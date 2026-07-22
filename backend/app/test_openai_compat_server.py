import asyncio
import json

import httpx
from fastapi import FastAPI

from .openai_compat import (
    ChatCompletionRequest,
    _complete_local_chat,
    _build_nanogpt_url,
    apply_extended_thinking_request,
    get_provider_attribution_headers,
    _resolve_local_model,
    _stream_local_chat,
    router,
)


def test_openrouter_requests_identify_mirid_without_user_data():
    headers = get_provider_attribution_headers({"url": "https://openrouter.ai/api/v1"})
    assert headers == {
        "HTTP-Referer": "https://mirid.ai",
        "X-OpenRouter-Title": "Mirid",
        "X-OpenRouter-Categories": "roleplay,general-chat",
    }
    assert get_provider_attribution_headers({"url": "https://example.com/openrouter.ai"}) == {}


def test_nanogpt_subscription_route_is_not_rewritten_for_thinking_models():
    url = _build_nanogpt_url(
        "https://nano-gpt.com/api/subscription/v1",
        {"model": "zhipu/glm-5:thinking", "thinking": True},
    )
    assert url == "https://nano-gpt.com/api/subscription/v1/chat/completions"


def test_nanogpt_reasoning_is_off_unless_requested():
    request_data = {"model": "zhipu/glm-5", "messages": []}
    enabled = apply_extended_thinking_request(
        {"url": "https://nano-gpt.com/api/v1"},
        request_data,
    )
    assert enabled is False
    assert request_data["reasoning_effort"] == "none"


def test_openrouter_thinking_suffix_still_enables_reasoning():
    request_data = {"model": "zhipu/glm-5:thinking", "messages": [], "max_tokens": 4096}
    enabled = apply_extended_thinking_request(
        {"url": "https://openrouter.ai/api/v1"},
        request_data,
    )
    assert enabled is True
    assert request_data["reasoning"]["enabled"] is True


class FakeModelManager:
    def __init__(self, available=None, loaded=None):
        self.available = available or []
        self.loaded = loaded or []

    def list_available_models(self):
        return {"available_models": self.available}

    def get_loaded_models(self):
        return {"loaded_models": self.loaded}

    def get_model(self, model_name, gpu_id):
        if any(model["name"] == model_name and model["gpu_id"] == gpu_id for model in self.loaded):
            return FakeChatModel()
        raise ValueError("Model not loaded")


class FakeChatModel:
    def create_chat_completion(self, messages, stream, **kwargs):
        assert messages == [{"role": "user", "content": "Hello"}]
        if stream:
            return iter([
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " there"}}]},
            ])
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello there"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }


def _request(stream):
    return ChatCompletionRequest(
        model="test.gguf",
        messages=[{"role": "user", "content": "Hello"}],
        stream=stream,
    )


def test_resolve_local_model_prefers_loaded_gpu():
    manager = FakeModelManager(
        available=["test.gguf"],
        loaded=[{"name": "test.gguf", "gpu_id": 1}],
    )
    assert _resolve_local_model(manager, "test.gguf", None) == ("test.gguf", 1)


def test_local_chat_completion_uses_openai_shape():
    result = asyncio.run(_complete_local_chat(FakeChatModel(), _request(False), "test.gguf"))
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Hello there"
    assert result["usage"]["total_tokens"] == 4


def test_local_chat_stream_emits_openai_sse_and_done():
    async def collect():
        return [chunk async for chunk in _stream_local_chat(FakeChatModel(), _request(True), "test.gguf")]

    chunks = asyncio.run(collect())
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: {")
    ]
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
    )
    assert content == "Hello there"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1] == "data: [DONE]\n\n"


def test_openai_routes_work_through_an_external_asgi_client():
    async def exercise_api():
        app = FastAPI()
        app.state.model_manager = FakeModelManager(
            available=["test.gguf"],
            loaded=[{"name": "test.gguf", "gpu_id": 0}],
        )
        app.include_router(router)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mirid") as client:
            models_response = await client.get("/v1/models")
            chat_response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test.gguf",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
        return models_response, chat_response

    models_response, chat_response = asyncio.run(exercise_api())
    assert models_response.status_code == 200
    assert models_response.json()["data"][0]["id"] == "test.gguf"
    assert chat_response.status_code == 200
    assert chat_response.headers["content-type"].startswith("text/event-stream")
    assert '"content": "Hello"' in chat_response.text
    assert chat_response.text.endswith("data: [DONE]\n\n")
