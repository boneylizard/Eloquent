"""Unit tests for character API generation (timeouts, streaming path)."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# character_intelligence imports heavy ML deps at module load; stub for unit tests.
_st_util = MagicMock()
sys.modules.setdefault("sentence_transformers", MagicMock(util=_st_util))
sys.modules.setdefault("torch", MagicMock())

from . import character_intelligence as ci  # noqa: E402
from . import openai_compat  # noqa: E402


def test_generate_with_api_uses_stream_aggregate_path():
    captured = {}

    async def fake_collect(endpoint_config, url, prepared, extra_headers=None):
        captured["endpoint_config"] = endpoint_config
        captured["url"] = url
        captured["prepared"] = prepared
        return '{"name": "Test"}'

    api_endpoint = {
        "id": "endpoint-test-1",
        "url": "https://api.example.com",
        "api_key": "sk-test",
        "name": "Test EP",
        "model": "gpt-4",
    }

    with patch.object(
        openai_compat,
        "prepare_endpoint_request",
        return_value=(api_endpoint, "https://api.example.com/v1/chat/completions", {"model": "gpt-4", "stream": True}),
    ) as prep, patch.object(
        openai_compat,
        "collect_openai_compatible_stream_text",
        new=AsyncMock(side_effect=fake_collect),
    ):
        out = asyncio.run(
            ci.generate_with_api("hello", api_endpoint, model_name="endpoint-test-1")
        )

    assert out == '{"name": "Test"}'
    prep.assert_called_once()
    _model_id, req = prep.call_args[0]
    assert _model_id == "endpoint-test-1"
    assert req["stream"] is True
    assert req["_skip_openai_message_pruning"] is True
    assert req["_max_stream_attempts"] == ci.CHARACTER_API_MAX_ATTEMPTS
    assert req["max_tokens"] == ci.CHARACTER_API_MAX_TOKENS
    assert captured["prepared"]["stream"] is True


def test_generate_with_api_retries_read_timeout():
    calls = {"n": 0}

    async def flaky_collect(*_a, **_k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ReadTimeout("read timed out")
        return "ok"

    api_endpoint = {
        "id": "endpoint-retry",
        "url": "https://api.example.com",
        "api_key": "",
        "name": "Retry EP",
        "model": "m",
    }

    with patch.object(
        openai_compat,
        "prepare_endpoint_request",
        return_value=(api_endpoint, "https://api.example.com/v1/chat/completions", {"model": "m"}),
    ), patch.object(
        openai_compat,
        "collect_openai_compatible_stream_text",
        new=AsyncMock(side_effect=flaky_collect),
    ), patch("asyncio.sleep", new=AsyncMock()):
        out = asyncio.run(
            ci.generate_with_api("p", api_endpoint, model_name="endpoint-retry")
        )

    assert out == "ok"
    assert calls["n"] == 2


def test_generate_with_api_raises_clear_error_after_retries():
    api_endpoint = {
        "id": "endpoint-fail",
        "url": "https://api.example.com",
        "api_key": "",
        "name": "Fail EP",
        "model": "m",
    }

    with patch.object(
        openai_compat,
        "prepare_endpoint_request",
        return_value=(api_endpoint, "https://api.example.com/v1/chat/completions", {"model": "m"}),
    ), patch.object(
        openai_compat,
        "collect_openai_compatible_stream_text",
        new=AsyncMock(side_effect=httpx.ReadTimeout("read timed out")),
    ), patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ci.CharacterApiError) as exc_info:
            asyncio.run(
                ci.generate_with_api("p", api_endpoint, model_name="endpoint-fail")
            )

    assert "read timeout" in str(exc_info.value).lower() or "stopped responding" in str(exc_info.value).lower()


def test_build_character_api_request_data_stream_and_attempts():
    req = ci._build_character_api_request_data(
        "prompt text",
        endpoint_model_id="endpoint-abc",
        configured_model="claude-3",
    )
    assert req["stream"] is True
    assert req["_max_stream_attempts"] == ci.CHARACTER_API_MAX_ATTEMPTS
    assert req["messages"][0]["content"] == "prompt text"


def test_remote_http_timeout_matches_openai_compat():
    from .openai_compat import REMOTE_HTTPC_TIMEOUT

    assert REMOTE_HTTPC_TIMEOUT.read == 3600.0
    assert REMOTE_HTTPC_TIMEOUT.connect == 60.0


def test_generate_character_json_surfaces_api_timeout():
    with patch.object(
        ci,
        "_generate_character_llm_text",
        new=AsyncMock(side_effect=ci.CharacterApiError(ci._CHARACTER_API_TIMEOUT_MSG)),
    ):
        result = asyncio.run(
            ci.generate_character_json(
            model_manager=None,
            messages=[{"role": "user", "content": "hi"}],
            character_analysis={},
            use_api=True,
            api_endpoint={"id": "endpoint-x", "url": "https://x", "model": "m"},
            )
        )

    assert result["status"] == "error"
    assert result.get("error_type") == "api_timeout"
    assert "timeout" in result["error"].lower() or "responding" in result["error"].lower()
