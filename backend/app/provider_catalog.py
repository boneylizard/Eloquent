from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/provider-catalog", tags=["Provider Catalog"])


PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "models_url": "https://api.openai.com/v1/models",
        "base_url": "https://api.openai.com/v1",
        "shape": "openai",
    },
    "anthropic": {
        "models_url": "https://api.anthropic.com/v1/models?limit=1000",
        "base_url": "https://api.anthropic.com/v1",
        "shape": "anthropic",
        "headers": {"anthropic-version": "2023-06-01"},
    },
    "gemini": {
        "models_url": "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "shape": "gemini",
        "key_header": "x-goog-api-key",
    },
    "mistral": {
        "models_url": "https://api.mistral.ai/v1/models",
        "base_url": "https://api.mistral.ai/v1",
        "shape": "mistral",
    },
    "xai": {
        "models_url": "https://api.x.ai/v1/language-models",
        "base_url": "https://api.x.ai/v1",
        "shape": "xai",
    },
    "meta": {
        "models_url": "https://api.meta.ai/v1/models",
        "base_url": "https://api.meta.ai/v1",
        "shape": "openai",
    },
}


class ProviderCatalogRequest(BaseModel):
    provider: str
    api_key: str


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _openai_chat_candidate(model_id: str) -> bool:
    lowered = model_id.lower()
    excluded = (
        "embedding", "moderation", "whisper", "tts", "transcribe", "dall-e",
        "image", "realtime", "audio", "sora", "search-preview",
    )
    return not any(fragment in lowered for fragment in excluded)


def normalize_provider_models(provider: str, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    shape = PROVIDERS[provider]["shape"]
    if shape in {"openai", "anthropic", "mistral"}:
        rows = payload.get("data", [])
    elif shape == "gemini":
        rows = payload.get("models", [])
    else:
        rows = payload.get("models", payload.get("data", []))
    if not isinstance(rows, list):
        return []

    models: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or row.get("name") or "").removeprefix("models/")
        if not model_id:
            continue

        if provider in {"openai", "meta"} and not _openai_chat_candidate(model_id):
            continue
        if provider == "gemini":
            methods = row.get("supportedGenerationMethods") or row.get("supported_actions") or []
            if methods and "generateContent" not in methods:
                continue
        if provider == "mistral" and row.get("capabilities", {}).get("completion_chat") is False:
            continue

        capabilities = row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
        input_modalities = row.get("input_modalities") or []
        models.append({
            "id": model_id,
            "name": str(row.get("display_name") or row.get("displayName") or model_id),
            "description": str(row.get("description") or ""),
            "context_length": _as_int(
                row.get("max_context_length")
                or row.get("inputTokenLimit")
                or row.get("context_length")
            ),
            "capabilities": {
                "vision": bool(capabilities.get("vision") or "image" in input_modalities),
                "tools": bool(capabilities.get("function_calling")),
                "reasoning": "reason" in model_id.lower() or "thinking" in model_id.lower(),
            },
            "created": row.get("created") or row.get("created_at"),
            "raw_pricing": {
                key: row.get(key)
                for key in (
                    "prompt_text_token_price", "completion_text_token_price", "pricing"
                )
                if row.get(key) is not None
            },
        })
    return models


@router.post("/models")
async def list_provider_models(request: ProviderCatalogRequest) -> dict[str, Any]:
    provider = request.provider.strip().lower()
    config = PROVIDERS.get(provider)
    if not config:
        raise HTTPException(status_code=400, detail="Unknown model provider.")
    api_key = request.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Add this provider's API key first.")

    headers = {"Accept": "application/json", **config.get("headers", {})}
    if config.get("key_header"):
        headers[config["key_header"]] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(config["models_url"], headers=headers)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="The provider's model catalogue could not be reached.") from error

    if response.status_code in {401, 403}:
        raise HTTPException(status_code=401, detail="The provider rejected that API key.")
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"The provider returned HTTP {response.status_code} while listing models.",
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise HTTPException(status_code=502, detail="The provider returned an unreadable model catalogue.") from error

    return {
        "provider": provider,
        "base_url": config["base_url"],
        "models": normalize_provider_models(provider, payload),
    }

