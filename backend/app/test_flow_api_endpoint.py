"""Flow intro / about API routing must not advance round-robin."""

import json
from pathlib import Path

import pytest

from . import openai_compat as oc


@pytest.fixture
def round_robin_settings(tmp_path, monkeypatch):
    settings = {
        "apiEndpointRoundRobinEnabled": True,
        "apiEndpointRoundRobinCursor": {"__manual_rotation__": 0},
        "customApiEndpoints": [
            {
                "id": "endpoint-a",
                "name": "A",
                "url": "https://a.example/v1",
                "apiKey": "ka",
                "model": "model-a",
                "enabled": True,
                "rotate_enabled": True,
            },
            {
                "id": "endpoint-b",
                "name": "B",
                "url": "https://b.example/v1",
                "apiKey": "kb",
                "model": "model-b",
                "enabled": True,
                "rotate_enabled": True,
            },
        ],
    }
    liang_dir = tmp_path / ".LiangLocal"
    liang_dir.mkdir(parents=True, exist_ok=True)
    settings_path = liang_dir / "settings.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return settings_path


def test_flow_purpose_never_rotates(round_robin_settings):
    first = oc.get_configured_endpoint("endpoint-a", request_purpose="character_intro")
    second = oc.get_configured_endpoint("endpoint-a", request_purpose="character_intro")
    assert first["url"] == "https://a.example/v1"
    assert second["url"] == "https://a.example/v1"


def test_chat_purpose_rotates(round_robin_settings):
    first = oc.get_configured_endpoint("endpoint-a")
    second = oc.get_configured_endpoint("endpoint-a")
    assert first["url"] == "https://a.example/v1"
    assert second["url"] == "https://b.example/v1"


def test_resolve_flow_without_flow_api_url_returns_none(round_robin_settings):
    cfg = oc.resolve_flow_api_endpoint_config(
        request_purpose="character_intro",
        model_name="endpoint-a",
    )
    assert cfg is None


def test_resolve_flow_dedicated_ignores_settings_lookup(round_robin_settings):
    """Dedicated API uses only request body fields; global rotation must not apply."""
    pinned_url = "https://pinned.example/v1"
    urls = []
    for _ in range(10):
        cfg = oc.resolve_flow_api_endpoint_config(
            request_purpose="character_intro",
            model_name="endpoint-a",
            flow_api_url=pinned_url,
            flow_api_model="pinned-model",
            flow_api_key="pk",
        )
        assert cfg is not None
        urls.append(cfg["url"])
    assert urls == [pinned_url.rstrip("/")] * 10
    assert all(
        oc.is_flow_dedicated_api_request("character_intro", pinned_url)
        for _ in range(1)
    )


def test_resolve_flow_dedicated_default_model():
    cfg = oc.resolve_flow_api_endpoint_config(
        request_purpose="system_intro",
        flow_api_url="https://x.example",
        flow_api_model="",
    )
    assert cfg["url"] == "https://x.example"
    assert cfg["model"] == "gpt-3.5-turbo"
