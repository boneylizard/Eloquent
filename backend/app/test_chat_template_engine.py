# backend/app/test_chat_template_engine.py
"""Basic tests for the custom Jinja chat-template engine."""

import json
import tempfile
from pathlib import Path

import pytest

from app.chat_template_engine import (
    DEFAULT_CHAT_TEMPLATES,
    SELECTABLE_CHAT_TEMPLATES,
    lookup,
    render,
    render_with_stops,
    merge_backend_context,
)


def test_default_froggeric_qwen_template_lookup():
    entry = lookup("Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated.i1-IQ4_XS.gguf")
    assert entry is not None
    assert "<|im_end|>" in entry["stop_tokens"]
    # Should also match generic Qwen 3.5/3.6 filenames.
    assert lookup("Qwen3.6-32B-Q4_K_M.gguf") is not None
    assert lookup("Qwen3.5-14B-Q4_K_M.gguf") is not None


def test_default_froggeric_qwen_template_rendering():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt, stops = render_with_stops(
        messages,
        "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated.i1-IQ4_XS.gguf",
    )
    assert "<|im_start|>system" in prompt
    assert "You are helpful." in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<|im_end|>" in stops
    assert "<|im_start|>user" in stops


def test_qwen_thinking_requires_an_explicit_request():
    prompt = render(
        [{"role": "user", "content": "Hello"}],
        "Qwen3.6-32B-Q4_K_M.gguf",
        enable_thinking=True,
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_user_template_overrides_default(tmp_path, monkeypatch):
    settings = {
        "modelChatTemplates": {
            "my-override": {
                "patterns": "huihui-qwen3.6",
                "template": "USER: {{ messages[-1].content }}\nBOT: ",
                "stop_tokens": "<STOP>",
            }
        }
    }
    settings_dir = tmp_path / ".LiangLocal"
    settings_dir.mkdir()
    settings_file = settings_dir / "settings.json"
    settings_file.write_text(json.dumps(settings))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    entry = lookup("Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated.i1-IQ4_XS.gguf")
    assert entry is not None
    assert entry["stop_tokens"] == ["<STOP>"]
    prompt = render([{"role": "user", "content": "hi"}], "Huihui-Qwen3.6-35B.gguf")
    assert prompt == "USER: hi\nBOT: "


def test_merge_backend_context_avoids_duplicate_user_query():
    messages = [
        {"role": "system", "content": "base system"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    merged = merge_backend_context(
        messages,
        "[SYSTEM TRUTH] concise",
        "User Query: What is 2+2?",
    )
    assert merged[0]["content"] == "[SYSTEM TRUTH] concise\n\nbase system"
    # Should not duplicate the query.
    assert merged[1]["content"] == "What is 2+2?"


def test_merge_backend_context_injects_rag_context():
    messages = [
        {"role": "user", "content": "Tell me about X."},
    ]
    merged = merge_backend_context(
        messages,
        "",
        "Relevant docs:\n- doc1\n\nUser Query: Tell me about X.",
    )
    assert "Relevant docs:" in merged[0]["content"]
    assert "Tell me about X." in merged[0]["content"]


def test_lookup_returns_none_for_unknown_model():
    assert lookup("some-random-model-name.gguf") is None


def test_selectable_generic_template_can_override_model_detection():
    messages = [
        {"role": "system", "content": "Stay in character."},
        {"role": "user", "content": "Hello"},
    ]
    prompt, stops = render_with_stops(
        messages,
        "Qwen3.6-32B-Q4_K_M.gguf",
        template_id="generic",
    )
    assert "System: Stay in character." in prompt
    assert "User: Hello" in prompt
    assert prompt.endswith("Assistant: ")
    assert "\nUser:" in stops


def test_selectable_chatml_template_can_override_unknown_model():
    assert "chatml" in SELECTABLE_CHAT_TEMPLATES
    prompt = render(
        [{"role": "user", "content": "Hello"}],
        "some-random-model-name.gguf",
        template_id="chatml",
    )
    assert prompt == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"


def test_explicit_custom_template_id_does_not_need_a_model_pattern_match(tmp_path, monkeypatch):
    settings = {
        "modelChatTemplates": {
            "manual-template": {
                "patterns": "a-different-model",
                "template": "CUSTOM {{ messages[-1].content }} -> ",
                "stop_tokens": "<STOP>",
            }
        }
    }
    settings_dir = tmp_path / ".LiangLocal"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(json.dumps(settings))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    prompt, stops = render_with_stops(
        [{"role": "user", "content": "Hello"}],
        "unmatched-model.gguf",
        template_id="custom:manual-template",
    )
    assert prompt == "CUSTOM Hello -> "
    assert stops == ["<STOP>"]
