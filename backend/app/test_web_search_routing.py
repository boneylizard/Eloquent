"""Tests for web search routing and structured formatting."""

import pytest

from .web_search_routing import (
    decompose_search_queries,
    endpoint_supports_native_search,
    format_structured_search_block,
    resolve_web_search_path,
)
from types import SimpleNamespace


def _result(**kwargs):
    return SimpleNamespace(
        scraped_successfully=kwargs.get("scraped_successfully", False),
        publisher=kwargs.get("publisher"),
        content=kwargs.get("content"),
        **{k: v for k, v in kwargs.items() if k not in ("scraped_successfully", "publisher", "content")},
    )


def test_resolve_auto_native_for_openrouter():
    cfg = {"url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o"}
    assert resolve_web_search_path(
        use_web_search=True,
        strategy="auto",
        model_name="endpoint-abc",
        endpoint_cfg=cfg,
    ) == "native"


def test_resolve_auto_eloquent_for_local():
    assert resolve_web_search_path(
        use_web_search=True,
        strategy="auto",
        model_name="some-gguf-model",
        endpoint_cfg=None,
    ) == "eloquent"


def test_retired_strategy_override_is_ignored():
    cfg = {"url": "https://openrouter.ai/api/v1", "model": "gpt-4o", "supports_native_search": True}
    assert resolve_web_search_path(
        use_web_search=True,
        strategy="eloquent",
        model_name="endpoint-1",
        endpoint_cfg=cfg,
    ) == "native"


def test_auto_uses_prefetch_for_article_research():
    cfg = {"url": "https://openrouter.ai/api/v1", "model": "gpt-4o"}
    assert resolve_web_search_path(
        use_web_search=True,
        strategy="auto",
        model_name="endpoint-1",
        endpoint_cfg=cfg,
        article_mode=True,
    ) == "eloquent"


def test_retired_endpoint_override_is_ignored():
    cfg = {"url": "https://openrouter.ai/api/v1", "supports_native_search": False}
    assert endpoint_supports_native_search(cfg) is True


def test_web_search_toggle_off_still_disables_search():
    assert resolve_web_search_path(
        use_web_search=False,
        strategy="native",
        model_name="endpoint-1",
        endpoint_cfg={"url": "https://openrouter.ai/api/v1"},
    ) == "off"


def test_decompose_comparison():
    queries, intent = decompose_search_queries("compare Python and Rust for web APIs")
    assert len(queries) >= 2
    assert "comparison" in intent or len(queries) >= 2


def test_format_structured_search_block():
    results = [
        _result(title="Example", url="https://example.com/a", snippet="Hello world"),
        _result(title="Other", url="https://example.com/b", snippet="More text"),
    ]
    block = format_structured_search_block(
        results,
        original_prompt="test query",
        optimized_queries=["test query"],
        search_intent="general",
        include_synthesis=True,
    )
    assert "[WEB SEARCH RESULTS]" in block
    assert "https://example.com/a" in block
    assert "[1]" in block and "[2]" in block
    assert "SYNTHESIS" in block
