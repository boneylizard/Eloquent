"""Automatic web-search routing for Mirid chat."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .openai_compat import get_configured_endpoint, is_api_endpoint

logger = logging.getLogger(__name__)

# Compatibility surface for callers that inspect supported strategies.
WEB_SEARCH_STRATEGIES = ("auto",)
DEFAULT_WEB_SEARCH_STRATEGY = "auto"

# Hosts that typically offer built-in or plugin web search (auto-detect when checkbox unset)
NATIVE_SEARCH_HOST_HINTS = (
    "openrouter.ai",
    "perplexity.ai",
    "api.perplexity",
    "nano-gpt.com",
    "nanogpt",
    "x.ai",
    "api.x.ai",
    "anthropic.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",
)

# Model id substrings that imply online/search-capable SKUs
NATIVE_SEARCH_MODEL_HINTS = (
    ":online",
    "sonar",
    "perplexity",
    "gpt-4o-search",
    "search-preview",
)

WEB_SEARCH_SYNTHESIS_INSTRUCTION = """[SYNTHESIS — use retrieved sources above]
- Answer using only information from the WEB SEARCH blocks and cite sources as [1], [2] matching the numbered entries.
- Include the source URL in parentheses after each citation when stating facts, e.g. ([1] https://example.com).
- If sources conflict, note the disagreement briefly.
- Do not invent URLs, dates, or quotes not present in the blocks above."""

_PATH_LABELS = {
    "off": "off",
    "eloquent": "eloquent_prefetch",
    "native": "provider_native",
    "native_tools": "provider_tools",
}


def load_web_search_settings() -> Dict[str, Any]:
    """Return the only supported web-search strategy."""
    return {"webSearchStrategy": DEFAULT_WEB_SEARCH_STRATEGY}


def endpoint_supports_native_search(
    endpoint_cfg: Optional[Dict[str, Any]] = None,
    *,
    model_name: Optional[str] = None,
) -> bool:
    """
    Whether this API endpoint can use provider-native web search (no Eloquent prefetch).
    Auto-detects support from the endpoint URL and model identifier.
    """
    url = ((endpoint_cfg or {}).get("url") or "").lower()
    model = (
        ((endpoint_cfg or {}).get("model") or "")
        + " "
        + (model_name or "")
    ).lower()

    if any(h in url for h in NATIVE_SEARCH_HOST_HINTS):
        return True
    if any(h in model for h in NATIVE_SEARCH_MODEL_HINTS):
        return True
    return False


def requires_eloquent_prefetch(
    *,
    article_mode: bool = False,
    deep_research: bool = False,
    research_urls: Optional[List[str]] = None,
    user_query: str = "",
) -> bool:
    """Article and deep-research flows need server-side fetch even when native search is on."""
    if article_mode or deep_research:
        return True
    if research_urls:
        return True
    try:
        from .eloquent_agent_tools import detect_article_research_intent

        if detect_article_research_intent(user_query):
            return True
    except Exception:
        pass
    return False


def resolve_web_search_path(
    *,
    use_web_search: bool,
    strategy: Optional[str] = None,
    model_name: Optional[str] = None,
    endpoint_cfg: Optional[Dict[str, Any]] = None,
    article_mode: bool = False,
    deep_research: bool = False,
    research_urls: Optional[List[str]] = None,
    user_query: str = "",
) -> str:
    """
    Automatically returns 'off', 'eloquent', or 'native'.

    ``strategy`` remains accepted for request compatibility but is ignored.
    """
    if not use_web_search:
        return "off"

    needs_prefetch = requires_eloquent_prefetch(
        article_mode=article_mode,
        deep_research=deep_research,
        research_urls=research_urls,
        user_query=user_query,
    )

    if needs_prefetch:
        return "eloquent"
    is_api_model = endpoint_cfg is not None or is_api_endpoint(model_name or "")
    if is_api_model and endpoint_supports_native_search(endpoint_cfg, model_name=model_name):
        return "native"
    return "eloquent"


def apply_native_web_search_request(
    request_data: Dict[str, Any],
    endpoint_cfg: Dict[str, Any],
) -> Tuple[Dict[str, str], str]:
    """
    Mutate request_data for provider-native search. Returns (extra_headers, method_label).

    Methods:
      - openrouter_tool: tools type openrouter/web_search (OpenRouter server tool)
      - openrouter_plugin: legacy plugins web (fallback)
      - online_suffix: append :online to model slug
      - pass_through: Perplexity-style — model already searches; no extra fields
    """
    url = (endpoint_cfg.get("url") or "").lower()
    model = (request_data.get("model") or endpoint_cfg.get("model") or "").strip()

    if "perplexity" in url or "sonar" in model.lower():
        return {}, "pass_through"

    if "openrouter.ai" in url:
        tools = list(request_data.get("tools") or [])
        has_or = any(
            isinstance(t, dict)
            and (
                t.get("type") in ("openrouter/web_search", "openrouter:web_search")
                or (t.get("type") == "function" and (t.get("function") or {}).get("name") == "web_search")
            )
            for t in tools
        )
        if not has_or:
            tools.append({"type": "openrouter/web_search"})
            request_data["tools"] = tools
        return {}, "openrouter_tool"

    if "nano-gpt.com" in url or "nanogpt" in url:
        # NanoGPT: prefer standard web_search tool if API accepts tools
        tools = list(request_data.get("tools") or [])
        if not tools:
            try:
                from .web_search_service import get_web_search_tool_definition

                tools.append(get_web_search_tool_definition(simple=True, unified=True))
                request_data["tools"] = tools
                request_data["tool_choice"] = "auto"
            except Exception:
                pass
        return {}, "nanogpt_tools"

    # Generic OpenAI-compatible: :online suffix or web plugin
    if model and ":online" not in model:
        request_data["model"] = f"{model}:online"
        return {}, "online_suffix"

    plugins = list(request_data.get("plugins") or [])
    if not any(isinstance(p, dict) and p.get("id") == "web" for p in plugins):
        plugins.append({"id": "web", "max_results": 8})
        request_data["plugins"] = plugins
    return {}, "openrouter_plugin"


def build_search_meta(
    *,
    path: str,
    status: str,
    source_count: int = 0,
    queries: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, str]]] = None,
    mode: str = "normal",
    strategy: str = "auto",
    native_method: Optional[str] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Frontend-friendly metadata for search status UI."""
    return {
        "path": _PATH_LABELS.get(path, path),
        "status": status,  # searching | complete | error | native_delegated
        "source_count": source_count,
        "queries": queries or [],
        "sources": sources or [],
        "mode": mode,
        "strategy": strategy,
        "native_method": native_method,
        "steps": steps or [],
    }


def sources_from_results(results: List[Any]) -> List[Dict[str, str]]:
    """Extract citation chips from SearchResult list."""
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for r in results or []:
        url = getattr(r, "url", None) or (r.get("url") if isinstance(r, dict) else "")
        if not url or url in seen:
            continue
        seen.add(url)
        title = getattr(r, "title", None) or (r.get("title") if isinstance(r, dict) else "") or url
        out.append({"title": str(title)[:200], "url": str(url)})
    return out[:24]


def format_structured_search_block(
    results: List[Any],
    *,
    original_prompt: str = "",
    optimized_queries: Optional[List[str]] = None,
    search_intent: str = "",
    include_synthesis: bool = True,
    max_snippet_chars: int = 800,
) -> str:
    """
    Structured WEB SEARCH block with numbered citations (URLs, titles) for Eloquent inject.
    """
    if not results:
        return f"[WEB SEARCH RESULTS]\nQuery: {original_prompt or (optimized_queries or [''])[0]}\nNo relevant results found."

    queries = optimized_queries or [original_prompt]
    lines = [
        "[WEB SEARCH RESULTS]",
        f"Original: {original_prompt[:300]}" if original_prompt else "",
        f"Intent: {search_intent}" if search_intent else "",
        f"Queries: {', '.join(queries)}",
        "---",
    ]
    lines = [ln for ln in lines if ln]

    for i, result in enumerate(results, 1):
        title = getattr(result, "title", None) or (result.get("title") if isinstance(result, dict) else "Untitled")
        url = getattr(result, "url", None) or (result.get("url") if isinstance(result, dict) else "")
        publisher = getattr(result, "publisher", None) or (result.get("publisher") if isinstance(result, dict) else None)
        lines.append(f"\n[{i}] {title}")
        lines.append(f"    url: {url}")
        if publisher:
            lines.append(f"    source: {publisher}")

        content = getattr(result, "content", None) or (result.get("content") if isinstance(result, dict) else None)
        snippet = getattr(result, "snippet", None) or (result.get("snippet") if isinstance(result, dict) else "")
        scraped = getattr(result, "scraped_successfully", False) or (
            result.get("scraped_successfully") if isinstance(result, dict) else False
        )

        if scraped and content:
            text = str(content)
            if len(text) > max_snippet_chars:
                text = text[:max_snippet_chars] + "…"
            lines.append(f"    content: {text}")
        elif snippet:
            sn = str(snippet)
            if len(sn) > max_snippet_chars:
                sn = sn[:max_snippet_chars] + "…"
            lines.append(f"    snippet: {sn}")

    if include_synthesis:
        lines.append("")
        lines.append(WEB_SEARCH_SYNTHESIS_INSTRUCTION)
    return "\n".join(lines)


def _basic_queries_fallback(user_prompt: str) -> Tuple[List[str], str]:
    """Lightweight query cleanup without importing web_search_service (test-friendly)."""
    text = (user_prompt or "").strip()
    if not text:
        return [user_prompt or ""], "general search"
    cleaned = re.sub(
        r"^(?:can you |could you |please |what is |what are |tell me |search for )",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip().rstrip("?")
    return [cleaned or text], "general search"


def decompose_search_queries(user_prompt: str, *, max_queries: int = 4) -> Tuple[List[str], str]:
    """
    Rule-based query decomposition (no LLM). Complements optimize_query for news/election-style asks.
    """
    try:
        from .web_search_service import web_search_service

        queries, intent = web_search_service._basic_query_optimization(user_prompt)
    except Exception:
        queries, intent = _basic_queries_fallback(user_prompt)
    low = (user_prompt or "").lower()

    # Multi-topic: "X and Y" / "X vs Y"
    split_match = re.search(
        r"^(?:compare|difference between)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)$",
        low,
        re.IGNORECASE,
    )
    if split_match:
        a, b = split_match.group(1).strip(), split_match.group(2).strip()
        if a and b:
            return [a, b][:max_queries], "comparison search"

    and_parts = re.split(r"\s+and\s+", user_prompt, maxsplit=2, flags=re.IGNORECASE)
    if len(and_parts) >= 2 and len(user_prompt) > 40:
        extra = [p.strip().rstrip("?") for p in and_parts if len(p.strip()) > 8]
        if len(extra) >= 2:
            merged = list(dict.fromkeys(extra + queries))[:max_queries]
            return merged, intent or "multi-topic search"

    # News / election style
    if re.search(r"\b(news|headlines|election|polls|primary)\b", low):
        if "news" not in queries[0].lower():
            queries[0] = f"{queries[0]} news"
        return queries[:max_queries], intent or "news search"

    return queries[:max_queries], intent


def get_endpoint_config_for_model(
    model_name: Optional[str],
    *,
    request_purpose: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Configured endpoint dict including supports_native_search when present."""
    if not model_name or not is_api_endpoint(model_name):
        return None
    return get_configured_endpoint(model_name, request_purpose=request_purpose)
