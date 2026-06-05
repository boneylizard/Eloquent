"""
Eloquent agent tool calling — shared registry, provider detection, and agent loops.

Used by /generate (chat web search), election assistant, and /v1/chat/completions/tools.
Supports OpenAI-style native tool_calls plus text fallbacks (GLM, DeepSeek, etc.).

Architecture grounding for agents: docs/ELOQUENT_SYSTEM_SPEC.md
(load via backend.app.eloquent_system_spec.load_eloquent_system_spec).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .openai_compat import get_configured_endpoint, is_api_endpoint
from .web_search_service import (
    get_web_search_tool_definition,
    handle_fetch_urls_tool_call,
    handle_web_search_tool_call,
    web_search_service,
)

logger = logging.getLogger(__name__)

# Host substrings that typically support OpenAI-compatible tools= + tool_choice=
NATIVE_TOOL_HOST_HINTS = (
    "deepseek",
    "openai.com",
    "openrouter.ai",
    "anthropic.com",
    "bigmodel.cn",
    "z.ai",
    "zhipu",
    "glm",
    "moonshot",
    "kimi",
    "siliconflow",
    "fireworks.ai",
    "together.xyz",
    "groq.com",
    "mistral.ai",
    "nanogpt",
    "novita.ai",
    "hyperbolic.xyz",
)

# Model id / name hints (custom endpoints often encode provider in model string)
NATIVE_TOOL_MODEL_HINTS = (
    "deepseek",
    "glm-4",
    "glm-5",
    "glm4",
    "glm5",
    "chatglm",
    "qwen",
)


def supports_native_tool_calling(
    model_name: Optional[str] = None,
    endpoint_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Whether to send tools= in the API payload (vs prompt injection fallback)."""
    url = ((endpoint_cfg or {}).get("url") or "").lower()
    model = (
        ((endpoint_cfg or {}).get("model") or "")
        + " "
        + (model_name or "")
    ).lower()

    if any(h in url for h in NATIVE_TOOL_HOST_HINTS):
        return True
    if any(h in model for h in NATIVE_TOOL_MODEL_HINTS):
        return True
    # Local OpenAI-compatible proxies on LAN often lack tool support
    if url and ("localhost" in url or "127.0.0.1" in url or "192.168." in url):
        return False
    # Unknown remote API — try native tools (most modern chat APIs support them)
    if url and url.startswith("http"):
        return True
    return False


def should_use_agent_tools_for_model(model_name: Optional[str]) -> bool:
    """Agentic tool loop (model chooses searches) vs legacy pre-inject web search."""
    if not model_name:
        return False
    if is_api_endpoint(model_name):
        return True
    lower = model_name.lower()
    # Local GGUF with tool-friendly templates may work later; API endpoints first.
    if any(x in lower for x in ("deepseek", "glm", "qwen", "devstral")):
        return False
    return False


def get_eloquent_chat_tools(
    *,
    simple: bool = True,
    include_news: bool = True,
    include_fetch_urls: bool = True,
) -> List[Dict[str, Any]]:
    """Tool definitions exposed to chat / generate agent loops."""
    tools: List[Dict[str, Any]] = [get_web_search_tool_definition(simple=simple, unified=True)]
    if include_news:
        tools.append(_web_search_news_tool_definition(simple=simple))
    if include_fetch_urls:
        tools.append(_fetch_urls_tool_definition())
    return tools


def _fetch_urls_tool_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "fetch_urls",
            "description": (
                "Download and read the full text of specific web pages when you already have the URLs. "
                "Use for reading multiple articles (e.g. all pieces on a publication site). "
                "Pass up to 8 URLs per call; call again for more batches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Full https:// URLs to fetch and read",
                        "maxItems": 8,
                    }
                },
                "required": ["urls"],
            },
        },
    }


def deepseek_likely_no_tools(model_name: Optional[str], endpoint_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Reasoner/R1 variants often ignore or break OpenAI-style tool calling."""
    combined = f"{model_name or ''} {(endpoint_cfg or {}).get('model', '')}".lower()
    return any(x in combined for x in ("reasoner", "r1", "deepseek-reasoner"))


def is_deepseek_model(model_name: Optional[str], endpoint_cfg: Optional[Dict[str, Any]] = None) -> bool:
    combined = f"{model_name or ''} {(endpoint_cfg or {}).get('model', '')}".lower()
    return "deepseek" in combined


_ARTICLE_RESEARCH_RE = re.compile(
    r"ux\s*mag(?:azine)?|uxmag\.com|"
    r"published\s+(?:like\s+)?\d*\s*articles?|"
    r"find\s+(?:as\s+many|all|my)\b.*\barticles?|"
    r"read\s+(?:all|my)\b.*\barticles?|"
    r"\d+\s+articles?\s+on\b|"
    r"research\b.*\b(?:articles?|publication|ux\s*mag)",
    re.IGNORECASE,
)


def detect_article_research_intent(query: str) -> bool:
    """User wants many full articles, not snippet-only web search."""
    q = (query or "").strip()
    if not q:
        return False
    if _ARTICLE_RESEARCH_RE.search(q):
        return True
    low = q.lower()
    return ("research" in low or "find" in low) and (
        "article" in low or "published" in low or "ux magazine" in low or "uxmag" in low
    )


WEB_SEARCH_MODEL_INSTRUCTIONS = """[WEB SEARCH RULES — READ BEFORE REPLYING]
The blocks above are the ONLY live web data Eloquent retrieved for this turn.
- If you see "No live results" or empty results, say clearly that web search did not return usable data.
- Do NOT claim you searched the web, browsed sites, or read articles unless that data appears above.
- Do NOT invent URLs, headlines, quotes, or article text that are not in the blocks above.
- Search snippets are summaries; full article text only appears in FETCHED sections.
- If URLs failed with 403/bot block, tell the user to paste article text or save .txt files in Transcript search — do not pretend you read those pages."""


def research_block_is_meaningful(block: str) -> bool:
    """True if injected research has real content, not only errors/headers."""
    if not block or len(block.strip()) < 80:
        return False
    low = block.lower()
    if "no results found" in low and "fetched 0/" in low:
        return False
    if low.count("403 forbidden") >= 3 and "content:" not in low:
        return False
    markers = (
        "web search results",
        "search completed",
        "fetched ",
        "transcript corpus",
        "url:",
        "content:",
        "snippet:",
    )
    return any(m in low for m in markers)


def build_web_search_receipt(
    *,
    ok: bool,
    steps: List[Dict[str, Any]],
    model_name: str,
    mode: str,
    path: str = "eloquent_prefetch",
    source_count: int = 0,
) -> str:
    tools_used = [s.get("tool") or "search" for s in (steps or [])]
    tools_line = ", ".join(tools_used[:12]) if tools_used else "snippet_search"
    status = "SUCCESS — data injected below" if ok else "NO USABLE DATA — tell the user search failed"
    return (
        f"[ELOQUENT WEB SEARCH RECEIPT]\n"
        f"Status: {status}\n"
        f"Path: {path}\n"
        f"Sources: {source_count}\n"
        f"Model: {model_name or 'unknown'}\n"
        f"Mode: {mode or 'normal'}\n"
        f"Server steps: {len(steps or [])} ({tools_line})\n"
        f"---"
    )


def prefer_programmatic_web_research(
    model_name: Optional[str],
    endpoint_cfg: Optional[Dict[str, Any]] = None,
    user_query: str = "",
) -> bool:
    """Use fetch_urls + site: batches instead of model tool APIs or snippet inject."""
    if deepseek_likely_no_tools(model_name, endpoint_cfg):
        return True
    if is_deepseek_model(model_name, endpoint_cfg):
        return True
    return detect_article_research_intent(user_query)


def extract_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    found = re.findall(r"https?://[^\s\]\)\"'<>]+", text)
    cleaned = []
    seen: set[str] = set()
    for u in found:
        u = u.rstrip(".,;)")
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


def build_site_search_queries(user_query: str, site: Optional[str] = None) -> List[str]:
    """Generate site:-scoped searches to find articles on a publication."""
    q = (user_query or "").strip()
    site = (site or "").strip().lower()
    if not site:
        for hint in ("uxmag.com", "ux magazine", "uxmag"):
            if hint in q.lower():
                site = "uxmag.com"
                break
    queries: List[str] = []
    if site:
        base = re.sub(r"\s+", " ", q)[:120]
        queries.append(f"site:{site} {base}")
        if "article" not in base.lower():
            queries.append(f"site:{site} articles")
        if "author" in base.lower() or "my " in base.lower() or "published" in base.lower():
            queries.append(f"site:{site} author")
    return queries[:8]


def _web_search_news_tool_definition(*, simple: bool = True) -> Dict[str, Any]:
    if simple:
        return {
            "type": "function",
            "function": {
                "name": "web_search_news",
                "description": (
                    "Search recent news headlines (RSS). Use for breaking news, elections, "
                    "politics, or time-sensitive events — not general how-to questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "News search keywords, e.g. '2026 senate polls Pennsylvania'",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    return {
        "type": "function",
        "function": {
            "name": "web_search_news",
            "description": "Search recent news headlines (RSS feeds). Prefer for current events and politics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "search_intent": {"type": "string"},
                },
                "required": ["search_queries"],
            },
        },
    }


def extract_tool_calls_from_text(content: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls emitted as text (GLM, some DeepSeek builds, legacy models).
    """
    if not content:
        return []

    tool_calls: List[Dict[str, Any]] = []

    # GLM / some APIs: <tool_call>name\n{json}\n</tool_call> or <tool_call>{json}</tool_call>
    glm_block = re.finditer(
        r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
        content,
        re.IGNORECASE,
    )
    for match in glm_block:
        block = match.group(1).strip()
        name = "web_search"
        args_raw = block
        if "\n" in block:
            first_line, rest = block.split("\n", 1)
            if first_line and not first_line.strip().startswith("{"):
                name = first_line.strip()
                args_raw = rest.strip()
        elif block.startswith("{") and '"name"' in block:
            try:
                parsed = json.loads(block)
                name = parsed.get("name") or parsed.get("function") or name
                args_raw = parsed.get("arguments") or parsed
            except json.JSONDecodeError:
                args_raw = block
        tool_calls.append(_make_tool_call(name, args_raw, len(tool_calls)))

    if tool_calls:
        return tool_calls

    # XML-style: <tool_call>{"name": "web_search", "arguments": {...}}</tool_call>
    tag_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    for match in re.finditer(tag_pattern, content, re.DOTALL | re.IGNORECASE):
        try:
            parsed = json.loads(match.group(1))
            name = parsed.get("name") or parsed.get("function") or "web_search"
            args = parsed.get("arguments") or parsed.get("parameters") or parsed
            tool_calls.append(_make_tool_call(name, args, len(tool_calls)))
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Markdown JSON fence with query / search_queries
    for match in re.finditer(r"```(?:json)?\s*({[\s\S]*?})\s*```", content):
        try:
            parsed = json.loads(match.group(1))
            name = parsed.get("name", "web_search")
            if "query" in parsed or "search_queries" in parsed or "search" in name.lower():
                args = parsed.get("arguments") or parsed.get("parameters") or parsed
                tool_calls.append(_make_tool_call(name, args, len(tool_calls)))
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Inline: web_search("query") or web_search(query="...")
    inline = r'web_search(?:_news)?\s*\(\s*(?:query\s*=\s*)?["\'](.+?)["\']\s*\)'
    for match in re.finditer(inline, content, re.IGNORECASE):
        name = "web_search_news" if "news" in match.group(0).lower() else "web_search"
        tool_calls.append(
            _make_tool_call(name, {"query": match.group(1)}, len(tool_calls))
        )

    # Bare JSON object with query key on its own line
    if not tool_calls:
        bare = re.search(
            r'^\s*\{\s*"query"\s*:\s*"([^"]+)"\s*\}\s*$',
            content.strip(),
            re.MULTILINE,
        )
        if bare:
            tool_calls.append(
                _make_tool_call("web_search", {"query": bare.group(1)}, 0)
            )

    return tool_calls


def _make_tool_call(name: str, arguments: Any, index: int) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        args_str = json.dumps(arguments, ensure_ascii=False)
    elif isinstance(arguments, str):
        args_str = arguments if arguments.strip().startswith("{") else json.dumps({"query": arguments})
    else:
        args_str = "{}"
    return {
        "id": f"call_{index}_{int(time.time() * 1000)}",
        "type": "function",
        "function": {"name": name or "web_search", "arguments": args_str},
    }


def _build_api_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


async def execute_eloquent_tool(
    tool_name: str,
    arguments: Any,
    *,
    max_results: int = 8,
    deep_research: bool = False,
    max_chars_per_result: int = 1200,
) -> str:
    """Run a registered Eloquent tool and return a string for the tool role message."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip().startswith("{") else {"query": arguments}
        except json.JSONDecodeError:
            arguments = {"query": arguments}

    name = (tool_name or "").strip().lower()
    chars = 8000 if deep_research else max_chars_per_result
    if name in ("web_search", "search_web", "internet_search"):
        return await handle_web_search_tool_call(
            arguments,
            max_results=max_results if not deep_research else 20,
            news=False,
            max_chars_per_result=chars,
        )
    if name in ("web_search_news", "news_search", "search_news"):
        return await handle_web_search_tool_call(
            arguments,
            max_results=min(max_results, 12),
            news=True,
            max_chars_per_result=chars,
        )
    if name in ("fetch_urls", "read_urls", "fetch_url"):
        return await handle_fetch_urls_tool_call(
            arguments,
            max_urls=8 if not deep_research else 20,
            max_chars_per_url=10000,
        )
    if name in ("search_transcript_corpus", "search_corpus", "corpus_search"):
        return await _search_corpus_tool(arguments)
    return f"Unknown tool '{tool_name}'. Available: web_search, web_search_news, fetch_urls, search_transcript_corpus."


async def _search_corpus_tool(arguments: Any) -> str:
    try:
        from .transcript_corpus import search_corpus, get_corpus_meta

        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip().startswith("{") else {"query": arguments}
        corpus_id = arguments.get("corpus_id") or arguments.get("corpus")
        query = arguments.get("query") or arguments.get("q") or ""
        if not corpus_id:
            return "Error: corpus_id required (index your .txt folder in Transcript search first)."
        if not query:
            return "Error: query required."
        top_k = int(arguments.get("top_k") or 30)
        data = search_corpus(
            corpus_id,
            query,
            top_k=top_k,
            min_score=float(arguments.get("min_score") or 0.12),
        )
        meta = get_corpus_meta(corpus_id) or {}
        lines = [
            f"Transcript corpus '{meta.get('name', corpus_id)}': {data.get('total_matches', 0)} matches",
            f"Query: {query}",
            "",
        ]
        for i, r in enumerate(data.get("results") or [], 1):
            lines.append(f"--- [{i}] {r.get('source_file')} (score {r.get('score')}) ---")
            lines.append(r.get("text", ""))
            lines.append("")
        return "\n".join(lines)[:120000]
    except Exception as e:
        logger.exception("corpus search tool failed")
        return f"search_transcript_corpus failed: {e}"


async def _call_chat_api(
    endpoint_cfg: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    native_tools: bool = True,
) -> Dict[str, Any]:
    url = _build_api_url(endpoint_cfg["url"])
    headers = {"Content-Type": "application/json"}
    api_key = endpoint_cfg.get("api_key") or endpoint_cfg.get("apiKey") or ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": endpoint_cfg.get("model"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools and native_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    timeout = httpx.Timeout(300.0, connect=60.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error("Agent tools API %s: %s", response.status_code, response.text[:500])
            response.raise_for_status()
        return response.json()


def _inject_tools_into_system(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> None:
    """Fallback when the API does not accept native tools."""
    lines = [
        "## Tools",
        "You may call tools by outputting JSON only, or use OpenAI tool_calls if supported.",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function") or {}
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')}")
    lines.append(
        'Example: {"name": "web_search", "arguments": {"query": "your keywords"}}'
    )
    block = "\n".join(lines)
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = (msg.get("content") or "") + "\n\n" + block
            return
    messages.insert(0, {"role": "system", "content": block})


AGENT_SEARCH_SYSTEM = """You are a research assistant inside Eloquent.

Tools:
- `web_search` — keyword web search (use site:domain.com when looking for articles on one publication)
- `fetch_urls` — read full pages when you have exact URLs (max 8 URLs per call; call multiple times for many articles)
- `web_search_news` — recent headlines only

For reading many articles: run site: searches, then `fetch_urls` with the links you find. Batch URLs 5–8 per fetch_urls call.
When done researching, reply starting with RESEARCH_COMPLETE: and a one-line summary (no more tool calls).
"""


async def gather_programmatic_article_research(
    user_query: str,
    *,
    research_urls: Optional[List[str]] = None,
    site_hint: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Multi-pass research without relying on model tool APIs (DeepSeek-safe).
    Fetches explicit URLs, runs site: searches, scrapes discovered links.
    """
    steps: List[Dict[str, Any]] = []
    parts = [
        "DEEP ARTICLE RESEARCH (programmatic — full page text, works with DeepSeek & all models):",
        "",
    ]

    urls: List[str] = []
    seen: set[str] = set()
    for u in (research_urls or []) + extract_urls_from_text(user_query):
        u = (u or "").strip()
        if u and u not in seen:
            seen.add(u)
            urls.append(u)

    for i in range(0, len(urls), 4):
        batch = urls[i : i + 4]
        result = await handle_fetch_urls_tool_call(
            {"urls": batch},
            max_urls=8,
            max_chars_per_url=10000,
        )
        parts.append(result)
        steps.append({"tool": "fetch_urls", "query": batch, "result_preview": result[:400]})

    discovered: List[str] = list(urls)
    site_queries = build_site_search_queries(user_query, site_hint)[:3]
    for sq in site_queries:
        # Snippets only — avoids dozens of 403 full-page scrapes
        result = await handle_web_search_tool_call(
            {"query": sq},
            max_results=8,
            max_chars_per_result=2500,
            scrape_full=False,
        )
        parts.append(f"--- site search: {sq} ---\n{result}")
        steps.append({"tool": "web_search", "query": sq, "result_preview": result[:400]})
        for u in extract_urls_from_text(result):
            on_site = "uxmag" in u.lower() or (site_hint and site_hint in u.lower())
            if u not in seen and on_site:
                seen.add(u)
                discovered.append(u)

    new_urls = [u for u in discovered if u not in urls][:12]
    for i in range(0, len(new_urls), 4):
        batch = new_urls[i : i + 4]
        if not batch:
            continue
        result = await handle_fetch_urls_tool_call(
            {"urls": batch},
            max_urls=8,
            max_chars_per_url=10000,
        )
        parts.append(result)
        steps.append({"tool": "fetch_urls", "query": batch, "result_preview": result[:400]})

    return "\n".join(parts).strip(), steps


async def gather_comprehensive_research(
    user_query: str,
    model_name: str,
    *,
    character_context: str = "",
    deep_research: bool = False,
    article_mode: bool = False,
    research_urls: Optional[List[str]] = None,
    transcript_corpus_id: Optional[str] = None,
    site_hint: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Best-effort research: corpus + programmatic fetches + optional API agent loop."""
    blocks: List[str] = []
    all_steps: List[Dict[str, Any]] = []

    if transcript_corpus_id:
        corp = await _search_corpus_tool({
            "corpus_id": transcript_corpus_id,
            "query": user_query,
            "top_k": 40,
        })
        blocks.append("TRANSCRIPT CORPUS (indexed .txt files):\n" + corp)
        all_steps.append({"tool": "search_transcript_corpus", "query": user_query})

    prog, prog_steps = await gather_programmatic_article_research(
        user_query,
        research_urls=research_urls,
        site_hint=site_hint,
    )
    if prog:
        blocks.append(prog)
    all_steps.extend(prog_steps)

    # Chat /generate uses gather_reliable_web_research only — no model tool loop here
    # (agent loop kept for election /v1/tools endpoints; unreliable across providers)

    return "\n\n".join(blocks).strip(), all_steps


async def gather_reliable_web_research(
    user_query: str,
    model_name: str,
    *,
    character_context: str = "",
    deep_research: bool = False,
    article_mode: bool = False,
    research_urls: Optional[List[str]] = None,
    transcript_corpus_id: Optional[str] = None,
    site_hint: Optional[str] = None,
    mode: str = "normal",
) -> Tuple[str, List[Dict[str, Any]], bool, List[Any]]:
    """
    Eloquent prefetch path for /generate — DuckDuckGo/RSS + optional article fetches.
    Returns (block, steps, ok, results_for_citations).
    """
    from .web_search_routing import decompose_search_queries, format_structured_search_block
    from .web_search_service import perform_smart_web_search, web_search_service

    all_steps: List[Dict[str, Any]] = []
    blocks: List[str] = []
    ok = False
    all_results: List[Any] = []

    try:
        max_results = 10 if deep_research else 8
        max_queries = 6 if deep_research else 4
        scrape_snippets_only = article_mode and not deep_research

        rule_queries, rule_intent = decompose_search_queries(user_query, max_queries=max_queries)
        smart = await perform_smart_web_search(
            user_query,
            max_results=max_results,
            use_optimization=True,
            max_queries=max_queries,
            scrape_full=not scrape_snippets_only,
        )
        optimized = smart.optimized_queries or rule_queries
        intent = smart.search_intent or rule_intent

        if not smart.results and rule_queries:
            seen_urls: set[str] = set()
            for q in rule_queries[:max_queries]:
                for r in await web_search_service.search(q, max_results):
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        smart.results.append(r)

        if smart.results:
            if scrape_snippets_only:
                pass
            elif not all(getattr(r, "scraped_successfully", False) for r in smart.results[:4]):
                to_scrape = smart.results[: max(4, min(max_results, len(smart.results)))]
                smart.results = await web_search_service.scrape_content(
                    to_scrape, max_to_scrape=min(4, max_results)
                )

            structured = format_structured_search_block(
                smart.results,
                original_prompt=user_query,
                optimized_queries=optimized,
                search_intent=intent,
                include_synthesis=True,
            )
            blocks.append(structured)
            all_results = list(smart.results)
            all_steps.append({
                "tool": "snippet_search",
                "query": optimized,
                "result_preview": structured[:300],
            })
            ok = True
    except Exception as e:
        logger.error("Snippet web search failed: %s", e)
        blocks.append(f"WEB SEARCH (snippets): failed — {e}")

    need_deep = bool(
        article_mode
        or deep_research
        or research_urls
        or transcript_corpus_id
        or detect_article_research_intent(user_query)
    )
    if need_deep:
        deep_block, deep_steps = await gather_comprehensive_research(
            user_query,
            model_name,
            character_context=character_context,
            deep_research=deep_research,
            article_mode=article_mode,
            research_urls=research_urls,
            transcript_corpus_id=transcript_corpus_id,
            site_hint=site_hint,
        )
        all_steps.extend(deep_steps)
        if deep_block and research_block_is_meaningful(deep_block):
            blocks.append(deep_block)
            ok = True

    combined = "\n\n".join(blocks).strip()
    return combined, all_steps, ok, all_results


async def gather_agent_tool_context(
    user_query: str,
    model_name: str,
    *,
    character_context: str = "",
    max_steps: int = 4,
    deep_research: bool = False,
    max_results_per_search: int = 8,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Run a short tool-calling loop against the configured API model.
    Returns (formatted context block for injection into /generate, tool_steps for UI).
    """
    endpoint_cfg = get_configured_endpoint(model_name)
    if not endpoint_cfg or not endpoint_cfg.get("model"):
        raise ValueError(
            "Agent tools require a configured custom API endpoint (Settings → API endpoints)."
        )

    native = supports_native_tool_calling(model_name, endpoint_cfg)
    if deepseek_likely_no_tools(model_name, endpoint_cfg):
        native = False
        logger.info("DeepSeek reasoner/R1 detected — using text tool fallback (no native tools)")
    tools = get_eloquent_chat_tools(simple=True, include_news=True, include_fetch_urls=True)
    if deep_research:
        max_steps = max(max_steps, 12)
        max_results_per_search = 20

    user_block = user_query.strip()
    if character_context:
        user_block = character_context + "\n\nUser question:\n" + user_block

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SEARCH_SYSTEM},
        {"role": "user", "content": user_block},
    ]
    if not native:
        _inject_tools_into_system(messages, tools)
    else:
        # DeepSeek chat: some gateways reject parallel tool calls
        pass

    tool_steps: List[Dict[str, Any]] = []

    for step in range(max_steps):
        logger.info(
            "Agent tool step %d/%d model=%s native_tools=%s",
            step + 1,
            max_steps,
            model_name,
            native,
        )
        response = await _call_chat_api(
            endpoint_cfg,
            messages,
            tools,
            temperature=0.2,
            max_tokens=900 if not deep_research else 1200,
            native_tools=native,
        )
        choice = (response.get("choices") or [{}])[0]
        message_obj = choice.get("message") or {}
        content = (message_obj.get("content") or "").strip()
        tool_calls = list(message_obj.get("tool_calls") or [])

        if not tool_calls:
            parsed = extract_tool_calls_from_text(content)
            if parsed:
                tool_calls = parsed
                logger.info("Parsed %d text tool call(s)", len(parsed))

        # Model finished without tools
        if not tool_calls:
            if content.upper().startswith("RESEARCH_COMPLETE:"):
                summary = content.split(":", 1)[-1].strip()
                if summary:
                    tool_steps.append({"tool": "summary", "result_preview": summary[:400]})
            break

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for idx, tool_call in enumerate(tool_calls):
            func = tool_call.get("function") or {}
            tool_name = func.get("name") or "web_search"
            raw_args = func.get("arguments", "{}")
            tool_call_id = tool_call.get("id") or f"tool_{step}_{idx}"

            try:
                arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {"query": str(raw_args)}

            result = await execute_eloquent_tool(
                tool_name,
                arguments,
                max_results=max_results_per_search,
                deep_research=deep_research,
                max_chars_per_result=8000 if deep_research else 1200,
            )
            query_hint = arguments.get("query") or arguments.get("search_queries")
            tool_steps.append({
                "tool": tool_name,
                "query": query_hint,
                "result_preview": result[:500],
            })
            logger.info("Tool %s done (%d chars)", tool_name, len(result))

            if native:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"[Tool result for {tool_name}]\n{result}",
                })

    if not tool_steps:
        return "", []

    parts = ["WEB SEARCH (agent — model-chosen queries):", ""]
    for i, step in enumerate(tool_steps, 1):
        if step.get("tool") == "summary":
            parts.append(f"Agent summary: {step.get('result_preview', '')}")
            continue
        parts.append(f"--- Step {i}: {step.get('tool')} — {step.get('query', '')} ---")
        preview = step.get("result_preview", "")
        parts.append(preview)
        parts.append("")

    combined = "\n".join(parts).strip()
    if len(combined) > 180000:
        combined = combined[:180000] + "\n\n[Research context truncated for prompt size]"
    return combined, tool_steps
