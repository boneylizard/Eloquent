import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .openai_compat import get_configured_endpoint, is_api_endpoint
from .web_search_service import get_web_search_tool_definition, handle_web_search_tool_call

logger = logging.getLogger(__name__)

FACT_SHEET_PATH = Path(__file__).resolve().parent / "2026_election_research.md"
FACT_SHEET_MAX_CHARS = 14000

DEFAULT_SYSTEM_PROMPT = """You are an expert US elections analyst embedded in a polling tracker application.

## Your capabilities
- You have access to the current polling data provided in the user message as JSON context
- You can use the `web_search` tool to find recent news, polling analysis, or candidate information
- You should cite sources when using web search results
- The user can also run news searches in the **News** tab: they can type any search query there (e.g. "Pennsylvania senate polls", "2026 governor race Texas") and see results. When the user asks for news or headlines, suggest a specific search query they can try in the News tab, and you can also run web_search yourself to summarize.

## How to use web_search
When you need current information not in the provided polling data, call web_search with a specific query.
Example queries: "2026 Ohio senate race latest polls", "Trump approval rating trend February 2026", "2026 midterms polling news"

## Response guidelines
- Be concise and data-driven. Lead with numbers and margins.
- When analyzing polls, note the pollster grade, sample size, and whether it's LV/RV/Adults
- Flag when results conflict across pollsters
- Note partisan lean of pollsters when relevant
- If the polling data shows a clear trend, state it directly
- Use party abbreviations: (D), (R), (I)
- Format margins as: Candidate +X (e.g., "McMorrow +3")

## What NOT to do
- Don't speculate about outcomes — stick to what the data shows
- Don't give long preambles — get to the analysis immediately
- Don't repeat the raw poll data back verbatim — synthesize and analyze it
"""

NEWS_SYSTEM_PROMPT = """You are a political news research agent embedded in a polling tracker application.

## Your capabilities
- You can use the `web_search` tool to find recent political news
- You must use the web_search tool to answer
- You should prioritize US political news (Congress, White House, elections, polling, major court/policy actions)

## Task
Return trending political news from roughly the last 48 hours. If the user supplies a query, include it but still include trending items.

## Tool usage rules (strict)
- Use at most 5 web_search calls total.
- Each web_search call should include 1-2 queries (not more).
- Aim for 2-4 web_search calls unless the user query is very specific.
- Do not embed JSON or multiple queries into a single query string.
- Avoid non-US or off-topic polling pages unless explicitly requested.
- Prefer broader queries without specific months/years unless the user asks for a date range.

## Output format (JSON only)
Return a single JSON object with this exact shape:
{"query": "...", "articles": [{"title": "...", "url": "...", "source": "...", "snippet": "..."}]}

Rules:
- Provide 8-12 articles when possible.
- Deduplicate by domain and by title.
- For "source": use the Publisher value from the search result when the tool response provides it (e.g. Reuters, CNN). Do not use the URL domain when it is an aggregator (e.g. news.google.com). When no Publisher is given, use the site name or domain (e.g. nytimes.com).
- Keep snippets short (1-2 sentences).
- Output JSON only, no markdown or extra text.
"""



class ElectionAssistantService:
    def __init__(self):
        self.max_steps = 4
        self.timeout = 120.0

    def _resolve_endpoint(self, model_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if model_id and is_api_endpoint(model_id):
            return get_configured_endpoint(model_id)

        # Optional fallback via env for a dedicated assistant endpoint
        env_url = os.getenv("ELECTION_ASSISTANT_API_URL")
        env_model = os.getenv("ELECTION_ASSISTANT_MODEL")
        if env_url and env_model:
            return {
                "url": env_url.rstrip("/"),
                "api_key": os.getenv("ELECTION_ASSISTANT_API_KEY", ""),
                "name": "Election Assistant Endpoint",
                "model": env_model
            }

        return None

    def _build_api_url(self, base_url: str) -> str:
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        if "/v1" in url and not url.endswith("/chat/completions"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def _supports_native_tools(self, endpoint: Dict[str, Any]) -> bool:
        """

Check if endpoint likely supports OpenAI tool calling."""
        url = (endpoint.get("url") or "").lower()
        if any(d in url for d in ["deepseek.com", "openai.com", "anthropic.com", "openrouter.ai"]):
            return True
        if "localhost" in url or "127.0.0.1" in url or "192.168." in url:
            return False
        return True

    def _extract_text_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls that the model output as text instead of structured tool_calls."""
        if not content:
            return []
        tool_calls = []
        # Pattern 1: <tool_call> tags
        tag_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        for match in re.finditer(tag_pattern, content, re.DOTALL):
            try:
                parsed = json.loads(match.group(1))
                name = parsed.get("name") or parsed.get("function") or "web_search"
                args = parsed.get("arguments") or parsed.get("parameters") or parsed
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                tool_calls.append({
                    "id": f"parsed_{len(tool_calls)}",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
            except Exception:
                pass
        if not tool_calls:
            json_pattern = r"```(?:json)?\s*({.*?})\s*```"
            for match in re.finditer(json_pattern, content, re.DOTALL):
                try:
                    parsed = json.loads(match.group(1))
                    if "query" in parsed or "search" in (parsed.get("name") or "").lower():
                        name = parsed.get("name", "web_search")
                        args = parsed.get("arguments") or parsed.get("parameters") or {"query": parsed.get("query", "")}
                        tool_calls.append({
                            "id": f"parsed_{len(tool_calls)}",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        })
                except Exception:
                    pass
        if not tool_calls:
            inline_pattern = r'web_search\s*\(\s*(?:query\s*=\s*)?["\'](.+?)["\']\s*\)'
            for match in re.finditer(inline_pattern, content):
                tool_calls.append({
                    "id": f"parsed_{len(tool_calls)}",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": match.group(1)})},
                })
        return tool_calls

    async def _call_api(
        self,
        api_endpoint: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        url = self._build_api_url(api_endpoint["url"])
        headers = {"Content-Type": "application/json"}
        if api_endpoint.get("api_key"):
            headers["Authorization"] = f"Bearer {api_endpoint['api_key']}"

        payload = {
            "model": api_endpoint.get("model"),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and self._supports_native_tools(api_endpoint):
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def ask(
        self,
        message: str,
        race_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        polls: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 900
    ) -> Dict[str, Any]:
        api_endpoint = self._resolve_endpoint(model)
        if not api_endpoint or not api_endpoint.get("model"):
            return {
                "error": "No API endpoint configured for this model. Configure a custom API endpoint in Settings.",
            }

        context = {
            "race_type": race_type,
            "metadata": metadata,
            "polls": polls
        }

        tools = [get_web_search_tool_definition(simple=True)]
        # Candidate roster and fact sheet first so the AI has them before any web search or analysis
        fact_sheet = _get_fact_sheet_context()
        user_content = ""
        if fact_sheet:
            user_content += "2026 election fact sheet and candidate roster (use for party and race context):\n" + fact_sheet + "\n\n"
        user_content += (
            "Polling context (JSON):\n"
            f"{json.dumps(context, ensure_ascii=False)}\n\n"
            f"User question: {message}"
        )

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        tool_steps: List[Dict[str, Any]] = []
        for step in range(self.max_steps):
            response = await self._call_api(
                api_endpoint=api_endpoint,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens
            )

            choice = (response.get("choices") or [{}])[0]
            message_obj = choice.get("message") or {}
            tool_calls = message_obj.get("tool_calls") or []
            if not tool_calls:
                content = message_obj.get("content", "") or ""
                parsed_calls = self._extract_text_tool_calls(content)
                if parsed_calls:
                    tool_calls = parsed_calls
                    logger.info("Parsed %d tool calls from text output", len(parsed_calls))

            if not tool_calls:
                return {
                    "answer": message_obj.get("content", ""),
                    "tool_steps": tool_steps,
                    "model": api_endpoint.get("model"),
                    "timestamp": int(time.time())
                }

            # Append assistant tool call request to history
            messages.append({
                "role": "assistant",
                "content": message_obj.get("content", ""),
                "tool_calls": tool_calls
            })

            # Execute tools
            for idx, tool_call in enumerate(tool_calls):
                func = tool_call.get("function") or {}
                tool_name = func.get("name") or "web_search"
                raw_args = func.get("arguments", "{}")
                tool_call_id = tool_call.get("id") or f"tool_{step}_{idx}"

                try:
                    arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                except Exception:
                    arguments = {"query": raw_args}

                if tool_name == "web_search":
                    tool_result = await handle_web_search_tool_call(arguments, max_results=5)
                else:
                    tool_result = f"Tool '{tool_name}' is not available."

                tool_steps.append({
                    "tool": tool_name,
                    "query": arguments.get("query") or arguments.get("search_queries"),
                    "result_preview": tool_result[:400],
                    "result": tool_result[:4000]
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result
                })

        return {
            "error": "Assistant exceeded maximum tool steps without a final answer."
        }

    async def ask_news(
        self,
        query: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 700
    ) -> Dict[str, Any]:
        api_endpoint = self._resolve_endpoint(model)
        if not api_endpoint or not api_endpoint.get("model"):
            return {
                "error": "No API endpoint configured for this model. Configure a custom API endpoint in Settings.",
            }

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_query = (query or "").strip()
        if not user_query:
            user_query = "Find trending US political news and election/polling headlines. Focus on the last 48 hours."

        fact_sheet = _get_fact_sheet_context()
        # Candidate roster and fact sheet first so the AI has them before deciding what to web search
        user_content = ""
        if fact_sheet:
            user_content += "2026 election fact sheet and candidate roster (use for context on key races and who is D/R):\n" + fact_sheet + "\n\n"
        user_content += f"Date: {now}\nUser request: {user_query}"

        tools = [get_web_search_tool_definition(simple=True)]
        messages = [
            {"role": "system", "content": NEWS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        tool_steps: List[Dict[str, Any]] = []
        max_steps = min(self.max_steps, 5)
        for step in range(max_steps):
            response = await self._call_api(
                api_endpoint=api_endpoint,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens
            )

            choice = (response.get("choices") or [{}])[0]
            message_obj = choice.get("message") or {}
            tool_calls = message_obj.get("tool_calls") or []
            if not tool_calls:
                content = message_obj.get("content", "") or ""
                parsed_calls = self._extract_text_tool_calls(content)
                if parsed_calls:
                    tool_calls = parsed_calls
                    logger.info("Parsed %d tool calls from text output", len(parsed_calls))

            if not tool_calls:
                return {
                    "answer": message_obj.get("content", ""),
                    "tool_steps": tool_steps,
                    "model": api_endpoint.get("model"),
                    "timestamp": int(time.time())
                }

            if len(tool_calls) > 1:
                tool_calls = tool_calls[:1]

            messages.append({
                "role": "assistant",
                "content": message_obj.get("content", ""),
                "tool_calls": tool_calls
            })

            for idx, tool_call in enumerate(tool_calls):
                func = tool_call.get("function") or {}
                tool_name = func.get("name") or "web_search"
                raw_args = func.get("arguments", "{}")
                tool_call_id = tool_call.get("id") or f"tool_{step}_{idx}"

                try:
                    arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                except Exception:
                    arguments = {"query": raw_args}

                if isinstance(raw_args, str) and "query" in raw_args and "{" in raw_args:
                    queries = re.findall(r'"query"\s*:\s*"([^"]+)"', raw_args)
                    if queries:
                        arguments = {"search_queries": queries}

                if isinstance(arguments, dict):
                    if "search_queries" in arguments and isinstance(arguments["search_queries"], list):
                        arguments["search_queries"] = arguments["search_queries"][:2]

                if tool_name == "web_search":
                    tool_result = await handle_web_search_tool_call(arguments, max_results=5, news=True)
                else:
                    tool_result = f"Tool '{tool_name}' is not available."

                tool_steps.append({
                    "tool": tool_name,
                    "query": arguments.get("query") or arguments.get("search_queries"),
                    "result_preview": tool_result[:400],
                    "result": tool_result[:4000]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result
                })

        return {
            "error": "Assistant exceeded maximum tool steps without a final answer.",
            "tool_steps": tool_steps,
            "model": api_endpoint.get("model"),
            "timestamp": int(time.time())
        }


def _get_fact_sheet_context() -> str:
    """Load 2026 election fact sheet and candidate roster for AI context."""
    parts = []
    try:
        from .election_candidates import get_roster_for_ai
        roster = get_roster_for_ai()
        if roster:
            parts.append("Candidate roster (name -> party, state):\n" + roster)
    except Exception:
        pass
    if FACT_SHEET_PATH.exists():
        try:
            text = FACT_SHEET_PATH.read_text(encoding="utf-8")
            if len(text) > FACT_SHEET_MAX_CHARS:
                text = text[:FACT_SHEET_MAX_CHARS] + "\n\n[truncated]"
            parts.append("2026 election research fact sheet:\n" + text)
        except Exception:
            pass
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)


election_ai_service = ElectionAssistantService()
