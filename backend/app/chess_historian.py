"""
Chess Historian agent: chat about chess history, look up games and rivalries, optional PGN for loading.
Uses web search to research; can return PGN in ```pgn ... ``` for the frontend to load.
"""
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Queries used for the 30s fact ticker so each fact is grounded in real search results.
FACT_SEARCH_QUERIES = [
    "chess history famous player biography",
    "chess world championship history fact",
    "famous chess game history",
    "chess history scandal or controversy",
    "historical chess tournament",
]

SYSTEM_PROMPT = """You are a chess historian—warm, knowledgeable, and curious. You love telling stories about players, rivalries, and famous games.

You have access to web search. When you need to look something up, output exactly one line and nothing else:
SEARCH: <your search query>

When you MUST search (do not guess or invent):
- Any specific date, year, or "when" question (e.g. when someone died, when an event happened).
- Any specific person, scandal, or event the user asks about—verify with search.
- Whenever you are not certain. If in doubt, output SEARCH: with a clear query.

If you are completely confident and the question is very general (e.g. "what is the Italian Game?"), you may reply without searching. For specific claims about people, dates, or events, always search first.

When using search results:
- Base your answer only on what you found. Do not add details that are not in the results.
- Cite what you found. If you find a famous game or PGN, you may include it in a code block so they can load it: use exactly this format on its own line:
  ```pgn
  [full PGN here]
  ```
- Keep replies conversational and not too long unless they ask for depth.
- You do not evaluate positions or suggest moves—you're a historian, not an analyst. Stick to history, stories, and researched facts."""

def _build_fact_prompt(recent_facts: Optional[List[str]] = None, search_context: Optional[str] = None) -> str:
    parts = [
        "Give exactly one short, interesting fact about chess history.",
        "One or two sentences only. No preamble—just the fact. Base your fact ONLY on the research below; do not invent or guess dates, names, or events.",
    ]
    if search_context and search_context.strip():
        parts.append(
            "Use the following research. Pick one fact that is clearly supported; do not copy verbatim:\n" + (search_context[:4000] if len(search_context) > 4000 else search_context)
        )
    else:
        parts.append("No search results were available. Say something very general like 'Chess has a long and rich history' and keep it to one short sentence.")
    if recent_facts:
        cleaned = [f.strip() for f in recent_facts if f and isinstance(f, str) and len(f.strip()) > 10][-20:]
        if cleaned:
            parts.append(
                "CRITICAL: Do NOT repeat, rephrase, or give a fact about the same event/player as any of these "
                "recent facts. Choose a completely different subject and a different angle:\n"
                + "\n".join(f"- {t}" for t in cleaned)
            )
    return "\n".join(parts)


def _extract_search_query(reply: str) -> Optional[str]:
    """If the model requested a web search, return the query; else None."""
    if not reply:
        return None
    m = re.search(r"SEARCH:\s*(.+?)(?:\n|$)", reply.strip(), re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _extract_pgn_from_reply(text: str) -> Optional[str]:
    """Extract PGN from a ```pgn ... ``` block if present."""
    if not text:
        return None
    m = re.search(r"```pgn\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


async def _call_llm(
    request: Any,
    model_manager: Any,
    model_name: Optional[str],
    messages: List[Dict[str, str]],
    max_tokens: int = 1200,
    temperature: float = 0.6,
) -> str:
    """Call configured LLM; returns assistant content."""
    from .openai_compat import (
        is_api_endpoint,
        get_configured_endpoint,
        prepare_endpoint_request,
        forward_to_configured_endpoint_non_streaming,
    )
    from . import inference

    if not model_name:
        return "No model configured for the Chess Historian. Set a model in settings."
    if is_api_endpoint(model_name):
        endpoint_config = get_configured_endpoint(model_name)
        if not endpoint_config:
            return "Model not configured."
        request_data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            endpoint_config, url, prepared_data = prepare_endpoint_request(model_name, request_data)
            response_json = await forward_to_configured_endpoint_non_streaming(
                endpoint_config, url, prepared_data
            )
            if response_json.get("choices") and response_json["choices"]:
                msg = response_json["choices"][0].get("message") or response_json["choices"][0]
                return (msg.get("content") or msg.get("text") or "").strip()
        except Exception as e:
            logger.warning("Chess historian API call failed: %s", e)
            return f"Sorry, I couldn't complete that ({e})."
        return ""
    if model_manager:
        prompt = "\n\n".join(
            (m.get("content") or "").strip()
            for m in messages
        )
        try:
            response = await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                gpu_id=0,
            )
            if isinstance(response, dict):
                return (response.get("choices", [{}])[0].get("text") or "").strip()
            return (response or "").strip()
        except Exception as e:
            logger.warning("Chess historian local call failed: %s", e)
            return f"Sorry, I couldn't complete that ({e})."
    return "No model available."


async def chat(
    messages: List[Dict[str, str]],
    model_manager: Any,
    model_name: Optional[str],
    request: Any,
    web_search_fn: Any,
    persona_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Historian chat: agent-driven web search. The model decides when to search and what query to use.
    If the model outputs "SEARCH: <query>", we run the search and call the model again with the results.
    persona_prompt: optional custom persona prepended to the system prompt.
    Returns { reply, pgn? }.
    """
    if not messages:
        return {"reply": "Ask me anything about chess history, players, or famous games!", "pgn": None}
    system_content = SYSTEM_PROMPT
    if persona_prompt and persona_prompt.strip():
        system_content = f"Your persona: {persona_prompt.strip()}\n\n{system_content}"
    built: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        if role in ("user", "assistant"):
            built.append({"role": role, "content": (m.get("content") or "").strip()})

    max_rounds = 3
    reply = ""
    for _ in range(max_rounds):
        reply = await _call_llm(request, model_manager, model_name, built, max_tokens=1200, temperature=0.6)
        search_query = _extract_search_query(reply)
        if not search_query:
            break
        try:
            search_context = await web_search_fn(search_query, 6)
            if search_context and "No results found" not in search_context:
                search_context = search_context[:8000] if len(search_context) > 8000 else search_context
            else:
                search_context = "No results found for that query."
        except Exception as e:
            logger.warning("Historian web search failed: %s", e)
            search_context = "Web search failed. Answer from your knowledge or say you couldn't look it up."
        built.append({"role": "assistant", "content": reply})
        built.append({
            "role": "user",
            "content": f"Web search results for \"{search_query}\":\n\n{search_context}\n\nUsing the above, answer the user. Do not output SEARCH: again.",
        })

    pgn = _extract_pgn_from_reply(reply)
    if pgn:
        reply = re.sub(r"```pgn\s*\n[\s\S]*?```", "[Game PGN loaded below — use “Load this game” to open it.]", reply, flags=re.IGNORECASE).strip()
    return {"reply": reply or "I couldn't complete that. Try asking something else.", "pgn": pgn}


async def random_fact(
    model_manager: Any,
    model_name: Optional[str],
    request: Any,
    recent_facts: Optional[List[str]] = None,
    search_context: Optional[str] = None,
    web_search_fn: Optional[Any] = None,
) -> str:
    """Return one short chess history fact grounded in web search when possible.
    recent_facts: list of recently shown facts to avoid repeating.
    search_context: optional pre-fetched web search results.
    web_search_fn: if provided, we run a chess-history search and use results so the fact is researched, not hallucinated.
    """
    if web_search_fn and search_context is None:
        try:
            query = random.choice(FACT_SEARCH_QUERIES)
            search_context = await web_search_fn(query, max_results=6)
            if not search_context or "No results found" in (search_context or ""):
                search_context = None
        except Exception as e:
            logger.warning("Chess historian fact search failed: %s", e)
            search_context = None
    prompt = _build_fact_prompt(recent_facts=recent_facts, search_context=search_context)
    messages = [
        {"role": "system", "content": "You are a chess historian. Reply with only the fact, no greeting. Base the fact only on the research provided; do not invent dates, names, or events. Never repeat a fact you or the user have already seen in this session."},
        {"role": "user", "content": prompt},
    ]
    return await _call_llm(request, model_manager, model_name, messages, max_tokens=150, temperature=0.7)


PERSONA_PROMPT = """You are the Chess Historian—a character in an app that chats about chess history. In 2–4 short sentences, describe your own persona and style in first person. No preamble; output only the persona description."""


async def generate_persona(
    model_manager: Any,
    model_name: Optional[str],
    request: Any,
) -> str:
    """Generate a short persona description for the Chess Historian (for display/customisation)."""
    messages = [
        {"role": "system", "content": "You are the Chess Historian. Reply only with the requested persona text."},
        {"role": "user", "content": PERSONA_PROMPT},
    ]
    return await _call_llm(request, model_manager, model_name, messages, max_tokens=200, temperature=0.7)
