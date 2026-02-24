# agentic_memory.py - Optional character-scoped agentic memory (AI-generated insights, JSON backend)

"""
Optional agentic memory: when enabled on a character, an AI agent analyzes each
user/bot exchange and writes structured insights to a per-(user, character) JSON file.
Those insights are then injected into the system prompt for that character so you get
character-specific memory profiles across different chats.
"""

from typing import List, Dict, Any, Optional
import json
import os
import logging
import re
import datetime
import uuid

logger = logging.getLogger("agentic_memory")

try:
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _USER_MEMORY_DIR = os.path.join(_CURRENT_DIR, "user_memories")
    _AGENTIC_DIR = os.path.join(_USER_MEMORY_DIR, "agentic")
    os.makedirs(_AGENTIC_DIR, exist_ok=True)
except Exception as e:
    logger.warning(f"agentic_memory path setup: {e}")
    _AGENTIC_DIR = os.path.join(os.getcwd(), "app", "user_memories", "agentic")
    os.makedirs(_AGENTIC_DIR, exist_ok=True)


def _safe_id(raw: Optional[str]) -> str:
    if not raw or not isinstance(raw, str):
        return "unknown"
    return "".join(c for c in raw if c.isalnum() or c in ("-", "_")) or "unknown"


def get_agentic_memory_path(user_id: str, character_id: str) -> str:
    uid = _safe_id(user_id)
    cid = _safe_id(character_id)
    return os.path.join(_AGENTIC_DIR, f"{uid}_{cid}.json")


def get_agentic_profile(user_id: str, character_id: str) -> Dict[str, Any]:
    """Load the agentic memory profile for (user_id, character_id)."""
    path = get_agentic_memory_path(user_id, character_id)
    if not os.path.exists(path):
        logger.info(f"[Agentic Memory] GET profile: no file yet for user={user_id!r} char={character_id!r} -> 0 insights")
        return {"insights": [], "meta": {"updated_at": None}}

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"agentic_memory: failed to read {path}: {e}")
        return {"insights": [], "meta": {"updated_at": None}}

    insights = data.get("insights")
    if not isinstance(insights, list):
        insights = []
    meta = data.get("meta") or {}
    logger.info(f"[Agentic Memory] GET profile: user={user_id!r} char={character_id!r} -> {len(insights)} insights")
    return {"insights": insights, "meta": meta}


def save_agentic_profile(user_id: str, character_id: str, insights: List[Dict[str, Any]]) -> bool:
    """Overwrite the agentic profile with the given insights list."""
    path = get_agentic_memory_path(user_id, character_id)
    meta = {"updated_at": datetime.datetime.utcnow().isoformat() + "Z"}
    payload = {"insights": insights, "meta": meta}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if os.path.exists(path):
            os.replace(tmp, path)
        else:
            os.rename(tmp, path)
        logger.info(f"[Agentic Memory] SAVED profile: user={user_id!r} char={character_id!r} -> {len(insights)} insights")
        return True
    except Exception as e:
        logger.error(f"agentic_memory: failed to save {path}: {e}")
        return False


def add_agentic_insights(
    user_id: str,
    character_id: str,
    new_insights: List[Dict[str, Any]],
    max_insights: int = 200,
    dedupe_content: bool = True,
) -> int:
    """
    Append new insights to the profile. Deduplicates by content (case-insensitive).
    Trims to max_insights (keeps newest). Returns number added.
    """
    profile = get_agentic_profile(user_id, character_id)
    existing = profile["insights"]
    existing_contents = {s.get("content", "").strip().lower() for s in existing if s.get("content")} if dedupe_content else set()
    added = 0
    for ins in new_insights:
        if not isinstance(ins, dict) or not ins.get("content"):
            continue
        content = (ins.get("content") or "").strip()
        if not content or len(content) < 3:
            continue
        if dedupe_content and content.lower() in existing_contents:
            continue
        obj = {
            "id": ins.get("id") or f"ins_{uuid.uuid4().hex[:12]}",
            "content": content,
            "category": ins.get("category") or "insight",
            "importance": max(0.0, min(1.0, float(ins.get("importance", 0.7)))),
            "created_at": ins.get("created_at") or datetime.datetime.utcnow().isoformat() + "Z",
        }
        existing.append(obj)
        if dedupe_content:
            existing_contents.add(content.lower())
        added += 1
    if added > 0:
        newly_added = existing[-added:]  # capture before sort
        # Keep most recent
        existing.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        trimmed = existing[:max_insights]
        save_agentic_profile(user_id, character_id, trimmed)
        for obj in newly_added[:5]:
            logger.info(f"[Agentic Memory] + insight: {(obj.get('content') or '')[:80]!r}")
        if added > 5:
            logger.info(f"[Agentic Memory] + ... and {added - 5} more")
    return added


def format_agentic_context(insights: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """Format insights for injection into system prompt."""
    if not insights:
        return ""
    lines = []
    total = 0
    for s in insights[:50]:
        content = (s.get("content") or "").strip()
        if not content:
            continue
        line = f"• {content}"
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    if not lines:
        return ""
    return "[CHARACTER MEMORY - What this character remembers about the user]\n" + "\n".join(lines)


async def run_agentic_agent(
    model_manager,
    user_message: str,
    ai_response: str,
    character_name: str,
    existing_insights: List[Dict[str, Any]],
    gpu_id: int = 0,
    single_gpu_mode: bool = False,
    api_base_url: Optional[str] = None,
    api_model_name: Optional[str] = None,
):
    """
    Use the LLM to analyze the exchange and output new insights (function-calling style:
    we ask for a JSON array of insight objects; the model 'calls' add_insight by emitting JSON).
    When api_base_url and api_model_name are set, calls that API /generate instead of local model.
    Returns list of new insight dicts to add.
    """
    from . import inference
    import httpx

    existing_preview = "\n".join([f"- {s.get('content', '')[:80]}" for s in existing_insights[-15:]]) if existing_insights else "(none yet)"
    prompt = f"""You are an agent that maintains a memory profile about the user, specific to the character "{character_name}".

Your only job is to decide whether this conversation exchange reveals something new and useful to remember about the user (preferences, facts, context, relationships, goals). If so, output a JSON array of new insights. If nothing worth remembering, output an empty array [].

RULES:
- Only add factual or preference insights that will help the character interact better in future chats.
- Keep each insight one short sentence.
- Do not duplicate what is already in existing memories.
- Output ONLY a valid JSON array. No markdown, no explanation. Example: [{{"content": "User prefers tea over coffee", "category": "preference", "importance": 0.8}}]

EXISTING MEMORIES (recent):
{existing_preview}

CONVERSATION:
User: {user_message[:800]}
{character_name}: {ai_response[:800]}

NEW INSIGHTS (JSON array only):"""

    try:
        logger.info(f"[Agentic Memory] AGENT running for char={character_name!r} (exchange ~{len(user_message)+len(ai_response)} chars)")
        text = None
        if api_base_url and api_model_name:
            base = api_base_url.rstrip("/")
            url = f"{base}/generate"
            logger.info(f"[Agentic Memory] Using API {url!r} model={api_model_name!r}")
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    url,
                    json={
                        "prompt": prompt,
                        "model_name": api_model_name,
                        "max_tokens": 512,
                        "temperature": 0.2,
                        "repetition_penalty": 1.05,
                        "stream": False,
                        "gpu_id": gpu_id,
                        "request_purpose": "continuation",
                    },
                )
                r.raise_for_status()
                data = r.json()
                text = (data.get("text") or data.get("response") or "").strip()
        if not text:
            model_name = await model_manager.find_suitable_model(gpu_id=gpu_id) if model_manager else None
            if not model_name:
                logger.warning("[Agentic Memory] No API response and no local model — skip")
                return []
            logger.info(f"[Agentic Memory] Using local model {model_name!r} on gpu_id={gpu_id}")
            text = await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=prompt,
                max_tokens=512,
                temperature=0.2,
                repetition_penalty=1.05,
                gpu_id=gpu_id,
            )
        if not text or not isinstance(text, str):
            return []
        text = text.strip()
        # Strip markdown code block if present
        if "```" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        # Find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            return []
        raw = text[start:end]
        arr = json.loads(raw)
        if not isinstance(arr, list):
            return []
        new_insights = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if len(content) < 5:
                continue
            new_insights.append({
                "content": content,
                "category": item.get("category") or "insight",
                "importance": max(0.1, min(1.0, float(item.get("importance", 0.7)))),
            })
        logger.info(f"[Agentic Memory] AGENT parsed {len(new_insights)} new insight(s)")
        return new_insights
    except json.JSONDecodeError as e:
        logger.warning(f"[Agentic Memory] Agent output not valid JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"[Agentic Memory] Agent run failed: {e}", exc_info=True)
        return []
