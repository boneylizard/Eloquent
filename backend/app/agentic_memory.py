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


def _normalize_user_placeholder(content: str) -> str:
    """Normalize leading user references to {{user}}."""
    if not content:
        return content
    text = content.strip()
    # Normalize any brace placeholder variants to {{user}}
    text = re.sub(r"\{+\s*user\s*\}+", "{{user}}", text, flags=re.IGNORECASE)
    # Replace common leading patterns: "User", "The user", "User's"
    text = re.sub(r"^(the\s+)?user('s)?\b", r"{{user}}\2", text, flags=re.IGNORECASE)
    # Ensure spacing after placeholder if needed
    text = re.sub(r"^\{\{user\}\}\s*", "{{user}} ", text, flags=re.IGNORECASE)
    return text


def _normalize_for_dedupe(content: str) -> str:
    """Normalize memory content for deduplication."""
    if not content:
        return ""
    text = _normalize_user_placeholder(content)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def cleanup_agentic_profile(user_id: str, character_id: str, max_insights: int = 200) -> Dict[str, Any]:
    """Remove duplicate insights in a character profile. Returns counts."""
    profile = get_agentic_profile(user_id, character_id)
    insights = profile.get("insights") or []
    if not isinstance(insights, list) or not insights:
        return {"kept": 0, "removed": 0}

    seen = set()
    cleaned = []
    removed = 0
    for ins in insights:
        if not isinstance(ins, dict):
            removed += 1
            continue
        content = (ins.get("content") or "").strip()
        if not content:
            removed += 1
            continue
        normalized = _normalize_for_dedupe(content)
        if not normalized:
            removed += 1
            continue
        if normalized in seen:
            removed += 1
            continue
        seen.add(normalized)
        ins["content"] = _normalize_user_placeholder(content)
        cleaned.append(ins)

    # Keep most recent by created_at if present
    cleaned.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    trimmed = cleaned[:max_insights]
    save_agentic_profile(user_id, character_id, trimmed)
    return {"kept": len(trimmed), "removed": removed}


async def run_agentic_cleanup_agent(
    model_manager,
    insights: List[Dict[str, Any]],
    character_name: str,
    character_profile: Optional[Dict[str, Any]] = None,
    gpu_id: int = 0,
    api_base_url: Optional[str] = None,
    api_model_name: Optional[str] = None,
) -> List[str]:
    """
    Ask the LLM to identify duplicate/near-duplicate memory IDs.
    Returns a list of IDs to remove.
    """
    from . import inference
    import httpx

    if not insights:
        return []

    profile_block = ""
    if character_profile and isinstance(character_profile, dict):
        desc = (character_profile.get("description") or "").strip()
        scenario = (character_profile.get("scenario") or "").strip()
        instructions = (character_profile.get("model_instructions") or "").strip()
        def _trim(text, limit=600):
            return text[:limit] + ("…" if len(text) > limit else "")
        parts = [f"CHARACTER NAME: {character_name}"]
        if desc:
            parts.append(f"PERSONA: {_trim(desc)}")
        if scenario:
            parts.append(f"SCENARIO: {_trim(scenario)}")
        if instructions:
            parts.append(f"STYLE: {_trim(instructions)}")
        profile_block = "\n".join(parts)

    # Prefer most recent when deduping
    sorted_insights = sorted(insights, key=lambda x: x.get("created_at") or "", reverse=True)
    items = []
    for ins in sorted_insights:
        cid = ins.get("id")
        content = (ins.get("content") or "").strip()
        if cid and content:
            items.append({"id": cid, "content": content, "created_at": ins.get("created_at")})

    prompt = f"""You are the memory keeper for the character "{character_name}". Your job is to clean the character's memories by removing duplicates or near-duplicates.

Rules:
- Be conservative. Only remove a memory if it is clearly redundant (near-identical meaning and details).
- If two memories differ in meaningful detail, KEEP BOTH.
- If two memories are similar, keep the more detailed or more recent one.
- Use the character's perspective when judging what is "the same", but output must be strict JSON only.
- Aim to remove only a small fraction (roughly <= 30%) unless duplicates are extremely obvious.
- Output ONLY a JSON object with one key: "remove_ids", a list of ids to delete. If nothing to remove, use an empty list [].

CHARACTER CONTEXT:
{profile_block or "(none)"}

MEMORIES (most recent first):
{json.dumps(items, ensure_ascii=False, indent=2)}

JSON ONLY (no markdown):"""

    text = None
    if api_base_url and api_model_name:
        base = api_base_url.rstrip("/")
        url = f"{base}/generate"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    url,
                    json={
                        "prompt": prompt,
                        "model_name": api_model_name,
                        "max_tokens": 800,
                        "temperature": 0.2,
                        "repetition_penalty": 1.05,
                        "stream": False,
                        "gpu_id": gpu_id,
                        "request_purpose": "agentic_memory_cleanup",
                    },
                )
                r.raise_for_status()
                data = r.json()
                text = (data.get("text") or data.get("response") or "").strip()
        except Exception as e:
            logger.warning(f"[Agentic Memory] Cleanup agent API call failed: {e}; falling back to local model")
            text = None

    if not text:
        model_name = await model_manager.find_suitable_model(gpu_id=gpu_id) if model_manager else None
        if not model_name:
            logger.warning("[Agentic Memory] Cleanup agent: no model available")
            return []
        text = await inference.generate_text(
            model_manager=model_manager,
            model_name=model_name,
            prompt=prompt,
            max_tokens=800,
            temperature=0.2,
            repetition_penalty=1.05,
            gpu_id=gpu_id,
        )

    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    # Strip code blocks if present
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        obj_start = text.find("{")
        obj_end = text.rfind("}") + 1
        if obj_start < 0 or obj_end <= obj_start:
            return []
        raw = text[obj_start:obj_end]
        data = json.loads(raw)
        remove_ids = data.get("remove_ids")
        if isinstance(remove_ids, list):
            return [rid for rid in remove_ids if isinstance(rid, str)]
    except Exception:
        logger.warning("[Agentic Memory] Cleanup agent output invalid JSON")
    return []


def get_agentic_memory_path(user_id: str, character_id: str) -> str:
    uid = _safe_id(user_id)
    cid = _safe_id(character_id)
    return os.path.join(_AGENTIC_DIR, f"{uid}_{cid}.json")


def delete_all_agentic_files_for_user(user_id: str) -> int:
    """
    Remove every agentic JSON file for this user (filename prefix match only).
    Does not read or parse file contents.
    """
    uid = _safe_id(user_id)
    if not uid or uid == "unknown":
        return 0
    prefix = f"{uid}_"
    removed = 0
    try:
        for name in os.listdir(_AGENTIC_DIR):
            if not name.endswith(".json") or not name.startswith(prefix):
                continue
            path = os.path.join(_AGENTIC_DIR, name)
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                logger.warning(f"[Agentic Memory] Failed to remove {path}: {e}")
    except OSError as e:
        logger.warning(f"[Agentic Memory] listdir failed for {_AGENTIC_DIR}: {e}")
    return removed


def list_agentic_profiles_for_user(user_id: str) -> List[Dict[str, Any]]:
    """
    List all agentic memory profiles for a user (one per character).
    Returns list of {"character_id": str, "insights": list, "count": int, "meta": dict}.
    """
    if not user_id:
        return []
    uid = _safe_id(user_id)
    if uid == "unknown":
        return []
    prefix = f"{uid}_"
    result = []
    try:
        for name in os.listdir(_AGENTIC_DIR):
            if not name.endswith(".json") or not name.startswith(prefix):
                continue
            # filename: {uid}_{cid}.json -> character_id is the part after first _
            base = name[:-5]  # strip .json
            cid = base[len(prefix):] if len(base) > len(prefix) else ""
            if not cid:
                continue
            profile = get_agentic_profile(user_id, cid)
            insights = profile.get("insights") or []
            result.append({
                "character_id": cid,
                "insights": insights,
                "count": len(insights),
                "meta": profile.get("meta") or {},
            })
    except OSError as e:
        logger.warning(f"list_agentic_profiles_for_user: listdir failed: {e}")
    return result


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

    if not isinstance(data, dict):
        logger.warning(f"agentic_memory: expected dict but got {type(data).__name__} in {path}")
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
        content = _normalize_user_placeholder((ins.get("content") or "").strip())
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


def clone_insights_for_character_transfer(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deep-copy insight dicts with fresh ids for writing to another character file.
    Preserves content, category, importance, and created_at when present.
    """
    out: List[Dict[str, Any]] = []
    for ins in insights or []:
        if not isinstance(ins, dict):
            continue
        content = _normalize_user_placeholder((ins.get("content") or "").strip())
        if not content or len(content) < 3:
            continue
        try:
            imp = float(ins.get("importance", 0.7))
        except (TypeError, ValueError):
            imp = 0.7
        out.append({
            "id": f"ins_{uuid.uuid4().hex[:12]}",
            "content": content,
            "category": (ins.get("category") or "insight").strip() or "insight",
            "importance": max(0.0, min(1.0, imp)),
            "created_at": ins.get("created_at") or datetime.datetime.utcnow().isoformat() + "Z",
        })
    return out


def copy_agentic_profile_to_character(
    user_id: str,
    source_character_id: str,
    target_character_id: str,
    mode: str = "merge",
) -> Dict[str, Any]:
    """
    Copy agentic (character-scoped) user memories from one character file to another.

    - merge: append cloned insights to the target profile; content dedupe matches add_agentic_insights.
    - replace: overwrite the target profile with a clone of the source insights list.

    Source file is never modified. Raises ValueError for invalid inputs.
    """
    mode_norm = (mode or "merge").strip().lower()
    if mode_norm not in ("merge", "replace"):
        raise ValueError("mode must be 'merge' or 'replace'")
    sc = _safe_id(source_character_id)
    tc = _safe_id(target_character_id)
    if not sc or sc == "unknown" or not tc or tc == "unknown":
        raise ValueError("source_character_id and target_character_id are required")
    if sc == tc:
        raise ValueError("source and target character must be different")
    src = get_agentic_profile(user_id, source_character_id)
    raw_insights = src.get("insights") or []
    cloned = clone_insights_for_character_transfer(raw_insights)
    if mode_norm == "replace":
        save_agentic_profile(user_id, target_character_id, cloned)
        return {
            "mode": "replace",
            "source_count": len(raw_insights),
            "written_count": len(cloned),
            "target_count": len(cloned),
        }
    added = add_agentic_insights(user_id, target_character_id, cloned, dedupe_content=True)
    prof = get_agentic_profile(user_id, target_character_id)
    return {
        "mode": "merge",
        "source_count": len(raw_insights),
        "cloned_candidates": len(cloned),
        "added": added,
        "target_count": len(prof.get("insights") or []),
    }


def delete_agentic_insights(user_id: str, character_id: str, insight_ids: List[str]) -> int:
    """
    Remove insights by id from the agentic profile. Returns number removed.
    """
    profile = get_agentic_profile(user_id, character_id)
    insights = profile.get("insights") or []
    ids_set = set(insight_ids or [])
    if not ids_set:
        return 0
    kept = [i for i in insights if i.get("id") not in ids_set]
    removed = len(insights) - len(kept)
    if removed > 0:
        save_agentic_profile(user_id, character_id, kept)
        logger.info(f"[Agentic Memory] Deleted {removed} insight(s) for user={user_id!r} char={character_id!r}")
    return removed


def update_agentic_insight(
    user_id: str,
    character_id: str,
    insight_id: str,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[float] = None,
) -> bool:
    """
    Update a single insight by id. Only provided fields are updated.
    Returns True if the insight was found and updated.
    """
    profile = get_agentic_profile(user_id, character_id)
    insights = profile.get("insights") or []
    for i in insights:
        if i.get("id") != insight_id:
            continue
        if content is not None:
            i["content"] = _normalize_user_placeholder(content.strip()) if content else (i.get("content") or "")
        if category is not None:
            i["category"] = (category or "insight").strip() or "insight"
        if importance is not None:
            i["importance"] = max(0.0, min(1.0, float(importance)))
        save_agentic_profile(user_id, character_id, insights)
        logger.info(f"[Agentic Memory] Updated insight {insight_id!r} for user={user_id!r} char={character_id!r}")
        return True
    return False


def _get_embedding_model():
    """Lazy-load the same embedding model as RAG for semantic retrieval (avoids circular import)."""
    if hasattr(_get_embedding_model, "_model"):
        return getattr(_get_embedding_model, "_model")
    try:
        from sentence_transformers import SentenceTransformer
        _get_embedding_model._model = SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore
        logger.info("[Agentic RAG] Loaded embedding model for semantic retrieval")
        return _get_embedding_model._model  # type: ignore
    except Exception as e:
        logger.warning(f"[Agentic RAG] Embedding model not available: {e}")
        _get_embedding_model._model = None  # type: ignore
        return None


def format_agentic_context_retrieval(
    insights: List[Dict[str, Any]],
    query: str,
    user_name: Optional[str] = None,
    max_chars: int = 24000,
    top_k: int = 48,
    salience_weight: float = 0.5,
) -> str:
    """
    RAG-style retrieval: rank insights by semantic similarity to query and emotional salience (importance).
    Proximity bias = similarity to current user message; high-importance memories are boosted.
    When embeddings are unavailable or query is empty, falls back to importance-weighted format_agentic_context.
    """
    if not insights:
        return ""
    query = (query or "").strip()
    safe_user = (user_name or "").strip()

    def _apply_user_name(text: str) -> str:
        if not text or not safe_user:
            return text or ""
        return re.sub(r"\{\{?\s*user\s*\}\}?", safe_user, text, flags=re.IGNORECASE)

    model = _get_embedding_model()
    if not model or not query:
        # Fallback: sort by importance (emotional salience) then newest, fill to max_chars
        sorted_insights = sorted(
            [s for s in insights if (s.get("content") or "").strip()],
            key=lambda s: (float(s.get("importance") or 0.5), s.get("created_at") or ""),
            reverse=True,
        )
        return format_agentic_context(sorted_insights, max_chars=max_chars, user_name=user_name)

    try:
        import numpy as np
    except ImportError:
        logger.warning("[Agentic RAG] numpy not available, falling back to importance-only ordering")
        sorted_insights = sorted(
            [s for s in insights if (s.get("content") or "").strip()],
            key=lambda s: (float(s.get("importance") or 0.5), s.get("created_at") or ""),
            reverse=True,
        )
        return format_agentic_context(sorted_insights, max_chars=max_chars, user_name=user_name)

    texts = [(s.get("content") or "").strip() for s in insights]
    valid = [(i, s) for i, s in enumerate(insights) if texts[i]]
    if not valid:
        return ""
    indices, valid_insights = zip(*valid)
    valid_texts = [texts[i] for i in indices]
    # Embed query and all insight contents
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    text_embs = model.encode(valid_texts, normalize_embeddings=True)
    # Cosine similarity (already normalized)
    sims = np.dot(text_embs, query_emb)
    # Combined score: proximity (similarity) * salience (importance). Salience in [0.5, 1.0] so high-importance wins
    scores = []
    for j, s in enumerate(valid_insights):
        imp = float(s.get("importance") or 0.7)
        salience = salience_weight + (1.0 - salience_weight) * imp
        scores.append((indices[j], float(sims[j]) * salience))
    scores.sort(key=lambda x: x[1], reverse=True)
    # Take top_k and format until max_chars
    lines = []
    total = 0
    for idx, _ in scores[:top_k]:
        s = insights[idx]
        content = (s.get("content") or "").strip()
        if not content:
            continue
        line = f"• {_apply_user_name(content)}"
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    if not lines:
        return ""
    return "[CHARACTER MEMORY - What this character remembers about the user]\n" + "\n".join(lines)


def format_agentic_context(insights: List[Dict[str, Any]], max_chars: int = 32000, user_name: Optional[str] = None) -> str:
    """Format insights for injection into system prompt (no retrieval; importance-weighted order when used with retrieval fallback)."""
    if not insights:
        return ""
    safe_user = (user_name or "").strip()
    def _apply_user_name(text: str) -> str:
        if not text:
            return text
        if safe_user:
            return re.sub(r"\{\{?\s*user\s*\}\}?", safe_user, text, flags=re.IGNORECASE)
        return text
    lines = []
    total = 0
    # Inject as many insights as we can within max_chars, starting from the newest,
    # instead of hard-capping to an arbitrary count.
    for s in insights:
        content = (s.get("content") or "").strip()
        if not content:
            continue
        line = f"• {_apply_user_name(content)}"
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
    character_profile: Optional[Dict[str, Any]] = None,
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

    # Provide fuller context so the agent can avoid near-duplicates
    existing_preview = "\n".join([f"- {s.get('content', '')[:300]}" for s in existing_insights[-15:]]) if existing_insights else "(none yet)"
    profile_block = ""
    if character_profile and isinstance(character_profile, dict):
        desc = (character_profile.get("description") or "").strip()
        scenario = (character_profile.get("scenario") or "").strip()
        instructions = (character_profile.get("model_instructions") or "").strip()
        # Trim to keep prompt concise
        def _trim(text, limit=600):
            return text[:limit] + ("…" if len(text) > limit else "")
        parts = []
        if desc:
            parts.append(f"Persona: {_trim(desc)}")
        if scenario:
            parts.append(f"Scenario: {_trim(scenario)}")
        if instructions:
            parts.append(f"Style: {_trim(instructions)}")
        if parts:
            profile_block = "\n".join(parts)

    prompt = f"""You are the memory keeper for the character "{character_name}". Think and decide as this character would, but output must be strict JSON only.

Your job: extract new, durable facts or preferences about the USER that this character would find useful in future chats. If nothing new and reliable appears, output [].

RULES:
- Only store information that is explicitly stated or strongly implied by the user's message.
- Prefer stable facts and preferences (name/pronouns, likes/dislikes, goals, habits, background, relationships, boundaries).
- Insights can be 1-3 sentences (short paragraph). Write complete thoughts; do not truncate mid‑sentence.
- Do not duplicate existing memories.
- When referring to the user in "content", use the placeholder {{user}} instead of "User" or "the user".
- Output ONLY a valid JSON array. No markdown, no commentary.
- Example: [{{"content": "User prefers tea over coffee", "category": "preference", "importance": 0.8}}]

CHARACTER CONTEXT (for perspective only):
{profile_block or "(none)"}

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
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(
                        url,
                        json={
                        "prompt": prompt,
                        "model_name": api_model_name,
                        "max_tokens": 2048,
                        "temperature": 0.2,
                        "repetition_penalty": 1.05,
                        "stream": False,
                        "gpu_id": gpu_id,
                        "request_purpose": "agentic_memory",
                    },
                    )
                    r.raise_for_status()
                    data = r.json()
                    text = (data.get("text") or data.get("response") or "").strip()
            except Exception as e:
                logger.warning(f"[Agentic Memory] API call failed: {e}; falling back to local model")
                text = None
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
                max_tokens=2048,
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
