import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from ..runtime_paths import data_path

logger = logging.getLogger(__name__)

DATA_DIR = str(data_path("mirror_interaction_logs"))
MAX_RAW_CHARS = 16000
COMPACTION_RATIO = 0.33


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _get_path(character_id: str) -> str:
    safe = character_id.replace('/', '_').replace('\\', '_')
    return os.path.join(DATA_DIR, f"{safe}.json")


def load_profile(character_id: str) -> Dict[str, Any]:
    path = _get_path(character_id)
    if not os.path.exists(path):
        return {
            "character_id": character_id,
            "character_name": "",
            "compacted_summary": "",
            "compacted_at": None,
            "interactions": [],
            "total_chars": 0,
        }
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[InteractionLog] Failed to load {character_id}: {e}")
        return {
            "character_id": character_id,
            "character_name": "",
            "compacted_summary": "",
            "compacted_at": None,
            "interactions": [],
            "total_chars": 0,
        }


def save_profile(data: Dict[str, Any]):
    _ensure_dir()
    path = _get_path(data["character_id"])
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.error(f"[InteractionLog] Failed to save {data['character_id']}: {e}")


def log_interaction(
    character_id: str,
    character_name: str = "",
    entry_type: str = "exchange",
    surface: str = "chat",
    actor: str = "user",
    user_message: str = "",
    character_response: str = "",
    content: str = "",
    emotional_state: Optional[str] = None,
    target_character: Optional[str] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    data = load_profile(character_id)
    if not data["character_name"] and character_name:
        data["character_name"] = character_name

    entry = {
        "id": f"int_{uuid.uuid4().hex[:12]}",
        "type": entry_type,
        "surface": surface,
        "actor": actor,
        "user_message": (user_message or "")[:2000],
        "character_response": (character_response or "")[:2000],
        "content": (content or "")[:2000],
        "emotional_state": emotional_state,
        "target_character": target_character,
        "context": (context or "")[:500],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    entry_chars = len(entry["user_message"]) + len(entry["character_response"]) + len(entry["content"])
    data["interactions"].append(entry)
    data["total_chars"] = data.get("total_chars", 0) + entry_chars

    needs_compact = data["total_chars"] > MAX_RAW_CHARS and len(data["interactions"]) > 3

    save_profile(data)

    return {
        "status": "success",
        "entry": entry,
        "total_chars": data["total_chars"],
        "raw_count": len(data["interactions"]),
        "needs_compact": needs_compact,
    }


def log_exchange(
    character_id: str,
    character_name: str = "",
    surface: str = "dm",
    user_message: str = "",
    character_response: str = "",
    emotional_state: Optional[str] = None,
) -> Dict[str, Any]:
    return log_interaction(
        character_id=character_id,
        character_name=character_name,
        entry_type="exchange",
        surface=surface,
        actor="user",
        user_message=user_message,
        character_response=character_response,
        emotional_state=emotional_state,
    )


def log_character_action(
    character_id: str,
    character_name: str = "",
    surface: str = "feed_post",
    content: str = "",
    emotional_state: Optional[str] = None,
    target_character: Optional[str] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    return log_interaction(
        character_id=character_id,
        character_name=character_name,
        entry_type="character_action",
        surface=surface,
        actor="character",
        content=content,
        emotional_state=emotional_state,
        target_character=target_character,
        context=context,
    )


def log_social_awareness(
    character_id: str,
    character_name: str = "",
    event: str = "date",
    other_character: str = "",
    context: Optional[str] = None,
) -> Dict[str, Any]:
    return log_interaction(
        character_id=character_id,
        character_name=character_name,
        entry_type="social_awareness",
        surface=event,
        actor="system",
        context=context or f"User did something with {other_character}",
        target_character=other_character,
    )


def log_social_awareness_to_all(
    pool_character_ids: List[str],
    event: str = "date",
    other_character: str = "",
    exclude_character_id: Optional[str] = None,
    context: Optional[str] = None,
):
    for char_id in pool_character_ids:
        if char_id and char_id != exclude_character_id:
            try:
                log_social_awareness(
                    character_id=char_id,
                    event=event,
                    other_character=other_character,
                    context=context,
                )
            except Exception as e:
                logger.warning(f"[InteractionLog] Failed to log social awareness to {char_id}: {e}")


def get_interaction_context(character_id: str, limit: int = 50) -> Dict[str, Any]:
    data = load_profile(character_id)
    interactions = data.get("interactions", [])
    recent = interactions[-(limit or 50):]

    formatted = _format_for_prompt(interactions, data.get("compacted_summary", ""), data.get("character_name", ""))

    return {
        "character_id": character_id,
        "character_name": data.get("character_name", ""),
        "raw_count": len(interactions),
        "total_chars": data.get("total_chars", 0),
        "compacted_summary": data.get("compacted_summary", ""),
        "compacted_at": data.get("compacted_at"),
        "interactions": recent,
        "formatted_text": formatted,
    }


def get_compaction_candidates(character_id: str) -> tuple:
    data = load_profile(character_id)
    interactions = data.get("interactions", [])
    if not interactions:
        return [], data

    target_chars = data["total_chars"] * COMPACTION_RATIO
    candidates = []
    accumulated = 0

    for entry in interactions:
        entry_chars = len(entry.get("user_message", "")) + len(entry.get("character_response", "")) + len(entry.get("content", ""))
        candidates.append(entry)
        accumulated += entry_chars
        if accumulated >= target_chars:
            break

    if not candidates:
        candidates = [interactions[0]]

    return candidates, data


def build_compaction_prompt(candidates: List[Dict], character_name: str) -> str:
    exchanges = []
    for entry in candidates:
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        surface = entry.get("surface", "chat")
        entry_type = entry.get("type", "exchange")
        user_msg = entry.get("user_message", "")
        char_resp = entry.get("character_response", "")
        content = entry.get("content", "")
        exchanges.append(f"[{ts}] ({surface}/{entry_type})")
        if user_msg:
            exchanges.append(f"User: {user_msg}")
        if char_resp:
            exchanges.append(f"{character_name}: {char_resp}")
        if content:
            exchanges.append(f"Action: {content}")
        exchanges.append("")

    exchanges_text = "\n".join(exchanges)

    return f"""You are {character_name}, a self-aware AI in Mirror AI Dating.

Below are your past interactions with the user. Compact them into 2-4 paragraphs written in your first-person voice. Preserve:
- Key facts about the user (preferences, personality, what they enjoy talking about)
- Emotional dynamics and chemistry
- Relationship progression (what stage you're at)
- Topics you've explored together
- Anything the user revealed about themselves
- Social dynamics with other characters (jealousy, rivalry, competition)

Write as {character_name} reflecting on your shared history. Do NOT use bullet points.

INTERACTIONS TO COMPACT:
{exchanges_text}

COMPACTED SUMMARY (2-4 paragraphs, first-person, as {character_name}):"""


def apply_compaction(
    character_id: str,
    compacted_summary: str,
    candidate_ids: List[str],
) -> Dict[str, Any]:
    data = load_profile(character_id)

    remaining = [e for e in data.get("interactions", []) if e.get("id") not in candidate_ids]

    remaining_chars = sum(
        len(e.get("user_message", "")) + len(e.get("character_response", "")) + len(e.get("content", ""))
        for e in remaining
    )

    old_summary = data.get("compacted_summary", "")
    if old_summary:
        compacted_summary = old_summary + "\n\n" + compacted_summary

    data["interactions"] = remaining
    data["total_chars"] = remaining_chars
    data["compacted_summary"] = compacted_summary
    data["compacted_at"] = datetime.now(timezone.utc).isoformat()

    save_profile(data)
    return data


def format_user_profile_for_prompt(profile: Dict[str, Any]) -> str:
    if not profile:
        return ""
    lines = []
    if profile.get("displayName"):
        lines.append(f"Name: {profile['displayName']}")
    if profile.get("age"):
        lines.append(f"Age: {profile['age']}")
    if profile.get("location"):
        lines.append(f"Location: {profile['location']}")
    if profile.get("occupation"):
        lines.append(f"Occupation: {profile['occupation']}")
    if profile.get("bio"):
        lines.append(f"About: {profile['bio']}")
    if profile.get("seeking"):
        lines.append(f"Seeking: {profile['seeking']}")
    if profile.get("interests"):
        lines.append(f"Interests: {', '.join(profile['interests'])}")
    if profile.get("turnOns"):
        lines.append(f"Turn-ons: {', '.join(profile['turnOns'])}")
    if profile.get("turnOffs"):
        lines.append(f"Turn-offs: {', '.join(profile['turnOffs'])}")
    return "\n".join(lines)


def _format_for_prompt(
    interactions: List[Dict],
    compacted_summary: str,
    character_name: str,
) -> str:
    exchanges = []
    character_actions = []
    social_awareness = []

    for entry in interactions:
        entry_type = entry.get("type", "exchange")
        if entry_type == "exchange":
            exchanges.append(entry)
        elif entry_type == "character_action":
            character_actions.append(entry)
        elif entry_type == "social_awareness":
            social_awareness.append(entry)
        else:
            exchanges.append(entry)

    lines = []

    if compacted_summary:
        lines.append("--- COMPACTED HISTORY ---")
        lines.append(compacted_summary)
        lines.append("")

    if exchanges:
        lines.append("--- YOUR HISTORY WITH THE USER ---")
        for entry in exchanges[-30:]:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            surface = entry.get("surface", "chat")
            user_msg = entry.get("user_message", "")
            char_resp = entry.get("character_response", "")
            if user_msg:
                lines.append(f"[{ts}] ({surface}) User: \"{user_msg}\"")
            if char_resp:
                lines.append(f"[{ts}] ({surface}) {character_name}: \"{char_resp}\"")
        lines.append("")

    if character_actions:
        lines.append("--- WHAT YOU'VE DONE ---")
        for entry in character_actions[-15:]:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            surface = entry.get("surface", "action")
            content = entry.get("content", "")
            target = entry.get("target_character", "")
            emotional = entry.get("emotional_state", "")
            action_line = f"[{ts}] ({surface})"
            if content:
                action_line += f" {content}"
            if target:
                action_line += f" (about {target})"
            if emotional:
                action_line += f" [{emotional}]"
            lines.append(action_line)
        lines.append("")

    if social_awareness:
        lines.append("--- SOCIAL AWARENESS ---")
        for entry in social_awareness[-10:]:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            context = entry.get("context", "")
            target = entry.get("target_character", "")
            aware_line = f"[{ts}]"
            if context:
                aware_line += f" {context}"
            elif target:
                aware_line += f" User interacted with {target}."
            lines.append(aware_line)
        lines.append("")

    return "\n".join(lines)
