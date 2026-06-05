# demo_showcase.py — install fabricated demo user / memories / agentic data for showcases

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("demo_showcase")

_PACK_PATH = Path(__file__).resolve().parent.parent.parent / "demo" / "showcase" / "pack.json"


def get_pack_path() -> Path:
    return _PACK_PATH


def load_pack() -> Dict[str, Any]:
    path = get_pack_path()
    if not path.is_file():
        raise FileNotFoundError(f"Demo showcase pack not found at {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Demo showcase pack must be a JSON object")
    return data


def get_demo_ids(pack: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    pack = pack or load_pack()
    ids = pack.get("ids") or {}
    profile_id = ids.get("profileId")
    character_id = ids.get("characterId")
    conversation_id = ids.get("conversationId")
    if not profile_id or not character_id:
        raise ValueError("Demo pack ids.profileId and ids.characterId are required")
    return {
        "profileId": str(profile_id),
        "characterId": str(character_id),
        "conversationId": str(conversation_id or ""),
    }


def get_status() -> Dict[str, Any]:
    from . import agentic_memory, memory_intelligence, user_utils

    try:
        pack = load_pack()
    except Exception as e:
        return {"available": False, "error": str(e)}

    ids = get_demo_ids(pack)
    user_id = ids["profileId"]
    char_id = ids["characterId"]

    memories = memory_intelligence.get_memory_store(user_id=user_id)
    agentic = agentic_memory.get_agentic_profile(user_id, char_id)
    insights = agentic.get("insights") or []
    active = user_utils.get_active_profile_id()

    expected_memories = len((pack.get("backend") or {}).get("profileMemories") or [])
    expected_insights = len((pack.get("backend") or {}).get("agenticInsights") or [])

    return {
        "available": True,
        "label": pack.get("label") or "Demo Showcase",
        "description": pack.get("description") or "",
        "ids": ids,
        "installed": len(memories) > 0 and len(insights) > 0,
        "memory_count": len(memories),
        "agentic_count": len(insights),
        "expected_memory_count": expected_memories,
        "expected_agentic_count": expected_insights,
        "active_profile_id": active,
        "demo_is_active": active == user_id,
    }


def install_backend(*, set_active: bool = True) -> Dict[str, Any]:
    from . import agentic_memory, memory_intelligence, user_utils

    pack = load_pack()
    ids = get_demo_ids(pack)
    backend = pack.get("backend") or {}
    user_id = ids["profileId"]
    char_id = ids["characterId"]

    memories = backend.get("profileMemories") or []
    insights = backend.get("agenticInsights") or []

    if not isinstance(memories, list) or not memories:
        raise ValueError("Demo pack backend.profileMemories must be a non-empty list")
    if not isinstance(insights, list) or not insights:
        raise ValueError("Demo pack backend.agenticInsights must be a non-empty list")

    ok_mem = memory_intelligence.save_memory_store(memories, user_id=user_id)
    ok_agentic = agentic_memory.save_agentic_profile(user_id, char_id, insights)
    if not ok_mem or not ok_agentic:
        raise RuntimeError("Failed to write demo memory or agentic files")

    if set_active:
        user_utils.save_active_profile_id(user_id)

    logger.info(
        "Demo showcase installed: user=%s char=%s memories=%s insights=%s",
        user_id,
        char_id,
        len(memories),
        len(insights),
    )
    return {
        "status": "success",
        "user_id": user_id,
        "character_id": char_id,
        "memories_saved": len(memories),
        "insights_saved": len(insights),
        "set_active": set_active,
    }
