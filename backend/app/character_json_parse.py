"""Robust JSON parse/repair for auto character generation."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .chatlog_condenser import (
    extract_first_json,
    json_closing_suffix,
    repair_truncated_json_blob,
)

logger = logging.getLogger("character_json_parse")

CHARACTER_SCHEMA_HINT = """{
  "name": "string",
  "description": "string",
  "model_instructions": "string",
  "scenario": "string",
  "first_message": "string",
  "example_dialogue": [{"role": "user"|"character", "content": "string"}],
  "loreEntries": [{"content": "string", "keywords": ["string"]}]
}"""

REQUIRED_FIELDS = ("name", "description")
OPTIONAL_DEFAULTS = {
    "model_instructions": "",
    "scenario": "",
    "first_message": "",
    "example_dialogue": [],
    "loreEntries": [],
}

# Salvage regexes for truncated / broken model output
_FIELD_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("name", re.compile(r'"name"\s*:\s*"((?:\\.|[^"\\])*)"', re.I)),
    ("description", re.compile(r'"description"\s*:\s*"((?:\\.|[^"\\])*)"', re.I | re.S)),
    (
        "model_instructions",
        re.compile(r'"model_instructions"\s*:\s*"((?:\\.|[^"\\])*)"', re.I | re.S),
    ),
    ("scenario", re.compile(r'"scenario"\s*:\s*"((?:\\.|[^"\\])*)"', re.I | re.S)),
    (
        "first_message",
        re.compile(r'"first_message"\s*:\s*"((?:\\.|[^"\\])*)"', re.I | re.S),
    ),
]


def _strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.I)
    if fence:
        return fence.group(1).strip()
    return cleaned


def _decode_json_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", "\n")


def _normalize_character_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    for key, default in OPTIONAL_DEFAULTS.items():
        if key not in out or out[key] is None:
            out[key] = default if not isinstance(default, list) else list(default)
    if not isinstance(out.get("example_dialogue"), list):
        out["example_dialogue"] = []
    if not isinstance(out.get("loreEntries"), list):
        out["loreEntries"] = []
    name = (out.get("name") or "").strip()
    if name and not (out.get("first_message") or "").strip():
        out["first_message"] = f"Hello! I'm {name}."
    return out


def _has_required_fields(data: Dict[str, Any]) -> bool:
    return all(
        isinstance(data.get(field), str) and data.get(field, "").strip()
        for field in REQUIRED_FIELDS
    )


def salvage_character_fields(blob: str) -> Dict[str, Any]:
    """Extract character fields from broken JSON via regex."""
    salvaged: Dict[str, Any] = {}
    for field, pattern in _FIELD_PATTERNS:
        match = pattern.search(blob or "")
        if match:
            salvaged[field] = _decode_json_string(match.group(1)).strip()
    if salvaged:
        salvaged = _normalize_character_dict(salvaged)
        salvaged["_salvaged"] = True
    return salvaged


def character_json_is_usable(data: Optional[Dict[str, Any]], *, partial_ok: bool = False) -> bool:
    if not data or not isinstance(data, dict):
        return False
    if _has_required_fields(data):
        return True
    if partial_ok:
        name = (data.get("name") or "").strip()
        desc = (data.get("description") or "").strip()
        return bool(name or desc)
    return False


def parse_character_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Parse model output into character JSON.

    Returns (character, partial, salvaged, error_message).
    """
    cleaned = _strip_fences(raw or "")
    if not cleaned:
        return None, None, False, "Empty model response"

    blob = extract_first_json(cleaned) or cleaned
    candidates = [blob]
    if blob:
        candidates.append(repair_truncated_json_blob(blob))

    last_error: Optional[str] = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                normalized = _normalize_character_dict(parsed)
                if _has_required_fields(normalized):
                    return normalized, None, False, None
                if character_json_is_usable(normalized, partial_ok=True):
                    return None, normalized, False, "Missing required name or description"
                last_error = "Parsed JSON missing required fields"
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)

    partial = salvage_character_fields(blob or cleaned)
    if character_json_is_usable(partial, partial_ok=True):
        return None, partial, True, last_error or "JSON repair failed; partial fields salvaged"

    return None, None, False, last_error or "Could not parse character JSON"


def build_character_repair_prompt(broken_json: str) -> str:
    from .chatlog_condenser_prompt import build_json_repair_user_message

    return (
        "System:\n"
        "You repair broken character profile JSON. Output ONLY one valid JSON object.\n"
        "Preserve all complete fields from the broken input; fill only what is missing.\n\n"
        + build_json_repair_user_message(
            broken_json=broken_json,
            schema_hint=CHARACTER_SCHEMA_HINT,
        )
    )


def save_character_generation_backup(
    raw_response: str,
    *,
    attempt: int = 0,
    conversation_id: str = "",
) -> str:
    """Persist raw model output for recovery."""
    base = Path.home() / ".LiangLocal" / "character_generation_backups"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{conversation_id}" if conversation_id else ""
    path = base / f"character_gen_{stamp}_a{attempt}{suffix}.txt"
    path.write_text(raw_response or "", encoding="utf-8")
    logger.info("Saved character generation backup: %s", path)
    return str(path)
