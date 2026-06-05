"""
In-character memory curator: build prompts so a chosen persona reviews profile or
agentic memories and returns structured JSON for deduplication / merge / reorder.

Apply endpoints validate and persist; users typically copy prompt → LLM → paste JSON.
"""

from __future__ import annotations

import datetime
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

PROFILE_JSON_SPEC = """
Return ONE JSON object only (no markdown fences, no commentary):

{
  "curator_voice_note": "Optional short line staying in voice — how you approached the cleanup.",
  "summary": "Plain-language summary of what you merged, dropped, or reordered.",
  "memories": [
    {
      "content": "Clear, non-redundant fact or preference; use {{user}} when referring to the human.",
      "category": "personal_info|preferences|interests|facts|skills|opinions|experiences|other",
      "importance": 0.75,
      "merged_from_indices": [0, 4]
    }
  ],
  "dropped_indices": [2, 5],
  "self_check": {
    "invented_facts": false,
    "comment": "If invented_facts is true, explain; otherwise empty string."
  }
}

Rules:
- Ground every retained memory in merged_from_indices (merge overlaps into one row).
- dropped_indices lists source indices you intentionally removed as redundant or harmful.
- Do not invent biographical facts not supported by MEMORIES_INDEXED.
- Prefer fewer high-quality rows over many repeats.
"""

AGENTIC_JSON_SPEC = """
Return ONE JSON object only (no markdown fences, no commentary):

{
  "curator_voice_note": "Optional — in character, one sentence.",
  "summary": "What you merged/dropped.",
  "insights": [
    {
      "content": "Durable note about {{user}} from this character's perspective.",
      "category": "preference|insight|behavior|habit|identity|background|plan|other",
      "importance": 0.8,
      "merged_from_ids": ["abc123", "def456"]
    }
  ],
  "dropped_ids": ["ghi789"],
  "self_check": {
    "invented_facts": false,
    "comment": ""
  }
}

Rules:
- Ground content in merged_from_ids; merge duplicates.
- dropped_ids lists insight ids removed as redundant.
- Use {{user}} in content where appropriate.
"""


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    # Strip markdown fence if present
    fence = re.match(r"^```(?:json)?\s*\n?", s)
    if fence:
        s = s[fence.end() :]
        if s.endswith("```"):
            s = s[:-3].strip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _character_persona_block(card: Optional[Dict[str, Any]], fallback_name: str) -> str:
    if not card or not isinstance(card, dict):
        return f"You are {fallback_name}. Answer as this persona would — voice, values, and tone."
    name = (card.get("name") or "").strip() or fallback_name
    parts = [f"You are {name}."]
    for label, key in (
        ("Persona / description", "description"),
        ("Personality", "personality"),
        ("Scenario", "scenario"),
        ("Speech style", "speech_style"),
        ("Background", "background"),
        ("Out-of-character model instructions", "model_instructions"),
    ):
        val = (card.get(key) or "").strip()
        if val:
            parts.append(f"{label}:\n{val}")
    return "\n\n".join(parts)


def build_profile_curator_prompt(
    *,
    user_id: str,
    user_display_name: Optional[str],
    user_profile_summary: Optional[str],
    memories: List[Dict[str, Any]],
    curator_character_card: Optional[Dict[str, Any]],
    curator_character_name: str,
    extra_notes: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    indexed = []
    for i, m in enumerate(memories or []):
        if not isinstance(m, dict):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        indexed.append(
            {
                "index": i,
                "content": content,
                "category": m.get("category") or "other",
                "importance": m.get("importance"),
                "created": m.get("created"),
                "type": m.get("type"),
            }
        )

    payload = json.dumps(indexed, ensure_ascii=False, indent=2)
    persona = _character_persona_block(curator_character_card, curator_character_name)
    notes = (extra_notes or "").strip() or "(none)"
    profile_snip = (user_profile_summary or "").strip() or "(no extra profile summary provided)"

    preamble = f"""{persona}

You are performing a structured housekeeping task for your human collaborator's backend USER MEMORY PROFILE (not roleplay fiction). Stay in character in curator_voice_note only; the JSON payload must be factual and grounded.

Target user id: {user_id}
They may be referred to as: {user_display_name or '{{user}}'} in memory text."""

    task = f"""TASK — CURATE USER PROFILE MEMORIES

USER PROFILE SUMMARY (from client; may be incomplete):
{profile_snip}

MEMORIES_INDEXED (each row is one stored memory; index matches merged_from_indices):
{payload}

OPERATOR NOTES:
{notes}

Your job: deduplicate, merge overlaps, drop noise, fix vague lines, reorder implicit priority by importance. Reduce count without losing distinct facts.

OUTPUT FORMAT:
{PROFILE_JSON_SPEC}

Produce valid JSON only."""

    combined = preamble.strip() + "\n\n" + task.strip()
    stats = {
        "mode": "profile",
        "indexed_rows": len(indexed),
        "bundle_chars": len(combined),
    }
    return combined, stats


def build_agentic_curator_prompt(
    *,
    user_id: str,
    target_character_id: str,
    target_character_name: str,
    user_display_name: Optional[str],
    insights: List[Dict[str, Any]],
    curator_character_card: Optional[Dict[str, Any]],
    curator_character_name: str,
    extra_notes: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    indexed = []
    for ins in insights or []:
        if not isinstance(ins, dict):
            continue
        cid = (ins.get("id") or "").strip()
        content = (ins.get("content") or "").strip()
        if not cid or not content:
            continue
        indexed.append(
            {
                "id": cid,
                "content": content,
                "category": ins.get("category") or "insight",
                "importance": ins.get("importance"),
                "created_at": ins.get("created_at"),
            }
        )

    payload = json.dumps(indexed, ensure_ascii=False, indent=2)
    persona = _character_persona_block(curator_character_card, curator_character_name)
    notes = (extra_notes or "").strip() or "(none)"

    preamble = f"""{persona}

You are performing structured housekeeping on CHARACTER AGENTIC MEMORY (facts this character remembers about the human). Stay in character lightly in curator_voice_note; JSON must be factual and grounded.

User id: {user_id}
Character whose memory file we are editing: {target_character_name} (id {target_character_id})
Human may appear as {{user}} in memory lines."""

    task = f"""TASK — CURATE AGENTIC INSIGHTS FOR THIS CHARACTER

INSIGHTS_INDEXED (id field must appear in merged_from_ids or dropped_ids):
{payload}

OPERATOR NOTES:
{notes}

Merge duplicates, remove repetition, tighten wording, adjust importance when warranted.

OUTPUT FORMAT:
{AGENTIC_JSON_SPEC}

Produce valid JSON only."""

    combined = preamble.strip() + "\n\n" + task.strip()
    stats = {
        "mode": "agentic",
        "indexed_rows": len(indexed),
        "bundle_chars": len(combined),
        "target_character_id": target_character_id,
    }
    return combined, stats


def normalize_profile_memories_for_save(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        content = (row.get("content") or "").strip()
        if len(content) < 3:
            continue
        cat = (row.get("category") or "other").strip() or "other"
        imp = row.get("importance")
        try:
            imp_f = float(imp) if imp is not None else 0.7
        except (TypeError, ValueError):
            imp_f = 0.7
        imp_f = max(0.0, min(1.0, imp_f))
        out.append(
            {
                "content": content,
                "category": cat,
                "importance": imp_f,
                "type": "curated",
                "created": datetime.datetime.now().isoformat(),
                "accessed": 0,
            }
        )
    return out


def normalize_agentic_insights_for_save(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        content = (row.get("content") or "").strip()
        if len(content) < 3:
            continue
        cat = (row.get("category") or "insight").strip() or "insight"
        imp = row.get("importance")
        try:
            imp_f = float(imp) if imp is not None else 0.7
        except (TypeError, ValueError):
            imp_f = 0.7
        imp_f = max(0.0, min(1.0, imp_f))
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append(
            {
                "id": f"ins_{uuid.uuid4().hex[:12]}",
                "content": content,
                "category": cat,
                "importance": imp_f,
                "created_at": ts,
            }
        )
    return out
