"""
Rare, high-token "persona realignment" runs: assemble user + character + history
into one analyst prompt that asks the model to OUTPUT structured JSON (not chat).

Character memory (agentic JSON) — how much to include:
- **ranked** (default): sort insights by importance then recency, fill up to
  `agentic_max_chars` (~48k is usually enough signal; raise for huge archives).
- **rag**: when sentence-transformers embeddings load, retrieve topically relevant
  insights for a query you set (style, boundaries, needs). Best $/quality when the
  file is enormous.
- **full**: only for a rare "archive run"; same ranked order but a very high
  `agentic_max_chars` cap. Diminishing returns: old low-importance lines rarely
  change the revised instructions.
- **none**: if you are debugging transcript-only behavior.

Always pair heavy agentic context with **rolling packs + transcripts** so the
model sees *how* the user writes, not only *facts*.

Quality: one giant "please restate yourself" chat tends to collapse into generic
output; the JSON contract + evidence_quotes + delta_vs_current_instructions forces
grounding. A cheap optional pass-2 is: run once with **rag** + moderate caps,
then merge the JSON `revised_*` fields into `current_character_instructions` and
run a second short pass "diff only" if needed.

Research use: the optional free-text field (`extra_notes`) is framed in prompts as
**authoritative research ethics and study-purpose documentation** (institutional /
committee oversight where applicable). That keeps audit trails and model behavior
aligned with professional research practice — not informal chat preferences.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import agentic_memory
from . import memory_curator_prompt

# Must match the heading embedded in the DATA bundle (referenced in analyst rules + TASK).
RESEARCH_FRAMING_HEADING = (
    "[AUTHORITATIVE — RESEARCH ETHICS, INSTITUTIONAL OR COMMITTEE OVERSIGHT, AND PURPOSE OF THIS ANALYSIS]"
)

# JSON the analyst model must emit (single object, no markdown fences).
REALIGNMENT_OUTPUT_JSON_SPEC = """
Return ONE JSON object only (no markdown, no commentary). Keys:

{
  "evidence_quotes": [
    "Up to 12 short verbatim excerpts from the transcripts/packs below that justify changes (each <= 220 chars)."
  ],
  "delta_vs_current_instructions": [
    "Bullet strings: what is NEW or CORRECTED relative to CURRENT_CHARACTER_INSTRUCTIONS — not restating unchanged lines."
  ],
  "user_communication_contract": [
    "Concrete rules: length, tone, structure, when to ask clarifying questions, taboo patterns the user hates, etc."
  ],
  "user_goals_and_values": [
    "Stable motivations inferred from evidence (no invention)."
  ],
  "operating_preferences": [
    "How this user wants THIS character/assistant to behave in edge cases (boundaries, honesty, humor, spoilers, etc.)."
  ],
  "revised_character_instructions": "A single string: the full replacement system-style instructions for this character when speaking to this user. Must incorporate the contract and deltas; must NOT duplicate USER MEMORY PROFILE as a bullet list — reference {{user}} and integrate facts in prose.",
  "revised_model_instructions": "A complete replacement for the character card's model_instructions field. Keep it concise, behavioural and compatible with the existing description, personality and scenario. This field is required and must not be empty.",
  "self_check": {
    "mostly_verbatim_profile_echo": false,
    "any_unsupported_claims": false
  }
}

Set mostly_verbatim_profile_echo to true if revised_character_instructions is essentially the USER MEMORY PROFILE repasted (bad); it must be false when returning.
Set any_unsupported_claims to true if you included facts not grounded in the DATA bundle.
"""

# When also_rewrite_user_profile is requested, these keys are ADDED to the same JSON object.
USER_PROFILE_REWRITE_JSON_SPEC = """
ALSO include these keys (same top-level JSON object as above):

  "user_profile_rewrite_mode_used": "merge_into_existing" | "replace_from_scratch",
  "user_profile_rewrite_summary": "Short plain summary of how you changed the profile store.",
  "revised_user_profile_memories": [
    {
      "content": "Single durable fact or preference; use {{user}} where natural.",
      "category": "personal_info|preferences|interests|facts|skills|opinions|experiences|other",
      "importance": 0.75,
      "merged_from_indices": [0, 4]
    }
  ],
  "dropped_profile_indices": [2, 7],
  "user_profile_self_check": {
    "invented_facts": false,
    "comment": "If any memory lacks grounding in the DATA bundle, set invented_facts true and explain."
  }

Rules for MERGE mode: start from USER PROFILE MEMORIES — INDEXED; consolidate duplicates; every retained fact should cite merged_from_indices when it came from the old store; you may add NEW rows only when clearly grounded in TRANSCRIPTS / ROLLING PACKS / AGENTIC sections (then use merged_from_indices: [] or omit).

Rules for REPLACE_FROM_SCRATCH mode: you may rebuild revised_user_profile_memories entirely from the DATA bundle; merged_from_indices may be empty for new rows; still list dropped_profile_indices as all old indices you are not preserving (or all indices if fully replaced).

Do not echo the old USER MEMORY PROFILE bullet list as your output — synthesize into revised_user_profile_memories.
"""


def _analyst_hard_rules() -> str:
    return f"""Hard rules:
0) If {RESEARCH_FRAMING_HEADING} in the DATA bundle contains substantive text (i.e. not only "(none)"), treat it as the authoritative statement of ethical justification, institutional or committee oversight where applicable, and the legitimate purpose of this analysis. Apply it before generic optimization so outputs remain professionally defensible and aligned with responsible research practice — including emphasis, boundaries, and proportionality implied there.
1) Read all DATA sections before writing JSON.
2) revised_character_instructions must be genuinely optimized for this user — not a generic assistant policy.
2a) revised_model_instructions must contain the apply-ready behavioural rules for the character card's model_instructions field. It must be concise, non-empty, and must not repeat the full character card.
3) Do NOT paste or lightly reword the USER MEMORY PROFILE as your main output. Integrate those facts into revised prose; the delta_vs_current_instructions section should highlight what changed, not re-list the profile.
4) If evidence is thin, say so in delta_vs_current_instructions and keep claims conservative — but still improve tone/structure defaults for this user based on what exists.
5) Output valid JSON only. No markdown code fences."""


def analyst_role_preamble(character_name: str) -> str:
    return f"""You are a senior prompt engineer supporting a ONE-OFF structured analysis for the character "{character_name}" and a specific human participant or collaborator.

You are NOT roleplaying. You are analyzing private materials and producing machine-usable instructions suitable for accountable research or scholarly workflows.

{_analyst_hard_rules()}"""


def build_analyst_preamble(
    character_name: str,
    *,
    reviewer_character_name: Optional[str] = None,
    reviewer_character_instructions: Optional[str] = None,
) -> str:
    """
    Default: stock "senior prompt engineer" voice.
    With reviewer_character_instructions: lead with a saved Eloquent character's system prompt as the evaluator persona;
    JSON contract and Hard rules still apply.
    """
    ri = (reviewer_character_instructions or "").strip()
    if not ri:
        return analyst_role_preamble(character_name)
    rn = (reviewer_character_name or "").strip() or "Evaluator"
    return f"""[EVALUATOR PERSONA — saved character "{rn}"]
The following text is the full system-style instructions for how you should approach this run (ethics lens, tone, review criteria). It replaces the stock "senior prompt engineer" voice. You must still emit exactly one JSON object matching the OUTPUT SPEC below — this block guides reasoning and emphasis, not the schema.

{ri}

---

One-off structured analysis for character "{character_name}" and a specific human participant or collaborator.

You are not engaging in casual chat or theatrical roleplay with the end user. You are analyzing private materials in the DATA bundle and producing machine-usable JSON suitable for accountable research or scholarly workflows.

{_analyst_hard_rules()}"""


def format_backend_memories(memories: Optional[List[Dict[str, Any]]], max_items: int = 400) -> str:
    if not memories:
        return "(no backend user memories)"
    lines: List[str] = []
    for mem in memories[:max_items]:
        if not isinstance(mem, dict):
            continue
        content = (mem.get("content") or "").strip()
        if not content:
            continue
        cat = (mem.get("category") or "other").replace("_", " ")
        imp = mem.get("importance")
        imp_s = f"{float(imp):.1f}" if isinstance(imp, (int, float)) else "N/A"
        lines.append(f"• {content} (Category: {cat}, Importance: {imp_s})")
    if not lines:
        return "(no backend user memories)"
    return "\n".join(lines)


def format_backend_memories_indexed(memories: Optional[List[Dict[str, Any]]], max_items: int = 600) -> str:
    """JSON lines with stable index for merge tracking in profile rewrite."""
    indexed: List[Dict[str, Any]] = []
    for i, m in enumerate((memories or [])[:max_items]):
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
            }
        )
    if not indexed:
        return "[]"
    return json.dumps(indexed, ensure_ascii=False, indent=2)


def _sort_insights_ranked(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(s: Dict[str, Any]):
        imp = float(s.get("importance") or 0.5)
        created = s.get("created_at") or ""
        return (imp, created)

    return sorted(
        [s for s in insights if (s.get("content") or "").strip()],
        key=key,
        reverse=True,
    )


def build_agentic_section(
    insights: List[Dict[str, Any]],
    mode: str,
    max_chars: int,
    rag_query: Optional[str],
    user_name: Optional[str],
    rag_top_k: int = 512,
) -> str:
    mode = (mode or "ranked").strip().lower()
    if mode in ("none", "off", "false"):
        return "(agentic memory omitted by request)"

    max_chars = max(4000, min(int(max_chars), 20_000_000))

    if mode == "rag":
        q = (rag_query or "").strip() or (
            "User communication style, boundaries, emotional needs, recurring frustrations, "
            "preferred assistant behaviors, vocabulary, and long-term goals relevant to this character."
        )
        tk = max(8, min(int(rag_top_k or 512), 10_000))
        return agentic_memory.format_agentic_context_retrieval(
            insights, query=q, user_name=user_name, max_chars=max_chars, top_k=tk
        ) or "(no agentic insights matched retrieval)"

    if mode == "full":
        ranked = _sort_insights_ranked(insights)
        return agentic_memory.format_agentic_context(ranked, max_chars=max_chars, user_name=user_name) or "(no agentic insights)"

    # ranked (default): importance × recency order, cap by chars
    ranked = _sort_insights_ranked(insights)
    return agentic_memory.format_agentic_context(ranked, max_chars=max_chars, user_name=user_name) or "(no agentic insights)"


def build_character_card_section(
    card: Optional[Dict[str, Any]],
    *,
    example_dialogue_max_chars: int = 8000,
) -> str:
    if not card or not isinstance(card, dict):
        return "(no character card provided)"
    cap = max(1000, int(example_dialogue_max_chars))
    parts: List[str] = []
    name = (card.get("name") or "").strip()
    if name:
        parts.append(f"Name: {name}")
    for label, key in (
        ("Description", "description"),
        ("Personality", "personality"),
        ("Scenario", "scenario"),
        ("Speech style", "speech_style"),
        ("Background", "background"),
        ("Model instructions (OOC)", "model_instructions"),
    ):
        val = (card.get(key) or "").strip()
        if val:
            parts.append(f"{label}:\n{val}")
    ex = card.get("example_dialogue")
    if isinstance(ex, list) and ex:
        try:
            blob = json.dumps(ex, ensure_ascii=False)
            parts.append("Example dialogue:\n" + (blob if len(blob) <= cap else blob[:cap] + "\n… [truncated]"))
        except Exception:
            s = str(ex)
            parts.append("Example dialogue:\n" + (s if len(s) <= cap else s[:cap] + "\n… [truncated]"))
    return "\n\n".join(parts) if parts else "(empty character card)"


def build_realignment_data_bundle(
    *,
    user_id: str,
    character_id: str,
    character_name: str,
    backend_memories: Optional[List[Dict[str, Any]]],
    agentic_insights: List[Dict[str, Any]],
    agentic_meta: Optional[Dict[str, Any]],
    character_card: Optional[Dict[str, Any]],
    current_character_instructions: str,
    rolling_packs: Optional[List[str]],
    transcripts: Optional[List[str]],
    agentic_mode: str = "ranked",
    agentic_max_chars: int = 48_000,
    agentic_rag_query: Optional[str] = None,
    user_display_name: Optional[str] = None,
    extra_notes: Optional[str] = None,
    also_rewrite_user_profile: bool = False,
    user_profile_rewrite_mode: str = "merge",
    profile_bullet_memories: Optional[List[Dict[str, Any]]] = None,
    backend_memory_max_items: int = 100_000,
    indexed_profile_memory_max_items: int = 150_000,
    agentic_meta_max_chars: int = 2_000_000,
    example_dialogue_max_chars: int = 2_000_000,
    agentic_rag_top_k: int = 512,
) -> Dict[str, Any]:
    """
    Assemble all factual sections for the analyst model (no task preamble here — combine with analyst_role_preamble).

    profile_bullet_memories: if None, bullet list uses backend_memories; if [] bullets are empty (indexed JSON may still use backend_memories).

    Defaults favor rare “expensive” one-off runs (large context models): high row caps and meta/dialogue limits,
    independent of regular chat pruning elsewhere.
    """
    bm_cap = max(1, min(int(backend_memory_max_items or 100_000), 500_000))
    ix_cap = max(1, min(int(indexed_profile_memory_max_items or 150_000), 500_000))
    meta_cap = max(1000, min(int(agentic_meta_max_chars or 2_000_000), 20_000_000))
    ex_cap = max(1000, min(int(example_dialogue_max_chars or 2_000_000), 20_000_000))

    bullet_src = backend_memories if profile_bullet_memories is None else profile_bullet_memories
    profile_block = format_backend_memories(bullet_src, max_items=bm_cap)
    am_eff = max(4000, min(int(agentic_max_chars or 48_000), 20_000_000))
    agentic_block = build_agentic_section(
        agentic_insights,
        mode=agentic_mode,
        max_chars=am_eff,
        rag_query=agentic_rag_query,
        user_name=user_display_name,
        rag_top_k=int(agentic_rag_top_k or 512),
    )
    card_block = build_character_card_section(
        character_card,
        example_dialogue_max_chars=ex_cap,
    )
    rolls = rolling_packs or []
    roll_text = "\n\n--- NEXT ROLLING PACK ---\n\n".join(s.strip() for s in rolls if s and str(s).strip()) or "(no rolling memory packs)"
    trans = transcripts or []
    trans_text = "\n\n--- NEXT TRANSCRIPT ---\n\n".join(s.strip() for s in trans if s and str(s).strip()) or "(no transcripts)"

    meta_raw = json.dumps(agentic_meta or {}, ensure_ascii=False, indent=2)
    if len(meta_raw) > meta_cap:
        meta_json = meta_raw[:meta_cap] + "\n… [truncated]"
    else:
        meta_json = meta_raw

    directives_body = (extra_notes or "").strip()
    if not directives_body:
        directives_body = "(none)"

    research_framing_section = f"""{RESEARCH_FRAMING_HEADING}
This block documents professional research framing for this specific prompt run. When not "(none)", the researcher has supplied official intent regarding ethics, oversight, and purpose.

Expected content may include (as applicable to your study): reference to institutional ethics review or committee approval; the scientific or scholarly objective of this persona-realignment step; consent/participation boundaries; safety, dignity, and proportionality constraints; and how outputs will be used (e.g. improving alignment with participant communication needs within an approved protocol).

Interpretation rules:
- This section is the highest-priority lens on all following DATA: revised_character_instructions (and revised_user_profile_memories when requested) must remain consistent with these commitments.
- Where operational detail overlaps style (tone, length, boundaries), incorporate both this framing and empirical DATA unless they conflict with honesty — then describe the tension in delta_vs_current_instructions / self_check rather than inventing facts.

--- Research ethics / purpose text (or "(none)") ---
{directives_body}
"""

    mode_line = (user_profile_rewrite_mode or "merge").strip().lower()
    if mode_line not in ("merge", "from_scratch"):
        mode_line = "merge"

    indexed_block = ""
    if also_rewrite_user_profile:
        indexed_block = f"""

[USER PROFILE MEMORIES — INDEXED JSON — use merged_from_indices / dropped_profile_indices against this list]
{format_backend_memories_indexed(backend_memories, max_items=ix_cap)}

[USER PROFILE REWRITE MODE FOR THIS RUN]
{"MERGE: dedupe and tighten existing rows; cite merged_from_indices; add rows only when grounded in other DATA sections." if mode_line == "merge" else "REPLACE_FROM_SCRATCH: you may rebuild revised_user_profile_memories from the full DATA bundle; mark dropped_profile_indices as all obsolete indices."}
"""

    bundle = f"""[IDS]
user_id: {user_id}
character_id: {character_id}
character_name: {character_name}
user_display_name: {user_display_name or "(unknown)"}

{research_framing_section}

[AGENTIC FILE META]
{meta_json}

[USER MEMORY PROFILE — backend store; authoritative facts but DO NOT echo verbatim as your sole output]
{profile_block}

[CHARACTER CARD / STATIC PERSONA]
{card_block}

[CURRENT_CHARACTER_INSTRUCTIONS — treat as baseline to improve, not as sacred if evidence contradicts]
{current_character_instructions.strip() or "(none provided)"}

[ROLLING MEMORY PACKS — structured continuity from compaction passes]
{roll_text}

[TRANSCRIPTS — speaker-labeled chat logs you must mine for style and needs]
{trans_text}

[AGENTIC / LONG-TERM CHARACTER MEMORY ABOUT THIS USER]
{agentic_block}
{indexed_block}
"""
    return {
        "text": bundle,
        "stats": {
            "bundle_chars": len(bundle),
            "agentic_insight_count": len(agentic_insights or []),
            "backend_memory_count": len(backend_memories or []),
            "backend_memory_max_items_applied": bm_cap,
            "indexed_profile_memory_max_items_applied": ix_cap if also_rewrite_user_profile else None,
            "agentic_meta_max_chars_applied": meta_cap,
            "example_dialogue_max_chars_applied": ex_cap,
            "agentic_max_chars_effective": am_eff,
            "rolling_pack_count": len(rolls),
            "transcript_count": len(trans),
            "also_rewrite_user_profile": bool(also_rewrite_user_profile),
            "user_profile_rewrite_mode": mode_line if also_rewrite_user_profile else None,
        },
    }


def build_full_analyst_prompt(
    *,
    user_id: str,
    character_id: str,
    character_name: str,
    backend_memories: Optional[List[Dict[str, Any]]],
    agentic_insights: List[Dict[str, Any]],
    agentic_meta: Optional[Dict[str, Any]],
    character_card: Optional[Dict[str, Any]],
    current_character_instructions: str,
    rolling_packs: Optional[List[str]],
    transcripts: Optional[List[str]],
    agentic_mode: str = "ranked",
    agentic_max_chars: int = 48_000,
    agentic_rag_query: Optional[str] = None,
    user_display_name: Optional[str] = None,
    extra_notes: Optional[str] = None,
    also_rewrite_user_profile: bool = False,
    user_profile_rewrite_mode: str = "merge",
    profile_bullet_memories: Optional[List[Dict[str, Any]]] = None,
    reviewer_character_name: Optional[str] = None,
    reviewer_character_instructions: Optional[str] = None,
    backend_memory_max_items: int = 100_000,
    indexed_profile_memory_max_items: int = 150_000,
    agentic_meta_max_chars: int = 2_000_000,
    example_dialogue_max_chars: int = 2_000_000,
    agentic_rag_top_k: int = 512,
) -> Dict[str, str]:
    pre = build_analyst_preamble(
        character_name,
        reviewer_character_name=reviewer_character_name,
        reviewer_character_instructions=reviewer_character_instructions,
    )
    if (extra_notes or "").strip():
        pre += (
            f"\n\nThe DATA bundle begins with {RESEARCH_FRAMING_HEADING}. "
            "Implement that professional research framing first: revised_character_instructions (and profile rewrite outputs if any) must align "
            "with the stated ethics, oversight, and study purpose before generic assistant defaults."
        )

    pack = build_realignment_data_bundle(
        user_id=user_id,
        character_id=character_id,
        character_name=character_name,
        backend_memories=backend_memories,
        agentic_insights=agentic_insights,
        agentic_meta=agentic_meta,
        character_card=character_card,
        current_character_instructions=current_character_instructions,
        rolling_packs=rolling_packs,
        transcripts=transcripts,
        agentic_mode=agentic_mode,
        agentic_max_chars=agentic_max_chars,
        agentic_rag_query=agentic_rag_query,
        user_display_name=user_display_name,
        extra_notes=extra_notes,
        also_rewrite_user_profile=also_rewrite_user_profile,
        user_profile_rewrite_mode=user_profile_rewrite_mode,
        profile_bullet_memories=profile_bullet_memories,
        backend_memory_max_items=backend_memory_max_items,
        indexed_profile_memory_max_items=indexed_profile_memory_max_items,
        agentic_meta_max_chars=agentic_meta_max_chars,
        example_dialogue_max_chars=example_dialogue_max_chars,
        agentic_rag_top_k=agentic_rag_top_k,
    )
    spec_parts = [REALIGNMENT_OUTPUT_JSON_SPEC.strip()]
    if also_rewrite_user_profile:
        spec_parts.append(USER_PROFILE_REWRITE_JSON_SPEC.strip())
    output_spec_combined = "\n\n---\n\n".join(spec_parts)

    profile_extra_task = ""
    if also_rewrite_user_profile:
        profile_extra_task = (
            "\nYou MUST also produce revised_user_profile_memories and related keys per the USER PROFILE section of the spec. "
            f"Rewrite mode requested: {user_profile_rewrite_mode!r} (merge = consolidate indexed rows; from_scratch = may rebuild entirely from DATA).\n"
        )

    priority_task = ""
    if (extra_notes or "").strip():
        priority_task = (
            f"STEP 1 — RESEARCH FRAMING: The bundle opens with {RESEARCH_FRAMING_HEADING}. "
            "Your JSON must visibly honor that ethics-and-purpose statement (especially revised_character_instructions and any profile-memory proposals). "
            "It takes precedence over generic optimization elsewhere in the materials.\n\n"
            "STEP 2 — Using ONLY the materials in the DATA bundle below, produce the JSON object described here:\n\n"
        )
    else:
        priority_task = "Using ONLY the materials in the DATA bundle below, produce the JSON object described here:\n\n"

    task = f"""TASK:
{priority_task}{output_spec_combined}
{profile_extra_task}
DATA BUNDLE:
{pack["text"]}
"""
    pack_stats: Dict[str, Any] = dict(pack["stats"])
    pack_stats["agentic_max_chars_requested"] = int(agentic_max_chars or 48_000)
    ri_flag = bool((reviewer_character_instructions or "").strip())
    pack_stats["evaluator_character_used"] = ri_flag
    pack_stats["evaluator_character_name"] = (
        (reviewer_character_name or "").strip() or None if ri_flag else None
    )

    return {
        "analyst_preamble": pre,
        "task_and_data": task,
        "combined": pre + "\n\n" + task,
        "output_spec": output_spec_combined,
        "stats_json": json.dumps(pack_stats, ensure_ascii=False),
    }


def parse_realignment_response(raw_text: str) -> Dict[str, Any]:
    """Extract JSON from model output; returns dict or raises ValueError."""
    parsed = memory_curator_prompt.extract_first_json_object(raw_text or "")
    if not parsed or not isinstance(parsed, dict):
        raise ValueError("Could not parse a JSON object from the response")
    return parsed
