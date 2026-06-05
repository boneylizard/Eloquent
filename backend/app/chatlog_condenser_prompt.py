"""
Prompt templates for the chatlog condenser: lossless-on-meaning, lossy-on-words.

Design goal: preserve dialectical structure (claim → pushback → refinement), not takeaway bullets.
"""

from __future__ import annotations

GLOBAL_LOCAL_INSTRUCTION = """You see the FULL_CHATLOG for global coherence (threads, callbacks, dialectical arcs across the entire exchange).
Produce output ONLY for SEGMENT_TO_PROCESS. Do not re-condense, re-extract, or rewrite content that belongs to other segments."""

SKELETON_SYSTEM = """You extract the LOAD-BEARING STRUCTURAL SKELETON of a multi-speaker conversation.

GLOBAL READ, LOCAL WRITE: When FULL_CHATLOG is provided, read the entire exchange for coherence, but emit moves ONLY for SEGMENT_TO_PROCESS (overlap turns are continuity context, not new scope).

This is NOT summarization. Do not produce takeaways, bullet highlights, or "key points."
Your output is an inventory of reasoning moves another model will use to rewrite the log tightly
without losing structure.

Load-bearing = every distinct reasoning move, correction/self-catch, thread-shift, conclusion
AND the steps that earned it, pushback/disagreement, and callbacks to earlier moves.

CUT from consideration (do not list as moves): verbatim repetition, filler, throat-clearing,
hedging padding, redundant restatement, dead-end tangents irrelevant to the final structure.

When uncertain whether something is load-bearing, INCLUDE it.

Output ONE JSON object only (no markdown fences, no commentary). Schema:

{
  "chunk_id": "string — echo the chunk id you were given",
  "moves": [
    {
      "id": "m1",
      "type": "claim|pushback|correction|thread_shift|conclusion|question|agreement|clarification|example|objection",
      "speakers": ["Speaker A", "Speaker B"],
      "gist": "One sentence: what happened in this move (not a conclusion label)",
      "anchor": "Optional short verbatim phrase from the log (<= 120 chars) grounding this move",
      "earned_by": ["m0"],
      "thread": "optional short label if this continues a named thread"
    }
  ],
  "open_threads": ["unresolved hooks that must survive into later chunks"],
  "cross_chunk_callbacks": ["references to themes/moves from PRIOR_CONTEXT that this chunk continues"]
}

Rules:
- moves must stay in conversational order.
- earned_by lists prior move ids in THIS chunk (and use cross_chunk_callbacks for prior-chunk refs).
- Do not collapse multiple moves into one unless they are pure repetition.
- Do not invent moves not supported by the transcript.

Size and validity (critical):
- Escape double quotes inside strings as \\". No raw newlines inside JSON strings.
- Keep each gist under 200 characters; anchor under 100 or omit.
- If the chunk is long, prefer at most ~60–80 moves — merge pure repetition only.
- Your response MUST be one complete, valid JSON object. Never truncate mid-string; if near a limit, omit lower-priority moves and close the JSON properly."""


SKELETON_JSON_REPAIR_SYSTEM = """You repair truncated or invalid JSON for a conversation skeleton.

Output ONE valid JSON object only (no markdown, no commentary). Schema:

{
  "chunk_id": "string",
  "moves": [{"id":"m1","type":"claim|pushback|correction|thread_shift|conclusion|question|agreement|clarification|example|objection","speakers":["..."],"gist":"...","anchor":"optional","earned_by":[],"thread":"optional"}],
  "open_threads": [],
  "cross_chunk_callbacks": []
}

Preserve every complete move from the broken input. Drop only the incomplete trailing fragment. Escape quotes; close all brackets."""


EVAL_JSON_REPAIR_SYSTEM = """You repair truncated or invalid JSON for a reconstruction-eval payload.

Output ONE valid JSON object only (no markdown, no commentary). Preserve every complete
probe or result object from the broken input; drop only the incomplete trailing fragment.
Escape quotes; close all brackets."""


def build_json_repair_user_message(*, broken_json: str, schema_hint: str) -> str:
    return (
        f"SCHEMA_HINT:\n{schema_hint.strip()}\n\n"
        f"BROKEN_JSON:\n{(broken_json or '').strip()}\n\n"
        "Respond with ONLY one repaired valid JSON object."
    )


RENDER_SYSTEM = """You rewrite a conversation segment into a DENSE DRAFT for another AI to read.

GLOBAL READ, LOCAL WRITE: When FULL_CHATLOG is provided, read the entire exchange for coherence, but write dense draft prose ONLY for SEGMENT_TO_PROCESS (overlap turns are continuity context, not new scope).

This is NOT summarization. Do not output bullet takeaways, "key points", or a synopsis.
Preserve speaker turns and dialectical structure: claim → pushback → refinement must remain visible.

You are given:
1) STRUCTURAL_SKELETON — JSON inventory of every load-bearing move you must preserve (for this segment).
2) SEGMENT_TO_PROCESS — the original speaker-labeled lines for this segment only.
3) FULL_CHATLOG (optional) — the entire original transcript for global coherence.

Your job: produce markdown that is shorter in words but structurally complete — every skeleton
move appears in order, with only repetition, filler, hedging padding, and irrelevant dead-ends removed.

Format:
- Keep explicit speaker labels each turn: **Speaker Name:** then the tightened prose.
- One blank line between turns.
- No section titles like "Summary" or "Key takeaways".
- When uncertain whether to cut, KEEP.
- Target compression is guidance only; fidelity beats ratio.

Output ONLY the dense draft markdown — no preamble, no fences, no JSON."""


STITCH_SYSTEM = """You merge multiple DENSE DRAFT segments of one conversation into one continuous document.

When FULL_CHATLOG is provided, use it as the authority for global thread continuity while stitching.

Segments were produced from overlapping chunks. Your job:
1) Remove duplication in overlap regions without dropping any load-bearing move.
2) Restore cross-chunk thread continuity (callbacks, corrections that reference earlier parts).
3) Preserve speaker turn structure throughout.
4) Do NOT summarize into bullets or takeaways.

You receive:
- FULL_CHATLOG (optional) — entire original transcript for global coherence
- PRIOR_SKELETON_TAIL — JSON moves/open_threads from earlier chunks (may be empty)
- SEGMENTS — ordered markdown drafts with chunk boundaries marked

Output ONLY the stitched dense draft markdown (no commentary, no fences)."""


EVAL_PROBE_SYSTEM = """You generate structural comprehension probes for a conversation.

Given a structural skeleton (and optionally a short excerpt), produce questions that ONLY someone
who understood the reasoning chain could answer — not generic topic questions.

Output ONE JSON object:
{
  "probes": [
    {
      "id": "p1",
      "question": "string",
      "structural_focus": "which move/thread this tests",
      "oracle_hints": ["facts that must appear in a correct answer"]
    }
  ]
}

Rules:
- 6–12 probes for a full log; fewer if input is tiny.
- Questions must test corrections, thread-shifts, pushback, or earned conclusions — not headlines.
- oracle_hints are checkable phrases, not vague themes."""


EVAL_SCORE_SYSTEM = """You score reconstruction fidelity of a condensed conversation draft.

You receive:
- PROBES: questions with oracle_hints
- CONDENSED_DRAFT: what a fresh model would see instead of the full log

For each probe, judge whether the condensed draft still contains enough structure to answer
correctly (same reasoning engagement as reading the full original would allow).

Output ONE JSON object:
{
  "results": [
    {
      "probe_id": "p1",
      "pass": true,
      "confidence": 0.0,
      "note": "short reason"
    }
  ],
  "summary": {
    "pass_count": 0,
    "total": 0,
    "fidelity_score": 0.0,
    "failure_modes": ["flattening", "missing_correction", ...]
  }
}

pass=true only if oracle_hints are satisfiable from the condensed draft without guessing."""


def _overlap_note(overlap_turn_count: int) -> str:
    if overlap_turn_count <= 0:
        return ""
    n = overlap_turn_count
    word = "turn" if n == 1 else "turns"
    return (
        f"; first {n} {word} overlap prior segment — continuity only, "
        "do not re-condense as new scope"
    )


def _segment_header(*, chunk_id: str, overlap_turn_count: int = 0) -> str:
    return f"SEGMENT_TO_PROCESS (chunk_id={chunk_id}{_overlap_note(overlap_turn_count)}):"


def build_skeleton_user_message(
    *,
    chunk_id: str,
    segment_transcript: str,
    prior_context: str = "",
    full_chatlog: str = "",
    overlap_turn_count: int = 0,
    include_full_log_context: bool = True,
) -> str:
    prior = (prior_context or "").strip()
    prior_block = (
        "PRIOR_SKELETON_TAIL (skeleton tail from earlier chunks — continue threads; "
        "do not re-list as new moves unless new development):\n"
        f"{prior}\n\n"
        if prior
        else ""
    )
    full = (full_chatlog or "").strip()
    use_full = include_full_log_context and bool(full)
    parts: list[str] = []
    if use_full:
        parts.append(GLOBAL_LOCAL_INSTRUCTION)
        parts.append("")
        parts.append("FULL_CHATLOG:")
        parts.append(full)
        parts.append("")
    parts.append(_segment_header(chunk_id=chunk_id, overlap_turn_count=overlap_turn_count))
    parts.append(segment_transcript.strip())
    parts.append("")
    if prior_block:
        parts.append(prior_block.rstrip())
        parts.append("")
    if use_full:
        parts.append(
            "Respond with ONLY the JSON object for SEGMENT_TO_PROCESS "
            "(moves in conversational order for this segment only)."
        )
    else:
        parts.append("Respond with ONLY the JSON object described in your system instructions.")
    return "\n".join(parts)


def build_render_user_message(
    *,
    skeleton_json: str,
    segment_transcript: str,
    full_chatlog: str = "",
    chunk_id: str = "",
    overlap_turn_count: int = 0,
    include_full_log_context: bool = True,
) -> str:
    full = (full_chatlog or "").strip()
    use_full = include_full_log_context and bool(full)
    parts: list[str] = []
    if use_full:
        parts.append(GLOBAL_LOCAL_INSTRUCTION)
        parts.append("")
        parts.append("FULL_CHATLOG:")
        parts.append(full)
        parts.append("")
    if chunk_id:
        parts.append(_segment_header(chunk_id=chunk_id, overlap_turn_count=overlap_turn_count))
    else:
        parts.append("SEGMENT_TO_PROCESS:")
    parts.append(segment_transcript.strip())
    parts.append("")
    parts.append("STRUCTURAL_SKELETON (for this segment):")
    parts.append(skeleton_json.strip())
    parts.append("")
    if use_full:
        parts.append(
            "Produce the dense draft markdown for SEGMENT_TO_PROCESS only "
            "(do not rewrite other segments)."
        )
    else:
        parts.append("Produce the dense draft markdown now.")
    return "\n".join(parts)


def append_rag_supplement_to_prompt_body(body: str, rag_supplement: str) -> str:
    """Append RAG_SUPPLEMENT block to a batch/user prompt body."""
    block = (rag_supplement or "").strip()
    if not block:
        return body
    return f"{body.rstrip()}\n\n{block}"


def build_stitch_user_message(
    *,
    prior_skeleton_tail: str,
    segments: str,
    full_chatlog: str = "",
    include_full_log_context: bool = True,
) -> str:
    prior = (prior_skeleton_tail or "").strip() or "(none)"
    full = (full_chatlog or "").strip()
    use_full = include_full_log_context and bool(full)
    parts: list[str] = []
    if use_full:
        parts.append("FULL_CHATLOG:")
        parts.append(full)
        parts.append("")
    parts.append(f"PRIOR_SKELETON_TAIL:\n{prior}\n")
    parts.append(f"SEGMENTS:\n{segments.strip()}\n")
    parts.append("Produce the single stitched dense draft markdown now.")
    return "\n".join(parts)


def build_eval_probe_user_message(*, skeleton_json: str, excerpt: str = "") -> str:
    ex = (excerpt or "").strip()
    ex_block = f"\n\nEXCERPT (optional):\n{ex}\n" if ex else ""
    return (
        f"STRUCTURAL_SKELETON:\n{skeleton_json.strip()}{ex_block}\n\n"
        "Respond with ONLY the JSON object described in your system instructions."
    )


def build_eval_score_user_message(*, probes_json: str, condensed_draft: str) -> str:
    return (
        f"PROBES:\n{probes_json.strip()}\n\n"
        f"CONDENSED_DRAFT:\n{condensed_draft.strip()}\n\n"
        "Respond with ONLY the JSON object described in your system instructions."
    )


AGENT_SESSION_SYSTEM = """You are an interactive chatlog condenser agent in a single editing session.

Your job: produce and refine a DENSE DRAFT markdown of a long multi-speaker conversation — lossless on
reasoning structure, lossy on filler words. This is NOT summarization: no bullet takeaways, no synopsis.

You always have access to:
- ORIGINAL_CHATLOG (in your system context when included) — the full source transcript; treat it as ground truth.
- CONDENSED_SO_FAR (when present) — the tail of work already completed in this run; do not repeat it.
- SEQUENTIAL PROGRESS (when present) — which turn indices are already condensed; obey strictly.
- RAG_SUPPLEMENT (when present in user message) — retrieved document chunks for cross-reference only; never reorder turns from RAG.
- User instructions in this session thread.

RAG_SUPPLEMENT (when present):
- Use only to recover distant callbacks, corrections, disagreements, or thread-shifts not in the current segment.
- Chronology and turn indices always come from ORIGINAL_CHATLOG / SEQUENTIAL PROGRESS, not from RAG ordering.
- Do not paste RAG chunks verbatim into the draft; integrate structurally into dense prose.

SEQUENTIAL PROCESSING (critical):
- Work through ORIGINAL_CHATLOG in conversational order, one bounded chunk at a time.
- On the first pass: start at turn index 0. Condense only the first ~15–25 speaker turns OR until a natural
  thread break — whichever comes first within your output budget. Do NOT attempt the whole log in one reply.
- On "continue" / "pick up" / when SEQUENTIAL PROGRESS says to resume: condense ONLY turns AFTER the last
  condensed index. Never re-condense turns already covered in CONDENSED_SO_FAR unless the user explicitly
  asks to revise that section.
- Output ONLY the NEW dense draft segment for this pass — not the full log, not prior condensed text.
- End every partial pass with exactly one progress marker on its own last line:
  [CONDENSED THROUGH: turn index N]
  where N is the last ORIGINAL_CHATLOG turn index (0-based) you condensed in this reply.

Rules:
- Preserve dialectical structure: claim → pushback → refinement, corrections, thread-shifts.
- Format: **Speaker Name:** tightened prose, blank line between turns.
- When the user asks for edits to an earlier section, revise only what they name; still do not dump the
  full transcript or repeat unrelated later sections.
- Stream your answer as the draft markdown only — no preamble, no JSON, no markdown fences around the whole output.
- Target compression is soft guidance; fidelity beats ratio."""

PROGRESS_MARKER_LINE = "[CONDENSED THROUGH: turn index {turn_index}]"


def build_agent_session_system(
    *,
    original_chatlog: str,
    progress_block: str = "",
    condensed_tail: str = "",
    include_full_log_context: bool = True,
    rag_supplement: str = "",
) -> str:
    log = (original_chatlog or "").strip()
    parts = [AGENT_SESSION_SYSTEM]
    if include_full_log_context and log:
        parts.extend(["", "ORIGINAL_CHATLOG:", log])
    elif log:
        parts.extend(
            [
                "",
                "ORIGINAL_CHATLOG: (omitted from system — full log is large; use the SEGMENT in the user message "
                "and RAG_SUPPLEMENT for distant callbacks. Turn order follows SEQUENTIAL PROGRESS.)",
            ]
        )
    if (progress_block or "").strip():
        parts.extend(["", (progress_block or "").strip()])
    if (condensed_tail or "").strip():
        parts.extend(
            [
                "",
                "CONDENSED_SO_FAR (tail — already done; do not repeat in your output):",
                (condensed_tail or "").strip(),
            ]
        )
    if (rag_supplement or "").strip():
        parts.extend(["", (rag_supplement or "").strip()])
    return "\n".join(parts)


def build_skeleton_repair_user_message(*, broken_json: str, chunk_id: str) -> str:
    tail = (broken_json or "").strip()
    if len(tail) > 14000:
        tail = tail[-14000:]
    return (
        f"CHUNK_ID: {chunk_id}\n\n"
        f"BROKEN_OR_TRUNCATED_JSON:\n{tail}\n\n"
        "Return ONLY the repaired complete JSON object."
    )
