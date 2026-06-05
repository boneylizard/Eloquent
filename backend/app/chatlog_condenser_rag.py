"""
RAG supplement for chatlog condenser (orchestrator, agent session, batch pipeline).

Reuses rag_utils.query_documents — no separate embedding stack.
Upload the transcript in Document Context, select doc ids, enable use_rag on condenser routes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("chatlog_condenser_rag")

# Smaller than chat RAG — condenser already carries segment + progress blocks
CONDENSER_RAG_TOP_K = 10
CONDENSER_RAG_QUERIES_MAX = 3

_RAG_SUPPLEMENT_HEADER = """RAG_SUPPLEMENT (retrieved document chunks — cross-reference only):
- Use for distant callbacks, corrections, thread-shifts, and continuity NOT visible in SEGMENT / CONDENSED_SO_FAR.
- Do NOT treat RAG as the canonical transcript; conversational order comes from ORIGINAL_CHATLOG turn indices.
- Do NOT repeat passages already reflected in CONDENSED_SO_FAR or the current segment.
- If RAG conflicts with the segment you are condensing, prefer the segment / progress markers."""


def _normalize_sig(text: str, *, max_len: int = 120) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t[:max_len]


def dedupe_rag_chunks(
    chunks: Sequence[Dict[str, Any]],
    *,
    partial_condensed: str = "",
    seen_sigs: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Drop chunks whose text already appears in partial_condensed or prior picks."""
    partial_sig = _normalize_sig(partial_condensed, max_len=2000) if partial_condensed else ""
    seen = set(seen_sigs or ())
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        body = (ch.get("chunk") or "").strip()
        if not body:
            continue
        sig = _normalize_sig(body)
        if sig in seen:
            continue
        if partial_sig and sig in partial_sig:
            continue
        if len(body) > 40 and body[:40].lower() in partial_sig:
            continue
        seen.add(sig)
        out.append(ch)
    return out


def build_step_rag_queries(
    *,
    segment_start: int,
    segment_end: int,
    partial_condensed: str = "",
    open_threads: Optional[Sequence[str]] = None,
    failsafe: bool = False,
) -> List[str]:
    """Construct up to CONDENSER_RAG_QUERIES_MAX retrieval queries for one condense step."""
    queries: List[str] = []
    queries.append(
        f"chat transcript turns {segment_start} through {segment_end} "
        f"reasoning corrections disagreements conclusions thread shifts"
    )
    tail = (partial_condensed or "").strip()
    if tail:
        excerpt = tail[-1200:] if len(tail) > 1200 else tail
        threads = [t.strip() for t in (open_threads or ()) if (t or "").strip()][:4]
        thread_bit = (" open threads: " + "; ".join(threads)) if threads else ""
        queries.append(
            f"continue condensation after: {excerpt[-400:]}{thread_bit} "
            "unresolved hooks callbacks prior claims"
        )
    if failsafe:
        queries.append(
            f"turns {segment_start}-{segment_end} corrections disagreements "
            "conclusions progress markers condensed through"
        )
    return queries[:CONDENSER_RAG_QUERIES_MAX]


def _query_once(
    question: str,
    doc_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        from . import rag_utils
    except ImportError:
        return [], "rag_utils unavailable"

    if not doc_ids:
        return [], None

    try:
        res = rag_utils.query_documents(
            question=question,
            doc_ids=doc_ids,
            top_k=CONDENSER_RAG_TOP_K,
            threshold=rag_utils.RAG_CHAT_SIMILARITY_THRESHOLD,
        )
    except Exception as e:
        logger.warning("Condenser RAG query failed: %s", e)
        return [], str(e)

    if res.get("status") != "success":
        err = res.get("error") or res.get("status")
        if err:
            logger.warning("Condenser RAG: %s", err)
        return [], None

    return list(res.get("chunks") or []), None


def query_rag_for_step(
    *,
    doc_ids: List[str],
    segment_start: int,
    segment_end: int,
    partial_condensed: str = "",
    open_threads: Optional[Sequence[str]] = None,
    failsafe: bool = False,
) -> str:
    """
    Run per-step RAG queries, dedupe, return RAG_SUPPLEMENT block (empty string if none).
    Never raises — failures log and return "".
    """
    ids = [d for d in (doc_ids or []) if (d or "").strip()]
    if not ids:
        return ""

    try:
        from . import rag_utils

        if not rag_utils.is_rag_available():
            logger.warning("Condenser RAG requested but RAG is not available")
            return ""
    except ImportError:
        logger.warning("Condenser RAG requested but rag_utils is missing")
        return ""

    queries = build_step_rag_queries(
        segment_start=segment_start,
        segment_end=segment_end,
        partial_condensed=partial_condensed,
        open_threads=open_threads,
        failsafe=failsafe,
    )

    merged: List[Dict[str, Any]] = []
    seen_sigs: Set[str] = set()
    for q in queries:
        batch, _ = _query_once(q, ids)
        if not batch:
            continue
        deduped = dedupe_rag_chunks(
            batch,
            partial_condensed=partial_condensed,
            seen_sigs=seen_sigs,
        )
        for ch in deduped:
            sig = _normalize_sig(ch.get("chunk") or "")
            seen_sigs.add(sig)
        merged.extend(deduped)
        if len(merged) >= CONDENSER_RAG_TOP_K * 2:
            break

    if not merged:
        logger.warning(
            "Condenser RAG: no chunks for turns %s–%s (docs=%s)",
            segment_start,
            segment_end,
            len(ids),
        )
        return ""

    merged = merged[: CONDENSER_RAG_TOP_K * 2]
    try:
        from . import rag_utils

        body = rag_utils.rag_processor.format_for_prompt(merged)
    except Exception:
        body = "\n\n".join(
            (ch.get("chunk") or "").strip() for ch in merged if (ch.get("chunk") or "").strip()
        )

    if not (body or "").strip():
        return ""

    return f"{_RAG_SUPPLEMENT_HEADER}\n\n{body.strip()}\n"


def append_rag_supplement_to_user(user: str, rag_block: str) -> str:
    block = (rag_block or "").strip()
    if not block:
        return user
    return f"{user.rstrip()}\n\n{block}"


def query_rag_for_batch_chunk(
    *,
    doc_ids: List[str],
    chunk_id: str,
    segment_start_turn: int,
    segment_end_turn: int,
    prior_context: str = "",
    open_threads: Optional[Sequence[str]] = None,
) -> str:
    """Single RAG supplement for batch skeleton/render passes."""
    return query_rag_for_step(
        doc_ids=doc_ids,
        segment_start=segment_start_turn,
        segment_end=segment_end_turn,
        partial_condensed=prior_context,
        open_threads=open_threads,
        failsafe=False,
    )
