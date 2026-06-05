"""
Chatlog condenser: lossless-on-meaning, lossy-on-words.

Two-stage pipeline per chunk (skeleton → dense draft), optional stitch pass, reconstruction eval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from .chatlog_condenser_rag import query_rag_for_batch_chunk
from .chatlog_condenser_prompt import (
    append_rag_supplement_to_prompt_body,
    EVAL_JSON_REPAIR_SYSTEM,
    EVAL_PROBE_SYSTEM,
    EVAL_SCORE_SYSTEM,
    RENDER_SYSTEM,
    SKELETON_JSON_REPAIR_SYSTEM,
    SKELETON_SYSTEM,
    STITCH_SYSTEM,
    build_eval_probe_user_message,
    build_eval_score_user_message,
    build_json_repair_user_message,
    build_render_user_message,
    build_skeleton_repair_user_message,
    build_skeleton_user_message,
    build_stitch_user_message,
)

logger = logging.getLogger("chatlog_condenser")

# ~4 chars per token heuristic (matches frontend packing)
CHARS_PER_TOKEN = 4

DEFAULT_CHUNK_TARGET_TOKENS = 16_000
DEFAULT_OVERLAP_TURNS = 5
DEFAULT_TARGET_RATIO = 0.4
DEFAULT_MAX_OUTPUT_TOKENS = 16_384
# Rough combined prompt size (full log + segment + prior) above which we warn (do not block).
FULL_CONTEXT_WARN_TOKENS_EST = 100_000
SKELETON_MIN_OUTPUT_TOKENS = 16_384
SKELETON_MAX_OUTPUT_TOKENS = 32_768

CONDENSER_API_MAX_ATTEMPTS = 3
_CONDENSER_API_DISCONNECT_MSG = (
    "Provider closed the connection during condensing (timeout or gateway reset). "
    "Try a smaller chunk budget or retry in a moment."
)


def _format_condenser_api_error(
    exc: HTTPException,
    *,
    endpoint_name: str = "API",
) -> str:
    """User-facing detail: config/auth vs transient provider disconnect vs other API errors."""
    detail = str(exc.detail or "").strip()
    low = detail.lower()
    if exc.status_code in (401, 403):
        return f"Endpoint authentication failed ({endpoint_name}): {detail}"
    if exc.status_code == 404 or "not configured" in low or "not found" in low:
        return f"Endpoint configuration error ({endpoint_name}): {detail}"
    if _condenser_llm_error_is_transient(exc):
        tech = f" — {detail}" if detail else ""
        return f"{_CONDENSER_API_DISCONNECT_MSG}{tech}"
    return detail or f"API error from {endpoint_name} (HTTP {exc.status_code})"


def _condenser_llm_error_is_transient(exc: BaseException) -> bool:
    """502 / disconnect errors worth retrying at the condenser layer."""
    import httpx

    from .openai_compat import _openai_compat_is_transient_upstream_for_retry

    if isinstance(exc, httpx.RequestError):
        return _openai_compat_is_transient_upstream_for_retry(
            exc, include_read_write_timeout=True
        )
    if isinstance(exc, HTTPException) and exc.status_code == 502:
        detail = str(exc.detail or "").lower()
        return any(
            marker in detail
            for marker in (
                "remoteprotocolerror",
                "server disconnected",
                "cannot connect",
            )
        )
    return False


@dataclass
class Turn:
    speaker: str
    content: str
    index: int = 0


@dataclass
class ChunkSpec:
    chunk_id: str
    turns: List[Turn]
    overlap_turn_count: int = 0


@dataclass
class CondenseStats:
    input_turns: int = 0
    input_tokens_est: int = 0
    output_tokens_est: int = 0
    chunk_count: int = 1
    target_ratio: float = DEFAULT_TARGET_RATIO
    achieved_ratio: float = 0.0
    include_full_log_context: bool = True
    context_tokens_est: int = 0
    context_warning: Optional[str] = None


@dataclass
class CondenseResult:
    condensed_markdown: str
    skeleton_full: Dict[str, Any]
    chunk_skeletons: List[Dict[str, Any]] = field(default_factory=list)
    stats: CondenseStats = field(default_factory=CondenseStats)
    eval_result: Optional[Dict[str, Any]] = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN)


def estimate_per_call_context_tokens(
    *,
    full_chatlog_md: str,
    segment_md: str,
    prior_context: str = "",
    include_full_log_context: bool = True,
) -> int:
    """Rough token estimate for one skeleton/render LLM prompt (input side)."""
    total = estimate_tokens(segment_md) + estimate_tokens(prior_context or "")
    if include_full_log_context:
        total += estimate_tokens(full_chatlog_md)
    return total


def context_size_warning(tokens_est: int) -> Optional[str]:
    if tokens_est <= FULL_CONTEXT_WARN_TOKENS_EST:
        return None
    return (
        f"Estimated per-call context ~{tokens_est:,} tokens (full log + segment + prior tail) "
        f"exceeds ~{FULL_CONTEXT_WARN_TOKENS_EST:,} — ensure your model context window is large enough."
    )


def normalize_endpoint_model_id(model_name: str) -> str:
    """Fix duplicated prefix from UI bug (endpoint-endpoint-* → endpoint-*)."""
    if model_name and model_name.startswith("endpoint-endpoint-"):
        return "endpoint-" + model_name[len("endpoint-endpoint-") :]
    return model_name


def format_turns_markdown(turns: List[Turn]) -> str:
    lines: List[str] = []
    for t in turns:
        body = (t.content or "").strip()
        if not body:
            continue
        lines.append(f"**{t.speaker}:** {body}")
        lines.append("")
    return "\n".join(lines).strip()


def extract_first_json(text: str) -> Optional[str]:
    """Extract outermost {...} respecting JSON string boundaries."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def json_closing_suffix(blob: str) -> str:
    """Suffix to close an unterminated JSON string and open brackets/braces."""
    in_string = False
    escape = False
    stack: List[str] = []
    for ch in blob:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    suffix = ""
    if in_string:
        suffix += '"'
    suffix += "".join(reversed(stack))
    return suffix


def trim_to_last_complete_move(blob: str) -> str:
    """Drop a truncated trailing move object inside the moves array."""
    moves_key = blob.find('"moves"')
    if moves_key == -1:
        return blob
    arr_start = blob.find("[", moves_key)
    if arr_start == -1:
        return blob
    last_comma = blob.rfind("},")
    if last_comma > arr_start:
        return blob[: last_comma + 1]
    last_brace = blob.rfind("}")
    if last_brace > arr_start:
        return blob[: last_brace + 1]
    return blob


def repair_truncated_json_blob(blob: str) -> str:
    """Best-effort repair for model output cut off mid-JSON."""
    blob = (blob or "").strip()
    if not blob:
        return blob
    candidates = [
        blob + json_closing_suffix(blob),
        trim_to_last_complete_move(blob) + json_closing_suffix(trim_to_last_complete_move(blob)),
    ]
    seen: set = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            json.loads(cand)
            return cand
        except json.JSONDecodeError:
            continue
    return candidates[0]


def salvage_moves_from_blob(blob: str) -> List[Dict[str, Any]]:
    """Extract complete move objects from broken skeleton JSON via raw_decode."""
    moves: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{\s*"id"\s*:\s*"[^"]+"', blob):
        start = match.start()
        try:
            obj, _end = decoder.raw_decode(blob, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            moves.append(obj)
    return moves


def minimal_skeleton(chunk_id: str = "") -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "moves": [],
        "open_threads": [],
        "cross_chunk_callbacks": [],
        "_degraded": True,
    }


def salvage_skeleton_object(blob: str, *, chunk_id: str = "") -> Dict[str, Any]:
    """Build valid skeleton from partial JSON; empty moves → minimal degraded skeleton."""
    moves = salvage_moves_from_blob(blob)
    if not moves:
        return minimal_skeleton(chunk_id)
    cid = chunk_id
    m = re.search(r'"chunk_id"\s*:\s*"([^"]*)"', blob)
    if m:
        cid = m.group(1) or cid
    open_threads: List[str] = []
    callbacks: List[str] = []
    ot = re.search(r'"open_threads"\s*:\s*\[(.*?)\]', blob, re.S)
    if ot:
        try:
            open_threads = json.loads("[" + ot.group(1) + "]")
        except json.JSONDecodeError:
            pass
    cb = re.search(r'"cross_chunk_callbacks"\s*:\s*\[(.*?)\]', blob, re.S)
    if cb:
        try:
            callbacks = json.loads("[" + cb.group(1) + "]")
        except json.JSONDecodeError:
            pass
    logger.warning(
        "Salvaged skeleton JSON: %d moves from truncated output (chunk_id=%s)",
        len(moves),
        cid,
    )
    return {
        "chunk_id": cid,
        "moves": moves,
        "open_threads": open_threads if isinstance(open_threads, list) else [],
        "cross_chunk_callbacks": callbacks if isinstance(callbacks, list) else [],
        "_salvaged": True,
    }


def _normalize_skeleton_dict(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed.get("moves"), list):
        parsed["moves"] = []
    parsed.setdefault("open_threads", [])
    parsed.setdefault("cross_chunk_callbacks", [])
    return parsed


def parse_json_object(
    raw: str,
    *,
    context: str = "",
    chunk_id: str = "",
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse model JSON with local repair/salvage; never raises on malformed JSON."""
    cleaned = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.I)
    if fence:
        cleaned = fence.group(1).strip()
    blob = extract_first_json(cleaned) or cleaned
    if not blob:
        if fallback is not None:
            return dict(fallback)
        if chunk_id or "skeleton" in (context or "").lower():
            return minimal_skeleton(chunk_id)
        return {}

    for candidate in (blob, repair_truncated_json_blob(blob)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if "moves" in parsed or chunk_id or "skeleton" in (context or "").lower():
                    return _normalize_skeleton_dict(parsed)
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    salvaged = salvage_skeleton_object(blob, chunk_id=chunk_id)
    if salvaged.get("moves") or salvaged.get("_degraded"):
        return _normalize_skeleton_dict(salvaged)
    if fallback is not None:
        return dict(fallback)
    if chunk_id or "skeleton" in (context or "").lower():
        return minimal_skeleton(chunk_id)
    return {}


def parse_json_object_optional(
    raw: str, *, context: str = ""
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Like parse_json_object but returns (None, error_message) instead of raising."""
    try:
        return parse_json_object(raw, context=context), None
    except ValueError as e:
        return None, str(e)


def output_tokens_for_chunk(chunk: ChunkSpec) -> int:
    """Scale condenser LLM output budget with chunk size (floor = DEFAULT_MAX_OUTPUT_TOKENS)."""
    est = estimate_tokens(format_turns_markdown(chunk.turns))
    scaled = max(DEFAULT_MAX_OUTPUT_TOKENS, est * 2)
    return min(SKELETON_MAX_OUTPUT_TOKENS, scaled)


def skeleton_max_tokens_for_chunk(chunk: ChunkSpec) -> int:
    """Alias for skeleton extraction (same scaling as output_tokens_for_chunk)."""
    return output_tokens_for_chunk(chunk)


_SPEAKER_LINE = re.compile(
    r"^(?:\*\*)?(?P<speaker>[^:*\n]+?)(?:\*\*)?\s*:\s*(?P<body>.*)$",
    re.MULTILINE,
)
_ROLE_PREFIX = re.compile(
    r"^(?P<role>user|assistant|human|ai|system|model)\s*:\s*(?P<body>.*)$",
    re.I | re.MULTILINE,
)


def parse_chatlog(text: str) -> List[Turn]:
    """Parse txt/md chatlogs into ordered speaker turns."""
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return []

    turns: List[Turn] = []
    idx = 0

    # Double-newline separated **Speaker:** blocks (export-style markdown)
    blocks = re.split(r"\n\s*\n(?=(?:\*\*)?[^:\n]+(?:\*\*)?\s*:)", raw)
    if len(blocks) > 1:
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            first_line = block.split("\n", 1)[0]
            m = _SPEAKER_LINE.match(first_line)
            if not m:
                continue
            speaker = m.group("speaker").strip()
            first_body = m.group("body").strip()
            rest = block.split("\n", 1)
            body = first_body
            if len(rest) > 1:
                body = (first_body + "\n" + rest[1]).strip() if first_body else rest[1].strip()
            elif not body:
                continue
            turns.append(Turn(speaker=speaker, content=body, index=idx))
            idx += 1
        if turns:
            return turns

    # Line-oriented User:/Assistant: or role:
    current_speaker: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal idx, current_speaker, buf
        if current_speaker and buf:
            body = "\n".join(buf).strip()
            if body:
                turns.append(Turn(speaker=current_speaker, content=body, index=idx))
                idx += 1
        buf = []

    for line in raw.split("\n"):
        sm = _SPEAKER_LINE.match(line)
        rm = _ROLE_PREFIX.match(line) if not sm else None
        if sm:
            flush()
            current_speaker = sm.group("speaker").strip()
            first = sm.group("body").strip()
            buf = [first] if first else []
        elif rm:
            flush()
            role = rm.group("role").strip().title()
            if role.lower() in ("ai", "model", "assistant"):
                role = "Assistant"
            elif role.lower() in ("human", "user"):
                role = "User"
            current_speaker = role
            buf = [rm.group("body").strip()] if rm.group("body").strip() else []
        elif line.strip() == "" and not buf:
            continue
        else:
            if current_speaker is None:
                current_speaker = "Speaker"
            buf.append(line)
    flush()

    if turns:
        return turns

    # Fallback: paragraph blocks separated by double newlines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    for i, block in enumerate(blocks):
        turns.append(Turn(speaker=f"Turn {i + 1}", content=block, index=i))
    return turns


def chunk_turns_with_overlap(
    turns: List[Turn],
    *,
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    overlap_turns: int = DEFAULT_OVERLAP_TURNS,
) -> List[ChunkSpec]:
    if not turns:
        return []

    chunk_target_tokens = max(1, int(chunk_target_tokens or DEFAULT_CHUNK_TARGET_TOKENS))
    max_chars = chunk_target_tokens * CHARS_PER_TOKEN
    overlap_turns = max(0, int(overlap_turns or 0))
    chunks: List[ChunkSpec] = []
    i = 0
    chunk_num = 0

    while i < len(turns):
        chunk_turns: List[Turn] = []
        chars = 0
        start_i = i
        if chunk_num > 0 and overlap_turns > 0:
            overlap_start = max(0, start_i - overlap_turns)
            for t in turns[overlap_start:start_i]:
                chunk_turns.append(t)
                chars += len(t.speaker) + len(t.content) + 8

        j = start_i
        while j < len(turns):
            t = turns[j]
            add = len(t.speaker) + len(t.content) + 8
            if chunk_turns and chars + add > max_chars:
                break
            chunk_turns.append(t)
            chars += add
            j += 1

        if j == start_i:
            chunk_turns.append(turns[j])
            j += 1

        overlap_count = 0
        if chunk_num > 0 and overlap_turns > 0:
            overlap_count = min(overlap_turns, len(chunk_turns))

        chunks.append(
            ChunkSpec(
                chunk_id=f"chunk_{chunk_num + 1}",
                turns=chunk_turns,
                overlap_turn_count=overlap_count,
            )
        )
        chunk_num += 1
        if j >= len(turns):
            break
        i = j

    return chunks


def estimate_condenser_llm_passes(
    chunk_count: int,
    *,
    run_eval: bool = False,
) -> int:
    """
    Expected LLM calls for condense (skeleton + render per chunk; optional stitch + eval).
    Does not count rare per-chunk skeleton JSON repair calls.
    """
    n = max(1, int(chunk_count or 1))
    passes = n * 2
    if n > 1:
        passes += 1
    if run_eval:
        passes += 2
    return passes


def skeleton_tail_for_context(skeleton: Dict[str, Any], *, max_moves: int = 24) -> str:
    """Compact JSON tail for cross-chunk continuity."""
    moves = skeleton.get("moves") or []
    if isinstance(moves, list) and len(moves) > max_moves:
        moves = moves[-max_moves:]
    tail = {
        "moves": moves,
        "open_threads": skeleton.get("open_threads") or [],
        "cross_chunk_callbacks": skeleton.get("cross_chunk_callbacks") or [],
    }
    return json.dumps(tail, ensure_ascii=False)


def merge_skeletons(chunk_skeletons: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_moves: List[Dict[str, Any]] = []
    open_threads: List[str] = []
    callbacks: List[str] = []
    for sk in chunk_skeletons:
        moves = sk.get("moves") or []
        if isinstance(moves, list):
            for m in moves:
                if isinstance(m, dict):
                    all_moves.append(m)
        for key, dest in (("open_threads", open_threads), ("cross_chunk_callbacks", callbacks)):
            val = sk.get(key)
            if isinstance(val, list):
                for x in val:
                    s = str(x).strip()
                    if s and s not in dest:
                        dest.append(s)
    return {
        "moves": all_moves,
        "open_threads": open_threads,
        "cross_chunk_callbacks": callbacks,
    }


def score_eval_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.get("pass") is True)
    fidelity = (passed / total) if total else 0.0
    failures = [r.get("note") for r in results if not r.get("pass")]
    modes: List[str] = []
    for note in failures:
        n = (note or "").lower()
        if "correction" in n or "catch" in n:
            modes.append("missing_correction")
        elif "thread" in n or "shift" in n:
            modes.append("missing_thread_shift")
        elif "pushback" in n or "disagree" in n:
            modes.append("missing_pushback")
        elif "flat" in n or "summary" in n:
            modes.append("flattening")
    return {
        "pass_count": passed,
        "total": total,
        "fidelity_score": round(fidelity, 4),
        "failure_modes": sorted(set(modes)),
    }


async def call_llm(
    *,
    model_manager: Any,
    model_name: str,
    system: str,
    user: str,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = 0.2,
    skip_pruning: bool = True,
    rotation_candidate_ids: Optional[List[str]] = None,
    rotation_cursor_key: Optional[str] = None,
) -> str:
    from . import inference
    from .openai_compat import (
        approx_openai_messages_payload_chars,
        collect_openai_compatible_stream_text,
        get_configured_endpoint,
        is_api_endpoint,
        prepare_endpoint_request,
    )

    model_name = normalize_endpoint_model_id(model_name)
    if not model_name:
        raise ValueError("model_name is required")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    request_data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": True,
    }
    if skip_pruning:
        request_data["_skip_openai_message_pruning"] = True

    if is_api_endpoint(model_name):
        endpoint_cfg = get_configured_endpoint(
            model_name,
            rotation_candidate_ids=rotation_candidate_ids,
            rotation_cursor_key=rotation_cursor_key,
        )
        if not endpoint_cfg:
            raise ValueError(f"API endpoint not configured: {model_name}")
        endpoint_name = endpoint_cfg.get("name") or model_name
        endpoint_config, url, prepared = prepare_endpoint_request(model_name, request_data)
        msg_chars = approx_openai_messages_payload_chars(prepared.get("messages", []))
        log_fn = logger.warning if msg_chars >= 80_000 else logger.info
        log_fn(
            "call_llm API request: ~%s message chars (system=%s, user=%s), "
            "max_tokens=%s, model=%s, upstream=stream_aggregate",
            msg_chars,
            len(system),
            len(user),
            max_tokens,
            model_name,
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(CONDENSER_API_MAX_ATTEMPTS):
            try:
                text_out = await collect_openai_compatible_stream_text(
                    endpoint_config, url, prepared
                )
                return (text_out or "").strip()
            except HTTPException as e:
                last_exc = e
                if (
                    _condenser_llm_error_is_transient(e)
                    and attempt < CONDENSER_API_MAX_ATTEMPTS - 1
                ):
                    delay = 2.0 * (attempt + 1)
                    logger.warning(
                        "call_llm transient upstream error (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1,
                        CONDENSER_API_MAX_ATTEMPTS,
                        delay,
                        e.detail,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise HTTPException(
                    status_code=e.status_code,
                    detail=_format_condenser_api_error(e, endpoint_name=endpoint_name),
                ) from e
        if last_exc is not None:
            if isinstance(last_exc, HTTPException):
                raise HTTPException(
                    status_code=last_exc.status_code,
                    detail=_format_condenser_api_error(
                        last_exc, endpoint_name=endpoint_name
                    ),
                ) from last_exc
            raise last_exc
        return ""

    if not model_manager:
        raise ValueError("Local model requested but model_manager is unavailable")
    prompt = f"{system}\n\n{user}"
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


def _parse_skeleton_response(raw: str, *, chunk_id: str) -> Dict[str, Any]:
    sk = parse_json_object(raw, context=f"skeleton {chunk_id}", chunk_id=chunk_id)
    sk.setdefault("chunk_id", chunk_id)
    return sk


def _parsed_json_is_usable(obj: Dict[str, Any], fallback: Dict[str, Any]) -> bool:
    if not obj:
        return False
    if "probes" in fallback:
        probes = obj.get("probes")
        return isinstance(probes, list) and len(probes) > 0
    if "results" in fallback:
        results = obj.get("results")
        return isinstance(results, list) and len(results) > 0
    if "moves" in fallback or obj.get("moves") is not None:
        moves = obj.get("moves")
        if isinstance(moves, list) and moves:
            return True
        return not obj.get("_degraded")
    return bool(obj)


async def _parse_json_with_llm_repair(
    raw: str,
    *,
    model_manager: Any,
    model_name: str,
    repair_system: str,
    schema_hint: str,
    context: str,
    chunk_id: str = "",
    fallback: Dict[str, Any],
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    """Local parse → one LLM repair → fallback. Never raises."""
    obj = parse_json_object(raw, context=context, chunk_id=chunk_id, fallback=None)
    if _parsed_json_is_usable(obj, fallback):
        return obj

    if model_name:
        try:
            repaired_raw = await call_llm(
                model_manager=model_manager,
                model_name=model_name,
                system=repair_system,
                user=build_json_repair_user_message(
                    broken_json=raw,
                    schema_hint=schema_hint,
                ),
                max_tokens=max_tokens,
                temperature=0.0,
            )
            repaired = parse_json_object(
                repaired_raw, context=context, chunk_id=chunk_id, fallback=None
            )
            if _parsed_json_is_usable(repaired, fallback):
                return repaired
        except Exception as e:
            logger.warning("JSON LLM repair failed (%s): %s", context, e)

    return dict(fallback)


async def extract_skeleton(
    *,
    model_manager: Any,
    model_name: str,
    chunk: ChunkSpec,
    prior_context: str = "",
    full_chatlog_md: str = "",
    include_full_log_context: bool = True,
    rag_supplement: str = "",
) -> Dict[str, Any]:
    segment_md = format_turns_markdown(chunk.turns)
    user = build_skeleton_user_message(
        chunk_id=chunk.chunk_id,
        segment_transcript=segment_md,
        prior_context=prior_context,
        full_chatlog=full_chatlog_md,
        overlap_turn_count=chunk.overlap_turn_count,
        include_full_log_context=include_full_log_context,
    )
    user = append_rag_supplement_to_prompt_body(user, rag_supplement)
    max_tok = output_tokens_for_chunk(chunk)
    raw = await call_llm(
        model_manager=model_manager,
        model_name=model_name,
        system=SKELETON_SYSTEM,
        user=user,
        max_tokens=max_tok,
    )
    sk = _parse_skeleton_response(raw, chunk_id=chunk.chunk_id)
    if sk.get("moves"):
        return sk

    logger.warning(
        "Skeleton JSON degraded for %s; attempting one repair call",
        chunk.chunk_id,
    )
    repair_user = build_skeleton_repair_user_message(
        broken_json=raw, chunk_id=chunk.chunk_id
    )
    repaired_raw = await call_llm(
        model_manager=model_manager,
        model_name=model_name,
        system=SKELETON_JSON_REPAIR_SYSTEM,
        user=repair_user,
        max_tokens=max_tok,
    )
    sk2 = _parse_skeleton_response(repaired_raw, chunk_id=chunk.chunk_id)
    if sk2.get("moves"):
        return sk2
    if not sk2.get("_degraded"):
        return sk2
    logger.warning(
        "Skeleton extraction degraded for %s (empty moves); continuing pipeline",
        chunk.chunk_id,
    )
    return sk2 if sk2.get("chunk_id") else minimal_skeleton(chunk.chunk_id)


async def render_dense_draft(
    *,
    model_manager: Any,
    model_name: str,
    chunk: ChunkSpec,
    skeleton: Dict[str, Any],
    full_chatlog_md: str = "",
    include_full_log_context: bool = True,
    rag_supplement: str = "",
) -> str:
    segment_md = format_turns_markdown(chunk.turns)
    user = build_render_user_message(
        skeleton_json=json.dumps(skeleton, ensure_ascii=False, separators=(",", ":")),
        segment_transcript=segment_md,
        full_chatlog=full_chatlog_md,
        chunk_id=chunk.chunk_id,
        overlap_turn_count=chunk.overlap_turn_count,
        include_full_log_context=include_full_log_context,
    )
    user = append_rag_supplement_to_prompt_body(user, rag_supplement)
    return await call_llm(
        model_manager=model_manager,
        model_name=model_name,
        system=RENDER_SYSTEM,
        user=user,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )


async def stitch_segments(
    *,
    model_manager: Any,
    model_name: str,
    segment_markdowns: List[str],
    prior_skeleton_tail: str,
    full_chatlog_md: str = "",
    include_full_log_context: bool = True,
) -> str:
    if len(segment_markdowns) == 1:
        return segment_markdowns[0].strip()
    parts = []
    for i, md in enumerate(segment_markdowns):
        parts.append(f"--- SEGMENT {i + 1} ---\n{md.strip()}\n")
    user = build_stitch_user_message(
        prior_skeleton_tail=prior_skeleton_tail,
        segments="\n".join(parts),
        full_chatlog=full_chatlog_md,
        include_full_log_context=include_full_log_context,
    )
    return await call_llm(
        model_manager=model_manager,
        model_name=model_name,
        system=STITCH_SYSTEM,
        user=user,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )


async def run_reconstruction_eval(
    *,
    model_manager: Any,
    model_name: str,
    skeleton: Dict[str, Any],
    condensed_markdown: str,
    probe_count_hint: int = 8,
) -> Dict[str, Any]:
    empty_summary = score_eval_results([])
    sk_json = json.dumps(skeleton, ensure_ascii=False, separators=(",", ":"))
    probe_user = build_eval_probe_user_message(skeleton_json=sk_json)
    probe_raw = await call_llm(
        model_manager=model_manager,
        model_name=model_name,
        system=EVAL_PROBE_SYSTEM,
        user=probe_user,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    probes_obj = await _parse_json_with_llm_repair(
        probe_raw,
        model_manager=model_manager,
        model_name=model_name,
        repair_system=EVAL_JSON_REPAIR_SYSTEM,
        schema_hint='{"probes":[{"id":"p1","question":"...","structural_focus":"...","oracle_hints":[]}]}',
        context="eval probes",
        fallback={"probes": []},
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    probes = probes_obj.get("probes") or []
    if not isinstance(probes, list):
        probes = []
    if not probes:
        logger.warning("Reconstruction eval skipped: probe JSON unusable after repair")
        return {
            "skipped": True,
            "reason": "probe_json_unparseable",
            "probes": [],
            "results": [],
            "summary": empty_summary,
        }
    if probe_count_hint > 0:
        probes = probes[: max(probe_count_hint, 12)]

    score_user = build_eval_score_user_message(
        probes_json=json.dumps({"probes": probes}, ensure_ascii=False, separators=(",", ":")),
        condensed_draft=condensed_markdown,
    )
    score_raw = await call_llm(
        model_manager=model_manager,
        model_name=model_name,
        system=EVAL_SCORE_SYSTEM,
        user=score_user,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    score_obj = await _parse_json_with_llm_repair(
        score_raw,
        model_manager=model_manager,
        model_name=model_name,
        repair_system=EVAL_JSON_REPAIR_SYSTEM,
        schema_hint='{"results":[{"probe_id":"p1","pass":true,"confidence":0.9,"note":"..."}],"summary":{"pass_count":0,"total":0,"fidelity_score":0.0,"failure_modes":[]}}',
        context="eval score",
        fallback={"results": [], "summary": empty_summary},
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    results = score_obj.get("results") or []
    if not isinstance(results, list):
        results = []
    summary = score_obj.get("summary")
    if not isinstance(summary, dict):
        summary = score_eval_results(results)
    else:
        summary.setdefault("fidelity_score", score_eval_results(results)["fidelity_score"])
    return {
        "probes": probes,
        "results": results,
        "summary": summary,
    }


async def condense_chatlog(
    *,
    model_manager: Any,
    model_name: str,
    text: str,
    target_ratio: float = DEFAULT_TARGET_RATIO,
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    overlap_turns: int = DEFAULT_OVERLAP_TURNS,
    run_eval: bool = True,
    eval_model_name: Optional[str] = None,
    include_full_log_context: bool = True,
    use_rag: bool = False,
    rag_docs: Optional[List[str]] = None,
) -> CondenseResult:
    turns = parse_chatlog(text)
    if not turns:
        raise ValueError("No speaker turns found in input")

    input_md = format_turns_markdown(turns)
    input_tokens = estimate_tokens(input_md)
    ratio = float(target_ratio if target_ratio is not None else DEFAULT_TARGET_RATIO)
    if ratio > 1:
        ratio = ratio / 100.0
    target_ratio = ratio if ratio > 0 else DEFAULT_TARGET_RATIO

    chunk_tokens = max(1, int(chunk_target_tokens or DEFAULT_CHUNK_TARGET_TOKENS))
    overlap = max(0, int(overlap_turns if overlap_turns is not None else DEFAULT_OVERLAP_TURNS))

    chunks = chunk_turns_with_overlap(
        turns,
        chunk_target_tokens=chunk_tokens,
        overlap_turns=overlap,
    )

    chunk_skeletons: List[Dict[str, Any]] = []
    segment_drafts: List[str] = []
    prior_tail = ""
    use_full_log = bool(include_full_log_context)
    rag_doc_ids = [d for d in (rag_docs or []) if (d or "").strip()]
    use_rag_supplement = bool(use_rag and rag_doc_ids)
    max_context_tokens = 0
    context_warning: Optional[str] = None

    for chunk in chunks:
        segment_md = format_turns_markdown(chunk.turns)
        seg_start = chunk.turns[0].index if chunk.turns else 0
        seg_end = chunk.turns[-1].index if chunk.turns else seg_start
        rag_block = ""
        if use_rag_supplement:
            open_threads: List[str] = []
            if prior_tail:
                try:
                    tail_sk = json.loads(prior_tail) if prior_tail.strip().startswith("{") else {}
                    open_threads = list(tail_sk.get("open_threads") or [])
                except (json.JSONDecodeError, TypeError, AttributeError):
                    open_threads = []
            rag_block = query_rag_for_batch_chunk(
                doc_ids=rag_doc_ids,
                chunk_id=chunk.chunk_id,
                segment_start_turn=seg_start,
                segment_end_turn=seg_end,
                prior_context=prior_tail,
                open_threads=open_threads,
            )
        ctx_est = estimate_per_call_context_tokens(
            full_chatlog_md=input_md,
            segment_md=segment_md,
            prior_context=prior_tail,
            include_full_log_context=use_full_log,
        )
        max_context_tokens = max(max_context_tokens, ctx_est)
        warn = context_size_warning(ctx_est)
        if warn:
            logger.warning(
                "Chatlog condenser large context for %s: ~%s tokens est.",
                chunk.chunk_id,
                ctx_est,
            )
            if context_warning is None:
                context_warning = warn

        sk = await extract_skeleton(
            model_manager=model_manager,
            model_name=model_name,
            chunk=chunk,
            prior_context=prior_tail,
            full_chatlog_md=input_md,
            include_full_log_context=use_full_log,
            rag_supplement=rag_block,
        )
        chunk_skeletons.append(sk)
        prior_tail = skeleton_tail_for_context(sk)
        draft = await render_dense_draft(
            model_manager=model_manager,
            model_name=model_name,
            chunk=chunk,
            skeleton=sk,
            full_chatlog_md=input_md,
            include_full_log_context=use_full_log,
            rag_supplement=rag_block,
        )
        segment_drafts.append(draft)

    full_skeleton = merge_skeletons(chunk_skeletons)
    prior_for_stitch = skeleton_tail_for_context(full_skeleton, max_moves=40)

    if len(segment_drafts) > 1:
        condensed = await stitch_segments(
            model_manager=model_manager,
            model_name=model_name,
            segment_markdowns=segment_drafts,
            prior_skeleton_tail=prior_for_stitch,
            full_chatlog_md=input_md,
            include_full_log_context=use_full_log,
        )
    else:
        condensed = segment_drafts[0].strip()

    out_tokens = estimate_tokens(condensed)
    stats = CondenseStats(
        input_turns=len(turns),
        input_tokens_est=input_tokens,
        output_tokens_est=out_tokens,
        chunk_count=len(chunks),
        target_ratio=target_ratio,
        achieved_ratio=round(out_tokens / max(1, input_tokens), 4),
        include_full_log_context=use_full_log,
        context_tokens_est=max_context_tokens,
        context_warning=context_warning,
    )

    eval_result = None
    if run_eval:
        eval_model = eval_model_name or model_name
        eval_result = await run_reconstruction_eval(
            model_manager=model_manager,
            model_name=eval_model,
            skeleton=full_skeleton,
            condensed_markdown=condensed,
        )

    return CondenseResult(
        condensed_markdown=condensed,
        skeleton_full=full_skeleton,
        chunk_skeletons=chunk_skeletons,
        stats=stats,
        eval_result=eval_result,
    )
