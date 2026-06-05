"""
In-memory agentic condenser sessions: chat with full log + run-scoped history, streaming drafts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import HTTPException

from .chatlog_condenser import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TARGET_RATIO,
    Turn,
    _format_condenser_api_error,
    format_turns_markdown,
    normalize_endpoint_model_id,
    parse_chatlog,
)
from .chatlog_condenser_prompt import build_agent_session_system
from .chatlog_condenser_rag import query_rag_for_step

logger = logging.getLogger("chatlog_condenser_session")

# Default user instructions (also used when API omits initial_user_message)
DEFAULT_AGENT_FIRST_USER_MESSAGE = (
    "Begin sequential partial condensing of ORIGINAL_CHATLOG from turn index 0. "
    "Condense only the first ~15–25 speaker turns (or until a natural break), not the whole log. "
    "Output only that new dense draft segment. "
    "End with [CONDENSED THROUGH: turn index N] where N is the last turn you condensed."
)
DEFAULT_AGENT_CONTINUE_USER_MESSAGE = (
    "Continue sequential condensing from the next turn after SEQUENTIAL PROGRESS. "
    "Output only the new dense draft segment for turns not yet condensed. "
    "Do not repeat CONDENSED_SO_FAR. "
    "End with [CONDENSED THROUGH: turn index N]."
)

CONDENSED_TAIL_EXCERPT_CHARS = 2400
INITIAL_SEGMENT_MAX_TURNS = 25

_PROGRESS_BRACKET_RE = re.compile(
    r"\[CONDENSED\s+THROUGH:\s*(?:.*?\bturn\s*(?:index\s*)?)?(\d+)\s*\]",
    re.IGNORECASE,
)
_PROGRESS_HTML_RE = re.compile(
    r"<!--\s*progress:\s*(?:(?:chunk\s*\d+\s*/\s*)?(?:through\s+)?turn\s*)?(\d+)(?:\s+of\s+(\d+))?\s*-->",
    re.IGNORECASE,
)
_CONTINUE_HINT_RE = re.compile(
    r"\b(continue|pick\s*up|resume|go\s+on|next\s+chunk|keep\s+going)\b",
    re.IGNORECASE,
)


def parse_progress_marker(text: str) -> Optional[int]:
    """Return last condensed turn index (0-based) from assistant output, or None."""
    if not (text or "").strip():
        return None
    last: Optional[int] = None
    for pat in (_PROGRESS_BRACKET_RE, _PROGRESS_HTML_RE):
        for m in pat.finditer(text):
            try:
                last = int(m.group(1))
            except (TypeError, ValueError):
                continue
    return last


def format_progress_marker(turn_index: int) -> str:
    return f"[CONDENSED THROUGH: turn index {turn_index}]"


def build_progress_block(
    *,
    last_turn_index: int,
    total_turns: int,
    segment_start_turn: int,
) -> str:
    lines = [
        "SEQUENTIAL PROGRESS:",
        f"- Total turns in ORIGINAL_CHATLOG: {total_turns}",
        f"- Already condensed through turn index: {last_turn_index} (inclusive)",
        f"- Continue from turn index: {segment_start_turn}",
        "- Do NOT re-condense earlier turns unless the user explicitly asks to revise them.",
    ]
    if segment_start_turn < total_turns:
        lines.append(
            f"- This pass: condense turns {segment_start_turn} onward "
            f"(~{INITIAL_SEGMENT_MAX_TURNS} turns or until a natural break)."
        )
    else:
        lines.append("- All turns appear condensed; only revise if the user requests edits.")
    return "\n".join(lines)


def condensed_tail_excerpt(partial: str, *, max_chars: int = CONDENSED_TAIL_EXCERPT_CHARS) -> str:
    text = (partial or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return "…\n" + text[-max_chars:]


def is_continue_instruction(message: str) -> bool:
    return bool(_CONTINUE_HINT_RE.search(message or ""))


def segment_transcript_for_turns(
    turns: List[Turn], *, start_index: int, max_turns: int = INITIAL_SEGMENT_MAX_TURNS
) -> str:
    if not turns or start_index >= len(turns):
        return ""
    end = min(len(turns), start_index + max(1, max_turns))
    return format_turns_markdown(turns[start_index:end])


@dataclass
class SessionMessage:
    role: str  # user | assistant
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass
class SessionSettings:
    model_name: str
    target_ratio: float = DEFAULT_TARGET_RATIO
    include_full_log_context: bool = True
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.2
    use_rag: bool = False
    rag_doc_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_ratio": self.target_ratio,
            "include_full_log_context": self.include_full_log_context,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "use_rag": self.use_rag,
            "rag_doc_ids": list(self.rag_doc_ids),
        }


@dataclass
class CondenserSession:
    session_id: str
    original_log: str
    settings: SessionSettings
    messages: List[SessionMessage] = field(default_factory=list)
    partial_condensed: str = ""
    status: str = "active"  # active | streaming
    total_turn_count: int = 0
    last_condensed_turn_index: int = -1
    progress_marker: str = ""
    parsed_turns: List[Turn] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def segment_start_turn(self) -> int:
        return self.last_condensed_turn_index + 1

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "original_log": self.original_log,
            "original_log_chars": len(self.original_log or ""),
            "partial_condensed": self.partial_condensed,
            "total_turn_count": self.total_turn_count,
            "last_condensed_turn_index": self.last_condensed_turn_index,
            "progress_marker": self.progress_marker,
            "segment_start_turn": self.segment_start_turn(),
            "messages": [m.to_dict() for m in self.messages],
            "settings": self.settings.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class CondenserSessionStore:
    """Process-local session store (v1)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, CondenserSession] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        original_log: str,
        settings: SessionSettings,
        initial_user_message: Optional[str] = None,
    ) -> CondenserSession:
        sid = str(uuid.uuid4())
        log = (original_log or "").strip()
        turns = parse_chatlog(log)
        session = CondenserSession(
            session_id=sid,
            original_log=log,
            settings=settings,
            parsed_turns=turns,
            total_turn_count=len(turns),
        )
        if initial_user_message and initial_user_message.strip():
            session.messages.append(
                SessionMessage(role="user", content=initial_user_message.strip())
            )
        async with self._lock:
            self._sessions[sid] = session
        return session

    async def get(self, session_id: str) -> Optional[CondenserSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def require(self, session_id: str) -> CondenserSession:
        session = await self.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    async def reset(self, session_id: str) -> CondenserSession:
        session = await self.require(session_id)
        async with self._lock:
            session.messages.clear()
            session.partial_condensed = ""
            session.last_condensed_turn_index = -1
            session.progress_marker = ""
            session.status = "active"
            session.touch()
        return session

    async def clear_streaming(self, session_id: str) -> CondenserSession:
        """Force-clear streaming lock (client abort, disconnect, or manual cancel)."""
        session = await self.require(session_id)
        async with self._lock:
            if session.status == "streaming":
                session.status = "active"
                session.touch()
        return session

    def append_user_message(self, session: CondenserSession, content: str) -> SessionMessage:
        msg = SessionMessage(role="user", content=(content or "").strip())
        session.messages.append(msg)
        session.touch()
        return msg

    def append_assistant_message(
        self, session: CondenserSession, content: str
    ) -> SessionMessage:
        text = (content or "").strip()
        if not text:
            session.status = "active"
            session.touch()
            return SessionMessage(role="assistant", content="")
        msg = SessionMessage(role="assistant", content=text)
        session.messages.append(msg)
        self._merge_assistant_output(session, text)
        session.status = "active"
        session.touch()
        return msg

    def _merge_assistant_output(self, session: CondenserSession, text: str) -> None:
        marker_index = parse_progress_marker(text)
        if marker_index is not None:
            session.last_condensed_turn_index = max(
                session.last_condensed_turn_index, marker_index
            )
            session.progress_marker = format_progress_marker(marker_index)

        prior = (session.partial_condensed or "").strip()
        if prior and session.last_condensed_turn_index >= 0:
            session.partial_condensed = prior + "\n\n" + text
        else:
            session.partial_condensed = text


# Module singleton used by routes
session_store = CondenserSessionStore()


def _augment_user_message(session: CondenserSession, content: str) -> str:
    base = (content or "").strip()
    if session.last_condensed_turn_index < 0:
        return base
    start = session.segment_start_turn()
    total = session.total_turn_count
    extra = (
        f"\n\n[Server: Already condensed through turn {session.last_condensed_turn_index} "
        f"of {total - 1 if total else 0}. Continue from turn {start}. "
        "Output ONLY new dense draft for later turns; end with progress marker.]"
    )
    if session.parsed_turns and start < len(session.parsed_turns):
        segment = segment_transcript_for_turns(
            session.parsed_turns, start_index=start
        )
        if segment:
            extra += f"\n\n[NEXT SEGMENT (turns {start}+ for reference — do not paste verbatim)]:\n{segment}"
    return base + extra


def build_agent_messages(session: CondenserSession) -> List[Dict[str, str]]:
    progress_block = ""
    if session.total_turn_count > 0 and session.last_condensed_turn_index >= 0:
        progress_block = build_progress_block(
            last_turn_index=session.last_condensed_turn_index,
            total_turns=session.total_turn_count,
            segment_start_turn=session.segment_start_turn(),
        )
    elif session.total_turn_count > 0:
        progress_block = (
            "SEQUENTIAL PROGRESS:\n"
            f"- Total turns in ORIGINAL_CHATLOG: {session.total_turn_count}\n"
            "- Already condensed through turn index: none (start at 0)\n"
            f"- This pass: condense from turn 0 (~{INITIAL_SEGMENT_MAX_TURNS} turns or a natural break)."
        )

    tail = condensed_tail_excerpt(session.partial_condensed)
    seg_start = session.segment_start_turn()
    seg_end = (
        min(seg_start + INITIAL_SEGMENT_MAX_TURNS - 1, session.total_turn_count - 1)
        if session.total_turn_count > 0
        else seg_start
    )
    rag_block = ""
    last_user_msg = ""
    for m in reversed(session.messages):
        if m.role == "user" and (m.content or "").strip():
            last_user_msg = m.content
            break
    if session.settings.use_rag and session.settings.rag_doc_ids:
        rag_block = query_rag_for_step(
            doc_ids=session.settings.rag_doc_ids,
            segment_start=seg_start,
            segment_end=seg_end,
            partial_condensed=session.partial_condensed,
            failsafe=is_continue_instruction(last_user_msg),
        )
    system = build_agent_session_system(
        original_chatlog=session.original_log,
        progress_block=progress_block,
        condensed_tail=tail,
        include_full_log_context=session.settings.include_full_log_context,
        rag_supplement=rag_block if session.settings.include_full_log_context else "",
    )
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    last_user_idx = -1
    for i, m in enumerate(session.messages):
        if m.role == "user" and (m.content or "").strip():
            last_user_idx = i
    for i, m in enumerate(session.messages):
        if m.role != "user" or not (m.content or "").strip():
            continue
        user_content = _augment_user_message(session, m.content)
        if (
            rag_block
            and not session.settings.include_full_log_context
            and i == last_user_idx
        ):
            user_content = f"{user_content.rstrip()}\n\n{rag_block}"
        msgs.append({"role": "user", "content": user_content})
    return msgs


def build_llm_messages(session: CondenserSession) -> List[Dict[str, str]]:
    """Alias for routes/tests."""
    return build_agent_messages(session)


def _messages_to_local_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"### User\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant\n{content}")
    parts.append("### Assistant\n")
    return "\n\n".join(parts)


def _sse_payload(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _append_stream_piece(collected: List[str], piece: str) -> str:
    """
    Append one streaming text fragment, tolerating providers that send cumulative
    text instead of pure deltas (avoids duplicated output in the UI).
    """
    if not piece:
        return ""
    current = "".join(collected)
    if current:
        if piece == current or current.endswith(piece):
            return ""
        if piece.startswith(current):
            piece = piece[len(current) :]
        elif current.endswith(piece):
            return ""
    if piece:
        collected.append(piece)
    return piece


async def stream_session_completion(
    *,
    model_manager: Any,
    session: CondenserSession,
) -> AsyncIterator[str]:
    """
    Stream condenser agent tokens as SSE lines:
    {type: token|done|error, ...}
    """
    from . import inference
    from .openai_compat import (
        forward_to_configured_endpoint_streaming,
        get_configured_endpoint,
        is_api_endpoint,
        prepare_endpoint_request,
    )

    collected: List[str] = []

    try:
        model_name = normalize_endpoint_model_id(session.settings.model_name)
        if not model_name:
            yield _sse_payload({"type": "error", "detail": "model_name is required"})
            return

        messages = build_agent_messages(session)
        request_data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": session.settings.max_output_tokens,
            "temperature": session.settings.temperature,
            "top_p": 0.9,
            "stream": True,
            "_skip_openai_message_pruning": True,
        }

        session.status = "streaming"
        if is_api_endpoint(model_name):
            endpoint_cfg = get_configured_endpoint(model_name)
            if not endpoint_cfg:
                yield _sse_payload(
                    {
                        "type": "error",
                        "detail": f"API endpoint not configured: {model_name}",
                    }
                )
                return
            endpoint_name = endpoint_cfg.get("name") or model_name
            _endpoint_config, url, prepared = prepare_endpoint_request(
                model_name, request_data
            )
            prepared["_max_stream_attempts"] = 1
            buffer = b""
            async for chunk_bytes in forward_to_configured_endpoint_streaming(
                _endpoint_config, url, prepared
            ):
                if isinstance(chunk_bytes, str):
                    buffer += chunk_bytes.encode("utf-8")
                else:
                    buffer += chunk_bytes
                while b"\n\n" in buffer:
                    message, buffer = buffer.split(b"\n\n", 1)
                    if not message.strip():
                        continue
                    try:
                        message_str = message.decode("utf-8", errors="ignore")
                        for line in message_str.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            json_str = line[6:].strip()
                            if json_str == "[DONE]":
                                continue
                            try:
                                chunk_data = json.loads(json_str)
                            except json.JSONDecodeError:
                                continue
                            err = chunk_data.get("error")
                            if isinstance(err, dict):
                                msg = err.get("message", str(err))
                                code = int(err.get("code", 502) or 502)
                                raise HTTPException(
                                    status_code=min(code, 599), detail=msg
                                )
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = (
                                    chunk_data["choices"][0].get("delta") or {}
                                )
                                piece = (
                                    delta.get("content")
                                    or delta.get("reasoning")
                                    or ""
                                )
                                if piece:
                                    out = _append_stream_piece(collected, piece)
                                    if out:
                                        yield _sse_payload(
                                            {"type": "token", "text": out}
                                        )
                    except HTTPException:
                        raise
        else:
            if not model_manager:
                yield _sse_payload(
                    {
                        "type": "error",
                        "detail": "Local model requested but model_manager unavailable",
                    }
                )
                return
            prompt = _messages_to_local_prompt(messages)
            async for raw in inference.generate_text_streaming(
                model_manager=model_manager,
                model_name=model_name,
                prompt=prompt,
                max_tokens=session.settings.max_output_tokens,
                temperature=session.settings.temperature,
                top_p=0.9,
            ):
                if not raw.startswith("data: "):
                    continue
                json_str = raw[6:].strip()
                if json_str == "[DONE]":
                    continue
                try:
                    chunk_data = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                if chunk_data.get("error"):
                    raise HTTPException(
                        status_code=500,
                        detail=str(chunk_data.get("error")),
                    )
                piece = (
                    chunk_data.get("text")
                    or (
                        (chunk_data.get("choices") or [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    or ""
                )
                if piece:
                    out = _append_stream_piece(collected, piece)
                    if out:
                        yield _sse_payload({"type": "token", "text": out})

        full_text = "".join(collected).strip()
        if full_text:
            session_store.append_assistant_message(session, full_text)
            yield _sse_payload(
                {
                    "type": "done",
                    "condensed_markdown": full_text,
                    "partial_condensed": session.partial_condensed,
                }
            )
        else:
            yield _sse_payload(
                {
                    "type": "done",
                    "condensed_markdown": "",
                    "partial_condensed": session.partial_condensed,
                    "interrupted": True,
                }
            )
    except HTTPException as e:
        detail = _format_condenser_api_error(
            e, endpoint_name=session.settings.model_name
        )
        yield _sse_payload({"type": "error", "detail": detail})
    except Exception as e:
        logger.exception("stream_session_completion failed")
        yield _sse_payload({"type": "error", "detail": str(e)})
    finally:
        session.status = "active"
