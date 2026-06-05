"""
Autonomous chatlog condenser orchestrator: sequential chunk condensing with API failover.

State machine per run_id; background asyncio loop or manual POST /tick steps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Literal, Optional

from fastapi import HTTPException

from .chatlog_condenser import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TARGET_RATIO,
    Turn,
    _format_condenser_api_error,
    call_llm,
    estimate_per_call_context_tokens,
    estimate_tokens,
    format_turns_markdown,
    normalize_endpoint_model_id,
    parse_chatlog,
)
from .chatlog_condenser_prompt import build_agent_session_system
from .chatlog_condenser_rag import query_rag_for_step
from .chatlog_condenser_session import (
    DEFAULT_AGENT_CONTINUE_USER_MESSAGE,
    DEFAULT_AGENT_FIRST_USER_MESSAGE,
    build_progress_block,
    condensed_tail_excerpt,
    format_progress_marker,
    parse_progress_marker,
    segment_transcript_for_turns,
)

__all__ = [
    "parse_progress_marker",
    "detect_no_advance",
    "orchestrator_store",
    "CONDENSER_RUNS_DIR",
    "load_settings_round_robin_enabled",
    "resolve_endpoint_display_name",
]

logger = logging.getLogger("chatlog_condenser_orchestrator")

CONDENSER_RUNS_DIR = Path.home() / ".LiangLocal" / "condenser_runs"

ApiRoutingMode = Literal["failover_on_failure", "rotate_every_step"]

WRANGLER_NUDGE = (
    "\n\n[WRANGLER — your last pass did not advance sequential progress. "
    "Do NOT repeat CONDENSED_SO_FAR. Output ONLY new dense draft for turns after the last "
    "condensed index. End with [CONDENSED THROUGH: turn index N] where N is strictly greater "
    "than the already-condensed index.]"
)

DEFAULT_CHUNK_TURNS = 20
DEFAULT_MAX_RETRIES_PER_STEP = 3
DEFAULT_AUTO_RUN = True

# Type: async (model_name, system, user, max_tokens, temperature) -> str
LlmStepRunner = Callable[..., Awaitable[str]]


@dataclass
class OrchestratorSettings:
    target_ratio: float = DEFAULT_TARGET_RATIO
    include_full_log_context: bool = True
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.2
    chunk_turns: int = DEFAULT_CHUNK_TURNS
    auto_run: bool = DEFAULT_AUTO_RUN
    max_retries_per_step: int = DEFAULT_MAX_RETRIES_PER_STEP
    api_routing_mode: ApiRoutingMode = "failover_on_failure"
    alternate_apis_every_step: bool = False
    use_global_round_robin: bool = False
    use_rag: bool = False
    rag_doc_ids: List[str] = field(default_factory=list)

    def effective_routing_mode(self) -> ApiRoutingMode:
        if self.alternate_apis_every_step:
            return "rotate_every_step"
        return self.api_routing_mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_ratio": self.target_ratio,
            "include_full_log_context": self.include_full_log_context,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "chunk_turns": self.chunk_turns,
            "auto_run": self.auto_run,
            "max_retries_per_step": self.max_retries_per_step,
            "api_routing_mode": self.api_routing_mode,
            "alternate_apis_every_step": self.alternate_apis_every_step,
            "use_global_round_robin": self.use_global_round_robin,
            "use_rag": self.use_rag,
            "rag_doc_ids": list(self.rag_doc_ids),
            "effective_routing_mode": self.effective_routing_mode(),
        }


@dataclass
class OrchestratorLogEntry:
    level: str  # info | warn | error
    message: str
    endpoint_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "endpoint_id": self.endpoint_id,
            "created_at": self.created_at,
        }


@dataclass
class OrchestratorRun:
    run_id: str
    original_log: str
    endpoint_ids: List[str]
    settings: OrchestratorSettings
    parsed_turns: List[Turn] = field(default_factory=list)
    total_turns: int = 0
    cursor_turn: int = -1
    current_endpoint_index: int = 0
    partial_condensed: str = ""
    status: str = "idle"  # idle | running | paused | completed | failed
    logs: List[OrchestratorLogEntry] = field(default_factory=list)
    last_step_output: str = ""
    wrangler_nudges: int = 0
    step_count: int = 0
    supervisor_instruction: str = ""
    last_step_tokens_est: int = 0
    last_resolved_endpoint_id: Optional[str] = None
    last_resolved_endpoint_name: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _cancel_requested: bool = False
    _step_in_progress: bool = False
    _subscribers: List[asyncio.Queue] = field(default_factory=list, repr=False)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def segment_start_turn(self) -> int:
        return self.cursor_turn + 1

    def is_complete(self) -> bool:
        if self.total_turns <= 0:
            return True
        return self.cursor_turn >= self.total_turns - 1

    def current_endpoint_id(self) -> Optional[str]:
        if not self.endpoint_ids:
            return None
        idx = min(self.current_endpoint_index, len(self.endpoint_ids) - 1)
        return self.endpoint_ids[idx]

    def current_endpoint_slot(self) -> int:
        """1-based slot (#1, #2) in the user's failover order."""
        if not self.endpoint_ids:
            return 0
        return min(self.current_endpoint_index, len(self.endpoint_ids) - 1) + 1

    def run_dir(self) -> Path:
        return CONDENSER_RUNS_DIR / self.run_id

    def to_public_dict(self) -> Dict[str, Any]:
        active_id = self.last_resolved_endpoint_id or self.current_endpoint_id()
        active_name = self.last_resolved_endpoint_name
        if not active_name and active_id:
            active_name = resolve_endpoint_display_name(active_id)
        return {
            "run_id": self.run_id,
            "status": self.status,
            "original_log_chars": len(self.original_log or ""),
            "total_turns": self.total_turns,
            "cursor_turn": self.cursor_turn,
            "segment_start_turn": self.segment_start_turn(),
            "endpoint_ids": list(self.endpoint_ids),
            "current_endpoint_index": self.current_endpoint_index,
            "current_endpoint_id": self.current_endpoint_id(),
            "current_endpoint_slot": self.current_endpoint_slot(),
            "active_endpoint_id": active_id,
            "active_endpoint_name": active_name,
            "partial_condensed": self.partial_condensed,
            "settings": self.settings.to_dict(),
            "logs": [e.to_dict() for e in self.logs],
            "wrangler_nudges": self.wrangler_nudges,
            "step_count": self.step_count,
            "supervisor_instruction": self.supervisor_instruction,
            "last_step_tokens_est": self.last_step_tokens_est,
            "is_complete": self.is_complete(),
            "checkpoint_dir": str(self.run_dir()),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _sse_payload(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def load_settings_round_robin_enabled() -> bool:
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    try:
        if not settings_path.exists():
            return False
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        return bool(settings.get("apiEndpointRoundRobinEnabled", False))
    except Exception:
        return False


def resolve_endpoint_display_name(endpoint_id: str) -> str:
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    try:
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            for ep in settings.get("customApiEndpoints", []):
                if ep.get("id") == endpoint_id:
                    return ep.get("name") or endpoint_id
    except Exception:
        pass
    return endpoint_id


def _run_state_path(run_id: str) -> Path:
    return CONDENSER_RUNS_DIR / run_id / "state.json"


def _serialize_run(run: OrchestratorRun) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "original_log": run.original_log,
        "endpoint_ids": run.endpoint_ids,
        "settings": run.settings.to_dict(),
        "total_turns": run.total_turns,
        "cursor_turn": run.cursor_turn,
        "current_endpoint_index": run.current_endpoint_index,
        "partial_condensed": run.partial_condensed,
        "status": run.status,
        "logs": [e.to_dict() for e in run.logs[-200:]],
        "wrangler_nudges": run.wrangler_nudges,
        "step_count": run.step_count,
        "supervisor_instruction": run.supervisor_instruction,
        "last_step_tokens_est": run.last_step_tokens_est,
        "last_resolved_endpoint_id": run.last_resolved_endpoint_id,
        "last_resolved_endpoint_name": run.last_resolved_endpoint_name,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _deserialize_run(data: Dict[str, Any]) -> OrchestratorRun:
    settings_raw = data.get("settings") or {}
    settings = OrchestratorSettings(
        target_ratio=float(settings_raw.get("target_ratio", DEFAULT_TARGET_RATIO)),
        include_full_log_context=bool(
            settings_raw.get("include_full_log_context", True)
        ),
        max_output_tokens=int(
            settings_raw.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        ),
        temperature=float(settings_raw.get("temperature", 0.2)),
        chunk_turns=int(settings_raw.get("chunk_turns", DEFAULT_CHUNK_TURNS)),
        auto_run=bool(settings_raw.get("auto_run", DEFAULT_AUTO_RUN)),
        max_retries_per_step=int(
            settings_raw.get("max_retries_per_step", DEFAULT_MAX_RETRIES_PER_STEP)
        ),
        api_routing_mode=settings_raw.get("api_routing_mode", "failover_on_failure"),
        alternate_apis_every_step=bool(
            settings_raw.get("alternate_apis_every_step", False)
        ),
        use_global_round_robin=bool(
            settings_raw.get("use_global_round_robin", False)
        ),
        use_rag=bool(settings_raw.get("use_rag", False)),
        rag_doc_ids=list(settings_raw.get("rag_doc_ids") or []),
    )
    log = (data.get("original_log") or "").strip()
    turns = parse_chatlog(log)
    run = OrchestratorRun(
        run_id=data["run_id"],
        original_log=log,
        endpoint_ids=list(data.get("endpoint_ids") or []),
        settings=settings,
        parsed_turns=turns,
        total_turns=int(data.get("total_turns", len(turns))),
        cursor_turn=int(data.get("cursor_turn", -1)),
        current_endpoint_index=int(data.get("current_endpoint_index", 0)),
        partial_condensed=data.get("partial_condensed") or "",
        status=data.get("status", "paused"),
        wrangler_nudges=int(data.get("wrangler_nudges", 0)),
        step_count=int(data.get("step_count", 0)),
        supervisor_instruction=data.get("supervisor_instruction") or "",
        last_step_tokens_est=int(data.get("last_step_tokens_est", 0)),
        last_resolved_endpoint_id=data.get("last_resolved_endpoint_id"),
        last_resolved_endpoint_name=data.get("last_resolved_endpoint_name"),
        created_at=data.get("created_at")
        or datetime.now(timezone.utc).isoformat(),
        updated_at=data.get("updated_at")
        or datetime.now(timezone.utc).isoformat(),
    )
    for entry in data.get("logs") or []:
        run.logs.append(
            OrchestratorLogEntry(
                level=entry.get("level", "info"),
                message=entry.get("message", ""),
                endpoint_id=entry.get("endpoint_id"),
                created_at=entry.get("created_at")
                or datetime.now(timezone.utc).isoformat(),
            )
        )
    return run


def persist_run_state(run: OrchestratorRun) -> None:
    run_dir = run.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = _run_state_path(run.run_id)
    tmp = state_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_serialize_run(run), f, ensure_ascii=False, indent=2)
    tmp.replace(state_path)


def write_step_checkpoint(run: OrchestratorRun) -> int:
    """Write checkpoint_N.json + draft_N.md after a successful step."""
    n = run.step_count
    run_dir = run.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": n,
        "cursor_turn": run.cursor_turn,
        "segment_start_turn": run.segment_start_turn(),
        "endpoint_id": run.last_resolved_endpoint_id,
        "endpoint_name": run.last_resolved_endpoint_name,
        "endpoint_slot": run.current_endpoint_slot(),
        "partial_condensed": run.partial_condensed,
        "last_step_tokens_est": run.last_step_tokens_est,
        "updated_at": run.updated_at,
    }
    ckpt_path = run_dir / f"checkpoint_{n}.json"
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    draft_path = run_dir / f"draft_{n}.md"
    draft_path.write_text(run.partial_condensed or "", encoding="utf-8")
    return n


def load_run_from_disk(run_id: str) -> Optional[OrchestratorRun]:
    state_path = _run_state_path(run_id)
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _deserialize_run(data)
    except Exception as e:
        logger.warning("Failed to load orchestrator run %s: %s", run_id, e)
        return None


def estimate_step_tokens(run: OrchestratorRun, messages: List[Dict[str, str]]) -> int:
    system = messages[0]["content"] if messages else ""
    user = messages[1]["content"] if len(messages) > 1 else ""
    segment_md = ""
    if run.parsed_turns:
        segment_md = segment_transcript_for_turns(
            run.parsed_turns,
            start_index=run.segment_start_turn(),
            max_turns=max(1, run.settings.chunk_turns),
        )
    return estimate_per_call_context_tokens(
        full_chatlog_md=format_turns_markdown(run.parsed_turns)
        if run.parsed_turns
        else run.original_log,
        segment_md=segment_md or user,
        prior_context=condensed_tail_excerpt(run.partial_condensed),
        include_full_log_context=run.settings.include_full_log_context,
    ) + estimate_tokens(system) + estimate_tokens(user)


def build_segment_user_message(
    run: OrchestratorRun,
    *,
    wrangler: bool = False,
) -> str:
    start = run.segment_start_turn()
    chunk = max(1, run.settings.chunk_turns)
    if run.cursor_turn < 0:
        base = DEFAULT_AGENT_FIRST_USER_MESSAGE.replace(
            "~15–25", f"~{chunk}"
        ).replace("15–25", str(chunk))
    else:
        base = DEFAULT_AGENT_CONTINUE_USER_MESSAGE
    extra = (
        f"\n\n[Orchestrator: condense turns {start} through at most "
        f"{min(start + chunk - 1, run.total_turns - 1)} only "
        f"(total turns: {run.total_turns}, 0-based indices).]"
    )
    if run.parsed_turns and start < len(run.parsed_turns):
        segment = segment_transcript_for_turns(
            run.parsed_turns,
            start_index=start,
            max_turns=chunk,
        )
        if segment:
            extra += (
                f"\n\n[NEXT SEGMENT (turns {start}+ for reference — do not paste verbatim)]:\n"
                f"{segment}"
            )
    if wrangler:
        extra += WRANGLER_NUDGE
    sup = (run.supervisor_instruction or "").strip()
    if sup:
        extra += f"\n\n[SUPERVISOR — apply on this step only; do not discard prior condensed work]:\n{sup}"
    return base + extra


def _orchestrator_segment_end(run: OrchestratorRun) -> int:
    start = run.segment_start_turn()
    chunk = max(1, run.settings.chunk_turns)
    if run.total_turns <= 0:
        return start
    return min(start + chunk - 1, run.total_turns - 1)


def build_orchestrator_messages(run: OrchestratorRun, *, wrangler: bool = False) -> List[Dict[str, str]]:
    progress_block = ""
    if run.total_turns > 0 and run.cursor_turn >= 0:
        progress_block = build_progress_block(
            last_turn_index=run.cursor_turn,
            total_turns=run.total_turns,
            segment_start_turn=run.segment_start_turn(),
        )
    elif run.total_turns > 0:
        chunk = max(1, run.settings.chunk_turns)
        progress_block = (
            "SEQUENTIAL PROGRESS:\n"
            f"- Total turns in ORIGINAL_CHATLOG: {run.total_turns}\n"
            "- Already condensed through turn index: none (start at 0)\n"
            f"- This pass: condense from turn 0 (~{chunk} turns or a natural break)."
        )

    tail = condensed_tail_excerpt(run.partial_condensed)
    seg_start = run.segment_start_turn()
    seg_end = _orchestrator_segment_end(run)
    rag_block = ""
    if run.settings.use_rag and run.settings.rag_doc_ids:
        rag_block = query_rag_for_step(
            doc_ids=run.settings.rag_doc_ids,
            segment_start=seg_start,
            segment_end=seg_end,
            partial_condensed=run.partial_condensed,
            failsafe=wrangler,
        )
    system = build_agent_session_system(
        original_chatlog=run.original_log,
        progress_block=progress_block,
        condensed_tail=tail,
        include_full_log_context=run.settings.include_full_log_context,
        rag_supplement=rag_block if run.settings.include_full_log_context else "",
    )
    user = build_segment_user_message(run, wrangler=wrangler)
    if rag_block and not run.settings.include_full_log_context:
        user = f"{user.rstrip()}\n\n{rag_block}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def merge_step_output(run: OrchestratorRun, text: str) -> int:
    """Merge assistant segment into partial_condensed; return new cursor (or -1 if no marker)."""
    marker_index = parse_progress_marker(text)
    prior = (run.partial_condensed or "").strip()
    if prior and marker_index is not None:
        run.partial_condensed = prior + "\n\n" + text
    elif prior and marker_index is None:
        run.partial_condensed = prior + "\n\n" + text
    else:
        run.partial_condensed = text
    if marker_index is not None:
        run.cursor_turn = max(run.cursor_turn, marker_index)
    return marker_index if marker_index is not None else -1


def detect_no_advance(run: OrchestratorRun, marker_index: Optional[int]) -> bool:
    if marker_index is None:
        return True
    return marker_index <= run.cursor_turn


class OrchestratorStore:
    """Process-local orchestrator runs (v1)."""

    def __init__(self) -> None:
        self._runs: Dict[str, OrchestratorRun] = {}
        self._lock = asyncio.Lock()
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._llm_runner: Optional[LlmStepRunner] = None

    def set_llm_runner(self, runner: Optional[LlmStepRunner]) -> None:
        self._llm_runner = runner

    def _log(
        self,
        run: OrchestratorRun,
        message: str,
        *,
        level: str = "info",
        endpoint_id: Optional[str] = None,
    ) -> None:
        run.logs.append(
            OrchestratorLogEntry(
                level=level, message=message, endpoint_id=endpoint_id
            )
        )
        if len(run.logs) > 500:
            run.logs = run.logs[-400:]
        run.touch()

    async def _broadcast(self, run: OrchestratorRun, event: Dict[str, Any]) -> None:
        line = _sse_payload(event)
        dead: List[asyncio.Queue] = []
        for q in run._subscribers:
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            if q in run._subscribers:
                run._subscribers.remove(q)

    async def set_supervisor_instruction(
        self, run_id: str, instruction: str
    ) -> OrchestratorRun:
        run = await self.require(run_id)
        run.supervisor_instruction = (instruction or "").strip()
        self._log(run, f"Supervisor note queued ({len(run.supervisor_instruction)} chars)")
        run.touch()
        persist_run_state(run)
        await self._broadcast(
            run,
            {
                "type": "supervisor",
                "instruction": run.supervisor_instruction,
            },
        )
        return run

    async def create(
        self,
        *,
        original_log: str,
        endpoint_ids: List[str],
        settings: OrchestratorSettings,
    ) -> OrchestratorRun:
        if not endpoint_ids:
            raise ValueError("endpoint_ids must not be empty")
        normalized = [
            normalize_endpoint_model_id(eid)
            for eid in endpoint_ids
            if (eid or "").strip()
        ]
        if not normalized:
            raise ValueError("endpoint_ids must not be empty")

        rid = str(uuid.uuid4())
        log = (original_log or "").strip()
        turns = parse_chatlog(log)
        run = OrchestratorRun(
            run_id=rid,
            original_log=log,
            endpoint_ids=normalized,
            settings=settings,
            parsed_turns=turns,
            total_turns=len(turns),
            status="idle",
        )
        mode = settings.effective_routing_mode()
        self._log(
            run,
            f"Parsed {run.total_turns} turns; endpoints: {', '.join(normalized)}; "
            f"routing={mode}",
        )
        if settings.use_rag and settings.rag_doc_ids:
            self._log(
                run,
                f"RAG supplement on ({len(settings.rag_doc_ids)} document(s))",
            )
        elif settings.use_rag:
            self._log(
                run,
                "RAG enabled but no rag_doc_ids — supplement skipped until docs selected",
                level="warn",
            )
        async with self._lock:
            self._runs[rid] = run
        persist_run_state(run)
        return run

    async def get(self, run_id: str) -> Optional[OrchestratorRun]:
        async with self._lock:
            run = self._runs.get(run_id)
        if run:
            return run
        disk = load_run_from_disk(run_id)
        if disk:
            async with self._lock:
                self._runs[run_id] = disk
            return disk
        return None

    async def require(self, run_id: str) -> OrchestratorRun:
        run = await self.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Orchestrator run not found")
        return run

    def subscribe(self, run: OrchestratorRun) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        run._subscribers.append(q)
        return q

    def unsubscribe(self, run: OrchestratorRun, q: asyncio.Queue) -> None:
        if q in run._subscribers:
            run._subscribers.remove(q)

    async def start_run(
        self,
        run: OrchestratorRun,
        *,
        model_manager: Any = None,
    ) -> OrchestratorRun:
        if run.status in ("running", "completed"):
            return run
        run._cancel_requested = False
        run.status = "running"
        run.touch()
        self._log(run, "Run started")
        if run.settings.auto_run:
            await self._ensure_background_loop(run.run_id, model_manager=model_manager)
        return run

    async def pause(self, run_id: str) -> OrchestratorRun:
        run = await self.require(run_id)
        if run.status == "running":
            run.status = "paused"
            run._cancel_requested = True
            self._log(run, "Paused by user")
            task = self._background_tasks.get(run_id)
            if task and not task.done():
                task.cancel()
            persist_run_state(run)
        return run

    async def resume(
        self, run_id: str, *, model_manager: Any = None
    ) -> OrchestratorRun:
        run = await self.require(run_id)
        if run.status not in ("paused", "idle", "stopped"):
            return run
        if run.is_complete():
            run.status = "completed"
            return run
        run._cancel_requested = False
        run.status = "running"
        self._log(run, "Resumed")
        persist_run_state(run)
        if run.settings.auto_run:
            await self._ensure_background_loop(run_id, model_manager=model_manager)
        return run

    async def stop(self, run_id: str) -> OrchestratorRun:
        """Stop without losing checkpoints or partial draft."""
        run = await self.require(run_id)
        run._cancel_requested = True
        task = self._background_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        if run.status not in ("completed",):
            run.status = "stopped"
        self._log(run, "Stopped — checkpoints preserved", level="warn")
        run.touch()
        persist_run_state(run)
        await self._broadcast(
            run,
            {
                "type": "stopped",
                "cursor_turn": run.cursor_turn,
                "partial_condensed": run.partial_condensed,
            },
        )
        return run

    async def cancel(self, run_id: str) -> OrchestratorRun:
        """Alias for stop (preserves progress on disk)."""
        return await self.stop(run_id)

    async def _ensure_background_loop(
        self, run_id: str, *, model_manager: Any = None
    ) -> None:
        existing = self._background_tasks.get(run_id)
        if existing and not existing.done():
            return

        async def _loop() -> None:
            try:
                while True:
                    run = await self.require(run_id)
                    if run._cancel_requested or run.status == "paused":
                        break
                    if run.is_complete():
                        run.status = "completed"
                        run.touch()
                        persist_run_state(run)
                        await self._broadcast(
                            run,
                            {
                                "type": "completed",
                                "cursor_turn": run.cursor_turn,
                                "partial_condensed": run.partial_condensed,
                            },
                        )
                        self._log(run, "All turns condensed — completed")
                        break
                    advanced = await self.execute_step(
                        run_id, model_manager=model_manager
                    )
                    if not advanced and run.status == "failed":
                        break
                    if run.is_complete():
                        run.status = "completed"
                        run.touch()
                        persist_run_state(run)
                        await self._broadcast(
                            run,
                            {
                                "type": "completed",
                                "cursor_turn": run.cursor_turn,
                                "partial_condensed": run.partial_condensed,
                            },
                        )
                        self._log(run, "All turns condensed — completed")
                        break
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                run = await self.get(run_id)
                if run and run.status == "running":
                    run.status = "paused"
                    run.touch()
            finally:
                self._background_tasks.pop(run_id, None)

        self._background_tasks[run_id] = asyncio.create_task(_loop())

    async def _call_llm(
        self,
        *,
        model_manager: Any,
        model_name: str,
        messages: List[Dict[str, str]],
        settings: OrchestratorSettings,
        run: OrchestratorRun,
    ) -> str:
        rotation_ids = None
        cursor_key = None
        if (
            settings.use_global_round_robin
            and settings.effective_routing_mode() != "rotate_every_step"
            and len(run.endpoint_ids) >= 2
        ):
            rotation_ids = list(run.endpoint_ids)
            cursor_key = f"__orch_{run.run_id}__"
        if self._llm_runner is not None:
            system = messages[0]["content"] if messages else ""
            user = messages[1]["content"] if len(messages) > 1 else ""
            return await self._llm_runner(
                model_name=model_name,
                system=system,
                user=user,
                max_tokens=settings.max_output_tokens,
                temperature=settings.temperature,
            )
        system = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else ""
        return await call_llm(
            model_manager=model_manager,
            model_name=model_name,
            system=system,
            user=user,
            max_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            rotation_candidate_ids=rotation_ids,
            rotation_cursor_key=cursor_key,
        )

    def _rotate_load_share(self, run: OrchestratorRun) -> None:
        """Advance to next endpoint after a successful step (load share)."""
        if len(run.endpoint_ids) <= 1:
            return
        run.current_endpoint_index = (run.current_endpoint_index + 1) % len(
            run.endpoint_ids
        )

    def _advance_failover(self, run: OrchestratorRun) -> bool:
        """Move to next endpoint in failover list. Returns False if exhausted."""
        if len(run.endpoint_ids) <= 1:
            return False
        next_idx = run.current_endpoint_index + 1
        if next_idx >= len(run.endpoint_ids):
            return False
        run.current_endpoint_index = next_idx
        return True

    def _record_resolved_endpoint(self, run: OrchestratorRun, endpoint_id: str) -> None:
        run.last_resolved_endpoint_id = endpoint_id
        run.last_resolved_endpoint_name = resolve_endpoint_display_name(endpoint_id)

    async def execute_step(
        self,
        run_id: str,
        *,
        model_manager: Any = None,
    ) -> bool:
        """
        Run one condense step. Returns True if cursor advanced or run completed.
        """
        run = await self.require(run_id)
        if run._step_in_progress:
            return False
        if run.is_complete():
            if run.status != "completed":
                run.status = "completed"
                run.touch()
            return True
        if run.status not in ("running", "idle"):
            return False

        run._step_in_progress = True
        run.status = "running"
        prior_cursor = run.cursor_turn
        max_attempts = max(1, run.settings.max_retries_per_step)
        wrangler = False
        step_ok = False
        routing = run.settings.effective_routing_mode()

        try:
            endpoint_id = run.current_endpoint_id()
            if endpoint_id:
                self._record_resolved_endpoint(run, endpoint_id)
            run.last_step_tokens_est = 0
            await self._broadcast(
                run,
                {
                    "type": "step_start",
                    "cursor_turn": run.cursor_turn,
                    "segment_start": run.segment_start_turn(),
                    "endpoint_id": endpoint_id,
                    "endpoint_slot": run.current_endpoint_slot(),
                    "endpoint_name": run.last_resolved_endpoint_name,
                    "routing_mode": routing,
                },
            )

            for attempt in range(max_attempts):
                if run._cancel_requested:
                    break
                endpoint_id = run.current_endpoint_id()
                if not endpoint_id:
                    break

                messages = build_orchestrator_messages(run, wrangler=wrangler)
                run.last_step_tokens_est = estimate_step_tokens(run, messages)
                self._log(
                    run,
                    f"Step attempt {attempt + 1}/{max_attempts} "
                    f"(turns from {run.segment_start_turn()}, "
                    f"API #{run.current_endpoint_slot()} {endpoint_id}, "
                    f"~{run.last_step_tokens_est:,} tok est)",
                    endpoint_id=endpoint_id,
                )
                await self._broadcast(
                    run,
                    {
                        "type": "step_timeline",
                        "phase": "llm_call",
                        "attempt": attempt + 1,
                        "endpoint_id": endpoint_id,
                        "endpoint_slot": run.current_endpoint_slot(),
                        "endpoint_name": run.last_resolved_endpoint_name,
                        "tokens_est": run.last_step_tokens_est,
                    },
                )

                try:
                    text = await self._call_llm(
                        model_manager=model_manager,
                        model_name=endpoint_id,
                        messages=messages,
                        settings=run.settings,
                        run=run,
                    )
                except HTTPException as e:
                    detail = _format_condenser_api_error(e, endpoint_name=endpoint_id)
                    self._log(run, f"API error: {detail}", level="error", endpoint_id=endpoint_id)
                    await self._broadcast(
                        run,
                        {"type": "error", "detail": detail, "endpoint_id": endpoint_id},
                    )
                    if self._advance_failover(run):
                        await self._broadcast(
                            run,
                            {
                                "type": "failover",
                                "from_endpoint_id": endpoint_id,
                                "to_endpoint_id": run.current_endpoint_id(),
                                "reason": detail,
                            },
                        )
                        self._log(
                            run,
                            f"Failover → {run.current_endpoint_id()}",
                            level="warn",
                        )
                        continue
                    run.status = "failed"
                    run.touch()
                    return False
                except Exception as e:
                    detail = str(e)
                    self._log(run, f"Step failed: {detail}", level="error")
                    await self._broadcast(run, {"type": "error", "detail": detail})
                    if self._advance_failover(run):
                        await self._broadcast(
                            run,
                            {
                                "type": "failover",
                                "from_endpoint_id": endpoint_id,
                                "to_endpoint_id": run.current_endpoint_id(),
                                "reason": detail,
                            },
                        )
                        continue
                    run.status = "failed"
                    run.touch()
                    return False

                run.last_step_output = (text or "").strip()
                if run.last_step_output:
                    await self._broadcast(
                        run,
                        {
                            "type": "token",
                            "text": run.last_step_output,
                            "aggregated": True,
                        },
                    )

                old_cursor_before_merge = run.cursor_turn
                marker_idx = parse_progress_marker(run.last_step_output)
                no_advance = marker_idx is None or marker_idx <= old_cursor_before_merge

                if no_advance:
                    self._log(
                        run,
                        "No progress marker advance — wrangler nudge or failover",
                        level="warn",
                        endpoint_id=endpoint_id,
                    )
                    if not wrangler and run.wrangler_nudges < 2:
                        wrangler = True
                        run.wrangler_nudges += 1
                        continue
                    if self._advance_failover(run):
                        await self._broadcast(
                            run,
                            {
                                "type": "failover",
                                "from_endpoint_id": endpoint_id,
                                "to_endpoint_id": run.current_endpoint_id(),
                                "reason": "no_progress_advance",
                            },
                        )
                        wrangler = False
                        continue
                    run.status = "failed"
                    self._log(run, "Stuck: no progress after retries", level="error")
                    return False

                merge_step_output(run, run.last_step_output)
                step_ok = True
                run.step_count += 1
                consumed_supervisor = run.supervisor_instruction
                run.supervisor_instruction = ""
                write_step_checkpoint(run)
                persist_run_state(run)
                if routing == "rotate_every_step":
                    self._rotate_load_share(run)
                    self._log(
                        run,
                        f"Load-share rotate → API #{run.current_endpoint_slot()} "
                        f"{run.current_endpoint_id()}",
                    )
                self._log(
                    run,
                    f"Step {run.step_count} done — {format_progress_marker(run.cursor_turn)}",
                    endpoint_id=endpoint_id,
                )
                await self._broadcast(
                    run,
                    {
                        "type": "step_done",
                        "step": run.step_count,
                        "cursor_turn": run.cursor_turn,
                        "marker": format_progress_marker(run.cursor_turn),
                        "partial_condensed": run.partial_condensed,
                        "endpoint_id": endpoint_id,
                        "endpoint_slot": run.current_endpoint_slot(),
                        "endpoint_name": run.last_resolved_endpoint_name,
                        "tokens_est": run.last_step_tokens_est,
                        "checkpoint": f"checkpoint_{run.step_count}.json",
                        "supervisor_applied": bool(consumed_supervisor),
                        "next_endpoint_id": run.current_endpoint_id(),
                    },
                )
                break

            if step_ok and run.cursor_turn > prior_cursor:
                return True
            if step_ok and run.is_complete():
                persist_run_state(run)
                return True
            return step_ok
        finally:
            run._step_in_progress = False
            run.touch()
            if run.status in ("running", "paused", "stopped"):
                persist_run_state(run)

    async def iter_sse_events(
        self, run_id: str, *, timeout_s: float = 3600.0
    ) -> AsyncIterator[str]:
        run = await self.require(run_id)
        q = self.subscribe(run)
        try:
            yield _sse_payload(
                {
                    "type": "status",
                    "run": run.to_public_dict(),
                }
            )
            deadline = asyncio.get_event_loop().time() + timeout_s
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    line = await asyncio.wait_for(q.get(), timeout=min(30.0, remaining))
                    yield line
                    try:
                        payload = json.loads(line[6:].strip().split("\n")[0])
                    except (json.JSONDecodeError, IndexError):
                        payload = {}
                    if payload.get("type") in ("completed", "error") and run.status in (
                        "completed",
                        "failed",
                    ):
                        if payload.get("type") == "completed":
                            break
                except asyncio.TimeoutError:
                    run = await self.require(run_id)
                    if run.status in ("completed", "failed", "paused", "stopped"):
                        yield _sse_payload({"type": "status", "run": run.to_public_dict()})
                        if run.status in ("completed", "failed", "stopped"):
                            break
        finally:
            self.unsubscribe(run, q)


orchestrator_store = OrchestratorStore()
