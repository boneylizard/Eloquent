"""
FastAPI routes for chatlog condenser (Settings → Chatlog condenser).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import chatlog_condenser
from .chatlog_condenser_orchestrator import (
    OrchestratorSettings,
    orchestrator_store,
)
from .chatlog_condenser_session import (
    DEFAULT_AGENT_FIRST_USER_MESSAGE,
    SessionSettings,
    session_store,
    stream_session_completion,
)
from .model_manager import ModelManager

logger = logging.getLogger("chatlog_condenser_routes")

chatlog_condenser_router = APIRouter(tags=["chatlog_condenser"])


async def get_model_manager_from_state(request: Request):
    yield request.app.state.model_manager


class CondenseChatlogRequest(BaseModel):
    text: str = Field(..., description="Raw chatlog (.txt / .md content)")
    model_name: str = Field(..., description="endpoint-* API model or local GGUF name")
    target_ratio: float = Field(
        0.4,
        gt=0,
        description="Soft target; structural fidelity overrides ratio (no upper cap)",
    )
    chunk_target_tokens: int = Field(
        chatlog_condenser.DEFAULT_CHUNK_TARGET_TOKENS,
        ge=1,
        description="Per-chunk input budget (token estimate); tuned for long-context models",
    )
    overlap_turns: int = Field(
        5,
        ge=0,
        description="Turns repeated at chunk boundaries for continuity",
    )
    run_eval: bool = True
    eval_model_name: Optional[str] = None
    include_full_log_context: bool = Field(
        True,
        description=(
            "Include FULL_CHATLOG on every skeleton/render/stitch LLM call for global coherence; "
            "disable for faster/cheaper runs on very long logs"
        ),
    )
    use_rag: bool = Field(
        False,
        description="Supplement each chunk with RAG retrieval from rag_docs (cross-ref only)",
    )
    rag_docs: List[str] = Field(
        default_factory=list,
        description="Document ids from Document Context store",
    )


class ParseChatlogRequest(BaseModel):
    text: str
    chunk_target_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="If set, estimate how many processing chunks this log will split into",
    )
    overlap_turns: Optional[int] = Field(None, ge=0)
    run_eval: bool = Field(
        False,
        description="If set with chunk_target_tokens, include eval passes in llm_passes estimate",
    )


class EvalOnlyRequest(BaseModel):
    condensed_markdown: str
    skeleton: Dict[str, Any]
    model_name: str
    eval_model_name: Optional[str] = None


@chatlog_condenser_router.post("/parse")
async def parse_chatlog_endpoint(body: ParseChatlogRequest):
    turns = chatlog_condenser.parse_chatlog(body.text or "")
    md = chatlog_condenser.format_turns_markdown(turns)
    tokens_est = chatlog_condenser.estimate_tokens(md)
    out: Dict[str, Any] = {
        "status": "success",
        "turn_count": len(turns),
        "speakers": sorted({t.speaker for t in turns}),
        "tokens_est": tokens_est,
        "preview_markdown": md[:8000],
    }
    if body.chunk_target_tokens is not None:
        chunk_tokens = max(1, int(body.chunk_target_tokens))
        overlap = max(0, int(body.overlap_turns if body.overlap_turns is not None else 5))
        chunks = chatlog_condenser.chunk_turns_with_overlap(
            turns,
            chunk_target_tokens=chunk_tokens,
            overlap_turns=overlap,
        )
        chunk_count = len(chunks) or (1 if turns else 0)
        out["estimated_chunk_count"] = chunk_count
        out["estimated_llm_passes"] = chatlog_condenser.estimate_condenser_llm_passes(
            chunk_count,
            run_eval=body.run_eval,
        )
    return out


@chatlog_condenser_router.post("/condense")
async def condense_chatlog_endpoint(
    body: CondenseChatlogRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    if not (body.text or "").strip():
        raise HTTPException(status_code=400, detail="text is required")
    if not body.model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    try:
        result = await chatlog_condenser.condense_chatlog(
            model_manager=model_manager,
            model_name=body.model_name,
            text=body.text,
            target_ratio=body.target_ratio,
            chunk_target_tokens=body.chunk_target_tokens,
            overlap_turns=body.overlap_turns,
            run_eval=body.run_eval,
            eval_model_name=body.eval_model_name,
            include_full_log_context=body.include_full_log_context,
            use_rag=body.use_rag,
            rag_docs=body.rag_docs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("condense_chatlog failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "status": "success",
        "condensed_markdown": result.condensed_markdown,
        "skeleton": result.skeleton_full,
        "chunk_skeletons": result.chunk_skeletons,
        "stats": {
            "input_turns": result.stats.input_turns,
            "input_tokens_est": result.stats.input_tokens_est,
            "output_tokens_est": result.stats.output_tokens_est,
            "chunk_count": result.stats.chunk_count,
            "target_ratio": result.stats.target_ratio,
            "achieved_ratio": result.stats.achieved_ratio,
            "include_full_log_context": result.stats.include_full_log_context,
            "context_tokens_est": result.stats.context_tokens_est,
            "context_warning": result.stats.context_warning,
        },
        "eval": result.eval_result,
    }


class StartSessionRequest(BaseModel):
    text: str = Field(..., description="Raw chatlog")
    model_name: str
    target_ratio: float = Field(0.4, gt=0)
    include_full_log_context: bool = True
    max_output_tokens: int = Field(
        chatlog_condenser.DEFAULT_MAX_OUTPUT_TOKENS, ge=256
    )
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    initial_user_message: Optional[str] = Field(
        None,
        description="Optional first user instruction (default applied if omitted)",
    )
    use_rag: bool = False
    rag_docs: List[str] = Field(default_factory=list)


class SessionMessageRequest(BaseModel):
    message: str = Field(..., description="User message or continuation instruction")


@chatlog_condenser_router.post("/session/start")
async def start_condenser_session(body: StartSessionRequest):
    if not (body.text or "").strip():
        raise HTTPException(status_code=400, detail="text is required")
    if not body.model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    settings = SessionSettings(
        model_name=chatlog_condenser.normalize_endpoint_model_id(body.model_name),
        target_ratio=body.target_ratio,
        include_full_log_context=body.include_full_log_context,
        max_output_tokens=body.max_output_tokens,
        temperature=body.temperature,
        use_rag=body.use_rag,
        rag_doc_ids=list(body.rag_docs or []),
    )
    initial = (body.initial_user_message or "").strip() or DEFAULT_AGENT_FIRST_USER_MESSAGE
    session = await session_store.create(
        original_log=body.text,
        settings=settings,
        initial_user_message=initial,
    )
    return {"status": "success", "session": session.to_public_dict()}


@chatlog_condenser_router.get("/session/{session_id}")
async def get_condenser_session(session_id: str):
    session = await session_store.require(session_id)
    return {"status": "success", "session": session.to_public_dict()}


@chatlog_condenser_router.post("/session/{session_id}/reset")
async def reset_condenser_session(session_id: str):
    await session_store.clear_streaming(session_id)
    session = await session_store.reset(session_id)
    return {"status": "success", "session": session.to_public_dict()}


@chatlog_condenser_router.post("/session/{session_id}/cancel")
async def cancel_condenser_session_stream(session_id: str):
    """Unlock streaming only; does not clear original_log, messages, or partial_condensed."""
    session = await session_store.clear_streaming(session_id)
    return {"status": "success", "session": session.to_public_dict()}


@chatlog_condenser_router.post("/session/{session_id}/message")
async def condenser_session_message(
    session_id: str,
    body: SessionMessageRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    if not (body.message or "").strip():
        raise HTTPException(status_code=400, detail="message is required")
    session = await session_store.require(session_id)
    if session.status == "streaming":
        raise HTTPException(
            status_code=409, detail="Session is already streaming a response"
        )
    session_store.append_user_message(session, body.message)

    async def event_stream():
        try:
            async for line in stream_session_completion(
                model_manager=model_manager,
                session=session,
            ):
                yield line
        except asyncio.CancelledError:
            session.status = "active"
            raise
        finally:
            if session.status == "streaming":
                session.status = "active"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class StartOrchestratorRequest(BaseModel):
    text: str = Field(..., description="Raw chatlog")
    endpoint_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered API endpoint ids for failover (endpoint-*)",
    )
    chunk_turns: int = Field(20, ge=1, le=80)
    target_ratio: float = Field(0.4, gt=0)
    include_full_log_context: bool = True
    max_output_tokens: int = Field(
        chatlog_condenser.DEFAULT_MAX_OUTPUT_TOKENS, ge=256
    )
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    auto_run: bool = Field(
        True, description="Background loop; false = manual POST /tick only"
    )
    max_retries_per_step: int = Field(3, ge=1, le=12)
    alternate_apis_every_step: bool = Field(
        False,
        description="Rotate #1→#2→#1 each successful step (load share)",
    )
    api_routing_mode: str = Field(
        "failover_on_failure",
        description="failover_on_failure | rotate_every_step",
    )
    use_global_round_robin: Optional[bool] = Field(
        None,
        description="Use Settings round-robin among selected endpoints; default reads settings.json",
    )
    resume_run_id: Optional[str] = Field(
        None,
        description="Resume a stopped/paused run from ~/.LiangLocal/condenser_runs/{id}/",
    )
    use_rag: bool = Field(
        False,
        description="Per-step RAG supplement from uploaded transcript / document store",
    )
    rag_docs: List[str] = Field(
        default_factory=list,
        description="Document ids (same as chat Document Context)",
    )


class OrchestratorSupervisorRequest(BaseModel):
    message: str = Field(
        ...,
        description="Boss instruction applied on the next step only",
    )


@chatlog_condenser_router.post("/orchestrator/start")
async def start_orchestrator_run(
    body: StartOrchestratorRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    routing_mode = body.api_routing_mode or "failover_on_failure"
    if body.alternate_apis_every_step:
        routing_mode = "rotate_every_step"
    use_rr = body.use_global_round_robin
    if use_rr is None:
        from .chatlog_condenser_orchestrator import load_settings_round_robin_enabled

        use_rr = load_settings_round_robin_enabled()

    settings = OrchestratorSettings(
        target_ratio=body.target_ratio,
        include_full_log_context=body.include_full_log_context,
        max_output_tokens=body.max_output_tokens,
        temperature=body.temperature,
        chunk_turns=body.chunk_turns,
        auto_run=body.auto_run,
        max_retries_per_step=body.max_retries_per_step,
        api_routing_mode=routing_mode,
        alternate_apis_every_step=body.alternate_apis_every_step,
        use_global_round_robin=bool(use_rr),
        use_rag=body.use_rag,
        rag_doc_ids=list(body.rag_docs or []),
    )
    try:
        if body.resume_run_id:
            from .chatlog_condenser_orchestrator import load_run_from_disk

            run = load_run_from_disk(body.resume_run_id.strip())
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"No saved run at {body.resume_run_id}",
                )
            async with orchestrator_store._lock:
                orchestrator_store._runs[run.run_id] = run
            run = await orchestrator_store.resume(
                run.run_id, model_manager=model_manager
            )
        else:
            if not (body.text or "").strip():
                raise HTTPException(status_code=400, detail="text is required")
            run = await orchestrator_store.create(
                original_log=body.text,
                endpoint_ids=body.endpoint_ids,
                settings=settings,
            )
            run = await orchestrator_store.start_run(
                run, model_manager=model_manager
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"status": "success", "run": run.to_public_dict()}


@chatlog_condenser_router.get("/orchestrator/{run_id}")
async def get_orchestrator_run(run_id: str):
    run = await orchestrator_store.require(run_id)
    return {"status": "success", "run": run.to_public_dict()}


@chatlog_condenser_router.get("/orchestrator/{run_id}/stream")
async def stream_orchestrator_run(run_id: str):
    await orchestrator_store.require(run_id)

    async def event_stream():
        try:
            async for line in orchestrator_store.iter_sse_events(run_id):
                yield line
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@chatlog_condenser_router.post("/orchestrator/{run_id}/pause")
async def pause_orchestrator_run(run_id: str):
    run = await orchestrator_store.pause(run_id)
    return {"status": "success", "run": run.to_public_dict()}


@chatlog_condenser_router.post("/orchestrator/{run_id}/resume")
async def resume_orchestrator_run(
    run_id: str,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    run = await orchestrator_store.resume(run_id, model_manager=model_manager)
    return {"status": "success", "run": run.to_public_dict()}


@chatlog_condenser_router.post("/orchestrator/{run_id}/cancel")
async def cancel_orchestrator_run(run_id: str):
    run = await orchestrator_store.stop(run_id)
    return {"status": "success", "run": run.to_public_dict()}


@chatlog_condenser_router.post("/orchestrator/{run_id}/stop")
async def stop_orchestrator_run(run_id: str):
    run = await orchestrator_store.stop(run_id)
    return {"status": "success", "run": run.to_public_dict()}


@chatlog_condenser_router.post("/orchestrator/{run_id}/supervisor")
async def orchestrator_supervisor_message(
    run_id: str,
    body: OrchestratorSupervisorRequest,
):
    if not (body.message or "").strip():
        raise HTTPException(status_code=400, detail="message is required")
    run = await orchestrator_store.set_supervisor_instruction(
        run_id, body.message
    )
    return {"status": "success", "run": run.to_public_dict()}


@chatlog_condenser_router.post("/orchestrator/{run_id}/tick")
async def tick_orchestrator_run(
    run_id: str,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    run = await orchestrator_store.require(run_id)
    if run.status == "running" and run.settings.auto_run:
        raise HTTPException(
            status_code=409,
            detail="Run is auto-running; pause first or disable auto_run at start",
        )
    if run.status in ("completed", "failed", "stopped"):
        if run.status == "stopped":
            run.status = "running"
            run._cancel_requested = False
        elif run.status in ("completed", "failed"):
            return {"status": "success", "run": run.to_public_dict(), "advanced": False}
    run.status = "running"
    run._cancel_requested = False
    advanced = await orchestrator_store.execute_step(
        run_id, model_manager=model_manager
    )
    run = await orchestrator_store.require(run_id)
    if run.is_complete() and run.status != "failed":
        run.status = "completed"
        run.touch()
    elif not advanced and not run.is_complete() and run.status == "running":
        run.status = "paused"
    run = await orchestrator_store.require(run_id)
    return {
        "status": "success",
        "run": run.to_public_dict(),
        "advanced": advanced,
    }


@chatlog_condenser_router.post("/eval")
async def eval_condensed_endpoint(
    body: EvalOnlyRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    if not body.model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    try:
        eval_result = await chatlog_condenser.run_reconstruction_eval(
            model_manager=model_manager,
            model_name=body.eval_model_name or body.model_name,
            skeleton=body.skeleton,
            condensed_markdown=body.condensed_markdown,
        )
    except Exception as e:
        logger.exception("reconstruction eval failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"status": "success", "eval": eval_result}
