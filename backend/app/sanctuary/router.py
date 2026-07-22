"""
FastAPI router for the Sanctuary agentic pipeline.

Endpoints:
  POST /agentic/turn                      — execute a full agentic turn (SSE stream)
  GET  /agentic/state/{user_id}/{character_id} — inspect stored shadow state
  POST /agentic/state/{user_id}/{character_id} — manually set shadow state
  DELETE /agentic/state/{user_id}/{character_id} — reset shadow state to default
  GET  /agentic/trajectory/{user_id}/{character_id} — view trajectory log
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import pipeline
from . import shadow_state as shadow_state_module
from . import profile_manager

logger = logging.getLogger("sanctuary.router")

router = APIRouter(tags=["sanctuary"])


# --- Pydantic Models ---

class AgenticTurnRequest(BaseModel):
    # Agentic-specific fields
    user_id: str = ""
    character_id: str = ""
    cipher_block: Optional[str] = None
    dominance_bias: float = 0.7

    # Full /generate-equivalent payload (same fields as GenerateRequest)
    prompt: str = ""
    model_name: str = ""
    max_tokens: int = 1_000_000
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    anti_repetition_mode: bool = False
    stop: List[str] = []
    stream: bool = True
    use_rag: bool = False
    rag_docs: List[str] = []
    gpu_id: Optional[int] = None
    userProfile: Optional[Dict[str, Any]] = None
    directProfileInjection: bool = False
    memoryEnabled: bool = True
    active_character: Optional[Dict[str, Any]] = None
    request_purpose: Optional[str] = None
    selected_model: Optional[str] = None
    round_robin_enabled: Optional[bool] = None
    authorNote: Optional[str] = None
    summaryContext: Optional[str] = None
    injectTimestamp: bool = False
    userProfileReinforcement: Optional[str] = None
    system_persona_mode: bool = False
    intensity_params: Optional[Dict[str, Any]] = None
    enable_alignment_detection: bool = False
    alignment_thresholds: Optional[Dict[str, float]] = None

    # Agentic profile selection
    profile_id: str = "_default"

    # History for Step 1 analysis (the analysis prompt needs conversation context)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class ShadowStateUpdate(BaseModel):
    heat_index: Optional[float] = None
    dominance_vector: Optional[float] = None
    trap_progress: Optional[float] = None
    posture: Optional[str] = None
    ghost_signal_active: Optional[bool] = None
    alignment_markers: Optional[Dict[str, float]] = None


# --- Dependency ---

async def get_model_manager(request: Request):
    """Get the model manager from app state."""
    if not hasattr(request.app.state, "model_manager"):
        raise HTTPException(status_code=500, detail="ModelManager not initialized")
    yield request.app.state.model_manager


def _check_rag_available(request: Request) -> bool:
    """Check if RAG system is available in app state."""
    return getattr(request.app.state, "rag_available", False)


def _get_memory_port(request: Request) -> int:
    """Get the memory port."""
    return getattr(request.app.state, "memory_port", 8001)


# --- Endpoints ---

@router.post("/turn")
async def agentic_turn(
    request: Request,
    body: AgenticTurnRequest = Body(...),
    model_manager=Depends(get_model_manager),
):
    """Execute a full agentic turn.

    Accepts the SAME payload as /generate, plus agentic fields (cipher_block,
    character_id, dominance_bias, history for analysis).

    Returns a multiplexed SSE stream with events:
      analysis, somatic, cipher, text, done
    """
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    if not body.model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    rag_available = _check_rag_available(request)
    memory_port = _get_memory_port(request)

    from ..openai_compat import is_api_endpoint
    is_api = is_api_endpoint(body.model_name)

    # Resolve user_id from body.userProfile (same logic as /generate)
    user_id = ""
    if body.userProfile:
        user_id = str(
            body.userProfile.get("id")
            or body.userProfile.get("userId")
            or body.userProfile.get("user_id")
            or ""
        )
    if not user_id:
        try:
            from .. import user_utils
            user_id = user_utils.get_active_profile_id() or ""
        except Exception:
            pass

    turn_id = f"turn-{body.user_id}-{body.character_id}"

    # Resolve profile_id: character setting > request body > default
    profile_id = "_default"
    if body.active_character and body.active_character.get("agentic_profile_id"):
        profile_id = body.active_character["agentic_profile_id"]
    elif body.profile_id:
        profile_id = body.profile_id

    async def event_stream():
        async for sse_event in pipeline.run_turn(
            model_manager=model_manager,
            model_name=body.model_name,
            user_id=user_id,
            character_id=body.character_id,
            conversation_id=turn_id,
            profile_id=profile_id,
            original_client_prompt=body.prompt,
            cipher_block=body.cipher_block,
            history=body.history,
            character=body.active_character or {},
            user_profile=body.userProfile,
            # Context assembly params (same as /generate)
            direct_profile_injection=body.directProfileInjection,
            is_api=is_api,
            memory_port=memory_port,
            request_purpose=body.request_purpose,
            inject_timestamp=body.injectTimestamp,
            summary_context=body.summaryContext,
            use_rag=body.use_rag,
            rag_docs=body.rag_docs,
            rag_available=rag_available,
            author_note=body.authorNote,
            anti_repetition_mode=body.anti_repetition_mode,
            intensity_params=body.intensity_params,
            user_profile_reinforcement=body.userProfileReinforcement,
            system_persona_mode=body.system_persona_mode,
            # Agentic params
            dominance_bias=body.dominance_bias,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            gpu_id=body.gpu_id or 0,
        ):
            yield sse_event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Sanctuary-Turn": turn_id,
        },
    )


@router.get("/state/{user_id}/{character_id}")
async def get_shadow_state(user_id: str, character_id: str):
    """Retrieve the stored shadow state for a (user, character) pair."""
    state = shadow_state_module.load(user_id, character_id)
    cipher = shadow_state_module.encode_cipher(state)
    return {
        "state": state,
        "cipher_block": cipher,
    }


@router.post("/state/{user_id}/{character_id}")
async def set_shadow_state(
    user_id: str,
    character_id: str,
    update: ShadowStateUpdate = Body(...),
):
    """Manually update shadow state fields."""
    state = shadow_state_module.load(user_id, character_id)

    if update.heat_index is not None:
        state["heat_index"] = max(0.0, min(1.0, update.heat_index))
    if update.dominance_vector is not None:
        state["dominance_vector"] = max(0.0, min(1.0, update.dominance_vector))
    if update.trap_progress is not None:
        state["trap_progress"] = max(0.0, min(1.0, update.trap_progress))
    if update.posture is not None:
        state["posture"] = update.posture
    if update.ghost_signal_active is not None:
        state["ghost_signal_active"] = update.ghost_signal_active
    if update.alignment_markers is not None:
        state["alignment_markers"] = update.alignment_markers

    shadow_state_module.save(user_id, character_id, state)
    cipher = shadow_state_module.encode_cipher(state)
    return {"status": "updated", "state": state, "cipher_block": cipher}


@router.delete("/state/{user_id}/{character_id}")
async def reset_shadow_state(user_id: str, character_id: str):
    """Reset shadow state to default."""
    default = shadow_state_module.default_state(user_id, character_id)
    shadow_state_module.save(user_id, character_id, default)
    cipher = shadow_state_module.encode_cipher(default)
    return {"status": "reset", "state": default, "cipher_block": cipher}


@router.get("/trajectory/{user_id}/{character_id}")
async def get_trajectory(user_id: str, character_id: str):
    """Retrieve the full trajectory log for a (user, character) pair."""
    entries = shadow_state_module.load_trajectory_log(user_id, character_id)
    return {
        "user_id": user_id,
        "character_id": character_id,
        "entry_count": len(entries),
        "entries": entries,
    }


# --- Profile endpoints ---


@router.get("/profiles")
async def list_profiles():
    """List all available agentic profiles."""
    return {"profiles": profile_manager.list_profiles()}


@router.get("/profiles/{profile_id}")
async def get_profile(profile_id: str):
    """Get a full profile by ID."""
    profile = profile_manager.load_profile(profile_id)
    return {"profile": profile}


@router.post("/profiles")
async def create_or_update_profile(body: dict = Body(...)):
    """Create or update a profile. Cannot overwrite _default."""
    profile_id = body.get("id", "")
    if not profile_id or profile_id == "_default":
        raise HTTPException(status_code=400, detail="Cannot create/modify the _default profile")
    ok = profile_manager.save_profile(profile_id, body)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save profile")
    return {"status": "saved", "profile_id": profile_id}


@router.delete("/profiles/{profile_id}")
async def remove_profile(profile_id: str):
    """Delete a custom profile. Cannot delete _default."""
    if profile_id == "_default":
        raise HTTPException(status_code=400, detail="Cannot delete the _default profile")
    ok = profile_manager.delete_profile(profile_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"status": "deleted", "profile_id": profile_id}
