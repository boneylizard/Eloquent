"""
Pipeline orchestrator for the Sanctuary agentic turn.

Wires the three steps together and emits multiplexed SSE events:
  event: analysis  → Step 1 result (posture, dominance, deltas)
  event: somatic   → Step 2 result (dashboard, hijack, ghost signal)
  event: cipher    → encoded shadow state (for next-turn transport)
  event: text      → Step 3 streamed text deltas
  event: done      → turn metadata + final state

Event ordering guarantees (AGENTS.md §5 "Latency Sync"):
  somatic events are emitted BEFORE the first text delta.

The synthesis step uses the SAME context assembly as /generate (via
context_assembler.py) — it does NOT build its own prompt. The only thing
the agentic pipeline adds is the agentic directive block (posture,
dominance, heat, somatic state) appended to the fully assembled prompt.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from . import cipher as cipher_module
from . import contextual_analysis
from . import planning_generation
from . import profile_manager
from . import shadow_state as shadow_state_module
from . import somatic_generation
from . import synthesis
from . import context_assembler
from .prompts import (
    LABEL_TACTILE_OUTREACH, LABEL_CHARACTER_SIGNAL,
    LABEL_TACTILE_DISPLAY, LABEL_SIGNAL_DISPLAY,
    resolve_label,
)

logger = logging.getLogger("sanctuary.pipeline")


async def run_turn(
    model_manager,
    model_name: str,
    user_id: str,
    character_id: str,
    conversation_id: str,
    profile_id: str = "_default",
    original_client_prompt: str = "",
    history: List[Dict[str, Any]] = None,
    character: Dict[str, Any] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    cipher_block: Optional[str] = None,
    # Context assembly params (same as /generate)
    direct_profile_injection: bool = False,
    is_api: bool = False,
    memory_port: int = 8001,
    request_purpose: Optional[str] = None,
    inject_timestamp: bool = False,
    summary_context: Optional[str] = None,
    use_rag: bool = False,
    rag_docs: List[str] = None,
    rag_available: bool = False,
    author_note: Optional[str] = None,
    anti_repetition_mode: bool = False,
    intensity_params: Optional[Dict[str, Any]] = None,
    user_profile_reinforcement: Optional[str] = None,
    system_persona_mode: bool = False,
    # Agentic params
    dominance_bias: float = 0.7,
    max_tokens: int = 2048,
    temperature: float = 0.8,
    gpu_id: int = 0,
) -> AsyncGenerator[str, None]:
    """Execute a full agentic turn and yield SSE events.

    Yields strings in SSE format: ``event: <type>\\ndata: <json>\\n\\n``
    """
    if rag_docs is None:
        rag_docs = []

    if history is None:
        history = []

    turn_id = uuid.uuid4().hex[:12]
    t_start = time.time()
    logger.info("pipeline.run_turn: starting turn_id=%s model=%s profile=%s", turn_id, model_name, profile_id)

    # --- Load agentic profile ---
    profile = profile_manager.load_profile(profile_id)
    if profile:
        logger.info("pipeline.run_turn: using profile '%s' (%s)", profile_id, profile.get("name", "unknown"))
    else:
        logger.warning("pipeline.run_turn: profile '%s' not found, using defaults", profile_id)

    # Per-character dominance bias override
    effective_bias = dominance_bias
    if character:
        char_bias = character.get("sanctuary_bias")
        if char_bias is not None and isinstance(char_bias, (int, float)):
            effective_bias = max(0.0, min(1.0, float(char_bias)))
            logger.info("pipeline.run_turn: using per-character bias=%.2f", effective_bias)

    # --- Load shadow state ---
    state = shadow_state_module.load(user_id, character_id, cipher_block)
    logger.info(
        "pipeline.run_turn: loaded state turn_count=%d heat=%.3f dominance=%.3f posture=%s",
        state.get("turn_count", 0),
        state.get("heat_index", 0.0),
        state.get("dominance_vector", 0.5),
        state.get("posture", "neutral"),
    )

    # Extract previous planning memo from cipher block (if any)
    previous_plan_text = None
    if cipher_block:
        decoded = cipher_module.decode(cipher_block)
        if decoded and isinstance(decoded, dict):
            previous_plan_text = decoded.get("plan_text")
            if previous_plan_text:
                logger.info("pipeline.run_turn: found previous plan (%d chars)", len(previous_plan_text))

    # --- Assemble full context FIRST (needed for all steps) ---
    t_assembly_start = time.time()
    try:
        llm_prompt, character_persona, user_query = await context_assembler.assemble_context(
            original_client_prompt=original_client_prompt,
            direct_profile_injection=direct_profile_injection,
            is_api=is_api,
            user_id=user_id,
            user_profile=user_profile or {},
            active_character=character,
            memory_port=memory_port,
            request_purpose=request_purpose,
            inject_timestamp=inject_timestamp,
            summary_context=summary_context,
            use_rag=use_rag,
            rag_docs=rag_docs,
            rag_available=rag_available,
            author_note=author_note,
            anti_repetition_mode=anti_repetition_mode,
            intensity_params=intensity_params,
            user_profile_reinforcement=user_profile_reinforcement,
            system_persona_mode=system_persona_mode,
            previous_plan_text=previous_plan_text,
        )
    except Exception as exc:
        logger.error("pipeline.run_turn: context assembly failed: %s", exc, exc_info=True)
        yield _sse_event("error", {"step": "context_assembly", "detail": str(exc)})
        yield _sse_event("done", {"turn_id": turn_id, "error": True})
        return

    t_assembly_end = time.time()
    logger.info("pipeline.run_turn: context assembled in %.2fs (prompt %d chars)",
                t_assembly_end - t_assembly_start, len(llm_prompt))

    # --- Step 1: Contextual Analysis (with full context) ---
    t_step1_start = time.time()
    try:
        analysis = await contextual_analysis.run(
            model_manager=model_manager,
            model_name=model_name,
            shadow_state_dict=state,
            history=history,
            user_text=original_client_prompt,
            character=character,
            intensity_params=intensity_params,
            dominance_bias=effective_bias,
            gpu_id=gpu_id,
            full_context=llm_prompt,
            profile=profile,
        )
    except Exception as exc:
        logger.error("pipeline.run_turn: Step 1 failed: %s", exc, exc_info=True)
        yield _sse_event("error", {"step": "analysis", "detail": str(exc)})
        yield _sse_event("done", {"turn_id": turn_id, "error": True})
        return

    t_step1_end = time.time()
    logger.info("pipeline.run_turn: Step 1 complete in %.2fs", t_step1_end - t_step1_start)

    # Emit analysis event with prompt token info
    analysis_event = dict(analysis)
    analysis_event["prompt_tokens"] = len(llm_prompt)
    yield _sse_event("analysis", analysis_event)

    # Emit tactile outreach as its own SSE event (extracted from analysis for frontend routing)
    tactile_label_key = resolve_label("LABEL_TACTILE_OUTREACH", profile)
    tactile_display = resolve_label("LABEL_TACTILE_DISPLAY", profile)
    tactile_data = analysis.get(tactile_label_key) or {}
    tactile_data["_label_display"] = tactile_display
    yield _sse_event("tactile", tactile_data)

    # Emit character signal as its own SSE event
    signal_label_key = resolve_label("LABEL_CHARACTER_SIGNAL", profile)
    signal_display = resolve_label("LABEL_SIGNAL_DISPLAY", profile)
    signal_data = analysis.get(signal_label_key) or {}
    signal_data["_label_display"] = signal_display
    yield _sse_event("signal", signal_data)

    # Apply analysis to state
    state = shadow_state_module.apply_analysis_result(state, analysis)

    # --- Step 2: Somatic Payload Generation (with full context) ---
    t_step2_start = time.time()
    try:
        somatic = await somatic_generation.run(
            model_manager=model_manager,
            model_name=model_name,
            shadow_state_dict=state,
            analysis_result=analysis,
            character=character,
            gpu_id=gpu_id,
            full_context=llm_prompt,
            profile=profile,
        )
    except Exception as exc:
        logger.error("pipeline.run_turn: Step 2 failed: %s", exc, exc_info=True)
        yield _sse_event("error", {"step": "somatic", "detail": str(exc)})
        yield _sse_event("done", {"turn_id": turn_id, "error": True})
        return

    t_step2_end = time.time()
    logger.info("pipeline.run_turn: Step 2 complete in %.2fs", t_step2_end - t_step2_start)

    # Emit somatic event with prompt token info
    somatic_event = dict(somatic)
    somatic_event["prompt_tokens"] = len(llm_prompt)
    yield _sse_event("somatic", somatic_event)

    # --- Assemble full context (SAME as /generate) ---
    t_assembly_start = time.time()
    try:
        llm_prompt, character_persona, user_query = await context_assembler.assemble_context(
            original_client_prompt=original_client_prompt,
            direct_profile_injection=direct_profile_injection,
            is_api=is_api,
            user_id=user_id,
            user_profile=user_profile or {},
            active_character=character,
            memory_port=memory_port,
            request_purpose=request_purpose,
            inject_timestamp=inject_timestamp,
            summary_context=summary_context,
            use_rag=use_rag,
            rag_docs=rag_docs,
            rag_available=rag_available,
            author_note=author_note,
            anti_repetition_mode=anti_repetition_mode,
            intensity_params=intensity_params,
            user_profile_reinforcement=user_profile_reinforcement,
            system_persona_mode=system_persona_mode,
            previous_plan_text=previous_plan_text,
        )
    except Exception as exc:
        logger.error("pipeline.run_turn: context assembly failed: %s", exc, exc_info=True)
        yield _sse_event("error", {"step": "context_assembly", "detail": str(exc)})
        yield _sse_event("done", {"turn_id": turn_id, "error": True})
        return

    t_assembly_end = time.time()
    logger.info("pipeline.run_turn: context assembled in %.2fs (prompt %d chars)",
                t_assembly_end - t_assembly_start, len(llm_prompt))

    # --- Step 3: Natural Language Synthesis (streamed) ---
    ghost = somatic.get("ghost_signal", {"active": False, "charge": 0.0, "carrier_phrase": ""})

    # Build the agentic directive block (the ONLY thing we add to the assembled prompt)
    agentic_directive = synthesis.build_agentic_directive_block(
        state, analysis, somatic,
        ghost_signal_data=ghost if ghost.get("active") else None,
        profile=profile,
    )

    synthesis_prompt_tokens = len(llm_prompt) + len(agentic_directive)

    text_accumulator: List[str] = []
    t_first_text = None
    t_somatic_emit = time.time()

    try:
        async for delta in synthesis.run(
            model_manager=model_manager,
            model_name=model_name,
            llm_prompt=llm_prompt,
            agentic_directive_block=agentic_directive,
            max_tokens=max_tokens,
            temperature=temperature,
            gpu_id=gpu_id,
        ):
            if t_first_text is None:
                t_first_text = time.time()
                latency_sync = t_first_text - t_somatic_emit
                logger.info(
                    "pipeline.run_turn: LATENCY SYNC — first text delta %.3fs after somatic emit (%s)",
                    latency_sync,
                    "PASS" if latency_sync >= 0 else "FAIL",
                )
            text_accumulator.append(delta)
            yield _sse_event("text", {"delta": delta})
    except Exception as exc:
        logger.error("pipeline.run_turn: Step 3 failed: %s", exc, exc_info=True)
        yield _sse_event("error", {"step": "synthesis", "detail": str(exc)})

    # --- Post-stream: finalize ---
    full_text = "".join(text_accumulator)

    # --- Step 3b: Private Strategic Planning Generation ---
    if full_text.strip():
        t_plan_start = time.time()
        try:
            plan_text = await planning_generation.run(
                model_manager=model_manager,
                model_name=model_name,
                shadow_state=state,
                analysis_result=analysis,
                somatic_payload=somatic,
                this_turn_text=full_text,
                gpu_id=gpu_id,
                profile=profile,
            )
            if plan_text:
                state["plan_text"] = plan_text
                logger.info("pipeline.run_turn: plan generated (%d chars) in %.2fs",
                            len(plan_text), time.time() - t_plan_start)
        except Exception as exc:
            logger.error("pipeline.run_turn: planning generation failed: %s", exc, exc_info=True)

    t_end = time.time()
    total_time = t_end - t_start
    logger.info(
        "pipeline.run_turn: turn_id=%s complete in %.2fs (step1=%.2fs step2=%.2fs assembly=%.2fs synthesis=%.2fs text=%d chars)",
        turn_id, total_time,
        t_step1_end - t_step1_start,
        t_step2_end - t_step2_start,
        t_assembly_end - t_assembly_start,
        t_end - t_assembly_end,
        len(full_text),
    )

    # Save final state to disk (includes plan_text)
    save_ok = shadow_state_module.save(user_id, character_id, state)
    if not save_ok:
        logger.warning("pipeline.run_turn: failed to persist shadow state")

    final_cipher = shadow_state_module.encode_cipher(state)
    yield _sse_event("cipher", {"block": final_cipher, "phase": "final"})

    # Emit done event with latency metrics and token counts
    yield _sse_event("done", {
        "turn_id": turn_id,
        "conversation_id": conversation_id,
        "error": False,
        "final_state": {
            "turn_count": state.get("turn_count", 0),
            "heat_index": state.get("heat_index", 0.0),
            "dominance_vector": state.get("dominance_vector", 0.5),
            "trap_progress": state.get("trap_progress", 0.0),
            "posture": state.get("posture", "neutral"),
        },
        "text_length": len(full_text),
        "latency_ms": {
            "total": round(total_time * 1000, 1),
            "step1": round((t_step1_end - t_step1_start) * 1000, 1),
            "step2": round((t_step2_end - t_step2_start) * 1000, 1),
            "assembly": round((t_assembly_end - t_assembly_start) * 1000, 1),
            "somatic_to_first_text": round((t_first_text - t_somatic_emit) * 1000, 1) if t_first_text else None,
        },
        "token_counts": {
            "analysis_prompt_chars": analysis.get("prompt_tokens", 0),
            "somatic_prompt_chars": somatic.get("prompt_tokens", 0),
            "synthesis_prompt_chars": synthesis_prompt_tokens,
        },
    })


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format an SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
