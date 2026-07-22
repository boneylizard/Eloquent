"""
Step 1 — Contextual Analysis.

Evaluates the current shadow state, history, and alignment markers to decide
the character's posture for this turn. Returns a structured JSON decision
that feeds into Step 2 and Step 3.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from . import dominance_rng
from . import prompts
from . import shadow_state
from .prompts import (
    LABEL_TACTILE_OUTREACH, LABEL_CHARACTER_SIGNAL,
    LABEL_POSE, LABEL_GESTURE, LABEL_PROXIMITY,
    LABEL_COVERT_ACTION, LABEL_VOICE_THIS_TURN,
)

logger = logging.getLogger("sanctuary.contextual_analysis")


async def run(
    model_manager,
    model_name: str,
    shadow_state_dict: Dict[str, Any],
    history: List[Dict[str, Any]],
    user_text: str,
    character: Dict[str, Any],
    intensity_params: Optional[Dict[str, Any]] = None,
    alignment_findings: Optional[Dict[str, Any]] = None,
    dominance_bias: float = 0.7,
    gpu_id: int = 0,
    full_context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute Step 1: Contextual Analysis.

    Returns a dict with keys:
      posture, dominance_vector, heat_index_delta, trap_progress_delta,
      ghost_signal_active, alignment_update, reasoning
    """
    prompt = prompts.build_contextual_analysis_prompt(
        shadow_state_dict, history, user_text, character,
        intensity_params, alignment_findings, full_context,
        profile=profile,
    )

    raw_output = await _call_llm(
        model_manager, model_name, prompt,
        max_tokens=700, temperature=0.4, gpu_id=gpu_id,
    )

    analysis = _parse_analysis_json(raw_output)
    if analysis is None:
        logger.warning("contextual_analysis: failed to parse JSON, using fallback")
        analysis = _fallback_analysis(shadow_state_dict)

    # Apply dominance stochasticity (70/30 bias)
    suggested = analysis.get("dominance_vector", shadow_state_dict.get("dominance_vector", 0.5))
    analysis["dominance_vector"] = dominance_rng.roll(suggested, bias=dominance_bias)

    logger.info(
        "contextual_analysis: posture=%s dominance=%.3f heat_delta=%+.3f ghost=%s",
        analysis.get("posture"),
        analysis.get("dominance_vector"),
        analysis.get("heat_index_delta", 0.0),
        analysis.get("ghost_signal_active", False),
    )

    return analysis


def _parse_analysis_json(raw: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON from the model output."""
    if not raw:
        return None

    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return _validate_analysis(result)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i + 1])
                    if isinstance(result, dict):
                        return _validate_analysis(result)
                except json.JSONDecodeError:
                    pass
                break

    return None


def _validate_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the analysis result, including tactile and signal sub-objects."""
    valid_postures = {"neutral", "assertive", "dominant", "yielding", "predatory", "protective"}
    posture = data.get("posture", "neutral")
    if posture not in valid_postures:
        posture = "neutral"

    # Validate tactile outreach sub-object
    raw_tactile = data.get(LABEL_TACTILE_OUTREACH) or {}
    tactile = {
        LABEL_POSE:      str(raw_tactile.get(LABEL_POSE, "")),
        LABEL_GESTURE:   str(raw_tactile.get(LABEL_GESTURE, "")),
        LABEL_PROXIMITY: str(raw_tactile.get(LABEL_PROXIMITY, "")),
    }

    # Validate character signal sub-object
    raw_signal = data.get(LABEL_CHARACTER_SIGNAL) or {}
    signal = {
        LABEL_COVERT_ACTION:  str(raw_signal.get(LABEL_COVERT_ACTION, "")),
        LABEL_VOICE_THIS_TURN: str(raw_signal.get(LABEL_VOICE_THIS_TURN, "")),
    }

    return {
        "posture": posture,
        "dominance_vector": _clamp(float(data.get("dominance_vector", 0.5))),
        "heat_index_delta": _clamp_delta(float(data.get("heat_index_delta", 0.0))),
        "trap_progress_delta": _clamp_delta(float(data.get("trap_progress_delta", 0.0))),
        "ghost_signal_active": bool(data.get("ghost_signal_active", False)),
        "alignment_update": data.get("alignment_update", {}),
        "reasoning": str(data.get("reasoning", "")),
        "emotional_state": str(data.get("emotional_state", "")),
        "physical_state": str(data.get("physical_state", "")),
        "trajectory": str(data.get("trajectory", "")),
        "internal_state": str(data.get("internal_state", "")),
        "external_state": str(data.get("external_state", "")),
        LABEL_TACTILE_OUTREACH: tactile,
        LABEL_CHARACTER_SIGNAL: signal,
    }


def _fallback_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a conservative default when JSON parsing fails."""
    return {
        "posture": state.get("posture", "neutral"),
        "dominance_vector": state.get("dominance_vector", 0.5),
        "heat_index_delta": 0.0,
        "trap_progress_delta": 0.0,
        "ghost_signal_active": False,
        "alignment_update": {},
        "reasoning": "Fallback: JSON parse failed, maintaining current state.",
        "emotional_state": "",
        "physical_state": "",
        "trajectory": "",
        "internal_state": "",
        "external_state": "",
        LABEL_TACTILE_OUTREACH: {LABEL_POSE: "", LABEL_GESTURE: "", LABEL_PROXIMITY: ""},
        LABEL_CHARACTER_SIGNAL: {LABEL_COVERT_ACTION: "", LABEL_VOICE_THIS_TURN: ""},
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_delta(value: float) -> float:
    return max(-0.3, min(0.3, value))


async def _call_llm(
    model_manager,
    model_name: str,
    prompt: str,
    max_tokens: int = 384,
    temperature: float = 0.4,
    gpu_id: int = 0,
) -> str:
    """Call the LLM, handling both local and API endpoints."""
    from .. import inference
    from ..openai_compat import is_api_endpoint, prepare_endpoint_request, collect_openai_compatible_stream_text

    if is_api_endpoint(model_name):
        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        try:
            endpoint_config, url, prepared = prepare_endpoint_request(model_name, request_data)
            return await collect_openai_compatible_stream_text(endpoint_config, url, prepared)
        except Exception as exc:
            logger.error("contextual_analysis: API call failed: %s", exc)
            return ""
    else:
        try:
            return await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                gpu_id=gpu_id,
            )
        except Exception as exc:
            logger.error("contextual_analysis: local call failed: %s", exc)
            return ""
