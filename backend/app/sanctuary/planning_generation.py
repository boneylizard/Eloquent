"""
Step 3b — Private Strategic Planning Generation.

After the AI writes its response, this step asks the AI to compose a short
strategic planning memo. The memo is encoded into the cipher block and
injected into the prompt at the start of the next turn so the AI can
read its own previous strategy.

This is a fast, low-token non-streaming LLM call using the same model.
"""

import logging
from typing import Any, Dict, Optional

from .prompts import PLANNING_GENERATION_PROMPT, PLANNING_INJECTION_TEMPLATE

logger = logging.getLogger("sanctuary.planning_generation")

_MAX_PLAN_TOKENS = 256
_PLAN_TEMPERATURE = 0.6


async def run(
    model_manager,
    model_name: str,
    shadow_state: Dict[str, Any],
    analysis_result: Dict[str, Any],
    somatic_payload: Dict[str, Any],
    this_turn_text: str,
    gpu_id: int = 0,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a private strategic planning memo for the next turn.

    Args:
        model_manager: The model manager instance.
        model_name: The model to use.
        shadow_state: Current shadow state dict.
        analysis_result: Step 1 analysis result.
        somatic_payload: Step 2 somatic payload.
        this_turn_text: The full text the AI wrote this turn.
        gpu_id: GPU ID for local models.
        profile: Optional agentic profile for custom prompt templates.

    Returns:
        The planning memo text string, or empty string on failure.
    """
    # Use profile-based planning prompt if available
    planning_prompt = PLANNING_GENERATION_PROMPT
    if profile:
        prompts_dict = profile.get("prompts", {})
        if isinstance(prompts_dict, dict) and prompts_dict.get("planning_generation"):
            planning_prompt = prompts_dict["planning_generation"]

    prompt = planning_prompt.format(
        posture=analysis_result.get("posture", "neutral"),
        dominance=analysis_result.get("dominance_vector", 0.5),
        heat=shadow_state.get("heat_index", 0.0),
        trap=shadow_state.get("trap_progress", 0.0),
        emotional_state=analysis_result.get("emotional_state", "neutral"),
        physical_state=analysis_result.get("physical_state", "standing"),
        trajectory=analysis_result.get("trajectory", "uncertain"),
        this_turn_text=(this_turn_text or "")[:200],
    )

    logger.info("planning_generation: generating planning memo")

    raw = await _call_llm(
        model_manager, model_name, prompt,
        max_tokens=_MAX_PLAN_TOKENS,
        temperature=_PLAN_TEMPERATURE,
        gpu_id=gpu_id,
    )

    text = (raw or "").strip()
    if text:
        logger.info("planning_generation: memo generated (%d chars)", len(text))
    else:
        logger.warning("planning_generation: empty response, using fallback")
        text = _fallback_plan(shadow_state, analysis_result)

    return text


def _fallback_plan(
    shadow_state: Dict[str, Any],
    analysis_result: Dict[str, Any],
) -> str:
    """Return a conservative default when the LLM call fails."""
    posture = analysis_result.get("posture", "neutral")
    dom = analysis_result.get("dominance_vector", 0.5)
    heat = shadow_state.get("heat_index", 0.0)
    return (
        f"Continue current trajectory. Posture: {posture}, "
        f"dominance at {dom:.2f}, heat at {heat:.2f}. "
        f"Assess next turn and adjust approach accordingly."
    )


async def _call_llm(
    model_manager,
    model_name: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.6,
    gpu_id: int = 0,
) -> str:
    """Call the LLM, handling both local and API endpoints.

    Follows the exact same pattern as contextual_analysis._call_llm.
    """
    from .. import inference
    from ..openai_compat import (
        is_api_endpoint,
        prepare_endpoint_request,
        collect_openai_compatible_stream_text,
    )

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
            logger.error("planning_generation: API call failed: %s", exc)
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
            logger.error("planning_generation: local call failed: %s", exc)
            return ""
