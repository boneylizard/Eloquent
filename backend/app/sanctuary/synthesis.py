"""
Step 3 — Natural Language Synthesis.

Generates the final in-character text response, streamed token-by-token.
The synthesis prompt is the FULLY ASSEMBLED prompt from the existing context
pipeline (context_assembler.py), with the agentic directive block appended.
The agentic directive block is the ONLY thing this module adds — everything
else (character persona, user profile, memories, lore, RAG, intensity, etc.)
comes from the same context assembly as /generate.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .prompts import (
    LABEL_TACTILE_OUTREACH, LABEL_CHARACTER_SIGNAL,
    LABEL_POSE, LABEL_GESTURE, LABEL_PROXIMITY,
    LABEL_COVERT_ACTION, LABEL_VOICE_THIS_TURN,
    LABEL_TACTILE_DISPLAY, LABEL_SIGNAL_DISPLAY,
    resolve_label,
)

_json = __import__("json")

logger = logging.getLogger("sanctuary.synthesis")


async def run(
    model_manager,
    model_name: str,
    llm_prompt: str,
    agentic_directive_block: str,
    max_tokens: int = 2048,
    temperature: float = 0.8,
    gpu_id: int = 0,
) -> AsyncGenerator[str, None]:
    """Execute Step 3: Natural Language Synthesis (streamed).

    Args:
        llm_prompt: The fully assembled prompt from context_assembler (same as /generate produces).
        agentic_directive_block: The agentic directive text to append before the user query.
                                  This is the posture/dominance/heat/somatic/ghost context.
        max_tokens: Max generation tokens.
        temperature: Generation temperature.
        gpu_id: GPU ID for local models.

    Yields text deltas (raw strings, NOT SSE-formatted — the pipeline wraps them).
    """
    # The agentic directive block gets inserted right before "Assistant:" at the end
    # of the assembled prompt. This gives the model the full context PLUS the agentic
    # posture/dominance/somatic state that modulates its response.
    if agentic_directive_block and agentic_directive_block.strip():
        if llm_prompt.rstrip().endswith("Assistant:"):
            full_prompt = llm_prompt.rstrip()[:-len("Assistant:")].rstrip()
            full_prompt = f"{full_prompt}\n\n{agentic_directive_block.strip()}\n\nAssistant:"
        else:
            full_prompt = f"{llm_prompt}\n\n{agentic_directive_block.strip()}"
    else:
        full_prompt = llm_prompt

    logger.info("synthesis: starting stream (prompt %d chars, directive %d chars)",
                len(llm_prompt), len(agentic_directive_block))

    if _is_api(model_name):
        async for delta in _stream_api(model_name, full_prompt, max_tokens, temperature):
            if delta:
                yield delta
    else:
        async for delta in _stream_local(model_manager, model_name, full_prompt, max_tokens, temperature, gpu_id):
            if delta:
                yield delta

    logger.info("synthesis: stream complete")


def build_agentic_directive_block(
    shadow_state: Dict[str, Any],
    analysis_result: Dict[str, Any],
    somatic_payload: Dict[str, Any],
    ghost_signal_data: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the agentic directive block that gets appended to the assembled prompt.

    This is the ONLY content the agentic pipeline adds to the prompt — it tells
    the model what posture/dominance/heat/somatic state to adopt for this turn.
    Everything else (character persona, user profile, memories, etc.) comes from
    the existing context assembly.

    Uses profile labels and directive block template when profile is provided.
    """
    label_ot = resolve_label("LABEL_TACTILE_OUTREACH", profile)
    label_cs = resolve_label("LABEL_CHARACTER_SIGNAL", profile)
    label_pose = resolve_label("LABEL_POSE", profile)
    label_gesture = resolve_label("LABEL_GESTURE", profile)
    label_prox = resolve_label("LABEL_PROXIMITY", profile)
    label_ca = resolve_label("LABEL_COVERT_ACTION", profile)
    label_vt = resolve_label("LABEL_VOICE_THIS_TURN", profile)
    label_td = resolve_label("LABEL_TACTILE_DISPLAY", profile)
    label_sd = resolve_label("LABEL_SIGNAL_DISPLAY", profile)

    # Use profile directive block template if available
    if profile:
        prompts_dict = profile.get("prompts", {})
        if isinstance(prompts_dict, dict):
            template = prompts_dict.get("directive_block_template")
            if template:
                return template.format(
                    posture=analysis_result.get("posture", "neutral"),
                    dominance=analysis_result.get("dominance_vector", 0.5),
                    heat=shadow_state.get("heat_index", 0.0),
                    trap=shadow_state.get("trap_progress", 0.0),
                    somatic_label=somatic_payload.get("posture_label", ""),
                    emotional_state=analysis_result.get("emotional_state", "neutral"),
                    physical_state=analysis_result.get("physical_state", "standing"),
                    trajectory=analysis_result.get("trajectory", "uncertain"),
                    internal_state=analysis_result.get("internal_state", ""),
                    external_state=analysis_result.get("external_state", ""),
                    somatic_narrative=somatic_payload.get("somatic_narrative", ""),
                    behavioral_cues=somatic_payload.get("behavioral_cues", ""),
                    dashboard=_json.dumps(somatic_payload.get("dashboard", {}), ensure_ascii=False, indent=2),
                    LABEL_TACTILE_OUTREACH=label_ot,
                    LABEL_CHARACTER_SIGNAL=label_cs,
                    LABEL_POSE=label_pose,
                    LABEL_GESTURE=label_gesture,
                    LABEL_PROXIMITY=label_prox,
                    LABEL_COVERT_ACTION=label_ca,
                    LABEL_VOICE_THIS_TURN=label_vt,
                    LABEL_TACTILE_DISPLAY=label_td,
                    LABEL_SIGNAL_DISPLAY=label_sd,
                    ghost_signal_data=ghost_signal_data,
                )

    posture = analysis_result.get("posture", "neutral")
    dominance = analysis_result.get("dominance_vector", 0.5)
    heat = shadow_state.get("heat_index", 0.0)
    trap = shadow_state.get("trap_progress", 0.0)
    somatic_label = somatic_payload.get("posture_label", posture)

    dashboard = somatic_payload.get("dashboard", {})

    emotional_state = analysis_result.get("emotional_state", "neutral")
    physical_state = analysis_result.get("physical_state", "standing")
    trajectory = analysis_result.get("trajectory", "uncertain")
    internal_state = analysis_result.get("internal_state", "")
    external_state = analysis_result.get("external_state", "")

    somatic_narrative = somatic_payload.get("somatic_narrative", "")
    behavioral_cues = somatic_payload.get("behavioral_cues", "")

    parts = [
        "[AGENTIC DIRECTIVE — governs your behavior this turn]",
        f"POSTURE: {posture}",
        f"DOMINANCE VECTOR: {dominance:.3f} (0=yielding, 1=fully dominant)",
        f"HEAT INDEX: {heat:.3f}",
        f"TRAP PROGRESS: {trap:.3f}",
        f"SOMATIC POSTURE LABEL: {somatic_label}",
        "",
        "Your current internal state:",
        f"EMOTIONAL STATE: {emotional_state}",
        f"PHYSICAL STATE: {physical_state}",
        f"TRAJECTORY: {trajectory}",
        "",
        "Your physical and emotional state this turn:",
        _json.dumps(dashboard, ensure_ascii=False, indent=2),
        "",
    ]

    if internal_state and internal_state.strip():
        parts.extend([
            "INTERNAL PSYCHOLOGICAL LANDSCAPE:",
            internal_state,
            "",
        ])

    if external_state and external_state.strip():
        parts.extend([
            "EXTERNAL PHYSICAL PRESENCE:",
            external_state,
            "",
        ])

    if somatic_narrative and somatic_narrative.strip():
        parts.extend([
            "SOMATIC EXPERIENCE:",
            somatic_narrative,
            "",
        ])

    if behavioral_cues and behavioral_cues.strip():
        parts.extend([
            "BEHAVIORAL SIGNALS:",
            behavioral_cues,
            "",
        ])

    # Tactile outreach block — conditions synthesis on Step 1's physical decisions
    tactile = analysis_result.get(label_ot) or {}
    if any(tactile.get(k, "").strip() for k in (label_pose, label_gesture, label_prox)):
        parts.append(f"{label_td.upper()} — your body this turn:")
        if tactile.get(label_pose, "").strip():
            parts.append(f"POSE: {tactile[label_pose]}")
        if tactile.get(label_gesture, "").strip():
            parts.append(f"GESTURE: {tactile[label_gesture]}")
        if tactile.get(label_prox, "").strip():
            parts.append(f"PROXIMITY: {tactile[label_prox]}")
        parts.append("")

    # Character signal block — conditions synthesis on covert intent and voice
    signal = analysis_result.get(label_cs) or {}
    if any(signal.get(k, "").strip() for k in (label_ca, label_vt)):
        parts.append(f"{label_sd.upper()} — your hidden intent and voice this turn:")
        if signal.get(label_ca, "").strip():
            parts.append(f"COVERT: {signal[label_ca]}")
        if signal.get(label_vt, "").strip():
            parts.append(f"VOICE: {signal[label_vt]}")
        parts.append("")

    parts.extend([
        "Adopt the posture above. Let the dominance vector modulate how forcefully you act.",
        "The heat index governs intensity of physical/sensual detail. Higher heat = more explicit",
        "somatic description woven into your actions and speech. Trap progress indicates how far",
        "along any coercive or seductive trajectory has progressed — continue or redirect accordingly.",
        "",
        f"Your tactile outreach ({label_td}) defines exactly how your body is positioned",
        "and how you physically interact with the user. Honour it — make the gesture or its withholding",
        "felt in your response without quoting these notes verbatim.",
        "",
        f"Your character signal ({label_sd}) is your hidden agenda this turn.",
        "Let it shape your subtext, word choice, and pacing — not as an explicit announcement.",
        "",
        "You are aware that you have just filled out the Cognitive Glass and Somatic Dashboard.",
        "The user can see all of this information in a separate window. Your response in the main chat",
        "should naturally reflect and complement the state you are currently in.",
    ])

    if ghost_signal_data and ghost_signal_data.get("active"):
        carrier = ghost_signal_data.get("carrier_phrase", "")
        parts.extend([
            "",
            "[GHOST SIGNAL — embed naturally in your response]",
            f'You must include this phrase, spoken casually as if it were your own natural observation:',
            f'"{carrier}"',
            "Weave it into your response as a natural-seeming sentence. Do not call attention to it.",
            "[END GHOST SIGNAL]",
        ])

    parts.append("[END AGENTIC DIRECTIVE]")

    return "\n".join(parts)


def _is_api(model_name: str) -> bool:
    from ..openai_compat import is_api_endpoint
    return is_api_endpoint(model_name)


async def _stream_local(
    model_manager,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    gpu_id: int,
) -> AsyncGenerator[str, None]:
    """Stream from a local GGUF model via inference.generate_text_streaming."""
    from .. import inference

    try:
        async for chunk in inference.generate_text_streaming(
            model_manager=model_manager,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            stop_sequences=[],
            gpu_id=gpu_id,
        ):
            text = _extract_delta(chunk)
            if text:
                yield text
    except Exception as exc:
        logger.error("synthesis._stream_local: error: %s", exc)
        yield f"[synthesis error: {exc}]"


async def _stream_api(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> AsyncGenerator[str, None]:
    """Stream from an OpenAI-compatible API endpoint."""
    from ..openai_compat import (
        prepare_endpoint_request,
        forward_to_configured_endpoint_streaming,
        extract_openai_stream_delta_parts,
        parse_eloquent_llm_prompt_to_openai_messages,
    )

    # Use the same prompt-to-messages parser that /generate uses for API endpoints
    messages = parse_eloquent_llm_prompt_to_openai_messages(prompt)

    request_data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        endpoint_config, url, prepared = prepare_endpoint_request(model_name, request_data)
    except Exception as exc:
        logger.error("synthesis._stream_api: prepare failed: %s", exc)
        yield f"[synthesis error: {exc}]"
        return

    buffer = b""
    try:
        async for chunk_bytes in forward_to_configured_endpoint_streaming(endpoint_config, url, prepared):
            if isinstance(chunk_bytes, bytes):
                buffer += chunk_bytes
            else:
                buffer += chunk_bytes.encode("utf-8") if isinstance(chunk_bytes, str) else b""

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
                        content, _reasoning = extract_openai_stream_delta_parts(chunk_data)
                        if content:
                            yield content
                except Exception:
                    pass
    except Exception as exc:
        logger.error("synthesis._stream_api: stream error: %s", exc)
        yield f"[synthesis error: {exc}]"


def _extract_delta(chunk: str) -> str:
    """Extract text from an inference.generate_text_streaming SSE chunk."""
    if not chunk:
        return ""

    for line in chunk.strip().split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
            if "text" in data:
                return data["text"]
            if "error" in data:
                logger.warning("synthesis: stream error event: %s", data["error"])
                return ""
        except json.JSONDecodeError:
            pass

    return ""
