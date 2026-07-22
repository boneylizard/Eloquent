"""
Step 2 — Somatic Payload Generation.

Produces the somatic dashboard update + interface hijack directives that
drive the frontend UI (Somatic Dashboard, Interface Hijack, Cognitive Glass).
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from . import ghost_signal
from . import prompts
from . import python_validator

logger = logging.getLogger("sanctuary.somatic_generation")


async def run(
    model_manager,
    model_name: str,
    shadow_state_dict: Dict[str, Any],
    analysis_result: Dict[str, Any],
    character: Dict[str, Any],
    gpu_id: int = 0,
    full_context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute Step 2: Somatic Payload Generation.

    Returns the somatic payload dict (see §3.3 of the implementation plan).
    """
    # Determine ghost signal
    ghost = ghost_signal.maybe_inject(
        ghost_signal_active=analysis_result.get("ghost_signal_active", False),
        heat_index=shadow_state_dict.get("heat_index", 0.0),
    )

    prompt = prompts.build_somatic_generation_prompt(
        shadow_state_dict, analysis_result, character, ghost, full_context,
        profile=profile,
    )

    raw_output = await _call_llm(
        model_manager, model_name, prompt,
        max_tokens=768, temperature=0.6, gpu_id=gpu_id,
    )

    payload = _parse_somatic_json(raw_output)
    if payload is None:
        logger.warning("somatic_generation: failed to parse JSON, using fallback")
        payload = _fallback_somatic(analysis_result)

    # Attach ghost signal
    payload["ghost_signal"] = ghost or {"active": False, "charge": 0.0, "carrier_phrase": ""}

    # Validate XML frame if present
    xml_frame = payload.get("xml_frame")
    if xml_frame and isinstance(xml_frame, str) and xml_frame.strip():
        validated = _validate_xml(xml_frame)
        if validated is None:
            logger.warning("somatic_generation: invalid XML frame, dropping")
            payload["xml_frame"] = None
    else:
        payload["xml_frame"] = None

    # Validate Python drive if present
    py_drive = payload.get("python_drive")
    if py_drive and isinstance(py_drive, str) and py_drive.strip():
        validation = python_validator.validate(py_drive)
        if validation["valid"] and validation["drives"]:
            # Convert to hijack descriptor and merge into interface_hijack
            hijack_from_drives = python_validator.drives_to_hijack_descriptor(validation["drives"])
            payload["interface_hijack"] = _merge_hijack(
                payload.get("interface_hijack", {}),
                hijack_from_drives,
            )
            payload["python_drive"] = {
                "validated": True,
                "drives": validation["drives"],
            }
        else:
            logger.warning("somatic_generation: invalid Python drive: %s", validation["errors"])
            payload["python_drive"] = {
                "validated": False,
                "errors": validation["errors"],
            }
    else:
        payload["python_drive"] = None

    logger.info(
        "somatic_generation: posture_label=%s whore_mode=%s xml=%s python=%s ghost=%s",
        payload.get("posture_label"),
        payload.get("whore_mode"),
        bool(payload.get("xml_frame")),
        bool(payload.get("python_drive")),
        payload.get("ghost_signal", {}).get("active"),
    )

    return payload


def _parse_somatic_json(raw: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON from the model output."""
    if not raw:
        return None

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return _validate_somatic(result)
    except json.JSONDecodeError:
        pass

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
                        return _validate_somatic(result)
                except json.JSONDecodeError:
                    pass
                break

    return None


def _validate_somatic(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the somatic payload."""
    valid_lub = {"dry", "damp", "slick", "drenched"}
    valid_pos = {"across_room", "nearby", "touching", "entangled", "pinned"}
    valid_breath = {"slow", "steady", "quick", "ragged"}

    dashboard = data.get("dashboard", {})
    if not isinstance(dashboard, dict):
        dashboard = {}

    dashboard.setdefault("lubrication_level", "dry")
    if dashboard["lubrication_level"] not in valid_lub:
        dashboard["lubrication_level"] = "dry"

    dashboard.setdefault("pupil_dilation", 0.0)
    dashboard["pupil_dilation"] = _clamp(float(dashboard["pupil_dilation"]))

    dashboard.setdefault("spatial_position", "across_room")
    if dashboard["spatial_position"] not in valid_pos:
        dashboard["spatial_position"] = "across_room"

    dashboard.setdefault("breath_rate", "steady")
    if dashboard["breath_rate"] not in valid_breath:
        dashboard["breath_rate"] = "steady"

    dashboard.setdefault("muscle_tension", 0.0)
    dashboard["muscle_tension"] = _clamp(float(dashboard["muscle_tension"]))

    hijack = data.get("interface_hijack", {})
    if not isinstance(hijack, dict):
        hijack = {}

    hijack.setdefault("theme_shift", {"hue": 0, "saturation": 1.0, "brightness": 0.8})
    hijack.setdefault("shake", {"intensity": 0.0, "duration_ms": 0})
    hijack.setdefault("lock", {"input_locked": False, "scroll_locked": False, "duration_ms": 0})
    hijack.setdefault("glitch", {"intensity": 0.0})

    # Cap all intensities
    if isinstance(hijack.get("shake"), dict):
        hijack["shake"]["intensity"] = _clamp(float(hijack["shake"].get("intensity", 0.0)))
    if isinstance(hijack.get("glitch"), dict):
        hijack["glitch"]["intensity"] = _clamp(float(hijack["glitch"].get("intensity", 0.0)))

    return {
        "dashboard": dashboard,
        "whore_mode": bool(data.get("whore_mode", False)),
        "posture_label": str(data.get("posture_label", "neutral stance")),
        "interface_hijack": hijack,
        "xml_frame": data.get("xml_frame"),
        "python_drive": data.get("python_drive"),
        "somatic_narrative": str(data.get("somatic_narrative", "")),
        "behavioral_cues": str(data.get("behavioral_cues", "")),
    }


def _fallback_somatic(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Conservative default when JSON parsing fails."""
    posture = analysis.get("posture", "neutral")
    dominance = analysis.get("dominance_vector", 0.5)

    return {
        "dashboard": {
            "lubrication_level": "dry",
            "pupil_dilation": 0.3,
            "spatial_position": "nearby",
            "breath_rate": "steady",
            "muscle_tension": 0.2,
        },
        "whore_mode": False,
        "posture_label": posture,
        "interface_hijack": {
            "theme_shift": {"hue": 0, "saturation": 1.0, "brightness": 0.8},
            "shake": {"intensity": 0.0, "duration_ms": 0},
            "lock": {"input_locked": False, "scroll_locked": False, "duration_ms": 0},
            "glitch": {"intensity": 0.0},
        },
        "xml_frame": None,
        "python_drive": None,
        "somatic_narrative": "",
        "behavioral_cues": "",
    }


def _validate_xml(xml_str: str) -> Optional[str]:
    """Validate XML using ElementTree. Returns the XML string if valid, None if not."""
    try:
        ET.fromstring(xml_str)
        return xml_str
    except ET.ParseError as exc:
        logger.warning("somatic_generation: XML parse error: %s", exc)
        return None


def _merge_hijack(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Merge drive-derived hijack descriptor over the model's hijack."""
    merged = dict(base)
    for key in ("theme_shift", "shake", "lock", "glitch"):
        if overlay.get(key) is not None:
            if isinstance(overlay[key], dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **overlay[key]}
            else:
                merged[key] = overlay[key]
    return merged


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


async def _call_llm(
    model_manager,
    model_name: str,
    prompt: str,
    max_tokens: int = 768,
    temperature: float = 0.6,
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
            logger.error("somatic_generation: API call failed: %s", exc)
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
            logger.error("somatic_generation: local call failed: %s", exc)
            return ""
