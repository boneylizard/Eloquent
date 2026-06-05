"""D-ID avatar pre-screening via OpenAI-compatible vision API (e.g. Kimi on NanoGPT)."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict

import httpx

from .openai_compat import collect_openai_compatible_stream_text, prepare_endpoint_request

logger = logging.getLogger("d_id_vision_screen")

SCORING_PROMPT = """You are evaluating a still image for use as a D-ID photo-avatar (talking head).

Score each criterion from 0 to 10 (10 best). Then give one overall score 0-10 (same scale).
Pass only if overall_score >= 7 and the face is usable for lip-sync (frontal, mouth visible, medium shot).

Criteria:
1) face_frontality — face directly facing camera
2) mouth_neutral — mouth closed or neutral, suitable for speech animation baseline
3) crop_framing — medium shot: face, neck, top of shoulders only; not too much body
4) face_size — face large enough for robust face detection
5) lighting — even, soft; no harsh shadows under nose or chin
6) occlusions — hair/jewellery/objects not covering mouth or jaw
7) realism — photorealistic enough for commercial face tracking (not heavy stylisation, not anime/cartoon)

Respond with ONLY a single JSON object (no markdown fences), keys:
{
  "overall_score": <number>,
  "pass": <true|false>,
  "criteria": {
    "face_frontality": <0-10>,
    "mouth_neutral": <0-10>,
    "crop_framing": <0-10>,
    "face_size": <0-10>,
    "lighting": <0-10>,
    "occlusions": <0-10>,
    "realism": <0-10>
  },
  "failure_reasons": [<short string>, ...]
}
If the image passes, failure_reasons should be []."""


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("Vision model did not return JSON.")
    return json.loads(m.group(0))


async def screen_image_bytes(
    image_bytes: bytes,
    *,
    vision_endpoint_model_id: str,
    content_type: str = "image/png",
) -> Dict[str, Any]:
    """
    vision_endpoint_model_id: Eloquent custom endpoint id, e.g. endpoint-xxxxx (must support vision).
    """
    if not vision_endpoint_model_id or not vision_endpoint_model_id.startswith("endpoint-"):
        raise ValueError(
            "Set D_ID_VISION_SCREEN_MODEL to your NanoGPT/Kimi custom endpoint id (endpoint-… from Settings)."
        )
    b64 = base64.b64encode(image_bytes).decode("ascii")
    mime = content_type if "/" in content_type else f"image/{content_type}"
    if not mime.startswith("image/"):
        mime = "image/png"
    data_url = f"data:{mime};base64,{b64}"

    messages = [
        {"role": "system", "content": "You output only valid JSON objects, no markdown."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SCORING_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    request_data: Dict[str, Any] = {
        "model": vision_endpoint_model_id,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1200,
        "stream": True,
        "_skip_openai_message_pruning": True,
    }
    endpoint_config, url, prepared = prepare_endpoint_request(vision_endpoint_model_id, request_data)
    text = await collect_openai_compatible_stream_text(endpoint_config, url, prepared, None)
    parsed = _extract_json_object(text)
    overall = float(parsed.get("overall_score", 0) or 0)
    passed = bool(parsed.get("pass")) and overall >= 7.0
    return {
        "overall_score": overall,
        "pass": passed,
        "criteria": parsed.get("criteria") or {},
        "failure_reasons": parsed.get("failure_reasons") or [],
        "raw_model_json": parsed,
    }


async def screen_image_from_url(image_url: str, *, vision_endpoint_model_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(image_url, follow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "image/png").split(";")[0].strip() or "image/png"
        return await screen_image_bytes(r.content, vision_endpoint_model_id=vision_endpoint_model_id, content_type=ct)
