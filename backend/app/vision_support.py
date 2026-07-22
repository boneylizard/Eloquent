import json
from typing import Optional


DEFAULT_VISION_SCHEMA = """description: A concise, factual account of the image
objects: The important objects, people, animals, or interface elements and where they appear
scene_type: The kind of scene, document, screenshot, or interface shown
text_content: Visible text, transcribed accurately; use an empty string when none is legible
colours: The dominant colours"""


def build_vision_messages(image_base64: str, schema_yaml: Optional[str], vision_mode: str):
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + image_base64},
    }
    if vision_mode == "extract":
        fields = (schema_yaml or DEFAULT_VISION_SCHEMA).strip()
        system_prompt = (
            "Extract the following from the image:\n\n"
            f"{fields}\n\n"
            "Respond with only a JSON object. Do not include any text outside the JSON."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [image_part]},
        ]
    return [
        {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": "Describe this image accurately and in useful detail."},
            ],
        }
    ]


def build_vision_completion_options(messages, vision_mode: str, max_tokens: int, temperature: float, repeat_penalty: float):
    if vision_mode == "extract":
        return {
            "messages": messages,
            "max_tokens": max(64, min(int(max_tokens or 512), 1024)),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "response_format": {"type": "json_object"},
        }
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": max(temperature, 0.3),
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": repeat_penalty,
    }


def parse_json_object(content: str):
    candidate = str(content or "").strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```json").removeprefix("```")
        candidate = candidate.removesuffix("```").strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
