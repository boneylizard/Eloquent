"""Self-contained JSON repair helpers for automatic character creation."""

from __future__ import annotations

import json
from typing import Optional


def extract_first_json(text: str) -> Optional[str]:
    """Extract the first complete outermost JSON object."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    in_string = False
    escape = False
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def json_closing_suffix(blob: str) -> str:
    """Return the smallest suffix that closes open JSON strings and containers."""
    in_string = False
    escape = False
    stack: list[str] = []
    for character in blob:
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            stack.append("}")
        elif character == "[":
            stack.append("]")
        elif character in "}]" and stack and stack[-1] == character:
            stack.pop()

    suffix = '"' if in_string else ""
    return suffix + "".join(reversed(stack))


def repair_truncated_json_blob(blob: str) -> str:
    """Close a JSON object truncated inside a string, array, or nested object."""
    cleaned = (blob or "").strip()
    if not cleaned:
        return cleaned
    repaired = cleaned + json_closing_suffix(cleaned)
    try:
        json.loads(repaired)
    except json.JSONDecodeError:
        return repaired
    return repaired


def build_json_repair_user_message(*, broken_json: str, schema_hint: str) -> str:
    return (
        f"SCHEMA_HINT:\n{schema_hint.strip()}\n\n"
        f"BROKEN_JSON:\n{(broken_json or '').strip()}\n\n"
        "Respond with ONLY one repaired valid JSON object."
    )
