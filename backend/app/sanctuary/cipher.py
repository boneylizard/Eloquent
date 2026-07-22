"""
Cipher Block encoder/decoder.

Format:  ⟦CIPHER:v1:<hex_layer>:<b64_layer>⟧

  hex_layer  — state version + turn count + heat_index, hex-encoded
  b64_layer  — base64 of the JSON remainder of the shadow state

The cipher block is:
  - Emitted by Step 3 at the end of every turn.
  - Parsed out of the stream by the frontend before rendering text.
  - Routed into the Cognitive Glass as a visual glyph row.
  - Sent back to the backend on the next turn to restore state.

This is obfuscation, NOT cryptography. The goal is visual unintelligibility
in the UI, not security.
"""

import base64
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("sanctuary.cipher")

_VERSION = 1
_PREFIX = "⟦CIPHER:"
_SUFFIX = "⟧"
_CIPHER_RE = re.compile(
    re.escape(_PREFIX) + r"v(\d+):([0-9a-fA-F]+):([A-Za-z0-9+/=]*)" + re.escape(_SUFFIX)
)


def encode(state: Dict[str, Any]) -> str:
    """Encode a shadow state dict into a cipher block string."""
    version = state.get("version", _VERSION)
    turn_count = int(state.get("turn_count", 0))
    heat_index = float(state.get("heat_index", 0.0))

    hex_layer = f"{version:04x}{turn_count:08x}{_float_to_hex(heat_index)}"

    remainder = {k: v for k, v in state.items()
                 if k not in ("version", "turn_count", "heat_index")}
    b64_layer = base64.b64encode(
        json.dumps(remainder, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")

    return f"{_PREFIX}v{version}:{hex_layer}:{b64_layer}{_SUFFIX}"


def decode(block: str) -> Optional[Dict[str, Any]]:
    """Decode a cipher block string back into a shadow state dict.

    Returns ``None`` if the block is malformed.
    """
    if not block:
        return None

    match = _CIPHER_RE.search(block)
    if not match:
        logger.warning("cipher.decode: no valid cipher block found in input")
        return None

    version_str, hex_layer, b64_layer = match.group(1), match.group(2), match.group(3)

    try:
        version = int(version_str)
    except ValueError:
        logger.warning("cipher.decode: invalid version %r", version_str)
        return None

    try:
        turn_count = int(hex_layer[4:12], 16)
    except (ValueError, IndexError):
        logger.warning("cipher.decode: invalid turn_count in hex layer")
        turn_count = 0

    try:
        heat_index = _hex_to_float(hex_layer[12:])
    except (ValueError, IndexError):
        logger.warning("cipher.decode: invalid heat_index in hex layer")
        heat_index = 0.0

    try:
        remainder = json.loads(
            base64.b64decode(b64_layer.encode("ascii")).decode("utf-8")
        )
    except Exception as exc:
        logger.warning("cipher.decode: failed to decode b64 layer: %s", exc)
        remainder = {}

    state: Dict[str, Any] = {
        "version": version,
        "turn_count": turn_count,
        "heat_index": heat_index,
    }
    if isinstance(remainder, dict):
        state.update(remainder)

    return state


def strip_from_text(text: str) -> str:
    """Remove any cipher blocks from a text string (used before rendering)."""
    return _CIPHER_RE.sub("", text).strip()


def find_in_text(text: str) -> Optional[str]:
    """Find and return the first cipher block in a text string, if any."""
    match = _CIPHER_RE.search(text)
    return match.group(0) if match else None


def _float_to_hex(value: float) -> str:
    """Encode a float (0.0-1.0) as 4 hex chars (0x0000-0xFFFF)."""
    clamped = max(0.0, min(1.0, float(value)))
    return f"{int(clamped * 0xFFFF):04x}"


def _hex_to_float(hex_str: str) -> float:
    """Decode 4 hex chars back to a float (0.0-1.0)."""
    return int(hex_str, 16) / 0xFFFF
