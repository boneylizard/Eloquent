"""
Write exactly-what-would-be-sent prompt text to disk for inspection before calling an LLM.

Files land under backend/data/preview_prompts/ (created on demand).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_PREVIEW_SUBDIR = ("data", "preview_prompts")


def preview_prompts_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, "..", *_PREVIEW_SUBDIR))


def _safe_kind(kind: str) -> str:
    s = (kind or "prompt").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return (s[:56] or "prompt").strip("_")


def save_preview_prompt(kind: str, text: str) -> Dict[str, Any]:
    """Save UTF-8 text; returns paths and byte length for API responses."""
    d = preview_prompts_dir()
    os.makedirs(d, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fn = f"{_safe_kind(kind)}_{ts}.txt"
    path = os.path.join(d, fn)
    payload = text if isinstance(text, str) else ""
    raw = payload.encode("utf-8")
    with open(path, "wb") as f:
        f.write(raw)
    abs_path = os.path.abspath(path)
    return {
        "filename": fn,
        "absolute_path": abs_path,
        "directory": os.path.abspath(d),
        "bytes_written": len(raw),
    }


def list_preview_prompts(limit: int = 25) -> List[Dict[str, Any]]:
    """Most recently modified first."""
    d = preview_prompts_dir()
    if not os.path.isdir(d):
        return []
    entries = []
    for name in os.listdir(d):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(d, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        entries.append((st.st_mtime, name, path, st.st_size))
    entries.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for _, name, path, size in entries[: max(1, min(limit, 100))]:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat()
        except OSError:
            mtime = ""
        out.append(
            {
                "filename": name,
                "absolute_path": os.path.abspath(path),
                "size_bytes": size,
                "modified_utc": mtime,
            }
        )
    return out
