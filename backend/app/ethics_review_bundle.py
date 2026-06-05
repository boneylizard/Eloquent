"""
Assemble whitelisted local source excerpts for ethics / framing review.

Paths are fixed relative to the repo root (no client-supplied paths).
Line ranges are 1-based inclusive; adjust if files drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# (relative path from repo root, line_start or None for whole file, line_end or None)
PARTS: Tuple[Tuple[str, Optional[int], Optional[int]], ...] = (
    ("backend/app/persona_realignment.py", None, None),
    # Request DTOs only (implementation lines drift often)
    ("backend/app/memory_routes.py", 1332, 1359),
    ("backend/app/memory_curator_prompt.py", 1, 182),
    ("frontend/src/components/PersonaRealignmentPanel.jsx", 350, 643),
)


def manifest_parts() -> List[Dict[str, Any]]:
    """Human-readable scope for each whitelisted excerpt (no disk read). Keep in sync with PARTS."""
    out: List[Dict[str, Any]] = []
    for rel, lo, hi in PARTS:
        if lo is not None and hi is not None:
            scope = f"lines {lo}-{hi}"
        else:
            scope = "entire file"
        out.append(
            {
                "path": rel,
                "scope": scope,
                "line_start": lo,
                "line_end": hi,
            }
        )
    return out


def _repo_root() -> Path:
    # backend/app/ethics_review_bundle.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def _read_sliced(rel: str, line_start: Optional[int], line_end: Optional[int]) -> str:
    root = _repo_root()
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"path outside repo: {rel}") from None

    if not path.is_file():
        raise FileNotFoundError(str(path))

    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines(keepends=True)
    if line_start is None or line_end is None:
        return raw

    lo = max(1, line_start)
    hi = min(len(lines), line_end)
    if hi < lo:
        return ""
    chunk = "".join(lines[lo - 1 : hi])
    return chunk


def build_bundle_markdown() -> Tuple[str, List[Dict[str, Any]]]:
    """Returns markdown body and per-part metadata."""
    parts_out: List[Dict[str, Any]] = []
    blocks: List[str] = []

    for rel, lo, hi in PARTS:
        meta: Dict[str, Any] = {"path": rel, "ok": False}
        if lo is not None and hi is not None:
            meta["lines"] = f"{lo}-{hi}"
        try:
            text = _read_sliced(rel, lo, hi)
            meta["ok"] = True
            meta["chars"] = len(text)
            heading = f"### `{rel}`"
            if lo is not None and hi is not None:
                heading += f" (lines {lo}-{hi})"
            blocks.append(f"{heading}\n\n```\n{text.rstrip()}\n```\n")
        except Exception as e:
            meta["error"] = str(e)
            blocks.append(f"### `{rel}` (failed)\n\n_{e}_\n")

        parts_out.append(meta)

    body = (
        "The following text is copied from this project only. "
        "Use it when you want an AI to compare how the app is described versus how it is implemented.\n\n"
        + "\n".join(blocks)
    )
    return body.strip() + "\n", parts_out


def appendix_for_prompt_pack() -> Tuple[str, List[Dict[str, Any]]]:
    """Markdown appendix string + metadata (same as bundle)."""
    return build_bundle_markdown()
