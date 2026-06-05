"""
Load the Eloquent system grounding spec for in-app agents and automations.

Spec file: docs/ELOQUENT_SYSTEM_SPEC.md (repo root relative).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SPEC_REL = Path("docs") / "ELOQUENT_SYSTEM_SPEC.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def eloquent_system_spec_path() -> Path:
    return _repo_root() / _SPEC_REL


@lru_cache(maxsize=1)
def load_eloquent_system_spec() -> str:
    """Return full markdown spec text; empty string if missing."""
    path = eloquent_system_spec_path()
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def eloquent_system_spec_block() -> str:
    """Wrapped block suitable to prepend to a system or author prompt."""
    body = load_eloquent_system_spec().strip()
    if not body:
        return ""
    return (
        "[ELOQUENT SYSTEM GROUNDING — factual architecture reference; "
        "do not contradict without verifying in code]\n\n"
        f"{body}\n\n"
        "[END ELOQUENT SYSTEM GROUNDING]\n"
    )
