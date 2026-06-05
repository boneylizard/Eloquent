"""Locate ffmpeg on PATH, in settings, or at common Windows install paths."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FFMPEG_INSTALL_HINT = (
    "ffmpeg not found — install from https://ffmpeg.org/download.html "
    "(Windows: winget install Gyan.FFmpeg, or Chocolatey: choco install ffmpeg), "
    "add ffmpeg to your system PATH, or set the path in Settings → Audio → FFmpeg path."
)

_SETTINGS_PATH = Path.home() / ".LiangLocal" / "settings.json"


def load_ffmpeg_path_from_settings() -> Optional[str]:
    try:
        if not _SETTINGS_PATH.is_file():
            return None
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        val = (data.get("ffmpegPath") or "").strip()
        return val or None
    except Exception as exc:
        logger.debug("load_ffmpeg_path_from_settings: %s", exc)
        return None


def _windows_ffmpeg_candidates() -> list[Path]:
    if os.name != "nt":
        return []
    candidates: list[Path] = []
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    for base in (
        Path(pf) / "ffmpeg" / "bin",
        Path(pf86) / "ffmpeg" / "bin",
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\tools\ffmpeg\bin"),
        Path(r"C:\ProgramData\chocolatey\bin"),
        Path.home() / "scoop" / "shims",
    ):
        candidates.append(base / "ffmpeg.exe")
    if local:
        candidates.append(Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
    return candidates


def _is_executable_ffmpeg(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        proc = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            timeout=8,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def find_ffmpeg(*, explicit: Optional[str] = None) -> Optional[str]:
    """
    Resolve ffmpeg executable. Order: explicit arg, FFMPEG_BIN env, Settings ffmpegPath,
    PATH (shutil.which), then common Windows install locations.
    """
    for raw in (
        explicit,
        (os.getenv("FFMPEG_BIN") or "").strip() or None,
        load_ffmpeg_path_from_settings(),
    ):
        if not raw:
            continue
        p = Path(raw)
        if p.is_file() and _is_executable_ffmpeg(p):
            return str(p.resolve())
        found = shutil.which(raw)
        if found and _is_executable_ffmpeg(Path(found)):
            return found

    found = shutil.which("ffmpeg")
    if found and _is_executable_ffmpeg(Path(found)):
        return found

    for candidate in _windows_ffmpeg_candidates():
        if _is_executable_ffmpeg(candidate):
            return str(candidate.resolve())

    return None


def resolve_ffmpeg_bin(fallback: str = "ffmpeg") -> str:
    """Return resolved ffmpeg path or fallback name (for error messages)."""
    return find_ffmpeg() or fallback


def prepend_ffmpeg_dir_to_path(ffmpeg_bin: str) -> None:
    ffmpeg_dir = os.path.dirname(os.path.abspath(ffmpeg_bin))
    if not ffmpeg_dir:
        return
    current = os.environ.get("PATH", "")
    if ffmpeg_dir.lower() not in current.lower().split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current


def apply_ffmpeg_config(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    """
    Apply ffmpeg from settings or discovery to FFMPEG_BIN and PATH.
    Returns resolved binary path, or None if not found.
    """
    explicit = (ffmpeg_path or "").strip() or load_ffmpeg_path_from_settings()
    resolved = find_ffmpeg(explicit=explicit)
    if resolved:
        os.environ["FFMPEG_BIN"] = resolved
        prepend_ffmpeg_dir_to_path(resolved)
        logger.info("ffmpeg resolved: %s", resolved)
    else:
        os.environ.pop("FFMPEG_BIN", None)
        logger.warning("ffmpeg not found on PATH or in Settings (ffmpegPath)")
    return resolved


def bootstrap_ffmpeg_from_settings() -> Optional[str]:
    """Call once at backend startup."""
    return apply_ffmpeg_config()
