"""
D-ID talking-head pipeline (V2 Photo Avatars via POST /talks).

Auth: Basic HTTP — set D_ID_API_KEY to "username:password" from Studio
(Account & API settings), or set D_ID_API_USER and D_ID_API_SECRET separately.

Env:
  D_ID_API_KEY          e.g. "api_user_xxx:secret_yyy" (recommended)
  D_ID_API_USER / D_ID_API_SECRET   alternative split form
  D_ID_API_BASE         default https://api.d-id.com

Flow: upload WAV → POST /audios → audio_url → POST /talks with source_url + script
→ poll GET /talks/{id} → download result MP4 → optional ffmpeg concat.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .runtime_paths import data_path

logger = logging.getLogger("d_id_service")

DEFAULT_BASE = "https://api.d-id.com"


def get_api_base() -> str:
    return (os.environ.get("D_ID_API_BASE") or DEFAULT_BASE).rstrip("/")


def d_id_credentials_configured() -> bool:
    """True if env has enough info to build Basic auth (does not validate with D-ID)."""
    try:
        _basic_auth_header()
        return True
    except ValueError:
        return False


def _basic_auth_header() -> str:
    raw = (os.environ.get("D_ID_API_KEY") or "").strip()
    if raw and ":" in raw:
        user, password = raw.split(":", 1)
    else:
        user = (os.environ.get("D_ID_API_USER") or "").strip()
        password = (os.environ.get("D_ID_API_SECRET") or raw).strip()
    if not password:
        raise ValueError(
            "D-ID is not configured: set D_ID_API_KEY to username:password from studio.d-id.com "
            "or set D_ID_API_USER and D_ID_API_SECRET."
        )
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _headers_json() -> Dict[str, str]:
    return {
        "Authorization": _basic_auth_header(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def get_d_id_export_dir() -> Path:
    out = data_path("d_id_exports")
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_d_id_batch_runs_dir() -> Path:
    out = data_path("d_id_batch_runs")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _extract_result_url(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    u = payload.get("result_url")
    if isinstance(u, str) and u.startswith("http"):
        return u
    res = payload.get("result")
    if isinstance(res, dict):
        for key in ("url", "video_url", "download_url", "mp4_url"):
            v = res.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return v
    return None


def _extract_talk_status(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    st = payload.get("status")
    if isinstance(st, str):
        return st.lower()
    return "unknown"


async def upload_audio_file(
    client: httpx.AsyncClient,
    audio_path: Path,
    *,
    field_name: str = "audio",
) -> str:
    """
    POST /audios (multipart). Returns a URL usable as script.audio_url for /talks.
    """
    url = f"{get_api_base()}/audios"
    headers = {"Authorization": _basic_auth_header(), "Accept": "application/json"}
    name = audio_path.name or "speech.wav"
    data_bytes = audio_path.read_bytes()
    last_err = None
    r: Optional[httpx.Response] = None
    for fname in (field_name, "file", "audio_file"):
        files = {fname: (name, data_bytes, "audio/wav")}
        r = await client.post(url, headers=headers, files=files, timeout=300.0)
        if r.status_code < 400:
            break
        last_err = f"{r.status_code}: {r.text[:400]}"
    if r is None or r.status_code >= 400:
        raise RuntimeError(f"D-ID /audios failed: {last_err or 'no response'}")
    try:
        body = r.json()
    except Exception:
        raise RuntimeError(f"D-ID /audios non-JSON response: {r.text[:500]}")
    # Response shapes seen in the wild
    for key in ("url", "audio_url", "s3_url"):
        v = body.get(key)
        if isinstance(v, str) and v:
            return v
    # Nested
    aud = body.get("audio") or body.get("data")
    if isinstance(aud, dict):
        for key in ("url", "audio_url"):
            v = aud.get(key)
            if isinstance(v, str) and v:
                return v
    raise RuntimeError(f"D-ID /audios: could not parse audio URL from: {body}")


def map_emotion_for_d_id(emotion: Optional[str]) -> Optional[str]:
    """Map UI emotion to D-ID Expression enum (neutral, happy, serious, surprise)."""
    if not emotion:
        return None
    e = emotion.strip().lower()
    mapping = {
        "happy": "happy",
        "neutral": "neutral",
        "serious": "serious",
        "surprised": "surprise",
        "surprise": "surprise",
    }
    return mapping.get(e)


def driver_url_for_movement(movement: Optional[str]) -> Optional[str]:
    """Movement style → D-ID driver_url bank (best-effort; API may ignore unknown values)."""
    if not movement:
        return None
    m = movement.strip().lower()
    if m == "active":
        return "bank://lively/driver-06"
    if m == "still":
        return "bank://subtle/driver-01"
    return None


def resolve_public_image_url(url_or_path: str) -> str:
    """
    D-ID requires a fetchable https URL for source images.
    If a relative /static/... path is passed, prefix D_ID_PUBLIC_BASE_URL (must be https, no trailing slash).
    """
    ref = (url_or_path or "").strip()
    if not ref:
        raise ValueError("Empty image URL.")
    if ref.startswith("https://") or ref.startswith("http://"):
        return ref
    base = (os.environ.get("D_ID_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base:
        raise ValueError(
            "Image must be a full https URL reachable by D-ID, or set D_ID_PUBLIC_BASE_URL to your "
            "public origin (e.g. https://your-host) so /static/... paths can be resolved."
        )
    if ref.startswith("/"):
        return f"{base}{ref}"
    return f"{base}/{ref}"


async def create_talk_audio(
    client: httpx.AsyncClient,
    *,
    source_url: str,
    audio_url: str,
    fluent: bool = True,
    pad_audio: float = 0.0,
    emotion: Optional[str] = None,
    movement: Optional[str] = None,
    background_url: Optional[str] = None,
) -> str:
    """POST /talks — V2 photo avatar + audio script. Returns talk id."""
    if not source_url or not source_url.startswith("http"):
        raise ValueError("source_url must be a public https URL to the avatar image.")
    if not audio_url:
        raise ValueError("audio_url is required (from /audios upload).")

    config: Dict[str, Any] = {
        "fluent": bool(fluent),
        "pad_audio": float(pad_audio),
    }
    expr = map_emotion_for_d_id(emotion)
    if expr:
        config["driver_expressions"] = {
            "expressions": [
                {"start_frame": 0, "expression": expr, "intensity": 1.0},
            ],
            "transition_frames": 15,
        }

    payload: Dict[str, Any] = {
        "source_url": source_url.strip(),
        "script": {
            "type": "audio",
            "audio_url": audio_url.strip(),
        },
        "config": config,
    }
    drv = driver_url_for_movement(movement)
    if drv:
        payload["driver_url"] = drv
    bg = (background_url or "").strip()
    if bg.startswith("http://") or bg.startswith("https://"):
        payload["background_url"] = bg

    url = f"{get_api_base()}/talks"
    r = await client.post(url, headers=_headers_json(), json=payload, timeout=120.0)
    if r.status_code not in (200, 201):
        detail = r.text[:800]
        try:
            detail = str(r.json())
        except Exception:
            pass
        raise RuntimeError(f"D-ID POST /talks failed {r.status_code}: {detail}")
    body = r.json()
    tid = body.get("id") or body.get("talk_id")
    if not isinstance(tid, str):
        raise RuntimeError(f"D-ID /talks: missing id in response: {body}")
    return tid


async def get_talk(client: httpx.AsyncClient, talk_id: str) -> Dict[str, Any]:
    url = f"{get_api_base()}/talks/{talk_id}"
    r = await client.get(url, headers=_headers_json(), timeout=60.0)
    r.raise_for_status()
    return r.json()


async def wait_talk_done(
    client: httpx.AsyncClient,
    talk_id: str,
    *,
    timeout_s: float = 900.0,
    interval_s: float = 2.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = await get_talk(client, talk_id)
        st = _extract_talk_status(last)
        if st in ("done", "completed", "success"):
            return last
        if st in ("error", "rejected", "failed"):
            err = last.get("error") or last.get("message") or last
            raise RuntimeError(f"D-ID talk failed: {err}")
        await asyncio.sleep(interval_s)
    raise TimeoutError(f"D-ID talk {talk_id} did not finish within {timeout_s}s; last={last}")


async def download_url_to_file(client: httpx.AsyncClient, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = await client.get(url, follow_redirects=True, timeout=300.0)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def resolve_source_url(avatar_ref: str, *, default_source_url: Optional[str] = None) -> str:
    """
    avatar_ref: either an https image URL or falls back to D_ID_DEFAULT_SOURCE_URL env.
    """
    ref = (avatar_ref or "").strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        return ref
    d = (default_source_url or os.environ.get("D_ID_DEFAULT_SOURCE_URL") or "").strip()
    if d.startswith("http"):
        return d
    raise ValueError(
        "Avatar image URL required: pass source_url/https avatar_ref, or set D_ID_DEFAULT_SOURCE_URL."
    )


async def wav_to_talk_mp4(
    wav_path: Path,
    avatar_ref: str,
    *,
    out_dir: Optional[Path] = None,
    talk_id_hint: Optional[str] = None,
    emotion: Optional[str] = None,
    movement: Optional[str] = None,
    background_url: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Full pipeline: upload WAV → create talk → poll → save MP4 under out_dir.
    Returns (mp4_path, last_talk_json).
    """
    out_dir = out_dir or get_d_id_export_dir()
    source_url = resolve_public_image_url(resolve_source_url(avatar_ref))
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(str(wav_path))

    async with httpx.AsyncClient() as client:
        audio_url = await upload_audio_file(client, wav_path)
        logger.info("D-ID audio uploaded, creating talk…")
        talk_id = await create_talk_audio(
            client,
            source_url=source_url,
            audio_url=audio_url,
            emotion=emotion,
            movement=movement,
            background_url=background_url,
        )
        logger.info("D-ID talk id=%s polling…", talk_id)
        final = await wait_talk_done(client, talk_id)
        result_url = _extract_result_url(final)
        if not result_url:
            raise RuntimeError(f"D-ID talk done but no result_url in payload keys={list(final.keys())}")
        stem = talk_id_hint or talk_id.replace("/", "_")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80]
        dest = out_dir / f"did_talk_{safe}.mp4"
        await download_url_to_file(client, result_url, dest)
        return dest, final


def concat_mp4_files_ffmpeg(input_paths: List[Path], output_path: Path) -> Path:
    """
    Same pattern as repo merge_videos.py: ffmpeg concat demuxer → single H.264/AAC MP4.
    """
    paths = [Path(p) for p in input_paths if Path(p).is_file()]
    if not paths:
        raise ValueError("No input MP4 files.")
    from .ffmpeg_utils import FFMPEG_INSTALL_HINT, find_ffmpeg
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(FFMPEG_INSTALL_HINT)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tf:
        list_file = tf.name
        for path in paths:
            ff_path = str(path.resolve()).replace("\\", "/").replace("'", r"'\''")
            tf.write(f"file '{ff_path}'\n")

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout or "ffmpeg failed")
    finally:
        try:
            os.unlink(list_file)
        except OSError:
            pass
    return output_path
