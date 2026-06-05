#!/usr/bin/env python3
"""
Standalone ASR Audio-to-Text Transcriber
Uses NVIDIA Parakeet (English), Parakeet-ZH (Chinese), or Whisper to transcribe
audio/video files to .txt. Supports MP4, WAV, MP3, and other common formats.

Usage (one-shot — loads model each time if no worker):
  python -m backend.app.asr_transcriber recording.mp4
  python -m backend.app.asr_transcriber a.mp4 b.wav c.m4a
  python -m backend.app.asr_transcriber C:\\recordings\\interviews   REM all audio/video in that folder
  python -m backend.app.asr_transcriber .\\clips -r                  REM folder + subfolders
  python -m backend.app.asr_transcriber video.mp4 -o transcript.txt
  python -m backend.app.asr_transcriber audio.wav --engine whisper

Persistent worker (load Parakeet once; other consoles auto-submit here):
  python -m backend.app.asr_transcriber --serve
  python -m backend.app.asr_transcriber --serve --port 18765 --engine parakeet

If 127.0.0.1:ASR_TRANSCRIBER_PORT (default 18765) accepts connections, one-shot
invocations delegate to the worker instead of reloading the model.

Run from project root with: python -m backend.app.asr_transcriber <args>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

# Ensure we can import from app
_backend_dir = Path(__file__).resolve().parent
_project_root = _backend_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_backend_dir.parent) not in sys.path:
    sys.path.insert(0, str(_backend_dir.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("asr_transcriber")

DEFAULT_WORKER_PORT = int(os.environ.get("ASR_TRANSCRIBER_PORT", "18765"))
WORKER_HOST = "127.0.0.1"

# Video extensions that need ffmpeg extraction
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
# Audio extensions that can be passed through (librosa handles most)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma", ".aac", ".webm", ".opus"}

# Files picked when a path is a directory (non-hidden; use -r for subfolders)
ALL_TRANSCRIBE_SUFFIXES = frozenset(
    e.lower() for e in (VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
)


def _import_transcribe_audio():
    try:
        from app.stt_service import transcribe_audio as ta

        return ta
    except ImportError:
        from backend.app.stt_service import transcribe_audio as ta

        return ta


def expand_inputs_to_files(paths: list[Path], *, recursive: bool) -> list[Path]:
    """
    Each path may be a file (included if it exists) or a directory (all matching
    audio/video extensions in that directory; with recursive=True, subfolders too).
    Order: each argument in order; within a directory, sorted by relative path then name.
    De-duplicated by resolved path.
    """
    out: list[Path] = []
    seen: set[str] = set()

    def add_one(f: Path) -> None:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(f)

    for raw in paths:
        p = raw.expanduser()
        if not p.exists():
            logger.warning("Skipping missing path: %s", p)
            continue
        if p.is_file():
            add_one(p)
            continue
        if p.is_dir():
            if recursive:
                found = [
                    f
                    for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in ALL_TRANSCRIBE_SUFFIXES
                ]
            else:
                found = [
                    f
                    for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in ALL_TRANSCRIBE_SUFFIXES
                ]
            for f in sorted(found, key=lambda x: (str(x.relative_to(p)).lower(), x.name.lower())):
                add_one(f)
            logger.info(
                "Expanded folder %s → %d file(s)%s",
                p,
                len(found),
                " (recursive)" if recursive else "",
            )
            continue
        logger.warning("Not a file or directory: %s", p)

    return out


def _import_stt_loaders():
    try:
        from app import stt_service as stt

        return stt
    except ImportError:
        from backend.app import stt_service as stt

        return stt


def ensure_audio_for_transcription(input_path: Path) -> Path:
    """
    If input is video (mp4, mkv, etc.), extract audio to a temp WAV using ffmpeg.
    Otherwise return the path as-is (librosa/audioread will handle it).
    """
    ext = input_path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return input_path

    from .ffmpeg_utils import FFMPEG_INSTALL_HINT, find_ffmpeg
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is required to transcribe video files (MP4, MKV, etc.). "
            + FFMPEG_INSTALL_HINT
        )

    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    temp_path = Path(temp_wav)

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(temp_path),
    ]

    logger.info("Extracting audio from %s...", input_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500] if result.stderr else result.stdout}")

    logger.info("Audio extracted successfully")
    return temp_path


async def prewarm_engine(engine: str) -> None:
    """Load STT model once in this process (cached in stt_service globals)."""
    stt = _import_stt_loaders()
    loop = asyncio.get_event_loop()
    if engine == "whisper":
        await loop.run_in_executor(None, stt.load_whisper_model)
    elif engine == "parakeet":
        await loop.run_in_executor(None, stt.load_parakeet_model)
    elif engine == "parakeet-v3":
        await loop.run_in_executor(None, stt.load_parakeet_v3_model)
    elif engine == "parakeet-zh":
        await loop.run_in_executor(None, stt.load_parakeet_zh_model)


async def transcribe_file_to_disk(
    input_path: Path,
    engine: str,
    output_path: Path,
) -> Tuple[bool, str]:
    """Returns (ok, message)."""
    if not input_path.exists():
        return False, f"Input file not found: {input_path}"

    ext = input_path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
        logger.warning("Unusual extension %s; attempting anyway", ext)

    temp_audio_path: Optional[Path] = None
    try:
        temp_audio_path = ensure_audio_for_transcription(input_path)
        transcribe_audio = _import_transcribe_audio()
        logger.info("Transcribing %s with %s...", input_path.name, engine)
        transcript = await transcribe_audio(str(temp_audio_path), engine=engine)

        if not transcript or not transcript.strip():
            logger.warning("Transcription returned empty text for %s", input_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript.strip(), encoding="utf-8")
        msg = f"Saved: {output_path} ({len(transcript):,} chars)"
        logger.info(msg)
        return True, msg
    except Exception as e:
        err = f"{input_path}: {e}"
        logger.error("Transcription failed: %s", err, exc_info=True)
        return False, err
    finally:
        if temp_audio_path is not None and temp_audio_path != input_path:
            try:
                temp_audio_path.unlink(missing_ok=True)
            except Exception:
                pass


def default_output_path(input_path: Path) -> Path:
    return input_path.with_stem(input_path.stem + "_transcript").with_suffix(".txt")


async def try_worker_transcribe(
    port: int,
    input_path: Path,
    output_path: Optional[Path],
    engine: str,
    connect_timeout: float = 0.75,
) -> Optional[Tuple[bool, str]]:
    """
    If a worker is listening, send one job and return (ok, message).
    If nothing is listening, return None so caller can run locally.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(WORKER_HOST, port),
            timeout=connect_timeout,
        )
    except (OSError, asyncio.TimeoutError):
        return None

    try:
        payload = {
            "input": str(input_path.resolve()),
            "output": str(output_path.resolve()) if output_path else None,
            "engine": engine,
        }
        writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=7200.0)
        if not line:
            return False, "Worker closed connection without response"
        data = json.loads(line.decode("utf-8", errors="replace"))
        if data.get("ok"):
            return True, data.get("message", "ok")
        return False, data.get("error", "worker error")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def run_worker(port: int, engine: str) -> int:
    """TCP server: one loaded model, jobs serialized with a lock."""
    job_lock = asyncio.Lock()

    async def process_job(payload: dict[str, Any]) -> dict[str, Any]:
        inp = Path(payload["input"])
        eng = str(payload.get("engine") or engine)
        out_raw = payload.get("output")
        out_path = Path(out_raw) if out_raw else default_output_path(inp)

        async with job_lock:
            ok, msg = await transcribe_file_to_disk(inp, eng, out_path)
        if ok:
            return {"ok": True, "message": msg, "saved": str(out_path)}
        return {"ok": False, "error": msg}

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as e:
                    writer.write(
                        (json.dumps({"ok": False, "error": f"invalid json: {e}"}, ensure_ascii=False) + "\n").encode(
                            "utf-8"
                        )
                    )
                    await writer.drain()
                    continue
                if "input" not in payload:
                    writer.write(
                        (json.dumps({"ok": False, "error": "missing input"}, ensure_ascii=False) + "\n").encode("utf-8")
                    )
                    await writer.drain()
                    continue
                result = await process_job(payload)
                writer.write((json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
        except Exception as e:
            logger.error("Client %s handler error: %s", addr, e, exc_info=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    logger.info("Loading %s model (once)...", engine)
    await prewarm_engine(engine)
    server = await asyncio.start_server(handle_client, WORKER_HOST, port)
    sockets = server.sockets
    if sockets:
        for s in sockets:
            logger.info("ASR worker listening on %s (engine=%s)", s.getsockname(), engine)
    else:
        logger.error("Server failed to bind")
        return 1

    try:
        async with server:
            await server.serve_forever()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        logger.info("Worker stopped (keyboard interrupt).")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video to text using Parakeet or Whisper (multi-file and optional local worker)."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[],
        help="Audio/video files and/or folders (each folder: all matching files; model loads once for the batch).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="When an argument is a folder, include matching files in subfolders too.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .txt path (only when exactly one input file is given).",
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=["parakeet", "parakeet-v3", "parakeet-zh", "whisper"],
        default="parakeet",
        help="STT engine (default: parakeet)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run a persistent worker on --port (keeps model loaded; other invocations auto-delegate).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WORKER_PORT,
        help=f"Worker TCP port (default {DEFAULT_WORKER_PORT}, or env ASR_TRANSCRIBER_PORT).",
    )
    parser.add_argument(
        "--no-worker",
        action="store_true",
        help="Do not try to connect to a local worker (always load model in this process).",
    )
    args = parser.parse_args()

    if args.serve:
        return await run_worker(args.port, args.engine)

    if not args.inputs:
        parser.error("Provide at least one input file or folder, or use --serve to start a worker.")

    file_list = expand_inputs_to_files(args.inputs, recursive=args.recursive)
    if not file_list:
        logger.error("No audio/video files found (check paths and extensions).")
        return 1

    if len(file_list) > 1 and args.output is not None:
        logger.error("-o/--output may only be used when exactly one input file is produced (not with multiple files or folders).")
        return 1

    # Try local worker first (unless disabled): same model reload avoided when worker is up.
    if not args.no_worker:
        all_via_worker = True
        results: list[tuple[Path, bool, str]] = []
        for inp in file_list:
            out = args.output if len(file_list) == 1 else None
            out_path = out if out is not None else default_output_path(inp)
            delegated = await try_worker_transcribe(args.port, inp, out_path if len(file_list) == 1 else None, args.engine)
            if delegated is None:
                all_via_worker = False
                break
            ok, msg = delegated
            results.append((inp, ok, msg))
        if all_via_worker:
            exit_code = 0
            for _, ok, msg in results:
                if not ok:
                    exit_code = 1
                    logger.error("%s", msg)
                else:
                    logger.info("%s", msg)
            return exit_code

    await prewarm_engine(args.engine)

    exit_code = 0
    for input_path in file_list:
        if len(file_list) == 1 and args.output is not None:
            out_path = args.output
        else:
            out_path = default_output_path(input_path)
        ok, msg = await transcribe_file_to_disk(input_path, args.engine, out_path)
        if not ok:
            exit_code = 1
            logger.error("%s", msg)
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
