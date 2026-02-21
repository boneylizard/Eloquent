#!/usr/bin/env python3
"""
Standalone ASR Audio-to-Text Transcriber
Uses Eloquent's NVIDIA Parakeet (or Whisper) STT engine to transcribe
audio/video files to .txt. Supports MP4, WAV, MP3, and other common formats.

Usage:
  python -m backend.app.asr_transcriber recording.mp4
  python -m backend.app.asr_transcriber video.mp4 -o transcript.txt
  python -m backend.app.asr_transcriber audio.wav --engine whisper

Run from project root with: python -m backend.app.asr_transcriber <audio_or_video_file>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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

# Video extensions that need ffmpeg extraction
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
# Audio extensions that can be passed through (librosa handles most)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma", ".aac", ".webm"}


def ensure_audio_for_transcription(input_path: Path) -> Path:
    """
    If input is video (mp4, mkv, etc.), extract audio to a temp WAV using ffmpeg.
    Otherwise return the path as-is (librosa/audioread will handle it).
    """
    ext = input_path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return input_path

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is required to transcribe video files (MP4, MKV, etc.). "
            "Install it and ensure it's on PATH, or use an audio-only file."
        )

    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    temp_path = Path(temp_wav)

    cmd = [
        ffmpeg_path,
        "-y",  # Overwrite output
        "-i", str(input_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(temp_path),
    ]

    logger.info(f"Extracting audio from {input_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500] if result.stderr else result.stdout}")

    logger.info("Audio extracted successfully")
    return temp_path


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video to text using Parakeet or Whisper"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to audio or video file (e.g. .mp4, .wav, .mp3)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .txt path (default: input_name_transcript.txt)",
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["parakeet", "whisper"],
        default="parakeet",
        help="STT engine (default: parakeet)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    ext = args.input.suffix.lower()
    if ext not in VIDEO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
        logger.warning(f"Unusual extension {ext}; attempting anyway")

    # Output path
    out_path = args.output
    if out_path is None:
        out_path = args.input.with_stem(args.input.stem + "_transcript").with_suffix(".txt")

    temp_audio_path = None
    try:
        temp_audio_path = ensure_audio_for_transcription(args.input)
        is_temp = temp_audio_path != args.input

        # Import STT here so errors are user-friendly after arg parse
        try:
            from app.stt_service import transcribe_audio
        except ImportError:
            try:
                from backend.app.stt_service import transcribe_audio
            except ImportError:
                logger.error(
                    "Could not import stt_service. Run from project root:\n"
                    "  python -m backend.app.asr_transcriber <audio_or_video_file>"
                )
                return 1

        logger.info(f"Transcribing with {args.engine}...")
        transcript = await transcribe_audio(str(temp_audio_path), engine=args.engine)

        if not transcript or not transcript.strip():
            logger.warning("Transcription returned empty text")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(transcript.strip(), encoding="utf-8")
        logger.info(f"Saved: {out_path} ({len(transcript):,} chars)")
        return 0

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return 1

    finally:
        if temp_audio_path and temp_audio_path != args.input:
            try:
                temp_audio_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
