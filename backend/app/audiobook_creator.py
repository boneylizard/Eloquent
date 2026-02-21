#!/usr/bin/env python3
"""
Standalone Audiobook Creator
Uses Eloquent's Chatterbox (Faster) or Chatterbox Turbo TTS engine with your selected voice clone
to convert .txt files into audiobook WAV files.

Usage:
  python -m app.audiobook_creator book.txt
  python -m app.audiobook_creator book.txt -o audiobook.wav
  python -m app.audiobook_creator book.txt --voice MyVoice.wav --engine chatterbox

Run from project root with: python -m backend.app.audiobook_creator <txt_file>
Or from backend/: python -m app.audiobook_creator <txt_file>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
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
logger = logging.getLogger("audiobook")


def load_voice_from_settings(engine: str) -> tuple[str | None, str]:
    """Load voice and engine from ~/.LiangLocal/settings.json.
    Returns (audio_prompt_path or None for default, engine)."""
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    if not settings_path.exists():
        return None, engine

    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read settings: {e}")
        return None, engine

    # Prefer explicit ttsVoice/ttsEngine if present
    tts_voice = settings.get("ttsVoice") or settings.get("tts_voice")
    tts_engine = settings.get("ttsEngine") or settings.get("tts_engine", engine)

    if tts_voice and tts_voice.lower() not in ("default", "af_heart"):
        voices_dir = _backend_dir / "static" / "voice_references"
        voice_path = voices_dir / tts_voice
        if voice_path.exists():
            return str(voice_path), tts_engine
        if (voices_dir / Path(tts_voice).name).exists():
            return str(voices_dir / Path(tts_voice).name), tts_engine

    # Fallback: most recent from voice_cache
    voice_cache = settings.get("voice_cache", [])
    for entry in reversed(voice_cache):
        vid = entry.get("voice_id")
        eng = entry.get("engine", "chatterbox")
        if vid and vid.lower() != "default":
            voices_dir = _backend_dir / "static" / "voice_references"
            voice_path = voices_dir / vid
            if voice_path.exists():
                return str(voice_path), eng

    return None, tts_engine


def read_text(path: Path) -> str:
    """Read and normalize text from file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Normalize line breaks, collapse excessive whitespace
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create audiobooks from .txt files using Chatterbox TTS"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to .txt file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output WAV path (default: input_name_audiobook.wav)",
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default=None,
        help="Voice reference: filename in voice_references, or path to .wav/.mp3 (default: from settings)",
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["chatterbox", "chatterbox_turbo"],
        default="chatterbox",
        help="TTS engine: chatterbox (Faster) or chatterbox_turbo (default: chatterbox)",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Use built-in default voice instead of voice clone",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    if args.input.suffix.lower() != ".txt":
        logger.warning(f"Expected .txt file, got {args.input.suffix}")

    # Resolve voice
    audio_prompt_path = None
    engine = args.engine
    if args.no_voice:
        audio_prompt_path = None
    elif args.voice:
        voice_name = Path(args.voice).name
        candidates = [
            Path(args.voice),  # Absolute or cwd-relative path
            Path.cwd() / voice_name,  # Current working directory
            args.input.parent / voice_name,  # Same directory as input file
            _backend_dir / "static" / "voice_references" / voice_name,  # Eloquent voice refs
        ]
        # Also try case-insensitive match in voice_references (e.g. Lizzo.wav vs lizzo.wav)
        voices_dir = _backend_dir / "static" / "voice_references"
        if voices_dir.exists():
            for f in voices_dir.iterdir():
                if f.is_file() and f.name.lower() == voice_name.lower():
                    candidates.append(f)
                    break
        audio_prompt_path = None
        for c in candidates:
            if c.exists():
                audio_prompt_path = str(c.resolve())
                logger.info(f"Using voice: {c.name}")
                break
        if not audio_prompt_path:
            logger.error(
                f"Voice file not found: {args.voice}\n"
                f"  Searched: cwd, input dir, {voices_dir}"
            )
            return 1
    else:
        audio_prompt_path, engine = load_voice_from_settings(args.engine)
        if audio_prompt_path:
            logger.info(f"Using voice from settings: {Path(audio_prompt_path).name}")
        else:
            logger.info("Using default built-in voice")

    logger.info(f"Engine: {engine}")

    # Output path
    out_path = args.output
    if out_path is None:
        out_path = args.input.with_stem(args.input.stem + "_audiobook").with_suffix(".wav")

    # Read text
    text = read_text(args.input)
    if not text:
        logger.error("Input file is empty")
        return 1

    logger.info(f"Loaded {len(text):,} characters from {args.input.name}")

    # Import TTS here so errors are user-friendly after arg parse
    try:
        from app.tts_service import synthesize_speech
    except ImportError:
        try:
            from backend.app.tts_service import synthesize_speech
        except ImportError:
            logger.error(
                "Could not import tts_service. Run from project root:\n"
                "  python -m backend.app.audiobook_creator <txt_file>"
            )
            return 1

    # Synthesize (tts_service handles chunking for long text)
    logger.info("Synthesizing... (this may take a while for long books)")
    try:
        audio_bytes = await synthesize_speech(
            text=text,
            voice="default" if not audio_prompt_path else "custom",
            engine=engine,
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.5,
            cfg=0.5,
        )
    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        return 1

    if not audio_bytes:
        logger.error("No audio generated")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    logger.info(f"Saved: {out_path} ({len(audio_bytes):,} bytes)")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
