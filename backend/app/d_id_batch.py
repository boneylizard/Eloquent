"""Batch WAV → D-ID MP4 segments with bounded concurrency, then ffmpeg concat."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from . import d_id_service

logger = logging.getLogger("d_id_batch")


def natural_sort_wav_paths(paths: List[Path]) -> List[Path]:
    def key(p: Path) -> Tuple:
        name = p.name.lower()
        parts = re.split(r"(\d+)", name)
        keyed = []
        for x in parts:
            if x.isdigit():
                keyed.append((0, int(x)))
            else:
                keyed.append((1, x))
        return tuple(keyed)

    wavs = [Path(p) for p in paths if p.suffix.lower() in (".wav", ".wave")]
    return sorted(wavs, key=key)


async def _one_segment(
    sem: asyncio.Semaphore,
    idx: int,
    wav_path: Path,
    avatar_ref: str,
    seg_out_dir: Path,
    emotion: Optional[str],
    movement: Optional[str],
    background_url: Optional[str],
) -> Tuple[int, Path, Dict[str, Any]]:
    async with sem:
        hint = f"batch_{idx:04d}_{wav_path.stem}"
        mp4, meta = await d_id_service.wav_to_talk_mp4(
            wav_path,
            avatar_ref,
            out_dir=seg_out_dir,
            talk_id_hint=hint,
            emotion=emotion,
            movement=movement,
            background_url=background_url,
        )
        return idx, mp4, meta


async def run_batch_ndjson(
    wav_paths: List[Path],
    avatar_ref: str,
    *,
    concurrency: int = 2,
    emotion: Optional[str] = None,
    movement: Optional[str] = None,
    background_url: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Yields NDJSON lines: {"event": "...", ...}\n
    """
    run_id = uuid.uuid4().hex[:12]
    base = d_id_service.get_d_id_batch_runs_dir() / run_id
    seg_out = base / "segments"
    seg_out.mkdir(parents=True, exist_ok=True)

    ordered = natural_sort_wav_paths(wav_paths)
    if not ordered:
        yield json.dumps({"event": "error", "message": "No WAV files in input."}) + "\n"
        return

    yield json.dumps(
        {
            "event": "started",
            "run_id": run_id,
            "segment_count": len(ordered),
            "concurrency": max(1, min(concurrency, 4)),
        }
    ) + "\n"

    sem = asyncio.Semaphore(max(1, min(int(concurrency), 4)))
    tasks = [
        asyncio.create_task(
            _one_segment(
                sem,
                i,
                p,
                avatar_ref,
                seg_out,
                emotion,
                movement,
                background_url,
            )
        )
        for i, p in enumerate(ordered)
    ]

    mp4_by_index: Dict[int, Path] = {}
    for coro in asyncio.as_completed(tasks):
        try:
            idx, mp4_path, meta = await coro
            mp4_by_index[idx] = mp4_path
            yield json.dumps(
                {
                    "event": "segment_done",
                    "index": idx,
                    "wav": ordered[idx].name,
                    "mp4_path": str(mp4_path),
                    "talk_id": meta.get("id"),
                }
            ) + "\n"
        except Exception as e:
            for t in tasks:
                t.cancel()
            logger.exception("batch segment failed")
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"
            return

    ordered_mp4s = [mp4_by_index[i] for i in range(len(ordered))]
    final_name = f"did_batch_{run_id}_final.mp4"
    final_path = base / final_name
    yield json.dumps({"event": "concat_started", "inputs": [str(p) for p in ordered_mp4s]}) + "\n"
    try:
        d_id_service.concat_mp4_files_ffmpeg(ordered_mp4s, final_path)
    except Exception as e:
        logger.exception("concat failed")
        yield json.dumps({"event": "error", "message": f"Concat failed: {e}"}) + "\n"
        return

    yield json.dumps(
        {
            "event": "complete",
            "run_id": run_id,
            "output_path": str(final_path),
            "segment_mp4s": [str(p) for p in ordered_mp4s],
        }
    ) + "\n"


def write_wav_uploads_to_temp_dir(files: List[Tuple[str, bytes]]) -> Path:
    """files: list of (filename, data). Returns temp dir containing WAVs."""
    tmp = Path(tempfile.mkdtemp(prefix="did_batch_wav_"))
    for name, data in files:
        if not data:
            continue
        p = tmp / Path(name).name
        p.write_bytes(data)
    return tmp
