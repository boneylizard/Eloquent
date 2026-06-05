"""
Voice reference merge pipeline: optional clean → timbre morph → optional RVC → normalize → voice_references/.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
from urllib.parse import unquote

from .ffmpeg_utils import FFMPEG_INSTALL_HINT, find_ffmpeg, resolve_ffmpeg_bin

logger = logging.getLogger(__name__)

_BACKEND_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_APP_DIR.parent.parent
_DEFAULT_WORK_DIR = _BACKEND_APP_DIR / "data" / "voice_sculpt"
_VOICE_REFERENCES_DIR = _BACKEND_APP_DIR / "static" / "voice_references"
_DEFAULT_APPLIO_DIR = _PROJECT_ROOT / "tools" / "Applio"
_SCULPT_ENV_BAT = _PROJECT_ROOT / "sculpt.env.bat"
_APPLIO_GIT_URL = "https://github.com/IAHispano/Applio.git"

_HF_HOST_RE = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/(?P<repo>[^/]+/[^/]+)"
    r"(?:/(?:tree|resolve|blob)/(?P<rev>[^/]+)(?:/(?P<file>.+))?)?/?$",
    re.IGNORECASE,
)
_HF_SKIP_PATH_PARTS = (
    "pretrained", "pretrains", "embedder", "hubert", "rmvpe", "contentvec",
    "pytorch_model", "hifi-gan", "refinegan", "/d/", "/g/", "discriminator",
)

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm", ".aac", ".mp4", ".mkv"}
_YOUTUBE_URL_RE = re.compile(
    r"^https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)",
    re.IGNORECASE,
)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


def parse_sculpt_sources(*, source: str = "", sources: Optional[list[str]] = None) -> list[str]:
    """One or many local audio paths (newline, pipe, or explicit list)."""
    raw: list[str] = []
    if sources:
        raw.extend(sources)
    text = (source or "").strip()
    if text:
        for part in text.replace("|", "\n").splitlines():
            cleaned = part.strip().strip('"').strip("'")
            if cleaned:
                raw.append(cleaned)

    ordered: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = str(Path(item).expanduser().resolve()).lower() if Path(item).expanduser().exists() else item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    if not ordered:
        raise ValueError("At least one source audio file is required")
    return ordered


def _ffmpeg_combine_filter(num_inputs: int, mode: str) -> str:
    if num_inputs < 1:
        raise ValueError("combine requires at least one input")
    if num_inputs == 1:
        return "[0:a]aformat=sample_rates=44100:channel_layouts=mono[out]"
    parts: list[str] = []
    labels: list[str] = []
    for i in range(num_inputs):
        label = f"a{i}"
        parts.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=mono[{label}]")
        labels.append(f"[{label}]")
    if mode == "mix":
        joined = "".join(labels)
        weights = " ".join(["0.5"] * num_inputs) if num_inputs == 2 else " ".join(
            [f"{1.0 / num_inputs:.4f}" for _ in range(num_inputs)]
        )
        parts.append(
            f"{joined}amix=inputs={num_inputs}:duration=longest:dropout_transition=0"
            f":normalize=0:weights={weights}[out]"
        )
    else:
        joined = "".join(labels)
        parts.append(f"{joined}concat=n={num_inputs}:v=0:a=1[out]")
    return ";".join(parts)


@dataclass
class MissingToolError:
    detail: str
    missing_tool: str
    env_var: str
    install_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "missing_tool": self.missing_tool,
            "env_var": self.env_var,
            "install_hint": self.install_hint,
            "status": 412,
        }


@dataclass
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass
class SculptRequest:
    source: str = ""
    sources: Optional[list[str]] = None
    source_type: str = "local_path"  # local_path | youtube_url
    output_name: Optional[str] = None
    accent_model: str = "default"  # default | target_accent | model name
    skip_rvc: bool = True  # merge output is the product; RVC is optional polish
    skip_uvr: bool = True  # inputs are usually pre-clipped clean references
    combine_mode: str = "morph"  # morph | mix | concat
    morph_balance: float = 0.5  # two-source morph: 0=first only, 1=second only
    source_weights: Optional[list[float]] = None
    pitch: int = 0
    index_rate: Optional[float] = None  # None = auto from index presence
    protect: float = 0.33
    volume_envelope: float = 1.0
    voice_prompt: Optional[str] = None  # saved as sidecar; engines may ignore today


class GPUQueue:
    """Ensures only one GPU subprocess (UVR or Applio) runs at a time."""

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(1)

    async def run(self, coro_factory: Callable[[], Awaitable[SubprocessResult]]) -> SubprocessResult:
        async with self._sem:
            return await coro_factory()


@dataclass
class PipelineTask:
    """Runs the sculpt chain for a single job."""

    job_id: str
    request: SculptRequest
    work_root: Path
    config: "AutomationConfig"
    gpu_queue: GPUQueue
    run_subprocess: Callable[..., Awaitable[SubprocessResult]]
    emit: ProgressCallback

    input_dir: Path = field(init=False)
    vocals_dir: Path = field(init=False)
    converted_dir: Path = field(init=False)
    final_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        base = self.work_root / self.job_id
        self.input_dir = base / "00_input"
        self.vocals_dir = base / "01_vocals"
        self.converted_dir = base / "02_converted"
        self.final_dir = base / "03_final"
        for d in (self.input_dir, self.vocals_dir, self.converted_dir, self.final_dir):
            d.mkdir(parents=True, exist_ok=True)

    async def _progress(self, step: int, total: int, phase: str, message: str) -> None:
        payload = {
            "type": "progress",
            "step": step,
            "total": total,
            "phase": phase,
            "message": message,
        }
        result = self.emit(payload)
        if asyncio.iscoroutine(result):
            await result

    async def run(self) -> Path:
        total_steps = 3
        input_paths = await self._resolve_inputs()
        clip_count = len(input_paths)

        if self.request.skip_uvr:
            stem_paths = list(input_paths)
            await self._progress(
                1, total_steps, "uvr",
                f"Step 1/3: Skipping UVR ({clip_count} clip(s) treated as clean references)…",
            )
        else:
            await self._progress(
                1, total_steps, "uvr",
                f"Step 1/3: Isolating vocals ({clip_count} clip(s))…",
            )
            stem_paths = []
            for idx, inp in enumerate(input_paths, start=1):
                if clip_count > 1:
                    await self._progress(
                        1, total_steps, "uvr",
                        f"Step 1/3: Isolating clip {idx}/{clip_count}…",
                    )
                clip_vocals_dir = self.vocals_dir / f"clip_{idx:03d}"
                clip_vocals_dir.mkdir(parents=True, exist_ok=True)
                stem_paths.append(await self._run_uvr(inp, clip_vocals_dir))

        if len(stem_paths) == 1:
            merged_stem = stem_paths[0]
        else:
            mode = (self.request.combine_mode or "morph").lower()
            if mode not in ("concat", "mix", "morph"):
                mode = "morph"
            if mode == "morph":
                verb = f"Merging timbre from {clip_count} voices (face-morph style)"
            elif mode == "mix":
                verb = f"Overlaying {clip_count} vocal stems"
            else:
                verb = f"Joining {clip_count} clips"
            await self._progress(
                2, total_steps, "merge",
                f"Step 2/3: {verb}…",
            )
            merged_stem = self.vocals_dir / "merged_stem.wav"
            await self._merge_stems(stem_paths, merged_stem, mode)

        if self.request.skip_rvc:
            await self._progress(
                2, total_steps, "rvc",
                "Step 2/3: Skipping optional RVC polish…",
            )
            shaped_path = merged_stem
        else:
            if clip_count > 1:
                await self._progress(
                    2, total_steps, "rvc",
                    f"Step 2/3: One RVC pass on the blended stem ({clip_count} sources)…",
                )
            else:
                await self._progress(2, total_steps, "rvc", "Step 2/3: Converting timbre…")
            shaped_path = await self._run_applio(
                merged_stem,
                output_path=self.converted_dir / "converted.wav",
            )

        await self._progress(3, total_steps, "normalize", "Step 3/3: Normalizing audio…")
        final_path = self.final_dir / "reference.wav"
        await self._run_normalize(shaped_path, final_path)

        voice_id = self._publish(final_path)
        self._last_voice_id = voice_id
        self._write_voice_prompt_sidecar(voice_id)
        return _VOICE_REFERENCES_DIR / voice_id

    @property
    def last_voice_id(self) -> str:
        return getattr(self, "_last_voice_id", "")

    async def _resolve_inputs(self) -> list[Path]:
        if self.request.source_type == "youtube_url":
            single = await self._resolve_youtube_input((self.request.source or "").strip())
            return [single]

        paths: list[Path] = []
        for raw in parse_sculpt_sources(source=self.request.source, sources=self.request.sources):
            local = Path(raw).expanduser()
            if local.is_dir():
                dir_files = sorted(
                    (p for p in local.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS),
                    key=lambda p: p.name.lower(),
                )
                if not dir_files:
                    raise ValueError(f"No audio files found in directory: {raw}")
                for f in dir_files:
                    paths.append(await self._stage_input_file(f))
                continue
            if not local.is_file():
                raise ValueError(f"Local file not found: {raw}")
            if local.suffix.lower() not in _AUDIO_EXTENSIONS:
                raise ValueError(f"Unsupported audio extension: {local.suffix}")
            paths.append(await self._stage_input_file(local))
        return paths

    async def _stage_input_file(self, local: Path) -> Path:
        dest = self.input_dir / local.name
        if dest.exists():
            dest = self.input_dir / f"{local.stem}_{uuid.uuid4().hex[:6]}{local.suffix.lower()}"
        if local.resolve() != dest.resolve():
            shutil.copy2(local, dest)
        return dest

    async def _resolve_youtube_input(self, source: str) -> Path:
        if not source:
            raise ValueError("source is required")
        if not _YOUTUBE_URL_RE.match(source):
            raise ValueError("Invalid YouTube URL")
        missing = _check_binary(
            self.config.yt_dlp_bin, "yt-dlp", "YT_DLP_BIN",
            "Install yt-dlp: https://github.com/yt-dlp/yt-dlp",
        )
        if missing:
            raise PreconditionError(missing)
        out_template = str(self.input_dir / "%(title)s.%(ext)s")
        cmd = [
            self.config.yt_dlp_bin,
            "--no-playlist",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", out_template,
            source,
        ]
        await self.run_subprocess(cmd, cwd=None, env=None, gpu=False, label="yt-dlp")
        candidates = sorted(self.input_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in candidates:
            if c.is_file() and c.suffix.lower() in _AUDIO_EXTENSIONS:
                return c
        raise RuntimeError("yt-dlp completed but no audio file was found in staging directory")

    async def _merge_stems(self, stems: list[Path], output_path: Path, mode: str) -> None:
        if mode == "morph":
            from .voice_morph import morph_voice_files, weights_from_balance

            if self.request.source_weights:
                weights = self.request.source_weights
            else:
                weights = weights_from_balance(len(stems), self.request.morph_balance)

            def _run_morph() -> None:
                morph_voice_files(stems, output_path, weights=weights)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_morph)
            if not output_path.is_file():
                raise RuntimeError("voice morph completed but output file was not created")
            return

        await self._combine_stems(stems, output_path, mode)

    async def _combine_stems(self, stems: list[Path], output_path: Path, mode: str) -> None:
        missing = _check_binary(
            self.config.ffmpeg_bin,
            "ffmpeg",
            "FFMPEG_BIN",
            FFMPEG_INSTALL_HINT,
        )
        if missing:
            raise PreconditionError(missing)

        filter_graph = _ffmpeg_combine_filter(len(stems), mode)
        cmd = [self.config.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
        for stem in stems:
            cmd.extend(["-i", str(stem)])
        cmd.extend([
            "-filter_complex", filter_graph,
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            str(output_path),
        ])
        result = await self.run_subprocess(cmd, cwd=None, env=None, gpu=False, label="ffmpeg-combine")
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg combine failed: {_tail(result.stderr)}")
        if not output_path.is_file():
            raise RuntimeError("ffmpeg combine completed but output file was not created")

    async def _run_uvr(self, input_audio: Path, output_dir: Path) -> Path:
        missing = _check_binary(
            self.config.audio_separator_bin,
            "audio-separator",
            "AUDIO_SEPARATOR_BIN",
            "pip install audio-separator — https://github.com/nomadkaraoke/python-audio-separator",
        )
        if missing:
            raise PreconditionError(missing)

        self.config.audio_separator_model_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.config.audio_separator_bin,
            str(input_audio),
            "--model_filename", self.config.uvr_model_filename,
            "--output_dir", str(output_dir),
            "--output_format", "WAV",
            "--model_file_dir", str(self.config.audio_separator_model_dir),
            "--single_stem", "Vocals",
            "--sample_rate", "44100",
        ]

        async def _uvr() -> SubprocessResult:
            return await self.run_subprocess(cmd, cwd=None, env=None, gpu=True, label="uvr")

        await self.gpu_queue.run(_uvr)
        return _find_newest_wav(output_dir, prefer_vocals=True)

    async def _run_applio(self, vocals_path: Path, *, output_path: Optional[Path] = None) -> Path:
        errors = self.config.validate_applio(self.request.accent_model)
        if errors:
            raise PreconditionError(errors[0])

        pth_path, index_path = self.config.resolve_applio_model(self.request.accent_model)
        has_index = index_path.is_file()
        if self.request.index_rate is not None:
            index_rate = float(self.request.index_rate)
        else:
            index_rate = 0.75 if has_index else 0.0
        index_rate = max(0.0, min(1.0, index_rate))
        pitch = int(self.request.pitch)
        protect = max(0.0, min(0.5, float(self.request.protect)))
        volume_envelope = max(0.0, min(1.0, float(self.request.volume_envelope)))
        output_path = output_path or (self.converted_dir / "converted.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.config.applio_python,
            "core.py",
            "infer",
            "--input_path", str(vocals_path),
            "--output_path", str(output_path),
            "--pth_path", str(pth_path),
            "--index_path", str(index_path) if has_index else "",
            "--export_format", "WAV",
            "--f0_method", "rmvpe",
            "--pitch", str(pitch),
            "--index_rate", str(index_rate),
            "--protect", str(protect),
            "--volume_envelope", str(volume_envelope),
            "--split_audio", "true",
            "--clean_audio", "false",
            "--embedder_model", "contentvec",
        ]
        if not has_index:
            logger.warning("No .index for %s — running Applio with index_rate=0", pth_path.name)

        async def _rvc() -> SubprocessResult:
            return await self.run_subprocess(
                cmd,
                cwd=str(self.config.applio_root),
                env=None,
                gpu=True,
                label="applio",
            )

        result = await self.gpu_queue.run(_rvc)
        if result.returncode != 0:
            raise RuntimeError(f"Applio inference failed: {_tail(result.stderr)}")

        if output_path.is_file():
            return output_path
        return _find_newest_wav(self.converted_dir)

    async def _run_normalize(self, input_path: Path, output_path: Path) -> None:
        missing = _check_binary(
            self.config.ffmpeg_bin,
            "ffmpeg",
            "FFMPEG_BIN",
            FFMPEG_INSTALL_HINT,
        )
        if missing:
            raise PreconditionError(missing)

        cmd = [
            self.config.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-af", "loudnorm=I=-16:TP=-3:LRA=11",
            "-ar", "44100",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
        result = await self.run_subprocess(cmd, cwd=None, env=None, gpu=False, label="ffmpeg")
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg normalization failed: {_tail(result.stderr)}")
        if not output_path.is_file():
            raise RuntimeError("ffmpeg completed but output file was not created")

    def _publish(self, final_path: Path) -> str:
        _VOICE_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        output_name = self.request.output_name or final_path.stem
        clean_name = _sanitize_voice_name(output_name)
        if not clean_name:
            clean_name = "sculpted_voice"
        voice_id = f"{clean_name}.wav"
        save_path = _VOICE_REFERENCES_DIR / voice_id
        counter = 1
        while save_path.exists():
            voice_id = f"{clean_name}_{counter}.wav"
            save_path = _VOICE_REFERENCES_DIR / voice_id
            counter += 1
        shutil.copy2(final_path, save_path)
        logger.info("Voice reference published: %s", save_path)
        return voice_id

    def _write_voice_prompt_sidecar(self, voice_id: str) -> None:
        prompt = (self.request.voice_prompt or "").strip()
        if not prompt:
            return
        sidecar = _VOICE_REFERENCES_DIR / f"{Path(voice_id).stem}.prompt.txt"
        sidecar.write_text(prompt + "\n", encoding="utf-8")
        logger.info("Wrote voice prompt sidecar: %s", sidecar)


class PreconditionError(Exception):
    def __init__(self, missing: MissingToolError) -> None:
        self.missing = missing
        super().__init__(missing.detail)


@dataclass
class AutomationConfig:
    work_dir: Path
    max_concurrent: int
    job_timeout_sec: float
    gpu_id: Optional[str]
    yt_dlp_bin: str
    audio_separator_bin: str
    audio_separator_model_dir: Path
    uvr_model_filename: str
    applio_root: Optional[Path]
    applio_python: Optional[str]
    applio_models_dir: Optional[Path]
    applio_default_pth: Optional[Path]
    applio_default_index: Optional[Path]
    applio_accent_pth: Optional[Path]
    applio_accent_index: Optional[Path]
    ffmpeg_bin: str

    @classmethod
    def from_env(cls, *, auto_discover: bool = True) -> "AutomationConfig":
        discovered = discover_environment() if auto_discover else {}
        work_dir = Path(os.getenv("VOICE_SCULPT_WORK_DIR", str(_DEFAULT_WORK_DIR)))
        applio_root_raw = (os.getenv("APPLIO_ROOT") or "").strip()
        applio_root = Path(applio_root_raw) if applio_root_raw else discovered.get("applio_root")
        if applio_root and not isinstance(applio_root, Path):
            applio_root = Path(applio_root)

        applio_python = (os.getenv("APPLIO_PYTHON") or "").strip() or None
        if not applio_python and applio_root:
            applio_python = _applio_python_for_root(applio_root)
        if not applio_python and discovered.get("applio_python"):
            applio_python = discovered["applio_python"]

        models_dir_raw = (os.getenv("APPLIO_MODELS_DIR") or "").strip()
        if models_dir_raw:
            applio_models_dir = Path(models_dir_raw)
        elif applio_root:
            applio_models_dir = applio_root / "logs"
        else:
            applio_models_dir = None

        def _path_env(key: str) -> Optional[Path]:
            val = (os.getenv(key) or "").strip()
            return Path(val) if val else None

        applio_default_pth = _path_env("APPLIO_DEFAULT_PTH") or discovered.get("applio_default_pth")
        applio_default_index = _path_env("APPLIO_DEFAULT_INDEX") or discovered.get("applio_default_index")
        if isinstance(applio_default_pth, str):
            applio_default_pth = Path(applio_default_pth)
        if isinstance(applio_default_index, str):
            applio_default_index = Path(applio_default_index)

        if applio_root and applio_default_pth:
            if not _is_voice_model_pth_file(Path(applio_default_pth), applio_root=applio_root):
                logger.warning("Ignoring invalid APPLIO_DEFAULT_PTH: %s", applio_default_pth)
                fallback_pth = discovered.get("applio_default_pth")
                fallback_index = discovered.get("applio_default_index")
                applio_default_pth = Path(fallback_pth) if fallback_pth else None
                applio_default_index = Path(fallback_index) if fallback_index else None

        audio_separator_bin = (os.getenv("AUDIO_SEPARATOR_BIN") or "").strip()
        if not audio_separator_bin:
            audio_separator_bin = discovered.get("audio_separator_bin") or _resolve_bin("AUDIO_SEPARATOR_BIN", "audio-separator")

        ffmpeg_bin = (os.getenv("FFMPEG_BIN") or "").strip()
        if not ffmpeg_bin:
            ffmpeg_bin = discovered.get("ffmpeg_bin") or resolve_ffmpeg_bin()

        model_dir_raw = (os.getenv("AUDIO_SEPARATOR_MODEL_DIR") or "").strip()
        model_dir = Path(model_dir_raw) if model_dir_raw else work_dir / "models" / "uvr"

        return cls(
            work_dir=work_dir,
            max_concurrent=max(1, int(os.getenv("VOICE_SCULPT_MAX_CONCURRENT", "1"))),
            job_timeout_sec=float(os.getenv("VOICE_SCULPT_JOB_TIMEOUT_SEC", "3600")),
            gpu_id=(os.getenv("VOICE_SCULPT_GPU_ID") or "").strip() or None,
            yt_dlp_bin=_resolve_bin("YT_DLP_BIN", "yt-dlp"),
            audio_separator_bin=audio_separator_bin,
            audio_separator_model_dir=model_dir,
            uvr_model_filename=os.getenv("UVR_MODEL_FILENAME", "UVR-MDX-NET-Voc_FT.onnx"),
            applio_root=applio_root,
            applio_python=applio_python,
            applio_models_dir=applio_models_dir,
            applio_default_pth=applio_default_pth,
            applio_default_index=applio_default_index,
            applio_accent_pth=_path_env("APPLIO_ACCENT_PTH"),
            applio_accent_index=_path_env("APPLIO_ACCENT_INDEX"),
            ffmpeg_bin=ffmpeg_bin,
        )

    def validate_applio(self, accent_model: str) -> list[MissingToolError]:
        errors: list[MissingToolError] = []
        if not self.applio_root or not self.applio_root.is_dir():
            errors.append(MissingToolError(
                detail="APPLIO_ROOT is not set or not a directory",
                missing_tool="applio",
                env_var="APPLIO_ROOT",
                install_hint="Clone/install Applio: https://github.com/IAHispano/Applio",
            ))
            return errors

        core_py = self.applio_root / "core.py"
        if not core_py.is_file():
            errors.append(MissingToolError(
                detail="Applio core.py not found under APPLIO_ROOT",
                missing_tool="applio",
                env_var="APPLIO_ROOT",
                install_hint=f"Expected core.py at {core_py}",
            ))

        if not self.applio_python or not Path(self.applio_python).is_file():
            errors.append(MissingToolError(
                detail="Applio Python interpreter not found",
                missing_tool="applio-python",
                env_var="APPLIO_PYTHON",
                install_hint="Set APPLIO_PYTHON to env\\Scripts\\python.exe inside Applio (run run-install.bat first)",
            ))

        pth, index = self.resolve_applio_model(accent_model)
        if not pth.is_file():
            key = "APPLIO_ACCENT_PTH" if accent_model == "target_accent" else "APPLIO_DEFAULT_PTH"
            errors.append(MissingToolError(
                detail=f"RVC model .pth not found: {pth}",
                missing_tool="applio-model",
                env_var=key,
                install_hint="Install a voice model via Hugging Face below, or set APPLIO_DEFAULT_PTH",
            ))
        return errors

    def applio_warnings(self, accent_model: str) -> list[str]:
        """Non-blocking notes (e.g. missing .index — sculpt still runs, lower quality)."""
        warnings: list[str] = []
        pth, index = self.resolve_applio_model(accent_model)
        if pth.is_file() and not index.is_file():
            warnings.append(
                f"No .index file for {pth.name}. Full sculpt will still run using index_rate=0 "
                f"(works, but often lower quality than with a .index sidecar)."
            )
        return warnings

    def resolve_applio_model(self, accent_model: str) -> tuple[Path, Path]:
        if accent_model == "target_accent":
            pth = self.applio_accent_pth or Path("")
            index = self.applio_accent_index
        elif accent_model and accent_model not in ("default", ""):
            named = self._resolve_named_applio_model(accent_model)
            if named:
                return named
            pth = self.applio_default_pth or Path("")
            index = self.applio_default_index
        else:
            pth = self.applio_default_pth or Path("")
            index = self.applio_default_index
            if self.applio_root:
                invalid = (
                    not pth.is_file()
                    or not _is_voice_model_pth_file(pth, applio_root=self.applio_root)
                )
                if invalid:
                    models = discover_rvc_models(self.applio_root)
                    if models:
                        pth = Path(models[0]["pth"])
                        index = Path(models[0]["index"]) if models[0].get("index") else None

        if pth.is_file() and (not index or not index.is_file()):
            index = _discover_index(pth)
        return pth, index or Path("")

    def _resolve_named_applio_model(self, name: str) -> Optional[tuple[Path, Path]]:
        candidate = Path(name.strip())
        if candidate.is_file() and candidate.suffix.lower() == ".pth":
            index = _discover_index(candidate)
            return candidate, index or Path("")

        if not self.applio_root or not self.applio_root.is_dir():
            return None

        needle = name.strip().lower()
        for model in discover_rvc_models(self.applio_root):
            pth = Path(model["pth"])
            if (
                model.get("name", "").lower() == needle
                or str(pth).lower() == needle
                or pth.stem.lower() == needle
                or pth.parent.name.lower() == needle
            ):
                index_path = Path(model["index"]) if model.get("index") else None
                if not index_path or not index_path.is_file():
                    index_path = _discover_index(pth)
                return pth, index_path or Path("")
        return None


def parse_huggingface_url(url: str) -> tuple[str, str, Optional[str]]:
    """Return (repo_id, revision, optional_file_path) from a Hugging Face URL."""
    cleaned = unquote(url.strip().rstrip("/"))
    match = _HF_HOST_RE.match(cleaned)
    if not match:
        raise ValueError(
            "Not a Hugging Face URL. Example: https://huggingface.co/Author/ModelName"
        )
    repo_id = match.group("repo")
    revision = match.group("rev") or "main"
    file_path = match.group("file")
    return repo_id, revision, unquote(file_path) if file_path else None


_MIN_VOICE_PTH_BYTES = 1_000_000  # real RVC voices are large; filters placeholders & trainer stubs
_LOGS_SKIP_DIRS = {"mute", "mute_spin", "mute_spin-v2", "zips", "reference", "__macosx"}
_PRETRAINED_STEM_RE = re.compile(
    r"^(f0[dgp]\d+k|d_.*|g_.*|.*_d$|.*_g$)$",
    re.IGNORECASE,
)


def _is_voice_model_pth_file(pth: Path, *, applio_root: Optional[Path] = None) -> bool:
    """True if this .pth under Applio logs/ looks like an inference voice, not training junk."""
    if not pth.is_file() or pth.suffix.lower() != ".pth":
        return False
    try:
        if pth.stat().st_size < _MIN_VOICE_PTH_BYTES:
            return False
    except OSError:
        return False

    parts = {part.lower() for part in pth.parts}
    if parts & _LOGS_SKIP_DIRS:
        return False

    rel = str(pth).lower().replace("\\", "/")
    if any(part in rel for part in _HF_SKIP_PATH_PARTS):
        return False
    if "/rvc/models/" in rel:
        return False

    stem = pth.stem.lower()
    if _PRETRAINED_STEM_RE.match(stem):
        return False
    if stem.startswith(("f0d", "f0g", "f0v")) and stem[-1].isdigit():
        return False

    if applio_root:
        logs = (applio_root / "logs").resolve()
        try:
            pth.resolve().relative_to(logs)
        except ValueError:
            return False

    return True


def _sanitize_model_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip())
    return cleaned.strip("_") or "voice_model"


def _is_voice_model_path(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    if not lower.endswith(".pth"):
        return False
    if any(part in lower for part in _HF_SKIP_PATH_PARTS):
        return False
    return True


def _pick_voice_pth(files: list[str]) -> Optional[str]:
    candidates = [f for f in files if _is_voice_model_path(f)]
    if not candidates:
        return None
    candidates.sort(key=lambda f: (f.count("/"), -len(f)))
    return candidates[0]


def _pick_matching_index(pth_file: str, index_files: list[str]) -> Optional[str]:
    if not index_files:
        return None
    pth_stem = Path(pth_file).stem.lower()
    pth_parent = str(Path(pth_file).parent).replace("\\", "/").lower()

    for idx in index_files:
        idx_lower = idx.lower()
        if pth_stem in Path(idx).stem.lower() or pth_stem in idx_lower:
            return idx
    for idx in index_files:
        if str(Path(idx).parent).replace("\\", "/").lower() == pth_parent:
            return idx
    return index_files[0]


def install_rvc_model_from_huggingface(url: str, applio_root: Path) -> dict[str, Any]:
    """
    Download RVC voice .pth (+ .index when present) using huggingface_hub.
    Applio's built-in downloader scrapes HTML for .zip only — this uses the real API.
    """
    if not applio_root.is_dir() or not (applio_root / "core.py").is_file():
        raise RuntimeError(f"Applio not found at {applio_root}")

    from huggingface_hub import hf_hub_download, list_repo_files

    repo_id, revision, file_hint = parse_huggingface_url(url)
    all_files = list_repo_files(repo_id, revision=revision)
    if not all_files:
        raise RuntimeError(f"Hugging Face repo is empty or private: {repo_id}")

    logs_dir = applio_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _copy_hub_file(remote_path: str, dest: Path) -> Path:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            revision=revision,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        return dest

    downloaded: list[str] = []

    if file_hint and file_hint.lower().endswith(".zip"):
        zip_dest = logs_dir / "zips"
        zip_dest.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=repo_id, filename=file_hint, revision=revision)
        model_name = _sanitize_model_name(Path(file_hint).stem)
        extract_dir = logs_dir / model_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(cached, "r") as zf:
            zf.extractall(extract_dir)
        pth_files = list(extract_dir.rglob("*.pth"))
        index_files = list(extract_dir.rglob("*.index"))
        if not pth_files:
            raise RuntimeError(f"Zip {file_hint} contains no .pth voice model")
        voice_pth = max(pth_files, key=lambda p: p.stat().st_size)
        voice_index = next((i for i in index_files if voice_pth.stem in i.stem), index_files[0] if index_files else None)
        final_pth = extract_dir / f"{model_name}.pth"
        shutil.move(str(voice_pth), final_pth)
        downloaded.append(str(final_pth))
        final_index = None
        if voice_index and voice_index.is_file():
            final_index = extract_dir / f"{model_name}.index"
            if voice_index != final_index:
                shutil.move(str(voice_index), final_index)
            downloaded.append(str(final_index))
        return {
            "model_name": model_name,
            "pth": str(final_pth),
            "index": str(final_index) if final_index else None,
            "repo_id": repo_id,
            "files": downloaded,
        }

    if file_hint and file_hint.lower().endswith(".pth"):
        pth_file = file_hint
        index_files = [f for f in all_files if f.endswith(".index")]
        index_file = _pick_matching_index(pth_file, index_files)
    elif file_hint and file_hint.lower().endswith(".index"):
        index_file = file_hint
        pth_candidates = [_pick_voice_pth(all_files)]
        pth_file = pth_candidates[0]
        if not pth_file:
            raise RuntimeError(f"No .pth found alongside {file_hint} in {repo_id}")
    else:
        zip_files = [f for f in all_files if f.lower().endswith(".zip")]
        if len(zip_files) == 1:
            return install_rvc_model_from_huggingface(
                f"https://huggingface.co/{repo_id}/resolve/{revision}/{zip_files[0]}",
                applio_root,
            )
        pth_file = _pick_voice_pth(all_files)
        if not pth_file:
            raise RuntimeError(
                f"No voice .pth found in {repo_id}. Repo may only contain training weights or need a direct file URL."
            )
        index_file = _pick_matching_index(pth_file, [f for f in all_files if f.endswith(".index")])

    model_name = _sanitize_model_name(Path(pth_file).stem or repo_id.split("/")[-1])
    dest_dir = logs_dir / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    final_pth = dest_dir / f"{model_name}.pth"
    _copy_hub_file(pth_file, final_pth)
    downloaded.append(str(final_pth))

    final_index = None
    if index_file:
        final_index = dest_dir / f"{model_name}.index"
        _copy_hub_file(index_file, final_index)
        downloaded.append(str(final_index))

    return {
        "model_name": model_name,
        "pth": str(final_pth),
        "index": str(final_index) if final_index else None,
        "repo_id": repo_id,
        "files": downloaded,
    }


def _count_pretrained_training_models(applio_root: Path) -> int:
    custom = applio_root / "rvc" / "models" / "pretraineds" / "custom"
    if not custom.is_dir():
        return 0
    return sum(1 for p in custom.glob("*.pth") if p.is_file() and p.stat().st_size > 1024)


def _build_voice_sculpt_guidance(
    found: dict[str, Any],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    applio_root = found.get("applio_root")
    root_path = Path(applio_root) if applio_root else None
    voice_count = len(models)
    pretrained_count = _count_pretrained_training_models(root_path) if root_path else 0
    logs_dir = str(root_path / "logs") if root_path else str(_DEFAULT_APPLIO_DIR / "logs")

    return {
        "voice_models_dir": logs_dir,
        "voice_model_count": voice_count,
        "pretrained_training_count": pretrained_count,
        "has_pretrained_only": pretrained_count > 0 and voice_count == 0,
        "applio_download_note": (
            "Applio's Model Link box does not use the Hugging Face API — it scrapes HTML for .zip files only. "
            "Use the Hugging Face installer below in LiangLocal, or drag .pth + .index into Applio."
        ),
        "quick_start": [
            "Add two or more clean voice clips (you already clip them — skip UVR unless needed).",
            "Merge morphs timbre in vocoder space (like celebrity face blends) — both voices should show in the hybrid.",
            "Optional RVC polish only if you want to nudge toward a .pth model afterward (off by default).",
            "Output is one .wav in voice_references/ for Eloquent / Chatterbox.",
        ],
        "applio_voice_model_steps": [
            "Paste a Hugging Face repo URL in the installer below (uses huggingface_hub API).",
            "Or open tools/Applio/run-applio.bat → Download → Drop files (.pth then .index).",
            "Do NOT use Applio 'Download Pretrained Models' or bare HF repo URLs in Applio's Model Link.",
            f"Voice files land in {logs_dir}/<ModelName>/",
            "Click Refresh setup when done.",
        ],
    }


class AutomationService:
    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        self.config = config or AutomationConfig.from_env()
        self.gpu_queue = GPUQueue()
        self._job_sem = asyncio.Semaphore(self.config.max_concurrent)
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

    def refresh_config(self) -> None:
        """Re-run discovery and reload config into this process (no restart needed)."""
        self.config = AutomationConfig.from_env(auto_discover=True)

    async def discover(self) -> dict[str, Any]:
        """Return auto-detected paths and model inventory."""
        found = discover_environment()
        models = []
        applio_root = found.get("applio_root")
        if applio_root:
            models = discover_rvc_models(Path(applio_root))
        guidance = _build_voice_sculpt_guidance(found, models)
        return {
            "discovered": _serialize_discovery(found),
            "models": models,
            "guidance": guidance,
            "default_applio_dir": str(_DEFAULT_APPLIO_DIR),
            "sculpt_env_bat": str(_SCULPT_ENV_BAT),
            "sculpt_env_bat_exists": _SCULPT_ENV_BAT.is_file(),
        }

    async def install_huggingface_model(self, url: str, applio_dest: Optional[str] = None) -> dict[str, Any]:
        """Download RVC voice model from Hugging Face into Applio logs/ using huggingface_hub."""
        dest = Path(applio_dest) if applio_dest else _DEFAULT_APPLIO_DIR
        if not (dest / "core.py").is_file():
            raise RuntimeError(f"Applio not installed at {dest}")

        result = await asyncio.to_thread(install_rvc_model_from_huggingface, url.strip(), dest)

        discovered = discover_environment(preferred_applio=dest)
        write_sculpt_env_bat(discovered)
        apply_discovered_env(discovered)
        self.refresh_config()

        pf_rvc = await self.preflight(for_rvc=True)
        return {
            "status": "ok",
            "install": result,
            "discovered": _serialize_discovery(discovered),
            "rvc_ready": pf_rvc["ready"],
            "preflight_rvc": pf_rvc,
            "next_steps": _next_setup_steps(pf_rvc),
        }

    async def auto_setup(
        self,
        *,
        clone_applio: bool = False,
        install_uvr: bool = True,
        write_env_file: bool = True,
        applio_dest: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        One-click bootstrap: install UVR tool, clone Applio if requested, discover paths,
        write sculpt.env.bat, apply env vars to this backend process immediately.
        """
        steps: list[str] = []
        dest = Path(applio_dest) if applio_dest else _DEFAULT_APPLIO_DIR

        if install_uvr:
            uvr_ok = await self._ensure_audio_separator()
            steps.append("audio-separator installed" if uvr_ok else "audio-separator already present or install skipped")

        if clone_applio and not (dest / "core.py").is_file():
            await self._clone_applio(dest)
            steps.append(f"cloned Applio to {dest}")
        elif clone_applio:
            steps.append(f"Applio already present at {dest}")

        discovered = discover_environment(preferred_applio=dest if (dest / "core.py").is_file() else None)
        if write_env_file and discovered:
            write_sculpt_env_bat(discovered)
            steps.append(f"wrote {_SCULPT_ENV_BAT.name}")

        apply_discovered_env(discovered)
        self.refresh_config()

        pf_uvr = await self.preflight(for_rvc=False)
        pf_rvc = await self.preflight(for_rvc=True)
        models = discover_rvc_models(dest) if (dest / "core.py").is_file() else []
        guidance = _build_voice_sculpt_guidance(discovered, models)
        return {
            "status": "ok",
            "steps": steps,
            "discovered": _serialize_discovery(discovered),
            "models": models,
            "guidance": guidance,
            "uvr_ready": pf_uvr["ready"],
            "rvc_ready": pf_rvc["ready"],
            "preflight_uvr": pf_uvr,
            "preflight_rvc": pf_rvc,
            "next_steps": _next_setup_steps(pf_rvc),
        }

    async def _ensure_audio_separator(self) -> bool:
        venv_sep = _PROJECT_ROOT / "venv" / "Scripts" / "audio-separator.exe"
        if venv_sep.is_file() or shutil.which("audio-separator"):
            return False
        pip = sys.executable
        proc = await asyncio.create_subprocess_exec(
            pip, "-m", "pip", "install", "audio-separator", "onnx>=1.17.0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        if proc.returncode != 0:
            err = (stderr_b or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"pip install audio-separator failed: {_tail(err)}")
        return True

    async def _clone_applio(self, dest: Path) -> None:
        git = shutil.which("git")
        if not git:
            raise RuntimeError("git not found on PATH — install Git for Windows to auto-clone Applio")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not (dest / "core.py").is_file():
            raise RuntimeError(f"{dest} exists but is not an Applio install")
        if (dest / "core.py").is_file():
            return
        proc = await asyncio.create_subprocess_exec(
            git, "clone", "--depth", "1", _APPLIO_GIT_URL, str(dest),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        if proc.returncode != 0:
            err = (stderr_b or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"git clone Applio failed: {_tail(err)}")
        logger.info("Cloned Applio to %s — run run-applio.bat once to finish Applio's venv setup", dest)

    async def preflight(
        self,
        *,
        for_youtube: bool = False,
        for_uvr: bool = True,
        for_rvc: bool = True,
        for_morph: bool = False,
        accent_model: str = "default",
    ) -> dict[str, Any]:
        missing: list[dict[str, Any]] = []
        ready: list[str] = []
        warnings: list[str] = []

        checks: list[tuple[str, str, str]] = [
            ("ffmpeg", self.config.ffmpeg_bin, "FFMPEG_BIN"),
        ]
        if for_uvr:
            checks.insert(0, ("audio-separator", self.config.audio_separator_bin, "AUDIO_SEPARATOR_BIN"))
        if for_youtube:
            checks.append(("yt-dlp", self.config.yt_dlp_bin, "YT_DLP_BIN"))

        for tool, binary, env_var in checks:
            err = _check_binary(binary, tool, env_var, _install_hint(tool))
            if err:
                missing.append(err.to_dict())
            else:
                ready.append(tool)

        if for_morph:
            try:
                import pyworld  # noqa: F401
                ready.append("pyworld")
            except ImportError:
                missing.append(MissingToolError(
                    detail="pyworld is required for voice timbre morph merging",
                    missing_tool="pyworld",
                    env_var="",
                    install_hint="pip install pyworld",
                ).to_dict())

        rvc_missing: list[dict[str, Any]] = []
        checked_model = accent_model or "default"
        if for_rvc:
            errors = self.config.validate_applio(checked_model)
            if errors and self.config.applio_root:
                models = discover_rvc_models(self.config.applio_root)
                if models and checked_model in ("default", ""):
                    checked_model = models[0]["name"]
                    errors = self.config.validate_applio(checked_model)
            for err in errors:
                item = err.to_dict()
                missing.append(item)
                rvc_missing.append(item)
            if not rvc_missing:
                ready.append("applio")
            warnings = self.config.applio_warnings(checked_model)

        core_missing = [m for m in missing if m.get("missing_tool") not in (
            "applio", "applio-python", "applio-model"
        )]

        return {
            "ready": len(missing) == 0,
            "merge_ready": len(core_missing) == 0,
            "uvr_ready": len(core_missing) == 0,
            "rvc_ready": len(rvc_missing) == 0,
            "missing": missing,
            "warnings": warnings,
            "accent_model_checked": checked_model,
            "available_tools": ready,
            "work_dir": str(self.config.work_dir),
            "voice_references_dir": str(_VOICE_REFERENCES_DIR),
            "config": _serialize_config(self.config),
        }

    async def sculpt_stream(self, request: SculptRequest) -> AsyncIterator[dict[str, Any]]:
        async with self._job_sem:
            try:
                source_count = (
                    len(parse_sculpt_sources(source=request.source, sources=request.sources))
                    if request.source_type == "local_path"
                    else 1
                )
                pf = await self.preflight(
                    for_youtube=request.source_type == "youtube_url",
                    for_uvr=not request.skip_uvr,
                    for_rvc=not request.skip_rvc,
                    for_morph=(
                        source_count > 1
                        and (request.combine_mode or "morph").lower() == "morph"
                    ),
                    accent_model=request.accent_model or "default",
                )
                if not pf["ready"]:
                    first = pf["missing"][0]
                    yield {"type": "error", **first}
                    return

                job_id = uuid.uuid4().hex[:12]
                events: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()

                async def emit(event: dict[str, Any]) -> None:
                    await events.put(event)

                task = PipelineTask(
                    job_id=job_id,
                    request=request,
                    work_root=self.config.work_dir,
                    config=self.config,
                    gpu_queue=self.gpu_queue,
                    run_subprocess=self._run_subprocess,
                    emit=emit,
                )

                async def run_pipeline() -> None:
                    try:
                        final_path = await task.run()
                        await events.put({
                            "type": "done",
                            "voice_id": task.last_voice_id,
                            "path": str(final_path),
                            "filename": task.last_voice_id,
                            "voice_references_dir": str(_VOICE_REFERENCES_DIR),
                            "source_count": (
                                len(parse_sculpt_sources(source=request.source, sources=request.sources))
                                if request.source_type == "local_path"
                                else 1
                            ),
                            "combine_mode": request.combine_mode,
                            "morph_balance": request.morph_balance,
                            "voice_prompt_saved": bool((request.voice_prompt or "").strip()),
                        })
                    except PreconditionError as pe:
                        await events.put({"type": "error", **pe.missing.to_dict()})
                    except Exception as exc:
                        logger.error("Voice sculpt pipeline failed: %s", exc, exc_info=True)
                        await events.put({
                            "type": "error",
                            "status": 500,
                            "detail": str(exc),
                        })
                    finally:
                        await events.put(None)

                runner = asyncio.create_task(run_pipeline())

                while True:
                    item = await events.get()
                    if item is None:
                        break
                    yield item

                await runner

            except Exception as exc:
                logger.error("Voice sculpt stream error: %s", exc, exc_info=True)
                yield {"type": "error", "status": 500, "detail": str(exc)}

    async def _run_subprocess(
        self,
        cmd: list[str],
        *,
        cwd: Optional[str],
        env: Optional[dict[str, str]],
        gpu: bool,
        label: str,
    ) -> SubprocessResult:
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        if gpu and self.config.gpu_id is not None:
            proc_env["CUDA_VISIBLE_DEVICES"] = self.config.gpu_id

        logger.info("[%s] Running: %s", label, " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=proc_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.job_timeout_sec,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{label} timed out after {self.config.job_timeout_sec}s")

        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")
        if proc.returncode != 0:
            logger.error("[%s] exit %s stderr: %s", label, proc.returncode, _tail(stderr))
            raise RuntimeError(f"{label} failed (exit {proc.returncode}): {_tail(stderr)}")
        return SubprocessResult(returncode=proc.returncode or 0, stdout=stdout, stderr=stderr)


def _resolve_bin(env_key: str, default_name: str) -> str:
    explicit = (os.getenv(env_key) or "").strip()
    if explicit:
        return explicit
    found = shutil.which(default_name)
    return found or default_name


def _check_binary(binary: str, tool: str, env_var: str, install_hint: str) -> Optional[MissingToolError]:
    path = Path(binary)
    if path.is_file():
        return None
    if shutil.which(binary):
        return None
    return MissingToolError(
        detail=f"{tool} not found ({binary})",
        missing_tool=tool,
        env_var=env_var,
        install_hint=install_hint,
    )


def _install_hint(tool: str) -> str:
    hints = {
        "audio-separator": "pip install audio-separator — https://github.com/nomadkaraoke/python-audio-separator",
        "ffmpeg": FFMPEG_INSTALL_HINT,
        "yt-dlp": "Install yt-dlp — https://github.com/yt-dlp/yt-dlp",
    }
    return hints.get(tool, f"Install {tool}")


def _sanitize_voice_name(name: str) -> str:
    clean = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    return clean.replace(" ", "_")


def _tail(text: str, limit: int = 2048) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _find_newest_wav(directory: Path, prefer_vocals: bool = False) -> Path:
    wavs = list(directory.glob("*.wav"))
    if prefer_vocals:
        vocal = [w for w in wavs if "vocal" in w.name.lower()]
        if vocal:
            return max(vocal, key=lambda p: p.stat().st_mtime)
    if not wavs:
        raise RuntimeError(f"No WAV output found in {directory}")
    return max(wavs, key=lambda p: p.stat().st_mtime)


def _discover_index(pth_path: Path) -> Optional[Path]:
    parent = pth_path.parent
    indices = sorted(parent.glob("*.index"), key=lambda p: p.stat().st_mtime, reverse=True)
    if indices:
        return indices[0]
    if pth_path.parent.parent and pth_path.parent.parent.is_dir():
        indices = sorted(pth_path.parent.parent.rglob("*.index"), key=lambda p: p.stat().st_mtime, reverse=True)
        if indices:
            return indices[0]
    return None


def _applio_python_for_root(applio_root: Path) -> Optional[str]:
    if sys.platform == "win32":
        candidates = [
            applio_root / "env" / "python.exe",
            applio_root / "env" / "Scripts" / "python.exe",
            applio_root / ".venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            applio_root / "env" / "bin" / "python",
            applio_root / ".venv" / "bin" / "python",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _is_applio_root(path: Path) -> bool:
    return path.is_dir() and (path / "core.py").is_file()


def _applio_search_paths() -> list[Path]:
    home = Path.home()
    candidates = [
        _DEFAULT_APPLIO_DIR,
        _PROJECT_ROOT / "Applio",
        Path("C:/Tools/Applio"),
        Path("D:/Tools/Applio"),
        home / "Applio",
        home / "Tools" / "Applio",
        home / "Documents" / "Applio",
    ]
    env_root = (os.getenv("APPLIO_ROOT") or "").strip()
    if env_root:
        candidates.insert(0, Path(env_root))
    seen: set[str] = set()
    out: list[Path] = []
    for c in candidates:
        key = str(c).lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def discover_rvc_models(applio_root: Path) -> list[dict[str, Any]]:
    logs = applio_root / "logs"
    if not logs.is_dir():
        return []
    models: list[dict[str, Any]] = []
    for pth in logs.rglob("*.pth"):
        if not _is_voice_model_pth_file(pth, applio_root=applio_root):
            continue
        index = _discover_index(pth)
        models.append({
            "name": pth.stem,
            "pth": str(pth),
            "index": str(index) if index else None,
            "has_index": bool(index and index.is_file()),
            "mtime": pth.stat().st_mtime,
        })
    models.sort(key=lambda m: m["mtime"], reverse=True)
    return models


def discover_environment(*, preferred_applio: Optional[Path] = None) -> dict[str, Any]:
    found: dict[str, Any] = {}

    # audio-separator: LiangLocal venv first
    venv_sep = _PROJECT_ROOT / "venv" / "Scripts" / "audio-separator.exe"
    if venv_sep.is_file():
        found["audio_separator_bin"] = str(venv_sep)
    else:
        which_sep = shutil.which("audio-separator")
        if which_sep:
            found["audio_separator_bin"] = which_sep

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        found["ffmpeg_bin"] = ffmpeg

    applio_root: Optional[Path] = None
    if preferred_applio and _is_applio_root(preferred_applio):
        applio_root = preferred_applio
    else:
        for candidate in _applio_search_paths():
            if _is_applio_root(candidate):
                applio_root = candidate
                break

    if applio_root:
        found["applio_root"] = applio_root
        py = _applio_python_for_root(applio_root)
        if py:
            found["applio_python"] = py
        models = discover_rvc_models(applio_root)
        if models:
            best = models[0]
            found["applio_default_pth"] = Path(best["pth"])
            if best.get("index"):
                found["applio_default_index"] = Path(best["index"])

    return found


def apply_discovered_env(discovered: dict[str, Any]) -> None:
    mapping = {
        "applio_root": "APPLIO_ROOT",
        "applio_python": "APPLIO_PYTHON",
        "audio_separator_bin": "AUDIO_SEPARATOR_BIN",
        "ffmpeg_bin": "FFMPEG_BIN",
    }
    for key, env_key in mapping.items():
        val = discovered.get(key)
        if val:
            os.environ[env_key] = str(val)
    pth = discovered.get("applio_default_pth")
    if pth:
        os.environ["APPLIO_DEFAULT_PTH"] = str(pth)
    index = discovered.get("applio_default_index")
    if index:
        os.environ["APPLIO_DEFAULT_INDEX"] = str(index)


def write_sculpt_env_bat(discovered: dict[str, Any], path: Path = _SCULPT_ENV_BAT) -> Path:
    lines = ["@echo off", "REM Auto-generated by LiangLocal voice sculpt auto-setup"]

    def _set(key: str, env_name: str) -> None:
        val = discovered.get(key)
        if val:
            lines.append(f'set {env_name}={val}')

    _set("applio_root", "APPLIO_ROOT")
    _set("applio_python", "APPLIO_PYTHON")
    _set("applio_default_pth", "APPLIO_DEFAULT_PTH")
    _set("applio_default_index", "APPLIO_DEFAULT_INDEX")
    _set("audio_separator_bin", "AUDIO_SEPARATOR_BIN")
    _set("ffmpeg_bin", "FFMPEG_BIN")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote voice sculpt env file: %s", path)
    return path


def _serialize_discovery(discovered: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in discovered.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def _serialize_config(config: AutomationConfig) -> dict[str, Any]:
    return {
        "applio_root": str(config.applio_root) if config.applio_root else None,
        "applio_python": config.applio_python,
        "applio_default_pth": str(config.applio_default_pth) if config.applio_default_pth else None,
        "applio_default_index": str(config.applio_default_index) if config.applio_default_index else None,
        "audio_separator_bin": config.audio_separator_bin,
        "ffmpeg_bin": config.ffmpeg_bin,
    }


def _next_setup_steps(pf_rvc: dict[str, Any]) -> list[str]:
    if pf_rvc.get("ready"):
        return []
    steps: list[str] = []
    missing_tools = {m.get("missing_tool") for m in pf_rvc.get("missing", [])}
    if "applio" in missing_tools:
        steps.append("Click 'Install Applio' to git-clone into tools/Applio")
        steps.append("Then run tools/Applio/run-install.bat to create Applio's env and install deps")
    if "applio-python" in missing_tools:
        steps.append("Run tools/Applio/run-install.bat (creates env\\ and installs PyTorch deps)")
    if "applio-model" in missing_tools:
        steps.append("Install a voice .pth via Hugging Face below (user + repo, or full URL)")
        steps.append("Pretrained trainers under rvc/models/pretraineds/ are not voice models")
        steps.append(".index files are optional — .pth alone is enough to sculpt")
    return steps
