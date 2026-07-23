# app/stt_service.py
# --- Multiple STT engines support (Whisper and Parakeet) ---
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import soundfile as sf
import tempfile
import logging
import os
import sys
import importlib
import asyncio
import shutil
import subprocess
import io
import json
from pathlib import Path

import numpy as np
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

# Monkey-patch Windows tempfile cleanup once at import time.
# NeMo writes manifest.json to TemporaryDirectory and its DataLoader workers
# hold file handles past cleanup, causing PermissionError on Windows.
# ignore_cleanup_errors=True (Python 3.10+) silently suppresses that.
import tempfile as _tf
_orig_TemporaryDirectory = _tf.TemporaryDirectory
_tf.TemporaryDirectory = lambda *a, **kw: _orig_TemporaryDirectory(*a, ignore_cleanup_errors=True, **{k: v for k, v in kw.items() if k != 'ignore_cleanup_errors'})

# --- STT Models ---
whisper_processor = None
whisper_model = None
parakeet_model = None
parakeet_v3_model = None
parakeet_zh_model = None  # NeMo Mandarin Chinese ASR
nemotron_model = None  # NVIDIA Nemotron Speech Streaming ASR

def get_device():
    """Determines the correct Torch device for the current process."""
    try:
        import torch
        force_cpu = os.environ.get("MIRID_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
        if torch.cuda.is_available() and not force_cpu:
            # Since the desktop host isolates the process to a single GPU,
            # that GPU will always be seen as 'cuda:0' by this process.
            device = "cuda:0"
            logger.info(f"✅ STT service will use the visible GPU: {device}")
            return device
        else:
            logger.warning("⚠️ CUDA not available. Falling back to CPU for STT.")
            return "cpu"
    except Exception as e:
        logger.error(f"❌ Error determining device, falling back to CPU: {e}")
        return "cpu"

DEVICE = get_device()
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
PARAKEET_V3_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
# NeMo Conformer-Transducer Large for Mandarin Chinese (16 kHz mono WAV)
PARAKEET_ZH_MODEL_ID = "nvidia/stt_zh_conformer_transducer_large"
# Nemotron Speech Streaming — FastConformer-CacheAware-RNNT English ASR
NEMOTRON_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
# Moonshine Streaming Tiny — Lightweight streaming ASR (34M params, English only)
MOONSHINE_MODEL_ID = "UsefulSensors/moonshine-streaming-tiny"

# --- NanoGPT Cloud STT Models ---
NANOGPT_API_BASE = "https://nano-gpt.com/api"
NANOGPT_STT_ENDPOINT = f"{NANOGPT_API_BASE}/transcribe"
# Known STT models on NanoGPT (can be discovered via /api/v1/audio-models?type=stt)
NANOGPT_STT_MODELS = {
    "fun-asr-flash-2026-06-15": "Alibaba Fun-ASR Flash (Multilingual, Diarization, Timestamps)",
    "Whisper-Large-V3": "Whisper Large V3 (High Accuracy)",
    "Wizper": "Wizper (Fast Processing)",
    "Elevenlabs-STT": "ElevenLabs STT (Async + Diarization)",
    "gpt-4o-mini-transcribe": "GPT-4o Mini Transcribe (Improved Accuracy)",
    "gpt-4o-mini-transcribe-2025-03-20": "GPT-4o Mini Transcribe (2025-03-20)",
    "gpt-4o-mini-transcribe-2025-12-15": "GPT-4o Mini Transcribe (2025-12-15)",
    "gpt-4o-mini-transcribe-latest": "GPT-4o Mini Transcribe (Latest)",
    "openai-whisper-with-video": "OpenAI Whisper with Video Support",
}
NANOGPT_DEFAULT_STT_MODEL = "fun-asr-flash-2026-06-15"

PARAKEET_TDT_ENGINE_IDS = frozenset({"parakeet", "parakeet-v3"})

# --- Parakeet.cpp GGUF Model Catalog ---
# Source: https://huggingface.co/mudler/parakeet-cpp-gguf
PARAKEET_CPP_HF_REPO = "mudler/parakeet-cpp-gguf"

PARAKEET_CPP_GGUF_MODELS = {
    "tdt_ctc-110m": {
        "label": "Parakeet TDT+CTC 110M (Hybrid, Fastest)",
        "source": "nvidia/parakeet-tdt_ctc-110m",
        "arch": "Hybrid TDT+CTC (FastConformer)",
        "params": "110M",
        "files": {
            "f16":  {"name": "tdt_ctc-110m-f16.gguf",  "size_mb": 267.5},
            "q8_0": {"name": "tdt_ctc-110m-q8_0.gguf", "size_mb": 177.8},
            "q6_k": {"name": "tdt_ctc-110m-q6_k.gguf", "size_mb": 155.9},
            "q5_k": {"name": "tdt_ctc-110m-q5_k.gguf", "size_mb": 143.3},
            "q4_k": {"name": "tdt_ctc-110m-q4_k.gguf", "size_mb": 131.4},
        },
        "recommended": "f16",
    },
    "realtime_eou_120m-v1": {
        "label": "Parakeet Realtime EOU 120M (Streaming)",
        "source": "nvidia/parakeet_realtime_eou_120m-v1",
        "arch": "Cache-aware streaming RNNT (FastConformer)",
        "params": "120M",
        "files": {
            "f16":  {"name": "realtime_eou_120m-v1-f16.gguf",  "size_mb": 266.5},
            "q8_0": {"name": "realtime_eou_120m-v1-q8_0.gguf", "size_mb": 176.0},
            "q6_k": {"name": "realtime_eou_120m-v1-q6_k.gguf", "size_mb": 153.9},
            "q5_k": {"name": "realtime_eou_120m-v1-q5_k.gguf", "size_mb": 141.2},
            "q4_k": {"name": "realtime_eou_120m-v1-q4_k.gguf", "size_mb": 129.1},
        },
        "recommended": "f16",
    },
    "ctc-0.6b": {
        "label": "Parakeet CTC 0.6B (English)",
        "source": "nvidia/parakeet-ctc-0.6b",
        "arch": "CTC (FastConformer)",
        "params": "0.6B",
        "files": {
            "f16":  {"name": "ctc-0.6b-f16.gguf",  "size_mb": 1373.4},
            "q8_0": {"name": "ctc-0.6b-q8_0.gguf", "size_mb": 875.4},
            "q6_k": {"name": "ctc-0.6b-q6_k.gguf", "size_mb": 746.8},
            "q5_k": {"name": "ctc-0.6b-q5_k.gguf", "size_mb": 676.3},
            "q4_k": {"name": "ctc-0.6b-q4_k.gguf", "size_mb": 609.9},
        },
        "recommended": "f16",
    },
    "rnnt-0.6b": {
        "label": "Parakeet RNNT 0.6B (English)",
        "source": "nvidia/parakeet-rnnt-0.6b",
        "arch": "RNNT transducer (FastConformer)",
        "params": "0.6B",
        "files": {
            "f16":  {"name": "rnnt-0.6b-f16.gguf",  "size_mb": 1402.8},
            "q8_0": {"name": "rnnt-0.6b-q8_0.gguf", "size_mb": 903.9},
            "q6_k": {"name": "rnnt-0.6b-q6_k.gguf", "size_mb": 776.3},
            "q5_k": {"name": "rnnt-0.6b-q5_k.gguf", "size_mb": 705.7},
            "q4_k": {"name": "rnnt-0.6b-q4_k.gguf", "size_mb": 639.2},
        },
        "recommended": "f16",
    },
    "tdt-0.6b-v2": {
        "label": "Parakeet TDT 0.6B v2 (English)",
        "source": "nvidia/parakeet-tdt-0.6b-v2",
        "arch": "TDT transducer (FastConformer)",
        "params": "0.6B",
        "files": {
            "f16":  {"name": "tdt-0.6b-v2-f16.gguf",  "size_mb": 1404.2},
            "q8_0": {"name": "tdt-0.6b-v2-q8_0.gguf", "size_mb": 903.8},
            "q6_k": {"name": "tdt-0.6b-v2-q6_k.gguf", "size_mb": 775.9},
            "q5_k": {"name": "tdt-0.6b-v2-q5_k.gguf", "size_mb": 705.0},
            "q4_k": {"name": "tdt-0.6b-v2-q4_k.gguf", "size_mb": 638.4},
        },
        "recommended": "f16",
    },
    "tdt-0.6b-v3": {
        "label": "Parakeet TDT 0.6B v3 (Multilingual)",
        "source": "nvidia/parakeet-tdt-0.6b-v3",
        "arch": "TDT transducer (FastConformer)",
        "params": "0.6B",
        "files": {
            "f16":  {"name": "tdt-0.6b-v3-f16.gguf",  "size_mb": 1441.0},
            "q8_0": {"name": "tdt-0.6b-v3-q8_0.gguf", "size_mb": 940.7},
            "q6_k": {"name": "tdt-0.6b-v3-q6_k.gguf", "size_mb": 812.7},
            "q5_k": {"name": "tdt-0.6b-v3-q5_k.gguf", "size_mb": 741.9},
            "q4_k": {"name": "tdt-0.6b-v3-q4_k.gguf", "size_mb": 675.2},
        },
        "recommended": "f16",
    },
    "ctc-1.1b": {
        "label": "Parakeet CTC 1.1B (English, High Accuracy)",
        "source": "nvidia/parakeet-ctc-1.1b",
        "arch": "CTC (FastConformer)",
        "params": "1.1B",
        "files": {
            "f16":  {"name": "ctc-1.1b-f16.gguf",  "size_mb": 2395.8},
            "q8_0": {"name": "ctc-1.1b-q8_0.gguf", "size_mb": 1526.3},
            "q6_k": {"name": "ctc-1.1b-q6_k.gguf", "size_mb": 1301.7},
            "q5_k": {"name": "ctc-1.1b-q5_k.gguf", "size_mb": 1178.5},
            "q4_k": {"name": "ctc-1.1b-q4_k.gguf", "size_mb": 1062.6},
        },
        "recommended": "f16",
    },
    "rnnt-1.1b": {
        "label": "Parakeet RNNT 1.1B (English, High Accuracy)",
        "source": "nvidia/parakeet-rnnt-1.1b",
        "arch": "RNNT transducer (FastConformer)",
        "params": "1.1B",
        "files": {
            "f16":  {"name": "rnnt-1.1b-f16.gguf",  "size_mb": 2425.2},
            "q8_0": {"name": "rnnt-1.1b-q8_0.gguf", "size_mb": 1554.7},
            "q6_k": {"name": "rnnt-1.1b-q6_k.gguf", "size_mb": 1331.2},
            "q5_k": {"name": "rnnt-1.1b-q5_k.gguf", "size_mb": 1207.9},
            "q4_k": {"name": "rnnt-1.1b-q4_k.gguf", "size_mb": 1091.9},
        },
        "recommended": "f16",
    },
    "tdt-1.1b": {
        "label": "Parakeet TDT 1.1B (English, High Accuracy)",
        "source": "nvidia/parakeet-tdt-1.1b",
        "arch": "TDT transducer (FastConformer)",
        "params": "1.1B",
        "files": {
            "f16":  {"name": "tdt-1.1b-f16.gguf",  "size_mb": 2425.3},
            "q8_0": {"name": "tdt-1.1b-q8_0.gguf", "size_mb": 1554.8},
            "q6_k": {"name": "tdt-1.1b-q6_k.gguf", "size_mb": 1331.2},
            "q5_k": {"name": "tdt-1.1b-q5_k.gguf", "size_mb": 1207.9},
            "q4_k": {"name": "tdt-1.1b-q4_k.gguf", "size_mb": 1091.9},
        },
        "recommended": "f16",
    },
    "tdt_ctc-1.1b": {
        "label": "Parakeet TDT+CTC 1.1B (Hybrid, Best Quality)",
        "source": "nvidia/parakeet-tdt_ctc-1.1b",
        "arch": "Hybrid TDT+CTC (FastConformer)",
        "params": "1.1B",
        "files": {
            "f16":  {"name": "tdt_ctc-1.1b-f16.gguf",  "size_mb": 2429.5},
            "q8_0": {"name": "tdt_ctc-1.1b-q8_0.gguf", "size_mb": 1559.0},
            "q6_k": {"name": "tdt_ctc-1.1b-q6_k.gguf", "size_mb": 1335.4},
            "q5_k": {"name": "tdt_ctc-1.1b-q5_k.gguf", "size_mb": 1212.1},
            "q4_k": {"name": "tdt_ctc-1.1b-q4_k.gguf", "size_mb": 1096.1},
        },
        "recommended": "f16",
    },
}


def _get_parakeet_cpp_models_dir() -> Path:
    backend_dir = Path(__file__).resolve().parent.parent
    models_dir = backend_dir / "parakeet_cpp_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _get_parakeet_cpp_binary() -> Optional[str]:
    backend_dir = Path(__file__).resolve().parent.parent
    candidates = [
        backend_dir / "parakeet.cpp" / "build" / "examples" / "cli" / "parakeet-cli.exe",
        backend_dir / "parakeet.cpp" / "build" / "examples" / "cli" / "parakeet-cli",
        backend_dir / "parakeet_cpp" / "parakeet-cli.exe",
        backend_dir / "parakeet_cpp" / "parakeet-cli",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    import shutil
    system_cli = shutil.which("parakeet-cli")
    if system_cli:
        return system_cli
    return None


def is_parakeet_cpp_available() -> bool:
    return _get_parakeet_cpp_binary() is not None


def list_parakeet_cpp_downloaded_models() -> list:
    models_dir = _get_parakeet_cpp_models_dir()
    downloaded = []
    for model_id, catalog in PARAKEET_CPP_GGUF_MODELS.items():
        for quant_key, file_info in catalog["files"].items():
            gguf_path = models_dir / file_info["name"]
            if gguf_path.exists():
                downloaded.append({
                    "model_id": model_id,
                    "quant": quant_key,
                    "filename": file_info["name"],
                    "size_mb": file_info["size_mb"],
                    "path": str(gguf_path),
                    "label": catalog["label"],
                    "arch": catalog["arch"],
                    "params": catalog["params"],
                    "source": catalog["source"],
                })
    return downloaded


async def download_parakeet_cpp_model(
    model_id: str,
    quant: str = "f16",
    progress_callback=None,
) -> tuple:
    model_id = model_id.strip()
    quant = quant.strip().lower()
    if model_id not in PARAKEET_CPP_GGUF_MODELS:
        return False, f"Unknown model: {model_id}"
    catalog = PARAKEET_CPP_GGUF_MODELS[model_id]
    if quant not in catalog["files"]:
        return False, f"Unknown quantization: {quant}. Available: {', '.join(catalog['files'].keys())}"
    file_info = catalog["files"][quant]
    filename = file_info["name"]
    models_dir = _get_parakeet_cpp_models_dir()
    dest_path = models_dir / filename
    if dest_path.exists():
        return True, f"Already downloaded: {filename}"

    try:
        from huggingface_hub import hf_hub_download
        if progress_callback:
            await progress_callback(f"Downloading {filename} ({file_info['size_mb']:.0f} MB)...")

        def _download():
            return hf_hub_download(
                repo_id=PARAKEET_CPP_HF_REPO,
                filename=filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            )

        loop = asyncio.get_event_loop()
        downloaded_path = await loop.run_in_executor(None, _download)
        logger.info(f"Downloaded {filename} -> {downloaded_path}")
        return True, f"Downloaded {filename} successfully"
    except ImportError:
        return False, "huggingface_hub not installed. Run: pip install huggingface_hub"
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}", exc_info=True)
        return False, f"Download failed: {e}"


async def delete_parakeet_cpp_model(filename: str) -> tuple:
    models_dir = _get_parakeet_cpp_models_dir()
    target = models_dir / filename
    if not target.exists():
        return False, f"File not found: {filename}"
    try:
        target.unlink()
        logger.info(f"Deleted {filename}")
        return True, f"Deleted {filename}"
    except Exception as e:
        return False, f"Delete failed: {e}"


async def transcribe_with_parakeet_cpp_array(audio, sr: int, model_id: str = None, quant: str = None) -> str:
    cli_binary = _get_parakeet_cpp_binary()
    if not cli_binary:
        raise RuntimeError(
            "Parakeet.cpp is unavailable in this Mirid runtime. "
            "Update Mirid or choose another speech-to-text engine."
        )

    if model_id is None:
        model_id = "tdt_ctc-110m"
    if quant is None:
        quant = "f16"

    catalog = PARAKEET_CPP_GGUF_MODELS.get(model_id)
    if not catalog:
        raise RuntimeError(f"Unknown parakeet-cpp model: {model_id}")
    file_info = catalog["files"].get(quant)
    if not file_info:
        raise RuntimeError(f"Unknown quantization '{quant}' for {model_id}")

    models_dir = _get_parakeet_cpp_models_dir()
    gguf_path = models_dir / file_info["name"]
    if not gguf_path.exists():
        raise RuntimeError(
            f"Model file not downloaded: {file_info['name']}. "
            f"Download it first from Settings > STT > Parakeet.cpp GGUF Models."
        )

    temp_wav_path = None
    try:
        temp_wav_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_wav_path, audio, sr)

        audio_duration = len(audio) / sr
        logger.info(f"Parakeet.cpp: transcribing {audio_duration:.2f}s audio with {file_info['name']}")

        cmd = [
            cli_binary, "transcribe",
            "--model", str(gguf_path),
            "--input", temp_wav_path,
        ]

        loop = asyncio.get_event_loop()

        def _run_cli():
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            return result

        result = await loop.run_in_executor(None, _run_cli)

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:500]
            raise RuntimeError(f"parakeet-cli failed (exit {result.returncode}): {stderr_snippet}")

        transcript = (result.stdout or "").strip()
        if not transcript:
            logger.warning("parakeet-cli returned empty transcript")
        else:
            logger.info(f"Parakeet.cpp transcription complete ({len(transcript)} chars)")
        return transcript

    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


async def transcribe_with_parakeet_cpp(audio_file_path: str, model_id: str = None, quant: str = None) -> str:
    audio, sr = _load_stt_audio_path(audio_file_path)
    return await transcribe_with_parakeet_cpp_array(audio, sr, model_id, quant)


# --- NanoGPT Settings Helper ---
def _load_nanogpt_settings() -> dict:
    """Load settings from ~/.LiangLocal/settings.json (same pattern as TTS service)."""
    from pathlib import Path
    import json
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load settings from {settings_path}: {e}")
    return {}


# FFmpeg decode only when librosa/audioread fails (file uploads), not on every mic clip.
_STT_FFMPEG_FALLBACK_EXTENSIONS = {
    ".webm", ".opus", ".ogg", ".m4a", ".aac", ".mp4", ".mkv", ".avi", ".mov",
}


def _convert_to_wav_with_ffmpeg(input_path: str) -> str:
    from .ffmpeg_utils import FFMPEG_INSTALL_HINT, find_ffmpeg
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is required to transcribe WebM/Opus recordings. " + FFMPEG_INSTALL_HINT
        )
    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        temp_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        try:
            os.remove(temp_wav)
        except OSError:
            pass
        err = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"FFmpeg could not decode audio: {err[:500]}")
    return temp_wav


def _read_wav_bytes(data: bytes) -> tuple:
    """In-memory 16 kHz mono float32 — fast path for browser WAV mic uploads."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="kaiser_fast")
    return audio, 16000


def _load_stt_audio_path(audio_file_path: str) -> tuple:
    """Load audio at 16 kHz; FFmpeg subprocess only if librosa cannot decode."""
    try:
        audio, sr = librosa.load(
            audio_file_path,
            sr=16000,
            mono=True,
            duration=None,
            res_type="kaiser_fast",
        )
        return audio, sr
    except Exception as first_err:
        ext = Path(audio_file_path).suffix.lower()
        if ext not in _STT_FFMPEG_FALLBACK_EXTENSIONS:
            raise
        logger.warning(
            "librosa could not decode %s (%s); one-shot FFmpeg fallback",
            audio_file_path,
            first_err,
        )
        temp_wav = _convert_to_wav_with_ffmpeg(audio_file_path)
        try:
            audio, sr = librosa.load(
                temp_wav,
                sr=16000,
                mono=True,
                duration=None,
                res_type="kaiser_fast",
            )
            return audio, sr
        finally:
            try:
                os.remove(temp_wav)
            except OSError:
                pass


def load_whisper_model():
    """Loads the Processor and Model using transformers if not already loaded."""
    global whisper_processor, whisper_model
    if whisper_model is None or whisper_processor is None:
        try:
            logger.info(f"Loading HF STT Processor and Model '{WHISPER_MODEL_ID}' onto device {DEVICE}...")
            # Use float16 on GPU for potential VRAM savings/speedup
            processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
                low_cpu_mem_usage=True, # Can help on systems with lower RAM
                use_safetensors=True
            ).to(DEVICE)

            whisper_processor = processor
            whisper_model = model
            logger.info(f"HF STT Processor and Model '{WHISPER_MODEL_ID}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HF STT model '{WHISPER_MODEL_ID}': {e}", exc_info=True)
            whisper_processor = None
            whisper_model = None
    # Return both, even if one failed (will be None)
    return whisper_processor, whisper_model


def _prepare_nemo_env() -> None:
    os.environ["NEMO_DISABLE_ONELOGGER"] = "True"
    os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
    for noisy_logger in [
        "nemo", "nemo_logging", "torch.distributed", "lhotse", "torio",
        "pytorch_lightning", "nv_one_logger", "nemo.utils.import_utils",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
        logging.getLogger(noisy_logger).propagate = False


def _import_nemo_asr():
    try:
        import nemo.collections.asr as nemo_asr
        return nemo_asr
    except (ImportError, TypeError) as e:
        logger.info(f"NeMo toolkit not found or failed to load ({e}). Attempting to install/fix automatically...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nemo_toolkit[asr]"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2"])
            logger.warning("PARAKEET FIX APPLIED: NumPy may have been downgraded — restart backend if needed.")
            importlib.invalidate_caches()
            import nemo.collections.asr as nemo_asr
            return nemo_asr
        except Exception as install_err:
            logger.error(f"Failed to automatically install NeMo: {install_err}")
            return None


def _load_parakeet_tdt_checkpoint(model_id: str):
    """Load an NVIDIA Parakeet TDT checkpoint (v2 English or v3 multilingual)."""
    _prepare_nemo_env()
    nemo_asr = _import_nemo_asr()
    if nemo_asr is None:
        return None
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.set_verbosity(nemo_logging.ERROR)
    except Exception:
        pass
    logger.info(f"Loading NVIDIA Parakeet TDT '{model_id}' onto device {DEVICE}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location="cpu")
    if DEVICE.startswith("cuda"):
        asr_model = asr_model.to(DEVICE)
    logger.info(f"NVIDIA Parakeet TDT '{model_id}' loaded successfully.")
    return asr_model


def load_parakeet_model():
    """Loads Parakeet TDT v2 (English)."""
    global parakeet_model
    if parakeet_model is None:
        try:
            print(f"[DEBUG] Loading Parakeet model: {PARAKEET_MODEL_ID} on device {DEVICE}", flush=True)
            parakeet_model = _load_parakeet_tdt_checkpoint(PARAKEET_MODEL_ID)
            print(f"[DEBUG] Parakeet model loaded: {parakeet_model is not None}", flush=True)
            if parakeet_model:
                print(f"[DEBUG] Parakeet model device: {next(parakeet_model.parameters()).device}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Parakeet load exception: {e}", flush=True)
            logger.error(f"Failed to load Parakeet v2 '{PARAKEET_MODEL_ID}': {e}", exc_info=True)
            parakeet_model = None
    else:
        print(f"[DEBUG] Parakeet model already loaded: {parakeet_model is not None}", flush=True)
    return parakeet_model


def load_parakeet_v3_model():
    """Loads Parakeet TDT v3 (multilingual, auto language detect)."""
    global parakeet_v3_model
    if parakeet_v3_model is None:
        try:
            parakeet_v3_model = _load_parakeet_tdt_checkpoint(PARAKEET_V3_MODEL_ID)
        except Exception as e:
            logger.error(f"Failed to load Parakeet v3 '{PARAKEET_V3_MODEL_ID}': {e}", exc_info=True)
            parakeet_v3_model = None
    return parakeet_v3_model


def load_parakeet_zh_model():
    """Loads the NVIDIA NeMo Mandarin Chinese ASR model (Conformer-Transducer Large)."""
    global parakeet_zh_model
    if parakeet_zh_model is None:
        os.environ.setdefault("NEMO_DISABLE_ONELOGGER", "True")
        os.environ.setdefault("NEMO_LOGGING_LEVEL", "ERROR")
        for noisy_logger in [
            "nemo", "nemo_logging", "torch.distributed", "lhotse", "torio",
            "pytorch_lightning", "nv_one_logger", "nemo.utils.import_utils",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)
            logging.getLogger(noisy_logger).propagate = False

        try:
            try:
                import nemo.collections.asr as nemo_asr
            except (ImportError, TypeError) as e:
                logger.info(f"NeMo not found ({e}). Attempting install...")
                try:
                    subprocess = __import__("subprocess")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "nemo_toolkit[asr]"])
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2"])
                    importlib.invalidate_caches()
                    import nemo.collections.asr as nemo_asr
                except Exception as install_err:
                    logger.error(f"Failed to install NeMo: {install_err}")
                    return None

            try:
                from nemo.utils import logging as nemo_logging
                nemo_logging.set_verbosity(nemo_logging.ERROR)
            except Exception:
                pass

            logger.info(f"Loading NeMo Chinese ASR model '{PARAKEET_ZH_MODEL_ID}' onto device {DEVICE}...")
            # EncDecRNNTModel for Conformer-Transducer (not ASRModel)
            asr_model = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name=PARAKEET_ZH_MODEL_ID, map_location="cpu")
            if DEVICE.startswith("cuda"):
                asr_model = asr_model.to(DEVICE)
            parakeet_zh_model = asr_model
            logger.info(f"NeMo Chinese ASR model '{PARAKEET_ZH_MODEL_ID}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NeMo Chinese ASR model '{PARAKEET_ZH_MODEL_ID}': {e}", exc_info=True)
            parakeet_zh_model = None
    return parakeet_zh_model


def load_nemotron_model():
    """Loads NVIDIA Nemotron Speech Streaming (English ASR)."""
    global nemotron_model
    if nemotron_model is None:
        try:
            nemotron_model = _load_parakeet_tdt_checkpoint(NEMOTRON_MODEL_ID)
        except Exception as e:
            logger.error(f"Failed to load Nemotron '{NEMOTRON_MODEL_ID}': {e}", exc_info=True)
            nemotron_model = None
    return nemotron_model


def _get_moonshine_python() -> str:
    """Path to the isolated Moonshine venv Python executable."""
    backend_dir = Path(__file__).resolve().parent.parent
    venv_python = backend_dir / "moonshine_env" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return None


def _get_moonshine_worker_script() -> str:
    """Path to moonshine_worker.py."""
    return str(Path(__file__).resolve().parent / "moonshine_worker.py")


def is_moonshine_available() -> bool:
    """Check if Moonshine isolated venv exists."""
    return _get_moonshine_python() is not None


# Persistent Moonshine worker process
_moonshine_worker_process = None
_moonshine_worker_lock = None


def _ensure_moonshine_worker():
    """Ensure persistent Moonshine worker is running. Returns the process."""
    global _moonshine_worker_process, _moonshine_worker_lock
    
    if _moonshine_worker_lock is None:
        import threading
        _moonshine_worker_lock = threading.Lock()
    
    with _moonshine_worker_lock:
        # Check if worker is still alive
        if _moonshine_worker_process is not None and _moonshine_worker_process.poll() is None:
            return _moonshine_worker_process
        
        # Worker not running or crashed, start new one
        moonshine_python = _get_moonshine_python()
        if not moonshine_python:
            raise RuntimeError("Moonshine venv not found")
        
        worker_script = _get_moonshine_worker_script()
        logger.info("Starting persistent Moonshine worker...")
        
        # Redirect stderr to a log file to prevent buffer blocking
        log_path = Path(__file__).resolve().parent.parent / "moonshine_worker.log"
        stderr_file = open(log_path, "a", encoding="utf-8")
        
        _moonshine_worker_process = subprocess.Popen(
            [moonshine_python, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        
        logger.info(f"Moonshine worker started (PID: {_moonshine_worker_process.pid})")
        return _moonshine_worker_process


async def setup_moonshine_venv(progress_callback=None) -> tuple[bool, str]:
    """Auto-create Moonshine venv with transformers 5.x. Returns (success, message)."""
    backend_dir = Path(__file__).resolve().parent.parent
    venv_dir = backend_dir / "moonshine_env"
    venv_python = venv_dir / "Scripts" / "python.exe"

    if venv_python.exists():
        return True, "Moonshine venv already exists"

    try:
        if progress_callback:
            await progress_callback("Creating Moonshine virtual environment...")

        def _create_venv():
            import subprocess
            import sys

            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _create_venv)

        if progress_callback:
            await progress_callback("Installing Moonshine dependencies (this may take a few minutes)...")

        def _install_deps():
            import subprocess
            pip_exe = str(venv_dir / "Scripts" / "pip.exe")

            subprocess.run(
                [pip_exe, "install", "torch==2.11.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128"],
                check=True,
                capture_output=True,
                text=True,
                timeout=600
            )

            subprocess.run(
                [pip_exe, "install", "transformers>=5.0.0", "librosa", "soundfile", "numpy", "huggingface-hub", "safetensors"],
                check=True,
                capture_output=True,
                text=True,
                timeout=600
            )

        await loop.run_in_executor(None, _install_deps)

        if progress_callback:
            await progress_callback("Moonshine setup complete!")

        return True, "Moonshine venv created successfully"

    except subprocess.CalledProcessError as e:
        error_msg = f"Setup failed: {e.stderr or e.stdout}"
        logger.error(error_msg)
        if venv_dir.exists():
            import shutil
            shutil.rmtree(venv_dir, ignore_errors=True)
        return False, error_msg
    except Exception as e:
        error_msg = f"Setup error: {str(e)}"
        logger.error(error_msg)
        if venv_dir.exists():
            import shutil
            shutil.rmtree(venv_dir, ignore_errors=True)
        return False, error_msg


async def transcribe_audio(audio_file_path: str, engine: str = "whisper") -> str:
    """Transcribes audio using the selected STT engine."""
    logger.info(f"Transcribing using engine: {engine}")
    audio, sr = _load_stt_audio_path(audio_file_path)
    return await _transcribe_audio_array(audio, sr, engine)


async def transcribe_audio_bytes(data: bytes, engine: str = "whisper") -> str:
    """Fast path: browser sends 16 kHz WAV bytes (no WebM decode, no disk write)."""
    if len(data) >= 4 and data[:4] == b"RIFF":
        audio, sr = _read_wav_bytes(data)
        return await _transcribe_audio_array(audio, sr, engine)
    suffix = ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)
        return await transcribe_audio(path, engine)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def _transcribe_audio_array(audio, sr: int, engine: str) -> str:
    if engine == "whisper":
        return await transcribe_with_whisper_array(audio, sr)
    if engine == "parakeet":
        model = load_parakeet_model()
        if model:
            return await transcribe_with_parakeet_array(audio, sr)
        logger.warning("Parakeet model failed to load, falling back to Whisper")
        return await transcribe_with_whisper_array(audio, sr)
    if engine == "parakeet-v3":
        model = load_parakeet_v3_model()
        if model:
            return await transcribe_with_parakeet_v3_array(audio, sr)
        logger.warning("Parakeet v3 model failed to load, falling back to Whisper")
        return await transcribe_with_whisper_array(audio, sr)
    if engine == "parakeet-zh":
        zh_model = load_parakeet_zh_model()
        if zh_model:
            return await transcribe_with_parakeet_zh_array(audio, sr)
        logger.warning("Parakeet-ZH model failed to load, falling back to Whisper")
        return await transcribe_with_whisper_array(audio, sr)
    if engine == "nemotron":
        model = load_nemotron_model()
        if model:
            return await transcribe_with_nemotron_array(audio, sr)
        logger.warning("Nemotron model failed to load, falling back to Whisper")
        return await transcribe_with_whisper_array(audio, sr)
    if engine == "moonshine":
        if is_moonshine_available():
            return await transcribe_with_moonshine_array(audio, sr)
        logger.warning("Moonshine venv not available, falling back to Whisper")
        return await transcribe_with_whisper_array(audio, sr)
    if engine.startswith("parakeet-cpp"):
        parts = engine.split(":", 1)
        model_ref = parts[1] if len(parts) > 1 else "tdt_ctc-110m:f16"
        model_parts = model_ref.split(":", 1)
        m_id = model_parts[0]
        m_quant = model_parts[1] if len(model_parts) > 1 else "f16"
        return await transcribe_with_parakeet_cpp_array(audio, sr, m_id, m_quant)
    if engine.startswith("nanogpt-"):
        return await transcribe_with_nanogpt_array(audio, sr, engine)
    logger.warning(f"Unknown STT engine: {engine}, falling back to Whisper")
    return await transcribe_with_whisper_array(audio, sr)


async def transcribe_with_whisper(audio_file_path: str) -> str:
    audio, sr = _load_stt_audio_path(audio_file_path)
    return await transcribe_with_whisper_array(audio, sr)


async def transcribe_with_whisper_array(audio_input, sampling_rate: int) -> str:
    """Transcribes audio using the loaded HF Whisper model."""
    processor, model = load_whisper_model()
    if not processor or not model:
        raise RuntimeError("STT processor or model is not loaded.")

    try:
        audio_duration = len(audio_input) / sampling_rate
        logger.info(f"Whisper audio ready. Sample rate: {sampling_rate}, Duration: {audio_duration:.2f}s")

        if audio_duration > 30:
            logger.info(f"Long audio detected ({audio_duration:.2f}s), using chunking approach")
            chunk_size = 20 * sampling_rate
            overlap = 2 * sampling_rate
            transcripts = []

            for i in range(0, len(audio_input), chunk_size - overlap):
                chunk = audio_input[i:i + chunk_size]
                if len(chunk) < sampling_rate * 1:
                    continue
                features = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt").input_features
                features = features.to(DEVICE)
                if DEVICE.startswith("cuda"):
                    features = features.half()
                with torch.no_grad():
                    pred_ids = model.generate(features, max_new_tokens=256, language="en")
                chunk_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                if chunk_text:
                    transcripts.append(chunk_text)

            transcript_text = " ".join(transcripts)
        else:
            input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
            input_features = input_features.to(DEVICE)
            if DEVICE.startswith("cuda"):
                input_features = input_features.half()
            with torch.no_grad():
                predicted_ids = model.generate(input_features, max_new_tokens=256, language="en")
            transcript_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        logger.info(f"Whisper transcription complete. Output length: {len(transcript_text)}")
        return transcript_text

    except Exception as e:
        logger.error(f"Error during HF transcription: {e}", exc_info=True)
        raise RuntimeError(f"HF Transcription failed: {str(e)}")


def _parakeet_chunk_paths(audio, sr: int):
    """Write NeMo-ready WAV chunk file paths. Uses ignore_cleanup_errors to handle Windows file-lock races."""
    audio_duration = len(audio) / sr
    temp_wav_path = None
    temp_dir = None
    chunk_paths = []

    if audio_duration > 30:
        chunk_size = 20 * sr
        overlap = 2 * sr
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        for i in range(0, len(audio), chunk_size - overlap):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < sr * 1:
                continue
            chunk_path = os.path.join(temp_dir.name, f"chunk_{i}.wav")
            sf.write(chunk_path, chunk, sr)
            chunk_paths.append(chunk_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav_path = tmp.name
        sf.write(temp_wav_path, audio, sr)
        chunk_paths.append(temp_wav_path)

    return chunk_paths, temp_wav_path, temp_dir


async def transcribe_with_parakeet(audio_file_path: str) -> str:
    if not parakeet_model:
        raise RuntimeError("Parakeet model is not loaded.")
    audio, sr = _load_stt_audio_path(audio_file_path)
    return await transcribe_with_parakeet_array(audio, sr)


async def transcribe_with_parakeet_v3(audio_file_path: str) -> str:
    if not parakeet_v3_model:
        raise RuntimeError("Parakeet v3 model is not loaded.")
    audio, sr = _load_stt_audio_path(audio_file_path)
    return await transcribe_with_parakeet_v3_array(audio, sr)


async def _transcribe_parakeet_tdt_array(audio, sr: int, model, engine_label: str) -> str:
    temp_wav_path = None
    temp_dir = None
    try:
        audio_duration = len(audio) / sr
        logger.info(f"{engine_label} audio ready. Duration: {audio_duration:.2f}s")

        chunk_paths, temp_wav_path, temp_dir = _parakeet_chunk_paths(audio, sr)
        logger.info(f"Transcribing with {engine_label} ({len(chunk_paths)} chunk(s))")

        torch.cuda.empty_cache()
        result = model.transcribe(chunk_paths)

        transcripts = []
        for item in result:
            if isinstance(item, str):
                text = item.strip()
            else:
                text = getattr(item, "text", None) or getattr(item, "transcript", None)
                text = text.strip() if isinstance(text, str) else ""
            if text:
                transcripts.append(text)

        transcript_text = " ".join(transcripts).strip()
        logger.info(f"{engine_label} transcription complete. Output length: {len(transcript_text)}")
        return transcript_text

    except Exception as e:
        logger.error(f"Error during {engine_label} transcription: {e}", exc_info=True)
        raise RuntimeError(f"{engine_label} transcription failed: {str(e)}")
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass
        if temp_dir:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


async def transcribe_with_parakeet_array(audio, sr: int) -> str:
    """Transcribes audio using Parakeet TDT v2 (English)."""
    if not parakeet_model:
        raise RuntimeError("Parakeet model is not loaded.")
    return await _transcribe_parakeet_tdt_array(audio, sr, parakeet_model, "Parakeet")


async def transcribe_with_parakeet_v3_array(audio, sr: int) -> str:
    """Transcribes audio using Parakeet TDT v3 (multilingual)."""
    if not parakeet_v3_model:
        raise RuntimeError("Parakeet v3 model is not loaded.")
    return await _transcribe_parakeet_tdt_array(audio, sr, parakeet_v3_model, "Parakeet v3")


async def transcribe_with_nemotron(audio_file_path: str) -> str:
    if not nemotron_model:
        raise RuntimeError("Nemotron model is not loaded.")
    audio, sr = _load_stt_audio_path(audio_file_path)
    return await transcribe_with_nemotron_array(audio, sr)


async def transcribe_with_nemotron_array(audio, sr: int) -> str:
    """Transcribes audio using Nemotron Speech Streaming (English)."""
    if not nemotron_model:
        raise RuntimeError("Nemotron model is not loaded.")
    return await _transcribe_parakeet_tdt_array(audio, sr, nemotron_model, "Nemotron")


async def transcribe_with_moonshine(audio_file_path: str) -> str:
    return await transcribe_with_moonshine_subprocess(audio_file_path)


async def transcribe_with_moonshine_array(audio, sr: int) -> str:
    """Transcribes audio using Moonshine via isolated subprocess."""
    temp_wav_path = None
    try:
        temp_wav_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_wav_path, audio, sr)
        return await transcribe_with_moonshine_subprocess(temp_wav_path)
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


async def transcribe_with_moonshine_subprocess(audio_file_path: str) -> str:
    """Call persistent Moonshine worker for transcription. Auto-sets up venv if needed."""
    moonshine_python = _get_moonshine_python()
    
    if not moonshine_python:
        logger.info("Moonshine venv not found, auto-setting up...")
        success, message = await setup_moonshine_venv()
        if not success:
            raise RuntimeError(f"Moonshine setup failed: {message}")
        moonshine_python = _get_moonshine_python()
        if not moonshine_python:
            raise RuntimeError("Moonshine setup completed but venv not found")

    audio_path_absolute = str(Path(audio_file_path).resolve())
    payload = json.dumps({"audio_path": audio_path_absolute, "sample_rate": 16000})

    loop = asyncio.get_event_loop()

    def _run():
        global _moonshine_worker_process
        
        # Get or create persistent worker
        worker = _ensure_moonshine_worker()
        
        # Send request
        worker.stdin.write(payload + "\n")
        worker.stdin.flush()
        
        # Read response
        output_line = worker.stdout.readline().strip()
        if not output_line:
            # Worker might have crashed, try to read stderr
            stderr_output = worker.stderr.read()
            raise RuntimeError(f"Moonshine worker returned empty output. Stderr: {stderr_output}")
        
        result = json.loads(output_line)
        if not result.get("ok"):
            raise RuntimeError(f"Moonshine error: {result.get('error', 'unknown')}")
        return result.get("transcript", "")

    transcript = await loop.run_in_executor(None, _run)
    logger.info(f"Moonshine transcription complete. Output length: {len(transcript)}")
    return transcript


async def transcribe_with_parakeet_zh(audio_file_path: str) -> str:
    if not parakeet_zh_model:
        raise RuntimeError("Parakeet-ZH (Chinese) model is not loaded.")
    audio, sr = _load_stt_audio_path(audio_file_path)
    return await transcribe_with_parakeet_zh_array(audio, sr)


async def transcribe_with_parakeet_zh_array(audio, sr: int) -> str:
    """Transcribes audio using the NeMo Mandarin Chinese ASR model (16 kHz mono)."""
    global parakeet_zh_model
    if not parakeet_zh_model:
        raise RuntimeError("Parakeet-ZH (Chinese) model is not loaded.")

    temp_wav_path = None
    temp_dir = None
    try:
        audio_duration = len(audio) / sr
        logger.info(f"Parakeet-ZH audio ready. Duration: {audio_duration:.2f}s")

        chunk_paths, temp_wav_path, temp_dir = _parakeet_chunk_paths(audio, sr)
        logger.info(f"Transcribing with Parakeet-ZH ({len(chunk_paths)} chunk(s))")
        raw_result = parakeet_zh_model.transcribe(chunk_paths)
        # RNNT transcribe can return (hypotheses, batch_lengths) tuple; use first element
        if isinstance(raw_result, (list, tuple)) and len(raw_result) == 2:
            result = raw_result[0]
        else:
            result = raw_result if isinstance(raw_result, (list, tuple)) else [raw_result]

        transcripts = []
        for item in result:
            # Only use .text or .transcript; never str(item) or we get Hypothesis repr dump
            text = getattr(item, "text", None) or getattr(item, "transcript", None)
            if isinstance(text, str):
                text = text.strip()
            else:
                text = ""
            if text:
                transcripts.append(text)
        transcript_text = " ".join(transcripts).strip()
        logger.info(f"Parakeet-ZH transcription complete. Output length: {len(transcript_text)}")
        return transcript_text
    except Exception as e:
        logger.error(f"Error during Parakeet-ZH transcription: {e}", exc_info=True)
        raise RuntimeError(f"Parakeet-ZH transcription failed: {str(e)}")
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass
        if temp_dir:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


async def transcribe_with_nanogpt_array(audio, sr: int, engine: str) -> str:
    """Transcribe audio using NanoGPT cloud STT API."""
    # Extract model ID from engine prefix (e.g., "nanogpt-fun-asr-flash-2026-06-15" -> "fun-asr-flash-2026-06-15")
    model_id = engine[len("nanogpt-"):] if engine.startswith("nanogpt-") else NANOGPT_DEFAULT_STT_MODEL
    
    # Load settings to get API key
    settings = _load_nanogpt_settings()
    api_key = settings.get('nanogpt_api_key') or settings.get('nanoGptApiKey')
    
    if not api_key:
        logger.error("❌ NanoGPT API key not configured in settings")
        raise RuntimeError("NanoGPT API key not configured. Please add it in Settings.")
    
    # Write audio to temp WAV file for upload
    import tempfile
    import soundfile as sf
    import subprocess
    temp_wav_path = None
    temp_opus_path = None
    try:
        fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_wav_path, audio, sr)
        
        # Check file size - NanoGPT direct upload limit is 3MB
        file_size = os.path.getsize(temp_wav_path)
        MAX_DIRECT_UPLOAD_SIZE = 3 * 1024 * 1024  # 3MB
        
        logger.info(f"🎵 Transcribing with NanoGPT model: {model_id} (file: {file_size/1024:.1f}KB)")
        
        upload_path = temp_wav_path
        upload_mime = "audio/wav"
        
        if file_size > MAX_DIRECT_UPLOAD_SIZE:
            # Compress with ffmpeg to Opus (~24kbps, ~10x smaller than WAV)
            logger.info(f"📦 File exceeds 3MB, compressing to Opus...")
            temp_opus_path = tempfile.mktemp(suffix=".opus")
            
            from .ffmpeg_utils import FFMPEG_INSTALL_HINT, find_ffmpeg
            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                raise RuntimeError("FFmpeg required to compress audio for NanoGPT. " + FFMPEG_INSTALL_HINT)
            
            # Opus at 24kbps mono - excellent for speech, ~10x compression
            cmd = [
                ffmpeg_path,
                "-y",
                "-v", "quiet",
                "-i", temp_wav_path,
                "-acodec", "libopus",
                "-b:a", "16k",
                "-compression_level", "0",
                "-frame_duration", "60",
                "-ac", "1",
                "-ar", "16000",
                temp_opus_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg Opus compression failed: {result.stderr[:500]}")
            
            compressed_size = os.path.getsize(temp_opus_path)
            logger.info(f"✅ Compressed: {file_size/1024:.1f}KB → {compressed_size/1024:.1f}KB ({file_size/compressed_size:.1f}x)")
            
            upload_path = temp_opus_path
            upload_mime = "audio/opus"
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)) as client:
            with open(upload_path, "rb") as f:
                files = {"audio": (os.path.basename(upload_path), f, upload_mime)}
                data = {
                    "model": model_id,
                    "language": "auto",
                }
                headers = {"x-api-key": api_key}
                
                response = await client.post(
                    NANOGPT_STT_ENDPOINT,
                    headers=headers,
                    files=files,
                    data=data,
                )
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get("transcription", "").strip()
            logger.info(f"✅ NanoGPT transcription complete. Length: {len(transcript)} chars")
            return transcript
        elif response.status_code == 202:
            # Async job - poll for result
            job_data = response.json()
            return await _poll_nanogpt_transcription(client, job_data, api_key)
        else:
            error_text = response.text
            logger.error(f"❌ NanoGPT STT API error {response.status_code}: {error_text}")
            raise RuntimeError(f"NanoGPT STT failed: {response.status_code} - {error_text}")
            
    except httpx.TimeoutException:
        logger.error("❌ NanoGPT STT request timed out")
        raise RuntimeError("NanoGPT STT request timed out")
    except Exception as e:
        logger.error(f"❌ NanoGPT STT error: {e}", exc_info=True)
        raise RuntimeError(f"NanoGPT STT failed: {str(e)}")
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass
        if temp_opus_path and os.path.exists(temp_opus_path):
            try:
                os.remove(temp_opus_path)
            except Exception:
                pass


async def _poll_nanogpt_transcription(client: httpx.AsyncClient, job_data: dict, api_key: str) -> str:
    """Poll NanoGPT transcription status endpoint until complete."""
    import time
    run_id = job_data.get("runId")
    if not run_id:
        raise RuntimeError("No runId returned from NanoGPT async transcription")
    
    status_url = f"{NANOGPT_API_BASE}/transcribe/status"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    
    status_data = {
        "runId": run_id,
        "cost": job_data.get("cost"),
        "paymentSource": job_data.get("paymentSource"),
        "isApiRequest": True,
        "fileName": job_data.get("fileName"),
        "fileSize": job_data.get("fileSize"),
        "chargedDuration": job_data.get("chargedDuration"),
        "diarize": job_data.get("diarize", False),
    }
    
    max_attempts = 60
    for attempt in range(max_attempts):
        await asyncio.sleep(5)
        try:
            response = await client.post(status_url, headers=headers, json=status_data)
            if response.status_code == 200:
                result = response.json()
                status = result.get("status")
                if status == "completed":
                    transcript = result.get("transcription", "").strip()
                    logger.info(f"✅ NanoGPT async transcription complete. Length: {len(transcript)} chars")
                    return transcript
                elif status == "failed":
                    error = result.get("error", "Unknown error")
                    logger.error(f"❌ NanoGPT transcription failed: {error}")
                    raise RuntimeError(f"NanoGPT transcription failed: {error}")
                # Still processing, continue polling
            else:
                logger.warning(f"Status poll returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"Status poll error (attempt {attempt + 1}): {e}")
    
    raise RuntimeError("NanoGPT transcription timed out")


def _load_nanogpt_settings() -> dict:
    """Load settings from ~/.LiangLocal/settings.json (same pattern as TTS service)."""
    from pathlib import Path
    import json
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load settings from {settings_path}: {e}")
    return {}


def is_engine_available(engine: str) -> bool:
    """Check if the specified STT engine is available."""
    if engine == "whisper":
        processor, model = load_whisper_model()
        return processor is not None and model is not None
    elif engine in PARAKEET_TDT_ENGINE_IDS:
        try:
            import importlib.util
            if importlib.util.find_spec("nemo") is None:
                logger.info("NeMo package not found")
                return False
            if importlib.util.find_spec("nemo.collections.asr") is None:
                logger.info("NeMo ASR module not found")
                return False
            try:
                import nemo.collections.asr
                return True
            except Exception as e:
                logger.info(f"Error importing NeMo ASR: {e}")
                return False
        except ImportError:
            logger.info("Import error checking for Parakeet")
            return False
    elif engine == "parakeet-zh":
        try:
            import importlib.util
            if importlib.util.find_spec("nemo") is None:
                return False
            if importlib.util.find_spec("nemo.collections.asr") is None:
                return False
            try:
                import nemo.collections.asr as nemo_asr
                # EncDecRNNTModel must be available for Chinese model
                _ = getattr(nemo_asr.models, "EncDecRNNTModel", None)
                return _ is not None
            except Exception as e:
                logger.info(f"Error checking Parakeet-ZH: {e}")
                return False
        except ImportError:
            return False
    elif engine == "nemotron":
        # Nemotron uses the same NeMo ASRModel loading as Parakeet TDT
        try:
            import importlib.util
            if importlib.util.find_spec("nemo") is None:
                return False
            if importlib.util.find_spec("nemo.collections.asr") is None:
                return False
            try:
                import nemo.collections.asr
                return True
            except Exception:
                return False
        except ImportError:
            return False
    elif engine == "moonshine":
        return True
    elif engine.startswith("parakeet-cpp"):
        return is_parakeet_cpp_available()
    elif engine.startswith("nanogpt-"):
        # NanoGPT engines are available if API key is configured
        settings = _load_nanogpt_settings()
        api_key = settings.get('nanogpt_api_key') or settings.get('nanoGptApiKey')
        return api_key is not None and len(api_key.strip()) > 0
    return False


def list_available_engines() -> list:
    """Returns a list of available STT engines."""
    engines = []
    if is_engine_available("whisper"):
        engines.append("whisper")
    if is_engine_available("parakeet"):
        engines.append("parakeet")
    if is_engine_available("parakeet-v3"):
        engines.append("parakeet-v3")
    if is_engine_available("parakeet-zh"):
        engines.append("parakeet-zh")
    if is_engine_available("nemotron"):
        engines.append("nemotron")
    if is_engine_available("moonshine"):
        engines.append("moonshine")
    if is_engine_available("parakeet-cpp"):
        engines.append("parakeet-cpp")
    # Add NanoGPT models if API key is configured
    settings = _load_nanogpt_settings()
    api_key = settings.get('nanogpt_api_key') or settings.get('nanoGptApiKey')
    if api_key and len(api_key.strip()) > 0:
        for model_id in NANOGPT_STT_MODELS.keys():
            engines.append(f"nanogpt-{model_id}")
    return engines
