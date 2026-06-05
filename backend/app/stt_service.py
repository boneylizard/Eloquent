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
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- STT Models ---
whisper_processor = None
whisper_model = None
parakeet_model = None
parakeet_v3_model = None
parakeet_zh_model = None  # NeMo Mandarin Chinese ASR

def get_device():
    """Determines the correct Torch device for the current process."""
    try:
        import torch
        if torch.cuda.is_available():
            # Since launch.py isolates the process to a single GPU,
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

PARAKEET_TDT_ENGINE_IDS = frozenset({"parakeet", "parakeet-v3"})

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
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    if DEVICE.startswith("cuda"):
        asr_model = asr_model.to(DEVICE)
    logger.info(f"NVIDIA Parakeet TDT '{model_id}' loaded successfully.")
    return asr_model


def load_parakeet_model():
    """Loads Parakeet TDT v2 (English)."""
    global parakeet_model
    if parakeet_model is None:
        try:
            parakeet_model = _load_parakeet_tdt_checkpoint(PARAKEET_MODEL_ID)
        except Exception as e:
            logger.error(f"Failed to load Parakeet v2 '{PARAKEET_MODEL_ID}': {e}", exc_info=True)
            parakeet_model = None
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
            asr_model = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name=PARAKEET_ZH_MODEL_ID)
            if DEVICE.startswith("cuda"):
                asr_model = asr_model.to(DEVICE)
            parakeet_zh_model = asr_model
            logger.info(f"NeMo Chinese ASR model '{PARAKEET_ZH_MODEL_ID}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NeMo Chinese ASR model '{PARAKEET_ZH_MODEL_ID}': {e}", exc_info=True)
            parakeet_zh_model = None
    return parakeet_zh_model


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
    """Write NeMo-ready WAV chunk file paths (Parakeet API needs paths, not WebM)."""
    audio_duration = len(audio) / sr
    temp_wav_path = None
    temp_dir = None
    chunk_paths = []

    if audio_duration > 30:
        chunk_size = 20 * sr
        overlap = 2 * sr
        temp_dir = tempfile.TemporaryDirectory()
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

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(chunk_paths),
        )

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
            except Exception as cleanup_err:
                logger.warning(f"Could not delete temp file {temp_wav_path}: {cleanup_err}")
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
        loop = asyncio.get_event_loop()
        raw_result = await loop.run_in_executor(
            None,
            lambda: parakeet_zh_model.transcribe(chunk_paths),
        )
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
    return engines
