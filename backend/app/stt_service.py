# app/stt_service.py
# --- Multiple STT engines support (Whisper and Parakeet) ---
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import soundfile as sf
import tempfile
import contextlib
import logging
import os
import sys
import importlib
import asyncio
import shutil

logger = logging.getLogger(__name__)

# --- STT Models ---
whisper_processor = None
whisper_model = None
parakeet_model = None

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

def load_parakeet_model():
    """Loads the NVIDIA Parakeet model using NeMo."""
    global parakeet_model
    if parakeet_model is None:
        # Crucial: Disable NeMo's buggy OneLogger telemetry integration
        # This prevents the TypeError: `OneLoggerPTLTrainer.save_checkpoint: weights_only must be a supertype...`
        os.environ["NEMO_DISABLE_ONELOGGER"] = "True"
        # Silence NeMo's internal logging level via environment variable
        os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
        
        # Aggressively suppress noisy/verbose logs from NeMo and its internal dependencies
        # We target specific sub-modules that are known to be chatty during setup/inference
        for noisy_logger in [
            'nemo', 'nemo_logging', 'torch.distributed', 'lhotse', 'torio', 
            'pytorch_lightning', 'nv_one_logger', 'nemo.utils.import_utils'
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)
            logging.getLogger(noisy_logger).propagate = False
        
        try:
            # Try to import nemo
            try:
                import nemo.collections.asr as nemo_asr
            except (ImportError, TypeError) as e:
                logger.info(f"NeMo toolkit not found or failed to load ({e}). Attempting to install/fix automatically...")
                try:
                    import subprocess
                    import sys
                    
                    # Ensure pip is up to date
                    logger.info("Ensuring pip is up to date...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

                    # Then install NeMo toolkit with ASR support
                    logger.info("Installing NeMo toolkit with ASR support...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "nemo_toolkit[asr]"
                    ])
                    
                    # FORCE FIX: Installation often pulls in numpy 2.x which breaks stuff
                    # So we explicitly downgrade it immediately after
                    logger.info("Applying NumPy compatibility fix (numpy<2)...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "numpy<2"
                    ])
                    logger.warning("************************************************************")
                    logger.warning("PARAKEET FIX APPLIED: NumPy has been downgraded.")
                    logger.warning("YOU MUST RESTART THE BACKEND FOR THIS TO TAKE EFFECT.")
                    logger.warning("************************************************************")
                    
                    logger.info("NeMo toolkit installed successfully!")
                    
                    # Now import nemo after installation
                    importlib.invalidate_caches()
                    import nemo.collections.asr as nemo_asr
                except Exception as install_err:
                    logger.error(f"Failed to automatically install NeMo: {install_err}")
                    return None
            
            # Additional silence for NeMo's custom logging system if it exists
            try:
                from nemo.utils import logging as nemo_logging
                nemo_logging.set_verbosity(nemo_logging.ERROR)
            except:
                pass

            logger.info(f"Loading NVIDIA Parakeet model '{PARAKEET_MODEL_ID}' onto device {DEVICE}...")
            # Load pre-trained model (this will download the model if not present)
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL_ID)
            
            # Move to the correct device
            if DEVICE.startswith("cuda"):
                asr_model = asr_model.to(DEVICE)
                
            parakeet_model = asr_model
            logger.info(f"NVIDIA Parakeet model '{PARAKEET_MODEL_ID}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NVIDIA Parakeet model '{PARAKEET_MODEL_ID}': {e}", exc_info=True)
            parakeet_model = None
    
    return parakeet_model

async def transcribe_audio(audio_file_path: str, engine: str = "whisper") -> str:
    """Transcribes audio using the selected STT engine."""
    logger.info(f"Transcribing using engine: {engine}")
    
    if engine == "whisper":
        return await transcribe_with_whisper(audio_file_path)
    elif engine == "parakeet":
        parakeet_model = load_parakeet_model()
        if parakeet_model:
            return await transcribe_with_parakeet(audio_file_path)
        else:
            logger.warning("Parakeet model failed to load, falling back to Whisper")
            return await transcribe_with_whisper(audio_file_path)
    else:
        logger.warning(f"Unknown STT engine: {engine}, falling back to Whisper")
        return await transcribe_with_whisper(audio_file_path)

async def transcribe_with_whisper(audio_file_path: str) -> str:
    """Transcribes audio using the loaded HF Whisper model."""
    processor, model = load_whisper_model()
    if not processor or not model:
        raise RuntimeError("STT processor or model is not loaded.")

    try:
        logger.info(f"Loading audio file: {audio_file_path}")
        # Load audio file using librosa, ensuring 16kHz sample rate
        # With resampy installed, we can use the faster resampling method
        audio_input, sampling_rate = librosa.load(
            audio_file_path, 
            sr=16000,
            mono=True,       # Force mono channel
            duration=None,   # Use entire file
            res_type='kaiser_fast'  # Faster resampling (requires resampy)
        )
        
        audio_duration = len(audio_input)/sampling_rate
        logger.info(f"Audio loaded. Sample rate: {sampling_rate}, Duration: {audio_duration:.2f}s")
        
        # For longer audio, use chunking approach
        if audio_duration > 30:  # If longer than 30 seconds
            logger.info(f"Long audio detected ({audio_duration:.2f}s), using chunking approach")
            
            # Calculate chunk size (20 sec chunks with 2 sec overlap)
            chunk_size = 20 * sampling_rate
            overlap = 2 * sampling_rate
            
            transcripts = []
            
            # Process audio in overlapping chunks
            for i in range(0, len(audio_input), chunk_size - overlap):
                chunk = audio_input[i:i + chunk_size]
                
                # Skip processing tiny chunks
                if len(chunk) < sampling_rate * 1:  # Less than 1 second
                    continue
                    
                logger.info(f"Processing chunk {i/sampling_rate:.2f}s to {(i+len(chunk))/sampling_rate:.2f}s")
                
                # Process chunk
                features = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt").input_features
                features = features.to(DEVICE)
                if DEVICE.startswith("cuda"):
                    features = features.half()
                    
                with torch.no_grad():
                    pred_ids = model.generate(
                        features, 
                        max_new_tokens=256,
                        language="en"
                    )
                
                chunk_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                if chunk_text:
                    transcripts.append(chunk_text)
                    
            # Join all chunks
            transcript_text = " ".join(transcripts)
            
        else:
            # Standard processing for shorter audio
            input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
            input_features = input_features.to(DEVICE)
            if DEVICE.startswith("cuda"):
                input_features = input_features.half()

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features, 
                    max_new_tokens=256,
                    language="en"
                )

            transcript_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        logger.info(f"Whisper transcription complete. Output length: {len(transcript_text)}")
        return transcript_text

    except Exception as e:
        logger.error(f"Error during HF transcription: {e}", exc_info=True)
        raise RuntimeError(f"HF Transcription failed: {str(e)}")

async def transcribe_with_parakeet(audio_file_path: str) -> str:
    """Transcribes audio using the NVIDIA Parakeet model."""
    if not parakeet_model:
        raise RuntimeError("Parakeet model is not loaded.")

    temp_wav_path = None
    try:
        # Workaround for FFmpeg/lhotse initialization errors on Windows
        # If the file is not a WAV, we convert it to a temporary WAV using librosa
        # which we know is working reliably for Whisper.
        if not audio_file_path.lower().endswith(('.wav', '.flac')):
            logger.info(f"🔄 Converting {audio_file_path} to temporary WAV for Parakeet...")
            try:
                # Load using librosa (handles .webm, etc. reliably via soundfile/audioread)
                audio, sr = librosa.load(audio_file_path, sr=16000)
                
                # Create a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_wav_path = tmp.name
                
                # Save as WAV
                sf.write(temp_wav_path, audio, sr)
                audio_file_path = temp_wav_path
                logger.info(f"✅ Conversion successful: {temp_wav_path}")
            except Exception as conv_err:
                logger.error(f"❌ Failed to convert audio: {conv_err}")
                # Fallback to trying original path, but it will likely fail with the FFmpeg error

        logger.info(f"Transcribing with Parakeet: {audio_file_path}")
        
        # Parakeet uses a different API than Whisper
        # Let's run it in a thread pool so it doesn't block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: parakeet_model.transcribe([audio_file_path])
        )
        
        # Extract the transcript text
        transcript_text = result[0].text
        
        logger.info(f"Parakeet transcription complete. Output length: {len(transcript_text)}")
        return transcript_text

    except Exception as e:
        logger.error(f"Error during Parakeet transcription: {e}", exc_info=True)
        raise RuntimeError(f"Parakeet transcription failed: {str(e)}")
    finally:
        # Clean up temporary WAV file if it was created
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                logger.info(f"🧹 Cleaned up temporary file: {temp_wav_path}")
            except Exception as cleanup_err:
                logger.warning(f"Could not delete temp file {temp_wav_path}: {cleanup_err}")

def is_engine_available(engine: str) -> bool:
    """Check if the specified STT engine is available."""
    if engine == "whisper":
        processor, model = load_whisper_model()
        return processor is not None and model is not None
    elif engine == "parakeet":
        try:
            # More thorough check - try to import specific required modules
            import importlib.util
            
            # Check if NeMo is installed
            nemo_spec = importlib.util.find_spec("nemo")
            if nemo_spec is None:
                logger.info("NeMo package not found")
                return False
                
            # Check if NeMo ASR is importable
            nemo_asr_spec = importlib.util.find_spec("nemo.collections.asr")
            if nemo_asr_spec is None:
                logger.info("NeMo ASR module not found")
                return False
                
            # Check if we can actually import it without errors
            try:
                import nemo.collections.asr
                return True
            except Exception as e:
                logger.info(f"Error importing NeMo ASR: {e}")
                return False
                
        except ImportError:
            logger.info("Import error checking for Parakeet")
            return False
    return False

def list_available_engines() -> list:
    """Returns a list of available STT engines."""
    engines = []
    if is_engine_available("whisper"):
        engines.append("whisper")
    if is_engine_available("parakeet"):
        engines.append("parakeet")
    return engines