# app/tts_service.py
# --- Uses the dedicated kokoro library + chatterbox ---
# --- Standard Imports ---
import torch
import soundfile as sf
import os
import logging
from pathlib import Path
import uuid
import io
import sys
import tempfile
import asyncio
import subprocess
import re

# --- FastAPI and WebSocket Imports ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware


# --- Initialize FastAPI App and CORS Middleware ---
app = FastAPI()
router = APIRouter()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Existing Library/Model Loading Code (No Changes Needed Here) ---
try:
    from kokoro import KPipeline
except ImportError:
    # Log critical error if kokoro is missing
    startup_logger = logging.getLogger(__name__) # Use logger here
    startup_logger.critical("\n--- ERROR ---")
    startup_logger.critical("'kokoro' library not found. Please install it:")
    startup_logger.critical("pip install kokoro>=0.9.2")
    startup_logger.critical("Also ensure 'espeak-ng' is installed via your system package manager (e.g., sudo apt-get install espeak-ng)")
    startup_logger.critical("-------------\n")
    KPipeline = None # Set to None if import fails

try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    startup_logger = logging.getLogger(__name__)
    startup_logger.warning("\n--- WARNING ---")
    startup_logger.warning("'chatterbox-tts' library not found. Please install it:")
    startup_logger.warning("pip install chatterbox-tts")
    startup_logger.warning("Chatterbox TTS will not be available")
    startup_logger.warning("-------------\n")
    ChatterboxTTS = None

try:
    from datasets import load_dataset
except ImportError:
    startup_logger = logging.getLogger(__name__)
    startup_logger.critical("\n--- ERROR ---")
    startup_logger.critical("'datasets' library not found. Please install it:")
    startup_logger.critical("pip install datasets")
    startup_logger.critical("-------------\n")
    load_dataset = None

logger = logging.getLogger(__name__)
tts_pipeline = None
chatterbox_model = None
speaker_embeddings = None
kyutai_tts_model = None

def get_device():
    try:
        import torch
        from main import SINGLE_GPU_MODE
        
        logger.info(f"🔍 [DEBUG] SINGLE_GPU_MODE = {SINGLE_GPU_MODE}")
        
        if torch.cuda.is_available():
            if SINGLE_GPU_MODE:
                logger.info("🔍 [DEBUG] Using GPU 0 (single GPU mode)")
                return "cuda:0"
            else:
                logger.info("🔍 [DEBUG] Using GPU 1 (dual GPU mode)")
                return "cuda:1"
        return "cpu"
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"🔍 [DEBUG] Import error: {e}")
        return "cpu"

DEVICE = get_device()

# --- All your existing functions (load_kyutai_model, clean_markdown_for_tts, etc.) go here ---
# --- No changes are needed in them. This just shows the correct file structure.      ---

def load_kyutai_model():
    """Load Kyutai model once and reuse it"""
    global kyutai_tts_model
    
    if kyutai_tts_model is None:
        try:
            import os
            os.environ['TORCH_DYNAMO_DISABLE'] = '1'
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            os.environ['TORCHDYNAMO_DISABLE'] = '1'
            os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'
            
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
            
            logger.info("🗣️ [Kyutai] Loading TTS model... (this may take a moment)")
            checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
            kyutai_tts_model = TTSModel.from_checkpoint_info(
                checkpoint_info, n_q=32, temp=0.6, device='cuda'
            )
            logger.info("🗣️ [Kyutai] TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"🗣️ [Kyutai] Failed to load model: {e}", exc_info=True)
            kyutai_tts_model = None
            raise RuntimeError(f"Failed to load Kyutai model: {str(e)}")
    
    return kyutai_tts_model


def clean_markdown_for_tts(text: str) -> str:
    """Removes common Markdown formatting and problematic characters for clearer TTS."""
    if not text:
        return ""
    import re
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Remove ALL punctuation marks
    text = re.sub(r'[.!?,:;(){}[\]"\'""''`~@#$%^&*+=<>|\\/_-]', '', text)
    
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def load_chatterbox_model():
    """Loads the Chatterbox TTS model if not already loaded."""
    global chatterbox_model
    if ChatterboxTTS is None:
        raise RuntimeError("chatterbox-tts library is not installed or import failed.")
    
    if chatterbox_model is None:
        try:
            detected_device = get_device()
            logger.info(f"🔍 [DEBUG] Detected device: {detected_device}")
            logger.info(f"🔍 [DEBUG] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"🔍 [DEBUG] CUDA device count: {torch.cuda.device_count()}")
            
            logger.info(f"Loading Chatterbox TTS model onto device {detected_device}...")
            chatterbox_model = ChatterboxTTS.from_pretrained(device=detected_device)
            logger.info("Chatterbox TTS model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Chatterbox TTS model: {e}", exc_info=True)
            chatterbox_model = None
            raise RuntimeError(f"Failed to load Chatterbox TTS model: {e}") from e
    
    return chatterbox_model

def load_tts_pipeline(lang_code='a'):
    """Loads the Kokoro TTS Pipeline and speaker embeddings if not already loaded."""
    global tts_pipeline, speaker_embeddings
    if KPipeline is None:
        raise RuntimeError("kokoro library is not installed or import failed.")
    if load_dataset is None:
         raise RuntimeError("datasets library is not installed or import failed.")

    if tts_pipeline is None or speaker_embeddings is None:
        try:
            if tts_pipeline is None:
                logger.info(f"Loading Kokoro TTS Pipeline (lang: {lang_code}) onto device {DEVICE}...")
                tts_pipeline = KPipeline(lang_code=lang_code)
                logger.info("Kokoro TTS Pipeline loaded successfully.")

            if speaker_embeddings is None:
                logger.info("Attempting to load speaker embeddings using 'datasets' library...")
                try:
                    logger.info("Loading dataset 'Matthijs/cmu-arctic-xvectors'...")
                    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                    speaker_index = 7306
                    logger.info(f"Extracting speaker embedding at index {speaker_index}...")
                    embedding_vector = embeddings_dataset[speaker_index]["xvector"]
                    speaker_embeddings = torch.tensor(embedding_vector).unsqueeze(0).to(DEVICE)
                    logger.info(f"Loaded speaker embeddings successfully from dataset index {speaker_index}.")
                except Exception as emb_err:
                     logger.error(f"Error loading speaker_embeddings from dataset: {emb_err}", exc_info=True)
                     logger.warning("Speaker embeddings failed to load! TTS might rely solely on 'voice' param.")
                     speaker_embeddings = None

        except AssertionError as ae:
             logger.error(f"AssertionError loading KPipeline: Invalid lang_code '{lang_code}'?", exc_info=True)
             tts_pipeline = None
             speaker_embeddings = None
             raise RuntimeError(f"Failed to load TTS Pipeline: Invalid lang_code '{lang_code}' provided.") from ae
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS Pipeline or Embeddings: {e}", exc_info=True)
            tts_pipeline = None
            speaker_embeddings = None
            raise RuntimeError(f"Unexpected error loading TTS components: {e}") from e

    return tts_pipeline

async def synthesize_speech(
    text: str, 
    voice: str = 'af_heart', 
    engine: str = 'kokoro',
    audio_prompt_path: str = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> bytes:
    """
    Cleans input text, synthesizes speech using the specified engine, and returns raw audio bytes.
    """
    cleaned_text = clean_markdown_for_tts(text)
    if not cleaned_text:
        logger.warning("🗣️ [TTS Service] Text became empty after cleaning, skipping synthesis.")
        return b""
    logger.info(f"🗣️ [TTS Service] Cleaned text for TTS: '{cleaned_text[:60]}...'")

    if engine.lower() == 'kyutai':
        return await _synthesize_with_kyutai_direct(cleaned_text, voice)
    if engine.lower() == 'chatterbox':
        return await _synthesize_with_chatterbox(cleaned_text, audio_prompt_path, exaggeration, cfg)
    else:
        return await _synthesize_with_kokoro(cleaned_text, voice)

async def _synthesize_with_kyutai_direct(text: str, voice: str = None) -> bytes:
    # ... (this function remains unchanged)
    return b""

async def _synthesize_with_chatterbox(
    text: str, 
    audio_prompt_path: str = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> bytes:
    # ... (this function remains unchanged)
    return b""

async def _synthesize_with_kokoro(text: str, voice: str) -> bytes:
    """Synthesize speech using Kokoro TTS and return raw audio bytes."""
    import time
    import numpy as np
    import soundfile as sf
    import tempfile
    import os

    try:
        total_start_time = time.perf_counter()

        # --- 1. Pipeline Loading ---
        load_start_time = time.perf_counter()
        assumed_lang_code = 'a'
        pipeline = load_tts_pipeline(lang_code=assumed_lang_code)
        load_end_time = time.perf_counter()
        
        if not pipeline:
            raise RuntimeError("Failed to load Kokoro TTS pipeline")
        
        logger.info(f"🗣️ [Kokoro] Synthesizing with voice '{voice}': '{text[:50]}...'")
        
        # --- 2. Core TTS Inference ---
        synth_start_time = time.perf_counter()
        generator = pipeline(text, voice=voice)
        audio_chunks = [audio for _, _, audio in generator] # More concise way to collect
        synth_end_time = time.perf_counter()
        
        if not audio_chunks:
            raise RuntimeError("No audio chunks generated")

        # --- 3. Audio Post-Processing (Concatenate and File I/O) ---
        post_start_time = time.perf_counter()
        full_audio = np.concatenate(audio_chunks, axis=0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            sf.write(temp_path, full_audio, 24000)
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        post_end_time = time.perf_counter()
        total_end_time = time.perf_counter()

        # --- New Timing Logs ---
        logger.info(f"⏱️ [Kokoro Breakdown] Pipeline Loading: {(load_end_time - load_start_time) * 1000:.2f}ms")
        logger.info(f"⏱️ [Kokoro Breakdown] Core Inference: {(synth_end_time - synth_start_time) * 1000:.2f}ms")
        logger.info(f"⏱️ [Kokoro Breakdown] Post-Processing: {(post_end_time - post_start_time) * 1000:.2f}ms")
        logger.info(f"⏱️ [Kokoro Total] Full function duration: {(total_end_time - total_start_time) * 1000:.2f}ms")
        
        logger.info(f"🗣️ [Kokoro] Generated {len(audio_bytes)} bytes of audio from {len(audio_chunks)} chunks")
        return audio_bytes
            
    except Exception as e:
        logger.error(f"🗣️ [Kokoro] Synthesis failed: {e}", exc_info=True)
        raise RuntimeError(f"Kokoro synthesis failed: {str(e)}")

# --- NEW BACKEND STREAMING LOGIC ---


class TTSStreamer:
    """Manages buffering and synthesizing text chunks for a single WebSocket connection."""
    def __init__(self, websocket: WebSocket):
        self._websocket = websocket
        self._text_buffer = ""
        self._synthesis_queue = asyncio.Queue()
        self._is_active = True
        self._synthesis_task = asyncio.create_task(self.synthesis_loop())

    async def add_text(self, text: str):
        if not self._is_active: return
        self._text_buffer += text
        self._find_and_queue_sentences()

    def _find_and_queue_sentences(self):
        # This regex finds sentences ending with punctuation followed by whitespace or end-of-string
        # It treats consecutive punctuation (like ...) as a single unit
        sentence_end_regex = r'[^.!?]*[.!?]+'
        match = re.search(sentence_end_regex, self._text_buffer)
        
        if match:
            sentence = match.group(0).strip()
            # Remove the processed sentence from the buffer.
            self._text_buffer = self._text_buffer[len(match.group(0)):]
            
            if sentence:
                logger.info(f"🧠 [Streamer] Queuing chunk: '{sentence[:60]}...'")
                self._synthesis_queue.put_nowait(sentence)

    async def synthesis_loop(self):
        """The 'consumer' loop that synthesizes sentences from the queue."""
        import time

        while self._is_active:
            try:
                sentence = await self._synthesis_queue.get()
                if sentence is None:  # Sentinel value to stop the loop
                    break

                logger.info(f"🎤 [Streamer] Synthesizing: '{sentence[:60]}...'")
                
                # --- Start Timing ---
                start_time = time.perf_counter()

                # The full synthesis process happens here
                audio_bytes = await synthesize_speech(text=sentence, voice='af_heart')

                # --- End Timing ---
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                logger.info(f"⏱️ [Streamer] Full synthesis task took {duration_ms:.2f}ms")

                if audio_bytes:
                    await self._websocket.send_bytes(audio_bytes)
                    logger.info(f"✅ [Streamer] Sent audio chunk of {len(audio_bytes)} bytes.")
                
                self._synthesis_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [Streamer] Synthesis loop error: {e}", exc_info=True)

    def finish(self):
        if self._is_active:
            # Process any remaining text in the buffer
            if self._text_buffer.strip():
                logger.info(f"🧠 [Streamer] Queuing final buffer text.")
                self._synthesis_queue.put_nowait(self._text_buffer.strip())
            # Add sentinel to shut down the loop gracefully
            self._synthesis_queue.put_nowait(None) 
            self._is_active = False
    
    async def cancel(self):
        self._is_active = False
        self._synthesis_task.cancel()
        try:
            await self._synthesis_task
        except asyncio.CancelledError:
            logger.info("🛑 [Streamer] Synthesis task cancelled successfully.")

@router.websocket("/tts-stream")
async def tts_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ [WebSocket] Connection accepted. Ready for streams.")
    
    try:
        # This outer loop keeps the connection alive for multiple messages
        while True:
            logger.info("👂 [WebSocket] Waiting for the start of a new message...")
            
            # The first text received will kick off the new streamer
            initial_text = await websocket.receive_text()

            # Create a NEW streamer instance for this specific message
            streamer = TTSStreamer(websocket)
            logger.info("✅ Streamer created for new message stream.")

            # Immediately process the first chunk of text
            await streamer.add_text(initial_text)

            # This inner loop handles all text for the CURRENT message
            while True:
                text_chunk = await websocket.receive_text()

                if text_chunk == "--END--":
                    logger.info("🏁 [WebSocket] Received end-of-stream signal for current message.")
                    streamer.finish()
                    # Wait for the streamer's background task to finish
                    if streamer._synthesis_task:
                        await streamer._synthesis_task 
                    break  # Breaks the INNER loop, ready for a new message
                else:
                    await streamer.add_text(text_chunk)
            
            logger.info("✅ [WebSocket] Finished processing message stream.")

    except WebSocketDisconnect:
        logger.warning("🔌 [WebSocket] Frontend disconnected.")
    except Exception as e:
        logger.error(f"❌ [WebSocket] An unexpected error occurred in the main handler: {e}", exc_info=True)