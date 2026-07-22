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
import json
import inspect
import wave
import aiohttp
from typing import Optional
# --- FastAPI and WebSocket Imports ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from backend.app.cors_policy import configure_cors


# --- Initialize FastAPI App and CORS Middleware ---
app = FastAPI()
router = APIRouter()

configure_cors(app)

# Include the router in the app
app.include_router(router)


# --- Kokoro TTS Library Loading ---
startup_logger = logging.getLogger(__name__)
try:
    from kokoro import KPipeline
    startup_logger.info("✅ Kokoro TTS library loaded successfully")
except Exception as e:
    startup_logger.warning(f"\n--- WARNING: Kokoro TTS not available ---")
    startup_logger.warning(f"Error: {e}")
    startup_logger.warning("This is usually a phonemizer/misaki dependency issue.")
    startup_logger.warning("Try: pip uninstall phonemizer && pip install phonemizer-fork")
    startup_logger.warning("Kokoro TTS will not be available - using Chatterbox instead")
    startup_logger.warning("-------------\n")
    KPipeline = None

try:
    from chatterbox import ChatterboxTTS
except ImportError:
    startup_logger = logging.getLogger(__name__)
    startup_logger.warning("\n--- WARNING ---")
    startup_logger.warning("'chatterbox-tts' library not found. Please install it:")
    startup_logger.warning("pip install chatterbox-tts")
    startup_logger.warning("Chatterbox TTS will not be available")
    startup_logger.warning("-------------\n")
    ChatterboxTTS = None

try:
    # Use the vendored version for Turbo to avoid conflicts
    try:
        from .chatterbox_turbo.tts_turbo import ChatterboxTurboTTS
    except ImportError:
        try:
            from app.chatterbox_turbo.tts_turbo import ChatterboxTurboTTS
        except ImportError:
            from chatterbox_turbo.tts_turbo import ChatterboxTurboTTS
    
    startup_logger.info("✅ Chatterbox Turbo (Vendored) library loaded successfully")
except Exception as e:
    startup_logger = logging.getLogger(__name__)
    startup_logger.warning(f"\n--- WARNING: Chatterbox Turbo (Vendored) not available: {e} ---")
    ChatterboxTurboTTS = None

try:
    # Vendored Nano loader
    try:
        from .chatterbox_turbo.tts_nano import ChatterboxNanoTTS
    except ImportError:
        try:
            from app.chatterbox_turbo.tts_nano import ChatterboxNanoTTS
        except ImportError:
            from chatterbox_turbo.tts_nano import ChatterboxNanoTTS
    
    startup_logger.info("✅ Chatterbox Nano (Vendored) library loaded successfully")
except Exception as e:
    startup_logger = logging.getLogger(__name__)
    startup_logger.warning(f"\n--- WARNING: Chatterbox Nano (Vendored) not available: {e} ---")
    ChatterboxNanoTTS = None

try:
    from voxcpm import VoxCPM
    startup_logger.info("✅ VoxCPM2 library loaded successfully")
except ImportError:
    startup_logger = logging.getLogger(__name__)
    startup_logger.warning("\n--- WARNING ---")
    startup_logger.warning("'voxcpm' library not found. Please install it:")
    startup_logger.warning("pip install voxcpm")
    startup_logger.warning("VoxCPM2 TTS will not be available")
    startup_logger.warning("-------------\n")
    VoxCPM = None

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
chatterbox_turbo_model = None # NEW: Turbo model global
chatterbox_nano_model = None # NEW: Nano model global
voxcpm_model = None # VoxCPM2 model global
speaker_embeddings = None
CHATTERBOX_VOICE_WARMED_UP = False
CHATTERBOX_TURBO_VOICE_WARMED_UP = False # NEW: Turbo warmup flag
CHATTERBOX_NANO_VOICE_WARMED_UP = False # NEW: Nano warmup flag
VOXCPM_VOICE_WARMED_UP = False
VOXCPM_EXECUTOR = None # Separate executor for VoxCPM2 — never share with Chatterbox
CHATTERBOX_EXECUTOR = None # Global executor to keep synthesis on one persistent thread

# --- VoxCPM2 GGUF Model Catalog ---
# Source: https://huggingface.co/DennisHuang648/VoxCPM2-GGUF
VOXCPM_GGUF_HF_REPO = "DennisHuang648/VoxCPM2-GGUF"

VOXCPM_GGUF_MODELS = {
    "BaseLM-Q8_0": {
        "label": "VoxCPM2 BaseLM Q8_0 (Recommended, ~1.6 GB)",
        "filename": "VoxCPM2-BaseLM-Q8_0.gguf",
        "size_mb": 1600,
        "component": "Base language model, 8-bit quantized",
        "required": True,
    },
    "BaseLM-F16": {
        "label": "VoxCPM2 BaseLM F16 (Full precision, ~3.0 GB)",
        "filename": "VoxCPM2-BaseLM-F16.gguf",
        "size_mb": 3000,
        "component": "Base language model (28-layer, n_embd=2048)",
        "required": True,
    },
    "Acoustic-F16": {
        "label": "VoxCPM2 Acoustic F16 (Required, ~1.7 GB)",
        "filename": "VoxCPM2-Acoustic-F16.gguf",
        "size_mb": 1700,
        "component": "Acoustic stack (ResidualLM + FSQ + LocEnc/LocDiT CFM + AudioVAE)",
        "required": True,
    },
}

def get_chatterbox_executor():
    """Returns a persistent ThreadPoolExecutor with 1 worker to ensure CUDA thread consistency."""
    global CHATTERBOX_EXECUTOR
    if CHATTERBOX_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor
        CHATTERBOX_EXECUTOR = ThreadPoolExecutor(max_workers=1)
        logger.info("🧵 [Chatterbox] Persistent thread executor initialized")
    return CHATTERBOX_EXECUTOR

def get_device(preferred_gpu_id: int = 0):
    """Determines the correct Torch device for the current process.
    
    Args:
        preferred_gpu_id: The GPU ID to prefer (default: 0)
    """
    try:
        import torch
        force_cpu = os.environ.get("MIRID_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
        if torch.cuda.is_available() and not force_cpu:
            # Use the first available GPU for TTS service
            device = "cuda:0"
            
            # Force CUDA device selection
            torch.cuda.set_device(0)
            current_device = torch.cuda.current_device()
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            compute_capability = torch.cuda.get_device_capability(0)
            
            logger.info(f"🔒 TTS service using: {device} ({gpu_name})")
            logger.info(f"🔒 GPU Memory: {gpu_memory:.1f} GB")
            logger.info(f"🔒 Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
            logger.info(f"🔒 Current CUDA device: {current_device}")
            logger.info(f"🔒 Total CUDA devices: {torch.cuda.device_count()}")
            
            # CRITICAL: Test GPU computation
            logger.info("🔒 Testing GPU computation...")
            test_tensor = torch.randn(1000, 1000, device='cuda:0')
            test_result = torch.mm(test_tensor, test_tensor)
            logger.info(f"🔒 GPU test successful: {test_result.shape}")
            
            return device
        else:
            logger.warning("⚠️ CUDA not available. Falling back to CPU for TTS.")
            return "cpu"
    except Exception as e:
        logger.error(f"❌ Error determining device, falling back to CPU: {e}")
        return "cpu"

DEVICE = None  # Will be set lazily when first needed

def get_tts_device():
    """Get device lazily, ONLY in the correct process"""
    global DEVICE
    if DEVICE is None:
        # Only initialize if we're in a process with CUDA isolation
        # or if we're explicitly in the main server process
        if os.environ.get('CUDA_VISIBLE_DEVICES') or os.getpid() != os.getppid():
            DEVICE = get_device(preferred_gpu_id=0)
            logger.info(f"🔧 TTS Device initialized (lazy): {DEVICE}")
        else:
            # We're in the parent/launcher process - don't initialize
            logger.warning(f"⚠️ Skipping TTS device init in parent process (PID: {os.getpid()})")
            return "cpu"  # Return safe fallback
    return DEVICE
# Additional safety: Log the current CUDA device to verify isolation
try:
    import torch
    force_cpu = os.environ.get("MIRID_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
    if torch.cuda.is_available() and not force_cpu:
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"🔒 CUDA current device verified: {current_device} ({current_device_name})")
        
        # Check if CUDA_VISIBLE_DEVICES is set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        logger.info(f"🔍 CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # Check if we're in an isolated process
        if cuda_visible and cuda_visible != 'Not set':
            if current_device != 0:  # In isolated process, should be cuda:0
                logger.warning(f"⚠️ WARNING: Current CUDA device is {current_device}, should be 0 in isolated process")
        else:
            # In single GPU mode, we prefer GPU 0
            preferred_gpu = 0
            if current_device != preferred_gpu:
                logger.warning(f"⚠️ WARNING: Current CUDA device is {current_device}, not the preferred {preferred_gpu}")
        
        # Log all available devices
        device_count = torch.cuda.device_count()
        logger.info(f"🔍 Total CUDA devices available: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            logger.info(f"🔍 Device {i}: {device_name}")
            
except Exception as e:
    logger.warning(f"⚠️ Could not verify CUDA device isolation: {e}")

def normalize_chatterbox_tags(text: str) -> str:
    """
    Normalizes a wide range of paralinguistic tag synonyms, variants, and hallucinations 
    to the specific canonical set supported by Chatterbox Turbo.
    
    Handles:
    1. Synonyms: "giggles" -> [laugh]
    2. Variations: *sighs*, (laughing) -> [tag]
    3. Plain text attempts in brackets: [guffaw] -> [laugh]
    4. Unsupported tags: [screams] -> removed
    """
    if not text:
        return ""
        
    canonical_map = {
        '[laugh]': [
            'laugh', 'laughs', 'laughing', 'laughter', 
            'giggle', 'giggles', 'giggling', 
            'chortle', 'chortling', 
            'snicker', 'snickering', 
            'guffaw', 'guffawing', 
            'haha', 'hahaha', 'lol', 'lmao', 'rofl'
        ],
        '[chuckle]': [
            'chuckle', 'chuckles', 'chuckling', 
            'soft laugh', 'teehee', 'hehe'
        ],
        '[sigh]': [
            'sigh', 'sighs', 'sighing', 'sighed', 
            'exhale', 'exhaling', 'exhales', 
            'phew', 'huff', 'huffing'
        ],
        '[groan]': [
            'groan', 'groans', 'groaning', 'groaned', 
            'moan', 'moans', 'moaning', 
            'grunt', 'grunting', 'argh', 'ugh'
        ],
        '[gasp]': [
            'gasp', 'gasps', 'gasping', 'gasped', 
            'sharp breath', 'intake of breath'
        ],
        '[clear throat]': [
            'clear throat', 'clears throat', 'clearing throat', 
            'ahem', 'throat clear'
        ],
        '[cough]': [
            'cough', 'coughs', 'coughing', 
            'hack', 'hacking'
        ],
        '[sniff]': [
            'sniff', 'sniffs', 'sniffing', 
            'sniffle', 'sniffles', 'sniffling'
        ],
        '[shush]': [
            'shush', 'shushes', 'shushing', 
            'hush', 'hushing', 
            'shh', 'shhh', 'be quiet'
        ],
        '[um]': [
            'um', 'umm', 'uh', 'uhh', 'er', 'err', 'erm', 'ah', 'ahh'
        ],
        '[hm]': [
            'hm', 'hmm', 'hmmm', 'hum', 'hums', 'humming'
        ]
    }
    
    # 1. Normalize Synonyms
    # We look for patterns like: [word], *word*, (word)
    # matching the synonyms case-insensitively
    
    normalized_text = text
    
    for canonical_tag, synonyms in canonical_map.items():
        # Create a giant OR regex for all synonyms of this tag
        # We perform replacement for each canonical tag group
        # Sort synonyms by length descending to catch 'soft laugh' before 'laugh'
        sorted_synonyms = sorted(synonyms, key=len, reverse=True)
        escaped_synonyms = [re.escape(s) for s in sorted_synonyms]
        pattern_str = '|'.join(escaped_synonyms)
        
        # Match [synonym], *synonym*, (synonym)
        # We allow flexible whitespace inside the wrappers
        regex = re.compile(
            r'(\[|\*|\()\s*(' + pattern_str + r')\s*(\]|\*|\))', 
            re.IGNORECASE
        )
        normalized_text = regex.sub(canonical_tag, normalized_text)

    # 2. Cleanup "Hallucinated" or Unsupported Tags
    # Chatterbox will read aloud any [...] that it doesn't recognize (or we strip the brackets and it reads the text).
    # The requirement is to STRIP unsupported tags entirely.
    
    supported_tags = set(canonical_map.keys())
    
    def tag_validator(match):
        content = match.group(0) # The full [tag]
        # Check if it is exactly one of our supported tags
        if content in supported_tags:
            return content
        else:
            # It's an unsupported bracketed sequence.
            # If it looks like a paralinguistic instruction (letters/spaces only), strip it.
            # If it looks like legitimate text (numbers, special chars), keep it (but stripped of brackets later by clean_markdown).
            # Heuristic: If it contains only letters and spaces/hyphens/underscores, assume it's a failed tag and delete.
            inner_text = match.group(1)
            if re.match(r'^[a-zA-Z\s\-_]+$', inner_text):
                return "" # Delete entirely
            return content # Keep (likely data)
            
    # Find all square bracket sequences
    # Note: normalized_text already has our canonical tags in it
    normalized_text = re.sub(r'\[(.*?)\]', tag_validator, normalized_text)
    
    # Clean up any double spaces created by deletions
    normalized_text = re.sub(r'\s{2,}', ' ', normalized_text).strip()
    
    return normalized_text

def clean_markdown_for_tts(text: str, engine: str = 'kokoro') -> str:
    """Removes common Markdown formatting and problematic characters for clearer TTS."""
    if not text:
        return ""
    import re
    
    # NEW: Apply Chatterbox Paralinguistic Normalization FIRST
    if engine.lower().startswith('chatterbox'):
        # Normalizes [giggles] -> [laugh], removes [screams], etc.
        text = normalize_chatterbox_tags(text)
        
    # Convert decimal points inside numbers to spoken form (e.g., "3.2" -> "3 point 2")
    text = re.sub(r'(?<=\d)\.(?=\d)', ' point ', text)
    # Speak "%" as "percent" (e.g. "80%" -> "80 percent") before we strip special chars
    text = text.replace('%', ' percent')

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
    
    # Remove punctuation marks but PRESERVE apostrophes, question marks, and exclamation marks
    # Remove punctuation marks but PRESERVE apostrophes, question marks, and exclamation marks
    if engine.lower().startswith('chatterbox'):
        # Preserve brackets [] and underscores/hyphens for Chatterbox tags like [laugh] or [clear_throat]
        text = re.sub(r'[:;(){}"``~@#$%^&*+=<>|\\/]', '', text)
    else:
        # Standard cleaning removes brackets and other special chars
        text = re.sub(r'[:;(){}[\]"""``~@#$%^&*+=<>|\\/_-]', '', text)
    
    # Collapse consecutive periods to prevent Kokoro click artifacts from solo "." chunks
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text
def load_settings():
    """Load settings from .LiangLocal/settings.json"""
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            return json.load(f)
    return {}


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
                logger.info(f"Loading Kokoro TTS Pipeline (lang: {lang_code}) onto device {get_device()}...")
                tts_pipeline = KPipeline(lang_code=lang_code)
                logger.info("✅ Kokoro TTS Pipeline loaded successfully.")

            if speaker_embeddings is None:
                logger.info("Attempting to load speaker embeddings using 'datasets' library...")
                try:
                    logger.info("Loading dataset 'Matthijs/cmu-arctic-xvectors'...")
                    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                    speaker_index = 7306
                    logger.info(f"Extracting speaker embedding at index {speaker_index}...")
                    embedding_vector = embeddings_dataset[speaker_index]["xvector"]
                    speaker_embeddings = torch.tensor(embedding_vector).unsqueeze(0).to(get_device())
                    logger.info(f"✅ Loaded speaker embeddings successfully from dataset index {speaker_index}.")
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
    cfg: float = 0.5,
    speed: float = 1.0,
    voxcpm_cfg_value: float = 2.0,
    voxcpm_inference_timesteps: int = 10,
    voxcpm_normalize: bool = True,
    voxcpm_denoise: bool = True,
    voxcpm_retry_badcase: bool = False,
    voxcpm_voice_design: str = None,
) -> bytes:
    """
    Cleans input text, synthesizes speech using the specified engine, and returns raw audio bytes.
    Available engines: kokoro, chatterbox, chatterbox_turbo, chatterbox_nano, voxcpm, nanogpt-*
    """

    cleaned_text = clean_markdown_for_tts(text, engine=engine)

    # Fallback: If using Chatterbox/Turbo/VoxCPM and no audio_prompt_path is provided,
    # try to use the 'voice' parameter as the path (if it's not 'default' or 'af_heart')
    if (engine.lower().startswith('chatterbox') or engine.lower() == 'voxcpm') and not audio_prompt_path:
        if voice and voice.lower() not in ('default', 'af_heart'):
            audio_prompt_path = voice
            logger.info(f"🗣️ [TTS Service] Adapting 'voice' parameter '{voice}' as audio_prompt_path")

    if not cleaned_text:
        logger.debug("🗣️ [TTS Service] Text became empty after cleaning, skipping synthesis.")
        return b""

    logger.debug(f"🗣️ [TTS Service] Using engine '{engine}' for: '{cleaned_text[:60]}...'")

    if engine.lower() == 'kokoro':
        # Use Kokoro TTS
        if KPipeline is None:
            logger.warning("⚠️ Kokoro not available, falling back to Chatterbox")
            return await _synthesize_with_chatterbox(
                cleaned_text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg=cfg
            )
        return await _synthesize_with_kokoro(cleaned_text, voice, speed=speed)
    elif engine.lower() == 'chatterbox':
        return await _synthesize_with_chatterbox(
            cleaned_text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )
    elif engine.lower() == 'chatterbox_turbo':
        return await _synthesize_with_chatterbox_turbo(
            cleaned_text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )
    elif engine.lower() == 'chatterbox_nano':
        return await _synthesize_with_chatterbox_nano(
            cleaned_text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )
    elif engine.lower() == 'voxcpm':
        return await _synthesize_with_voxcpm(
            cleaned_text,
            audio_prompt_path=audio_prompt_path,
            cfg_value=voxcpm_cfg_value,
            inference_timesteps=voxcpm_inference_timesteps,
            normalize=voxcpm_normalize,
            denoise=voxcpm_denoise,
            retry_badcase=voxcpm_retry_badcase,
            voice_design=voxcpm_voice_design,
        )
    elif engine.lower() == 'voxcpm-gguf':
        return await _synthesize_with_voxcpm_gguf(
            cleaned_text,
            audio_prompt_path=audio_prompt_path,
            cfg_value=voxcpm_cfg_value,
            inference_timesteps=voxcpm_inference_timesteps,
            voice_design=voxcpm_voice_design,
        )
    elif engine.lower().startswith('nanogpt-'):
        model = engine.split('-', 1)[1] if '-' in engine else 'Qwen-3-TTS-1.7B'
        return await _synthesize_with_nanogpt(cleaned_text, voice, model=model, speed=speed)
    else:
        logger.warning(f"⚠️ Unknown engine '{engine}', using Kokoro.")
        if KPipeline is not None:
            return await _synthesize_with_kokoro(cleaned_text, voice, speed=speed)
        return await _synthesize_with_chatterbox(
            cleaned_text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )



NANOGPT_TTS_MODEL_PROFILES = {
    "Kokoro-82m": {
        "voice_default": "af_bella",
        "voice_valid": [
            "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
            "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
            "am_onyx", "am_puck",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            "ff_siwis",
            "hf_alpha", "hf_beta",
            "hm_omega", "hm_psi",
            "if_sara",
            "im_nicola",
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
            "jm_kumo",
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
        ],
        "extra_params": [],
        "max_input_chars": 10000,
    },
    "Qwen-3-TTS-1.7B": {
        "voice_default": "Vivian",
        "voice_valid": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"],
        "extra_params": ["language", "top_k", "top_p", "temperature", "repetition_penalty",
                         "subtalker_dosample", "subtalker_top_k", "subtalker_top_p",
                         "subtalker_temperature", "max_new_tokens"],
        "max_input_chars": None,
    },
}


def _get_nanogpt_model_profile(model: str) -> dict:
    """Get the profile for a NanoGPT TTS model, with fallback to minimal profile for unknown models."""
    # For known models (Kokoro, Qwen), return full profile with voice validation
    # For unknown models, return minimal profile - no voice validation (let API handle it)
    known = NANOGPT_TTS_MODEL_PROFILES.get(model)
    if known:
        return known
    return {
        "voice_default": None,
        "voice_valid": [],  # Empty = don't validate, pass voice through to API
        "extra_params": [],
        "max_input_chars": 10000,
    }


async def _synthesize_with_nanogpt(text: str, voice: str, model: str = 'Qwen-3-TTS-1.7B', speed: float = 1.0) -> bytes:
    """
    Synthesize speech using NanoGPT TTS API with model-aware payload construction.
    Supports Kokoro-82m, Qwen-3-TTS-1.7B, and future models via NANOGPT_TTS_MODEL_PROFILES.
    Handles both synchronous (HTTP 200) and asynchronous (HTTP 202) responses by polling
    the status endpoint when the job is queued.
    """
    import time
    import asyncio

    try:
        settings = load_settings()
        nanogpt_api_key = settings.get('nanogpt_api_key') or settings.get('nanoGptApiKey')

        if not nanogpt_api_key:
            logger.error("❌ NanoGPT API key not configured in settings")
            raise RuntimeError("NanoGPT API key not configured")

        base_url = "https://nano-gpt.com"
        tts_url = f"{base_url}/api/tts"

        headers = {
            "Content-Type": "application/json",
            "x-api-key": nanogpt_api_key
        }

        profile = _get_nanogpt_model_profile(model)
        valid_voices = profile["voice_valid"]
        default_voice = profile["voice_default"]
        max_input_chars = profile["max_input_chars"]

        # For known models: treat valid_voices as a suggestion list, not a hard restriction.
        # If the user explicitly provides a voice name, respect it even if unknown.
        # Only fall back to default when no voice was given (None or sentinel values).
        if voice and voice not in ('default', ''):
            resolved_voice = voice
            if valid_voices and voice not in valid_voices:
                logger.warning(f"Voice '{voice}' is not in the known list for {model}, passing through anyway")
        elif default_voice:
            resolved_voice = default_voice
        else:
            resolved_voice = None

        if max_input_chars and len(text) > max_input_chars:
            logger.warning(f"⚠️ Input text ({len(text)} chars) exceeds {model} limit ({max_input_chars}), truncating")
            text = text[:max_input_chars]

        speed = max(0.25, min(4.0, float(speed or 1.0)))

        payload = {
            "input": text,
            "speed": speed,
            "model": model,
        }
        if resolved_voice is not None:
            payload["voice"] = resolved_voice

        if model == "Qwen-3-TTS-1.7B":
            payload.update({
                "language": "Auto",
                "top_k": 50,
                "top_p": 1,
                "temperature": 0.9,
                "repetition_penalty": 1.05,
                "subtalker_dosample": True,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1,
                "subtalker_temperature": 0.9,
                "max_new_tokens": 8192,
            })

        logger.info(f"🎵 Calling NanoGPT TTS API (model='{model}', voice='{resolved_voice}', speed={speed}) for text: '{text[:60]}...'")

        start_time = time.perf_counter()

        # Magic-byte signatures for known audio container formats.
        def _looks_like_audio(data: bytes) -> bool:
            if not data or len(data) < 4:
                return False
            return (
                data[:4] == b"RIFF"        # WAV
                or data[:3] == b"ID3"      # MP3 with ID3 tag
                or data[:2] == b"\xff\xfb" # MP3 frame
                or data[:2] == b"\xff\xf3" # MP3 frame
                or data[:2] == b"\xff\xf2" # MP3 frame
                or data[:4] == b"OggS"     # Ogg/Opus
                or data[:4] == b"fLaC"     # FLAC
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(tts_url, headers=headers, json=payload) as response:
                # NanoGPT may return the async "job queued" ticket either as HTTP 202
                # OR as HTTP 200 with a JSON body (status=pending/queued/processing).
                # Detect a ticket regardless of status code and route into polling.
                ticket = None
                if response.status in (200, 202):
                    content_type = (response.headers.get("Content-Type") or "").lower()
                    if "application/json" in content_type:
                        try:
                            maybe = await response.json()
                            if isinstance(maybe, dict) and (
                                maybe.get("runId")
                                or maybe.get("status") in ("pending", "queued", "processing")
                            ):
                                ticket = maybe
                        except Exception:
                            ticket = None

                if ticket is not None:
                    run_id = ticket.get("runId")
                    if not run_id:
                        raise RuntimeError(f"NanoGPT API returned a queued ticket but no runId: {ticket}")

                    logger.info(f"⏳ NanoGPT TTS queued (HTTP {response.status}, runId={run_id}), polling for completion...")

                    # Poll on a TIME budget, not a fixed attempt count. Long text can take
                    # several minutes to synthesize, so allow plenty of headroom. As long as
                    # the job keeps reporting pending/processing, we keep waiting up to the
                    # wall-clock budget. Interval backs off so we don't hammer the API.
                    poll_interval = 1.0
                    max_poll_interval = 5.0
                    poll_budget_seconds = 30 * 60  # 30 minutes of wall-clock headroom
                    poll_deadline = time.perf_counter() + poll_budget_seconds
                    cost = ticket.get("cost")
                    payment_source = ticket.get("paymentSource")
                    is_api_request = ticket.get("isApiRequest", True)

                    attempt = 0
                    while time.perf_counter() < poll_deadline:
                        attempt += 1
                        await asyncio.sleep(poll_interval)
                        poll_interval = min(poll_interval * 1.2, max_poll_interval)

                        params = {
                            "runId": run_id,
                            "model": model,
                        }
                        if cost is not None:
                            params["cost"] = cost
                        if payment_source is not None:
                            params["paymentSource"] = payment_source
                        if is_api_request is not None:
                            params["isApiRequest"] = str(is_api_request).lower()

                        status_url = f"{base_url}/api/tts/status"
                        async with session.get(status_url, headers=headers, params=params) as status_resp:
                            if status_resp.status != 200:
                                error_detail = await status_resp.text()
                                logger.warning(f"⚠️ NanoGPT status poll returned {status_resp.status}: {error_detail}")
                                continue

                            status_data = await status_resp.json()
                            job_status = status_data.get("status")

                            if job_status == "completed":
                                audio_url = status_data.get("audioUrl")
                                if not audio_url:
                                    raise RuntimeError(f"NanoGPT TTS completed but no audioUrl: {status_data}")

                                logger.info(f"📥 NanoGPT TTS completed, downloading audio from {audio_url}")

                                async with session.get(audio_url, headers=headers) as audio_resp:
                                    if audio_resp.status != 200:
                                        raise RuntimeError(f"Failed to download audio from {audio_url}: HTTP {audio_resp.status}")
                                    audio_bytes = await audio_resp.read()

                                end_time = time.perf_counter()
                                duration_ms = (end_time - start_time) * 1000
                                logger.info(f"✅ NanoGPT TTS completed in {duration_ms:.2f}ms, {len(audio_bytes)} bytes")
                                return audio_bytes

                            elif job_status in ("failed", "error"):
                                error_msg = status_data.get("error", "Unknown error")
                                raise RuntimeError(f"NanoGPT TTS failed: {error_msg}")

                            elif job_status in ("pending", "queued", "processing"):
                                if attempt % 5 == 0:
                                    elapsed = time.perf_counter() - start_time
                                    logger.info(f"⏳ NanoGPT TTS still {job_status} ({elapsed:.0f}s elapsed)...")
                                continue

                            else:
                                logger.warning(f"⚠️ NanoGPT TTS unknown status '{job_status}', retrying...")
                                continue

                    raise RuntimeError(f"NanoGPT TTS timed out after {poll_budget_seconds}s (runId={run_id})")

                if response.status != 200:
                    error_detail = await response.text()
                    logger.error(f"❌ NanoGPT API request failed with status {response.status}: {error_detail}")

                    if response.status == 401:
                        raise RuntimeError("Invalid API key for NanoGPT")
                    elif response.status == 403:
                        raise RuntimeError("Access forbidden - check API key permissions")
                    elif response.status == 429:
                        raise RuntimeError("Rate limit exceeded - too many requests")
                    elif response.status == 400:
                        try:
                            error_json = await response.json()
                            raise RuntimeError(f"Bad request: {error_json.get('detail', error_detail)}")
                        except:
                            raise RuntimeError(f"Bad request: {error_detail}")
                    else:
                        raise RuntimeError(f"API request failed with status {response.status}")

                audio_bytes = await response.read()

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        if not audio_bytes:
            logger.error("❌ NanoGPT API returned empty audio response")
            raise RuntimeError("Empty audio response from NanoGPT")

        if not _looks_like_audio(audio_bytes):
            preview = audio_bytes[:200]
            try:
                preview_text = preview.decode("utf-8", errors="replace")
            except Exception:
                preview_text = repr(preview)
            logger.error(f"❌ NanoGPT returned non-audio payload ({len(audio_bytes)} bytes): {preview_text}")
            raise RuntimeError(f"NanoGPT returned non-audio response: {preview_text}")

        logger.info(f"✅ NanoGPT TTS completed in {duration_ms:.2f}ms, {len(audio_bytes)} bytes")

        return audio_bytes

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"❌ NanoGPT TTS synthesis failed: {e}", exc_info=True)
        raise RuntimeError(f"NanoGPT TTS synthesis failed: {str(e)}")


def _extract_paragraph_chunk(text: str, max_tokens: int = 200) -> dict:
    """
    Extract sentence-level chunks that stay under token limits for Chatterbox TTS.
    Chatterbox has a max_cache_len of ~1500, so chunks must be small.
    
    Returns dict with 'text' and 'end_pos' or None if no complete chunk found.
    """
    if not text.strip():
        return None
    
    import re
    
    # Look for sentence endings (period, exclamation, question mark)
    sentence_pattern = r'([.!?])(?:\s|$)'
    matches = list(re.finditer(sentence_pattern, text))
    
    # Calculate char limit from token limit (Chatterbox uses ~1.5 tokens per char)
    char_limit = int(max_tokens / 1.5)  # ~133 chars for 200 tokens
    
    if not matches:
        # No clear break points, use char limit
        if len(text) <= char_limit:
            return {'text': text, 'end_pos': len(text)}
        else:
            # Find last space before char limit
            chunk_text = text[:char_limit]
            last_space = chunk_text.rfind(' ')
            if last_space > 0:
                return {'text': text[:last_space], 'end_pos': last_space}
            else:
                return {'text': text[:char_limit], 'end_pos': char_limit}
    
    # Find the best break point that stays under token limit
    best_end = 0
    for match in matches:
        chunk_end = match.end()
        chunk_text = text[:chunk_end]
        estimated_tokens = int(len(chunk_text) * 1.5)  # Chatterbox token estimation
        
        if estimated_tokens <= max_tokens:
            best_end = chunk_end  # This is still safe
        else:
            break  # This would exceed limit, stop here
    
    if best_end > 0:
        return {'text': text[:best_end].strip(), 'end_pos': best_end}
    else:
        # No sentence break found within limit, force break at word boundary
        if len(text) <= char_limit:
            return {'text': text, 'end_pos': len(text)}
        chunk_text = text[:char_limit]
        last_space = chunk_text.rfind(' ')
        if last_space > 0:
            return {'text': text[:last_space], 'end_pos': last_space}
        return {'text': text[:char_limit], 'end_pos': char_limit}


def _split_text_for_chunked_generation(text: str, max_tokens: int = 200) -> list:
    """Split text into sentence-level chunks for Chatterbox TTS"""
    chunks = []
    remaining_text = text
    
    # Calculate char limit (Chatterbox uses ~1.5 tokens per char)
    char_limit = int(max_tokens / 1.5)
    
    while remaining_text.strip():
        chunk_info = _extract_paragraph_chunk(remaining_text, max_tokens)
        
        if chunk_info:
            chunks.append(chunk_info['text'])
            remaining_text = remaining_text[chunk_info['end_pos']:].lstrip()
        else:
            # No good break point found, force break at safe character limit
            if len(remaining_text) <= char_limit:
                chunks.append(remaining_text.strip())
                break
            else:
                # Force break at last space before limit
                chunk_text = remaining_text[:char_limit]
                last_space = chunk_text.rfind(' ')
                if last_space > 0:
                    chunks.append(remaining_text[:last_space].strip())
                    remaining_text = remaining_text[last_space:].lstrip()
                else:
                    # No space found, just break here
                    chunks.append(chunk_text)
                    remaining_text = remaining_text[char_limit:]
    
    return [chunk for chunk in chunks if chunk.strip()]

async def _synthesize_with_chatterbox(
    text: str, 
    audio_prompt_path: str = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> bytes:
    """Synthesize speech using Chatterbox TTS and return raw audio bytes."""
    global CHATTERBOX_VOICE_WARMED_UP
    import time
    import tempfile
    import os
    import soundfile as sf
    import asyncio
    import torch
    import inspect
    from concurrent.futures import ThreadPoolExecutor
    
    loop = asyncio.get_event_loop()
    total_start_time = time.perf_counter()

    if audio_prompt_path and not os.path.isabs(audio_prompt_path):
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        full_path = voices_dir / audio_prompt_path
        if full_path.exists():
            audio_prompt_path = str(full_path)
        else:
            audio_prompt_path = None
    
    try:
        model = load_chatterbox_model()
        if not model:
            raise RuntimeError("Failed to load Chatterbox model")

        # One-time voice warmup - MUST BE IN EXECUTOR
        if audio_prompt_path and not CHATTERBOX_VOICE_WARMED_UP:
            def _warmup():
                with torch.inference_mode():
                    # model.generate("warm up", audio_prompt_path=audio_prompt_path)
                    _call_chatterbox_generate(model, "warm up", audio_prompt_path=audio_prompt_path)
            
            logger.info("🔥 [Chatterbox] Warming up voice reference on persistent thread...")
            await loop.run_in_executor(get_chatterbox_executor(), _warmup)
            CHATTERBOX_VOICE_WARMED_UP = True
        
        # UPDATED: Added temperature to enable sampling so exaggeration works
        generation_kwargs = {
            'temperature': 0.7,  # Enable sampling (crucial for emotion exaggeration)
            'exaggeration': exaggeration,
            'cfg_weight': cfg,
        }
        
        # if audio_prompt_path and os.path.exists(audio_prompt_path):
        #     generation_kwargs['audio_prompt_path'] = audio_prompt_path
        
        synthesis_start = time.perf_counter()
        
        # CHECK IF TEXT NEEDS CHUNKING
        # Chatterbox has max_cache_len of ~1500, so we need to keep chunks small
        # Rough estimate: ~1.5 tokens per char for Chatterbox
        estimated_tokens = int(len(text) * 1.5)
        
        if estimated_tokens > 400:  # Use chunked generation for texts that might hit cache limits  
            logger.debug(f"🔀 Long text detected ({len(text)} chars, ~{estimated_tokens} tokens), using chunked generation")
            
            # Split text into smaller chunks - Chatterbox needs chunks under ~800 chars to stay within cache
            text_chunks = _split_text_for_chunked_generation(text, max_tokens=200)
            logger.debug(f"🔀 Split into {len(text_chunks)} chunks")
            
            # Generate all chunks and concatenate audio
            all_audio_chunks = []
            
            # Use persistent executor
            executor = get_chatterbox_executor()
            for i, chunk in enumerate(text_chunks):
                logger.debug(f"🔀 Generating chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
                
                def _generate_chunk(c=chunk): # Bind chunk locally
                    with torch.inference_mode():
                        return _call_chatterbox_generate(model, c, audio_prompt_path, **generation_kwargs)
                
                # Generate this chunk
                try:
                    # Check for cancellation before starting next chunk
                    if asyncio.current_task() and asyncio.current_task().cancelled():
                        logger.warning("🛑 [Chatterbox] Synthesis cancelled before chunk generation.")
                        raise asyncio.CancelledError()

                    chunk_audio = await loop.run_in_executor(executor, _generate_chunk)
                    all_audio_chunks.append(chunk_audio)
                    
                    logger.info(f"✅ Chunk {i+1}/{len(text_chunks)} complete")

                    # Check for cancellation after chunk generation
                    if asyncio.current_task() and asyncio.current_task().cancelled():
                        logger.warning("🛑 [Chatterbox] Synthesis cancelled after chunk generation.")
                        raise asyncio.CancelledError()

                except asyncio.CancelledError:
                    logger.warning("🛑 [Chatterbox] Synthesis loop cancelled.")
                    raise # Re-raise to stop the loop
            
            # Concatenate all audio chunks
            audio_tensor = torch.cat(all_audio_chunks, dim=-1)
            logger.info(f"🔀 Concatenated {len(all_audio_chunks)} chunks into final audio")
            
        else:
            # STANDARD SINGLE GENERATION for short texts
            def _generate():
                with torch.inference_mode():
                    return _call_chatterbox_generate(model, text, audio_prompt_path, **generation_kwargs)
            
            # Run the blocking operation in the persistent thread
            audio_tensor = await loop.run_in_executor(get_chatterbox_executor(), _generate)
        
        # Calculate and log RTF
        synthesis_time = time.perf_counter() - synthesis_start
        if hasattr(model, 'sr') and model.sr:
            total_audio_length = audio_tensor.shape[-1] if hasattr(audio_tensor, 'shape') else len(audio_tensor)
            audio_duration = total_audio_length / model.sr
            rtf = synthesis_time / audio_duration
            logger.info(f"🚀 RTF: {rtf:.3f} ({synthesis_time:.2f}s for {audio_duration:.2f}s audio)")

        # Convert to audio bytes (rest of the function stays the same)
        if hasattr(audio_tensor, 'detach'):
            audio_tensor = audio_tensor.detach()
        if hasattr(audio_tensor, 'to'):
            audio_tensor = audio_tensor.to('cpu')
        
        if hasattr(audio_tensor, 'numpy'):
            audio_numpy = audio_tensor.numpy()
        elif hasattr(audio_tensor, 'cpu'):
            audio_numpy = audio_tensor.cpu().numpy()
        else:
            audio_numpy = audio_tensor
        
        if len(audio_numpy.shape) > 1:
            audio_numpy = audio_numpy.squeeze()
        
        # File I/O can also be done in executor if needed
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            sf.write(temp_path, audio_numpy, model.sr)
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return audio_bytes

    except Exception as e:
        logger.error(f"Chatterbox synthesis failed: {e}")
        raise RuntimeError(f"Chatterbox synthesis failed: {str(e)}")

def _call_chatterbox_generate(model, text, audio_prompt_path=None, **kwargs):
    """
    Dynamically adapts to the Chatterbox model's generate signature.
    Handles versions that require 'language_id' vs those that don't.
    """
    sig = inspect.signature(model.generate)
    params = sig.parameters
    
    call_args = [text]
    call_kwargs = kwargs.copy()
    
    # 1. Handle audio_prompt_path (Position 2 in some versions)
    # Check if 'audio_prompt_path' is a named parameter
    if 'audio_prompt_path' in params:
        call_kwargs['audio_prompt_path'] = audio_prompt_path
    else:
        # If not named but we have enough positional args, assume pos 2
        # But safer to just look at the list of params
        param_names = list(params.keys())
        if len(param_names) > 1 and param_names[1] == 'audio_prompt_path':
             call_kwargs['audio_prompt_path'] = audio_prompt_path
        elif len(param_names) > 1:
             # Fallback: some versions might take it as 2nd arg?
             # For now, let's assume if it's not in kwargs it might be positional 2
             # But usually it is a kwarg in new versions. 
             # Refined strategy: If explicitly present in signature, use it.
             pass

    # Actually, simplistic approach:
    # Version A: (text, audio_prompt_path=None, ...)
    # Version B: (text, audio_prompt_path, language_id, ...)
    
    # Let's try to construct args based on known patterns
    if 'language_id' in params:
        # Version B likely: might need language_id passed
        # Check if it has a default
        if params['language_id'].default == inspect.Parameter.empty:
             # Mandatory language_id
             if 'audio_prompt_path' in params and list(params.keys()).index('audio_prompt_path') < list(params.keys()).index('language_id'):
                 # It's likely (text, audio_prompt_path, language_id)
                 return model.generate(text, audio_prompt_path, "en", **kwargs)
             else:
                 call_kwargs['language_id'] = "en"

    # Standard call for Version A (text, audio_prompt_path=...)
    if audio_prompt_path:
        call_kwargs['audio_prompt_path'] = audio_prompt_path
        
    return model.generate(text, **call_kwargs)


async def _synthesize_with_kokoro(text: str, voice: str, speed: float = 1.0) -> bytes:
    """Synthesize speech using Kokoro TTS and return raw audio bytes."""
    import time
    import numpy as np
    import tempfile

    # Validate voice — Kokoro only supports native voice codes (e.g. af_heart, am_adam)
    # or local .pt files. Chatterbox-style filenames (.wav, .wav.pt, etc.) will fail.
    if not voice or voice == 'default':
        voice = 'af_heart'
    elif not voice.endswith('.pt') and any(c in voice for c in '.\\/'):
        logger.warning(f"⚠️ [Kokoro] Voice '{voice}' looks like a file path, not a Kokoro native voice. Falling back to 'af_heart'")
        voice = 'af_heart'

    speed = max(0.25, min(4.0, float(speed or 1.0)))

    try:
        total_start_time = time.perf_counter()

        # --- 1. Pipeline Loading ---
        load_start_time = time.perf_counter()
        assumed_lang_code = 'a'
        pipeline = load_tts_pipeline(lang_code=assumed_lang_code)
        load_end_time = time.perf_counter()
        
        if not pipeline:
            raise RuntimeError("Failed to load Kokoro TTS pipeline")
        
        logger.info(f"🗣️ [Kokoro] Synthesizing with voice '{voice}', speed={speed}x: '{text[:50]}...'")
        
        # --- 2. Core TTS Inference ---
        synth_start_time = time.perf_counter()
        generator = pipeline(text, voice=voice, speed=speed)
        audio_chunks = [audio for _, _, audio in generator]
        synth_end_time = time.perf_counter()
        
        if not audio_chunks:
            raise RuntimeError("No audio chunks generated")

        # --- 3. Audio Post-Processing ---
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

        # --- Timing Logs ---
        logger.info(f"⏱️ [Kokoro] Pipeline Loading: {(load_end_time - load_start_time) * 1000:.2f}ms")
        logger.info(f"⏱️ [Kokoro] TTS Synthesis: {(synth_end_time - synth_start_time) * 1000:.2f}ms")
        logger.info(f"⏱️ [Kokoro] Post-Processing: {(post_end_time - post_start_time) * 1000:.2f}ms")
        logger.info(f"⏱️ [Kokoro] Total Time: {(total_end_time - total_start_time) * 1000:.2f}ms")
        logger.info(f"✅ [Kokoro] Generated {len(audio_bytes)} bytes of audio")
        
        return audio_bytes

    except Exception as e:
        logger.error(f"❌ [Kokoro] Synthesis failed: {e}", exc_info=True)
        raise RuntimeError(f"Kokoro synthesis failed: {str(e)}")


async def _synthesize_with_chatterbox_turbo(
    text: str, 
    audio_prompt_path: str = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> bytes:
    """Synthesize speech using Chatterbox TTS (Turbo) and return raw audio bytes."""
    global CHATTERBOX_TURBO_VOICE_WARMED_UP
    import time
    import tempfile
    import os
    import soundfile as sf
    import asyncio
    import torch
    
    loop = asyncio.get_event_loop()
    
    if audio_prompt_path and not os.path.isabs(audio_prompt_path):
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        full_path = voices_dir / audio_prompt_path
        if full_path.exists():
            audio_prompt_path = str(full_path)
        else:
            audio_prompt_path = None
    
    try:
        model = load_chatterbox_turbo_model()
        if not model:
            raise RuntimeError("Failed to load Chatterbox Turbo model")

        # One-time voice warmup - MUST BE IN EXECUTOR
        if audio_prompt_path and not CHATTERBOX_TURBO_VOICE_WARMED_UP:
            def _warmup_turbo():
                with torch.inference_mode():
                    # Turbo generate signature: text, audio_prompt_path=None
                    model.generate("warm up", audio_prompt_path=audio_prompt_path)
            
            logger.info("🔥 [Chatterbox Turbo] Warming up voice reference on persistent thread...")
            await loop.run_in_executor(get_chatterbox_executor(), _warmup_turbo)
            CHATTERBOX_TURBO_VOICE_WARMED_UP = True
        
        # Generation kwargs for Turbo
        generation_kwargs = {
            'temperature': 0.8,
            # 'exaggeration': exaggeration, # Not supported in Turbo
            # 'cfg_weight': cfg, # Not supported in Turbo
        }
        
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            generation_kwargs['audio_prompt_path'] = audio_prompt_path
        
        synthesis_start = time.perf_counter()
        
        # CHECK IF TEXT NEEDS CHUNKING
        # Using same logic as standard Chatterbox for paranoia/safety, though Turbo handles long text better
        estimated_tokens = int(len(text) * 1.5)
        
        if estimated_tokens > 400:  # Use chunked generation for texts that might hit cache limits  
            logger.info(f"🔀 [Turbo] Long text detected ({len(text)} chars, ~{estimated_tokens} tokens), using chunked generation")
            
            text_chunks = _split_text_for_chunked_generation(text, max_tokens=200)
            logger.info(f"🔀 [Turbo] Split into {len(text_chunks)} chunks")
            
            all_audio_chunks = []
            executor = get_chatterbox_executor()
            
            for i, chunk in enumerate(text_chunks):
                logger.info(f"🔀 [Turbo] Generating chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
                
                def _generate_chunk_turbo(c=chunk): 
                    with torch.inference_mode():
                        return model.generate(c, **generation_kwargs)
                
                chunk_audio = await loop.run_in_executor(executor, _generate_chunk_turbo)
                all_audio_chunks.append(chunk_audio)
                
                logger.info(f"✅ [Turbo] Chunk {i+1}/{len(text_chunks)} complete")
            
            audio_tensor = torch.cat(all_audio_chunks, dim=-1)
            
        else:
            # STANDARD SINGLE GENERATION
            def _generate_turbo():
                with torch.inference_mode():
                    return model.generate(text, **generation_kwargs)
            
            audio_tensor = await loop.run_in_executor(get_chatterbox_executor(), _generate_turbo)
        
        # Calculate and log RTF
        synthesis_time = time.perf_counter() - synthesis_start
        if hasattr(model, 'sr') and model.sr:
            total_audio_length = audio_tensor.shape[-1] if hasattr(audio_tensor, 'shape') else len(audio_tensor)
            audio_duration = total_audio_length / model.sr
            rtf = synthesis_time / audio_duration
            logger.info(f"🚀 [Turbo] RTF: {rtf:.3f} ({synthesis_time:.2f}s for {audio_duration:.2f}s audio)")

        # Convert to audio bytes
        if hasattr(audio_tensor, 'detach'):
            audio_tensor = audio_tensor.detach()
        if hasattr(audio_tensor, 'to'):
            audio_tensor = audio_tensor.to('cpu')
        
        if hasattr(audio_tensor, 'numpy'):
            audio_numpy = audio_tensor.numpy()
        elif hasattr(audio_tensor, 'cpu'):
            audio_numpy = audio_tensor.cpu().numpy()
        else:
            audio_numpy = audio_tensor
        
        if len(audio_numpy.shape) > 1:
            audio_numpy = audio_numpy.squeeze()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            sf.write(temp_path, audio_numpy, model.sr)
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return audio_bytes

    except Exception as e:
        logger.error(f"Chatterbox Turbo synthesis failed: {e}")
        raise RuntimeError(f"Chatterbox Turbo synthesis failed: {str(e)}")


async def _synthesize_with_chatterbox_nano(
    text: str, 
    audio_prompt_path: str = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> bytes:
    """Synthesize speech using Chatterbox TTS (Nano) and return raw audio bytes."""
    global CHATTERBOX_NANO_VOICE_WARMED_UP
    import time
    import tempfile
    import os
    import soundfile as sf
    import asyncio
    import torch
    
    loop = asyncio.get_event_loop()
    
    if audio_prompt_path and not os.path.isabs(audio_prompt_path):
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        full_path = voices_dir / audio_prompt_path
        if full_path.exists():
            audio_prompt_path = str(full_path)
        else:
            audio_prompt_path = None
    
    try:
        model = load_chatterbox_nano_model()
        if not model:
            raise RuntimeError("Failed to load Chatterbox Nano model")

        # One-time voice warmup - MUST BE IN EXECUTOR
        if audio_prompt_path and not CHATTERBOX_NANO_VOICE_WARMED_UP:
            def _warmup_nano():
                with torch.inference_mode():
                    model.generate("warm up", audio_prompt_path=audio_prompt_path)
            
            logger.info("🔥 [Chatterbox Nano] Warming up voice reference on persistent thread...")
            await loop.run_in_executor(get_chatterbox_executor(), _warmup_nano)
            CHATTERBOX_NANO_VOICE_WARMED_UP = True
        
        # Generation kwargs for Nano
        generation_kwargs = {
            'temperature': 0.8,
        }
        
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            generation_kwargs['audio_prompt_path'] = audio_prompt_path
        
        synthesis_start = time.perf_counter()
        
        # CHECK IF TEXT NEEDS CHUNKING
        # Nano has small max_text_tokens/max_speech_tokens, so chunking is important
        estimated_tokens = int(len(text) * 1.5)
        
        if estimated_tokens > 400:  # Use chunked generation for texts that might hit cache limits  
            logger.info(f"🔀 [Nano] Long text detected ({len(text)} chars, ~{estimated_tokens} tokens), using chunked generation")
            
            text_chunks = _split_text_for_chunked_generation(text, max_tokens=200)
            logger.info(f"🔀 [Nano] Split into {len(text_chunks)} chunks")
            
            all_audio_chunks = []
            executor = get_chatterbox_executor()
            
            for i, chunk in enumerate(text_chunks):
                logger.info(f"🔀 [Nano] Generating chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
                
                def _generate_chunk_nano(c=chunk): 
                    with torch.inference_mode():
                        return model.generate(c, **generation_kwargs)
                
                chunk_audio = await loop.run_in_executor(executor, _generate_chunk_nano)
                all_audio_chunks.append(chunk_audio)
                
                logger.info(f"✅ [Nano] Chunk {i+1}/{len(text_chunks)} complete")
            
            audio_tensor = torch.cat(all_audio_chunks, dim=-1)
            
        else:
            # STANDARD SINGLE GENERATION
            def _generate_nano():
                with torch.inference_mode():
                    return model.generate(text, **generation_kwargs)
            
            audio_tensor = await loop.run_in_executor(get_chatterbox_executor(), _generate_nano)
        
        # Calculate and log RTF
        synthesis_time = time.perf_counter() - synthesis_start
        if hasattr(model, 'sr') and model.sr:
            total_audio_length = audio_tensor.shape[-1] if hasattr(audio_tensor, 'shape') else len(audio_tensor)
            audio_duration = total_audio_length / model.sr
            rtf = synthesis_time / audio_duration
            logger.info(f"🚀 [Nano] RTF: {rtf:.3f} ({synthesis_time:.2f}s for {audio_duration:.2f}s audio)")

        # Convert to audio bytes
        if hasattr(audio_tensor, 'detach'):
            audio_tensor = audio_tensor.detach()
        if hasattr(audio_tensor, 'to'):
            audio_tensor = audio_tensor.to('cpu')
        
        if hasattr(audio_tensor, 'numpy'):
            audio_numpy = audio_tensor.numpy()
        elif hasattr(audio_tensor, 'cpu'):
            audio_numpy = audio_tensor.cpu().numpy()
        else:
            audio_numpy = audio_tensor
        
        if len(audio_numpy.shape) > 1:
            audio_numpy = audio_numpy.squeeze()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            sf.write(temp_path, audio_numpy, model.sr)
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return audio_bytes

    except Exception as e:
        logger.error(f"Chatterbox Nano synthesis failed: {e}")
        raise RuntimeError(f"Chatterbox Nano synthesis failed: {str(e)}")


async def _synthesize_with_voxcpm(
    text: str,
    audio_prompt_path: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 8,
    normalize: bool = False,
    denoise: bool = False,
    retry_badcase: bool = False,
    voice_design: str = None,
) -> bytes:
    """Synthesize speech using VoxCPM2 and return raw audio bytes (WAV)."""
    import time
    import asyncio

    loop = asyncio.get_event_loop()

    if audio_prompt_path and not os.path.isabs(audio_prompt_path):
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        full_path = voices_dir / audio_prompt_path
        if full_path.exists():
            audio_prompt_path = str(full_path)
        else:
            audio_prompt_path = None

    try:
        model = load_voxcpm_model()
        if not model:
            raise RuntimeError("Failed to load VoxCPM2 model")

        synthesis_start = time.perf_counter()

        generation_kwargs = {
            'cfg_value': cfg_value,
            'inference_timesteps': inference_timesteps,
            'normalize': normalize,
            'denoise': denoise,
            'retry_badcase': retry_badcase,
            'max_len': 2048,
        }

        if voice_design and voice_design.strip():
            text = f"({voice_design.strip()}){text}"
        elif audio_prompt_path and os.path.exists(audio_prompt_path):
            generation_kwargs['reference_wav_path'] = audio_prompt_path

        voxcpm_executor = get_voxcpm_executor()

        def _generate():
            return model.generate(text=text, **generation_kwargs)

        audio_numpy = await loop.run_in_executor(voxcpm_executor, _generate)

        # Calculate RTF
        synthesis_time = time.perf_counter() - synthesis_start
        sr = model.tts_model.sample_rate if hasattr(model, 'tts_model') else 48000
        audio_duration = len(audio_numpy) / sr
        rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"🚀 [VoxCPM2] RTF: {rtf:.3f} ({synthesis_time:.2f}s for {audio_duration:.2f}s audio)")

        # Convert to WAV bytes
        if hasattr(audio_numpy, 'detach'):
            audio_numpy = audio_numpy.detach().cpu().numpy()
        if len(audio_numpy.shape) > 1:
            audio_numpy = audio_numpy.squeeze()

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            sf.write(temp_path, audio_numpy, sr)
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return audio_bytes

    except Exception as e:
        logger.error(f"VoxCPM2 synthesis failed: {e}")
        raise RuntimeError(f"VoxCPM2 synthesis failed: {str(e)}")


# --- NEW BACKEND STREAMING LOGIC ---


class TTSStreamer:
    """Manages buffering and synthesizing text chunks for a single WebSocket connection."""
    def __init__(self, websocket: WebSocket, tts_settings=None):
        self._websocket = websocket
        self._text_buffer = ""
        self._synthesis_queue = asyncio.Queue()
        self._is_active = True
        self._has_queued_before = False  # FIX: Track if we've ever queued anything
        
        # Store TTS settings
        self._tts_settings = tts_settings or {
            'engine': 'kokoro',
            'voice': 'af_heart',
            'exaggeration': 0.5,
            'cfg': 0.5,
            'audio_prompt_path': None
        }
        
        self._synthesis_task = asyncio.create_task(self.synthesis_loop())

    async def add_text(self, text_data: str):
        if not self._is_active: return
        
        try:
            # Parse JSON data
            data = json.loads(text_data)
            text = data.get('text', '')
            if not text:
                logger.warning(f"⚠️ [Streamer] Received empty text in JSON: {text_data}")
                return
                
            self._text_buffer += text
            self._find_and_queue_chunks()
            
        except json.JSONDecodeError:
            logger.error(f"❌ [Streamer] Invalid JSON: {text_data}")
            return
        except Exception as e:
            logger.error(f"❌ [Streamer] Error processing text: {e}")

    def _clear_synthesis_queue(self):
        """Helper method to clear all items from the synthesis queue"""
        cleared_count = 0
        while not self._synthesis_queue.empty():
            try:
                item = self._synthesis_queue.get_nowait()
                if item is not None:  # Don't count sentinel values
                    cleared_count += 1
                self._synthesis_queue.task_done()
            except:
                break
        if cleared_count > 0:
            logger.info(f"🧹 [Streamer] Cleared {cleared_count} items from synthesis queue")

    def _find_and_queue_chunks(self):
        """
        Fast chunking for all chunks: minimum 8 words, then break at next punctuation.
        This ensures consistent chunk sizes and better RTF for uninterrupted playback.
        """
        import re
        
        # Chatterbox benefits from fast first chunk extraction
        is_slow_engine = self._tts_settings.get('engine') in ('chatterbox', 'chatterbox_turbo', 'chatterbox_nano', 'voxcpm')

        # For slow engines (Chatterbox, VoxCPM): configurable sentence grouping.
        # Minimum 3 sentences/chunk — the buffer waits for enough text to arrive
        # from the LLM before queueing synthesis, so chunks respect the grouping
        # setting. Leftover text is flushed by finish() at stream end.
        if is_slow_engine:
            target_sentences = self._get_stream_chunk_sentence_count()
            logger.debug(f"🧩 [Streamer] Slow engine active, target={target_sentences} sentences/chunk, buffer={len(self._text_buffer)} chars")
            while True:
                chunk_info = self._extract_n_sentence_chunk(self._text_buffer, target_sentences)
                if not chunk_info:
                    # Only fall back to comma-based splitting for very long run-on
                    # text (30+ words with no sentence boundary) to avoid stalling
                    # on malformed input.
                    words = self._text_buffer.split()
                    if len(words) >= 30:
                        chunk_info = self._extract_smart_chunk(self._text_buffer)
                if chunk_info:
                    chunk_text = chunk_info['text']
                    self._text_buffer = self._text_buffer[chunk_info['end_pos']:]
                    if chunk_text.strip():
                        self._synthesis_queue.put_nowait(chunk_text.strip())
                        self._has_queued_before = True
                        logger.debug(f"🧠 [Streamer] Queued {target_sentences}-sentence chunk: '{chunk_text[:60]}...'")
                else:
                    break
            return

        is_first_extraction = is_slow_engine and not self._has_queued_before

        if is_first_extraction:
            # For slow engines (Chatterbox) first extraction: prioritize fast first chunk
            # Look for first sentence or comma after 5 words for immediate playback
            first_chunk = self._extract_first_chunk_fast(self._text_buffer)
            
            if first_chunk:
                engine_name = self._tts_settings.get('engine', 'Unknown').title()
                logger.debug(f"✅ [CHUNK LOGIC - {engine_name} First] Fast first chunk: '{first_chunk['text'][:100]}...'")
                
                # Update buffer by removing what we processed
                self._text_buffer = self._text_buffer[first_chunk['end_pos']:]
                
                self._synthesis_queue.put_nowait(first_chunk['text'])
                self._has_queued_before = True  # Mark that we've queued
                logger.debug(f"🧠 [Streamer] Queued fast first {engine_name} chunk: '{first_chunk['text'][:60]}...'")

        else:
            # Standard behavior: use fast chunking for ALL subsequent chunks too
            while True:
                chunk_info = self._extract_fast_chunk(self._text_buffer)
                
                if chunk_info:
                    chunk_text = chunk_info['text']
                    logger.debug(f"✅ [CHUNK LOGIC] Found fast chunk: '{chunk_text}'")
                    logger.debug(f"✅ [CHUNK LOGIC] Buffer before removal: '{self._text_buffer[:100]}...'")

                    self._text_buffer = self._text_buffer[chunk_info['end_pos']:]
                    logger.debug(f"✅ [CHUNK LOGIC] Buffer after removal: '{self._text_buffer[:100]}...'")

                    if chunk_text.strip():
                        self._synthesis_queue.put_nowait(chunk_text.strip())
                        self._has_queued_before = True  # Mark that we've queued
                        logger.debug(f"🧠 [Streamer] Queued fast chunk for synthesis: '{chunk_text[:60]}...'")
                else:
                    break

    async def synthesis_loop(self):
        """The 'consumer' loop that synthesizes sentences from the queue."""
        import time

        while True:
            try:
                sentence = await self._synthesis_queue.get()

                if sentence is None:
                    logger.info("🛑 [Streamer] Sentinel received. Synthesis loop is shutting down.")
                    break

                # Check active status just before expensive synthesis
                if not self._is_active:
                     logger.critical("🛑 [Streamer] Streamer is inactive. Skipping synthesis.")
                     self._clear_synthesis_queue()
                     break

                logger.debug(f"🎤 [Streamer] Synthesizing: '{sentence[:60]}...'")
                
                start_time = time.perf_counter()

                # Pass a check function? No, just rely on task cancellation or check after.
                audio_bytes = await synthesize_speech(
                    text=sentence,
                    voice=self._tts_settings['voice'],
                    engine=self._tts_settings['engine'],
                    audio_prompt_path=self._tts_settings.get('audio_prompt_path'),
                    exaggeration=self._tts_settings.get('exaggeration', 0.5),
                    cfg=self._tts_settings.get('cfg', 0.5),
                    speed=self._tts_settings.get('speed', 1.0),
                    voxcpm_cfg_value=self._tts_settings.get('voxcpm_cfg_value', 2.0),
                    voxcpm_inference_timesteps=self._tts_settings.get('voxcpm_inference_timesteps', 10),
                    voxcpm_normalize=self._tts_settings.get('voxcpm_normalize', True),
                    voxcpm_denoise=self._tts_settings.get('voxcpm_denoise', True),
                    voxcpm_retry_badcase=self._tts_settings.get('voxcpm_retry_badcase', True),
                    voxcpm_voice_design=self._tts_settings.get('voxcpm_voice_design'),
                )

                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                logger.debug(f"⏱️ [Streamer] Synthesis task took {duration_ms:.2f}ms")

                if audio_bytes:
                    # Check if WebSocket is still connected before sending
                    if self._websocket.client_state.value == 1:  # CONNECTED state
                        chunk_duration_ms = None
                        try:
                            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                                frames = wf.getnframes()
                                rate = wf.getframerate() or 1
                                chunk_duration_ms = (frames / rate) * 1000
                        except Exception:
                            chunk_duration_ms = None

                        try:
                            await self._websocket.send_text(json.dumps({
                                "type": "tts_chunk",
                                "text": sentence,
                                "duration_ms": chunk_duration_ms
                            }))
                            logger.info(f"🎛️ [Subtitle Cue] Sent chunk cue ({chunk_duration_ms}ms): '{sentence[:60]}...'")
                        except Exception:
                            pass

                        await self._websocket.send_bytes(audio_bytes)
                        logger.info(f"✅ [Streamer] Sent audio chunk of {len(audio_bytes)} bytes.")
                    else:
                        logger.warning(f"⚠️ [Streamer] WebSocket in state {self._websocket.client_state.value}, clearing synthesis queue")
                        self._clear_synthesis_queue()
                        break  # Exit synthesis loop since client disconnected
                
                self._synthesis_queue.task_done()

            except asyncio.CancelledError:
                logger.info("🛑 [Streamer] Synthesis task was cancelled.")
                self._clear_synthesis_queue()
                break
            except Exception as e:
                error_msg = str(e)
                if "Cannot call" in error_msg and "close" in error_msg:
                    logger.warning("⚠️ [Streamer] WebSocket closed during send, clearing synthesis queue")
                    self._clear_synthesis_queue()
                    break
                else:
                    logger.error(f"❌ [Streamer] Error in synthesis loop: {e}", exc_info=True)

    def _extract_word_chunk(self, text: str) -> dict:
        """
        Extract a chunk based on word count: minimum 10 words, 
        then break at next comma/period/question mark/exclamation mark.
        
        Returns dict with 'text' and 'end_pos' or None if no complete chunk found.
        """
        if not text.strip():
            return None
            
        words = text.split()
        if len(words) < 10:
            return None  # Need at least 10 words
        
        # Find the position after the 10th word
        word_positions = []
        current_pos = 0
        
        for i, word in enumerate(words):
            start_pos = text.find(word, current_pos)
            end_pos = start_pos + len(word)
            word_positions.append((start_pos, end_pos))
            current_pos = end_pos
            
            if i == 9:  # After 10th word (0-indexed)
                break
        
        # Now look for punctuation after the 10th word
        search_start = word_positions[9][1]  # End of 10th word
        remaining_text = text[search_start:]
        
        # Find next punctuation mark
        punctuation_match = re.search(r'[,.!?]', remaining_text)
        
        if punctuation_match:
            # Found punctuation, extract up to and including it
            chunk_end = search_start + punctuation_match.end()
            chunk_text = text[:chunk_end].strip()
            return {'text': chunk_text, 'end_pos': chunk_end}
        else:
            # No punctuation found after 10 words, no chunk available yet
            return None

    def _extract_smart_chunk(self, text: str) -> dict:
        """
        Unified smart chunking logic:
        Only split on sentence-ending punctuation (. ! ?).
        Never split on commas — Kokoro is fast enough for full sentences.
        Skip consecutive periods (ellipsis) and title abbreviations (Mr., Dr., etc).
        """
        if not text.strip():
            return None
            
        import re
        
        pattern = r'(?<!\.)\.(?!\.)|[!?]'
            
        for match in re.finditer(pattern, text):
            # Exception Logic: Check for Title Abbreviations (Mr., Dr., etc.)
            if match.group() == '.':
                # Check what comes before this dot
                end_pos = match.start()
                prefix = text[:end_pos].rstrip()
                
                # Common abbreviations that end in dot but aren't sentence ends
                # Case-insensitive check for common titles at the end of the prefix
                if re.search(r'(?i)\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Rev|Capt|Gen|Col|Lt|St)$', prefix):
                    # It's likely a title (e.g., "Hello Mr."), skip this split point
                    continue

            # If we're here, it's a valid split point
            chunk_end = match.end()
            chunk_text = text[:chunk_end].strip()
            
            if chunk_text:
                return {'text': chunk_text, 'end_pos': chunk_end}
            
        return None

    def _extract_first_chunk_fast(self, text: str) -> dict:
        """Wrapper for smart chunking (logic is now unified)"""
        return self._extract_smart_chunk(text)

    def _extract_fast_chunk(self, text: str) -> dict:
         """Wrapper for smart chunking (logic is now unified)"""
         return self._extract_smart_chunk(text)

    def _extract_one_sentence_chunk(self, text: str) -> dict:
        """
        First full sentence using . ! ? as enders; ignore decimal points (e.g. 3.2).
        """
        return self._extract_n_sentence_chunk(text, 1)

    def _extract_two_sentence_chunk(self, text: str) -> dict:
        """
        Extract exactly two full sentences using simple regex markers.
        Avoids complex heuristics for faster, more predictable chunking.
        """
        return self._extract_n_sentence_chunk(text, 2)

    def _extract_n_sentence_chunk(self, text: str, sentence_count: int) -> dict:
        """Extract exactly N full sentences using . ! ? as enders; ignore decimal points."""
        if not text or not text.strip():
            return None

        import re
        target = max(1, int(sentence_count or 1))
        pattern = re.compile(r'(?:(?<!\d)(?<!\.)\.(?!\.)(?=\s|$)|[!?](?=\s|$))')
        matches = list(pattern.finditer(text))
        if len(matches) < target:
            return None

        chunk_end = matches[target - 1].end()
        chunk_text = text[:chunk_end].strip()
        if not chunk_text:
            return None
        return {'text': chunk_text, 'end_pos': chunk_end}

    def _get_stream_chunk_sentence_count(self) -> int:
        """Read per-stream sentence grouping from settings payload (clamped to 3–12)."""
        try:
            raw = self._tts_settings.get('stream_chunk_sentences', 3)
            value = int(raw)
        except Exception:
            value = 3
        return max(3, min(12, value))

    def finish(self):
        # Check if there's any leftover text in the buffer and queue it
        if self._is_active and self._text_buffer.strip():
            logger.info(f"🧠 [Streamer] Queuing final buffered text: '{self._text_buffer.strip()[:60]}...'")
            self._synthesis_queue.put_nowait(self._text_buffer.strip())
        
        # Add the 'None' sentinel to the queue
        logger.info("🏁 [Streamer] Queuing sentinel to gracefully stop the synthesis loop.")
        self._synthesis_queue.put_nowait(None) 
        
        # DO NOT mark inactive here! Wait for sentinel to stop loop.
        # self._is_active = False
    
    async def cancel(self):
        self._is_active = False
        self._clear_synthesis_queue()  # Clear queue before cancelling task
        self._synthesis_task.cancel()
        try:
            await self._synthesis_task
        except asyncio.CancelledError:
            logger.info("🛑 [Streamer] Synthesis task cancelled successfully.")

@router.websocket("/tts-stream")
async def tts_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ [WebSocket] Connection accepted. Ready for new message streams.")
    streamer = None

    try:
        # This primary loop handles multiple, separate message streams over a single connection.
        while True:
            # 1. Expect TTS settings at the start of each new message stream.
            logger.info("👂 [WebSocket] Waiting for new stream (expecting settings)...")
            settings_data = await websocket.receive_text()

            try:
                import json
                tts_settings = json.loads(settings_data)
                logger.info(f"📋 [WebSocket] stream_chunk_sentences={tts_settings.get('stream_chunk_sentences')}, engine={tts_settings.get('engine')}")
            except json.JSONDecodeError:
                logger.warning("⚠️ [WebSocket] Expected settings JSON but received invalid data")
                # If we don't get settings, we can't proceed with this stream.
                # The loop will continue, waiting for the next valid settings message.
                continue

            # 2. A new, valid stream is starting. Create a streamer instance for it.
            # If an old one exists (e.g., from a client error), cancel it.
            if streamer and streamer._synthesis_task and not streamer._synthesis_task.done():
                await streamer.cancel()
            
            streamer = TTSStreamer(websocket, tts_settings)
            logger.info("✅ Streamer created for new message.")

            # 3. This inner loop processes all text chunks for the CURRENT message stream.
            while True:
                text_chunk = await websocket.receive_text()

                if text_chunk == "--END--":
                    logger.info("🏁 [WebSocket] End-of-stream signal received.")
                    streamer.finish()
                    # Wait for any queued synthesis to complete.
                    if streamer._synthesis_task:
                        await streamer._synthesis_task
                    try:
                        await websocket.send_text(json.dumps({"type": "tts_done"}))
                        logger.info("✅ [WebSocket] Sent tts_done after all audio chunks.")
                    except Exception:
                        logger.warning("⚠️ [WebSocket] Failed to send tts_done.")
                    streamer = None # Clear the streamer.
                    break  # Exit the inner loop to await the next message stream's settings.
                else:
                    # It's a regular text chunk, add it to the current streamer's queue.
                    await streamer.add_text(text_chunk)
            
            logger.info("✅ [WebSocket] Message stream finished. Ready for next.")

    except WebSocketDisconnect:
        logger.warning("🔌 [WebSocket] Frontend disconnected.")
        if streamer and streamer._synthesis_task and not streamer._synthesis_task.done():
            await streamer.cancel() # Clean up the task on disconnect.
    except Exception as e:
        logger.error(f"❌ [WebSocket] An unexpected error occurred in the stream handler: {e}", exc_info=True)
        if streamer and streamer._synthesis_task and not streamer._synthesis_task.done():
            await streamer.cancel() # Attempt cleanup on error.

# --- TTS SETTINGS ENDPOINTS ---

@router.post("/tts/save-speed-mode")
async def save_speed_mode(request: dict):
    """Save TTS speed mode to settings"""
    try:
        speed_mode = request.get('tts_speed_mode', 'ultra_fast')
        
        # Load current settings
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        settings = {}
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        
        # Update speed mode
        settings['tts_speed_mode'] = speed_mode
        
        # Save settings
        settings_path.parent.mkdir(exist_ok=True)
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"🔧 [Settings] TTS speed mode saved: {speed_mode}")
        return {"status": "success", "tts_speed_mode": speed_mode}
        
    except Exception as e:
        logger.error(f"❌ [Settings] Error saving speed mode: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/tts/unload-chatterbox")
async def unload_chatterbox():
    """Unload Chatterbox model from VRAM to free up memory"""
    global chatterbox_model, CHATTERBOX_VOICE_WARMED_UP
    
    try:
        if chatterbox_model is None:
            logger.info("🔓 [Chatterbox] Model is already unloaded")
            return {
                "status": "success", 
                "message": "Chatterbox model was already unloaded",
                "vram_freed": 0
            }
        
        logger.info("🔓 [Chatterbox] Unloading model to free VRAM...")
        
        # Delete the model
        del chatterbox_model
        chatterbox_model = None
        CHATTERBOX_VOICE_WARMED_UP = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🔓 [Chatterbox] CUDA cache cleared")
        
        logger.info("✅ [Chatterbox] Model unloaded successfully, VRAM freed")
        return {
            "status": "success",
            "message": "Chatterbox model unloaded successfully",
            "vram_freed": "~5GB"
        }
        
    except Exception as e:
        logger.error(f"❌ [Chatterbox] Error unloading model: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/tts/reload-chatterbox")
async def reload_chatterbox():
    """Reload Chatterbox model for use"""
    global chatterbox_model
    
    try:
        if chatterbox_model is not None:
            logger.info("🔄 [Chatterbox] Model is already loaded")
            return {
                "status": "success",
                "message": "Chatterbox model is already loaded",
                "already_loaded": True
            }
        
        logger.info("🔄 [Chatterbox] Reloading model...")
        
        # Load the model (will trigger full loading + warmup)
        model = load_chatterbox_model()
        
        if model is None:
            raise RuntimeError("Failed to load Chatterbox model")
        
        logger.info("✅ [Chatterbox] Model reloaded successfully")
        return {
            "status": "success",
            "message": "Chatterbox model loaded and ready for use",
            "already_loaded": False
        }
        
    except Exception as e:
        logger.error(f"❌ [Chatterbox] Error reloading model: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/tts/unload-chatterbox-turbo")
async def unload_chatterbox_turbo():
    """Unload Chatterbox Turbo model from VRAM"""
    global chatterbox_turbo_model, CHATTERBOX_TURBO_VOICE_WARMED_UP
    
    try:
        if chatterbox_turbo_model is None:
            return {"status": "success", "message": "Chatterbox Turbo already unloaded"}
        
        del chatterbox_turbo_model
        chatterbox_turbo_model = None
        CHATTERBOX_TURBO_VOICE_WARMED_UP = False
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ [Chatterbox Turbo] Model unloaded successfully")
        return {"status": "success", "message": "Chatterbox Turbo unloaded"}
        
    except Exception as e:
        logger.error(f"❌ [Chatterbox Turbo] Error unloading: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/tts/reload-chatterbox-turbo")
async def reload_chatterbox_turbo():
    """Reload Chatterbox Turbo model"""
    global chatterbox_turbo_model
    try:
        if chatterbox_turbo_model is not None:
             return {"status": "success", "message": "Already loaded"}
        
        model = load_chatterbox_turbo_model()
        if model is None:
            raise RuntimeError("Failed to load Chatterbox Turbo")
            
        return {"status": "success", "message": "Chatterbox Turbo loaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/tts/unload-chatterbox-nano")
async def unload_chatterbox_nano():
    """Unload Chatterbox Nano model from VRAM"""
    global chatterbox_nano_model, CHATTERBOX_NANO_VOICE_WARMED_UP
    
    try:
        if chatterbox_nano_model is None:
            return {"status": "success", "message": "Chatterbox Nano already unloaded"}
        
        del chatterbox_nano_model
        chatterbox_nano_model = None
        CHATTERBOX_NANO_VOICE_WARMED_UP = False
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ [Chatterbox Nano] Model unloaded successfully")
        return {"status": "success", "message": "Chatterbox Nano unloaded"}
        
    except Exception as e:
        logger.error(f"❌ [Chatterbox Nano] Error unloading: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/tts/reload-chatterbox-nano")
async def reload_chatterbox_nano():
    """Reload Chatterbox Nano model"""
    global chatterbox_nano_model
    try:
        if chatterbox_nano_model is not None:
             return {"status": "success", "message": "Already loaded"}
        
        model = load_chatterbox_nano_model()
        if model is None:
            raise RuntimeError("Failed to load Chatterbox Nano")
            
        return {"status": "success", "message": "Chatterbox Nano loaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/tts/unload-voxcpm")
async def unload_voxcpm():
    """Unload VoxCPM2 model from VRAM to free up memory"""
    global voxcpm_model, VOXCPM_VOICE_WARMED_UP

    try:
        if voxcpm_model is None:
            return {"status": "success", "message": "VoxCPM2 already unloaded"}

        logger.info("🔓 [VoxCPM2] Unloading model to free VRAM...")

        del voxcpm_model
        voxcpm_model = None
        VOXCPM_VOICE_WARMED_UP = False

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🔓 [VoxCPM2] CUDA cache cleared")

        logger.info("✅ [VoxCPM2] Model unloaded successfully")
        return {
            "status": "success",
            "message": "VoxCPM2 model unloaded successfully",
            "vram_freed": "~8GB"
        }

    except Exception as e:
        logger.error(f"❌ [VoxCPM2] Error unloading model: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/tts/reload-voxcpm")
async def reload_voxcpm():
    """Reload VoxCPM2 model for use"""
    global voxcpm_model

    try:
        if voxcpm_model is not None:
            return {"status": "success", "message": "VoxCPM2 already loaded", "already_loaded": True}

        logger.info("🔄 [VoxCPM2] Reloading model...")
        model = load_voxcpm_model()
        if model is None:
            raise RuntimeError("Failed to load VoxCPM2")

        return {"status": "success", "message": "VoxCPM2 loaded", "already_loaded": False}
    except Exception as e:
        logger.error(f"❌ [VoxCPM2] Error reloading model: {e}")
        return {"status": "error", "message": str(e)}


def get_voxcpm_executor():
    """Returns a dedicated ThreadPoolExecutor for VoxCPM2 (separate from Chatterbox)."""
    global VOXCPM_EXECUTOR
    if VOXCPM_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor
        VOXCPM_EXECUTOR = ThreadPoolExecutor(max_workers=1)
        logger.info("🧵 [VoxCPM2] Dedicated thread executor initialized")
    return VOXCPM_EXECUTOR


def load_voxcpm_model():
    """Load the VoxCPM2 model with torch.compile optimization enabled."""
    global voxcpm_model

    if VoxCPM is None:
        raise RuntimeError("voxcpm library is not installed or import failed.")

    if voxcpm_model is None:
        try:
            target_device = get_tts_device()
            # VoxCPM's optimize() method requires exact device string "cuda" (not "cuda:0")
            voxcpm_device = "cuda" if target_device.startswith("cuda") else target_device
            logger.info(f"Loading VoxCPM2 model onto device {voxcpm_device} (optimize=True)...")

            voxcpm_model = VoxCPM.from_pretrained(
                "openbmb/VoxCPM2",
                load_denoiser=False,
                optimize=True,
                device=voxcpm_device,
            )

            logger.info("✅ VoxCPM2 model loaded successfully (optimize=True).")

        except Exception as e:
            logger.error(f"Failed to load VoxCPM2 model: {e}", exc_info=True)
            voxcpm_model = None
            raise RuntimeError(f"Failed to load VoxCPM2 model: {e}") from e

    return voxcpm_model


# --- VoxCPM2 GGUF Functions ---

def _get_voxcpm_gguf_models_dir() -> Path:
    """Get the directory for VoxCPM2 GGUF model files."""
    from .runtime_paths import data_path

    models_dir = data_path("models", "voxcpm_gguf")
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _get_voxcpm_cli_binary() -> Optional[str]:
    """Find the voxcpm2-cli binary."""
    backend_dir = Path(__file__).resolve().parent.parent
    candidates = [
        backend_dir / "llama.cpp-omni" / "build" / "tools" / "omni" / "voxcpm2-cli.exe",
        backend_dir / "llama.cpp-omni" / "build" / "tools" / "omni" / "voxcpm2-cli",
        backend_dir / "llama.cpp-omni" / "build" / "bin" / "voxcpm2-cli.exe",
        backend_dir / "llama.cpp-omni" / "build" / "bin" / "voxcpm2-cli",
        backend_dir / "voxcpm2-cli.exe",
        backend_dir / "voxcpm2-cli",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    import shutil
    system_cli = shutil.which("voxcpm2-cli")
    if system_cli:
        return system_cli
    return None


def is_voxcpm_gguf_available() -> bool:
    """Check if voxcpm2-cli binary is available."""
    return _get_voxcpm_cli_binary() is not None


def list_voxcpm_gguf_downloaded_models() -> list:
    """List downloaded VoxCPM2 GGUF model files."""
    models_dir = _get_voxcpm_gguf_models_dir()
    downloaded = []
    for model_id, info in VOXCPM_GGUF_MODELS.items():
        gguf_path = models_dir / info["filename"]
        if gguf_path.exists():
            downloaded.append({
                "model_id": model_id,
                "filename": info["filename"],
                "size_mb": info["size_mb"],
                "path": str(gguf_path),
                "label": info["label"],
                "component": info["component"],
            })
    return downloaded


async def download_voxcpm_gguf_model(model_id: str, progress_callback=None) -> tuple:
    """Download a VoxCPM2 GGUF model file from HuggingFace."""
    model_id = model_id.strip()
    if model_id not in VOXCPM_GGUF_MODELS:
        return False, f"Unknown model: {model_id}"
    
    info = VOXCPM_GGUF_MODELS[model_id]
    filename = info["filename"]
    models_dir = _get_voxcpm_gguf_models_dir()
    dest_path = models_dir / filename
    
    if dest_path.exists():
        return True, f"Already downloaded: {filename}"

    try:
        from huggingface_hub import hf_hub_download
        if progress_callback:
            await progress_callback(f"Downloading {filename} ({info['size_mb']} MB)...")

        def _download():
            return hf_hub_download(
                repo_id=VOXCPM_GGUF_HF_REPO,
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


async def delete_voxcpm_gguf_model(filename: str) -> tuple:
    """Delete a downloaded VoxCPM2 GGUF model file."""
    models_dir = _get_voxcpm_gguf_models_dir()
    target = models_dir / filename
    if not target.exists():
        return False, f"File not found: {filename}"
    try:
        target.unlink()
        logger.info(f"Deleted {filename}")
        return True, f"Deleted {filename}"
    except Exception as e:
        return False, f"Delete failed: {e}"


async def _synthesize_with_voxcpm_gguf(
    text: str,
    audio_prompt_path: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    voice_design: str = None,
) -> bytes:
    """Synthesize speech using VoxCPM2 GGUF via voxcpm2-cli and return raw audio bytes."""
    import time
    import asyncio

    cli_binary = _get_voxcpm_cli_binary()
    if not cli_binary:
        raise RuntimeError(
            "voxcpm2-cli binary not found. Build llama.cpp-omni first: "
            "git clone https://github.com/tc-mb/llama.cpp-omni && cd llama.cpp-omni && "
            "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target voxcpm2-cli -j"
        )

    models_dir = _get_voxcpm_gguf_models_dir()
    baselm_path = models_dir / "VoxCPM2-BaseLM-Q8_0.gguf"
    acoustic_path = models_dir / "VoxCPM2-Acoustic-F16.gguf"
    
    if not baselm_path.exists():
        raise RuntimeError(
            f"BaseLM model not downloaded. Download VoxCPM2-BaseLM-Q8_0.gguf from Settings > TTS > VoxCPM2 GGUF Models."
        )
    if not acoustic_path.exists():
        raise RuntimeError(
            f"Acoustic model not downloaded. Download VoxCPM2-Acoustic-F16.gguf from Settings > TTS > VoxCPM2 GGUF Models."
        )

    if audio_prompt_path and not os.path.isabs(audio_prompt_path):
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        full_path = voices_dir / audio_prompt_path
        if full_path.exists():
            audio_prompt_path = str(full_path)
        else:
            audio_prompt_path = None

    if voice_design and voice_design.strip():
        text = f"({voice_design.strip()}){text}"

    temp_wav_path = None
    try:
        temp_wav_path = tempfile.mktemp(suffix=".wav")
        
        cmd = [
            cli_binary,
            "-t", text,
            "-o", temp_wav_path,
            "--cfg", str(cfg_value),
            "--timesteps", str(inference_timesteps),
        ]
        
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            cmd.extend(["-r", audio_prompt_path])
        
        cmd.extend([str(baselm_path), str(acoustic_path)])

        logger.info(f"VoxCPM2 GGUF: synthesizing with cfg={cfg_value}, timesteps={inference_timesteps}")

        loop = asyncio.get_event_loop()

        def _run_cli():
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            return result

        synthesis_start = time.perf_counter()
        result = await loop.run_in_executor(None, _run_cli)
        synthesis_time = time.perf_counter() - synthesis_start

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:500]
            raise RuntimeError(f"voxcpm2-cli failed (exit {result.returncode}): {stderr_snippet}")

        if not os.path.exists(temp_wav_path):
            raise RuntimeError("voxcpm2-cli did not produce output file")

        with open(temp_wav_path, 'rb') as f:
            audio_bytes = f.read()

        logger.info(f"✅ [VoxCPM2 GGUF] Synthesis completed in {synthesis_time:.2f}s, {len(audio_bytes)} bytes")
        return audio_bytes

    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


def comprehensive_model_warmup():
    """Warm up all models and compilation paths on startup using the persistent executor"""
    import time
    global chatterbox_model, CHATTERBOX_VOICE_WARMED_UP
    
    executor = get_chatterbox_executor()

    def _warmup_logic():
        try:
            model = load_chatterbox_model()
            if not model:
                return
            
            # 2. Basic model warm-up (default voice)
            with torch.inference_mode():
                _call_chatterbox_generate(model, "Warming up the model for optimal performance.")
            
            # 3. Additional synthesis warm-up
            with torch.inference_mode():
                _call_chatterbox_generate(model, "Testing additional synthesis for compilation.")
            
            # 4. Voice cloning warm-up (if default voice file exists)
            voices_dir = Path(__file__).parent / "static" / "voice_references"
            default_voice_files = ["default.wav", "narrator.wav", "sample.wav"]
            
            for voice_file in default_voice_files:
                voice_path = voices_dir / voice_file
                if voice_path.exists():
                    try:
                        with torch.inference_mode():
                            _call_chatterbox_generate(
                                model,
                                "Voice cloning warm up test.",
                                audio_prompt_path=str(voice_path)
                            )
                        global CHATTERBOX_VOICE_WARMED_UP
                        CHATTERBOX_VOICE_WARMED_UP = True
                        break 
                    except Exception as e:
                        logger.warning(f"⚠️ Voice cloning warm-up failed for {voice_file}: {e}")
            
            # 5. Clear artifacts
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Model warm-up logic failed: {e}")

    # Submit to persistent thread and WAIT for it to complete
    logger.info("🔥 [Chatterbox] Submitting comprehensive warmup to persistent thread...")
    future = executor.submit(_warmup_logic)
    future.result() # Synchronous wait
    logger.info("✅ [Chatterbox] Comprehensive warmup complete")


def load_chatterbox_turbo_model():
    """Load the vendored Chatterbox Turbo model"""
    global chatterbox_turbo_model
    
    if ChatterboxTurboTTS is None:
         raise RuntimeError("Chatterbox Turbo library failed to import.")

    if chatterbox_turbo_model is None:
        try:
            target_device = get_tts_device()
            logger.info(f"Loading Chatterbox TURBO TTS model onto device {target_device}...")
            
            # Force CUDA context if applicable (same logic as main Chatterbox)
            if target_device.startswith('cuda:'):
                import torch
                device_id = int(target_device.split(':')[1])
                torch.cuda.set_device(device_id)

            chatterbox_turbo_model = ChatterboxTurboTTS.from_pretrained(device=target_device)
            
            # Basic warnup
            if not os.environ.get('CUDA_VISIBLE_DEVICES', ''):
                 with torch.inference_mode():
                     chatterbox_turbo_model.generate("Turbo model warmup.", temperature=0.8)
            
            logger.info("✅ Chatterbox TURBO model loaded successfully.")

            # Pre-cache voices from settings (similar logic to main Chatterbox)
            settings = load_settings()
            voice_cache = settings.get('voice_cache', [])
            if voice_cache:
                voices_dir = Path(__file__).parent / "static" / "voice_references"
                for voice_entry in voice_cache:
                    if voice_entry.get('engine') == 'chatterbox_turbo':
                        voice_id = voice_entry.get('voice_id')
                        if voice_id and voice_id != 'default':
                            voice_path = voices_dir / voice_id
                            if voice_path.exists():
                                try:
                                    chatterbox_turbo_model.prepare_conditionals(str(voice_path))
                                    logger.info(f"🔥 [Turbo] Cached voice: {voice_id}")
                                except Exception as e:
                                    logger.warning(f"⚠️ [Turbo] Failed to cache voice {voice_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to load Chatterbox Turbo model: {e}", exc_info=True)
            chatterbox_turbo_model = None
            raise RuntimeError(f"Failed to load Chatterbox Turbo model: {e}") from e

    return chatterbox_turbo_model

def load_chatterbox_nano_model():
    """Load the vendored Chatterbox Nano model"""
    global chatterbox_nano_model
    
    if ChatterboxNanoTTS is None:
         raise RuntimeError("Chatterbox Nano library failed to import.")

    if chatterbox_nano_model is None:
        try:
            target_device = get_tts_device()
            logger.info(f"Loading Chatterbox NANO TTS model onto device {target_device}...")
            
            # Force CUDA context if applicable
            if target_device.startswith('cuda:'):
                import torch
                device_id = int(target_device.split(':')[1])
                torch.cuda.set_device(device_id)

            chatterbox_nano_model = ChatterboxNanoTTS.from_pretrained(device=target_device)
            
            # Basic warmup
            if not os.environ.get('CUDA_VISIBLE_DEVICES', ''):
                 with torch.inference_mode():
                     chatterbox_nano_model.generate("Nano model warmup.", temperature=0.8)
            
            logger.info("✅ Chatterbox NANO model loaded successfully.")

            # Pre-cache voices from settings
            settings = load_settings()
            voice_cache = settings.get('voice_cache', [])
            if voice_cache:
                voices_dir = Path(__file__).parent / "static" / "voice_references"
                for voice_entry in voice_cache:
                    if voice_entry.get('engine') == 'chatterbox_nano':
                        voice_id = voice_entry.get('voice_id')
                        if voice_id and voice_id != 'default':
                            voice_path = voices_dir / voice_id
                            if voice_path.exists():
                                try:
                                    chatterbox_nano_model.prepare_conditionals(str(voice_path))
                                    logger.info(f"🔥 [Nano] Cached voice: {voice_id}")
                                except Exception as e:
                                    logger.warning(f"⚠️ [Nano] Failed to cache voice {voice_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to load Chatterbox Nano model: {e}", exc_info=True)
            chatterbox_nano_model = None
            raise RuntimeError(f"Failed to load Chatterbox Nano model: {e}") from e

    return chatterbox_nano_model

def load_chatterbox_model():
    """Enhanced model loading with comprehensive warm-up"""
    global chatterbox_model
    
    # Add logging to track when this function is called
    import traceback
    caller_info = traceback.extract_stack()[-2]
    logger.info(f"🔍 load_chatterbox_model() called from: {caller_info.filename}:{caller_info.lineno}")
    logger.info(f"🔍 Process environment: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if ChatterboxTTS is None:
        raise RuntimeError("chatterbox-tts library is not installed or import failed.")
    
    if chatterbox_model is None:
        try:
            logger.info(f"Loading Chatterbox TTS model onto device {get_tts_device()}...")
            
            # CRITICAL: Force CUDA to use only our assigned device
            import torch
            if torch.cuda.is_available():
                # Get the device from our device detection logic
                target_device = get_tts_device()
                if target_device.startswith('cuda:'):
                    device_id = int(target_device.split(':')[1])
                    torch.cuda.set_device(device_id)
                    logger.info(f"🔒 CUDA device locked to: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
                    
                    # Check device count for logging
                    device_count = torch.cuda.device_count()
                    logger.info(f"🔍 Process sees {device_count} GPU(s)")
                    
                    # In single GPU mode, we expect to see 1 GPU; in dual GPU mode with isolation, we expect 1 GPU
                    if device_count == 1:
                        logger.info(f"✅ Single GPU mode confirmed: only {device_count} GPU visible")
                    else:
                        logger.info(f"🔍 Multi-GPU mode: {device_count} GPUs visible")
                else:
                    logger.info(f"🔒 Using device: {target_device}")
            
            # CRITICAL: Force the model to load on the specified device
            target_device = get_tts_device()
            logger.info(f"🔒 Loading model on device: {target_device}")
            
            # Force CUDA device before model loading
            if target_device.startswith('cuda:'):
                device_id = int(target_device.split(':')[1])
                torch.cuda.set_device(device_id)
                logger.info(f"🔒 CUDA device locked to: {torch.cuda.current_device()}")
            
            chatterbox_model = ChatterboxTTS.from_pretrained(device=target_device)
            
            # DEBUG: Log all available attributes of the model
            logger.info("🔍 ChatterboxTTS model attributes:")
            for attr in dir(chatterbox_model):
                if not attr.startswith('_'):
                    try:
                        value = getattr(chatterbox_model, attr)
                        if not callable(value):
                            logger.info(f"🔍   {attr}: {type(value)} = {value}")
                    except Exception as e:
                        logger.info(f"🔍   {attr}: <error accessing: {e}>")
            
            # CRITICAL: Verify model is on correct device
            if hasattr(chatterbox_model, 'device'):
                actual_device = str(chatterbox_model.device)
                logger.info(f"🔍 Model reports device: {actual_device}")
                if actual_device != target_device:
                    logger.warning(f"⚠️ Model loaded on {actual_device}, expected {target_device}")
                    # Force move to correct device
                    chatterbox_model = chatterbox_model.to(target_device)
                    logger.info(f"🔒 Model moved to {target_device}")
            
            # CRITICAL: Check model parameters device (ChatterboxTTS might not have named_parameters)
            try:
                if hasattr(chatterbox_model, 'named_parameters'):
                    for name, param in chatterbox_model.named_parameters():
                        if param.device.type != target_device.split(':')[0]:
                            logger.warning(f"⚠️ Parameter {name} on wrong device: {param.device}")
                            break
                    else:
                        logger.info(f"✅ All model parameters on correct device: {target_device}")
                else:
                    # ChatterboxTTS doesn't have named_parameters, check alternative attributes
                    logger.info("🔍 ChatterboxTTS model - checking alternative device attributes")
                    
                    # Check if model has a device attribute
                    if hasattr(chatterbox_model, 'device'):
                        logger.info(f"✅ Model device attribute: {chatterbox_model.device}")
                    
                    # Check if model has a model attribute that might contain parameters
                    if hasattr(chatterbox_model, 'model'):
                        logger.info(f"✅ Model has 'model' attribute: {type(chatterbox_model.model)}")
                    
                    # Check if model has a tts attribute
                    if hasattr(chatterbox_model, 'tts'):
                        logger.info(f"✅ Model has 'tts' attribute: {type(chatterbox_model.tts)}")
                    
                    logger.info("✅ ChatterboxTTS model device verification complete")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not verify model parameters device: {e}")
                logger.info("✅ Continuing with model loading...")
            
            # Double-check: Ensure the model is actually on the correct device
            if hasattr(chatterbox_model, 'device'):
                actual_device = str(chatterbox_model.device)
                logger.info(f"🔍 Model reports device: {actual_device}")
                if actual_device != DEVICE:
                    logger.warning(f"⚠️ Model loaded on {actual_device}, expected {DEVICE}")
            
            logger.info("Chatterbox TTS model loaded successfully.")

            # === ADD THIS: Comprehensive warm-up ===
            # Only run warm-up if we're in the main process (not model service)
            # This prevents double warm-up across multiple processes
            if not os.environ.get('CUDA_VISIBLE_DEVICES', ''):
                logger.info("🔥 Running comprehensive warm-up in main process...")
                comprehensive_model_warmup()
            else:
                logger.info("🔒 Skipping warm-up in isolated process (model service)")

            # === REPLACE THIS SECTION: Pre-cache voice references from settings ===
            settings = load_settings()
            voice_cache = settings.get('voice_cache', [])
            
            if voice_cache:
                logger.info(f"🔥 [Chatterbox] Pre-caching {len(voice_cache)} voice references from settings...")
                voices_dir = Path(__file__).parent / "static" / "voice_references"
                
                for voice_entry in voice_cache:
                    if voice_entry.get('engine') == 'chatterbox':
                        voice_id = voice_entry.get('voice_id')
                        if voice_id and voice_id != 'default':
                            voice_path = voices_dir / voice_id
                            if voice_path.exists():
                                try:
                                    logger.info(f"🔥 [Chatterbox] Pre-caching voice: {voice_id}")
                                    chatterbox_model.prepare_conditionals(str(voice_path))
                                    logger.info(f"✅ [Chatterbox] Cached voice: {voice_id}")
                                except Exception as e:
                                    logger.warning(f"⚠️ [Chatterbox] Failed to cache voice {voice_id}: {e}")
                            else:
                                logger.warning(f"⚠️ [Chatterbox] Voice file not found: {voice_path}")
                
                logger.info("✅ [Chatterbox] Voice pre-caching complete.")
            else:
                logger.info("📝 [Chatterbox] No voices found in settings cache")

        except Exception as e:
            logger.error(f"Failed to load Chatterbox TTS model: {e}", exc_info=True)
            chatterbox_model = None
            raise RuntimeError(f"Failed to load Chatterbox TTS model: {e}") from e
    
    return chatterbox_model
