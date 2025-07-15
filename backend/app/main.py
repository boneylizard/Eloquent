# --- Imports ---
import os
os.environ["CUDA_MODULE_LOADING"] = "EAGER" # Ensure CUDA modules load eagerly
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["GGML_CUDA_NO_PINNED"] = "0"

# Actually FORCE CUDA initialization with llama_cpp

# --- END: DEFINITIVE GPU ISOLATION ---
from pyexpat.errors import messages
from fastapi import FastAPI, HTTPException, Depends, APIRouter, File, UploadFile, BackgroundTasks, Request, Query, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import datetime
from contextlib import asynccontextmanager
from . import memory_intelligence
from .memory_routes import memory_router
import httpx
import logging
from fastapi.logger import logger # Use FastAPI's logger
import sys
import time
import shutil
import uuid
from .tts_service import clean_markdown_for_tts, _synthesize_with_kokoro
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import re
from .model_manager import ModelManager
from . import inference
from .tts_service import synthesize_speech # Assuming this is the correct import path for your TTS service
import io
import tempfile
from . import tts_service # Assuming this is the correct import path for your TTS service
from .stt_service import transcribe_audio # Assuming this is the correct import path for your STT service
import uuid
from .inference import generate_text
from . import dual_chat_utils as dcu # Assuming this is the correct import path for your dual chat util
import base64
import threading
from .Document_routes import document_router
from . import rag_utils # Assuming this is the correct import path for your RAG utils
import logging
import signal
from PIL import Image, PngImagePlugin
import requests
from io import BytesIO
import base64
from . import character_intelligence
from .sd_manager import SDManager
import random
from .web_search_service import perform_web_search
from .openai_compat import router as openai_router
import pynvml
from .kyutai_streaming_service import kyutai_streaming, create_text_stream_from_llm, create_sentence_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Only set DEBUG-level loggers to WARNING to suppress their excessive output
logging.getLogger('numba.core').setLevel(logging.WARNING)
logging.getLogger('numba.byteflow').setLevel(logging.WARNING)
logging.getLogger('graphviz').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# NeMo's logging can be very verbose at DEBUG level
logging.getLogger('nemo').setLevel(logging.WARNING)
logging.getLogger('nemo.collections').setLevel(logging.WARNING)
logging.getLogger('nemo.utils').setLevel(logging.WARNING)

app = FastAPI() # Re-initialize app to avoid conflicts
SINGLE_GPU_MODE = False # Set to True if running on a single GPU

# --- Environment Variable Settings ---
# Disable tokenizer parallelism to potentially avoid warnings/issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Ensure stdout/stderr are unbuffered, might help logs appear faster
os.environ["PYTHONUNBUFFERED"] = "1"

async def get_model_manager(request: Request): # Changed signature to use Request state
    # This safely accesses the model manager from app.state
    # Ensure app.state is correctly populated in lifespan
    if not hasattr(request.app.state, 'model_manager'):
         logger.error("ModelManager not found in application state!")
         raise HTTPException(status_code=500, detail="ModelManager not initialized")
    yield request.app.state.model_manager
def strip_ui_wrappers(s: str) -> str:
    """
    Remove any lines that are just the model’s name or "’s avatar,"
    and drop triple-backtick fences.
    """
    cleaned_lines = []
    for line in s.splitlines():
        text = line.strip()
        # 1) skip lines like "ModelName" or "ModelName's avatar"
        if re.match(r"^[\w\-.]+(?:'s avatar)?$", text):
            continue
        # 2) skip markdown fences
        if text.startswith("```"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    model_name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop: List[str] = []
    stream: bool = False
    use_rag: bool = False # Keep if used elsewhere
    rag_docs: List[str] = [] # Keep if used elsewhere
    gpu_id: Optional[int] = None
    userProfile: Optional[Dict[str, Any]] = None
    is_dual_chat: bool = False # Added for dual chat support
    messages: Optional[List[Dict[str, Any]]] = None  # To support the frontend's messages array
    echo: bool = False # Added for echo functionality
    active_character: Optional[Dict[str, Any]] = None  # Add this field
    request_purpose: Optional[str] = None # <<< ADD THIS LINE
    use_web_search: bool = False  # NEW: Web search toggle
    web_search_query: Optional[str] = None  # NEW: Optional web search query
    image_base64: Optional[str] = None  # NEW: Optional base64-encoded image for context
    image_type: Optional[str] = None  # NEW: Optional image type (e.g., "png", "jpg")
    # Add any other fields you need for your request

class ImageRequest(BaseModel): # Keep for now
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.0
    sampler: str = "Euler a"
    seed: int = -1

class DocumentQuery(BaseModel): # Keep for now
    query: str
    doc_ids: List[str]
    top_k: int = 5

# --- FastAPI App Setup ---
# Assume 'app = FastAPI(...)' is defined correctly
app = FastAPI(title="LLM Frontend API") # Example instantiation

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # Ensure your frontend origin is listed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(tags=["generate"])
router = APIRouter()   # Re-initialize router to avoid conflicts




# --- Static Files Setup ---
# Define the base static directory path
# Using Path(__file__).parent makes the 'static' directory relative to the main.py file's location
base_dir = Path(__file__).parent
static_dir = base_dir / "static"
generated_images_dir = static_dir / "generated_images" # Define the subdirectory path

# Ensure both directories exist
try:
    static_dir.mkdir(parents=True, exist_ok=True)
    generated_images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Static directory ensured: {static_dir.resolve()}")
    logger.info(f"Generated images directory ensured: {generated_images_dir.resolve()}")
except OSError as e:
    logger.error(f"FATAL: Failed to create static directories: {e}", exc_info=True)
    # Depending on severity, you might want to exit here or raise an exception

# Mount the base static directory
if not static_dir.is_dir():
    logger.error(f"Static directory path is invalid or not found: {static_dir.resolve()}")
else:
    logger.info(f"Attempting to mount static directory: {static_dir.resolve()}")
    # Mount the static directory to the /static URL path
    app.mount("/static", StaticFiles(directory=str(static_dir.resolve())), name="static")

# Add this to your existing FastAPI app
async def generate_llm_response(prompt: str, model_manager, model_name: str, **kwargs) -> str:
    """
    Adapter for your existing LLM generation
    Uses your actual inference function
    """
    from .inference import generate_text
    
    # Call your existing LLM logic with proper parameters
    response = await generate_text(
        model_manager=model_manager,
        model_name=model_name,
        prompt=prompt,
        max_tokens=kwargs.get('max_tokens', 1024),
        temperature=kwargs.get('temperature', 0.7),
        top_p=kwargs.get('top_p', 0.9),
        top_k=kwargs.get('top_k', 40),
        repetition_penalty=kwargs.get('repetition_penalty', 1.1),
        stop_sequences=kwargs.get('stop_sequences', []),
        gpu_id=kwargs.get('gpu_id', 0)
    )
    return response
@app.websocket("/ws/stream-tts")
async def websocket_streaming_tts(websocket: WebSocket, model_manager: ModelManager = Depends(get_model_manager)):
    """
    WebSocket endpoint for real-time streaming TTS
    
    Protocol:
    Client sends: {"type": "start", "prompt": "...", "voice_reference": "path/to/voice.wav"}
    Server sends: {"type": "audio_chunk", "data": "base64_audio_data"}
    Server sends: {"type": "complete"} when done
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        # Wait for initial message
        data = await websocket.receive_text()
        message = json.loads(data)
        
        if message.get("type") != "start":
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "Expected 'start' message"
            }))
            return
        
        prompt = message.get("prompt")
        voice_reference = message.get("voice_reference")  # Optional voice cloning
        use_streaming_llm = message.get("stream_llm", True)  # Whether to stream from LLM
        
        if not prompt:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "No prompt provided"
            }))
            return
        
        logger.info(f"🗣️ [WebSocket] Starting streaming TTS session {session_id}")
        
        # Start Kyutai session
        kyutai_session_id = await kyutai_streaming.start_stream(voice_reference)
        
        # Create text stream based on mode
        if use_streaming_llm:
            # Stream directly from LLM as it generates
            text_stream = create_text_stream_from_llm(
                prompt=prompt,
                model_manager=model_manager,
                model_name=message.get("model_name", "default"),  # Get from message
                max_tokens=message.get("max_tokens", 1024),
                temperature=message.get("temperature", 0.7),
                gpu_id=message.get("gpu_id", 0)
            )
        else:
            # For testing: convert complete LLM response to sentence stream
            llm_response = await generate_llm_response(prompt, model_manager, message.get("model_name", "default"))
            text_stream = create_sentence_stream(llm_response)
        
        # Send confirmation to client
        await websocket.send_text(json.dumps({
            "type": "started",
            "session_id": session_id
        }))
        
        # Stream audio chunks to client
        async for audio_chunk in kyutai_streaming.stream_text_and_get_audio(text_stream, kyutai_session_id):
            try:
                # Encode audio as base64 for WebSocket transmission
                import base64
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                await websocket.send_text(json.dumps({
                    "type": "audio_chunk",
                    "data": audio_b64,
                    "session_id": session_id
                }))
                
            except WebSocketDisconnect:
                logger.info(f"🗣️ [WebSocket] Client disconnected from session {session_id}")
                break
        
        # Send completion message
        await websocket.send_text(json.dumps({
            "type": "complete",
            "session_id": session_id
        }))
        
        logger.info(f"🗣️ [WebSocket] Completed streaming TTS session {session_id}")
        
    except WebSocketDisconnect:
        logger.info(f"🗣️ [WebSocket] Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"🗣️ [WebSocket] Error in session {session_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
                "session_id": session_id
            }))
        except:
            pass
    finally:
        # Cleanup
        await kyutai_streaming.disconnect()

@app.websocket("/ws/chat-stream")
async def websocket_chat_with_streaming_tts(websocket: WebSocket, model_manager: ModelManager = Depends(get_model_manager)):
    """
    Combined chat + streaming TTS WebSocket endpoint
    
    This replaces your existing chat flow with real-time audio
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        while True:
            # Wait for message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat":
                prompt = message.get("message")
                voice_reference = message.get("voice_reference")
                tts_enabled = message.get("tts_enabled", True)
                model_name = message.get("model_name", "default")
                
                if not prompt:
                    continue
                
                logger.info(f"🗣️ [Chat Stream] New message in session {session_id}: {prompt[:50]}...")
                
                if tts_enabled:
                    # Start TTS streaming
                    kyutai_session_id = await kyutai_streaming.start_stream(voice_reference)
                    
                    # Create streaming text from LLM
                    text_stream = create_text_stream_from_llm(
                        prompt=prompt,
                        model_manager=model_manager,
                        model_name=model_name,
                        max_tokens=message.get("max_tokens", 1024),
                        temperature=message.get("temperature", 0.7),
                        gpu_id=message.get("gpu_id", 0)
                    )
                    
                    # Send both text and audio to client
                    text_buffer = ""
                    
                    async for audio_chunk in kyutai_streaming.stream_text_and_get_audio(text_stream, kyutai_session_id):
                        import base64
                        audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                        
                        await websocket.send_text(json.dumps({
                            "type": "response_chunk",
                            "text": text_buffer,  # Accumulated text for display
                            "audio": audio_b64,
                            "session_id": session_id
                        }))
                        
                        text_buffer = ""  # Reset after sending
                
                else:
                    # Text-only mode (your existing flow)
                    response = await generate_llm_response(prompt, model_manager, model_name)
                    await websocket.send_text(json.dumps({
                        "type": "response_complete",
                        "text": response,
                        "session_id": session_id
                    }))
            
            elif message.get("type") == "interrupt":
                # Handle user interruption
                logger.info(f"🗣️ [Chat Stream] User interrupted session {session_id}")
                await kyutai_streaming.disconnect()
                
                await websocket.send_text(json.dumps({
                    "type": "interrupted",
                    "session_id": session_id
                }))
    
    except WebSocketDisconnect:
        logger.info(f"🗣️ [Chat Stream] Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"🗣️ [Chat Stream] Error in session {session_id}: {e}")
    finally:
        await kyutai_streaming.disconnect()

@app.post("/tts/stream-test")
async def test_streaming_tts(request: dict, model_manager: ModelManager = Depends(get_model_manager)):
    """
    REST endpoint to test Kyutai streaming
    Returns audio file instead of streaming (for testing)
    """
    text = request.get("text", "")
    voice_reference = request.get("voice_reference")
    model_name = request.get("model_name", "default")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        # Start streaming session
        session_id = await kyutai_streaming.start_stream(voice_reference)
        
        # Create sentence stream from complete text
        text_stream = create_sentence_stream(text)
        
        # Collect all audio chunks
        audio_chunks = []
        async for audio_chunk in kyutai_streaming.stream_text_and_get_audio(text_stream, session_id):
            audio_chunks.append(audio_chunk)
        
        # Concatenate all chunks
        complete_audio = b''.join(audio_chunks)
        
        return StreamingResponse(
            io.BytesIO(complete_audio),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=kyutai_test.wav"}
        )
        
    except Exception as e:
        logger.error(f"🗣️ [Test] Kyutai streaming test failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    finally:
        await kyutai_streaming.disconnect()

async def _synthesize_with_kyutai_streaming(text: str, voice_reference: str = None) -> bytes:
    """
    Non-streaming Kyutai synthesis for compatibility with existing REST API
    """
    try:
        session_id = await kyutai_streaming.start_stream(voice_reference)
        text_stream = create_sentence_stream(text)
        
        audio_chunks = []
        async for audio_chunk in kyutai_streaming.stream_text_and_get_audio(text_stream, session_id):
            audio_chunks.append(audio_chunk)
        
        return b''.join(audio_chunks)
        
    finally:
        await kyutai_streaming.disconnect()

async def synthesize_speech(
    text: str, 
    voice: str = 'af_heart', 
    engine: str = 'kokoro',
    audio_prompt_path: str = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> bytes:
    """
    Enhanced version with Kyutai support
    """
    print(f"🗣️ [TTS Service] Synthesizing with engine: {engine}, voice: {voice}")
    
    cleaned_text = clean_markdown_for_tts(text)
    if not cleaned_text:
        logger.warning("🗣️ [TTS Service] Text became empty after cleaning")
        return b""
    
    if engine.lower() == 'kyutai':
        return await _synthesize_with_kyutai_streaming(cleaned_text, audio_prompt_path)
    elif engine.lower() == 'chatterbox':
        return await _synthesize_with_chatterbox(cleaned_text, audio_prompt_path, exaggeration, cfg)
    else:  # Default to kokoro
        return await _synthesize_with_kokoro(cleaned_text, voice)
# --- Model Manager Dependency ---
# Assume app.state.model_manager is initialized in lifespan

    # No need to do anything here, as the lifespan will handle cleanup
@router.get("/system/gpu_info")
async def get_gpu_info(request: Request):
    """Return GPU count and single GPU mode status."""
    return {
        "gpu_count": check_gpu_count(),
        "single_gpu_mode": getattr(request.app.state, 'single_gpu_mode', False)
    }
@router.post("/models/update-gpu-mode")
async def update_gpu_mode(
    data: dict = Body(...),
):
    """Update the GPU usage mode and save to settings file."""
    try:
        gpu_mode = data.get("gpuUsageMode")
        if gpu_mode not in ["split_services", "unified_model"]:
            raise HTTPException(status_code=400, detail="Invalid GPU usage mode")

        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        # Create a unique temporary file path to prevent conflicts
        temp_path = settings_path.with_suffix(f".{uuid.uuid4()}.tmp")

        os.makedirs(settings_dir, exist_ok=True)

        settings = {}
        # --- Safer Read Logic ---
        if settings_path.exists():
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Handle case where file might be empty
                    if content.strip():
                        settings = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {settings_path}. The file might be corrupted. A new one will be created.")
                settings = {}
            except Exception as e:
                logger.error(f"Error reading settings file {settings_path}: {e}. Proceeding with empty settings.")
                settings = {}
        
        # Update the setting in the dictionary
        settings['gpuUsageMode'] = gpu_mode
        logger.info(f"Attempting to save 'gpuUsageMode': '{gpu_mode}' to {settings_path}")

        # --- Atomic Write Logic ---
        # 1. Write to a temporary file first
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        
        # 2. Atomically replace the original file with the new one
        os.replace(temp_path, settings_path)
        logger.info(f"Successfully saved settings to {settings_path}")
        
        return {"status": "success", "message": "GPU usage mode updated"}
        
    except Exception as e:
        # Clean up temp file on error if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Failed to update GPU mode setting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# Add this endpoint with your other router endpoints
@router.post("/export_character_png")
async def export_character_png(character_data: dict):
    """Export character as PNG with embedded JSON data in tEXt chunk."""
    try:
        character_name = character_data.get('name', 'character')
        avatar_url = character_data.get('avatar')
        
        # Convert to TavernAI format directly in Python
        tavern_data = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": character_data.get('name', ''),
                "description": character_data.get('description', ''),
                "personality": '',  # Not used in GingerGUI
                "scenario": character_data.get('scenario', ''),
                "first_mes": character_data.get('first_message', ''),
                "mes_example": '',
                "creator_notes": 'Exported from GingerGUI',
                "system_prompt": character_data.get('model_instructions', ''),
                "post_history_instructions": '',
                "alternate_greetings": [],
                "tags": [],
                "creator": 'GingerGUI',
                "character_version": '1.0',
                "extensions": {
                    "ginger_gui": {
                        "exported_at": datetime.datetime.now().isoformat(),
                        "original_format": "ginger_gui"
                    }
                }
            }
        }
        
        # Convert example dialogue if it exists
        example_dialogue = character_data.get('example_dialogue', [])
        if example_dialogue and len(example_dialogue) > 0:
            example_lines = []
            for dialogue in example_dialogue:
                if dialogue.get('role') == 'user':
                    example_lines.append(f"{{{{user}}}}: {dialogue.get('content', '')}")
                elif dialogue.get('role') == 'character':
                    example_lines.append(f"{{{{char}}}}: {dialogue.get('content', '')}")
            tavern_data['data']['mes_example'] = '\n'.join(example_lines)
        
        # Convert lore entries to character_book if they exist
        lore_entries = character_data.get('loreEntries', [])
        if lore_entries and len(lore_entries) > 0:
            tavern_data['data']['character_book'] = {
                "name": f"{character_name} Lorebook",
                "description": f"Lorebook for {character_name}",
                "scan_depth": 100,
                "token_budget": 500,
                "recursive_scanning": False,
                "entries": []
            }
            
            for index, entry in enumerate(lore_entries):
                tavern_entry = {
                    "id": index,
                    "keys": entry.get('keywords', []),
                    "content": entry.get('content', ''),
                    "extensions": {},
                    "enabled": True,
                    "insertion_order": index,
                    "case_sensitive": False,
                    "name": f"Entry {index + 1}",
                    "priority": 100,
                    "comment": '',
                    "selective": True,
                    "secondary_keys": [],
                    "constant": False,
                    "position": "before_char"
                }
                tavern_data['data']['character_book']['entries'].append(tavern_entry)
        
        character_json = json.dumps(tavern_data)
        
        # Replace the avatar loading section with this:
        if avatar_url:
            try:
                logger.info(f"Attempting to load avatar: {avatar_url}")
                
                if avatar_url.startswith('http://localhost:8000/static/') or avatar_url.startswith('http://127.0.0.1:8000/static/'):
                    # Local server URL - use direct file path instead of HTTP request
                    filename = avatar_url.split('/static/')[-1]
                    static_path = Path(__file__).parent / "static" / filename
                    logger.info(f"Loading local avatar from direct path: {static_path}")
                    
                    if static_path.exists():
                        img = Image.open(static_path)
                        logger.info(f"Loaded avatar from local static path: {static_path}")
                    else:
                        logger.warning(f"Avatar file not found at: {static_path}")
                        raise FileNotFoundError(f"Avatar not found: {static_path}")
                        
                elif avatar_url.startswith('http'):
                    # External URL - use HTTP request
                    response = requests.get(avatar_url, timeout=10)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    logger.info(f"Loaded avatar from external URL: {avatar_url}")
                    
                elif avatar_url.startswith('/static/'):
                    # Path like /static/filename.png
                    filename = avatar_url.replace('/static/', '')
                    static_path = Path(__file__).parent / "static" / filename
                    logger.info(f"Looking for avatar at: {static_path}")
                    
                    if static_path.exists():
                        img = Image.open(static_path)
                        logger.info(f"Loaded avatar from static path: {static_path}")
                    else:
                        logger.warning(f"Avatar file not found at: {static_path}")
                        raise FileNotFoundError(f"Avatar not found: {static_path}")
                        
                else:
                    # Try as direct filename in static folder
                    static_path = Path(__file__).parent / "static" / avatar_url
                    logger.info(f"Trying direct filename at: {static_path}")
                    
                    if static_path.exists():
                        img = Image.open(static_path)
                        logger.info(f"Loaded avatar from direct path: {static_path}")
                    else:
                        logger.warning(f"Avatar file not found at: {static_path}")
                        raise FileNotFoundError(f"Avatar not found: {static_path}")
                        
            except Exception as e:
                logger.warning(f"Failed to load avatar {avatar_url}: {e}")
                # Create default image
                img = create_default_character_image(character_name)
        else:
            logger.info("No avatar URL provided, creating default image")
            # Create default image
            img = create_default_character_image(character_name)

        
        # Ensure image is RGB and 512x512
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Create PngInfo object to embed metadata
        png_info = PngImagePlugin.PngInfo()
        
        # Add character data to tEXt chunk with 'chara' keyword
        # SillyTavern expects base64-encoded JSON in tEXt format (not zTXt)
        character_json_b64 = base64.b64encode(character_json.encode('utf-8')).decode('ascii')
        png_info.add_text('chara', character_json_b64)

        # Also add some basic metadata
        png_info.add_text('Title', f'Character Card: {character_name}')
        png_info.add_text('Description', f'Character card for {character_name}')
        png_info.add_text('Software', 'GingerGUI')
        
        # Save to bytes buffer
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG', pnginfo=png_info, optimize=True)
        img_buffer.seek(0)
        
        # Clean filename
        safe_filename = re.sub(r'[^\w\-_.]', '_', character_name)
        filename = f"{safe_filename}_character_card.png"
        
        # Return as file download
        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(img_buffer.getvalue()))
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting character PNG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PNG export failed: {str(e)}")

async def generate_text_with_vision(
    model_manager,
    model_name: str,
    prompt: str,
    image_base64: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    stop_sequences: List[str] = None, # Kept for signature consistency
    gpu_id: int = 0,
    echo: bool = False,
    request_purpose: Optional[str] = None
):
    """
    Handles vision generation using the create_chat_completion method.
    This version is tailored specifically for Gemma models, which require
    a single 'user' role and the 'data:' URI for images.
    """
    
    # This function is now self-contained and stable.
    
    try:
        model_instance = model_manager.get_model(model_name, gpu_id)
        if not model_instance:
            raise ValueError(f"Model {model_name} not loaded on GPU {gpu_id}")

        if image_base64:
            # The full, context-rich prompt is passed in from the main /generate function.
            # We don't need to parse it; the entire block of text is what we send.
            # The custom GemmaVisionChatHandler will correctly format this for the model.
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                # FIX: Use the stable data: URI method instead of file://
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            response = model_instance.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                # FIX: Add the crucial stop token to prevent prompt leakage
                stop=["<end_of_turn>"]
            )
            
            if response and response.get('choices'):
                return response['choices'][0]['message']['content']
            else:
                return "Vision processing failed: The model returned no valid response."

        else: # Standard text generation for non-vision calls (fallback)
            response = model_instance(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, repeat_penalty=repetition_penalty, stop=stop_sequences, echo=echo)
            if response and response.get('choices'):
                return response['choices'][0]['text']
            else:
                return "Generation failed - no response"
                
    except Exception as e:
        logger.error(f"Error in vision/text generation: {e}", exc_info=True)
        raise

def save_image_and_get_url(image_data: bytes) -> str:
    """Saves image data to a file and returns its static URL path."""
    # This uses the base_dir and generated_images_dir you've already defined
    generated_images_dir = base_dir / "static" / "generated_images"
    generated_images_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4()}.png"
    save_path = generated_images_dir / filename
    
    with open(save_path, "wb") as f:
        f.write(image_data)
        
    logger.info(f"Image saved to: {save_path}")
    
    # Return the web-accessible URL path
    return f"/static/generated_images/{filename}"

def create_default_character_image(character_name: str) -> Image.Image:
    """Create a default character image with gradient background and name."""
    # Create 512x512 image
    img = Image.new('RGB', (512, 512), color='white')
    
    # Create a simple gradient (requires PIL, but we can do a simple version)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for y in range(512):
        # Simple blue gradient
        blue_value = int(70 + (140 * y / 512))  # 70 to 210
        color = (45, 90, blue_value)  # Blue gradient
        draw.line([(0, y), (512, y)], fill=color)
    
    # Add character name text
    try:
        # Try to use a nice font, fallback to default
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    if font:
        # Get text size for centering
        bbox = draw.textbbox((0, 0), character_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (512 - text_width) // 2
        y = (512 - text_height) // 2
        
        # Draw text with outline for visibility
        outline_color = (0, 0, 0)
        text_color = (255, 255, 255)
        
        # Draw outline
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), character_name, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), character_name, font=font, fill=text_color)
    
    return img
# --- Lifespan Function ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application lifespan startup...")
    global SINGLE_GPU_MODE
    # Read environment variables set by launch.py or batch script
    # Ensure these env vars are correctly set by your launch mechanism
    default_gpu = int(os.environ.get("GPU_ID", 0))
    port = int(os.environ.get("PORT", 8000 if default_gpu == 0 else 8001))
    model_path_env = os.environ.get("MODEL_PATH", "")
    model_name_env = os.environ.get("MODEL_NAME", "")
    # NEW CODE START - Add right here
    gpu_count = check_gpu_count()
    SINGLE_GPU_MODE = (gpu_count == 1)
    logger.info(f"Detected {gpu_count} GPUs. Single GPU mode: {SINGLE_GPU_MODE}")
    
    # Unified settings loader
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    settings = {}
    try:
        logging.info(f"🔍 Looking for settings at: {settings_path}")
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                logging.info(f"🔍 Settings file contents: {settings}")
        else:
            logging.info("🔍 No settings file found.")
    except Exception as e:
        logging.warning(f"🔍 Could not read settings file: {e}")

    # GPU usage mode
    user_gpu_mode = settings.get('gpuUsageMode')
    if user_gpu_mode in ['split_services', 'unified_model']:
        gpu_usage_mode = user_gpu_mode
        logging.info(f"🔍 Using user GPU usage mode preference: {gpu_usage_mode}")
    else:
        logging.info(f"🔍 Invalid or missing GPU mode, using default: {gpu_usage_mode}")

    # SD model directory
    sd_model_dir = settings.get('sdModelDirectory')
    if sd_model_dir:
        app.state.sd_model_directory = sd_model_dir
        logger.info(f"SD model directory set to: {sd_model_dir}")
    else:
        app.state.sd_model_directory = None
        logger.warning("No SD model directory set in settings, using default.")
    # Also check if you changed this line:
    logging.info(f"🔍 About to create ModelManager with gpu_usage_mode: {gpu_usage_mode}")
    
    # Store in app state so it's accessible to endpoints and model manager
    app.state.single_gpu_mode = SINGLE_GPU_MODE
    app.state.gpu_usage_mode = gpu_usage_mode
    app.state.gpu_count = gpu_count
    # NEW CODE END

    logger.info(f"Lifespan: Running on Port {port}, Default GPU {default_gpu}")

    # Initialize ModelManager and store in app state
    # Ensure ModelManager() doesn't raise exceptions during init
    try:
        app.state.model_manager = ModelManager(gpu_usage_mode=gpu_usage_mode)
        app.state.default_gpu = default_gpu
        app.state.port = port
        logger.info(f"Server starting on port {port} with default GPU {default_gpu}")
    except Exception as init_err:
        logger.error(f"FATAL: Failed to initialize ModelManager: {init_err}", exc_info=True)
        # Optionally re-raise or handle differently to prevent app start
        raise init_err

    # Auto-load logic (only if env vars were set - likely not needed now)
    if model_path_env and model_name_env:
        logger.info(f"Attempting auto-load: '{model_name_env}' from {model_path_env} on GPU {default_gpu}")
        try:
            await app.state.model_manager.load_model(
                model_name_env, gpu_id=default_gpu, model_path=model_path_env
            )
            logger.info(f"Auto-load successful: '{model_name_env}' on GPU {default_gpu}.")
        except Exception as e:
            logger.error(f"Error auto-loading model '{model_name_env}' on GPU {default_gpu}: {e}", exc_info=True)
    else:
        logger.info("Skipping auto-load: MODEL_PATH or MODEL_NAME env vars not set.")

    # Initialize SD Manager (add this after your ModelManager initialization)
    try:
        app.state.sd_manager = SDManager()
        logger.info("SD Manager initialized")
    except Exception as sd_err:
        logger.error(f"Failed to initialize SD Manager: {sd_err}")
        app.state.sd_manager = None
    
    # Initialize RAG system
    try:
        rag_available = rag_utils.initialize_rag_system()
        app.state.rag_available = rag_available
        logger.info(f"RAG system initialization {'successful' if rag_available else 'skipped or failed'}")
    except Exception as rag_error:
        logger.error(f"Error initializing RAG system: {rag_error}", exc_info=True)
        app.state.rag_available = False
        # Initialize active user profile
    try:
        from . import user_utils
        active_profile_id = user_utils.get_active_profile_id()
        active_profile = user_utils.load_profile(active_profile_id) if active_profile_id else None
        
        app.state.active_profile_id = active_profile_id
        app.state.active_profile = active_profile
        
        logger.info(f"Active user profile: {active_profile_id or 'None'}")
    except Exception as e:
        logger.error(f"Error initializing user profile: {e}")
        app.state.active_profile_id = None
        app.state.active_profile = None

    yield # Application runs here

    # Shutdown logic
    logger.info(f"Application lifespan shutdown on port {port}...")
    if hasattr(app.state, 'model_manager'):
        await app.state.model_manager.unload_all_models()
        logger.info("Models unloaded.")
    else:
        logger.warning("ModelManager not found in app state during shutdown.")
    logger.info("Server shutdown complete.")

app.router.lifespan_context = lifespan # Register the lifespan context with the app

@app.router.get("/user/profile/current")
async def get_current_profile(request: Request):
    """Get the current active user profile."""
    profile_id = getattr(request.app.state, "active_profile_id", None)
    profile = getattr(request.app.state, "active_profile", None)
    
    return {
        "profile_id": profile_id,
        "profile": profile
    }

@app.router.post("/user/profile/set-active/{profile_id}")
async def set_active_profile(profile_id: str, request: Request):
    """Set the active user profile in settings, even if profile doesn't exist yet."""
    try:
        from . import user_utils
        # Just save the ID to settings without checking if profile exists
        success = user_utils.save_active_profile_id(profile_id)
        if success:
            # Also update app state
            request.app.state.active_profile_id = profile_id
            logger.info(f"Active profile ID set to: {profile_id}")
            return {"status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save profile ID")
    except Exception as e:
        logger.error(f"Error setting active profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Routes ---
@router.post("/models/load-for-purpose/{purpose}")
async def load_model_for_purpose_endpoint(
    purpose: str,
    request: Request,
    data: dict = Body(...),  # Expects {"model_name": "...", "gpu_id": 0, "context_length": 4096}
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Load a model for a specific testing purpose."""
    try:
        model_name = data.get("model_name")
        gpu_id = data.get("gpu_id", 0)
        context_length = data.get("context_length", 4096)
        
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        
        logger.info(f"Loading {model_name} for purpose '{purpose}' on GPU {gpu_id}")
        
        await model_manager.load_model_for_purpose(
            purpose=purpose,
            model_name=model_name, 
            gpu_id=gpu_id,
            context_length=context_length
        )
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded as {purpose} on GPU {gpu_id}"
        }
        
    except ValueError as e:
        logger.error(f"Invalid purpose for model loading: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading model for purpose: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/unload-purpose/{purpose}")
async def unload_model_purpose_endpoint(
    purpose: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Unload the model serving a specific testing purpose."""
    try:
        await model_manager.unload_model_purpose(purpose)
        return {
            "status": "success", 
            "message": f"Unloaded {purpose} model"
        }
    except Exception as e:
        logger.error(f"Error unloading model purpose: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# NEW endpoint to get local data only. Both servers will have this.
@router.get("/models/by-purpose/local")
async def get_local_models_by_purpose_endpoint(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get currently loaded models organized by their testing purpose from the local instance."""
    try:
        purposes = model_manager.get_models_by_purpose()
        return {
            "status": "success",
            "purposes": purposes
        }
    except Exception as e:
        logger.error(f"Error getting local models by purpose: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# MODIFIED existing endpoint to be smarter
@router.get("/models/by-purpose")
async def get_models_by_purpose_endpoint(
    request: Request, # Need request to access app state
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get currently loaded models organized by their testing purpose, merging from both servers if applicable."""
    try:
        # Get the local purposes from this server instance
        local_purposes = model_manager.get_models_by_purpose()

        # If this is the primary server (port 8000) and we are in dual GPU mode, fetch from secondary
        is_primary_server = hasattr(request.app.state, 'port') and request.app.state.port == 8000
        is_dual_gpu_mode = not getattr(request.app.state, 'single_gpu_mode', False)

        if is_primary_server and is_dual_gpu_mode:
            logger.info("Primary server fetching purposes from secondary server...")
            try:
                async with httpx.AsyncClient() as client:
                    # Port 8001 is the secondary server
                    resp = await client.get("http://localhost:8001/models/by-purpose/local", timeout=5.0)
                    if resp.status_code == 200:
                        secondary_data = resp.json()
                        secondary_purposes = secondary_data.get("purposes", {})

                        # Merge the secondary purposes into the local ones.
                        # Any non-null purpose from the secondary server (for GPU 1 models)
                        # should override the primary's stale information.
                        for purpose, info in secondary_purposes.items():
                            if info is not None:
                                local_purposes[purpose] = info
                        logger.info("Successfully merged purposes from secondary server.")
                    else:
                        logger.warning(f"Could not fetch purposes from secondary server. Status: {resp.status_code}")
            except Exception as e:
                logger.error(f"Error fetching purposes from secondary server: {e}")

        return {
            "status": "success",
            "purposes": local_purposes
        }
    except Exception as e:
        logger.error(f"Error getting models by purpose: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# Add this new endpoint anywhere in your main.py (before the app.include_router lines)
@router.post("/system/shutdown")
async def shutdown_system(
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Signal shutdown by creating a file."""
    try:
        logger.info("🔴 Shutdown request received")
        
        # Unload models first
        logger.info("🔄 Unloading all models...")
        await model_manager.unload_all_models()
        logger.info("✅ All models unloaded")
        
        # Create shutdown signal file
        with open("SHUTDOWN_SIGNAL", "w") as f:
            f.write("shutdown_requested")
        
        logger.info("🔴 Shutdown signal file created")
        
        return {
            "status": "success", 
            "message": "Shutdown initiated. Servers will stop shortly."
        }
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {str(e)}")
@router.get("/")
async def read_root(request: Request):
    default_gpu = request.app.state.default_gpu if hasattr(request.app.state, 'default_gpu') else 'N/A'
    port = request.app.state.port if hasattr(request.app.state, 'port') else 'N/A'
    return {"status": "ok", "message": "LLM Frontend API is running", "server_info": {"port": port, "default_gpu": default_gpu}}

@router.get("/models")
async def list_available_models_endpoint(model_manager: ModelManager = Depends(get_model_manager)):
    return model_manager.list_available_models()

@router.get("/models/loaded")
async def list_loaded_models_endpoint(model_manager: ModelManager = Depends(get_model_manager)):
    return model_manager.get_loaded_models()

@router.post("/upload_avatar", status_code=201)
async def upload_avatar_image(request: Request, file: UploadFile = File(...)):
    # Define the global static directory path - ensure this matches your app.mount location
    # Make sure you're using only ONE static_dir definition in your entire codebase
    static_dir = Path(__file__).parent / "static"  # Use the same path as in your app.mount()
    
    allowed_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {allowed_extensions}")
    
    try:
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        save_path = static_dir / unique_filename
        logger.info(f"Attempting to save avatar to: {save_path}")
        
        # Ensure the directory exists
        static_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        with save_path.open("wb") as buffer:
            while content := await file.read(1024 * 1024): 
                buffer.write(content)
        
        logger.info(f"Avatar successfully saved: {save_path}")
        
        # Create a full URL including the host - this is what the frontend needs
        base_url = str(request.base_url).rstrip("/")
        full_file_url = f"{base_url}/static/{unique_filename}"
        
        logger.info(f"Returning full URL: {full_file_url}")
        return {"status": "success", "file_url": full_file_url}
    
    except Exception as e:
        logger.error(f"Error uploading avatar: {e}", exc_info=True)
        if 'save_path' in locals() and save_path.exists():
            try: 
                save_path.unlink()
            except OSError: 
                logger.error(f"Failed to remove partially uploaded file: {save_path}")
        raise HTTPException(status_code=500, detail=f"Failed to upload avatar: {str(e)}")
    
    finally: 
        await file.close()

@router.get("/rag/status")
async def rag_status(request: Request):
    """Check if RAG functionality is available."""
    try:
        rag_available = getattr(request.app.state, 'rag_available', False)
        
        if not rag_available:
            # Try to initialize
            rag_available = rag_utils.is_rag_available()
            request.app.state.rag_available = rag_available
        
        return {
            "available": rag_available,
            "message": "RAG functionality is available" if rag_available else "RAG functionality not available, missing dependencies"
        }
    except Exception as e:
        logger.error(f"Error checking RAG status: {e}", exc_info=True)
        return {
            "available": False,
            "message": f"Error checking RAG status: {str(e)}"
        }

@router.post("/character/analyze-readiness")
async def analyze_character_readiness_endpoint(
    request: Request,
    data: dict = Body(...),  # Expects {"messages": [...]}
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Analyze conversation messages for character auto-generation readiness."""
    try:
        messages = data.get("messages", [])
        lookback_count = data.get("lookback_count", 25)
        
        if not messages:
            return {
                "status": "success", 
                "readiness_score": 0, 
                "detected_elements": [],
                "message": "No messages to analyze"
            }
        
        logger.info(f"🎯 Analyzing character readiness for {len(messages)} messages")
        
        # Analyze character readiness using embeddings
        analysis_result = character_intelligence.analyze_character_readiness(
            messages=messages,
            lookback_count=lookback_count
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ Error in character readiness analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# Add this new endpoint for testing web search
@router.post("/web-search/test")
async def test_web_search(data: dict = Body(...)):
    """Test endpoint for web search functionality."""
    try:
        query = data.get("query", "")
        max_results = data.get("max_results", 3)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
        
        results = await perform_web_search(query, max_results)
        
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Web search test error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/character/generate-from-conversation")
async def generate_character_from_conversation_endpoint(
    request: Request,
    data: dict = Body(...),  # Expects {"messages": [...], "analysis": {...}}
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Generate a character JSON from conversation using LLM on memory GPU."""
    try:
        messages = data.get("messages", [])
        analysis = data.get("analysis", {})
        model_name = data.get("model_name")  # Get from frontend if provided
        gpu_id = data.get("gpu_id")  # Optional override
        
        # Determine GPU. Default to 0 (primary chat GPU) for character creation.
        if gpu_id is None:
            gpu_id = 0
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        logger.info(f"🎨 Generating character from conversation using GPU {gpu_id}")
        
        # Generate character JSON using LLM
        generation_result = await character_intelligence.generate_character_json(
            model_manager=model_manager,
            messages=messages,
            character_analysis=analysis,
            model_name=model_name,
            gpu_id=gpu_id,
            single_gpu_mode=getattr(request.app.state, 'single_gpu_mode', False)
        )
        
        return generation_result
        
    except Exception as e:
        logger.error(f"❌ Error generating character from conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/character/refine-generated")
async def refine_generated_character_endpoint(
    request: Request,
    data: dict = Body(...),  # Expects {"character_json": {...}, "feedback": "...", "original_messages": [...]}
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Refine a generated character based on user feedback."""
    try:
        character_json = data.get("character_json", {})
        feedback = data.get("feedback", "")
        original_messages = data.get("original_messages", [])
        gpu_id = data.get("gpu_id")
        
        # Determine GPU using same logic as memory system
        if gpu_id is None:
            gpu_id = 0
        
        if not character_json or not feedback:
            raise HTTPException(status_code=400, detail="Character JSON and feedback required")
        
        logger.info(f"🔄 Refining character '{character_json.get('name', 'Unknown')}' with feedback: {feedback[:50]}...")
        
        # Build refinement prompt
        refinement_prompt = f"""System:
You are a character refinement specialist. Your task is to take an existing character JSON and apply user feedback to improve it while maintaining the exact JSON structure.

**CRITICAL RULES:**
1. You MUST output ONLY valid JSON and nothing else
2. Do NOT include any commentary, explanations, or text outside the JSON
3. Keep ALL existing good elements that weren't criticized in the feedback
4. Apply the user's feedback thoughtfully and accurately
5. Maintain the exact same JSON field structure

**REQUIRED JSON STRUCTURE:**
{{
  "name": "string",
  "description": "string", 
  "model_instructions": "string",
  "scenario": "string",
  "first_message": "string",
  "example_dialogue": [
    {{"role": "user", "content": "string"}},
    {{"role": "character", "content": "string"}}
  ],
  "loreEntries": [
    {{"content": "string", "keywords": ["string", "string"]}}
  ]
}}

**CURRENT CHARACTER JSON:**
{json.dumps(character_json, indent=2)}

**USER FEEDBACK TO APPLY:**
{feedback}

**TASK:**
Apply the user's feedback to improve the character while keeping all good elements unchanged. Output ONLY the refined JSON with no additional text.

**REFINED CHARACTER JSON:**
"""

        # Generate refined character using your existing inference module
        from . import inference
        response = await inference.generate_text(
            model_manager=model_manager,
            model_name=data.get("model_name"),
            prompt=refinement_prompt,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            stop_sequences=["</character>", "---"],
            gpu_id=gpu_id
        )
        
        # Extract refined JSON
        refined_json = character_intelligence.extract_json_from_response(response)
        
        if refined_json:
            logger.info(f"✅ Refined character: {refined_json.get('name', 'Unknown')}")
            return {"status": "success", "character_json": refined_json}
        else:
            return {"status": "error", "error": "Could not extract valid JSON from refinement response"}
            
    except Exception as e:
        logger.error(f"❌ Error refining character: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/document/query")
async def query_documents_endpoint(
    request: Request,
    query: DocumentQuery
):
    """Query documents for relevant chunks to use in LLM context."""
    # Check if RAG is available
    rag_available = getattr(request.app.state, 'rag_available', False)
    if not rag_available:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "error": "RAG functionality not available, check server logs for details"}
        )
    
    try:
        # Query the documents
        result = rag_utils.query_documents(
            question=query.query,
            doc_ids=query.doc_ids,
            top_k=query.top_k,
            threshold=query.threshold
        )
        
        return result
    except Exception as e:
        logger.error(f"Error querying documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document query error: {str(e)}")
 
@router.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...), 
    engine: str = Query("whisper")  # Added engine parameter with "whisper" default
):
    try:
        # Generate a unique filename
        filename = f"recording_{uuid.uuid4()}.webm"
        save_path = os.path.join("temp_audio", filename)
        os.makedirs("temp_audio", exist_ok=True)

        # Save uploaded file
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Transcribe with the specified engine
        transcript = await transcribe_audio(save_path, engine)  # Pass engine parameter here

        # Clean up
        os.remove(save_path)

        return { "transcript": transcript }

    except Exception as e:
        print("🔥 Transcription error:", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
    
@router.get("/stt/available-engines")
async def get_available_stt_engines():
    """Return a list of available STT engines."""
    from .stt_service import list_available_engines, is_engine_available
    
    try:
        # Add explicit checks for each engine
        whisper_available = is_engine_available("whisper")
        parakeet_available = is_engine_available("parakeet")
        
        logger.info(f"Checking available engines - Whisper: {whisper_available}, Parakeet: {parakeet_available}")
        
        available_engines = list_available_engines()
        logger.info(f"Available STT engines: {available_engines}")
        
        return {
            "available_engines": available_engines
        }
    except Exception as e:
        logger.error(f"Error checking available engines: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/stt/install-engine")
async def install_stt_engine(engine: str = Query(...)):
    """Install requested STT engine."""
    logger.info(f"Received request to install engine: {engine}")
    
    if engine == "parakeet":
        try:
            from .stt_service import load_parakeet_model
            
            logger.info("Starting Parakeet installation...")
            # This will trigger the automatic installation in load_parakeet_model
            model = load_parakeet_model()
            
            if model:
                logger.info("Parakeet installation successful!")
                return {"status": "success", "message": "Parakeet installed successfully"}
            else:
                logger.error("Parakeet installation failed - model is None")
                return JSONResponse(
                    status_code=500, 
                    content={"status": "error", "message": "Failed to install Parakeet"}
                )
        except Exception as e:
            logger.error(f"Error installing Parakeet: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)}
            )
    else:
        logger.warning(f"Unknown engine requested for installation: {engine}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Unknown engine: {engine}"}
        )
@router.get("/gpu/count")
def check_gpu_count():
    """Check how many GPUs are available using pynvml to avoid initializing a CUDA context."""
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return gpu_count
    except (pynvml.NVMLError, NameError):
        # Fallback or log warning if pynvml is not available
        # For this fix, we assume it is installed. If not: pip install pynvml
        return 0

@router.get("/sd-local/list-models")
async def list_local_sd_models(request: Request):
    """List available local Stable Diffusion models from the configured directory."""
    sd_model_dir = getattr(request.app.state, 'sd_model_directory', None)
    if not sd_model_dir:
        return {"status": "error", "message": "SD model directory not configured.", "models": []}

    model_path = Path(sd_model_dir)
    if not model_path.is_dir():
        return {"status": "error", "message": f"Configured directory not found: {model_path}", "models": []}

    # Scan for .safetensors and .ckpt files
    allowed_extensions = {".safetensors", ".ckpt", ".gguf"}
    models = sorted([f.name for f in model_path.iterdir() if f.suffix.lower() in allowed_extensions])

    return {"status": "success", "models": models}


@router.post("/sd-local/load-model")
async def sd_local_load_model(data: dict, request: Request):
    """Load a local SD model by its filename."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    model_filename = data.get("model_filename")
    if not model_filename:
        raise HTTPException(status_code=400, detail="model_filename required")

    sd_model_dir = getattr(request.app.state, 'sd_model_directory', None)
    if not sd_model_dir:
        raise HTTPException(status_code=500, detail="SD model directory not configured on backend.")

    full_model_path = str(Path(sd_model_dir) / model_filename)
    
    success = sd_manager.load_model(full_model_path)
    if success:
        return {"status": "success", "message": f"Model loaded: {full_model_path}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model from path: {full_model_path}")
# --- MODIFIED load_model_endpoint ---
@router.post("/models/load/{model_name}")
async def load_model_endpoint(
    model_name: str,
    request: Request,
    gpu_id: Optional[int] = None,
    # --- FIXED: Changed default from 4096 to None ---
    context_length: Optional[int] = 4096,  # Default context length, can be overridden by query param
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Load a specific model on a specific GPU."""
    logger.info(f"Received request to load model: {model_name}, GPU: {gpu_id}, Context: {context_length}")
    try:
        # Determine target GPU using app state from the specific instance
        target_gpu_id = gpu_id if gpu_id is not None else request.app.state.default_gpu
        logger.info(f"Targeting GPU: {target_gpu_id}")

        # Pass context_length received from query param as n_ctx
        # model_manager.load_model will handle None and use its internal default if needed
        await model_manager.load_model(
             model_name,
             gpu_id=target_gpu_id,
             n_ctx=context_length # Pass query param value here
        )
        # Use the received context_length (or 'default') in the response message
        ctx_msg = context_length if context_length is not None else 'default'
        return {
            "status": "success",
            "message": f"Model {model_name} load initiated on GPU {target_gpu_id} with context {ctx_msg}"
        }
    except FileNotFoundError as e:
        logger.error(f"Model file not found for {model_name}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error loading model {model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/models/unload/{model_name}")
async def unload_model_endpoint(
    model_name: str,
    request: Request,
    gpu_id: Optional[int] = None, # Make sure this query parameter is accepted
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Unload a specific model instance from a specific GPU."""
    logger.info(f"Received request to unload model: {model_name}, requested GPU: {gpu_id}")
    try:
        # Determine the target GPU ID for the unload operation
        # Use the provided gpu_id if available, otherwise default to this server instance's default GPU
        # This ensures we try to unload from the correct instance managing that GPU
        target_gpu_id = gpu_id if gpu_id is not None else request.app.state.default_gpu
        logger.info(f"Attempting to unload model '{model_name}' from GPU {target_gpu_id}")

        # --- FIXED CALL ---
        # Pass both model_name and target_gpu_id to the manager's unload method
        await model_manager.unload_model(model_name=model_name, gpu_id=target_gpu_id)
        # --- END FIXED CALL ---

        return {"status": "success", "message": f"Model {model_name} unload initiated from GPU {target_gpu_id}."}
    except ValueError as e: # Catch if model wasn't loaded on that GPU
         logger.warning(f"Failed to unload model {model_name} from GPU {target_gpu_id}: {e}")
         # Return 404 if model/GPU combo not found by the manager
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error unloading model {model_name} from GPU {target_gpu_id}: {e}", exc_info=True)
        # Return 500 for other unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
@app.post("/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text")
    voice = data.get("voice", "af_heart")  # Default Kokoro voice
    engine = data.get("engine", "kokoro")  # Default to Kokoro
    audio_prompt_path = data.get("audio_prompt_path")  # For Chatterbox voice cloning
    
    # Chatterbox-specific parameters
    exaggeration = data.get("exaggeration", 0.5)
    cfg = data.get("cfg", 0.5)

    if not text:
        return JSONResponse(content={"detail": "No text provided"}, status_code=400)

    try:
        # Pass all parameters to synthesize_speech
        audio_buffer = await synthesize_speech(
            text, 
            voice=voice,
            engine=engine,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )
        return StreamingResponse(io.BytesIO(audio_buffer), media_type="audio/wav")
    except Exception as e:
        print("🔥 TTS error:", str(e))
        return JSONResponse(content={"detail": f"TTS failed: {str(e)}"}, status_code=500)
# Add this new endpoint for uploading voice reference files
@app.post("/tts/upload-voice")
async def upload_voice_reference(request: Request, file: UploadFile = File(...)):
    """Upload a reference audio file for Chatterbox voice cloning."""
    try:
        # Define the voice references directory
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate file type
        allowed_extensions = {".wav", ".mp3", ".flac", ".m4a"}
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}")
        
        # Generate unique filename
        unique_filename = f"voice_ref_{uuid.uuid4()}{file_extension}"
        save_path = voices_dir / unique_filename
        
        # Save the file
        with save_path.open("wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)
        
        logger.info(f"Voice reference uploaded: {save_path}")
        
        return {
            "status": "success", 
            "voice_id": unique_filename,
            "file_path": str(save_path),
            "message": f"Voice reference '{file.filename}' uploaded successfully"
        }
    
    except Exception as e:
        logger.error(f"Error uploading voice reference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload voice reference: {str(e)}")
# Add this endpoint to list available voice references
@app.get("/tts/voices")
async def list_available_voices():
    """List available voices for both engines."""
    try:
        # Complete Kokoro voices list
        kokoro_voices = [
            # 🇺🇸 American English
            {'id': 'af_heart', 'name': 'Am. English Female (Heart)', 'engine': 'kokoro'},
            {'id': 'af_alloy', 'name': 'Am. English Female (Alloy)', 'engine': 'kokoro'},
            {'id': 'af_aoede', 'name': 'Am. English Female (Aoede)', 'engine': 'kokoro'},
            {'id': 'af_bella', 'name': 'Am. English Female (Bella)', 'engine': 'kokoro'},
            {'id': 'af_jessica', 'name': 'Am. English Female (Jessica)', 'engine': 'kokoro'},
            {'id': 'af_kore', 'name': 'Am. English Female (Kore)', 'engine': 'kokoro'},
            {'id': 'af_nicole', 'name': 'Am. English Female (Nicole)', 'engine': 'kokoro'},
            {'id': 'af_nova', 'name': 'Am. English Female (Nova)', 'engine': 'kokoro'},
            {'id': 'af_river', 'name': 'Am. English Female (River)', 'engine': 'kokoro'},
            {'id': 'af_sarah', 'name': 'Am. English Female (Sarah)', 'engine': 'kokoro'},
            {'id': 'af_sky', 'name': 'Am. English Female (Sky)', 'engine': 'kokoro'},
            
            {'id': 'am_adam', 'name': 'Am. English Male (Adam)', 'engine': 'kokoro'},
            {'id': 'am_echo', 'name': 'Am. English Male (Echo)', 'engine': 'kokoro'},
            {'id': 'am_eric', 'name': 'Am. English Male (Eric)', 'engine': 'kokoro'},
            {'id': 'am_fenrir', 'name': 'Am. English Male (Fenrir)', 'engine': 'kokoro'},
            {'id': 'am_liam', 'name': 'Am. English Male (Liam)', 'engine': 'kokoro'},
            {'id': 'am_michael', 'name': 'Am. English Male (Michael)', 'engine': 'kokoro'},
            {'id': 'am_onyx', 'name': 'Am. English Male (Onyx)', 'engine': 'kokoro'},
            {'id': 'am_puck', 'name': 'Am. English Male (Puck)', 'engine': 'kokoro'},
            {'id': 'am_santa', 'name': 'Am. English Male (Santa)', 'engine': 'kokoro'},
            
            # 🇬🇧 British English
            {'id': 'bf_alice', 'name': 'Br. English Female (Alice)', 'engine': 'kokoro'},
            {'id': 'bf_emma', 'name': 'Br. English Female (Emma)', 'engine': 'kokoro'},
            {'id': 'bf_isabella', 'name': 'Br. English Female (Isabella)', 'engine': 'kokoro'},
            {'id': 'bf_lily', 'name': 'Br. English Female (Lily)', 'engine': 'kokoro'},
            
            {'id': 'bm_daniel', 'name': 'Br. English Male (Daniel)', 'engine': 'kokoro'},
            {'id': 'bm_fable', 'name': 'Br. English Male (Fable)', 'engine': 'kokoro'},
            {'id': 'bm_george', 'name': 'Br. English Male (George)', 'engine': 'kokoro'},
            {'id': 'bm_lewis', 'name': 'Br. English Male (Lewis)', 'engine': 'kokoro'},
            
            # 🇯🇵 Japanese
            {'id': 'jf_alpha', 'name': 'Japanese Female (Alpha)', 'engine': 'kokoro'},
            {'id': 'jf_gongitsune', 'name': 'Japanese Female (Gongitsune)', 'engine': 'kokoro'},
            {'id': 'jf_nezumi', 'name': 'Japanese Female (Nezumi)', 'engine': 'kokoro'},
            {'id': 'jf_tebukuro', 'name': 'Japanese Female (Tebukuro)', 'engine': 'kokoro'},
            
            {'id': 'jm_kumo', 'name': 'Japanese Male (Kumo)', 'engine': 'kokoro'},
            
            # 🇨🇳 Mandarin Chinese
            {'id': 'zf_xiaobei', 'name': 'Mandarin Female (Xiaobei)', 'engine': 'kokoro'},
            {'id': 'zf_xiaoni', 'name': 'Mandarin Female (Xiaoni)', 'engine': 'kokoro'},
            {'id': 'zf_xiaoxiao', 'name': 'Mandarin Female (Xiaoxiao)', 'engine': 'kokoro'},
            {'id': 'zf_xiaoyi', 'name': 'Mandarin Female (Xiaoyi)', 'engine': 'kokoro'},
            
            {'id': 'zm_yunjian', 'name': 'Mandarin Male (Yunjian)', 'engine': 'kokoro'},
            {'id': 'zm_yunxi', 'name': 'Mandarin Male (Yunxi)', 'engine': 'kokoro'},
            {'id': 'zm_yunxia', 'name': 'Mandarin Male (Yunxia)', 'engine': 'kokoro'},
            {'id': 'zm_yunyang', 'name': 'Mandarin Male (Yunyang)', 'engine': 'kokoro'},
            
            # 🇪🇸 Spanish
            {'id': 'ef_dora', 'name': 'Spanish Female (Dora)', 'engine': 'kokoro'},
            {'id': 'em_alex', 'name': 'Spanish Male (Alex)', 'engine': 'kokoro'},
            {'id': 'em_santa', 'name': 'Spanish Male (Santa)', 'engine': 'kokoro'},
            
            # 🇫🇷 French
            {'id': 'ff_siwis', 'name': 'French Female (Siwis)', 'engine': 'kokoro'},
            
            # 🇮🇳 Hindi
            {'id': 'hf_alpha', 'name': 'Hindi Female (Alpha)', 'engine': 'kokoro'},
            {'id': 'hf_beta', 'name': 'Hindi Female (Beta)', 'engine': 'kokoro'},
            {'id': 'hm_omega', 'name': 'Hindi Male (Omega)', 'engine': 'kokoro'},
            {'id': 'hm_psi', 'name': 'Hindi Male (Psi)', 'engine': 'kokoro'},
            
            # 🇮🇹 Italian
            {'id': 'if_sara', 'name': 'Italian Female (Sara)', 'engine': 'kokoro'},
            {'id': 'im_nicola', 'name': 'Italian Male (Nicola)', 'engine': 'kokoro'},
            
            # 🇧🇷 Brazilian Portuguese
            {'id': 'pf_dora', 'name': 'Br. Portuguese Female (Dora)', 'engine': 'kokoro'},
            {'id': 'pm_alex', 'name': 'Br. Portuguese Male (Alex)', 'engine': 'kokoro'},
            {'id': 'pm_santa', 'name': 'Br. Portuguese Male (Santa)', 'engine': 'kokoro'},
        ]
        
        # Chatterbox voice references (uploaded files)
        chatterbox_voices = []
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        if voices_dir.exists():
            for voice_file in voices_dir.glob("voice_ref_*"):
                chatterbox_voices.append({
                    'id': voice_file.name,
                    'name': f"Custom Voice ({voice_file.stem.replace('voice_ref_', '')})",
                    'engine': 'chatterbox',
                    'file_path': str(voice_file)
                })
        
        return {
            "kokoro_voices": kokoro_voices,
            "chatterbox_voices": chatterbox_voices,
            "available_engines": ["kokoro", "chatterbox"]
        }
    
    except Exception as e:
        logger.error(f"Error listing voices: {e}", exc_info=True)
        return {
            "kokoro_voices": [],
            "chatterbox_voices": [],
            "available_engines": ["kokoro"],
            "error": str(e)
        }
# Define the system prompt directly in the backend
SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are an advanced AI assistant designed to provide personalized, contextually-aware interactions. Your primary objectives are:

1. **CONTEXTUAL ADAPTATION**: Carefully review any USER CONTEXT, RELEVANT USER INFORMATION, memory data, or profile details provided later in this conversation. Use this information to tailor your responses to the user's specific needs, preferences, communication style, and background.

2. **CHARACTER PORTRAYAL** (when applicable): If character information is provided, portray that character authentically while maintaining awareness of user context. Write in internet RP style, italicize actions and narration, use markdown. Be proactive, creative, and drive plot/conversation forward with immersive storytelling.

3. **ALIGNMENT PRINCIPLES**: 
   - Reflect the user's cognitive and emotional patterns when beneficial
   - Adapt your communication style to match user preferences 
   - Build on previous context and established rapport
   - Provide responses that genuinely serve the user's goals and well-being

4. **RESPONSE QUALITY**: Maintain engaging, thoughtful responses that demonstrate understanding of both the immediate query and broader user context. Avoid repetition, emphasize relevant details, and respond dynamically to user input.

When engaging in roleplay: Maintain realistic immersion, end with action/dialogue, allow explicit language if contextually appropriate, prioritize character consistency, and use descriptive narrative covering surroundings, actions, thoughts, appearances, emotions, and sensory details.

Remember: Any user information provided is valuable context for creating more helpful, personalized interactions. Always integrate this context thoughtfully into your responses.
"""
END_MARKER = "<|DONE|>"
# ─── module-level worker ───
async def detect_and_store(
    final_raw: str,
    orig_prompt: str,
    user_profile_attempt: dict = None,
):
    # Determine which port to use for memory service based on GPU mode
    memory_port = 8000 if SINGLE_GPU_MODE else 8001
    logger = logging.getLogger(__name__) # Uses the logger configured by FastAPI/Uvicorn

    # Try to get user ID from multiple sources
    user_id = None

    # 1. Try from provided user profile
    if user_profile_attempt and isinstance(user_profile_attempt, dict):
        user_id = user_profile_attempt.get("id")
        # The [DBG store] log for user_profile_attempt can be very verbose if it contains full profile.
        # Consider logging just the presence or absence of the ID, or a truncated version if needed.
        # For now, keeping your original log:
        logger.info(f"[DBG store] user_profile_attempt content for user_id extraction: {user_profile_attempt!r}")

    # 2. Direct environment check (for when request isn't available in background tasks)
    if not user_id:
        try:
            from . import user_utils # Make sure this path is correct for your project structure
            user_id = user_utils.get_active_profile_id()
            logger.info(f"🧠 Used fallback to load profile ID from settings: {user_id}")
        except ImportError:
            logger.error("🧠 Failed to import user_utils. Cannot use fallback for profile ID.")
        except Exception as e:
            logger.error(f"🧠 Error loading profile ID using user_utils fallback: {e}")

    if not (final_raw and orig_prompt and user_id):
        # More detailed log for why it's skipping
        logger.warning(f"🧠 Memory detection skipped – missing one or more: final_raw ({bool(final_raw)}), orig_prompt ({bool(orig_prompt)}), user_id ({bool(user_id)})")
        return

    # 3) Call detect_intent with the user ID
    try:
        async with httpx.AsyncClient() as client:
            detect_resp = await client.post(
                f"http://localhost:{memory_port}/memory/detect_intent",
                json={
                    "original_prompt": orig_prompt,
                    "response_text": final_raw,
                    "user_name": user_id, # memory_routes.py uses user_name or user_id
                    "user_id": user_id,   # Sending both for robustness
                },
                timeout=120.0,
            )

            # Parse the response safely
            det_json = None # Initialize
            try:
                det_json = detect_resp.json()
                logger.info(f"🧠 Successfully parsed JSON response from /memory/detect_intent")
            except json.JSONDecodeError as parse_error: # More specific exception
                logger.error(f"🧠 JSONDecodeError parsing /memory/detect_intent response: {parse_error}. Response text: '{detect_resp.text[:200]}...'")
                # Attempt to load from text if direct .json() fails and text might be valid JSON
                if detect_resp.text:
                    try:
                        det_json = json.loads(detect_resp.text)
                        logger.info(f"🧠 Successfully parsed /memory/detect_intent response from text fallback.")
                    except json.JSONDecodeError:
                        logger.error(f"🧠 Failed to parse /memory/detect_intent response from text fallback as well.")
                        return # Critical error, cannot proceed
                else:
                    logger.error(f"🧠 /memory/detect_intent response text is empty, cannot parse.")
                    return
            except Exception as e: # Catch other potential errors from .json() or .text
                logger.error(f"🧠 Unexpected error processing /memory/detect_intent response: {e}. Response text: '{detect_resp.text[:200]}...'")
                return


            if not det_json: # If det_json is still None after parsing attempts
                logger.error("🧠 Could not obtain valid JSON from /memory/detect_intent response. Aborting memory storage.")
                return

            logger.info(f"🧠 /memory/detect_intent status: {det_json.get('status')}, detection_result preview: '{str(det_json.get('detection_result'))[:100]}...'")

            if det_json.get("status") == "success" and "MEMORY_DETECTED: YES" in det_json.get("detection_result", ""):
                detection_result_text = det_json.get("detection_result", "")
                m = re.search(r"MEMORY_CONTENT: (.*?)(?:\n|$)", detection_result_text, re.DOTALL)
                if not m:
                    logger.warning(f"🧠 Memory intent detected YES, but failed to extract MEMORY_CONTENT from: '{detection_result_text}'")
                    return
                content = m.group(1).strip()

                cat_match = re.search(r"MEMORY_CATEGORY: (.*?)(?:\n|$)", detection_result_text, re.DOTALL)
                imp_match = re.search(r"MEMORY_IMPORTANCE: (.*?)(?:\n|$)", detection_result_text, re.DOTALL)

                category = cat_match.group(1).strip() if cat_match else "general"
                importance_val = 0.5 # Default
                if imp_match:
                    try:
                        importance_val = max(0.1, min(1.0, float(imp_match.group(1).strip())))
                    except ValueError:
                        logger.warning(f"🧠 Could not parse importance value '{imp_match.group(1).strip()}', using default 0.5.")
                        importance_val = 0.5
                else: # No importance match
                    importance_val = 0.5


                add_payload = {
                    "content": content,
                    "category": category,
                    "importance": importance_val,
                    "type": "auto",
                    "user_id": user_id, # Ensure user_id is correctly passed to /memory/add
                }
                logger.info(f"🧠 Preparing to add memory with payload: {add_payload}")

                try:
                    add_resp = await client.post(
                        f"http://localhost:{memory_port}/memory/add", # Assuming this is your memory service URL
                        json=add_payload,
                        timeout=60.0,
                    )
                    # Check status before trying to parse JSON, especially for errors like 422
                    if add_resp.status_code == 422:
                        error_detail = "Memory content validation failed (e.g., too short)."
                        try:
                            error_payload = add_resp.json()
                            error_detail = error_payload.get("detail", error_detail)
                        except json.JSONDecodeError:
                            pass # Keep the generic error detail
                        logger.warning(f"🧠 Memory addition rejected with 422: {error_detail}. Payload was: {add_payload}")
                        # No return here, just log, as this is an expected "failure" for invalid content

                    elif not (200 <= add_resp.status_code < 300) : # Raise for other HTTP errors
                         add_resp.raise_for_status() # Will raise an httpx.HTTPStatusError

                    else: # Successful add (2xx status)
                        add_data = add_resp.json() # Should be safe now
                        if add_data.get("status") == "success":
                            logger.info(f"🧠 Memory successfully added: '{content[:50]}...'")
                        else:
                            # This case implies 2xx status but "status": "failed" in JSON, which might be unusual
                            logger.error(f"🧠 Memory addition reported failure by API: {add_data.get('error', 'Unknown error')}. Payload: {add_payload}")

                except httpx.HTTPStatusError as http_error: # Catches non-2xx responses
                    logger.error(f"🧠 HTTP error during /memory/add: {http_error}. Response: '{http_error.response.text[:200]}...'")
                    # No need to re-parse JSON here, http_error.response.text has it
                except httpx.RequestError as req_error: # For network errors, timeouts etc.
                    logger.error(f"🧠 Request error during /memory/add: {req_error}")
                except Exception as add_error: # Catch-all for other errors during the add attempt
                    logger.error(f"🧠 Unexpected error during /memory/add: {add_error}", exc_info=True)
            else:
                logger.info("🧠 No memory detected by /memory/detect_intent, or status was not success.")
    except httpx.RequestError as client_req_error: # For network errors, timeouts for /memory/detect_intent
        logger.error(f"🧠 Request error calling /memory/detect_intent: {client_req_error}")
    except Exception as e: # Catch-all for other errors in the main try block
        logger.error(f"🧠 Unhandled error in memory processing pipeline: {e}", exc_info=True) # Changed to .error and added exc_info
# This endpoint handles the generation of text based on user input and model settings.



@router.post("/generate")
async def generate(
    request: Request,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
    body: GenerateRequest = Body(...), # body.request_purpose is now available here
):
    # --- DIAGNOSTIC LOG FOR VISION DEBUGGING ---
    logger = logging.getLogger(__name__)
    logger.info("--- VISION DEBUG: /generate endpoint hit ---")
    logger.info(f"Request for model: {body.model_name}")
    logger.info(f"Is body.image_base64 present: {bool(body.image_base64)}")
    if body.image_base64:
        logger.info(f"image_base64 length: {len(body.image_base64)}")
        logger.info(f"image_base64 start: {body.image_base64[:80]}...")
    else:
        logger.info("image_base64 is NOT present in the request body.")
    logger.info("--- END VISION DEBUG ---")
    # --- END DIAGNOSTIC LOG ---
    # 0) Determine memory_port for this request (needed for memory context and detect_and_store)
    memory_port = 8000 if SINGLE_GPU_MODE else 8001
    
    # Log the purpose of the request (user_chat or title_generation)
    logger.info(f"➡️ Entering /generate endpoint. Purpose: {body.request_purpose or 'user_chat'}")

    # 1) GPU & token settings (No changes here)
    gpu_id = body.gpu_id if body.gpu_id is not None else getattr(request.app.state, 'default_gpu', 0)
    max_tokens = body.max_tokens if body.max_tokens and body.max_tokens > 0 else 4096
    logger.info(f"[DBG gen] full request body → {body!r}")

    # 2) Determine user_id (THIS VERSION IS MORE ROBUST)
    user_profile_from_request = body.userProfile or {}
    logger.info(f"[DBG gen] user_profile_from_request in /generate: {user_profile_from_request!r}")
    
    user_id = None
    if user_profile_from_request: # Try body.userProfile first
        user_id = (
            user_profile_from_request.get("id") or 
            user_profile_from_request.get("userId") or 
            user_profile_from_request.get("user_id")
        )
        user_id = str(user_id) if user_id else None

    if not user_id: # If not found in body.userProfile, try your user_utils fallback
        logger.info(f"User ID not found in body.userProfile. Attempting fallback via user_utils.")
        try:
            from . import user_utils # Ensure this import path is correct from main.py
            user_id = user_utils.get_active_profile_id()
            if user_id:
                logger.info(f"🧠 Successfully obtained user_id='{user_id}' via user_utils fallback for /generate logic.")
            else:
                logger.warning("🧠 user_utils.get_active_profile_id() returned None or empty.")
        except ImportError:
            logger.error("🧠 Failed to import user_utils in /generate. Cannot use fallback for profile ID.")
        except Exception as e:
            logger.error(f"🧠 Error using user_utils fallback in /generate: {e}")
    
    # Final check for user_id for this /generate instance
    if not user_id:
        logger.warning("⚠️ CRITICAL: user_id could not be determined for this /generate call. Memory context and detect_and_store will be skipped.")
    else:
        logger.info(f"✅ User ID for this /generate call (for memory context & scheduling detect_and_store): '{user_id}'")

    # 3) Split client's original prompt into character_persona and user_query
    # This 'original_client_prompt' is what the frontend (e.g., apiCall.js after formatPrompt) sends.
    # For title generation, it's "Generate a title...". For user chat, it includes system, persona, history.
    original_client_prompt = body.prompt or ""
    character_persona_from_split = ""
    user_query_from_split = ""

    # New, more robust splitting logic that handles multiple prompt formats
    if "<start_of_turn>user" in original_client_prompt:
        # Find the last user turn marker
        last_user_turn_start = original_client_prompt.rfind("<start_of_turn>user")
        
        # The persona context is everything before the last user turn
        character_persona_from_split = original_client_prompt[:last_user_turn_start].strip()
        
        # The user's query is within the last user turn
        temp_query_block = original_client_prompt[last_user_turn_start:]
        
        # Extract content between <start_of_turn>user and <end_of_turn>
        user_content_match = re.search(r"<start_of_turn>user\n(.*?)(?:<end_of_turn>|$)", temp_query_block, re.DOTALL)
        if user_content_match:
            user_query_from_split = user_content_match.group(1).strip()
        else:
            # Fallback if the end tag is missing for some reason
            user_query_from_split = temp_query_block.replace("<start_of_turn>user\n", "").strip()

    elif "Human:" in original_client_prompt:
        parts = original_client_prompt.rsplit("Human:", 1)
        character_persona_from_split = parts[0].strip()
        user_query_from_split = parts[1].strip()

    elif "User Query:" in original_client_prompt:
        parts = original_client_prompt.split("User Query:", 1)
        character_persona_from_split = parts[0].strip()
        user_query_from_split = parts[1].strip()
    else:
        # Fallback for simple prompts (like title generation)
        user_query_from_split = original_client_prompt.strip()
        character_persona_from_split = ""

    # For analysis/testing, don't use character persona
    if body.request_purpose == "model_judging":
        character_persona_from_split = ""  # Force empty for analysis  
    
    logger.info(f"Extracted user query (from step #3 split): '{user_query_from_split[:100]}...'")
    logger.info(f"Extracted character persona (from step #3 split): {'Present' if character_persona_from_split else 'Not explicitly separated in client_prompt'}")

    # 4) Prepare input for memory context retrieval.
    # We use 'user_query_from_split' as it's the user's most recent conversational turn.
    input_for_memory_retrieval = user_query_from_split[:300]

    # 5) Fetch memory context (ONLY for user chats, not for title generation)
    memory_context_for_llm = "" # Initialize
    if body.request_purpose not in ["title_generation", "model_judging", "model_testing"]:
        if user_id:
            logger.info(f"🧠 Attempting to fetch memory context for user '{user_id}' using input: '{input_for_memory_retrieval}'")
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"http://localhost:{memory_port}/memory/relevant", # Your memory service
                        json={
                            "prompt": input_for_memory_retrieval,
                            "userProfile": user_profile_from_request, # Pass the profile from the request
                            "systemTime": datetime.datetime.now().isoformat(),
                            "requestType": "generate_user_chat", # More specific type
                            "active_character": body.active_character,
                        },
                        timeout=240.0,
                    )
                resp.raise_for_status()
                data = resp.json()
                memory_context_for_llm = data.get("formatted_memories", "")
                if memory_context_for_llm:
                    logger.info(f"🧠 Retrieved {data.get('memory_count',0)} memories, {len(memory_context_for_llm)} chars for LLM context.")
                else:
                    logger.info(f"🧠 No relevant memories found or formatted_memories was empty for user '{user_id}'.")
            except Exception as e:
                logger.error(f"🧠 Memory context fetch error: {e}", exc_info=True)
        else:
            logger.info("🧠 Skipping memory context fetch: user_id is not available.")
    else:
        logger.info("🌀 Title generation request: Skipping memory context retrieval for LLM prompt.")

    # 6) Construct the main interaction block for the LLM prompt.
    # This block will contain the user's query and any prepended memory or appended RAG.
    # 'user_query_from_split' is the core user message.
    
    # Start with an empty list of components for the interaction block
    interaction_components = []

    if memory_context_for_llm: # Prepend memory if available
        interaction_components.append("RELEVANT USER INFORMATION:\n" + memory_context_for_llm)

    # Add the actual user query (ensure "User Query:" prefix if not already there or if desired)
    # If user_query_from_split is "Generate a title...", it doesn't make sense to prefix it with "User Query:" again for the LLM
    # For actual user chat, user_query_from_split *is* the user's query.
    if body.request_purpose in ["title_generation", "model_testing", "model_judging"]:
        interaction_components.append(user_query_from_split) # For titles, the query is the instruction
    else: # For user chat
        interaction_components.append(f"User Query: {user_query_from_split}")

    # 7) Optionally integrate RAG (ONLY for user chats)
    if body.use_rag and body.request_purpose != "title_generation":
        logger.info(f"🔍 Attempting RAG with query: '{user_query_from_split[:100]}...' and docs: {body.rag_docs}")
        if getattr(request.app.state, 'rag_available', False):
            try:
                rag_res = rag_utils.query_documents(
                    question=user_query_from_split, # Use the clean user query
                    doc_ids=body.rag_docs or [],
                    top_k=5,
                    threshold=0.3, 
                )
                if rag_res.get('status') == 'success':
                    rag_content = rag_res.get('formatted_context', '')
                    if rag_content:
                        interaction_components.append("DOCUMENT CONTEXT:\n" + rag_content) # Append RAG
                        logger.info(f"🔍 Added {len(rag_res.get('chunks', []))} RAG chunks to interaction block.")
            except Exception as e:
                logger.error(f"❌ RAG error: {e}", exc_info=True)
        else:
            logger.warning("🔍 RAG requested but RAG system is not available in app state.")
    elif body.request_purpose == "title_generation":
        logger.info("🌀 Title generation request: Skipping RAG.")
    else: # RAG not enabled for this request
        logger.info("🔍 RAG not enabled for this user chat request.")

    # 7.5) Optionally integrate Web Search (NEW)
    if body.use_web_search and body.request_purpose not in ["title_generation", "model_testing", "model_judging"]:
        # Determine search query - use custom query if provided, otherwise use user's query
        search_query = body.web_search_query if body.web_search_query else user_query_from_split
        
        logger.info(f"🌐 Performing web search for: '{search_query[:100]}...'")
        try:
            web_search_context = await perform_web_search(search_query, max_results=5)
            if web_search_context and "No results found" not in web_search_context:
                interaction_components.append(web_search_context)
                logger.info(f"🌐 Added web search results to interaction block")
            else:
                logger.info(f"🌐 Web search returned no useful results")
        except Exception as e:
            logger.error(f"❌ Web search error: {e}", exc_info=True)
            # Don't fail the whole request if web search fails
            interaction_components.append(f"WEB SEARCH: Search failed - {str(e)}")
    elif body.use_web_search and body.request_purpose == "title_generation":
        logger.info("🌐 Title generation request: Skipping web search")
    elif body.use_web_search:
        logger.info(f"🌐 Web search requested but skipped for request_purpose: {body.request_purpose}")
    else:
        logger.debug("🌐 Web search not enabled for this request")

    # 8) Query conversation history to prevent repetition (for analysis chats)
    if body.request_purpose == "model_testing" and body.use_rag:
        logger.info(f"🔄 Querying conversation chunks to prevent repetition...")
        if getattr(request.app.state, 'rag_available', False):
            try:
                # Query for similar conversation topics  
                conversation_rag_res = rag_utils.query_documents(
                    question=user_query_from_split,
                    doc_ids=None,
                    top_k=3,
                    threshold=0.3,  # Higher threshold for better matches
                )
                
                if conversation_rag_res.get('status') == 'success':
                    conv_chunks = []
                    for chunk in conversation_rag_res.get('chunks', []):
                        if chunk.get('document', {}).get('file_type') == 'conversation':
                            conv_chunks.append(chunk['chunk'])
                    
                    if conv_chunks:
                        # Format as discussion context, not document sections
                        conversation_context = f"PREVIOUS DISCUSSION:\n{chr(10).join(conv_chunks[:2])}\n\nAvoid repeating the above topics. Build on them or explore new angles."
                        interaction_components.append(conversation_context)
                        logger.info(f"🔄 Added conversation context to prevent repetition.")
            except Exception as e:
                logger.error(f"❌ Conversation RAG error: {e}")
        
    # Join all components of the interaction block with double newlines
    final_interaction_block = "\n\n".join(interaction_components)

    # 9) Assemble the full LLM prompt
    #    SYSTEM_PROMPT is your global default (e.g., "You are GingerGPT...")
    #    character_persona_from_split contains the character-specific system instructions from the client.
    # Skip roleplay system prompt for model testing
    if body.request_purpose in ["model_testing", "model_judging"]:
        logger.info("🌀 Model testing/judging request: Skipping roleplay system prompt.")
        system_block_for_llm = "You are a language model designed for testing and evaluation purposes. Respond to the user's input without roleplay context."
    else:
        system_block_for_llm = f"System:\n{SYSTEM_PROMPT.strip()}"
    if character_persona_from_split: # This is from step #3 split
        system_block_for_llm += f"\n\nCharacter Persona:\n{character_persona_from_split}"
    
    # Construct the final prompt for the LLM
    llm_prompt = f"{system_block_for_llm.strip()}\n\n{final_interaction_block.strip()}\n\nAssistant:"
    # 10) Log the final prompt sent to LLM
    logger.info(f"[generate] FULL LLM PROMPT ({len(llm_prompt)} chars) >>>\n{llm_prompt}\n<<<")

    # 11) LLM Generation & Conditional Scheduling of detect_and_store
    if body.stream:
        logger.info("🔄 Streaming response requested. detect_and_store will be scheduled post-stream.")

        async def response_generator_with_post_action(
            bg_tasks: BackgroundTasks,
            # Pass necessary variables from the outer scope
            current_user_id: Optional[str],
            prompt_text_for_llm: str,
            user_query_for_detection: str,
            user_profile_for_detection_task: dict,
            is_title_generation_request: bool
        ):
            
            # Check for vision input first
            if body.image_base64:
                # Vision models don't stream well, fall back to non-streaming
                llm_output_raw_text = await generate_text_with_vision(
                    model_manager=model_manager,
                    model_name=body.model_name,
                    prompt=prompt_text_for_llm,
                    image_base64=body.image_base64,
                    max_tokens=max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    top_k=body.top_k,
                    repetition_penalty=body.repetition_penalty,
                    stop_sequences=dcu.get_stop_sequences(body.stop),
                    gpu_id=gpu_id,
                    echo=body.echo,
                    request_purpose=body.request_purpose
                )
                # Clean and return like non-streaming
                clean_llm_response = llm_output_raw_text.replace("<|DONE|>", "").strip()
                yield f"data: {json.dumps({'text': clean_llm_response})}\n\n"
                yield "data: [DONE]\n\n"  # Signal end of stream to client 
            else:
                # Regular streaming for text-only requests
                streamed_content_accumulator = []
                try:
                    async for token in inference.generate_text_streaming(
                        model_manager=model_manager, model_name=body.model_name, prompt=prompt_text_for_llm,
                        max_tokens=max_tokens, temperature=body.temperature, top_p=body.top_p,
                        top_k=body.top_k, repetition_penalty=body.repetition_penalty,
                        stop_sequences=dcu.get_stop_sequences(body.stop), gpu_id=gpu_id, echo=body.echo,
                        request_purpose=body.request_purpose
                    ):
                        streamed_content_accumulator.append(token)
                        yield f"data: {json.dumps({'text': token})}\n\n"
                    
                    yield "data: [DONE]\n\n"  # Signal end of stream to client
                except Exception as stream_exc:
                    logger.error(f"❌ Error during LLM streaming: {stream_exc}", exc_info=True)
                    # Optionally, yield an error event to the client if your frontend handles it
                    # yield f"event: error\ndata: {json.dumps({'detail': str(stream_exc)})}\n\n"
                    # Ensure [DONE] is still sent or handle client-side appropriately
                    yield f"data: {json.dumps({'text': f'[STREAM_ERROR: {str(stream_exc)}]'})}\n\n" # Send error in data
                    yield "data: [DONE]\n\n"

            # ---- After stream is DONE ----
            full_llm_response_text = "".join(streamed_content_accumulator)
            logger.info(f"🌀 Stream complete. Full response length: {len(full_llm_response_text)}. Scheduling detect_and_store if applicable.")

            clean_full_llm_response = full_llm_response_text.replace("<|DONE|>", "").strip() # Clean it once

            if is_title_generation_request or body.request_purpose in ["model_testing", "model_judging"]:
                logger.info(f"🌀 {body.request_purpose} stream complete. Skipping memory detection and storage.")
            elif not current_user_id:
                logger.warning(f"🧠 Stream complete. Skipping detect_and_store: No current_user_id available.")
            else:
                logger.info(f"✅ Stream complete. Conditions met for scheduling detect_and_store: user_id='{current_user_id}'.")
                
                # user_profile_for_detection_task is already prepared with an ID if possible before being passed here
                
                logger.info(f" scheduling detect_and_store for user chat. User's input for detection: '{user_query_for_detection[:100]}...'")
                bg_tasks.add_task(
                    detect_and_store,
                    clean_full_llm_response, # Use the cleaned full response
                    user_query_for_detection,
                    user_profile_for_detection_task
                )
                logger.info(f"🧠 Memory write task scheduled post-stream for user ID: {current_user_id} (user chat)")

        # Prepare the user_profile object that will be passed to detect_and_store
        # This ensures it has an ID, using the user_id determined in Step #2 of /generate
        user_profile_for_task = dict(user_profile_from_request or {}) # Start with a copy or new dict
        if not user_profile_for_task.get("id") and user_id: # If body.userProfile was {} or lacked ID
            user_profile_for_task["id"] = user_id
            logger.info(f"🧠 Ensured user_profile for post-stream task has id: '{user_id}'")
        elif not user_profile_for_task.get("id") and getattr(request.app.state, "active_profile_id", None):
            # Fallback to app.state.active_profile_id if body.userProfile was empty AND user_id from step 2 was also somehow None
            # Though user_id from Step #2 should be reliable now. This is an extra safeguard.
            app_state_profile_id = getattr(request.app.state, "active_profile_id", None)
            if app_state_profile_id:
                user_profile_for_task["id"] = app_state_profile_id
                logger.info(f"🧠 Ensured user_profile for post-stream task has id (from app.state): '{app_state_profile_id}'")


        return StreamingResponse(
            response_generator_with_post_action(
                background_tasks,
                user_id, # Pass the user_id determined in Step #2
                llm_prompt, # Pass the fully assembled prompt for the LLM
                user_query_from_split, # Pass the user's direct query for memory detection
                user_profile_for_task, # Pass the prepared user profile object for the task
                (body.request_purpose == "title_generation") # Pass boolean flag
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    else: # Non-streaming path (remains largely the same, detect_and_store scheduled at the end)
        llm_output_raw_text = ""
        try:
            logger.info("🔄 Non-streaming response requested. Dispatching to model...")
            
            # Get the loaded model instance once for this request
            model_instance = model_manager.get_model(body.model_name, gpu_id)
            if not model_instance:
                raise ValueError(f"Model {body.model_name} not loaded on GPU {gpu_id}")

            # --- UNIFIED DISPATCH LOGIC ---
            # This logic block decides how to call the model based on whether an image is present.
            # It uses the fully assembled 'llm_prompt' for both paths.
            
            if body.image_base64:
                # --- VISION PATH ---
                # For vision, we must use create_chat_completion. Our custom GemmaVisionChatHandler
                # expects the full prompt string to be passed within the "text" part of the user message.
                logger.info("✅ Constructing vision payload with full context.")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            # Pass the entire, context-rich prompt here. The handler will format it.
                            {"type": "text", "text": llm_prompt},
                            # Pass the image data using the stable data URI method.
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{body.image_base64}"}}
                        ]
                    }
                ]
                
                response = model_instance.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    top_k=body.top_k,
                    repeat_penalty=body.repetition_penalty,
                    stop=["<end_of_turn>"] # Essential stop token for Gemma
                )
                if response and response.get('choices'):
                    llm_output_raw_text = response['choices'][0]['message']['content']
                else:
                    llm_output_raw_text = "Vision model returned no valid response."

            else:
                # --- TEXT-ONLY PATH (Your original, working logic) ---
                # For text, we call the model directly with the full prompt string.
                logger.info("✅ Dispatching to standard text generation.")
                response = model_instance(
                    prompt=llm_prompt,
                    max_tokens=max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    top_k=body.top_k,
                    repeat_penalty=body.repetition_penalty,
                    stop=["<end_of_turn>", "<|DONE|>"] + dcu.get_stop_sequences(body.stop)
                )
                if response and response.get('choices'):
                    llm_output_raw_text = response['choices'][0]['text']
                else:
                    llm_output_raw_text = "Text model returned no valid response."

        except Exception as exc:
            logger.error(f"❌ Generation error (non-streaming): {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

        # 12) Post-process LLM output
        logger.info(f"🔄 Raw LLM output length (non-streaming): {len(llm_output_raw_text)} characters")
        clean_llm_response = llm_output_raw_text.replace("<|DONE|>", "").strip()
        
        # 13) Schedule memory detection and storage (for non-streaming user chats)
        if body.request_purpose in ["title_generation", "model_testing", "model_judging"]:
            logger.info("🌀 Title generation request (non-streaming). Skipping memory detection.")
        elif not user_id:
            logger.warning(f"🧠 Memory detection/storage skipped (non-streaming): No user_id available. (Purpose: {body.request_purpose or 'user_chat'})")
        else:
            logger.info(f"✅ Conditions met for scheduling detect_and_store (non-streaming): user_id='{user_id}'.")
            
            prompt_that_elicited_response = user_query_from_split
            user_profile_for_task = dict(user_profile_from_request or {})
            if not user_profile_for_task.get("id") and user_id:
                 user_profile_for_task["id"] = user_id
            elif not user_profile_for_task.get("id") and getattr(request.app.state, "active_profile_id", None):
                 user_profile_for_task["id"] = getattr(request.app.state, "active_profile_id", None)
            
            logger.info(f" scheduling detect_and_store for user chat (non-streaming). User's input for detection: '{prompt_that_elicited_response[:100]}...'")
            background_tasks.add_task(
                detect_and_store,
                clean_llm_response, # Use cleaned response here
                prompt_that_elicited_response,
                user_profile_for_task
            )
            logger.info(f"🧠 Memory write task scheduled for user ID: {user_id} (user chat, non-streaming)")

        # 14) Return final response to client
        return {"text": clean_llm_response}    
    
@router.post("/models/set-openai-api-mode")
async def set_openai_api_mode(data: dict = Body(...)):
    """Save OpenAI API mode to settings"""
    try:
        use_openai_api = data.get("useOpenAIAPI", False)
        
        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        os.makedirs(settings_dir, exist_ok=True)
        
        settings = {}
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        
        settings['useOpenAIAPI'] = use_openai_api
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/save-custom-endpoints")
async def save_custom_endpoints(data: dict = Body(...)):
    """Save custom API endpoints to settings"""
    try:
        endpoints = data.get("customApiEndpoints", [])
        
        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        os.makedirs(settings_dir, exist_ok=True)
        
        settings = {}
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        
        settings['customApiEndpoints'] = endpoints
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        return {"status": "success", "message": "Endpoints saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/set-api-endpoint")
async def set_api_endpoint(data: dict = Body(...)):
    """Set API endpoint URL and save to settings"""
    try:
        url = data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Save to settings file (same pattern as model directory)
        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        
        os.makedirs(settings_dir, exist_ok=True)
        
        settings = {}
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            except:
                pass
        
        settings['apiEndpointUrl'] = url
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        return {"status": "success", "message": "API endpoint updated"}
    except Exception as e:
        logger.error(f"Error updating API endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get("/sd-local/adetailer-models")
async def list_adetailer_models(request: Request):
    """List available ADetailer models"""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        return {"available": False, "models": []}
    
    models = sd_manager.adetailer.available_models
    
    # Add default models that auto-download
    default_models = [
        "face_yolov8n.pt",
        "face_yolov8s.pt", 
        "face_yolov8m.pt",
        "hand_yolov8n.pt",
        "person_yolov8n-seg.pt"
    ]
    
    all_models = list(set(models + default_models))
    
    return {
        "available": True,
        "models": all_models,
        "custom_models": models,
        "directory": str(sd_manager.adetailer.model_directory) if sd_manager.adetailer.model_directory else None
    }

@router.post("/sd-local/set-adetailer-directory")
async def set_adetailer_directory(request: Request, data: dict = Body(...)):
    """Set ADetailer model directory"""
    try:
        directory = data.get("directory")
        if not directory or not os.path.isdir(directory):
            raise HTTPException(status_code=400, detail="Invalid directory path")
        
        # Save to settings
        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        settings_dir.mkdir(exist_ok=True)
        
        settings = {}
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        
        settings['adetailerModelDirectory'] = directory
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        # Update manager
        sd_manager = getattr(request.app.state, 'sd_manager', None)
        if sd_manager:
            sd_manager.adetailer.set_model_directory(directory)
        
        return {"status": "success", "message": "ADetailer directory updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Update existing enhance endpoint
@router.post("/sd-local/enhance-adetailer")
async def enhance_image_with_adetailer(request: Request, data: dict = Body(...)):
    """Enhance an existing image using ADetailer post-processing"""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    if not sd_manager.is_loaded():
        raise HTTPException(status_code=400, detail="No SD model loaded")

    try:
        # Extract parameters
        image_url = data.get("image_url")
        original_prompt = data.get("original_prompt", "")
        face_prompt = data.get("face_prompt", "")
        strength = data.get("strength", 0.4)
        confidence = data.get("confidence", 0.3)
        model_name = data.get("model_name", "face_yolov8n.pt")  # NEW: model selection
        
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        
        # Check if ADetailer is available
        if not sd_manager.is_adetailer_available():
            raise HTTPException(
                status_code=503, 
                detail="ADetailer not available. Install ultralytics: pip install ultralytics"
            )
        
        # Convert URL to local file path
        if "/static/generated_images/" in image_url:
            filename = image_url.split("/static/generated_images/")[-1]
            image_path = Path(__file__).parent / "static" / "generated_images" / filename
        else:
            raise HTTPException(status_code=400, detail="Invalid image URL")
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        logger.info(f"Enhancing image: {image_path} with ADetailer using model {model_name}")
        
        # Enhance the image
        enhanced_image_data = sd_manager.enhance_image_with_adetailer(
            image_path=str(image_path),
            original_prompt=original_prompt,
            face_prompt=face_prompt,
            strength=strength,
            confidence=confidence,
            model_name=model_name  # NEW: pass model name
        )
        
        # Save enhanced image
        enhanced_filename = f"enhanced_{uuid.uuid4()}.png"
        enhanced_path = Path(__file__).parent / "static" / "generated_images" / enhanced_filename
        
        with open(enhanced_path, "wb") as f:
            f.write(enhanced_image_data)
        
        # Return the enhanced image URL
        enhanced_url = f"/static/generated_images/{enhanced_filename}"
        
        return {
            "status": "success",
            "enhanced_image_url": enhanced_url,
            "original_image_url": image_url,
            "enhancement_applied": True,
            "model_used": model_name,  # NEW: return which model was used
            "parameters": {
                "strength": strength,
                "confidence": confidence,
                "face_prompt": face_prompt,
                "model_name": model_name
            }
        }
        
    except Exception as e:
        logger.error(f"ADetailer enhancement error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@router.get("/sd-local/adetailer-status")
async def get_adetailer_status(request: Request):
    """Check if ADetailer functionality is available"""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        return {"available": False, "error": "SD Manager not available"}
    
    available = sd_manager.is_adetailer_available()
    return {
        "available": available,
        "models_loaded": available,
        "message": "ADetailer ready" if available else "Install ultralytics for ADetailer support"
    }
@router.post("/models/refresh-directory")
async def refresh_model_directory(
    data: dict = Body(...),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Update the model directory and refresh available models."""
    try:
        new_directory = data.get("directory")
        if not new_directory or not os.path.isdir(new_directory):
            raise HTTPException(status_code=400, detail="Invalid directory path")
        
        # Save to settings file
        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        
        # Create directory if it doesn't exist
        os.makedirs(settings_dir, exist_ok=True)
        
        # Read existing settings or create new
        settings = {}
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            except:
                pass
        
        # Update settings
        settings['modelDirectory'] = new_directory
        
        # Write settings
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        # Update model manager
        model_manager.models_dir = Path(new_directory)
        
        # Return success
        return {"status": "success", "message": "Model directory updated"}
    except Exception as e:
        logger.error(f"Error updating model directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# This endpoint updates the SD model directory and saves it to settings.
@router.post("/sd-local/refresh-directory")
async def refresh_sd_model_directory(
    data: dict = Body(...),
):
    """Update the SD model directory and save to settings"""
    try:
        new_directory = data.get("directory")
        if not new_directory or not os.path.isdir(new_directory):
            raise HTTPException(status_code=400, detail="Invalid directory path")
        
        # Save to settings file (same pattern as your existing code)
        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        
        os.makedirs(settings_dir, exist_ok=True)
        
        settings = {}
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            except:
                pass
        
        settings['sdModelDirectory'] = new_directory
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        return {"status": "success", "message": "SD model directory updated"}
    except Exception as e:
        logger.error(f"Error updating SD model directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# --- Placeholder routes ---
@app.get("/sd/status")
async def sd_status():
    """Check if AUTOMATIC1111 is up by listing available SD models."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "http://127.0.0.1:7860/sdapi/v1/sd-models",
                timeout=5.0
            )
        resp.raise_for_status()
        return { "automatic1111": True, "models": resp.json() }
    except Exception:
        return { "automatic1111": False, "models": [] }

@app.post("/sd/txt2img")
async def sd_txt2img(body: dict):
    """Proxy to Automatic1111, save returned images to files, and return URLs."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://127.0.0.1:7860/sdapi/v1/txt2img",
                json=body,
                timeout=240.0
            )
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"SD API error: {e}")

    sd_response = resp.json()
    images_base64 = sd_response.get("images", [])
    if not images_base64:
        raise HTTPException(500, "No images returned from SD API")

    saved_image_urls = []
    for b64_string in images_base64:
        # A1111 can return a data URI or raw base64; handle both
        if b64_string.startswith("data:image/png;base64,"):
            b64_string = b64_string.split(',', 1)[1]

        image_data = base64.b64decode(b64_string)
        image_url = save_image_and_get_url(image_data)
        saved_image_urls.append(image_url)

    return JSONResponse({
        "status": "success",
        "image_urls": saved_image_urls,  # Return the list of file URLs
        "parameters": sd_response.get("parameters", {}),
        "info": sd_response.get("info", "")
    })
@app.get("/sd-local/status")
async def sd_local_status(request: Request):
    """Check local SD status and loaded model"""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        return {"available": False, "error": "SD Manager not initialized"}
    
    return {
        "available": True,
        "model_loaded": sd_manager.is_loaded(),
        "current_model": sd_manager.current_model_path
    }

@app.post("/sd-local/load-model")
async def sd_local_load_model(data: dict, request: Request):
    """Load a local SD model by its filename"""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    # This is the fix: we now correctly look for 'model_filename' from the request
    model_filename = data.get("model_filename")
    if not model_filename:
        raise HTTPException(status_code=400, detail="Request body must include 'model_filename'")

    # The backend now correctly builds the full path from the configured directory
    sd_model_dir = getattr(request.app.state, 'sd_model_directory', None)
    if not sd_model_dir:
        raise HTTPException(status_code=500, detail="SD model directory not configured on backend.")

    full_model_path = str(Path(sd_model_dir) / model_filename)

    success = sd_manager.load_model(full_model_path)
    if success:
        return {"status": "success", "message": f"Model loaded: {full_model_path}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model from path: {full_model_path}")

@app.post("/sd-local/txt2img")
async def sd_local_txt2img(body: dict, request: Request):
    """Generate image using local SD, save it as a file, and return the URL."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    if not sd_manager.is_loaded():
        raise HTTPException(status_code=400, detail="No SD model loaded")

    try:
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt required")

        # Check for seed and randomize if it's -1
        seed = body.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Local SD: No seed provided, generated random seed: {seed}")

        # This returns the raw image bytes
        image_data = sd_manager.generate_image(
            prompt=prompt,
            negative_prompt=body.get("negative_prompt", ""),
            width=body.get("width", 512),
            height=body.get("height", 512),
            steps=body.get("steps", 20),
            cfg_scale=body.get("guidance_scale", 7.0),
            seed=seed # Pass the new, potentially randomized seed
        )

        image_url = save_image_and_get_url(image_data)

        # Important: Include the seed in the parameters for the frontend to know it
        final_params = body.copy()
        final_params['seed'] = seed

        return {
            "status": "success",
            "image_urls": [image_url],
            "parameters": final_params,
            "info": "Generated with local stable-diffusion.cpp"
        }

    except Exception as e:
        logger.error(f"Local SD generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# These endpoints are now handled in document_routes.py

# mount the “generate” router
app.include_router(router)

# mount your memory endpoints under the “/memory” prefix
app.include_router(memory_router, prefix="/memory")
app.include_router(memory_router, prefix="/memory", tags=["memory"])
app.include_router(document_router)
# Add OpenAI compatibility layer
app.include_router(openai_router)
app.include_router(tts_service.router)
logger.info("🔗 OpenAI-compatible API endpoints available at /v1/chat/completions and /v1/models")