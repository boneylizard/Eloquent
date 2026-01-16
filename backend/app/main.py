import os
# Disable problematic Torch optimizations for Python 3.12+ (MUST BE AT TOP)
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# MONKEYPATCH: Disable torch.compile to avoid Dynamo error on Python 3.12+
try:
    import torch
    if not hasattr(torch, '_original_compile'):
        torch._original_compile = torch.compile
        def dummy_compile(f, *args, **kwargs): return f
        torch.compile = dummy_compile
except Exception:
    pass

os.environ["CUDA_MODULE_LOADING"] = "EAGER" # Ensure CUDA modules load eagerly
# REMOVED: CUDA_LAUNCH_BLOCKING="1" - This can hurt GPU performance!
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["GGML_CUDA_NO_PINNED"] = "0"

# Actually FORCE CUDA initialization with llama_cpp

# --- END: DEFINITIVE GPU ISOLATION ---
from pyexpat.errors import messages
from fastapi import FastAPI, HTTPException, Depends, APIRouter, File, UploadFile, BackgroundTasks, Request, Query, Body, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os
import json
import threading
import pandas as pd
import json
from .model_manager import DevstralHandler
import xml.etree.ElementTree as ET
import yaml
import io
import subprocess
import fnmatch
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
from . import openai_compat
import shutil
from .forensic_linguistics_service import ForensicLinguisticsService, TextDocument, SimilarityScore
import uuid
from .tts_client import TTSClient  # Use TTS client instead of direct service
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import re
from .model_manager import ModelManager
from . import inference
import io
import yaml
import tempfile
from .stt_service import transcribe_audio # Assuming this is the correct import path for your STT service
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
# Configure logging BEFORE importing modules that use it
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from .sd_manager import SDManager
from .upscale_manager import UpscaleManager
import random
from .web_search_service import perform_web_search
from .openai_compat import router as openai_router, is_api_endpoint, get_configured_endpoint, forward_to_configured_endpoint_streaming, forward_to_configured_endpoint_non_streaming, prepare_endpoint_request
import pynvml
from .devstral_service import devstral_service, DevstralService
# Only set DEBUG-level loggers to WARNING to suppress their excessive output
logging.getLogger('numba.core').setLevel(logging.WARNING)
logging.getLogger('numba.byteflow').setLevel(logging.WARNING)
logging.getLogger('graphviz').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('transformers.modeling_utils').setLevel(logging.WARNING)
logging.getLogger('transformers.configuration_utils').setLevel(logging.WARNING)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# NeMo's logging can be very verbose at DEBUG level
logging.getLogger('nemo').setLevel(logging.WARNING)
logging.getLogger('nemo.collections').setLevel(logging.WARNING)
logging.getLogger('nemo.utils').setLevel(logging.WARNING)

app = FastAPI() # Re-initialize app to avoid conflicts
SINGLE_GPU_MODE = False # Set to True if running on a single GPU

# TTS API URL (for forwarding requests to TTS service on port 8000)
# Note: TTS service endpoints are on main backend (port 8000) not separate service
TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:8000")

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
    Remove any lines that are just the model's name or "'s avatar,"
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

class DevstralToolCallParser:
    """Parse tool calls from Devstral's text-based tool calling format"""
    
    @staticmethod
    def extract_tool_calls_from_content(content: str):
        """
        Extract tool calls from content like:
        list_directory{"path": "/path/to/directory"}
        read_file{"filepath": "README.md"}
        """
        if not content:
            return [], content
        
        tool_calls = []
        remaining_content = content
        
        # Find all potential tool call patterns
        # Look for function_name followed by { and try to parse as JSON
        # This is more robust than regex for complex JSON
        import re
        
        # First, find all function names followed by {
        function_pattern = r'(\w+)\s*\{'
        function_matches = list(re.finditer(function_pattern, content))
        
        for match in function_matches:
            function_name = match.group(1)
            start_pos = match.start()
            
            # Find the matching closing brace by counting braces
            brace_count = 0
            json_start = match.end() - 1  # Start at the opening {
            json_end = -1
            
            for i in range(json_start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > 0:
                json_args_str = content[json_start:json_end]
                
                try:
                    # Parse the JSON arguments
                    args = json.loads(json_args_str)
                    
                    # Create a tool call in OpenAI format
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(args)
                        }
                    }
                    
                    tool_calls.append(tool_call)
                    
                    # Remove this tool call from the content
                    tool_call_text = content[start_pos:json_end]
                    remaining_content = remaining_content.replace(tool_call_text, "").strip()
                    
                except json.JSONDecodeError:
                    # If we can't parse the JSON, skip this match
                    continue
        
        return tool_calls, remaining_content

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    directProfileInjection: bool = False # <-- ADD THIS
    prompt: str
    model_name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    anti_repetition_mode: bool = False
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
    authorNote: Optional[str] = None  # Author's note for custom session instructions
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

class FileOperationRequest(BaseModel):
    filepath: str
    content: Optional[str] = None

class DirectoryListRequest(BaseModel):
    path: str = "."
    include_hidden: bool = False

class SearchFilesRequest(BaseModel):
    query: str
    path: str = "."
    file_pattern: str = "*"
    max_results: int = 100

class RunCommandRequest(BaseModel):
    command: str
    working_dir: Optional[str] = None
    timeout: int = 30

class BackupRequest(BaseModel):
    filepath: str
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

# Helper functions for path safety
def is_safe_path(basedir: str, path: str) -> bool:
    """Check if path is safe (no directory traversal)"""
    try:
        basedir = os.path.abspath(basedir)
        requested_path = os.path.abspath(os.path.join(basedir, path))
        return requested_path.startswith(basedir)
    except (ValueError, OSError):
        return False

def get_safe_path(basedir: str, path: str) -> Optional[str]:
    """Get safe absolute path or None if unsafe"""
    if is_safe_path(basedir, path):
        return os.path.abspath(os.path.join(basedir, path))
    return None

CODE_EDITOR_BASE_DIR = os.getcwd()  # You can change this to a specific project directory


# Add this to your existing FastAPI app
async def generate_llm_response(prompt: str, model_manager, model_name: str, **kwargs) -> str:
    """
    Adapter for your existing LLM generation, handling both local and API models.
    """
    from .inference import generate_text
    from .openai_compat import is_api_endpoint, _prepare_endpoint_request, forward_to_configured_endpoint_non_streaming

    if is_api_endpoint(model_name):
        logger.info(f"Routing generate_llm_response for API endpoint: {model_name}")
        
        # Construct request data compatible with OpenAI API
        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 0.9),
            # Add other params if needed
        }
        
        try:
            endpoint_config, url, prepared_data = _prepare_endpoint_request(model_name, request_data)
            response_json = await forward_to_configured_endpoint_non_streaming(endpoint_config, url, prepared_data)
            
            # Extract content from OpenAI response format
            if 'choices' in response_json and len(response_json['choices']) > 0:
                content = response_json['choices'][0]['message']['content']
                return content
            else:
                logger.error(f"Unexpected API response format: {response_json}")
                return ""
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    else:
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
# WebSocket TTS streaming endpoint moved to TTS service on port 8002
# Use /tts-stream endpoint on port 8002 for TTS WebSocket connections

# WebSocket chat + TTS streaming endpoint moved to TTS service on port 8002
# Use /tts-stream endpoint on port 8002 for TTS WebSocket connections
async def get_forensic_service(request: Request) -> ForensicLinguisticsService:
    if not hasattr(request.app.state, 'forensic_service') or request.app.state.forensic_service is None:
        raise HTTPException(status_code=503, detail="Forensic Linguistics Service is not available.")
    return request.app.state.forensic_service

async def synthesize_speech(
    text: str, 
    voice: str = 'af_heart', 
    engine: str = 'kokoro',
    audio_prompt_path: str = None,
    exaggeration: float = 0.5,
    cfg: float = 0.5
) -> bytes:
    """
    TTS synthesis with Kokoro and Chatterbox support
    """
    print(f"🗣️ [TTS Service] Synthesizing with engine: {engine}, voice: {voice}")
    
    cleaned_text = clean_markdown_for_tts(text)
    if not cleaned_text:
        logger.warning("🗣️ [TTS Service] Text became empty after cleaning")
        return b""
    
    if engine.lower() == 'chatterbox':
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

# Add these endpoints to your router in main.py

@router.post("/forensic/build-corpus")
async def build_forensic_corpus(
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
    
):
    """Build a comprehensive stylometric corpus for a public figure."""
    try:
        person_name = data.get("person_name")
        platforms = data.get("platforms", ["twitter", "speeches", "press_releases", "interviews"])
        max_documents = data.get("max_documents", 1000)
        
        if not person_name:
            raise HTTPException(status_code=400, detail="person_name is required")
        
        logger.info(f"🔍 [Forensic] Building corpus for {person_name}")
        
        # Check if corpus already exists
        existing_corpus = forensic_service._load_cached_corpus(person_name)
        if existing_corpus and len(existing_corpus) > 50:
            return {
                "status": "exists",
                "message": f"Corpus for {person_name} already exists with {len(existing_corpus)} documents",
                "corpus_size": len(existing_corpus),
                "person_name": person_name,
                "platforms": list(set(doc.platform for doc in existing_corpus))
            }
        
        # Build corpus in background
        async def build_corpus_task():
            try:
                corpus = await forensic_service.build_corpus(
                    person_name=person_name,
                    platforms=platforms,
                    max_documents=max_documents
                )
                logger.info(f"✅ [Forensic] Completed corpus building for {person_name}: {len(corpus)} documents")
            except Exception as e:
                logger.error(f"❌ [Forensic] Corpus building failed for {person_name}: {e}")

        corpus = await forensic_service.build_corpus(
            person_name=person_name,
            platforms=platforms,
            max_documents=max_documents
        )

        return {
            "status": "building",
            "message": f"Corpus building started for {person_name}",
            "person_name": person_name,
            "platforms": platforms,
            "max_documents": max_documents
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error in build-corpus: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forensic/corpus-status/{person_name}")
async def get_corpus_status(person_name: str, forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)):
    """Check the status of a person's corpus."""
    try:
        corpus = forensic_service._load_cached_corpus(person_name)
        
        if not corpus:
            return {
                "status": "not_found",
                "person_name": person_name,
                "corpus_size": 0,
                "message": "No corpus found for this person"
            }
        
        # Analyze corpus composition
        platform_breakdown = {}
        for doc in corpus:
            platform_breakdown[doc.platform] = platform_breakdown.get(doc.platform, 0) + 1
        
        # Calculate date range
        dates = [doc.date for doc in corpus if doc.date]
        date_range = {
            "earliest": min(dates).isoformat() if dates else None,
            "latest": max(dates).isoformat() if dates else None
        }
        
        return {
            "status": "ready",
            "person_name": person_name,
            "corpus_size": len(corpus),
            "platform_breakdown": platform_breakdown,
            "date_range": date_range,
            "message": f"Corpus ready with {len(corpus)} documents"
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error checking corpus status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forensic/analyze-file")
async def analyze_file_content(
    file: UploadFile = File(...),
    person_name: str = Query(None, description="Public figure to compare against"),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Analyze uploaded file content for forensic linguistics."""
    try:
        # Validate file size (10MB limit)
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Use the cleaning function
        content = await process_uploaded_file_with_cleaning(file)
        
        if not content.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or unreadable after cleaning")
        
        logger.info(f"🧹 [Forensic] Cleaned content from {file.filename}: {len(content)} characters")
        
        if person_name:
            # Compare against specific person's corpus
            logger.info(f"🔍 [Forensic] Analyzing uploaded file against {person_name}")
            
            corpus = forensic_service._load_cached_corpus(person_name)
            if not corpus:
                raise HTTPException(status_code=404, detail=f"No corpus found for {person_name}")
            
            similarity_scores = forensic_service.analyze_authorship(content, corpus)
            interpretation = forensic_service._interpret_similarity_scores(similarity_scores)
            
            return {
                "status": "success",
                "analysis_type": "authorship_attribution",
                "file_name": file.filename,
                "person_analyzed": person_name,
                "similarity_scores": {
                    "overall_similarity": similarity_scores.overall_similarity,
                    "lexical_similarity": similarity_scores.lexical_similarity,
                    "syntactic_similarity": similarity_scores.syntactic_similarity,
                    "semantic_similarity": similarity_scores.semantic_similarity,
                    "stylistic_similarity": similarity_scores.stylistic_similarity
                },
                "interpretation": interpretation,
                "confidence_level": forensic_service._calculate_confidence(similarity_scores),
                "analysis_timestamp": datetime.now().isoformat(),
                "cleaned_content_length": len(content)
            }
        else:
            # Extract features only
            logger.info(f"🔍 [Forensic] Extracting features from uploaded file: {file.filename}")
            
            style_vector = forensic_service.extract_style_vector(content)
            
            return {
                "status": "success",
                "analysis_type": "feature_extraction",
                "file_name": file.filename,
                "features": {
                    "lexical_features": {
                        "avg_word_length": round(style_vector.avg_word_length, 2),
                        "avg_sentence_length": round(style_vector.avg_sentence_length, 2),
                        "vocab_richness": round(style_vector.vocab_richness, 3),
                        "hapax_legomena_ratio": round(style_vector.hapax_legomena_ratio, 3),
                        "yule_k": round(style_vector.yule_k, 2)
                    },
                    "syntactic_features": {
                        "pos_distribution": {k: round(v, 3) for k, v in style_vector.pos_distribution.items()},
                        "sentence_complexity": round(style_vector.sentence_complexity, 2)
                    },
                    "stylistic_features": {
                        "modal_verb_usage": round(style_vector.modal_verb_usage, 3),
                        "passive_voice_ratio": round(style_vector.passive_voice_ratio, 3),
                        "question_ratio": round(style_vector.question_ratio, 3),
                        "exclamation_ratio": round(style_vector.exclamation_ratio, 3),
                        "capitalization_ratio": round(style_vector.capitalization_ratio, 3)
                    },
                    "punctuation_features": {k: round(v, 4) for k, v in style_vector.punctuation_ratios.items()},
                    "function_word_features": {k: round(v, 3) for k, v in style_vector.function_word_ratios.items()},
                    "sentiment_features": {k: round(v, 3) for k, v in style_vector.sentiment_scores.items()},
                    "text_statistics": {
                        "character_count": len(content),
                        "word_count": len(content.split()),
                        "sentence_count": len(content.split('.')),
                        "paragraph_count": len(content.split('\n\n'))
                    }
                },
                "text_preview": content[:300] + "..." if len(content) > 300 else content,
                "analysis_timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error analyzing uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_uploaded_file_with_cleaning(file: UploadFile) -> str:
    """Enhanced file processing with a new robust cleaning function."""
    content_bytes = await file.read()
    raw_text = ""
    
    # Try to decode the file content
    try:
        raw_text = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # Fallback to another common encoding if UTF-8 fails
        try:
            raw_text = content_bytes.decode('latin-1')
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Unable to decode file content: {e}"
            )
            
    # Apply our new, single, robust cleaning function
    cleaned_text = robust_text_cleaning(raw_text)
    
    is_valid, validation_message = validate_cleaned_content(cleaned_text)
    if not is_valid:
        logger.warning(f"Content validation failed for {file.filename}: {validation_message}")
        # Return the raw text if cleaning makes it invalid
        return raw_text
        
    logger.info(f"Successfully cleaned {file.filename}: {len(raw_text)} -> {len(cleaned_text)} chars")
    return cleaned_text
@router.post("/forensic/compare-texts")
async def compare_texts(
    data: dict = Body(...),
    model_manager: ModelManager = Depends(get_model_manager),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Compare two texts for stylometric similarity without using a pre-built corpus."""
    try:
        text1 = data.get("text1")
        text2 = data.get("text2")
        text1_label = data.get("text1_label", "Text 1")
        text2_label = data.get("text2_label", "Text 2")
        
        if not text1 or not text2:
            raise HTTPException(status_code=400, detail="Both text1 and text2 are required")
        
        if len(text1.strip()) < 50 or len(text2.strip()) < 50:
            raise HTTPException(status_code=400, detail="Both texts must be at least 50 characters for meaningful analysis")
        
        logger.info(f"🔍 [Forensic] Comparing two texts directly")
        
        # Extract style vectors
        vector1 = forensic_service.extract_style_vector(text1)
        vector2 = forensic_service.extract_style_vector(text2)
        
        # Compare the vectors
        similarity = forensic_service.compare_styles(text1, [vector2])
        
        # Generate comparison report
        comparison_result = {
            "text1_label": text1_label,
            "text2_label": text2_label,
            "text1_preview": text1[:200] + "..." if len(text1) > 200 else text1,
            "text2_preview": text2[:200] + "..." if len(text2) > 200 else text2,
            "similarity_scores": {
                "overall_similarity": round(similarity.overall_score, 3),
                "lexical_similarity": round(similarity.lexical_score, 3),
                "syntactic_similarity": round(similarity.syntactic_score, 3),
                "semantic_similarity": round(similarity.semantic_score, 3),
                "stylistic_similarity": round(similarity.stylistic_score, 3),
                "confidence": round(similarity.confidence, 3)
            },
            "interpretation": forensic_service._interpret_similarity_score(similarity.overall_score),
            "detailed_breakdown": similarity.breakdown,
            "style_comparison": {
                "text1_features": {
                    "avg_word_length": round(vector1.avg_word_length, 2),
                    "avg_sentence_length": round(vector1.avg_sentence_length, 2),
                    "vocab_richness": round(vector1.vocab_richness, 3),
                    "question_ratio": round(vector1.question_ratio, 3),
                    "exclamation_ratio": round(vector1.exclamation_ratio, 3)
                },
                "text2_features": {
                    "avg_word_length": round(vector2.avg_word_length, 2),
                    "avg_sentence_length": round(vector2.avg_sentence_length, 2),
                    "vocab_richness": round(vector2.vocab_richness, 3),
                    "question_ratio": round(vector2.question_ratio, 3),
                    "exclamation_ratio": round(vector2.exclamation_ratio, 3)
                }
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "comparison": comparison_result
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error in text comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forensic/available-figures")
async def get_available_figures(forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)):
    """Get list of public figures with available corpora."""
    try:
        cache_dir = Path(forensic_service.cache_dir)
        
        if not cache_dir.exists():
            return {"figures": [], "count": 0}
        
        figures = []
        
        for cache_file in cache_dir.glob("*_corpus.pkl"):
            try:
                # Extract person name from filename
                person_name = cache_file.stem.replace("_corpus", "").replace("_", " ").title()
                
                # Load corpus to get stats
                corpus = forensic_service._load_cached_corpus(person_name)
                
                if corpus:
                    platform_breakdown = {}
                    for doc in corpus:
                        platform_breakdown[doc.platform] = platform_breakdown.get(doc.platform, 0) + 1
                    
                    figures.append({
                        "name": person_name,
                        "corpus_size": len(corpus),
                        "platforms": list(platform_breakdown.keys()),
                        "platform_breakdown": platform_breakdown,
                        "last_updated": cache_file.stat().st_mtime
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing cache file {cache_file}: {e}")
                continue
        
        # Sort by corpus size (descending)
        figures.sort(key=lambda x: x["corpus_size"], reverse=True)
        
        return {
            "figures": figures,
            "count": len(figures)
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error listing available figures: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/forensic/corpus/{person_name}")
async def delete_corpus(person_name: str, forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)):
    """Delete a person's cached corpus."""
    try:
        cache_file = forensic_service.cache_dir / f"{person_name.lower().replace(' ', '_')}_corpus.pkl"
        
        if not cache_file.exists():
            raise HTTPException(status_code=404, detail=f"No corpus found for {person_name}")
        
        cache_file.unlink()
        
        return {
            "status": "success",
            "message": f"Corpus for {person_name} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error deleting corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forensic/batch-analyze-files")
async def batch_analyze_files(
    files: List[UploadFile] = File(...),
    person_name: str = Query(None, description="Public figure to compare against"),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Analyze multiple files in batch."""
    try:
        if len(files) > 10000:
            raise HTTPException(status_code=400, detail="Maximum 10,000 files allowed")
        
        results = []
        combined_content = []
        
        for file in files:
            try:
                content = await process_uploaded_file_with_cleaning(file)
                combined_content.append(f"=== {file.filename} ===\n{content}")
                
                logger.info(f"🧹 [Forensic] Cleaned {file.filename}: {len(content)} characters")
                
                # Individual file analysis
                if person_name:
                    corpus = forensic_service._load_cached_corpus(person_name)
                    if corpus:
                        similarity_scores = forensic_service.analyze_authorship(content, corpus)
                        results.append({
                            "file_name": file.filename,
                            "similarity_score": similarity_scores.overall_similarity,
                            "interpretation": forensic_service._interpret_similarity_scores(similarity_scores),
                            "cleaned_length": len(content),
                            "detailed_scores": {
                                "lexical_similarity": similarity_scores.lexical_similarity,
                                "syntactic_similarity": similarity_scores.syntactic_similarity,
                                "semantic_similarity": similarity_scores.semantic_similarity,
                                "stylistic_similarity": similarity_scores.stylistic_similarity
                            }
                        })
                
            except Exception as e:
                logger.warning(f"Failed to process file {file.filename}: {e}")
                results.append({
                    "file_name": file.filename,
                    "error": str(e)
                })
        
        # Combined analysis
        full_content = "\n\n".join(combined_content)
        
        if person_name and full_content.strip():
            corpus = forensic_service._load_cached_corpus(person_name)
            if corpus:
                combined_similarity = forensic_service.analyze_authorship(full_content, corpus)
                
                return {
                    "status": "success",
                    "batch_analysis": {
                        "files_processed": len(files),
                        "person_analyzed": person_name,
                        "combined_similarity": {
                            "overall_similarity": combined_similarity.overall_similarity,
                            "lexical_similarity": combined_similarity.lexical_similarity,
                            "syntactic_similarity": combined_similarity.syntactic_similarity,
                            "semantic_similarity": combined_similarity.semantic_similarity,
                            "stylistic_similarity": combined_similarity.stylistic_similarity
                        },
                        "individual_results": results,
                        "interpretation": forensic_service._interpret_similarity_scores(combined_similarity),
                        "confidence_level": forensic_service._calculate_confidence(combined_similarity),
                        "total_content_length": len(full_content)
                    },
                    "analysis_timestamp": datetime.now().isoformat()
                }
        
        # Feature extraction for combined content
        if full_content.strip():
            style_vector = forensic_service.extract_style_vector(full_content)
            
            return {
                "status": "success",
                "batch_features": {
                    "files_processed": len(files),
                    "combined_word_count": len(full_content.split()),
                    "combined_character_count": len(full_content),
                    "lexical_diversity": round(style_vector.vocab_richness, 3),
                    "avg_sentence_length": round(style_vector.avg_sentence_length, 2),
                    "stylistic_markers": {
                        "question_ratio": round(style_vector.question_ratio, 3),
                        "exclamation_ratio": round(style_vector.exclamation_ratio, 3),
                        "passive_voice_ratio": round(style_vector.passive_voice_ratio, 3)
                    },
                    "individual_results": [r for r in results if "error" not in r],
                    "detailed_features": {
                        "lexical_features": {
                            "avg_word_length": round(style_vector.avg_word_length, 2),
                            "vocab_richness": round(style_vector.vocab_richness, 3),
                            "hapax_legomena_ratio": round(style_vector.hapax_legomena_ratio, 3)
                        },
                        "syntactic_features": {
                            "pos_distribution": {k: round(v, 3) for k, v in style_vector.pos_distribution.items()},
                            "sentence_complexity": round(style_vector.sentence_complexity, 2)
                        },
                        "stylistic_features": {
                            "modal_verb_usage": round(style_vector.modal_verb_usage, 3),
                            "passive_voice_ratio": round(style_vector.passive_voice_ratio, 3),
                            "question_ratio": round(style_vector.question_ratio, 3),
                            "exclamation_ratio": round(style_vector.exclamation_ratio, 3)
                        }
                    }
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        raise HTTPException(status_code=400, detail="No valid content found in uploaded files")
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error in batch file analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forensic/preview-cleaning")
async def preview_text_cleaning(file: UploadFile = File(...)):
    """Preview how text cleaning affects a file (for debugging/testing)."""
    try:
        # Get raw content
        file_extension = file.filename.split('.')[-1].lower()
        raw_content = await file.read()
        raw_text = raw_content.decode('utf-8')
        
        # Get cleaned content
        await file.seek(0)  # Reset file pointer
        cleaned_text = await process_uploaded_file_with_cleaning(file)
        
        # Validation info
        is_valid, validation_message = validate_cleaned_content(cleaned_text)
        
        # Calculate what was removed
        removed_percentage = round((len(raw_text) - len(cleaned_text)) / len(raw_text) * 100, 1) if len(raw_text) > 0 else 0
        
        return {
            "status": "success",
            "file_name": file.filename,
            "file_type": file_extension,
            "raw_stats": {
                "character_count": len(raw_text),
                "word_count": len(raw_text.split()),
                "line_count": len(raw_text.split('\n'))
            },
            "cleaned_stats": {
                "character_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "line_count": len(cleaned_text.split('\n'))
            },
            "validation": {
                "is_valid": is_valid,
                "message": validation_message
            },
            "preview": {
                "raw_sample": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
                "cleaned_sample": cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
            },
            "reduction_percentage": removed_percentage,
            "content_removed": len(raw_text) - len(cleaned_text)
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error previewing cleaning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forensic/batch-analyze")
async def batch_analyze_statements(
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Analyze multiple statements against a corpus in batch."""
    try:
        statements = data.get("statements", [])
        person_name = data.get("person_name")
        
        if not statements or not person_name:
            raise HTTPException(status_code=400, detail="Both statements list and person_name are required")
        
        if len(statements) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 statements per batch")
        
        # Process batch analysis in background
        async def batch_analysis_task():
            results = []
            
            for i, statement in enumerate(statements):
                try:
                    analysis = forensic_service.analyze_statement(statement, person_name)
                    results.append({
                        "index": i,
                        "statement": statement[:100] + "..." if len(statement) > 100 else statement,
                        "analysis": analysis
                    })
                    logger.info(f"🔍 [Forensic] Batch analysis {i+1}/{len(statements)} completed")
                    
                except Exception as e:
                    logger.error(f"❌ [Forensic] Error in batch item {i}: {e}")
                    results.append({
                        "index": i,
                        "statement": statement[:100] + "..." if len(statement) > 100 else statement,
                        "error": str(e)
                    })
            
            # Cache batch results
            batch_id = hashlib.md5(f"{person_name}_{time.time()}".encode()).hexdigest()[:8]
            batch_cache_file = forensic_service.cache_dir / f"batch_{batch_id}.json"
            
            with open(batch_cache_file, 'w') as f:
                json.dump({
                    "person_name": person_name,
                    "timestamp": datetime.now().isoformat(),
                    "results": results
                }, f, indent=2)
            
            logger.info(f"✅ [Forensic] Batch analysis completed for {person_name}: {len(results)} statements")
        
        background_tasks.add_task(batch_analysis_task)
        
        return {
            "status": "processing",
            "message": f"Batch analysis of {len(statements)} statements started for {person_name}",
            "statements_count": len(statements),
            "person_name": person_name
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error in batch analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forensic/corpus-preview/{person_name}")
async def get_corpus_preview(person_name: str, limit: int = 10, forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)):
    """Get a preview of documents in a person's corpus."""
    try:
        corpus = forensic_service._load_cached_corpus(person_name)
        
        if not corpus:
            raise HTTPException(status_code=404, detail=f"No corpus found for {person_name}")
        
        # Create preview of documents
        preview_docs = []
        for i, doc in enumerate(corpus[:limit]):
            preview_docs.append({
                "index": i,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "platform": doc.platform,
                "date": doc.date.isoformat() if doc.date else None,
                "source_url": doc.source_url,
                "title": doc.title,
                "word_count": len(doc.content.split()),
                "char_count": len(doc.content)
            })
        
        return {
            "status": "success",
            "person_name": person_name,
            "total_documents": len(corpus),
            "preview_count": len(preview_docs),
            "documents": preview_docs
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error getting corpus preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forensic/extract-features")
async def extract_stylometric_features(data: dict = Body(...), forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)):
    """Extract detailed stylometric features from a text."""
    try:
        text = data.get("text")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if len(text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Text too short for feature extraction (minimum 20 characters)")
        
        logger.info(f"🔍 [Forensic] Extracting features from text ({len(text)} chars)")
        
        # Extract comprehensive style vector
        style_vector = forensic_service.extract_style_vector(text)
        
        # Convert to serializable format
        features = {
            "lexical_features": {
                "avg_word_length": round(style_vector.avg_word_length, 2),
                "avg_sentence_length": round(style_vector.avg_sentence_length, 2),
                "vocab_richness": round(style_vector.vocab_richness, 3),
                "hapax_legomena_ratio": round(style_vector.hapax_legomena_ratio, 3),
                "yule_k": round(style_vector.yule_k, 2)
            },
            "syntactic_features": {
                "pos_distribution": {k: round(v, 3) for k, v in style_vector.pos_distribution.items()},
                "sentence_complexity": round(style_vector.sentence_complexity, 2)
            },
            "stylistic_features": {
                "modal_verb_usage": round(style_vector.modal_verb_usage, 3),
                "passive_voice_ratio": round(style_vector.passive_voice_ratio, 3),
                "question_ratio": round(style_vector.question_ratio, 3),
                "exclamation_ratio": round(style_vector.exclamation_ratio, 3),
                "capitalization_ratio": round(style_vector.capitalization_ratio, 3)
            },
            "punctuation_features": {k: round(v, 4) for k, v in style_vector.punctuation_ratios.items()},
            "function_word_features": {k: round(v, 3) for k, v in style_vector.function_word_ratios.items()},
            "sentiment_features": {k: round(v, 3) for k, v in style_vector.sentiment_scores.items()},
            "text_statistics": {
                "character_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(text.split('.')),
                "paragraph_count": len(text.split('\n\n'))
            }
        }
        
        return {
            "status": "success",
            "features": features,
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error extracting features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
def robust_text_cleaning(content: str) -> str:
    """
    A simplified and more robust text cleaning function that avoids complex regex.
    """
    # Remove URLs
    content = re.sub(r'http[s]?://\S+', '', content)
    
    # Remove email addresses
    content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', content)

    # Remove bracketed metadata like [APPLAUSE] or (inaudible)
    content = re.sub(r'\[.*?\]', '', content)
    content = re.sub(r'\(.*?\)', '', content)
    
    # Remove speaker annotations like "TRUMP:" or "MODERATOR:"
    content = re.sub(r'^[A-Z\s]+:', '', content, flags=re.MULTILINE)

    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n\s*\n+', '\n\n', content)

    return content.strip()
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



def remove_common_metadata(content: str) -> str:
    """Remove common metadata patterns found across all file types."""
    
    # Remove attribution lines
    attribution_patterns = [
        r'^.*(?:said|stated|remarked|declared|announced).*$',
        r'^.*(?:according to|as reported by|source:).*$',
        r'^.*(?:transcript|remarks|speech) (?:by|from|of).*$',
        r'^\s*-+\s*$',  # Horizontal lines
        r'^\s*=+\s*$',  # Equal sign lines
    ]
    
    for pattern in attribution_patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove bracketed metadata
    metadata_brackets = [
        r'\[.*(?:applause|laughter|cheering|booing|interruption|inaudible).*\]',
        r'\(.*(?:applause|laughter|cheering|booing|interruption|inaudible).*\)',
        r'\[.*(?:date|time|location|venue).*\]',
        r'\[.*(?:begin|end) (?:transcript|recording).*\]',
    ]
    
    for pattern in metadata_brackets:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Remove stage directions and speaker annotations
    stage_directions = [
        r'^[A-Z\s]+:',  # Speaker names like "TRUMP:" or "THE PRESIDENT:"
        r'^\s*(?:MODERATOR|INTERVIEWER|REPORTER|AUDIENCE MEMBER):.*$',
        r'^\s*\[.*\]\s*$',  # Lines that are just bracketed content
        r'^\s*\(.*\)\s*$',  # Lines that are just parenthetical content
    ]
    
    for pattern in stage_directions:
        content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove question/answer markers that aren't the actual content
    content = re.sub(r'^Q[:\.]?\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'^A[:\.]?\s*', '', content, flags=re.MULTILINE)
    
    return content

def clean_whitespace(content: str) -> str:
    """Clean up whitespace and formatting issues."""
    
    # Replace multiple whitespace with single space
    content = re.sub(r'\s+', ' ', content)
    
    # Remove empty lines and excessive line breaks
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Strip leading/trailing whitespace
    content = content.strip()
    
    return content

def validate_cleaned_content(content: str, min_length: int = 50) -> Tuple[bool, str]:
    """
    Validate that the cleaned content is suitable for forensic analysis.
    Returns (is_valid, reason)
    """
    
    if len(content.strip()) < min_length:
        return False, f"Content too short after cleaning ({len(content)} chars)"
    
    # Count actual words vs potential metadata
    words = content.split()
    if len(words) < 10:
        return False, f"Too few words after cleaning ({len(words)} words)"
    
    # Check for reasonable sentence structure
    sentences = content.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    if avg_sentence_length < 3:
        return False, "Content appears to be metadata or fragmented text"
    
    # Check for excessive metadata markers
    metadata_ratio = len(re.findall(r'[\[\(\{].*?[\]\)\}]', content)) / len(words)
    if metadata_ratio > 0.1:  # More than 10% metadata markers
        return False, "Content contains too much metadata"
    
    return True, "Content validated successfully"

@router.post("/forensic/analyze")
async def analyze_statement_endpoint(
    data: dict = Body(...), 
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Start an analysis task in a real background thread."""
    try:
        statement = data.get("statement")
        person_name = data.get("person_name")
        
        if not statement or not person_name:
            raise HTTPException(status_code=400, detail="Both statement and person_name are required")
        
        task_id = str(uuid.uuid4())
        logger.info(f"🔍 [Forensic] Starting analysis task {task_id} for {person_name}")
        
        # Run in actual background thread instead of FastAPI background_tasks
        def run_analysis():
            try:
                asyncio.run(forensic_service.analyze_statement(task_id, statement, person_name))
            except Exception as e:
                logger.error(f"Analysis task {task_id} failed: {e}")
        
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()
        
        logger.info(f"🚀 Task {task_id} started in thread, returning immediately")
        
        # This should now return immediately
        return {"status": "processing_started", "task_id": task_id}
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error starting analysis task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/forensic/progress/{task_id}")
async def get_analysis_progress(task_id: str, forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)):
    """Get the progress of a forensic analysis task."""
    progress = forensic_service.get_progress(task_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    return progress
# --- Lifespan Function ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application lifespan startup...")
    global SINGLE_GPU_MODE
    # Read environment variables set by launch.py or batch script
    # Ensure these env vars are correctly set by your launch mechanism
    default_gpu = int(os.environ.get("GPU_ID", 0))
    port = int(os.environ.get("PORT", 8000 if default_gpu == 0 else 8001))
    tts_port = int(os.environ.get("TTS_PORT", 8002))  # TTS service port
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
        gpu_usage_mode = 'unified_model'  # ✅ Add this default
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

    logger.info(f"Lifespan: Running on Port {port}, Default GPU {default_gpu}, GPU Mode: {gpu_usage_mode}")

    # --- CRITICAL FIX: Set CUDA_VISIBLE_DEVICES at startup ---
    if gpu_usage_mode == "split_services":
        # In split mode, isolate this server instance to its assigned GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(default_gpu)
        logging.info(f"✅ [Split Mode] Set CUDA_VISIBLE_DEVICES to {default_gpu}")
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        # In unified mode, ensure the environment variable is UNSET
        # so that llama.cpp can see all available GPUs.
        del os.environ["CUDA_VISIBLE_DEVICES"]
        logging.info(f"✅ [Unified Mode] Unset CUDA_VISIBLE_DEVICES to enable multi-GPU visibility.")
    # --- END CRITICAL FIX ---

    # Initialize ModelManager and store in app state
    try:
        app.state.model_manager = ModelManager(gpu_usage_mode=gpu_usage_mode)
        app.state.default_gpu = default_gpu
        app.state.port = port
        logger.info(f"Server starting on port {port} with default GPU {default_gpu}")
        
        # Initialize Devstral service with model manager
        devstral_service.model_manager = app.state.model_manager
        logger.info("✅ Devstral service initialized")
        
    except Exception as init_err:
        logger.error(f"FATAL: Failed to initialize ModelManager: {init_err}", exc_info=True)
        raise init_err

    # Initialize SD Manager (add this after your ModelManager initialization)
    try:
        app.state.sd_manager = SDManager()
        logger.info("SD Manager initialized")
    except Exception as sd_err:
        logger.error(f"Failed to initialize SD Manager: {sd_err}")
        app.state.sd_manager = None
      
    # === TTS SERVICE INTEGRATION ===
    # TTS now runs as a separate service on port 8002 to avoid resource conflicts
    # The main backend focuses on model inference, while TTS runs independently
    
    if port == 8000:  # Main backend
        try:
            logger.info("🔗 Checking TTS service availability...")
            import httpx
            import asyncio
            
            # Wait a moment for TTS service to start
            await asyncio.sleep(2)
            
            # Check if TTS service is running
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(f"http://localhost:{tts_port}/health", timeout=5.0)
                    if response.status_code == 200:
                        tts_status = response.json()
                        logger.info(f"✅ TTS service is running: {tts_status}")
                    else:
                        logger.warning(f"⚠️ TTS service responded with status {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ TTS service not yet available: {e}")
                    logger.info(f"📌 TTS service will start independently on port {tts_port}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not check TTS service: {e}")
    
    logger.info(f"📌 Main backend on port {port} - TTS runs separately on port {tts_port}")
    
    # Initialize TTS client for forwarding requests to TTS service
    try:
        app.state.tts_client = TTSClient(base_url=f"http://localhost:{tts_port}")
        logger.info(f"✅ TTS client initialized for forwarding to TTS service on port {tts_port}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize TTS client: {e}")
        app.state.tts_client = None
   
    # Initialize RAG system
    try:
        rag_available = rag_utils.initialize_rag_system()
        app.state.rag_available = rag_available
        logger.info(f"RAG system initialization {'successful' if rag_available else 'skipped or failed'}")
    except Exception as rag_error:
        logger.error(f"Error initializing RAG system: {rag_error}", exc_info=True)
        app.state.rag_available = False
    
    # Initialize Forensic Linguistics Service
    try:
        logger.info("Initializing Forensic Linguistics Service...")
        app.state.forensic_service = ForensicLinguisticsService(
            model_manager=app.state.model_manager,
            cache_dir="./forensic_cache"
        )
        logger.info("✅ Forensic Linguistics Service initialized (no embedding model loaded - use Settings to load manually)")
    except Exception as forensic_error:
        logger.error(f"Error initializing Forensic Linguistics Service: {forensic_error}", exc_info=True)
        app.state.forensic_service = None
        
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
@router.post("/stt/load-engine")
async def load_stt_engine_endpoint(data: dict = Body(...)):
    """Manually load an STT engine on a specific GPU."""
    try:
        engine = data.get("engine", "whisper")
        gpu_id = data.get("gpu_id", 0) # Default to GPU 0 (3090) for peripheral STT service
        
        from . import stt_service
        # The STT service already has the device initialized
        # No need to re-detect device

        if engine == "whisper":
            stt_service.load_whisper_model()
        elif engine == "parakeet":
            stt_service.load_parakeet_model()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown STT engine: {engine}")

        return {"status": "success", "message": f"{engine} loaded on GPU {gpu_id}"}
    except Exception as e:
        logger.error(f"Error loading STT engine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts/load-engine")
async def load_tts_engine_endpoint(data: dict = Body(...)):
    """Forward TTS engine loading request to TTS service on port 8002."""
    try:
        engine = data.get("engine", "kokoro")
        gpu_id = data.get("gpu_id", 0) # Default to GPU 0 (3090) for peripheral TTS service
        
        # Forward request to TTS service
        if hasattr(request.app.state, 'tts_client') and request.app.state.tts_client:
            response = await request.app.state.tts_client.load_engine(engine=engine, gpu_id=gpu_id)
            return response
        else:
            raise HTTPException(status_code=503, detail="TTS service not available")

    except Exception as e:
        logger.error(f"Error loading TTS engine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DEVSTRAL 2 CODE EDITOR ENDPOINTS
# ============================================================================

@router.post("/devstral/chat")
async def devstral_chat_endpoint(
    request: Request,
    data: dict = Body(...),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Main chat endpoint for Devstral Small 2 with tool calling support.
    This is the primary endpoint for the code editor.
    
    Supports two modes:
    1. External API mode (koboldcpp, ollama) - set DEVSTRAL_EXTERNAL=true
    2. Direct model mode - requires a model loaded via model_manager
    """
    from .devstral_service import EXTERNAL_LLM_ENABLED, EXTERNAL_LLM_URL
    
    try:
        messages = data.get("messages", [])
        working_dir = data.get("working_dir", devstral_service.base_dir)
        temperature = data.get("temperature", 0.15)
        max_tokens = data.get("max_tokens", 4096)
        image_base64 = data.get("image_base64")  # Optional vision input
        auto_execute = data.get("auto_execute", True)  # Auto-execute tool calls
        
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")
        
        # Check for model parameter (from frontend selector)
        requested_model = data.get("model")
        logger.info(f"📨 Devstral Chat Request. Model: {requested_model}")

        api_config = None
        model_instance = None
        model_name = None
        
        # 1. Check if requested model is an API endpoint (e.g. OpenRouter)
        is_api = requested_model and is_api_endpoint(requested_model)
        logger.info(f"❓ Is API Endpoint? {requested_model} -> {is_api}")

        if is_api:
            endpoint_config = get_configured_endpoint(requested_model)
            logger.info(f"⚙️ Config lookup result: {endpoint_config is not None}")
            
            if endpoint_config:
                logger.info(f"🌐 Devstral requested API model: {requested_model}")
                api_config = {
                    "url": endpoint_config.get("url"),
                    "api_key": endpoint_config.get("api_key"),
                    "model": endpoint_config.get("model")
                }
                model_name = requested_model
                logger.info(f"✅ Found API config for {requested_model}")
            else:
                logger.warning(f"⚠️ API endpoint {requested_model} not found in settings")

        if not api_config:
            # 2. Try to find a loaded local model matching the request or default to Devstral
            
            # First pass: try to match requested_model exactly or loosely
            if requested_model:
                for key, model_info in model_manager.loaded_models.items():
                    name, gpu_id = key
                    if requested_model.lower() in name.lower():
                        model_instance = model_manager.get_model(name, gpu_id)
                        model_name = name
                        logger.info(f"🔧 Found matching local model: {name} (requested: {requested_model})")
                        break
            
            # Second pass: Any Devstral model
            if not model_instance:
                for key, model_info in model_manager.loaded_models.items():
                    name, gpu_id = key
                    if devstral_service.is_devstral_model(name):
                        model_instance = model_manager.get_model(name, gpu_id)
                        model_name = name
                        logger.info(f"🔧 Using default Devstral model: {name}")
                        break

            # 3. Last Resort: Use Legacy External API if enabled AND no local model found
            if not model_instance and EXTERNAL_LLM_ENABLED:
                logger.info(f"🌐 Using legacy external LLM API (fallback): {EXTERNAL_LLM_URL}")
                model_name = "external-api"
            
            # 4. Fallback: Any available model
            if not model_instance and not EXTERNAL_LLM_ENABLED and model_manager.loaded_models:
                 key = next(iter(model_manager.loaded_models.keys()))
                 name, gpu_id = key
                 model_instance = model_manager.get_model(name, gpu_id)
                 model_name = name
                 logger.info(f"🔧 Using fallback model (any): {name}")

            if not model_instance and not EXTERNAL_LLM_ENABLED:
                raise HTTPException(status_code=400, detail="No model loaded. Please load a model or check external API settings.")
        
        # Add system prompt if not present
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, {
                'role': 'system',
                'content': devstral_service.get_system_prompt(working_dir)
            })
        
        # Get response from model with tools
        # Get response from model with tools
        response = await devstral_service.chat_with_tools(
            messages=messages,
            model_instance=model_instance,
            working_dir=working_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            image_base64=image_base64,
            api_config=api_config  # Pass the new config
        )
        
        # If auto_execute is enabled and we have tool calls, execute them
        if auto_execute and response.get('choices'):
            choice = response['choices'][0]
            message = choice.get('message', {})
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                tool_results = []
                for tool_call in tool_calls:
                    func = tool_call.get('function', {})
                    tool_name = func.get('name')
                    try:
                        arguments = json.loads(func.get('arguments', '{}'))
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    logger.info(f"🔧 Auto-executing tool: {tool_name}")
                    success, result = await devstral_service.execute_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        base_dir=working_dir
                    )
                    
                    tool_results.append({
                        'tool_call_id': tool_call.get('id'),
                        'name': tool_name,
                        'success': success,
                        'result': result
                    })
                
                # Add tool results to response
                response['tool_results'] = tool_results
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Devstral chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devstral/execute-tool")
async def devstral_execute_tool_endpoint(data: dict = Body(...)):
    """Execute a specific tool call manually."""
    try:
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})
        working_dir = data.get("working_dir", devstral_service.base_dir)
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")
        
        logger.info(f"🔧 Executing tool: {tool_name} with args: {arguments}")
        
        success, result = await devstral_service.execute_tool(
            tool_name=tool_name,
            arguments=arguments,
            base_dir=working_dir
        )
        
        return {
            "success": success,
            "result": result,
            "tool_name": tool_name
        }
        
    except Exception as e:
        logger.error(f"❌ Tool execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devstral/tools")
async def get_devstral_tools():
    """Get the list of available tools for Devstral."""
    return {
        "tools": devstral_service.get_tools_definition(),
        "version": "2.0",
        "model": "Devstral Small 2 24B"
    }


@router.get("/devstral/status")
async def get_devstral_status(model_manager: ModelManager = Depends(get_model_manager)):
    """Get Devstral model status and capabilities."""
    from .devstral_service import EXTERNAL_LLM_ENABLED, EXTERNAL_LLM_URL
    
    try:
        devstral_loaded = False
        devstral_model = None
        devstral_version = None
        using_external = EXTERNAL_LLM_ENABLED
        
        if using_external:
            # Check if external API is reachable
            import httpx
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{EXTERNAL_LLM_URL.rstrip('/chat/completions')}/models")
                    devstral_loaded = response.status_code == 200
                    devstral_model = "external-api"
                    devstral_version = "2"  # Assume Devstral 2 for external
            except:
                devstral_loaded = False
        else:
            for key, model_info in model_manager.loaded_models.items():
                name, gpu_id = key
                if devstral_service.is_devstral_model(name):
                    devstral_loaded = True
                    devstral_model = name
                    devstral_version = "2" if devstral_service.is_devstral_2(name) else "1"
                    break
        
        return {
            "devstral_loaded": devstral_loaded,
            "model_name": devstral_model,
            "version": devstral_version,
            "external_api": using_external,
            "external_url": EXTERNAL_LLM_URL if using_external else None,
            "capabilities": {
                "tool_calling": True,
                "vision": devstral_version == "2",
                "context_length": 256000 if devstral_version == "2" else 32768,
            },
            "working_directory": devstral_service.base_dir
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting Devstral status: {e}")
        return {"error": str(e)}
@router.post("/forensic/initialize-gme")
async def initialize_gme_endpoint(
    request: Request,
    data: dict = Body(...),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Initialize GME model for enhanced forensic embeddings"""
    try:
        model_name = data.get("model_name", "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")
        gpu_id = data.get("gpu_id", 0)
        
        logger.info(f"🔍 [Forensic] Initializing GME: {model_name} on GPU {gpu_id}")
        
        success = await forensic_service.initialize_gme_model(model_name, gpu_id)
        
        if success:
            status = forensic_service.get_embedding_status()
            return {
                "status": "success",
                "message": f"GME model {model_name} initialized on GPU {gpu_id}",
                "embedding_status": status
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize GME model")
            
    except Exception as e:
        logger.error(f"❌ [Forensic] GME initialization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forensic/embedding-status")
async def get_embedding_status(forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)):
    """Get current status of all embedding models"""
    try:
        return forensic_service.get_embedding_status()
    except Exception as e:
        logger.error(f"❌ [Forensic] Error getting embedding status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Log which backend instance received this request
        backend_gpu_id = request.app.state.default_gpu
        logger.info(f"📡 Backend instance (GPU {backend_gpu_id}) received load request for GPU {gpu_id}")
        
        # When CUDA_VISIBLE_DEVICES is set, backend only sees one GPU as device 0
        # So we need to accept the request if it matches the backend's physical GPU
        # But we'll normalize to device 0 for actual loading
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            # Backend is restricted to one GPU - validate it matches
            if gpu_id != backend_gpu_id:
                error_msg = (
                    f"GPU routing error: Requested GPU {gpu_id} but this backend instance "
                    f"(port {request.app.state.port}) is configured for GPU {backend_gpu_id}. "
                    f"Request should be routed to {'PRIMARY_API_URL (port 8000)' if gpu_id == 0 else 'SECONDARY_API_URL (port 8001)'}."
                )
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            # Normalize to device 0 for actual loading since CUDA_VISIBLE_DEVICES restricts visibility
            actual_device_id = 0
            logger.info(f"✅ Request validated. Will load on device {actual_device_id} (maps to physical GPU {gpu_id})")
        else:
            # Backend can see all GPUs - use requested GPU directly
            actual_device_id = gpu_id
            logger.info(f"✅ Backend can see all GPUs. Will load on GPU {gpu_id}")
        
        # Check VRAM availability before loading
        try:
            import torch
            if torch.cuda.is_available():
                # Use actual_device_id (0 when restricted, or gpu_id when not)
                vram_check_gpu = actual_device_id
                logger.info(f"📊 Checking VRAM on device {vram_check_gpu} (physical GPU {gpu_id})")
                
                # Get available GPU memory
                mem_free, mem_total = torch.cuda.mem_get_info(vram_check_gpu)
                mem_free_gb = mem_free / (1024**3)
                mem_total_gb = mem_total / (1024**3)
                mem_used_gb = (mem_total - mem_free) / (1024**3)
                
                logger.info(f"📊 Physical GPU {gpu_id} (device {vram_check_gpu}) VRAM: {mem_free_gb:.2f}GB free / {mem_total_gb:.2f}GB total ({(mem_used_gb/mem_total_gb)*100:.1f}% used)")
                
                # Warn if VRAM is nearly full (less than 2GB free)
                if mem_free_gb < 2.0:
                    warning_msg = (
                        f"⚠️ Warning: GPU {gpu_id} has only {mem_free_gb:.2f}GB free VRAM. "
                        f"Model loading may fail or fall back to system RAM. "
                        f"Consider unloading other models first."
                    )
                    logger.warning(warning_msg)
                    # Don't block, but log the warning
                    
                # Error if VRAM is critically low (less than 500MB free)
                if mem_free_gb < 0.5:
                    error_msg = (
                        f"❌ Insufficient VRAM: GPU {gpu_id} has only {mem_free_gb:.2f}GB free VRAM. "
                        f"Cannot load model. Please unload other models first."
                    )
                    logger.error(error_msg)
                    raise HTTPException(status_code=507, detail=error_msg)
        except ImportError:
            logger.warning("PyTorch not available - cannot check VRAM")
        except RuntimeError as e:
            # GPU not accessible (e.g., CUDA_VISIBLE_DEVICES restriction)
            error_msg = f"GPU {gpu_id} is not accessible from this backend instance: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            logger.warning(f"Could not check VRAM: {e}")
        
        logger.info(f"🚀 Loading {model_name} for purpose '{purpose}' on physical GPU {gpu_id} (device {actual_device_id})")
        
        # Pass the original gpu_id - load_model_for_purpose will handle device normalization
        await model_manager.load_model_for_purpose(
            purpose=purpose,
            model_name=model_name, 
            gpu_id=gpu_id,  # Pass original physical GPU ID
            context_length=context_length
        )
        
        # Override the tracking to use the physical GPU ID (for routing)
        # The load_model_for_purpose might have set it to device 0, but we want to track the physical GPU
        if hasattr(model_manager, 'model_purposes') and model_manager.model_purposes.get(purpose):
            model_manager.model_purposes[purpose]['gpu_id'] = gpu_id
            logger.info(f"✅ Tracked {model_name} as {purpose} on physical GPU {gpu_id} (loaded on device {actual_device_id})")
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded as {purpose} on GPU {gpu_id}"
        }
        
    except HTTPException:
        raise
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

@router.post("/forensic/initialize-embedding")
async def initialize_embedding_endpoint(
    request: Request,
    data: dict = Body(...),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Initialize any embedding model"""
    try:
        model_type = data.get("model_type")
        gpu_id = data.get("gpu_id", 0)

        if not model_type:
            raise HTTPException(status_code=400, detail="model_type is required")

        logger.info(f"🔍 [Forensic] Initializing {model_type} on GPU {gpu_id}")

        success = await forensic_service.initialize_embedding_model(model_type, gpu_id)

        if success:
            status = forensic_service.get_embedding_status()
            return {
                "status": "success",
                "message": f"{model_type} model initialized successfully",
                "embedding_status": status
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to initialize {model_type}")

    except Exception as e:
        logger.error(f"❌ [Forensic] Embedding initialization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/forensic/set-active-embedding-model")
async def set_active_embedding_model_endpoint(
    data: dict = Body(...),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Set the active embedding model for forensic analysis."""
    try:
        model_key = data.get("model_key")
        if not model_key:
            raise HTTPException(status_code=400, detail="model_key is required")

        success = forensic_service.set_active_embedding_model(model_key)

        if success:
            status = forensic_service.get_embedding_status()
            return {
                "status": "success", 
                "message": f"Active embedding model set to {model_key}",
                "embedding_status": status
            }
        else:
            raise HTTPException(status_code=400, detail=f"Could not set active model to {model_key}")
    except Exception as e:
        logger.error(f"Error setting active embedding model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forensic/unload-models")
async def unload_forensic_models_endpoint(
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Unload roberta and star models from memory to free VRAM"""
    try:
        success = await forensic_service.unload_forensic_models()
        
        if success:
            status = forensic_service.get_embedding_status()
            return {
                "status": "success",
                "message": "Forensic models (roberta/star) unloaded successfully",
                "embedding_status": status
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to unload forensic models")
    except Exception as e:
        logger.error(f"❌ [Forensic] Error unloading models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts/shutdown")
async def shutdown_tts_service():
    """Shutdown TTS service running on port 8002"""
    try:
        import socket
        import platform
        
        port = int(os.environ.get("TTS_PORT", 8002))
        logger.info(f"🛑 Attempting to shutdown TTS service on port {port}...")
        
        # Check if port is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result != 0:
            return {
                "status": "info",
                "message": f"TTS service on port {port} is not running"
            }
        
        # Find and kill process using port 8002
        system = platform.system()
        if system == "Windows":
            # Windows: use netstat to find PID, then taskkill
            try:
                # Find PID using netstat (Windows format)
                netstat_cmd = ["netstat", "-ano"]
                result = subprocess.run(netstat_cmd, capture_output=True, text=True, shell=True)
                lines = result.stdout.split('\n')
                
                pid = None
                for line in lines:
                    if f":{port}" in line and "LISTENING" in line.upper():
                        # Parse Windows netstat output format
                        parts = line.strip().split()
                        # PID is the last column in Windows netstat -ano output
                        if len(parts) >= 5:
                            pid = parts[-1]
                            # Validate PID is numeric
                            try:
                                int(pid)
                                break
                            except ValueError:
                                pid = None
                
                if pid:
                    # Kill the process
                    kill_cmd = ["taskkill", "/F", "/PID", pid]
                    subprocess.run(kill_cmd, capture_output=True)
                    logger.info(f"✅ TTS service process (PID: {pid}) terminated")
                    return {
                        "status": "success",
                        "message": f"TTS service on port {port} has been shut down (PID: {pid})"
                    }
                else:
                    return {
                        "status": "warning",
                        "message": f"Port {port} is in use but PID could not be determined"
                    }
            except Exception as e:
                logger.error(f"Error shutting down TTS service: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to shutdown TTS service: {str(e)}")
        else:
            # Linux/Mac: use lsof or fuser
            try:
                lsof_cmd = ["lsof", "-ti", f":{port}"]
                result = subprocess.run(lsof_cmd, capture_output=True, text=True)
                pid = result.stdout.strip()
                
                if pid:
                    subprocess.run(["kill", "-9", pid], capture_output=True)
                    logger.info(f"✅ TTS service process (PID: {pid}) terminated")
                    return {
                        "status": "success",
                        "message": f"TTS service on port {port} has been shut down (PID: {pid})"
                    }
                else:
                    return {
                        "status": "warning",
                        "message": f"Port {port} is in use but PID could not be determined"
                    }
            except Exception as e:
                logger.error(f"Error shutting down TTS service: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to shutdown TTS service: {str(e)}")
                
    except Exception as e:
        logger.error(f"❌ Error shutting down TTS service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts/restart")
async def restart_tts_service(request: Request):
    """Restart TTS service"""
    try:
        import platform
        from pathlib import Path
        
        port = int(os.environ.get("TTS_PORT", 8002))
        logger.info(f"🔄 Attempting to restart TTS service on port {port}...")
        
        # First, shutdown existing service
        try:
            await shutdown_tts_service()
            await asyncio.sleep(2)  # Wait for process to fully terminate
        except:
            pass  # Ignore errors during shutdown
        
        # Get project root (assuming main.py is in backend/app/)
        project_root = Path(__file__).parent.parent.parent
        
        # Start TTS service
        tts_script = project_root / "launch_tts.py"
        
        if not tts_script.exists():
            raise HTTPException(status_code=500, detail="TTS launch script not found")
        
        system = platform.system()
        
        # Launch TTS service in background
        if system == "Windows":
            # Use subprocess.Popen to start in background
            cmd = [sys.executable, str(tts_script)]
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            # Linux/Mac
            cmd = [sys.executable, str(tts_script)]
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Wait a moment for service to start
        await asyncio.sleep(3)
        
        # Check if service is running
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            logger.info(f"✅ TTS service restarted successfully on port {port}")
            return {
                "status": "success",
                "message": f"TTS service restarted on port {port} (PID: {process.pid})"
            }
        else:
            return {
                "status": "warning",
                "message": f"TTS service process started (PID: {process.pid}) but port {port} is not yet responding. It may still be starting up."
            }
            
    except Exception as e:
        logger.error(f"❌ Error restarting TTS service: {e}", exc_info=True)
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
    

@router.post("/system/initialize-services")
async def initialize_services_endpoint(
    request: Request,
    data: dict = Body(...),
    model_manager: ModelManager = Depends(get_model_manager)
):
    try:
        gpu_id = data.get("gpu_id", 1)
        device_str = f"cuda:{gpu_id}"
        logger.info(f"--- Manually initializing embedding services on GPU {gpu_id} ---")

        # --- ADD THIS NEW SECTION ---
        # 1. Initialize Memory Intelligence Model
        if not hasattr(request.app.state, 'similarity_model_initialized') or not request.app.state.similarity_model_initialized:
            logger.info("Initializing Memory Intelligence similarity model...")
            from . import memory_intelligence
            memory_intelligence.initialize_similarity_model(device=device_str)
            request.app.state.similarity_model_initialized = True
            logger.info("✅ Memory Intelligence similarity model initialized.")
        else:
            logger.info("Memory Intelligence similarity model is already initialized.")

        return {"status": "success", "message": "Services initialized successfully."}
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
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

# @router.post("/character/analyze-readiness")
# async def analyze_character_readiness_endpoint(
#     request: Request,
#     data: dict = Body(...),  # Expects {"messages": [...]}
#     model_manager: ModelManager = Depends(get_model_manager)
# ):
#     """Analyze conversation messages for character auto-generation readiness."""
#     try:
#         messages = data.get("messages", [])
#         lookback_count = data.get("lookback_count", 25)
#         
#         if not messages:
#             return {
#                 "status": "success", 
#                 "readiness_score": 0, 
#                 "detected_elements": [],
#                 "message": "No messages to analyze"
#             }
#         
#         logger.info(f"🎯 Analyzing character readiness for {len(messages)} messages")
#         
#         # Analyze character readiness using embeddings
#         analysis_result = character_intelligence.analyze_character_readiness(
#             messages=messages,
#             lookback_count=lookback_count
#         )
#         
#         return analysis_result
#         
#     except Exception as e:
#         logger.error(f"❌ Error in character readiness analysis: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/save-voice-preference")
async def save_voice_preference(request: dict):
    """Save voice preference to settings.json for pre-caching"""
    try:
        print(f"🔧 [Voice Preference] Received request: {request}")

        settings_dir = Path.home() / ".LiangLocal"
        settings_dir.mkdir(exist_ok=True)
        settings_path = settings_dir / "settings.json"
        print(f"🔧 [Voice Preference] Settings path: {settings_path.absolute()}")
        print(f"🔧 [Voice Preference] Path exists: {settings_path.exists()}")
        
        # Load existing settings
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            print(f"🔧 [Voice Preference] Loaded existing settings with keys: {settings.keys()}")
        else:
            settings = {}
            print("🔧 [Voice Preference] No existing settings.json, creating new")
        
        # Initialize voice_cache list if not exists
        if 'voice_cache' not in settings:
            settings['voice_cache'] = []
            print("🔧 [Voice Preference] Initialized voice_cache list")
        
        voice_entry = {
            'voice_id': request.get('voice_id'),  # Use .get() for safety
            'engine': request.get('engine', 'chatterbox')
        }
        print(f"🔧 [Voice Preference] Voice entry to add: {voice_entry}")
        
        # Check if already exists
        existing_voices = [v.get('voice_id') for v in settings['voice_cache']]
        print(f"🔧 [Voice Preference] Existing voices: {existing_voices}")
        
        # Add or update
        if voice_entry['voice_id'] not in existing_voices:
            settings['voice_cache'].append(voice_entry)
            print(f"🔧 [Voice Preference] Added new voice")
        else:
            # Update existing entry
            for i, v in enumerate(settings['voice_cache']):
                if v.get('voice_id') == voice_entry['voice_id']:
                    settings['voice_cache'][i] = voice_entry
                    print(f"🔧 [Voice Preference] Updated existing voice at index {i}")
                    break
        
        # Keep only last 5 voices
        settings['voice_cache'] = settings['voice_cache'][-5:]
        print(f"🔧 [Voice Preference] Final voice_cache: {settings['voice_cache']}")
        
        # Save settings
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"✅ [Voice Preference] Settings saved successfully to {settings_path.absolute()}")
        
        # Verify it was written
        with open(settings_path, 'r') as f:
            verify = json.load(f)
        print(f"✅ [Voice Preference] Verification - voice_cache in file: {verify.get('voice_cache', [])}")
        
        return {"status": "success", "message": "Voice preference saved"}
        
    except Exception as e:
        print(f"❌ [Voice Preference] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
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
    """Generate a character JSON from conversation using LLM (local or API)."""
    try:
        messages = data.get("messages", [])
        analysis = data.get("analysis", {})
        model_name = data.get("model_name")  # Get from frontend if provided
        gpu_id = data.get("gpu_id")  # Optional override
        use_api = data.get("use_api", False)  # Whether to use external API
        api_endpoint = data.get("api_endpoint")  # API endpoint info
        
        # Determine GPU. Default to 0 (primary chat GPU) for character creation.
        if gpu_id is None:
            gpu_id = 0
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        logger.info(f"🎨 Generating character from conversation (use_api={use_api})")
        
        # Generate character JSON using LLM
        generation_result = await character_intelligence.generate_character_json(
            model_manager=model_manager,
            messages=messages,
            character_analysis=analysis,
            model_name=model_name,
            gpu_id=gpu_id,
            single_gpu_mode=getattr(request.app.state, 'single_gpu_mode', False),
            use_api=use_api,
            api_endpoint=api_endpoint
        )
        
        return generation_result
        
    except Exception as e:
        logger.error(f"❌ Error generating character from conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def warmup_chatterbox_voices():
    """Warm up voices from settings.json"""
    import json
    from pathlib import Path
    
    try:
        # Read settings.json
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        if not settings_path.exists():
            logger.info("No settings.json found, skipping voice warmup")
            return
            
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            
        voice_cache = settings.get('voice_cache', [])
        if not voice_cache:
            logger.info("No voices in cache, skipping voice warmup")
            return
            
        logger.info(f"🔥 Warming up {len(voice_cache)} cached voices...")
        
        from . import tts_service
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        
        for voice_entry in voice_cache:
            if voice_entry.get('engine') == 'chatterbox':
                voice_id = voice_entry.get('voice_id')
                if voice_id:
                    voice_path = voices_dir / voice_id
                    if voice_path.exists():
                        try:
                            logger.info(f"🔥 Warming up voice: {voice_id}")
                            # This will trigger conditional preparation and caching
                            if hasattr(request.app.state, 'tts_client') and request.app.state.tts_client:
                                await request.app.state.tts_client.synthesize_speech(
                                    text="Warmup test",
                                    engine="chatterbox",
                                    audio_prompt_path=str(voice_path),
                                    exaggeration=0.5,
                                    cfg=0.3
                                )
                            else:
                                logger.warning("TTS client not available for warmup")
                            logger.info(f"✅ Warmed up voice: {voice_id}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to warm up voice {voice_id}: {e}")
                    else:
                        logger.warning(f"⚠️ Voice file not found: {voice_path}")
        
        logger.info("✅ Voice warmup complete")
        
    except Exception as e:
        logger.error(f"Voice warmup failed: {e}", exc_info=True)

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

@router.post("/stt/fix-parakeet-numpy")
async def fix_parakeet_numpy():
    """Force install numpy<2 to fix Parakeet/NeMo issues."""
    logger.info("Received request to force fix Parakeet NumPy dependency")
    try:
        import sys
        import subprocess
        
        logger.info("Running: pip install \"numpy<2\"")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "numpy<2"
        ])
        
        logger.info("NumPy fix applied successfully via pip")
        return {"status": "success", "message": "Successfully downgraded NumPy (numpy<2). Please restart the app if issues persist."}
    except Exception as e:
        logger.error(f"Error applying NumPy fix: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to fix NumPy: {str(e)}"}
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
    """Load a local SD model by its filename on specified GPU"""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    model_filename = data.get("model_filename")
    gpu_id = data.get("gpu_id", 0)  # Default to GPU0
    
    if not model_filename:
        raise HTTPException(status_code=400, detail="model_filename required")

    sd_model_dir = getattr(request.app.state, 'sd_model_directory', None)
    if not sd_model_dir:
        raise HTTPException(status_code=500, detail="SD model directory not configured on backend.")

    full_model_path = str(Path(sd_model_dir) / model_filename)
    
    success = sd_manager.load_model(full_model_path, gpu_id=gpu_id)
    if success:
        return {
            "status": "success", 
            "message": f"Model loaded: {full_model_path} on GPU {gpu_id}"
        }
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

    # ADD THIS: For Chatterbox, use voice as the audio_prompt_path if not explicitly set
    if engine == "chatterbox" and not audio_prompt_path and voice != "default":
        audio_prompt_path = voice
        logger.info(f"🔊 [TTS] Chatterbox mode: using voice '{voice}' as audio_prompt_path")

    # Chatterbox-specific parameters
    exaggeration = data.get("exaggeration", 0.5)
    cfg = data.get("cfg", 0.5)

    if not text:
        return JSONResponse(content={"detail": "No text provided"}, status_code=400)

    try:
        # Call tts_service directly since it's integrated
        from .tts_service import synthesize_speech
        
        audio_bytes = await synthesize_speech(
            text=text,
            voice=voice,
            engine=engine,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg=cfg
        )
        
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
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
        
        # Create a clean filename based on original name
        original_name = Path(file.filename).stem  # Remove extension
        clean_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_name = clean_name.replace(' ', '_')  # Replace spaces with underscores
        
        # Ensure filename is not empty
        if not clean_name:
            clean_name = "uploaded_voice"
        
        # Create the final filename with original extension
        unique_filename = f"{clean_name}{file_extension}"
        save_path = voices_dir / unique_filename
        
        # Handle duplicates by adding a number suffix
        counter = 1
        while save_path.exists():
            unique_filename = f"{clean_name}_{counter}{file_extension}"
            save_path = voices_dir / unique_filename
            counter += 1
        
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
        # Check which TTS engines are available
        available_engines = []
        
        # Check Kokoro availability - use the already-imported module check from tts_service
        try:
            from .tts_service import KPipeline
            kokoro_available = KPipeline is not None
        except:
            kokoro_available = False
        
        if kokoro_available:
            available_engines.append("kokoro")
        
        # Chatterbox is always available (primary engine)
        available_engines.append("chatterbox")
        
        # Kokoro voices (built-in voices)
        kokoro_voices = []
        if kokoro_available:
            # Kokoro has built-in voices - list common ones
            kokoro_voices = [
                {'id': 'af_heart', 'name': 'Heart (Female)', 'engine': 'kokoro'},
                {'id': 'af_bella', 'name': 'Bella (Female)', 'engine': 'kokoro'},
                {'id': 'af_sarah', 'name': 'Sarah (Female)', 'engine': 'kokoro'},
                {'id': 'af_nicole', 'name': 'Nicole (Female)', 'engine': 'kokoro'},
                {'id': 'am_adam', 'name': 'Adam (Male)', 'engine': 'kokoro'},
                {'id': 'am_michael', 'name': 'Michael (Male)', 'engine': 'kokoro'},
                {'id': 'bf_emma', 'name': 'Emma (British Female)', 'engine': 'kokoro'},
                {'id': 'bf_isabella', 'name': 'Isabella (British Female)', 'engine': 'kokoro'},
                {'id': 'bm_george', 'name': 'George (British Male)', 'engine': 'kokoro'},
                {'id': 'bm_lewis', 'name': 'Lewis (British Male)', 'engine': 'kokoro'},
            ]
        
        # Chatterbox voice references (uploaded files)
        chatterbox_voices = []
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        if voices_dir.exists():
            for voice_file in voices_dir.glob("*"):
                if voice_file.is_file() and voice_file.suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a'}:
                    # Handle both old UUID format and new readable format
                    if voice_file.name.startswith('voice_ref_'):
                        # Old UUID format - extract UUID part for display
                        display_name = f"Custom Voice ({voice_file.stem.replace('voice_ref_', '')[:8]}...)"
                    else:
                        # New readable format - use the actual filename
                        display_name = voice_file.stem.replace('_', ' ').title()
                    
                    chatterbox_voices.append({
                        'id': voice_file.name,
                        'name': display_name,
                        'engine': 'chatterbox',
                        'file_path': str(voice_file)
                    })
        
        return {
            "kokoro_voices": kokoro_voices,
            "chatterbox_voices": chatterbox_voices,
            "available_engines": available_engines
        }
    
    except Exception as e:
        logger.error(f"Error listing voices: {e}", exc_info=True)
        return {
            "kokoro_voices": [],
            "chatterbox_voices": [],
            "available_engines": ["chatterbox"],
            "error": str(e)
        }



async def prewarm_chatterbox_voices():
    """Pre-warm voices from settings.json - called from main app with correct paths"""
    try:
        # Load settings
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        if not settings_path.exists():
            logger.info("📝 No settings.json found, skipping voice pre-warming")
            return
            
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        voice_cache = settings.get('voice_cache', [])
        if not voice_cache:
            logger.info("📝 No voices in cache, skipping voice pre-warming")
            return
            
        logger.info(f"🔥 Pre-warming {len(voice_cache)} voices from settings...")
        
        # Voice references directory (relative to this file)
        voices_dir = Path(__file__).parent / "static" / "voice_references"
        logger.info(f"🔍 Looking for voices in: {voices_dir.absolute()}")
        
        for voice_entry in voice_cache:
            if voice_entry.get('engine') == 'chatterbox':
                voice_id = voice_entry.get('voice_id')
                if voice_id:
                    voice_path = voices_dir / voice_id
                    if voice_path.exists():
                        try:
                            logger.info(f"🔥 Pre-warming voice: {voice_id}")
                            # Call your TTS service to warm up this voice
                            # Note: This is a background task, so we can't access request.app.state
                            # We'll need to handle this differently or skip warmup in background tasks
                            logger.info(f"⚠️ Skipping voice warmup in background task (no TTS client access)")
                            logger.info(f"✅ Pre-warmed voice: {voice_id}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to pre-warm {voice_id}: {e}")
                    else:
                        logger.warning(f"⚠️ Voice file not found: {voice_path}")
        
        logger.info("✅ Voice pre-warming complete")
        
    except Exception as e:
        logger.error(f"⚠️ Voice pre-warming failed: {e}", exc_info=True)
# System prompt removed - relying on base model instructions and character personas
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

@router.post("/forensic/build-corpus-from-files")
async def build_corpus_from_files(
    background_tasks: BackgroundTasks,
    person_name: str = Form(...),
    files: List[UploadFile] = File(...),
    forensic_service: ForensicLinguisticsService = Depends(get_forensic_service)
):
    """Build a corpus from uploaded files instead of auto-scraping."""
    try:
        if len(files) > 10000:  # Limit to 10,000 files to prevent abuse
            raise HTTPException(status_code=400, detail="Maximum 10,000 files allowed")
        
        if not person_name.strip():
            raise HTTPException(status_code=400, detail="Person name is required")
        
        logger.info(f"🏗️ [Forensic] Building corpus for {person_name} from {len(files)} uploaded files")
        
        # Process all uploaded files
        corpus_documents = []
        total_chars = 0
        
        for i, file in enumerate(files):
            try:
                # Clean the content
                content = await process_uploaded_file_with_cleaning(file)
                
                if len(content.strip()) < 50:  # Skip very short content
                    logger.warning(f"Skipping {file.filename}: too short after cleaning")
                    continue
                
                # Create a document object for the corpus
                doc = TextDocument(  # ✅ Correct
                    content=content,
                    platform="uploaded_file",
                    date=datetime.datetime.now(),
                    author=person_name,  # Add this line
                    source_url=f"uploaded:{file.filename}",
                    title=file.filename,
                    metadata={"file_size": file.size, "file_type": file.filename.split('.')[-1]}
                )
                
                corpus_documents.append(doc)
                total_chars += len(content)
                
                logger.info(f"✅ Processed {file.filename}: {len(content)} chars")
                
            except Exception as e:
                logger.warning(f"❌ Failed to process {file.filename}: {e}")
                continue
        
        if len(corpus_documents) == 0:
            raise HTTPException(status_code=400, detail="No valid files could be processed")
        
        if len(corpus_documents) < 3:
            logger.warning(f"Only {len(corpus_documents)} files processed - corpus may be too small for reliable analysis")
        
        # Save the corpus
        forensic_service._cache_corpus(person_name, corpus_documents)
        
        logger.info(f"🎉 [Forensic] Successfully built corpus for {person_name}: {len(corpus_documents)} documents, {total_chars:,} total characters")
        
        return {
            "status": "success",
            "message": f"Corpus built successfully for {person_name}",
            "corpus_stats": {
                "person_name": person_name,
                "total_documents": len(corpus_documents),
                "total_characters": total_chars,
                "total_words": sum(len(doc.content.split()) for doc in corpus_documents),
                "files_processed": len([f for f in files if any(doc.title == f.filename for doc in corpus_documents)]),
                "files_skipped": len(files) - len(corpus_documents),
                "platform_breakdown": {
                    "uploaded_file": len(corpus_documents)
                },
                "average_document_length": total_chars // len(corpus_documents) if corpus_documents else 0
            },
            "build_timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ [Forensic] Error building corpus from files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

    # 5) Fetch memory context (ONLY for user chats, not for title generation or direct injection)
    memory_context_for_llm = "" # Initialize
    if body.request_purpose not in ["title_generation", "model_judging", "model_testing"] and not body.directProfileInjection:
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
        
    # 8.5) Add Author's Note if provided (custom session instructions)
    if body.authorNote and body.authorNote.strip() and body.request_purpose not in ["title_generation", "model_testing", "model_judging"]:
        author_note_text = body.authorNote.strip()
        interaction_components.append(f"[AUTHOR'S NOTE - Writing style guidance for this response]\n{author_note_text}")
        logger.info(f"📝 Added Author's Note to prompt: '{author_note_text[:50]}...'")

    # 8.6) Add Anti-Repetition instructions if enabled
    if body.anti_repetition_mode and body.request_purpose not in ["title_generation", "model_testing", "model_judging"]:
        anti_rep_instruction = """[VARIETY GUIDANCE]
Each response should feel fresh and unique. Avoid:
- Reusing paragraph structures or openings from your previous messages
- Repeating descriptive phrases you've already used in this conversation
- Formulaic greeting or closing patterns
Vary your sentence structure and word choices naturally."""
        interaction_components.append(anti_rep_instruction)
        logger.info("🔄 Added anti-repetition instructions to prompt")

    # Join all components of the interaction block with double newlines
    final_interaction_block = "\n\n".join(interaction_components)

    # 9) Assemble the full LLM prompt
    #    System prompt removed - relying on base model instructions and character personas
    #    character_persona_from_split contains the character-specific system instructions from the client.
    # Skip roleplay system prompt for model testing
    if body.request_purpose in ["model_testing", "model_judging"]:
        logger.info("🌀 Model testing/judging request: Skipping roleplay system prompt.")
        system_block_for_llm = "You are a language model designed for testing and evaluation purposes. Respond to the user's input without roleplay context."
    else:
        # Start with empty system block - base model instructions will provide default behavior
        system_block_for_llm = ""
    if character_persona_from_split: # This is from step #3 split
        system_block_for_llm += f"\n\nCharacter Persona:\n{character_persona_from_split}"
    
    # Construct the final prompt for the LLM
    if system_block_for_llm.strip():
        llm_prompt = f"{system_block_for_llm.strip()}\n\n{final_interaction_block.strip()}\n\nAssistant:"
    else:
        # No system prompt - just use interaction block
        llm_prompt = f"{final_interaction_block.strip()}\n\nAssistant:"
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
                yield f"data: {json.dumps({'done': True})}\n\n"  # Signal end of stream to client 
            else:
                # Regular streaming for text-only requests
                streamed_content_accumulator = []
                try:
                    # Check if this is an API endpoint - if so, route to OpenAI-compatible endpoint
                    is_api = is_api_endpoint(body.model_name)
                    logger.info(f"[generate] Model check: model_name='{body.model_name}', is_api_endpoint={is_api}")
                    if is_api:
                        logger.info(f"[generate] Detected API endpoint: {body.model_name}. Routing to OpenAI-compatible endpoint.")
                        
                        # Improved prompt parsing to extract multiple conversation turns
                        messages = []
                        
                        # Use regex to find all segments like <start_of_turn>user\n...\n<end_of_turn>
                        # This works for Gemma and similar formats
                        segments = re.findall(r'<start_of_turn>(user|model)\n(.*?)(?:<end_of_turn>|$)', prompt_text_for_llm, re.DOTALL)
                        
                        if segments:
                            logger.info(f"[generate] Parsed {len(segments)} segments from prompt using Gemma format logic.")
                            for role, content in segments:
                                messages.append({
                                    "role": "assistant" if role == "model" else "user",
                                    "content": content.strip()
                                })
                            
                            # If there's a system part before the first turn, add it as system message
                            system_part = prompt_text_for_llm.split("<start_of_turn>")[0].strip()
                            if system_part:
                                messages.insert(0, {"role": "system", "content": system_part})
                        else:
                            # Fallback to simple split logic for other formats or if regex fails
                            logger.info("[generate] Falling back to manual splitting for prompt parsing.")
                            if "Character Persona:" in prompt_text_for_llm:
                                parts = prompt_text_for_llm.split("Character Persona:", 1)
                                if parts[0].strip():
                                    messages.append({"role": "system", "content": parts[0].strip()})
                                if len(parts) > 1:
                                    persona_and_user = parts[1]
                                    if "User Query:" in persona_and_user:
                                        persona, user_query = persona_and_user.split("User Query:", 1)
                                        if persona.strip():
                                            messages.append({"role": "system", "content": f"Character Persona:\n{persona.strip()}"})
                                        messages.append({"role": "user", "content": user_query.strip()})
                                    else:
                                        messages.append({"role": "user", "content": persona_and_user.replace("Assistant:", "").strip()})
                            elif "User Query:" in prompt_text_for_llm:
                                parts = prompt_text_for_llm.split("User Query:", 1)
                                if parts[0].strip():
                                    messages.append({"role": "system", "content": parts[0].strip()})
                                messages.append({"role": "user", "content": parts[1].replace("Assistant:", "").strip()})
                            else:
                                # Simple prompt - treat as user message
                                clean_prompt = prompt_text_for_llm.replace("Assistant:", "").strip()
                                messages.append({"role": "user", "content": clean_prompt})
                        
                        # Prepare request data for API endpoint
                        request_data = {
                            "model": body.model_name,
                            "messages": messages,
                            "temperature": body.temperature,
                            "top_p": body.top_p,
                            "max_tokens": max_tokens,
                            "stream": True,
                        }
                        
                        if body.top_k:
                            request_data["top_k"] = body.top_k
                        if body.repetition_penalty:
                            request_data["repetition_penalty"] = body.repetition_penalty
                        stop_seqs = dcu.get_stop_sequences(body.stop)
                        if stop_seqs:
                            request_data["stop"] = stop_seqs
                        
                        # Use centralized helper for config, URL, and CONTEXT PRUNING
                        try:
                            endpoint_config, url, request_data = prepare_endpoint_request(body.model_name, request_data)
                        except HTTPException as e:
                            logger.error(f"[generate] {e.detail}")
                            yield f"data: {json.dumps({'error': e.detail})}\n\n"
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            return
                        
                        logger.info(f"[generate] Forwarding {body.model_name} to {endpoint_config['name']} at {url}")
                        
                        # Stream from the API endpoint
                        # The forward_to_configured_endpoint_streaming function yields raw bytes in SSE format
                        # We need to buffer them, parse OpenAI format, and convert to frontend format
                        buffer = b""
                        async for chunk_bytes in forward_to_configured_endpoint_streaming(endpoint_config, url, request_data):
                            # Accumulate bytes in buffer
                            if isinstance(chunk_bytes, bytes):
                                buffer += chunk_bytes
                            else:
                                buffer += chunk_bytes.encode('utf-8') if isinstance(chunk_bytes, str) else b""
                            
                            # Process complete SSE messages (separated by \n\n)
                            while b'\n\n' in buffer:
                                message, buffer = buffer.split(b'\n\n', 1)
                                if not message.strip():
                                    continue
                                
                                try:
                                    message_str = message.decode('utf-8', errors='ignore')
                                    # SSE format: "data: {...}\n" or just "data: {...}"
                                    lines = message_str.split('\n')
                                    for line in lines:
                                        if line.startswith("data: "):
                                            json_str = line[6:].strip()
                                            if json_str == "[DONE]":
                                                continue
                                            
                                            try:
                                                chunk_data = json.loads(json_str)
                                                
                                                # Extract content from OpenAI format: {"choices": [{"delta": {"content": "..."}}]}
                                                content = ""
                                                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                                    delta = chunk_data["choices"][0].get("delta", {})
                                                    content = delta.get("content", "")
                                                
                                                # Convert to frontend format: {"text": "..."}
                                                if content:
                                                    streamed_content_accumulator.append(content)
                                                    # Yield in the format the frontend expects
                                                    yield f"data: {json.dumps({'text': content})}\n\n"
                                            except json.JSONDecodeError:
                                                # If it's not JSON, might be an error message - forward as-is
                                                if json_str:
                                                    yield f"data: {json_str}\n\n"
                                except Exception as e:
                                    logger.debug(f"Error processing API chunk: {e}")
                        
                        # Yield done message after API streaming completes
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    else:
                        # Local model - use existing inference path
                        async for token in inference.generate_text_streaming(
                            model_manager=model_manager, model_name=body.model_name, prompt=prompt_text_for_llm,
                            max_tokens=max_tokens, temperature=body.temperature, top_p=body.top_p,
                            top_k=body.top_k, repetition_penalty=body.repetition_penalty,
                            stop_sequences=dcu.get_stop_sequences(body.stop), gpu_id=gpu_id, echo=body.echo,
                            request_purpose=body.request_purpose
                        ):
                            # Extract just the text content for memory processing
                            try:
                                if token.startswith("data: "):
                                    token_data = json.loads(token[6:])  # Remove "data: " prefix
                                    if "text" in token_data:
                                        streamed_content_accumulator.append(token_data["text"])
                            except (json.JSONDecodeError, KeyError):
                                # If parsing fails, just append the raw token
                                streamed_content_accumulator.append(token)
                            
                            yield token
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception as stream_exc:
                    logger.error(f"❌ Error during LLM streaming: {stream_exc}", exc_info=True)
                    # Optionally, yield an error event to the client if your frontend handles it
                    # yield f"event: error\ndata: {json.dumps({'detail': str(stream_exc)})}\n\n"
                    # Ensure [DONE] is still sent or handle client-side appropriately
                    yield f"data: {json.dumps({'error': f'[STREAM_ERROR: {str(stream_exc)}]'})}\n\n" # Send error in data
                    yield f"data: {json.dumps({'done': True})}\n\n"

            # ---- After stream is DONE ----
            full_llm_response_text = "".join(streamed_content_accumulator)
            logger.info(f"🌀 Stream complete. Full response length: {len(full_llm_response_text)}. Scheduling detect_and_store if applicable.")

            clean_full_llm_response = full_llm_response_text.replace("<|DONE|>", "").strip() # Clean it once

            if body.directProfileInjection:
                logger.info(f"🌀 Direct profile injection enabled. Stream complete.")
            elif is_title_generation_request or body.request_purpose in ["model_testing", "model_judging"]:
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
            
            # Check if this is an API endpoint - if so, route to OpenAI-compatible endpoint
            is_api = is_api_endpoint(body.model_name)
            logger.info(f"[generate] Model check (non-streaming): model_name='{body.model_name}', is_api_endpoint={is_api}")
            if is_api:
                logger.info(f"[generate] Detected API endpoint: {body.model_name}. Routing to OpenAI-compatible endpoint (non-streaming).")
                
                # Improved prompt parsing for non-streaming
                messages = []
                segments = re.findall(r'<start_of_turn>(user|model)\n(.*?)(?:<end_of_turn>|$)', llm_prompt, re.DOTALL)
                
                if segments:
                    for role, content in segments:
                        messages.append({
                            "role": "assistant" if role == "model" else "user",
                            "content": content.strip()
                        })
                    system_part = llm_prompt.split("<start_of_turn>")[0].strip()
                    if system_part:
                        messages.insert(0, {"role": "system", "content": system_part})
                else:
                    if "Character Persona:" in llm_prompt:
                        parts = llm_prompt.split("Character Persona:", 1)
                        if parts[0].strip():
                            messages.append({"role": "system", "content": parts[0].strip()})
                        if len(parts) > 1:
                            persona_and_user = parts[1]
                            if "User Query:" in persona_and_user:
                                persona, user_query = persona_and_user.split("User Query:", 1)
                                if persona.strip():
                                    messages.append({"role": "system", "content": f"Character Persona:\n{persona.strip()}"})
                                messages.append({"role": "user", "content": user_query.strip()})
                            else:
                                messages.append({"role": "user", "content": persona_and_user.replace("Assistant:", "").strip()})
                    elif "User Query:" in llm_prompt:
                        parts = llm_prompt.split("User Query:", 1)
                        if parts[0].strip():
                            messages.append({"role": "system", "content": parts[0].strip()})
                        messages.append({"role": "user", "content": parts[1].replace("Assistant:", "").strip()})
                    else:
                        clean_prompt = llm_prompt.replace("Assistant:", "").strip()
                        messages.append({"role": "user", "content": clean_prompt})
                
                # Prepare request data for API endpoint
                request_data = {
                    "model": body.model_name,
                    "messages": messages,
                    "temperature": body.temperature,
                    "top_p": body.top_p,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                
                if body.top_k:
                    request_data["top_k"] = body.top_k
                if body.repetition_penalty:
                    request_data["repetition_penalty"] = body.repetition_penalty
                stop_seqs = dcu.get_stop_sequences(body.stop)
                if stop_seqs:
                    request_data["stop"] = stop_seqs
                
                # Use centralized helper for config, URL, and CONTEXT PRUNING
                endpoint_config, url, request_data = prepare_endpoint_request(body.model_name, request_data)
                
                logger.info(f"[generate] Forwarding {body.model_name} to {endpoint_config['name']} at {url}")
                
                # Call the API endpoint
                result = await forward_to_configured_endpoint_non_streaming(endpoint_config, url, request_data)
                
                # Extract text from OpenAI-compatible response
                if result and "choices" in result and len(result["choices"]) > 0:
                    llm_output_raw_text = result["choices"][0].get("message", {}).get("content", "")
                else:
                    llm_output_raw_text = "API endpoint returned no valid response."
            else:
                # Local model - use existing path
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
        if body.directProfileInjection:
            logger.info("🧠 Direct Profile Injection is ON. Skipping memory creation task (non-streaming).")
        elif body.request_purpose in ["title_generation", "model_testing", "model_judging"]:
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
    
@router.post("/models/performance-test")
async def performance_test_endpoint(
    request: Request,
    data: dict = Body(...),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Test endpoint to benchmark model performance in different modes.
    Expects: {"model_name": "...", "gpu_id": 0, "test_prompt": "...", "max_tokens": 100}
    """
    try:
        model_name = data.get("model_name")
        gpu_id = data.get("gpu_id", 0)
        test_prompt = data.get("test_prompt", "Write a short story about a robot learning to paint.")
        max_tokens = data.get("max_tokens", 100)
        
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        
        logger.info(f"🚀 [Performance Test] Starting benchmark for {model_name} on GPU {gpu_id}")
        
        # Ensure model is loaded
        try:
            await model_manager.load_model(model_name, gpu_id=gpu_id)
            logger.info(f"✅ [Performance Test] Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"❌ [Performance Test] Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        # Get model instance
        model = model_manager.get_model(model_name, gpu_id)
        
        # Check if it's unified mode
        is_unified_mode = isinstance(model, model_manager.RemoteModelWrapper)
        mode_name = "unified_model" if is_unified_mode else "split_services"
        
        logger.info(f"🚀 [Performance Test] Testing {mode_name} mode")
        
        # Run performance test
        start_time = time.time()
        
        try:
            # Generate response
            response = model(
                prompt=test_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract text and calculate metrics
            if response and "choices" in response and response["choices"]:
                generated_text = response["choices"][0]["text"]
                estimated_tokens = len(generated_text) // 4
                tokens_per_second = estimated_tokens / generation_time if generation_time > 0 else 0
                
                logger.info(f"🚀 [Performance Test] {mode_name} mode results:")
                logger.info(f"   Generation time: {generation_time:.2f}s")
                logger.info(f"   Estimated tokens: {estimated_tokens}")
                logger.info(f"   Speed: {tokens_per_second:.1f} tokens/second")
                
                return {
                    "status": "success",
                    "mode": mode_name,
                    "model_name": model_name,
                    "gpu_id": gpu_id,
                    "performance_metrics": {
                        "generation_time": generation_time,
                        "estimated_tokens": estimated_tokens,
                        "tokens_per_second": tokens_per_second,
                        "test_prompt": test_prompt,
                        "generated_text": generated_text
                    }
                }
            else:
                raise Exception("Invalid response format from model")
                
        except Exception as e:
            logger.error(f"❌ [Performance Test] Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ [Performance Test] Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

@router.post("/models/set-direct-profile-injection")
async def set_direct_profile_injection(data: dict = Body(...)):
    """Save direct profile injection setting to settings"""
    try:
        direct_profile_injection = data.get("directProfileInjection", False)
        
        settings_dir = Path.home() / ".LiangLocal"
        settings_path = settings_dir / "settings.json"
        os.makedirs(settings_dir, exist_ok=True)
        
        settings = {}
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        
        settings['directProfileInjection'] = direct_profile_injection
        
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        logger.info(f"✅ Direct Profile Injection setting saved: {direct_profile_injection}")
        return {"status": "success", "directProfileInjection": direct_profile_injection}
    except Exception as e:
        logger.error(f"❌ Error saving direct profile injection setting: {str(e)}")
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
    """Enhance an existing image using ADetailer post-processing on a specific GPU."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    try:
        # Extract parameters
        image_url = data.get("image_url")
        gpu_id = data.get("gpu_id", 0) # CRITICAL FIX: Get the GPU ID from the request

        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        
        # Convert URL to local file path
        if "/static/generated_images/" in image_url:
            filename = image_url.split("/static/generated_images/")[-1]
            image_path = Path(__file__).parent / "static" / "generated_images" / filename
        else:
            raise HTTPException(status_code=400, detail="Invalid image URL")
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        logger.info(f"Enhancing image: {image_path} with ADetailer using model {data.get('model_name')} on GPU {gpu_id}")
        
        # Enhance the image
        enhanced_image_data = sd_manager.enhance_image_with_adetailer(
            image_path=str(image_path),
            original_prompt=data.get("original_prompt", ""),
            face_prompt=data.get("face_prompt", ""),
            strength=data.get("strength", 0.4),
            confidence=data.get("confidence", 0.3),
            model_name=data.get("model_name", "face_yolov8n.pt"),
            gpu_id=gpu_id # CRITICAL FIX: Pass the GPU ID to the manager
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
            "model_used": data.get("model_name", "face_yolov8n.pt"),
            "parameters": {
                "strength": data.get("strength", 0.4),
                "confidence": data.get("confidence", 0.3),
                "face_prompt": data.get("face_prompt", ""),
                "model_name": data.get("model_name", "face_yolov8n.pt")
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
    """Check local SD status and loaded models on all GPUs."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        return {"available": False, "error": "SD Manager not initialized", "loaded_models": {}}
    
    # Return a dictionary of loaded models keyed by GPU ID
    return {
        "available": True,
        "loaded_models": sd_manager.current_model_paths 
    }

# ============================================================================
# ComfyUI API Integration - Complete Package
# ============================================================================

COMFYUI_BASE_URL = "http://127.0.0.1:8188"

# Standard ComfyUI samplers and schedulers
COMFY_SAMPLERS = [
    "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
    "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc",
    "uni_pc_bh2"
]

COMFY_SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform",
    "beta"
]

def _build_comfy_txt2img_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    checkpoint: str = "",
    batch_size: int = 1,
    denoise: float = 1.0
) -> dict:
    """Build a ComfyUI txt2img workflow JSON."""
    import random
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint if checkpoint else "v1-5-pruned-emaonly.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "Eloquent_ComfyUI",
                "images": ["8", 0]
            }
        }
    }
    return workflow

def _build_comfy_img2img_workflow(
    prompt: str,
    negative_prompt: str = "",
    image_base64: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    checkpoint: str = "",
    denoise: float = 0.75
) -> dict:
    """Build a ComfyUI img2img workflow JSON."""
    import random
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    workflow = {
        "1": {
            "class_type": "LoadImageBase64",
            "inputs": {
                "image": image_base64
            }
        },
        "2": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["4", 2]
            }
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["2", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint if checkpoint else "v1-5-pruned-emaonly.safetensors"
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "Eloquent_ComfyUI_i2i",
                "images": ["8", 0]
            }
        }
    }
    return workflow

def _build_comfy_upscale_workflow(
    image_base64: str = "",
    upscale_model: str = "RealESRGAN_x4plus.pth",
    scale_factor: float = 2.0
) -> dict:
    """Build a ComfyUI upscale workflow JSON."""
    workflow = {
        "1": {
            "class_type": "LoadImageBase64",
            "inputs": {
                "image": image_base64
            }
        },
        "2": {
            "class_type": "UpscaleModelLoader",
            "inputs": {
                "model_name": upscale_model
            }
        },
        "3": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["2", 0],
                "image": ["1", 0]
            }
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "Eloquent_Upscaled",
                "images": ["3", 0]
            }
        }
    }
    return workflow

async def _comfy_queue_and_wait(workflow: dict, timeout_seconds: int = 300) -> dict:
    """Queue a workflow and wait for completion. Returns output info."""
    import asyncio
    
    async with httpx.AsyncClient() as client:
        # Queue the prompt
        queue_resp = await client.post(
            f"{COMFYUI_BASE_URL}/prompt",
            json={"prompt": workflow},
            timeout=10.0
        )
        queue_resp.raise_for_status()
        result = queue_resp.json()
        prompt_id = result.get("prompt_id")
        
        if not prompt_id:
            raise HTTPException(500, "ComfyUI did not return a prompt_id")
        
        # Check for immediate errors
        if "error" in result:
            raise HTTPException(500, f"ComfyUI workflow error: {result['error']}")
        if "node_errors" in result and result["node_errors"]:
            raise HTTPException(500, f"ComfyUI node errors: {result['node_errors']}")
        
        # Poll for completion
        for _ in range(timeout_seconds):
            await asyncio.sleep(1)
            
            # Check queue status
            queue_resp = await client.get(f"{COMFYUI_BASE_URL}/queue", timeout=5.0)
            if queue_resp.status_code == 200:
                queue_data = queue_resp.json()
                running = queue_data.get("queue_running", [])
                pending = queue_data.get("queue_pending", [])
                
                # Check if our job is still in queue
                our_job_running = any(job[1] == prompt_id for job in running)
                our_job_pending = any(job[1] == prompt_id for job in pending)
                
                if not our_job_running and not our_job_pending:
                    # Job finished, check history
                    history_resp = await client.get(
                        f"{COMFYUI_BASE_URL}/history/{prompt_id}",
                        timeout=10.0
                    )
                    
                    if history_resp.status_code == 200:
                        history = history_resp.json()
                        if prompt_id in history:
                            job_data = history[prompt_id]
                            
                            # Check for execution errors
                            if job_data.get("status", {}).get("status_str") == "error":
                                error_msg = job_data.get("status", {}).get("messages", [])
                                raise HTTPException(500, f"ComfyUI execution error: {error_msg}")
                            
                            return {
                                "prompt_id": prompt_id,
                                "outputs": job_data.get("outputs", {}),
                                "status": job_data.get("status", {})
                            }
        
        # Timeout - try to cancel the job
        try:
            await client.post(f"{COMFYUI_BASE_URL}/queue", json={"delete": [prompt_id]})
        except:
            pass
        
        raise HTTPException(504, "ComfyUI generation timed out")

async def _comfy_fetch_images(outputs: dict) -> list:
    """Fetch generated images from ComfyUI outputs and save locally."""
    saved_urls = []
    
    async with httpx.AsyncClient() as client:
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    img_resp = await client.get(
                        f"{COMFYUI_BASE_URL}/view",
                        params={
                            "filename": img_info["filename"],
                            "subfolder": img_info.get("subfolder", ""),
                            "type": img_info.get("type", "output")
                        },
                        timeout=30.0
                    )
                    if img_resp.status_code == 200:
                        image_url = save_image_and_get_url(img_resp.content)
                        saved_urls.append(image_url)
    
    return saved_urls

@app.get("/sd-comfy/status")
async def sd_comfy_status():
    """Check if ComfyUI is running and get full configuration options."""
    try:
        async with httpx.AsyncClient() as client:
            # Check system stats
            resp = await client.get(f"{COMFYUI_BASE_URL}/system_stats", timeout=5.0)
            resp.raise_for_status()
            system_stats = resp.json()
            
            # Get available checkpoints
            checkpoints = []
            try:
                obj_resp = await client.get(f"{COMFYUI_BASE_URL}/object_info/CheckpointLoaderSimple", timeout=5.0)
                if obj_resp.status_code == 200:
                    obj_info = obj_resp.json()
                    checkpoints = obj_info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            except:
                pass
            
            # Get available VAEs
            vaes = []
            try:
                vae_resp = await client.get(f"{COMFYUI_BASE_URL}/object_info/VAELoader", timeout=5.0)
                if vae_resp.status_code == 200:
                    vae_info = vae_resp.json()
                    vaes = vae_info.get("VAELoader", {}).get("input", {}).get("required", {}).get("vae_name", [[]])[0]
            except:
                pass
            
            # Get available LoRAs
            loras = []
            try:
                lora_resp = await client.get(f"{COMFYUI_BASE_URL}/object_info/LoraLoader", timeout=5.0)
                if lora_resp.status_code == 200:
                    lora_info = lora_resp.json()
                    loras = lora_info.get("LoraLoader", {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
            except:
                pass
            
            # Get available upscale models
            upscalers = []
            try:
                up_resp = await client.get(f"{COMFYUI_BASE_URL}/object_info/UpscaleModelLoader", timeout=5.0)
                if up_resp.status_code == 200:
                    up_info = up_resp.json()
                    upscalers = up_info.get("UpscaleModelLoader", {}).get("input", {}).get("required", {}).get("model_name", [[]])[0]
            except:
                pass
            
            return {
                "comfyui": True,
                "system": system_stats,
                "checkpoints": checkpoints,
                "vaes": vaes,
                "loras": loras,
                "upscalers": upscalers,
                "samplers": COMFY_SAMPLERS,
                "schedulers": COMFY_SCHEDULERS
            }
    except Exception as e:
        return {
            "comfyui": False,
            "error": str(e),
            "checkpoints": [],
            "vaes": [],
            "loras": [],
            "upscalers": [],
            "samplers": COMFY_SAMPLERS,
            "schedulers": COMFY_SCHEDULERS
        }

@app.get("/sd-comfy/queue")
async def sd_comfy_queue():
    """Get current ComfyUI queue status."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{COMFYUI_BASE_URL}/queue", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(502, f"ComfyUI connection error: {e}")

@app.post("/sd-comfy/interrupt")
async def sd_comfy_interrupt():
    """Interrupt current ComfyUI generation."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{COMFYUI_BASE_URL}/interrupt", timeout=5.0)
            return {"status": "interrupted"}
    except Exception as e:
        raise HTTPException(502, f"ComfyUI connection error: {e}")

@app.post("/sd-comfy/clear-queue")
async def sd_comfy_clear_queue():
    """Clear all pending ComfyUI jobs."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{COMFYUI_BASE_URL}/queue",
                json={"clear": True},
                timeout=5.0
            )
            return {"status": "queue_cleared"}
    except Exception as e:
        raise HTTPException(502, f"ComfyUI connection error: {e}")

@app.post("/sd-comfy/txt2img")
async def sd_comfy_txt2img(body: dict):
    """Generate image using ComfyUI txt2img."""
    try:
        workflow = _build_comfy_txt2img_workflow(
            prompt=body.get("prompt", ""),
            negative_prompt=body.get("negative_prompt", ""),
            width=body.get("width", 512),
            height=body.get("height", 512),
            steps=body.get("steps", 20),
            cfg_scale=body.get("cfg_scale", 7.0),
            seed=body.get("seed", -1),
            sampler=body.get("sampler", "euler"),
            scheduler=body.get("scheduler", "normal"),
            checkpoint=body.get("checkpoint", ""),
            batch_size=body.get("batch_size", 1),
            denoise=body.get("denoise", 1.0)
        )
        
        result = await _comfy_queue_and_wait(workflow, timeout_seconds=body.get("timeout", 300))
        saved_urls = await _comfy_fetch_images(result["outputs"])
        
        return JSONResponse({
            "status": "success",
            "image_urls": saved_urls,
            "prompt_id": result["prompt_id"]
        })
        
    except HTTPException:
        raise
    except httpx.RequestError as e:
        raise HTTPException(502, f"ComfyUI connection error: {e}")
    except Exception as e:
        raise HTTPException(500, f"ComfyUI error: {e}")

@app.post("/sd-comfy/img2img")
async def sd_comfy_img2img(body: dict):
    """Generate image using ComfyUI img2img."""
    image_base64 = body.get("image", "")
    if not image_base64:
        raise HTTPException(400, "image (base64) is required for img2img")
    
    # Strip data URI prefix if present
    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    
    try:
        workflow = _build_comfy_img2img_workflow(
            prompt=body.get("prompt", ""),
            negative_prompt=body.get("negative_prompt", ""),
            image_base64=image_base64,
            width=body.get("width", 512),
            height=body.get("height", 512),
            steps=body.get("steps", 20),
            cfg_scale=body.get("cfg_scale", 7.0),
            seed=body.get("seed", -1),
            sampler=body.get("sampler", "euler"),
            scheduler=body.get("scheduler", "normal"),
            checkpoint=body.get("checkpoint", ""),
            denoise=body.get("denoise", 0.75)
        )
        
        result = await _comfy_queue_and_wait(workflow, timeout_seconds=body.get("timeout", 300))
        saved_urls = await _comfy_fetch_images(result["outputs"])
        
        return JSONResponse({
            "status": "success",
            "image_urls": saved_urls,
            "prompt_id": result["prompt_id"]
        })
        
    except HTTPException:
        raise
    except httpx.RequestError as e:
        raise HTTPException(502, f"ComfyUI connection error: {e}")
    except Exception as e:
        raise HTTPException(500, f"ComfyUI error: {e}")

@app.post("/sd-comfy/upscale")
async def sd_comfy_upscale(body: dict):
    """Upscale image using ComfyUI."""
    image_base64 = body.get("image", "")
    if not image_base64:
        raise HTTPException(400, "image (base64) is required for upscaling")
    
    # Strip data URI prefix if present
    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    
    try:
        workflow = _build_comfy_upscale_workflow(
            image_base64=image_base64,
            upscale_model=body.get("upscale_model", "RealESRGAN_x4plus.pth"),
            scale_factor=body.get("scale_factor", 2.0)
        )
        
        result = await _comfy_queue_and_wait(workflow, timeout_seconds=body.get("timeout", 300))
        saved_urls = await _comfy_fetch_images(result["outputs"])
        
        return JSONResponse({
            "status": "success",
            "image_urls": saved_urls,
            "prompt_id": result["prompt_id"]
        })
        
    except HTTPException:
        raise
    except httpx.RequestError as e:
        raise HTTPException(502, f"ComfyUI connection error: {e}")
    except Exception as e:
        raise HTTPException(500, f"ComfyUI error: {e}")

@app.post("/sd-comfy/workflow")
async def sd_comfy_custom_workflow(body: dict):
    """Execute a custom ComfyUI workflow JSON."""
    workflow = body.get("workflow")
    if not workflow:
        raise HTTPException(400, "workflow JSON is required")
    
    try:
        result = await _comfy_queue_and_wait(workflow, timeout_seconds=body.get("timeout", 300))
        saved_urls = await _comfy_fetch_images(result["outputs"])
        
        return JSONResponse({
            "status": "success",
            "image_urls": saved_urls,
            "prompt_id": result["prompt_id"],
            "outputs": result["outputs"]
        })
        
    except HTTPException:
        raise
    except httpx.RequestError as e:
        raise HTTPException(502, f"ComfyUI connection error: {e}")
    except Exception as e:
        raise HTTPException(500, f"ComfyUI error: {e}")

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
    """Generate image using local SD on a specific GPU."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    try:
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt required")

        # Get GPU ID from the request, default to 0
        gpu_id = body.get("gpu_id", 0)

        # Check for seed and randomize if it's -1
        seed = body.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Local SD: No seed provided, generated random seed: {seed}")

        # This returns the raw image bytes
        image_data = sd_manager.generate_image(
            prompt=prompt,
            gpu_id=gpu_id, # Pass the GPU ID to the manager
            negative_prompt=body.get("negative_prompt", ""),
            width=body.get("width", 768), # Changed from 512 to 768 to match user's working aspect ratio
            height=body.get("height", 512),
            steps=body.get("steps", 20),
            cfg_scale=body.get("guidance_scale", 7.0),
            seed=seed
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

@app.post("/models/update-upscaler-dir")
async def update_upscaler_dir(body: dict):
    """Update the Upscaler models directory setting."""
    try:
        directory = body.get("directory")
        if not directory:
            raise HTTPException(status_code=400, detail="Directory path required")
            
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        
        # Load existing
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
            
        # Update
        settings["upscaler_model_directory"] = directory
        
        # Save
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
            
        return {"status": "success", "message": f"Upscaler directory updated to {directory}"}
        
    except Exception as e:
        logger.error(f"Error updating upscaler directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sd-local/upscalers")
async def get_upscalers(request: Request):
    """List available upscaler models."""
    upscale_manager = getattr(request.app.state, 'upscale_manager', None)
    if not upscale_manager:
        # Quick init check if not initialized (reuse logic or simple check)
        # For simplicity, if not init, try to init with settings or default
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        models_dir = r"C:\stable-diffusion-webui\models\ESRGAN"
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    if settings.get("upscaler_model_directory"):
                        models_dir = settings["upscaler_model_directory"]
            except: pass
            
        try:
            from .upscale_manager import UpscaleManager
            upscale_manager = UpscaleManager(models_dir)
            request.app.state.upscale_manager = upscale_manager
        except Exception:
            return {"models": []} # Return empty if init fails

    return {"models": list(upscale_manager.models.keys())}

@app.post("/sd-local/upscale")
async def sd_upscale(body: dict, request: Request, model_manager: ModelManager = Depends(get_model_manager)):
    """Upscale an image using custom UpscaleManager (ESRGAN)."""
    # Lazy initialization of UpscaleManager
    upscale_manager = getattr(request.app.state, 'upscale_manager', None)
    if not upscale_manager:
        # Load from settings
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        models_dir = r"C:\stable-diffusion-webui\models\ESRGAN" # Default fallback
        
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    if settings.get("upscaler_model_directory"):
                        models_dir = settings["upscaler_model_directory"]
            except Exception as e:
                logger.error(f"Error reading upscaler setting: {e}")

        logger.info(f"Initializing UpscaleManager with models dir: {models_dir}")
        try:
            from .upscale_manager import UpscaleManager
            upscale_manager = UpscaleManager(models_dir)
            request.app.state.upscale_manager = upscale_manager
        except Exception as e:
             logger.error(f"Failed to initialize UpscaleManager: {e}")
             raise HTTPException(status_code=500, detail=f"Upscale Manager failed to initialize: {str(e)}")

    try:
        image_url = body.get("image_url")
        image_data_b64 = body.get("image_data")
        scale_factor = float(body.get("scale_factor", 2.0)) # Note: Many ESRGAN models are fixed 4x, manager handles this
        model_name = body.get("model_name") # Optional specific model
        
        image_bytes = None
        
        # Handle URL or Base64 (prefer base64 if provided, else path from URL)
        if image_data_b64:
            image_bytes = base64.b64decode(image_data_b64)
        elif image_url:
            # Convert URL to local path if possible
            # Assumes URL is like /static/generated_images/...
            if "/static/generated_images/" in image_url:
                filename = image_url.split("/")[-1]
                file_path = generated_images_dir / filename
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        image_bytes = f.read()
                else:
                    raise HTTPException(status_code=404, detail="Source image file not found")
            else:
                raise HTTPException(status_code=400, detail="Only local generated images supported for now")
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image provided")

        # Load PIL Image
        input_image = Image.open(io.BytesIO(image_bytes))

        # Perform Upscale
        upscaled_image = upscale_manager.upscale(
            image=input_image,
            model_name=model_name,
            scale_factor=scale_factor
        )

        # Save result
        with io.BytesIO() as output:
            upscaled_image.save(output, format="PNG")
            upscaled_bytes = output.getvalue()

        new_image_url = save_image_and_get_url(upscaled_bytes)

        return {
            "status": "success",
            "image_url": new_image_url,
            "original_url": image_url,
            "scale_factor": scale_factor,
            "model_used": upscale_manager.current_model_name
        }

    except Exception as e:
        logger.error(f"Upscale error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sd-local/visualize")
async def sd_local_visualize(body: dict, request: Request, model_manager: ModelManager = Depends(get_model_manager)):
    """Generate an image based on the chat context."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    try:
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages required")
        
        # Use primary model for prompt generation
        # Find a suitable model similar to chat completion logic
        model_name = body.get("model_name")
        if not model_name and model_manager.loaded_models:
             model_name = next(iter(model_manager.loaded_models.keys()))[0]

        if not model_name:
             raise HTTPException(status_code=500, detail="No LLM loaded for prompt generation")

        # 1. Summarize context into an image prompt
        # We limit context to last 10 messages for speed and relevance
        recent_context = messages[-10:]
        context_str = "\\n".join([f"{m['role']}: {m['content']}" for m in recent_context])
        
        system_prompt = "You are an expert stable diffusion prompt engineer. Your task is to visualize the current scene described in the conversation."
        user_prompt = f"""Based on the following conversation, create a detailed Stable Diffusion prompt to visualize the current scene. 
Include details about characters, setting, lighting, and mood.
Format the output as a SINGLE paragraph of comma-separated keywords.
Do NOT use bullet points, newlines, or lists.
Do NOT include negative prompts or explanations. Just the prompt keywords.

Conversation:
{context_str}

Image Prompt:"""

        # Generate prompt using the LLM
        # using standard generation helper or direct inference call
        generated_prompt = await generate_llm_response(
            prompt=f"{system_prompt}\\n\\n{user_prompt}", 
            model_manager=model_manager,
            model_name=model_name,
            max_tokens=150,
            temperature=0.7
        )
        
        # Clean up prompt (remove "Image Prompt:" prefix if model excessively chattered)
        clean_prompt = generated_prompt.replace("Image Prompt:", "").strip()
        logger.info(f"Visualizing scene with prompt: {clean_prompt}")

        # 2. Generate Image
        gpu_id = body.get("gpu_id", 0)
        
        image_bytes = sd_manager.generate_image(
            prompt=clean_prompt,
            gpu_id=gpu_id,
            steps=25,
            width=512,
            height=512,
            cfg_scale=7.5
        )
        
        image_url = save_image_and_get_url(image_bytes)
        
        return {
            "status": "success",
            "image_url": image_url,
            "generated_prompt": clean_prompt
        }

    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/code_editor/read_file")
async def read_file(request: FileOperationRequest):
    """Read contents of a file"""
    try:
        safe_path = get_safe_path(CODE_EDITOR_BASE_DIR, request.filepath)
        if not safe_path:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if not os.path.isfile(safe_path):
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Check file size (limit to 10MB for safety)
        file_size = os.path.getsize(safe_path)
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="File too large")
        
        with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return {
            "success": True,
            "filepath": request.filepath,
            "content": content,
            "size": file_size,
            "modified": datetime.datetime.fromtimestamp(os.path.getmtime(safe_path)).isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file {request.filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/code_editor/write_file")
async def write_file(request: FileOperationRequest):
    """Write content to a file"""
    try:
        if request.content is None:
            raise HTTPException(status_code=400, detail="Content is required")
        
        safe_path = get_safe_path(CODE_EDITOR_BASE_DIR, request.filepath)
        if not safe_path:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        # Create backup if file exists
        if os.path.exists(safe_path):
            backup_path = f"{safe_path}.backup.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(safe_path, backup_path)
        
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(request.content)
        
        return {
            "success": True,
            "filepath": request.filepath,
            "size": len(request.content.encode('utf-8')),
            "message": "File written successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error writing file {request.filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing file: {str(e)}")

@app.post("/code_editor/list_directory")
async def list_directory(request: DirectoryListRequest):
    """List contents of a directory"""
    try:
        safe_path = get_safe_path(CODE_EDITOR_BASE_DIR, request.path)
        if not safe_path:
            raise HTTPException(status_code=400, detail="Invalid directory path")
        
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        if not os.path.isdir(safe_path):
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        items = []
        for item_name in os.listdir(safe_path):
            if not request.include_hidden and item_name.startswith('.'):
                continue
            
            item_path = os.path.join(safe_path, item_name)
            is_dir = os.path.isdir(item_path)
            
            try:
                stat = os.stat(item_path)
                items.append({
                    "name": item_name,
                    "type": "folder" if is_dir else "file",
                    "size": stat.st_size if not is_dir else None,
                    "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": os.path.relpath(item_path, CODE_EDITOR_BASE_DIR)
                })
            except (OSError, PermissionError):
                # Skip items we can't access
                continue
        
        # Sort: directories first, then files, both alphabetically
        items.sort(key=lambda x: (x["type"] == "file", x["name"].lower()))
        
        return {
            "success": True,
            "path": request.path,
            "items": items
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing directory {request.path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing directory: {str(e)}")

@app.post("/code_editor/search_files")
async def search_files(request: SearchFilesRequest):
    """Search for text within files (grep-like functionality)"""
    try:
        safe_path = get_safe_path(CODE_EDITOR_BASE_DIR, request.path)
        if not safe_path:
            raise HTTPException(status_code=400, detail="Invalid search path")
        
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="Search path not found")
        
        results = []
        count = 0
        start_time = time.time()
        max_search_time = 30  # 30 second timeout
        
        # Walk through directory tree
        for root, dirs, files in os.walk(safe_path):
            # Check timeout
            if time.time() - start_time > max_search_time:
                logger.warning(f"⚠️ Search timeout reached after {max_search_time}s")
                break
                
            # Skip hidden directories unless requested
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # Skip certain directories that are usually not relevant for code search
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', 'node_modules', '.git', 'wheels', 'upgrade']]
            
            for file in files:
                # Check timeout
                if time.time() - start_time > max_search_time:
                    break
                    
                if not fnmatch.fnmatch(file, request.file_pattern):
                    continue
                
                if file.startswith('.'):
                    continue
                
                # Skip certain file types that are usually not relevant for code search
                if file.endswith(('.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.dat')):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, CODE_EDITOR_BASE_DIR)
                
                try:
                    # Skip binary files and large files
                    if os.path.getsize(file_path) > 1024 * 1024:  # 1MB limit
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if request.query.lower() in line.lower():
                                results.append({
                                    "file": rel_path,
                                    "line": line_num,
                                    "content": line.strip(),
                                    "match": request.query
                                })
                                count += 1
                                
                                if count >= request.max_results:
                                    break
                    
                    if count >= request.max_results:
                        break
                        
                except (UnicodeDecodeError, PermissionError, OSError):
                    # Skip files we can't read
                    continue
            
            if count >= request.max_results or time.time() - start_time > max_search_time:
                break
        
        # Check if this was a file name search (common case)
        if request.query.endswith('.py') or request.query.endswith('.js') or request.query.endswith('.jsx'):
            # For file name searches, also try to find the file directly
            try:
                direct_file_path = os.path.join(safe_path, request.query)
                if os.path.exists(direct_file_path) and os.path.isfile(direct_file_path):
                    # Add the file itself to results if not already there
                    file_already_found = any(r['file'] == request.query for r in results)
                    if not file_already_found:
                        results.insert(0, {
                            "file": request.query,
                            "line": 1,
                            "content": f"File: {request.query}",
                            "match": "file_found"
                        })
                        count += 1
            except Exception as e:
                logger.warning(f"⚠️ Error in direct file search: {e}")
        
        return {
            "success": True,
            "query": request.query,
            "path": request.path,
            "results": results,
            "total_matches": count,
            "truncated": count >= request.max_results or time.time() - start_time > max_search_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching files: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching files: {str(e)}")

@app.post("/code_editor/run_command")
async def run_command(request: RunCommandRequest):
    """Run a shell command (use with caution!)"""
    try:
        # Basic command validation - you might want to make this more restrictive
        dangerous_commands = ['rm -rf', 'del', 'format', 'mkfs', 'dd if=', ':(){']
        if any(dangerous in request.command.lower() for dangerous in dangerous_commands):
            raise HTTPException(status_code=403, detail="Command not allowed")
        
        working_dir = CODE_EDITOR_BASE_DIR
        if request.working_dir:
            safe_dir = get_safe_path(CODE_EDITOR_BASE_DIR, request.working_dir)
            if safe_dir and os.path.isdir(safe_dir):
                working_dir = safe_dir
        
        # Run command with timeout
        result = subprocess.run(
            request.command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=request.timeout
        )
        
        return {
            "success": result.returncode == 0,
            "command": request.command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "working_dir": working_dir
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running command {request.command}: {e}")
        raise HTTPException(status_code=500, detail=f"Error running command: {str(e)}")

@app.post("/code_editor/create_backup")
async def create_backup(request: BackupRequest):
    """Create a timestamped backup of a file"""
    try:
        safe_path = get_safe_path(CODE_EDITOR_BASE_DIR, request.filepath)
        if not safe_path:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{safe_path}.backup.{timestamp}"
        
        shutil.copy2(safe_path, backup_path)
        
        return {
            "success": True,
            "original_file": request.filepath,
            "backup_file": os.path.relpath(backup_path, CODE_EDITOR_BASE_DIR),
            "timestamp": timestamp
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating backup for {request.filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating backup: {str(e)}")

@app.get("/code_editor/get_tree")
async def get_file_tree(path: str = ".", max_depth: int = 5):
    """Get a file tree structure for the explorer"""
    try:
        safe_path = get_safe_path(CODE_EDITOR_BASE_DIR, path)
        if not safe_path:
            raise HTTPException(status_code=400, detail="Invalid path")
        
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="Path not found")
        
        def build_tree(dir_path: str, current_depth: int = 0) -> dict:
            if current_depth >= max_depth:
                return None
            
            try:
                name = os.path.basename(dir_path) or "root"
                relative_path = os.path.relpath(dir_path, CODE_EDITOR_BASE_DIR)
                
                if os.path.isfile(dir_path):
                    return {
                        "name": name,
                        "type": "file",
                        "path": relative_path
                    }
                
                children = []
                try:
                    for item in sorted(os.listdir(dir_path)):
                        if item.startswith('.'):
                            continue
                        
                        item_path = os.path.join(dir_path, item)
                        child = build_tree(item_path, current_depth + 1)
                        if child:
                            children.append(child)
                except PermissionError:
                    pass
                
                return {
                    "name": name,
                    "type": "folder",
                    "path": relative_path,
                    "children": children,
                    "expanded": current_depth < 2  # Auto-expand first 2 levels
                }
            
            except (OSError, PermissionError):
                return None
        
        tree = build_tree(safe_path)
        if not tree:
            raise HTTPException(status_code=500, detail="Could not build file tree")
        
        return {
            "success": True,
            "tree": tree,
            "base_path": CODE_EDITOR_BASE_DIR
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file tree: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting file tree: {str(e)}")

# Streaming helper functions
async def stream_tool_calling_response(model_instance, messages, tools, tool_choice, temperature, max_tokens, seed):
    """Stream tool calling response with true token-by-token streaming"""
    try:
        # Handle both async generators and regular generators
        # Use original max_tokens for tool calling to ensure complete responses
        generator = model_instance.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            seed=seed
        )
        
        # Check if it's an async generator or regular generator
        if hasattr(generator, '__aiter__'):
            # Async generator
            async for chunk in generator:
                if chunk:
                    # Extract and stream individual tokens
                    async for token in _stream_chunk_tokens_async(chunk):
                        yield token
        else:
            # Regular generator - convert to async
            accumulated_content = ""
            for chunk in generator:
                if chunk:
                    # Extract and stream individual tokens
                    async for token in _stream_chunk_tokens_async(chunk):
                        yield token
                    
                    # Track accumulated content to detect incomplete tool calls
                    if isinstance(chunk, dict) and 'choices' in chunk and chunk['choices']:
                        choice = chunk['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            accumulated_content += choice['delta']['content']
            
            # Check if we have an incomplete tool call
            if accumulated_content and not accumulated_content.strip().endswith('}'):
                logger.warning(f"⚠️ [STREAM DEBUG] Incomplete tool call detected: {accumulated_content}")
                # Could implement completion logic here if needed
        
        # Send done signal
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        logger.error(f"❌ Streaming tool calling error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

async def stream_standard_response(model_instance, messages, temperature, max_tokens, seed):
    """Stream standard chat completion response with true token-by-token streaming"""
    try:
        # Handle both async generators and regular generators
        # Use original max_tokens for standard responses
        generator = model_instance.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            seed=seed
        )
        
        # Check if it's an async generator or regular generator
        if hasattr(generator, '__aiter__'):
            # Async generator
            async for chunk in generator:
                if chunk:
                    # Extract and stream individual tokens
                    async for token in _stream_chunk_tokens_async(chunk):
                        yield token
        else:
            # Regular generator - convert to async
            for chunk in generator:
                if chunk:
                    # Extract and stream individual tokens
                    async for token in _stream_chunk_tokens_async(chunk):
                        yield token
        
        # Send done signal
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        logger.error(f"❌ Streaming standard response error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

async def _stream_chunk_tokens_async(chunk):
    """Extract individual tokens from a chunk and stream them one by one (async version)"""
    try:
        # DEBUG: Log the actual chunk structure

        
        # Handle different chunk formats
        if isinstance(chunk, dict):
            # OpenAI-style chunk format
            if 'choices' in chunk and chunk['choices']:
                choice = chunk['choices'][0]
                if 'delta' in choice and 'content' in choice['delta']:
                    content = choice['delta']['content']
                    
                    if content:
                        # Stream each character as a separate token for true streaming
                        for char in content:
                            token_chunk = {
                                'choices': [{
                                    'delta': {'content': char},
                                    'index': 0,
                                    'finish_reason': None
                                }]
                            }
                            yield f"data: {json.dumps(token_chunk)}\n\n"
                        return
                elif 'delta' in choice and 'tool_calls' in choice['delta']:
                    # Handle tool call chunks
                    yield f"data: {json.dumps(chunk)}\n\n"
                    return
            # If no content delta, stream the whole chunk
            yield f"data: {json.dumps(chunk)}\n\n"
        elif isinstance(chunk, str):
            # String chunk - stream character by character
            for char in chunk:
                token_chunk = {
                    'choices': [{
                        'delta': {'content': char},
                        'index': 0,
                        'finish_reason': None
                    }]
                }
                yield f"data: {json.dumps(token_chunk)}\n\n"
        else:
            # Unknown chunk format - stream as is
            yield f"data: {json.dumps(chunk)}\n\n"
            
    except Exception as e:
        logger.error(f"❌ Error streaming chunk tokens: {e}")
        # Fallback: stream the original chunk
        yield f"data: {json.dumps(chunk)}\n\n"

# Add endpoint to set the working directory for code editor
@app.post("/code_editor/set_base_dir")
async def set_base_directory(path: str):
    """Set the base directory for code editor operations"""
    global CODE_EDITOR_BASE_DIR
    
    try:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        CODE_EDITOR_BASE_DIR = abs_path
        
        return {
            "success": True,
            "base_directory": CODE_EDITOR_BASE_DIR,
            "message": "Base directory updated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting base directory: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting base directory: {str(e)}")
    
@app.post("/v1/chat/completions/tools")
async def chat_completions_with_tools(
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """DEBUG: Find the None/empty content issue"""
    try:
        body = await request.json()
        
        messages = body.get('messages', [])
        tools = body.get('tools', [])
        tool_choice = body.get('tool_choice', 'auto')
        model = body.get('model', 'current')
        temperature = body.get('temperature', 0.7)
        max_tokens = body.get('max_tokens', 2048)
        stream = body.get('stream', False)
        seed = body.get('seed')  # ADD THIS LINE
        
        # Generate random seed if not provided
        if seed is None:
            import random
            seed = random.randint(0, 2147483647)
            logger.info(f"🎲 [DEBUG] Generated random seed: {seed}")
        else:
            logger.info(f"🎲 [DEBUG] Using provided seed: {seed}")

        # DEBUG: Check messages for None/empty content
        logger.info(f"🔍 [DEBUG] Checking {len(messages)} messages:")
        for i, msg in enumerate(messages):
            content = msg.get('content')
            role = msg.get('role')
            logger.info(f"  Message {i}: role='{role}', content_type={type(content)}, content_length={len(str(content)) if content else 0}")
            if content is None or content == "":
                logger.warning(f"  ⚠️  Message {i} has empty/None content!")
            if not isinstance(content, str):
                logger.warning(f"  ⚠️  Message {i} content is not a string: {repr(content)}")
        
        # Find a suitable model to use
        model_instance = None
        model_name = None
        gpu_id = None
        is_devstral = False
        
        # Strategy 1: Try to get a model assigned to 'test_model' purpose
        purposes = model_manager.get_models_by_purpose()
        if purposes.get('test_model') and purposes['test_model']['is_loaded']:
            model_info = purposes['test_model']
            model_name = model_info['name']
            gpu_id = model_info['gpu_id']
            model_instance = model_manager.get_model(model_name, gpu_id)
            is_devstral = DevstralHandler.is_devstral_model(model_name)
            logger.info(f"🎯 Using test_model: {model_name} on GPU {gpu_id}")
        
        # Strategy 2: Try to get any loaded model
        elif model_manager.loaded_models:
            # Get the first available loaded model
            model_key = next(iter(model_manager.loaded_models.keys()))
            model_name, gpu_id = model_key
            model_instance = model_manager.get_model(model_name, gpu_id)
            is_devstral = DevstralHandler.is_devstral_model(model_name)
            logger.info(f"🎯 Using first available model: {model_name} on GPU {gpu_id}")
        
        # No models loaded
        else:
            loaded_info = model_manager.get_loaded_models()
            raise HTTPException(
                status_code=400, 
                detail=f"No models are currently loaded. Available models: {loaded_info}"
            )
        
        logger.info(f"📝 Chat request - Model: {model_name}, GPU: {gpu_id}, Devstral: {is_devstral}, Tools: {len(tools) if tools else 0}")
        
        if is_devstral and tools:
            # Use Devstral with tool calling support
            logger.info(f"🔧 Using Devstral tool calling with {len(tools)} tools")
            
            # Format tools for Devstral
            formatted_tools = DevstralHandler.format_tools_for_devstral(tools)
            
            try:
                # Clean messages before sending to model
                cleaned_messages = []
                for msg in messages:
                    content = msg.get('content')
                    if content is not None and content.strip() != "":
                        cleaned_messages.append(msg)
                    elif msg.get('tool_calls'):
                        # Keep messages with tool calls - preserve original content and tool calls
                        cleaned_messages.append(msg)
                    else:
                        logger.warning(f"🧹 Skipping message with empty content: {msg}")
                
                logger.info(f"🔍 [DEBUG] Cleaned messages count: {len(cleaned_messages)} (was {len(messages)})")
                
                # Handle streaming vs non-streaming for tool calling
                if stream:
                    logger.info(f"🔄 Streaming tool calling response")
                    return StreamingResponse(
                        stream_tool_calling_response(
                            model_instance, cleaned_messages, formatted_tools, 
                            tool_choice, temperature, max_tokens, seed
                        ),
                        media_type="text/plain"
                    )
                else:
                    # Non-streaming tool calling
                    response = model_instance.create_chat_completion(
                        messages=cleaned_messages,
                        tools=formatted_tools,
                        tool_choice=tool_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                        seed=seed
                    )
                    
                    # Parse response for text-based tool calls
                    if isinstance(response, dict) and 'choices' in response and response['choices']:
                        choice = response['choices'][0]
                        message = choice.get('message', {})
                        content = message.get('content', '')
                        
                        # Check for structured tool calls first
                        if message.get('tool_calls'):
                            logger.info(f"🔧 Found structured tool calls: {len(message['tool_calls'])}")
                            return response
                        
                        # Parse text-based tool calls
                        logger.info(f"🔍 [DEBUG] Attempting to parse tool calls from content: {repr(content)}")
                        parsed_tool_calls, remaining_content = DevstralToolCallParser.extract_tool_calls_from_content(content)
                        
                        if parsed_tool_calls:
                            logger.info(f"🔧 Parsed {len(parsed_tool_calls)} tool calls from content")
                            logger.info(f"🔧 Tool calls: {[tc['function']['name'] for tc in parsed_tool_calls]}")
                            
                            # Modify the response to include structured tool calls
                            message['tool_calls'] = parsed_tool_calls
                            # For tool call messages, preserve the original content if it exists
                            # This ensures the tool call information is maintained
                            if content and content.strip():
                                message['content'] = content
                            else:
                                # If no content, use a placeholder that indicates this was a tool call
                                message['content'] = f"[Tool call: {', '.join([tc['function']['name'] for tc in parsed_tool_calls])}]"
                            
                            return response
                        else:
                            logger.warning(f"⚠️ No tool calls found in content: {content[:100]}...")
                            logger.info(f"🔍 [DEBUG] Content type: {type(content)}, length: {len(content) if content else 0}")
                            logger.info(f"🔍 [DEBUG] Full content: {repr(content)}")
                    
                    return response
                
            except Exception as tool_error:
                logger.error(f"❌ Devstral tool calling failed: {tool_error}")
                # Fallback to manual injection if needed
                logger.info("🔄 Falling back to manual tool injection")
        
        # Fallback: Manual tool injection or standard chat
        if tools and not is_devstral:
            logger.info(f"🔧 Using manual tool injection for non-Devstral model")
            
            # Your existing manual tool injection code
            tool_descriptions = []
            for tool in tools:
                func = tool.get('function', {})
                tool_descriptions.append(f"- {func.get('name')}: {func.get('description')}")
            
            tools_text = f"\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
            
            # Find system message or create one
            system_message = None
            for msg in messages:
                if msg.get('role') == 'system':
                    system_message = msg
                    break
            
            if system_message:
                system_message['content'] += tools_text
            else:
                messages.insert(0, {
                    'role': 'system',
                    'content': f"You are a helpful coding assistant with access to tools.{tools_text}"
                })
        
        # Clean messages for standard completion too
        cleaned_messages = []
        for msg in messages:
            content = msg.get('content')
            if content is not None and content.strip() != "":
                cleaned_messages.append(msg)
            elif msg.get('tool_calls'):
                # Keep messages with tool calls - preserve original content and tool calls
                cleaned_messages.append(msg)
            else:
                logger.warning(f"🧹 Skipping message with empty content in standard completion: {msg}")
        
        if not cleaned_messages:
            raise HTTPException(status_code=400, detail="No valid messages after cleaning")
        
        logger.info(f"🔍 [DEBUG] Standard completion with {len(cleaned_messages)} cleaned messages")
        
        # Handle streaming vs non-streaming for standard completion
        if stream:
            logger.info(f"🔄 Streaming standard chat completion")
            return StreamingResponse(
                stream_standard_response(
                    model_instance, cleaned_messages, temperature, max_tokens, seed
                ),
                media_type="text/plain"
            )
        else:
            # Standard chat completion (non-streaming)
            logger.info(f"💬 Standard chat completion (non-streaming)")
            response = model_instance.create_chat_completion(
                messages=cleaned_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                seed=seed
            )
            
            # DEBUG: Check response structure
            logger.info(f"🔍 [DEBUG] Response type: {type(response)}")
            if response is None:
                logger.error(f"❌ [DEBUG] Response is None!")
                raise HTTPException(status_code=500, detail="Model returned None response")
            
            if isinstance(response, dict):
                logger.info(f"🔍 [DEBUG] Response keys: {response.keys()}")
                choices = response.get('choices')
                logger.info(f"🔍 [DEBUG] Choices: {choices} (type: {type(choices)})")
                if choices is None or len(choices) == 0:
                    logger.error(f"❌ [DEBUG] No choices in response!")
                    raise HTTPException(status_code=500, detail="Model returned no choices")
            
            return response
        
    except Exception as e:
        logger.error(f"❌ [DEBUG] Error in chat completions: {e}")
        import traceback
        logger.error(f"❌ [DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/current-status")
async def get_current_model_status(model_manager: ModelManager = Depends(get_model_manager)):
    """Get current model status for debugging"""
    try:
        purposes = model_manager.get_models_by_purpose()
        loaded_models = model_manager.get_loaded_models()
        
        # Find which model would be used for chat
        active_model = None
        if purposes.get('test_model') and purposes['test_model']['is_loaded']:
            active_model = {
                "source": "test_model_purpose",
                "name": purposes['test_model']['name'],
                "gpu_id": purposes['test_model']['gpu_id'],
                "is_devstral": DevstralHandler.is_devstral_model(purposes['test_model']['name'])
            }
        elif model_manager.loaded_models:
            model_key = next(iter(model_manager.loaded_models.keys()))
            model_name, gpu_id = model_key
            active_model = {
                "source": "first_available",
                "name": model_name,
                "gpu_id": gpu_id,
                "is_devstral": DevstralHandler.is_devstral_model(model_name)
            }
        
        return {
            "active_model": active_model,
            "purposes": purposes,
            "loaded_models": loaded_models
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return {"error": str(e)}

async def load_devstral_with_tools(model_path: str):
    """Load Devstral model with tool calling support"""
    try:
        # When loading with llama.cpp, use these flags:
        # --jinja --chat-template-file path/to/mistral-tool-template.jinja
        
        # This would integrate with your existing model loading system
        # You'd need to modify your model loading to include tool calling flags
        
        command = [
            "llama-server",  # or whatever binary you use
            "--model", model_path,
            "--jinja",  # Enable jinja templating
            "--host", "0.0.0.0",
            "--port", "8001",  # or whatever port
            "--ctx-size", "32768",
            "--n-gpu-layers", "99",  # Adjust for your GPU
            # Add tool calling specific flags here
        ]
        
        # Start the server process
        # You'd integrate this with your existing model management
        
        logger.info(f"Loading Devstral with tool calling: {' '.join(command)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading Devstral with tools: {e}")
        return False    

@app.get("/models/devstral-info")
async def get_devstral_info(model_manager: ModelManager = Depends(get_model_manager)):
    """Get information about loaded Devstral model"""
    try:
        if not model_manager.primary_model:
            return {"error": "No model loaded"}
        
        model_instance = model_manager.primary_model
        is_devstral = getattr(model_instance, '_is_devstral', False)
        
        if not is_devstral:
            return {"is_devstral": False}
        
        # Try to get model metadata
        model_info = {
            "is_devstral": True,
            "tool_calling_supported": True,
            "chat_format": getattr(model_instance, 'chat_format', 'unknown'),
            "model_path": getattr(model_instance, 'model_path', 'unknown')
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting Devstral info: {e}")
        return {"error": str(e)}

@app.post("/code_editor/execute_tool")
async def execute_tool_call(tool_name: str, arguments: dict):
    """Execute a tool call and return the result"""
    try:
        if tool_name == "read_file":
            result = await read_file(FileOperationRequest(**arguments))
            return {"success": True, "result": result}
            
        elif tool_name == "write_file":
            result = await write_file(FileOperationRequest(**arguments))
            return {"success": True, "result": result}
            
        elif tool_name == "search_files":
            result = await search_files(SearchFilesRequest(**arguments))
            return {"success": True, "result": result}
            
        elif tool_name == "list_directory":
            result = await list_directory(DirectoryListRequest(**arguments))
            return {"success": True, "result": result}
            
        elif tool_name == "run_command":
            result = await run_command(RunCommandRequest(**arguments))
            return {"success": True, "result": result}
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution error: {str(e)}")

@router.post("/models/update-tensor-split")
async def update_tensor_split(data: dict = Body(...)):
    """Update tensor split settings for unified model mode - supports 2+ GPUs"""
    try:
        tensor_split = data.get("tensor_split")
        
        if not tensor_split:
            raise HTTPException(status_code=400, detail="tensor_split is required")
        
        if not isinstance(tensor_split, list) or len(tensor_split) < 2:
            raise HTTPException(status_code=400, detail="tensor_split must be a list of at least 2 values")
        
        # Normalize values to sum to 1.0
        total = sum(tensor_split)
        if total <= 0:
            raise HTTPException(status_code=400, detail="tensor_split values must be positive")
        
        normalized_split = [val / total for val in tensor_split]
        
        # Load existing settings
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
        
        # Update tensor split
        settings["tensor_split"] = normalized_split
        
        # Save back to file
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"✅ Updated tensor_split to {normalized_split}")
        
        return {
            "status": "success",
            "message": "Tensor split updated successfully. Reload model for changes to take effect.",
            "tensor_split": normalized_split
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating tensor split: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/get-tensor-split")
async def get_tensor_split():
    """Get current tensor split settings"""
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                tensor_split = settings.get("tensor_split")
                
                if tensor_split:
                    return {
                        "status": "success",
                        "tensor_split": tensor_split
                    }
        
        # Return default if not found
        # Default: CUDA0 (5090 32GB) gets 57%, CUDA1 (3090 24GB) gets 43%
        return {
            "status": "success",
            "tensor_split": [0.57, 0.43],
            "is_default": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting tensor split: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/get-settings")
async def get_settings():
    """Get all settings from settings.json"""
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                return {
                    "status": "success",
                    "settings": settings
                }
        
        # Return empty settings if file doesn't exist
        return {
            "status": "success",
            "settings": {}
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# These endpoints are now handled in document_routes.py

# mount the "generate" router
app.include_router(router)

# mount your memory endpoints under the "/memory" prefix
app.include_router(memory_router, prefix="/memory")
app.include_router(memory_router, prefix="/memory", tags=["memory"])
app.include_router(document_router)
# Add OpenAI compatibility layer
app.include_router(openai_router)
# TTS service router removed - TTS now runs separately on port 8002
logger.info("🔗 OpenAI-compatible API endpoints available at /v1/chat/completions and /v1/models")

