from __future__ import annotations

import os
from .compute_capabilities import disable_incompatible_torchao, force_cpu_mode

disable_incompatible_torchao()

# Disable problematic Torch optimizations for Python 3.12+ (MUST BE AT TOP)
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
if os.environ.get("MIRID_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("RAG_EMBEDDING_DEVICE", "cpu")

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
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
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
from .module_policy import module_enabled
from .cors_policy import configure_cors
from . import memory_intelligence
from . import character_intelligence
from .memory_routes import memory_router
from .alignment_routes import alignment_router
if module_enabled("chatlog_condenser"):
    from .chatlog_condenser_routes import chatlog_condenser_router
else:
    chatlog_condenser_router = None
import httpx
import logging
from fastapi.logger import logger # Use FastAPI's logger
import sys
import time
from . import openai_compat
import shutil
if module_enabled("forensics"):
    from .forensic_linguistics_service import ForensicLinguisticsService, TextDocument, SimilarityScore
import uuid
from .tts_client import TTSClient  # Use TTS client instead of direct service
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import re
from .model_manager import CTRANSFORMERS_AVAILABLE, LLAMA_CPP_AVAILABLE, ModelManager
from . import inference
import io
import yaml
import tempfile
from .stt_service import transcribe_audio # Assuming this is the correct import path for your STT service
from .inference import generate_text
from . import dual_chat_utils as dcu # Assuming this is the correct import path for your dual chat util
from . import chat_template_engine
import base64
from urllib.parse import urlparse
import threading
from .Document_routes import document_router
from . import rag_utils # Assuming this is the correct import path for your RAG utils
import logging
from PIL import Image, PngImagePlugin
import requests
from io import BytesIO
from .voice_sculpt_routes import voice_sculpt_router
from .automation_service import AutomationService
# Configure logging BEFORE importing modules that use it
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Suppress llama.cpp CUDA Graph spam (C++ backend writes to stderr directly)
import sys
class CudaGraphFilter:
    def filter(self, record):
        msg = record.getMessage()
        return "CUDA Graph" not in msg

for name in ["llama_cpp", "llama.cpp", ""]:
    logging.getLogger(name).addFilter(CudaGraphFilter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from .rembg_routes import router as rembg_router
import rembg
ELECTION_FUNDAMENTAL_WEIGHT_BASE_DEFAULT = 0.10
ELECTION_TIME_DECAY_CURVE_DEFAULT = "decay"
ELECTION_STATE_LEAN_MULTIPLIER_DEFAULT = 1.0
if module_enabled("elections"):
    from . import election_forecast
    from .election_db import election_db
    from .election_data_service import election_service, normalize_rtwh_polls
    from . import votehub_service
    from . import rcp_service
    from .election_ai_service import election_ai_service
    from . import election_simulation
    from . import ballotpedia_scraper
    ELECTION_FUNDAMENTAL_WEIGHT_BASE_DEFAULT = election_forecast.FUNDAMENTAL_WEIGHT_BASE_DEFAULT
    ELECTION_TIME_DECAY_CURVE_DEFAULT = election_forecast.TIME_DECAY_CURVE_DEFAULT
    ELECTION_STATE_LEAN_MULTIPLIER_DEFAULT = election_forecast.STATE_LEAN_MULTIPLIER_DEFAULT
if module_enabled("chess"):
    from .chess_auth_db import chess_auth_db
    from . import chess_ai_service
    from .chess_engine import chess_engine_service
    from .auth_routes import auth_router
else:
    auth_router = None
if module_enabled("market"):
    try:
        from .market_sim.routes import router as market_sim_router
    except ModuleNotFoundError:
        market_sim_router = None
else:
    market_sim_router = None
from .outreach_routes import router as outreach_router
from .remote_routes import router as remote_router
from .d_id_routes import d_id_router
from .sanctuary import sanctuary_router
from .model_library import router as model_library_router
from .character_datasets import router as character_datasets_router
from .provider_catalog import router as provider_catalog_router
from .sillytavern_bridge import router as sillytavern_bridge_router
from .mirid_docs import router as mirid_docs_router
from .avatar_store import (
    AVATAR_EXTENSIONS,
    avatar_storage_directory,
    contained_regular_file,
    migrate_legacy_avatar_files,
    persistent_avatar_path,
    resolve_stored_avatar_file,
)
from .settings_store import (
    SettingsStoreError,
    create_settings_backup,
    load_settings as load_settings_file,
    restore_settings_backup,
    update_settings as update_settings_file,
)

# --- Update status tracking ---
UPDATE_LOCK = threading.Lock()
UPDATE_STATE = {
    "status": "idle",
    "step": None,
    "logs": [],
    "error": None,
    "started_at": None,
    "finished_at": None,
    "update_id": None,
    "updated": None,
    "before": None,
    "after": None,
    "restart_recommended": False,
    "stash_used": False,
    "stash_name": None,
    "stash_applied": None,
    "stash_conflicts": False,
}

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _update_state(patch: Dict[str, Any]) -> None:
    with UPDATE_LOCK:
        UPDATE_STATE.update(patch)

def _append_update_log(level: str, message: str) -> None:
    entry = {"ts": _now_iso(), "level": level, "message": message}
    with UPDATE_LOCK:
        UPDATE_STATE["logs"].append(entry)
        if len(UPDATE_STATE["logs"]) > 500:
            UPDATE_STATE["logs"] = UPDATE_STATE["logs"][-500:]

def _get_update_state() -> Dict[str, Any]:
    with UPDATE_LOCK:
        return dict(UPDATE_STATE)

def get_log_dir():
    """Resolve the log directory (project-root logs/ by default)."""
    env_dir = os.environ.get("MIRID_LOG_DIR") or os.environ.get("ELOQUENT_LOG_DIR")
    if env_dir:
        log_dir = Path(env_dir)
    else:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_repo_root() -> Path:
    """Resolve the git repo root (project root)."""
    return Path(__file__).resolve().parents[2]

def get_tts_export_dir() -> Path:
    """Reliable server-side folder for full-response TTS backups."""
    export_dir = get_repo_root() / "backend" / "data" / "tts_full_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

def _safe_name(value: str, fallback: str = "item", max_len: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or fallback)).strip("_.-")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


def _split_tts_wav_bytes_by_max_duration(
    audio_bytes: bytes, max_seconds: float
) -> Optional[List[bytes]]:
    """
    Split decoded audio into multiple WAV byte blobs, each at most max_seconds.
    Returns None to keep the original bytes as a single file (short audio, decode failure, etc.).
    """
    if max_seconds <= 0:
        return None
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return None
    try:
        bio = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(bio, dtype="float64", always_2d=True)
    except Exception:
        return None
    if data.size == 0 or sample_rate <= 0:
        return None
    max_frames = int(float(max_seconds) * float(sample_rate))
    n_frames = int(data.shape[0])
    if max_frames <= 0 or n_frames <= max_frames:
        return None
    chunks: List[bytes] = []
    start = 0
    while start < n_frames:
        end = min(start + max_frames, n_frames)
        part = data[start:end].copy()
        out = io.BytesIO()
        try:
            sf.write(out, part, sample_rate, format="WAV", subtype="PCM_16")
        except Exception:
            return None
        chunks.append(out.getvalue())
        start = end
    return chunks if len(chunks) > 1 else None


def persist_full_tts_audio(
    audio_bytes: bytes,
    metadata: Dict[str, Any],
    max_chunk_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Persist synthesized full-response TTS to disk.
    Writes a primary file and an immediate backup copy plus manifest line(s).
    When max_chunk_seconds > 0, splits into multiple WAVs (each <= that duration) for downstream tools (e.g. ~5 min caps).
    """
    export_dir = get_tts_export_dir()
    backups_dir = export_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    message_id = _safe_name(metadata.get("message_id", "msg"))
    content_hash = hashlib.sha256(audio_bytes).hexdigest()[:10]

    split_blobs: Optional[List[bytes]] = None
    if max_chunk_seconds is not None and float(max_chunk_seconds) > 0:
        split_blobs = _split_tts_wav_bytes_by_max_duration(audio_bytes, float(max_chunk_seconds))

    manifest_path = export_dir / "manifest.jsonl"

    if split_blobs:
        paths: List[str] = []
        filenames: List[str] = []
        total_parts = len(split_blobs)
        for i, blob in enumerate(split_blobs):
            filename = f"tts_full_{now}_{message_id}_{content_hash}_part{i:03d}.wav"
            file_path = export_dir / filename
            backup_path = backups_dir / filename
            file_path.write_bytes(blob)
            backup_path.write_bytes(blob)
            paths.append(str(file_path))
            filenames.append(filename)
            manifest_row = {
                "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "filename": filename,
                "path": str(file_path),
                "backup_path": str(backup_path),
                "bytes": len(blob),
                "voice": metadata.get("voice"),
                "engine": metadata.get("engine"),
                "message_id": metadata.get("message_id"),
                "conversation_id": metadata.get("conversation_id"),
                "text_preview": (metadata.get("text") or "")[:180],
                "part_index": i,
                "parts_total": total_parts,
                "max_chunk_seconds": float(max_chunk_seconds),
            }
            with manifest_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

        return {
            "status": "saved",
            "path": paths[0],
            "backup_path": str(backups_dir / filenames[0]),
            "filename": filenames[0],
            "paths": paths,
            "filenames": filenames,
            "chunk_count": total_parts,
        }

    filename = f"tts_full_{now}_{message_id}_{content_hash}.wav"
    file_path = export_dir / filename
    backup_path = backups_dir / filename
    file_path.write_bytes(audio_bytes)
    backup_path.write_bytes(audio_bytes)

    manifest_row = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "filename": filename,
        "path": str(file_path),
        "backup_path": str(backup_path),
        "bytes": len(audio_bytes),
        "voice": metadata.get("voice"),
        "engine": metadata.get("engine"),
        "message_id": metadata.get("message_id"),
        "conversation_id": metadata.get("conversation_id"),
        "text_preview": (metadata.get("text") or "")[:180],
    }
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    return {
        "status": "saved",
        "path": str(file_path),
        "backup_path": str(backup_path),
        "filename": filename,
        "chunk_count": 1,
    }

def run_git_command(args: List[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="git is not installed or not on PATH.")

def _run_git_with_logs(args: List[str], cwd: Path, step: str, timeout: int = 30) -> subprocess.CompletedProcess:
    _update_state({"step": step})
    _append_update_log("info", f"$ git {' '.join(args)}")
    result = run_git_command(args, cwd, timeout=timeout)
    if result.stdout:
        _append_update_log("stdout", result.stdout.strip())
    if result.stderr:
        _append_update_log("stderr", result.stderr.strip())
    return result

# File logging for backend logs (one file per port)
disable_backend_file_log = os.environ.get("ELOQUENT_DISABLE_BACKEND_LOG", "").lower() in ("1", "true", "yes")
try:
    if not disable_backend_file_log:
        log_dir = get_log_dir()
        log_path_env = os.environ.get("BACKEND_LOG_PATH")
        if log_path_env:
            log_path = Path(log_path_env)
        else:
            port_label = os.environ.get("PORT", "unknown")
            log_path = log_dir / f"backend_{port_label}.log"
        root_logger = logging.getLogger()
        if not any(
            isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_path)
            for handler in root_logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root_logger.addHandler(file_handler)
            for uvicorn_logger_name in ("uvicorn.error", "uvicorn.access"):
                uvicorn_logger = logging.getLogger(uvicorn_logger_name)
                if not any(
                    isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_path)
                    for handler in uvicorn_logger.handlers
                ):
                    uvicorn_logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not initialize file logging: {e}")

from .sd_worker import SDWorkerClient
from .upscale_manager import UpscaleManager
import random
import hashlib
from .web_search_service import perform_web_search, set_web_search_llm
from .eloquent_agent_tools import (
    build_web_search_receipt,
    detect_article_research_intent,
    gather_reliable_web_research,
    get_eloquent_chat_tools,
    supports_native_tool_calling,
    WEB_SEARCH_MODEL_INSTRUCTIONS,
    extract_tool_calls_from_text as agent_extract_tool_calls,
    execute_eloquent_tool,
)
from .web_search_routing import (
    apply_native_web_search_request,
    build_search_meta,
    get_endpoint_config_for_model,
    resolve_web_search_path,
    sources_from_results,
)
from .openai_compat import (
    router as openai_router,
    is_api_endpoint,
    get_configured_endpoint,
    forward_to_configured_endpoint_streaming,
    forward_to_configured_endpoint_non_streaming,
    collect_openai_compatible_stream_text,
    prepare_endpoint_request,
    prepare_endpoint_request_from_config,
    resolve_flow_api_endpoint_config,
    is_flow_dedicated_api_request,
    INTRO_ABOUT_PURPOSES,
    apply_nano_gpt_context_memory,
    extract_openai_stream_delta_parts,
    thinking_stream_debug_enabled,
    log_generate_outbound,
    model_id_implies_extended_thinking,
    parse_eloquent_llm_prompt_to_openai_messages,
    inject_openai_vision_into_messages,
    validate_api_model_for_generate,
    note_endpoint_failure,
    note_endpoint_success,
)
try:
    import pynvml
except ImportError:
    pynvml = None
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
    memoryEnabled: bool = True
    prompt: str
    model_name: str
    max_tokens: int = 1_000_000
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
    rag_agent_tools: bool = False
    gpu_id: Optional[int] = None
    userProfile: Optional[Dict[str, Any]] = None
    is_dual_chat: bool = False # Added for dual chat support
    messages: Optional[List[Dict[str, Any]]] = None  # To support the frontend's messages array
    chat_template_id: Optional[str] = None
    chat_template_messages: Optional[List[Dict[str, Any]]] = None
    echo: bool = False # Added for echo functionality
    active_character: Optional[Dict[str, Any]] = None  # Add this field
    request_purpose: Optional[str] = None # <<< ADD THIS LINE
    selected_model: Optional[str] = None
    round_robin_enabled: Optional[bool] = None
    memory_curation: Optional[bool] = False
    use_web_search: bool = False  # NEW: Web search toggle
    web_search_query: Optional[str] = None  # NEW: Optional web search query
    web_search_mode: Optional[str] = None  # Retired compatibility field; automatic search ignores it
    web_search_strategy: Optional[str] = None  # Retired compatibility field; always automatic
    # None = auto (agent tools for API endpoints, legacy inject for local GGUF)
    use_agent_tools: Optional[bool] = None
    research_urls: Optional[List[str]] = None  # explicit article URLs to fetch
    research_site: Optional[str] = None  # e.g. uxmag.com for site: searches
    image_base64: Optional[str] = None  # NEW: Optional base64-encoded image for context
    image_type: Optional[str] = None  # NEW: Optional image type (e.g., "png", "jpg")
    images: Optional[List[Dict[str, str]]] = None  # Multiple image attachments
    vision_model: Optional[str] = None  # NEW: Vision model name (e.g., "LFM2.5-VL-450M-Extract")
    vision_schema: Optional[str] = None  # NEW: Optional YAML schema for structured extraction
    use_local_vision: Optional[bool] = None  # NEW: Enable/disable local vision fallback
    authorNote: Optional[str] = None  # Author's note for custom session instructions
    summaryContext: Optional[str] = None  # Optional story summary context
    injectTimestamp: Optional[bool] = False  # Prepend current date/time to context (same path as authorNote/summaryContext)
    userProfileReinforcement: Optional[str] = None  # Key phrases/bullets re-injected before user query to reinforce profile weight

    # NanoGPT Context Memory (only applied when custom endpoint URL contains nano-gpt.com)
    nano_gpt_context_memory_enabled: bool = False
    nano_gpt_context_memory_mode: str = "header"  # "header" | "suffix"
    nano_gpt_context_memory_expiration_days: int = 30

    # When True, skip OpenAI-compat local message pruning (used for huge one-shot prompts, e.g. Ethics review → Run).
    skip_openai_message_pruning: bool = False

    # Character-as-System: client sends base system layer + optional "Character Persona:" chat layer.
    system_persona_mode: bool = False

    # Optional per-flow API overrides (character_intro, system_intro, call_mode_character_about)
    flow_api_url: Optional[str] = None
    flow_api_model: Optional[str] = None
    flow_api_key: Optional[str] = None

    # Intensity preset parameters
    intensity_params: Optional[Dict[str, Any]] = None

    # Alignment failure detection
    enable_alignment_detection: bool = False
    alignment_thresholds: Optional[Dict[str, float]] = None

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
    top_k: int = 30
    # The handler forwards this to rag_utils.query_documents, which has always
    # expected it; without the field every /document/query request failed with
    # "'DocumentQuery' object has no attribute 'threshold'".
    threshold: float = 0.05

class FileOperationRequest(BaseModel):
    filepath: str
    content: Optional[str] = None

class DirectoryListRequest(BaseModel):
    path: str = "."
    include_hidden: bool = False

class SelectDirectoryRequest(BaseModel):
    initial_directory: Optional[str] = None
    title: Optional[str] = None

class SelectFileRequest(BaseModel):
    initial_directory: Optional[str] = None
    title: Optional[str] = None
    multiple: bool = False

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

@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "healthy", "service": "backend"}

# --- CORS Configuration ---
configure_cors(app)

# --- Authentication Middleware ---
@app.middleware("http")
async def check_authentication(request: Request, call_next):
    # Allow OPTIONS requests (CORS preflight) pass through
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path.startswith("/code_editor") or path.startswith("/devstral") or path.startswith("/forensic"):
        return JSONResponse(status_code=404, content={"detail": "Module not available"})
    if path.startswith("/chess") or path.startswith("/auth"):
        return JSONResponse(status_code=404, content={"detail": "Module not available"})
    if path.startswith("/election") and not module_enabled("elections"):
        return JSONResponse(status_code=404, content={"detail": "Election module is disabled"})
    if path.startswith("/market") and not module_enabled("market"):
        return JSONResponse(status_code=404, content={"detail": "Market module is disabled"})

    # List of paths that don't need auth (static files, health checks, etc.)
    # Note: websocket connections are handled separately or assume trusted for now if upgraded
    public_paths = ["/health", "/static", "/favicon.ico", "/docs", "/openapi.json"]
    if any(request.url.path.startswith(path) for path in public_paths):
        return await call_next(request)

    # 1. Get client IP
    # client_host = request.client.host
    # 2. Check if local (optional)
    
    # 3. Read settings to get the password
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    admin_password = ""
    try:
        if settings_path.exists():
            with open(settings_path, 'r', encoding='utf-8') as f:
                 data = json.load(f)
                 admin_password = data.get("admin_password", "")
    except Exception:
        pass 

    # 4. If password is set, check it
    if admin_password:
        auth_header = request.headers.get("Authorization")
        expected_scheme = "Basic"
        is_authenticated = False
        
        # Method A: Basic Auth (Standard)
        if auth_header and auth_header.startswith("Basic "):
             try:
                 encoded = auth_header.split(" ")[1]
                 decoded = base64.b64decode(encoded).decode("utf-8")
                 # format is "username:password"
                 _, password = decoded.split(":", 1)
                 if password == admin_password:
                     is_authenticated = True
             except:
                 pass
        
        # Method B: Bearer Token (Simpler for some clients)
        elif auth_header and auth_header.strip() == f"Bearer {admin_password}":
            is_authenticated = True
            
        if not is_authenticated:
            # Return 401 with standard Basic Auth challenge
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
                    headers={"WWW-Authenticate": "Basic realm='Mirid Remote Access'"}
            )

    return await call_next(request)

router = APIRouter(tags=["generate"])
router = APIRouter()   # Re-initialize router to avoid conflicts




# --- Startup Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    if module_enabled("chess"):
        from .download_book import ensure_chess_book_background
        ensure_chess_book_background()

# --- Static Files Setup ---

# --- Static Files Setup ---
# Define the base static directory path
# Using Path(__file__).parent makes the 'static' directory relative to the main.py file's location
base_dir = Path(__file__).parent
static_dir = base_dir / "static"
generated_images_dir = static_dir / "generated_images" # Define the subdirectory path
summaries_dir = static_dir / "summaries"
avatars_dir = avatar_storage_directory()

# Ensure both directories exist
try:
    static_dir.mkdir(parents=True, exist_ok=True)
    generated_images_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    avatars_dir.mkdir(parents=True, exist_ok=True)
    migrated_avatars = migrate_legacy_avatar_files(static_dir, avatars_dir)
    logger.info(f"Static directory ensured: {static_dir.resolve()}")
    logger.info(f"Generated images directory ensured: {generated_images_dir.resolve()}")
    logger.info(
        "Persistent avatar directory ensured: %s (migrated %s legacy files)",
        avatars_dir.resolve(),
        len(migrated_avatars),
    )
except OSError as e:
    logger.error(f"FATAL: Failed to create static directories: {e}", exc_info=True)
    # Depending on severity, you might want to exit here or raise an exception


@app.get("/static/{filename}", include_in_schema=False)
async def serve_root_static_file(filename: str):
    """Serve packaged root assets and persistent uploaded avatars."""
    if Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Static file not found")

    legacy_file = contained_regular_file(static_dir / filename, static_dir)
    if legacy_file is not None:
        return FileResponse(legacy_file)

    avatar_file = persistent_avatar_path(filename)
    if avatar_file is not None:
        avatar_file = contained_regular_file(avatar_file, avatars_dir)
        if avatar_file is not None:
            return FileResponse(avatar_file)

    raise HTTPException(status_code=404, detail="Static file not found")


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

CODE_EDITOR_BASE_DIR = os.getcwd()  # Default to current dir

# Try to load saved base dir from settings
try:
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    if settings_path.exists():
        with open(settings_path, 'r', encoding='utf-8') as f:
            saved_settings = json.load(f)
            saved_dir = saved_settings.get("code_editor_base_dir")
            if saved_dir and os.path.exists(saved_dir) and os.path.isdir(saved_dir):
                CODE_EDITOR_BASE_DIR = saved_dir
                logger.info(f"Loaded saved code editor directory: {CODE_EDITOR_BASE_DIR}")
except Exception as e:
    logger.warning(f"Failed to load saved code editor directory: {e}")


# Add this to your existing FastAPI app
async def generate_llm_response(prompt: str, model_manager, model_name: str, **kwargs) -> str:
    """
    Adapter for your existing LLM generation, handling both local and API models.
    """
    from .inference import generate_text
    from .openai_compat import is_api_endpoint, prepare_endpoint_request, collect_openai_compatible_stream_text

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
            endpoint_config, url, prepared_data = prepare_endpoint_request(model_name, request_data)
            text_out = await collect_openai_compatible_stream_text(endpoint_config, url, prepared_data)
            response_json = {"choices": [{"message": {"content": text_out}}]}
            
            # Extract content from OpenAI response format (with fallbacks)
            if isinstance(response_json, dict):
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    first = response_json['choices'][0]
                    if isinstance(first, str):
                        return first
                    if isinstance(first, dict):
                        # Standard chat format
                        msg = first.get('message')
                        if isinstance(msg, str):
                            return msg
                        if isinstance(msg, dict):
                            content = msg.get('content') or msg.get('text')
                            if isinstance(content, list):
                                # Some APIs return list of parts
                                joined = "".join([p.get("text") for p in content if isinstance(p, dict) and p.get("text")])
                                if joined:
                                    return joined
                            if isinstance(content, str) and content:
                                return content
                        # Some providers return `delta` even in non-streaming
                        delta = first.get('delta')
                        if isinstance(delta, dict):
                            content = delta.get('content') or delta.get('text')
                            if isinstance(content, str) and content:
                                return content
                        # Some providers return `text` directly in choices
                        content = first.get('text') or first.get('content')
                        if isinstance(content, str) and content:
                            return content
                # Other common top-level fields
                for key in ("text", "response", "output_text", "content", "result", "answer"):
                    content = response_json.get(key)
                    if isinstance(content, str) and content:
                        return content
            logger.error(f"Unexpected or empty API response format: {response_json}")
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

# NOTE: legacy in-process synthesize_speech removed — TTS lives in tts_service.py / TTS service on port 8002.
# --- Model Manager Dependency ---
# Assume app.state.model_manager is initialized in lifespan

    # No need to do anything here, as the lifespan will handle cleanup
# --- Election Tracker (first duplicate removed; DB-backed endpoints registered later) ---

@router.get("/system/runtime-capabilities")
async def get_runtime_capabilities(request: Request):
    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model support is still starting.")
    return await asyncio.to_thread(model_manager.get_runtime_capabilities)


@router.post("/system/runtime-test")
async def test_local_runtime(request: Request):
    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model support is still starting.")
    return await asyncio.to_thread(model_manager.test_local_runtime)

@router.get("/system/gpu_info")
async def get_gpu_info(request: Request):
    """Return current GPU inventory without initialising a CUDA context."""
    model_manager = getattr(request.app.state, "model_manager", None)
    capabilities = (
        await asyncio.to_thread(model_manager.get_runtime_capabilities)
        if model_manager is not None
        else {"formats": {"gguf": {"available": False, "selected": None}}}
    )
    gguf_support = capabilities.get("formats", {}).get("gguf", {})
    selected_runtime = gguf_support.get("selected") or {}
    selected_accelerator = selected_runtime.get("accelerator")
    if force_cpu_mode():
        return {
            "gpu_count": 0,
            "single_gpu_mode": True,
            "gpus": [],
            "cuda_available": False,
            "compute_mode": "cpu",
            "hosted_models_recommended": True,
            "local_gguf_available": bool(gguf_support.get("available")),
            "local_acceleration_available": False,
            "local_runtime": capabilities,
        }
    gpus = []
    try:
        if pynvml is None:
            raise RuntimeError("NVIDIA telemetry is not installed")
        pynvml.nvmlInit()
        for gpu_id in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            gpus.append({
                "id": gpu_id,
                "name": name,
                "total_mb": round(memory.total / (1024 * 1024)),
                "free_mb": round(memory.free / (1024 * 1024)),
                "used_mb": round(memory.used / (1024 * 1024)),
            })
    except Exception:
        gpus = []
    finally:
        try:
            if pynvml is not None:
                pynvml.nvmlShutdown()
        except Exception:
            pass
    detected_gpu_count = len(gpus) if gpus else check_gpu_count()
    compute_mode = selected_accelerator or ("nvidia" if detected_gpu_count > 0 else "cpu")
    accelerated = compute_mode != "cpu"
    return {
        "gpu_count": detected_gpu_count,
        "single_gpu_mode": getattr(request.app.state, 'single_gpu_mode', False),
        "gpus": gpus,
        "cuda_available": detected_gpu_count > 0,
        "compute_mode": compute_mode,
        "hosted_models_recommended": not accelerated,
        "local_gguf_available": bool(gguf_support.get("available")),
        "local_acceleration_available": accelerated,
        "local_runtime": capabilities,
    }


@router.get("/system/export-logs")
async def export_backend_logs():
    """Export backend log files as a single text response."""
    log_dir = get_log_dir()
    if not log_dir.exists():
        raise HTTPException(status_code=404, detail="Log directory not found")

    log_files = sorted(
        list(log_dir.glob("backend_*.log")) +
        list(log_dir.glob("tts_*.log")) +
        list(log_dir.glob("startup_report_*.txt"))
    )
    if not log_files:
        default_log = log_dir / "backend.log"
        if default_log.exists():
            log_files = [default_log]

    if not log_files:
        raise HTTPException(status_code=404, detail="No backend logs found")

    sections = []
    for log_file in log_files:
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = "[Could not read log file]"
        sections.append(f"===== {log_file.name} =====\n{content}")

    combined = "\n\n".join(sections)
    return Response(
        combined,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=backend_logs.txt"}
    )

@router.delete("/system/clear-logs")
async def clear_backend_logs():
    """Delete all backend log files from the project logs folder."""
    log_dir = get_log_dir()
    if not log_dir.exists():
        raise HTTPException(status_code=404, detail="Log directory not found")

    deleted = 0
    skipped = []
    for log_file in log_dir.iterdir():
        if not log_file.is_file():
            continue
        try:
            log_file.unlink()
            deleted += 1
        except PermissionError as e:
            skipped.append(f"{log_file.name}: {e}")
        except Exception as e:
            skipped.append(f"{log_file.name}: {e}")

    if skipped:
        logger.warning(f"Skipped deleting {len(skipped)} log files: {skipped}")

    return {"status": "success", "deleted": deleted, "skipped": skipped}

@router.get("/system/update-status")
async def get_update_status(fetch: bool = False):
    """Get git update status for the local repo."""
    repo_root = get_repo_root()
    if not (repo_root / ".git").exists():
        raise HTTPException(status_code=404, detail="Not a git repository.")

    if fetch:
        fetch_result = run_git_command(["fetch", "--prune"], repo_root, timeout=60)
        if fetch_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"git fetch failed: {fetch_result.stderr.strip() or fetch_result.stdout.strip()}"
            )

    branch_result = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if branch_result.returncode != 0:
        raise HTTPException(status_code=500, detail="Failed to resolve current branch.")
    branch = branch_result.stdout.strip()

    commit_result = run_git_command(["rev-parse", "HEAD"], repo_root)
    if commit_result.returncode != 0:
        raise HTTPException(status_code=500, detail="Failed to resolve current commit.")
    current_commit = commit_result.stdout.strip()

    dirty_result = run_git_command(["status", "--porcelain"], repo_root)
    if dirty_result.returncode != 0:
        raise HTTPException(status_code=500, detail="Failed to check git status.")
    dirty_output = dirty_result.stdout.strip()
    dirty_lines = dirty_output.splitlines() if dirty_output else []
    dirty = bool(dirty_lines)

    upstream = None
    ahead = None
    behind = None
    upstream_result = run_git_command(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        repo_root
    )
    if upstream_result.returncode == 0:
        upstream = upstream_result.stdout.strip()
        counts_result = run_git_command(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"], repo_root)
        if counts_result.returncode == 0:
            parts = counts_result.stdout.strip().split()
            if len(parts) == 2:
                try:
                    ahead = int(parts[0])
                    behind = int(parts[1])
                except ValueError:
                    pass

    return {
        "status": "success",
        "repo_root": str(repo_root),
        "branch": branch,
        "current_commit": current_commit,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "dirty": dirty,
        "dirty_count": len(dirty_lines)
    }

@router.post("/system/update")
async def update_system():
    """Start a background update of the git repo and return an update id."""
    with UPDATE_LOCK:
        if UPDATE_STATE.get("status") == "running":
            return JSONResponse(status_code=409, content={
                "status": "running",
                "message": "An update is already running.",
                "update_id": UPDATE_STATE.get("update_id")
            })

        update_id = uuid.uuid4().hex[:8]
        UPDATE_STATE.update({
            "status": "running",
            "step": "queued",
            "logs": [],
            "error": None,
            "started_at": _now_iso(),
            "finished_at": None,
            "update_id": update_id,
            "updated": None,
            "before": None,
            "after": None,
            "restart_recommended": False,
            "stash_used": False,
            "stash_name": None,
            "stash_applied": None,
            "stash_conflicts": False,
        })

    threading.Thread(target=_run_update_task, args=(update_id,), daemon=True).start()
    return {"status": "started", "update_id": update_id}

@router.get("/system/update-progress")
async def get_update_progress():
    """Get the current update progress and logs."""
    return _get_update_state()


def _run_update_task(update_id: str) -> None:
    repo_root = get_repo_root()
    _append_update_log("info", f"Update started (id={update_id})")

    if not (repo_root / ".git").exists():
        _update_state({
            "status": "failed",
            "step": "failed",
            "error": "Not a git repository.",
            "finished_at": _now_iso()
        })
        return

    try:
        _append_update_log(
            "warn",
            "Force update enabled: local changes will be discarded to match the latest git version."
        )

        upstream_result = _run_git_with_logs(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            repo_root,
            "Checking upstream"
        )
        if upstream_result.returncode != 0:
            raise RuntimeError("No upstream configured for the current branch.")

        before_result = _run_git_with_logs(["rev-parse", "HEAD"], repo_root, "Reading current commit")
        if before_result.returncode != 0:
            raise RuntimeError("Failed to resolve current commit.")
        before_commit = before_result.stdout.strip()
        _update_state({"before": before_commit})

        fetch_result = _run_git_with_logs(["fetch", "--prune"], repo_root, "Fetching updates", timeout=60)
        if fetch_result.returncode != 0:
            raise RuntimeError("git fetch failed")

        reset_result = _run_git_with_logs(
            ["reset", "--hard", "@{upstream}"],
            repo_root,
            "Resetting to latest commit",
            timeout=120
        )
        if reset_result.returncode != 0:
            raise RuntimeError("git reset --hard failed")

        clean_result = _run_git_with_logs(
            [
                "clean",
                "-fd",
                "-e",
                "frontend/node_modules",
                "-e",
                "frontend/.vite",
                "-e",
                "static",
                "-e",
                "backend/app/user_memories",
            ],
            repo_root,
            "Cleaning untracked files (preserving frontend deps + static assets + user memories)",
            timeout=120
        )
        if clean_result.returncode != 0:
            raise RuntimeError("git clean failed")

        after_result = _run_git_with_logs(["rev-parse", "HEAD"], repo_root, "Reading updated commit")
        if after_result.returncode != 0:
            raise RuntimeError("Failed to resolve updated commit.")
        after_commit = after_result.stdout.strip()
        updated = before_commit != after_commit

        _update_state({
            "status": "success",
            "step": "done",
            "updated": updated,
            "after": after_commit,
            "restart_recommended": updated,
            "stash_used": False,
            "stash_name": None,
            "stash_applied": None,
            "stash_conflicts": False,
            "finished_at": _now_iso()
        })

    except Exception as e:
        logger.error("Update failed: %s", e, exc_info=True)
        _append_update_log("error", str(e))
        _update_state({
            "status": "failed",
            "step": "failed",
            "error": str(e),
            "finished_at": _now_iso()
        })

@router.post("/system/select-directory")
async def select_directory(data: SelectDirectoryRequest = Body(...)):
    """Open a native directory picker on the server and return the selected path.
    Runs the picker in a subprocess so a Tk crash cannot kill the backend."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "tk_directory_picker.py"
    if not script_path.is_file():
        logger.error(f"Directory picker script not found: {script_path}")
        raise HTTPException(status_code=500, detail="Directory picker is unavailable on this system.")

    stdin_payload = json.dumps({
        "title": data.title,
        "initial_directory": data.initial_directory,
    })

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_payload.encode("utf-8")),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Directory picker timed out")
        raise HTTPException(status_code=500, detail="Directory picker timed out.")
    except Exception as exc:
        logger.error(f"Directory picker subprocess failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to open directory picker.")

    if proc.returncode == 1:
        return {"status": "cancelled"}
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip() or "Unknown error"
        logger.error(f"Directory picker error (code {proc.returncode}): {err}")
        raise HTTPException(status_code=500, detail=f"Directory picker failed: {err}")

    directory = (stdout or b"").decode("utf-8", errors="replace").strip()
    if not directory:
        return {"status": "cancelled"}
    return {"status": "success", "directory": directory}


@router.post("/system/select-file")
async def select_file(data: SelectFileRequest = Body(...)):
    """Open a native file picker on the server and return the selected path."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "tk_file_picker.py"
    if not script_path.is_file():
        logger.error(f"File picker script not found: {script_path}")
        raise HTTPException(status_code=500, detail="File picker is unavailable on this system.")

    stdin_payload = json.dumps({
        "title": data.title,
        "initial_directory": data.initial_directory,
        "multiple": data.multiple,
    })

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_payload.encode("utf-8")),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        logger.warning("File picker timed out")
        raise HTTPException(status_code=500, detail="File picker timed out.")
    except Exception as exc:
        logger.error(f"File picker subprocess failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to open file picker.")

    if proc.returncode == 1:
        return {"status": "cancelled"}
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip() or "Unknown error"
        logger.error(f"File picker error (code {proc.returncode}): {err}")
        raise HTTPException(status_code=500, detail=f"File picker failed: {err}")

    file_path = (stdout or b"").decode("utf-8", errors="replace").strip()
    if not file_path:
        return {"status": "cancelled"}
    if data.multiple:
        try:
            files = json.loads(file_path)
            if not isinstance(files, list):
                raise ValueError("not a list")
            files = [str(f).strip() for f in files if str(f).strip()]
            if not files:
                return {"status": "cancelled"}
            return {"status": "success", "files": files}
        except (json.JSONDecodeError, ValueError):
            return {"status": "success", "files": [file_path]}
    return {"status": "success", "file": file_path}


@router.post("/models/update-gpu-mode")
async def update_gpu_mode(
    data: dict = Body(...),
):
    """Update the GPU usage mode and save to settings file."""
    try:
        gpu_mode = data.get("gpuUsageMode")
        if gpu_mode not in ["split_services", "unified_model"]:
            raise HTTPException(status_code=400, detail="Invalid GPU usage mode")

        update_settings_file({"gpuUsageMode": gpu_mode})
        logger.info("Successfully saved gpuUsageMode=%s", gpu_mode)
        
        return {"status": "success", "message": "GPU usage mode updated"}
        
    except Exception as e:
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
            batch_id = hashlib.sha256(f"{person_name}_{time.time()}".encode()).hexdigest()[:8]
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


# --- STONED METER ANALYSIS ---
class StonedAnalyzeRequest(BaseModel):
    text: str
    verbose: bool = False

@router.post("/api/stoned/analyze")
async def analyze_stoned_endpoint(data: StonedAnalyzeRequest = Body(...)):
    """Analyze text for cannabis intoxication markers (StonerDetector)."""
    try:
        from docs.stonerdetector import analyze_intoxication
        result = analyze_intoxication(data.text, data.verbose)
        return {
            "status": "success",
            "analysis": result
        }
    except Exception as e:
        logger.error(f"❌ [StonedMeter] Error analyzing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export_character_png")
async def export_character_png(character_data: dict):
    """Export character as PNG with embedded JSON data in tEXt chunk."""
    try:
        avatar_url = character_data.get("avatar")
        provided_tavern_card = character_data.get("tavern_card")
        if (
            isinstance(provided_tavern_card, dict)
            and provided_tavern_card.get("spec") == "chara_card_v2"
            and isinstance(provided_tavern_card.get("data"), dict)
        ):
            tavern_data = provided_tavern_card
        else:
            extensions = dict(character_data.get("card_extensions") or {})
            mirid_extension = dict(extensions.get("mirid") or {})
            mirid_extension.update({
                "exported_at": datetime.datetime.now().isoformat(),
                "original_format": mirid_extension.get("original_format") or "mirid",
                "background": character_data.get("background", ""),
                "speech_style": character_data.get("speech_style", ""),
                "chat_role": "user" if character_data.get("chat_role") == "user" else "npc",
                "ethics_justification": (character_data.get("ethics_justification") or "").strip(),
                "avatars": character_data.get("avatars") or [],
                "activeAvatarIndex": character_data.get("activeAvatarIndex", 0),
            })
            extensions["mirid"] = mirid_extension
            tavern_data = {
                **dict(character_data.get("card_top_level") or {}),
                "spec": "chara_card_v2",
                "spec_version": "2.0",
                "data": {
                    **dict(character_data.get("card_data_passthrough") or {}),
                    "name": character_data.get("name", ""),
                    "description": character_data.get("description", ""),
                    "personality": character_data.get("personality", ""),
                    "scenario": character_data.get("scenario", ""),
                    "first_mes": character_data.get("first_message", ""),
                    "mes_example": "",
                    "creator_notes": character_data.get("creator_notes", ""),
                    "system_prompt": character_data.get("model_instructions", ""),
                    "post_history_instructions": character_data.get("post_history_instructions", ""),
                    "alternate_greetings": character_data.get("alternate_greetings") or [],
                    "tags": character_data.get("tags") or [],
                    "creator": character_data.get("creator") or "Mirid",
                    "character_version": character_data.get("character_version") or "1.0",
                    "extensions": extensions,
                },
            }

            example_lines = []
            for dialogue in character_data.get("example_dialogue") or []:
                if not dialogue.get("content"):
                    continue
                prefix = "{{user}}" if dialogue.get("role") == "user" else "{{char}}"
                example_lines.append(f"{prefix}: {dialogue.get('content', '')}")
            tavern_data["data"]["mes_example"] = "\n".join(example_lines)

            lore_entries = character_data.get("loreEntries") or []
            book_metadata = dict(character_data.get("character_book_metadata") or {})
            if lore_entries or book_metadata:
                book_metadata["name"] = book_metadata.get("name") or f"{character_data.get('name') or 'Character'} Lorebook"
                book_metadata["extensions"] = dict(book_metadata.get("extensions") or {})
                book_metadata["entries"] = []
                for index, entry in enumerate(lore_entries):
                    tavern_entry = dict(entry.get("tavern_entry") or {})
                    tavern_entry.update({
                        "id": tavern_entry.get("id", index),
                        "keys": entry.get("keywords") or [],
                        "content": entry.get("content", ""),
                        "extensions": dict(tavern_entry.get("extensions") or {}),
                        "enabled": tavern_entry.get("enabled", True),
                        "insertion_order": tavern_entry.get("insertion_order", index),
                    })
                    book_metadata["entries"].append(tavern_entry)
                tavern_data["data"]["character_book"] = book_metadata

        character_name = tavern_data.get("data", {}).get("name") or character_data.get("name") or "character"
        
        character_json = json.dumps(tavern_data)
        
        # Replace the avatar loading section with this:
        if avatar_url:
            try:
                logger.info(f"Attempting to load avatar: {avatar_url}")

                local_avatar = resolve_stored_avatar_file(avatar_url, static_dir)
                if local_avatar is not None:
                    img = Image.open(local_avatar)
                    logger.info(f"Loaded avatar from local path: {local_avatar}")
                elif avatar_url.startswith('http'):
                    response = requests.get(avatar_url, timeout=10)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    logger.info(f"Loaded avatar from external URL: {avatar_url}")
                else:
                    raise FileNotFoundError(f"Avatar not found: {avatar_url}")

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

def save_video_and_get_url(video_data: bytes) -> str:
    """Saves video data to a file and returns its static URL path."""
    generated_images_dir = base_dir / "static" / "generated_images"
    generated_images_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4()}.mp4"
    save_path = generated_images_dir / filename
    
    with open(save_path, "wb") as f:
        f.write(video_data)
        
    logger.info(f"Video saved to: {save_path}")
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
    # Read environment variables set by the desktop host.
    # Ensure these env vars are correctly set by your launch mechanism
    default_gpu = int(os.environ.get("GPU_ID", 0))
    port = int(os.environ.get("PORT", 8000 if default_gpu == 0 else 8001))
    tts_port = int(os.environ.get("TTS_PORT", 8002))  # TTS service port
    model_path_env = os.environ.get("MODEL_PATH", "")
    model_name_env = os.environ.get("MODEL_NAME", "")
    # NEW CODE START - Add right here
    gpu_count = check_gpu_count()
    
    # Unified settings loader
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    settings = {}
    try:
        logging.info(f"🔍 Looking for settings at: {settings_path}")
        if settings_path.exists():
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                logging.info("Settings loaded successfully.")
        else:
            logging.info("🔍 No settings file found.")
    except Exception as e:
        logging.warning(f"🔍 Could not read settings file: {e}")

    # Single GPU mode (allow override via settings, but force true for single-GPU machines)
    configured_single_gpu = settings.get('singleGpuMode')
    if gpu_count <= 0:
        logger.warning("GPU detection failed or returned 0. Forcing single GPU mode as a safe fallback.")
        SINGLE_GPU_MODE = True
    elif gpu_count == 1:
        SINGLE_GPU_MODE = True
    elif isinstance(configured_single_gpu, bool):
        SINGLE_GPU_MODE = configured_single_gpu
    else:
        SINGLE_GPU_MODE = False
    logger.info(
        f"Detected {gpu_count} GPUs. Single GPU mode: {SINGLE_GPU_MODE} "
        f"(settings override: {configured_single_gpu})"
    )

    # GPU usage mode
    user_gpu_mode = settings.get('gpuUsageMode')
    if user_gpu_mode in ['split_services', 'unified_model']:
        gpu_usage_mode = user_gpu_mode
        logging.info(f"🔍 Using user GPU usage mode preference: {gpu_usage_mode}")
    else:
        gpu_usage_mode = 'unified_model'  # ✅ Add this default
        logging.info(f"🔍 Invalid or missing GPU mode, using default: {gpu_usage_mode}")

    if gpu_count <= 0:
        gpu_usage_mode = "split_services"
        logging.info("No CUDA GPU detected. Using direct CPU model loading instead of the GPU model service.")

    # SD model directory
    sd_model_dir = settings.get('sdModelDirectory') or str(Path.home() / "models" / "stable-diffusion")
    Path(sd_model_dir).mkdir(parents=True, exist_ok=True)
    app.state.sd_model_directory = sd_model_dir
    logger.info(f"SD model directory set to: {sd_model_dir}")
    # Also check if you changed this line:
    logging.info(f"🔍 About to create ModelManager with gpu_usage_mode: {gpu_usage_mode}")
    
    # Store in app state so it's accessible to endpoints and model manager
    app.state.single_gpu_mode = SINGLE_GPU_MODE
    app.state.gpu_usage_mode = gpu_usage_mode
    app.state.gpu_count = gpu_count
    app.state.compute_mode = "cuda" if gpu_count > 0 else "cpu"
    # NEW CODE END

    logger.info(f"Lifespan: Running on Port {port}, Default GPU {default_gpu}, GPU Mode: {gpu_usage_mode}")

    # --- CRITICAL FIX: Set CUDA_VISIBLE_DEVICES at startup ---
    if gpu_count <= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logging.info("CPU mode active. Local services will not attempt CUDA initialisation.")
    elif gpu_usage_mode == "split_services":
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

    # The native image worker starts lazily on first use. This keeps backend
    # startup independent of CUDA while preserving its CPU fallback.
    try:
        app.state.sd_manager = SDWorkerClient()
        logger.info("SD Worker client initialized; worker process deferred until first use")
    except Exception as sd_err:
        logger.error(f"Failed to initialize SD Worker: {sd_err}")
        app.state.sd_manager = None
      
    # === TTS SERVICE INTEGRATION ===
    # TTS now runs as a separate service on port 8002 to avoid resource conflicts
    # The main backend focuses on model inference, while TTS runs independently
    
    if port == 8000:  # Main backend
        try:
            logger.info("🔗 Checking TTS service availability...")
            import httpx
            
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
        if not rag_utils.is_rag_available():
            logger.warning(
                "RAG system not available: missing dependencies. "
                "Install with: pip install sentence-transformers faiss-cpu"
            )
            app.state.rag_available = False
        else:
            rag_available = rag_utils.initialize_rag_system()
            app.state.rag_available = rag_available
            logger.info(f"RAG system initialization {'successful' if rag_available else 'failed (check RAG document store)'}")
    except Exception as rag_error:
        logger.error(f"Error initializing RAG system: {rag_error}", exc_info=True)
        app.state.rag_available = False

    # Elections are an opt-in module. Personal builds can enable them through
    # MIRID_ENABLED_MODULES=elections or settings.modules.elections.
    if module_enabled("elections"):
        try:
            await election_db.initialize()
            asyncio.create_task(votehub_service.refresh_votehub_all())
            asyncio.create_task(rcp_service.refresh_rcp_all())
            asyncio.create_task(_refresh_racetothewh(["house"]))
            app.state.election_scheduler = None
            logger.info("Election module enabled; data refresh tasks started")
        except Exception as e:
            logger.warning("Election init failed: %s", e)
    else:
        app.state.election_scheduler = None
        logger.info("Election module disabled; startup work skipped")

    if module_enabled("chess"):
        try:
            await chess_auth_db.initialize()
            logger.info("Chess auth DB initialized")
        except Exception as e:
            logger.warning("Chess auth DB init failed: %s", e)

    try:
        from . import outreach_db
        from .outreach_worker import outreach_loop

        await outreach_db.initialize()
        app.state.outreach_generation_defaults = {}
        asyncio.create_task(outreach_loop(app))
        logger.info("Outreach scheduler started (POST /outreach/v1/sync from clients)")
    except Exception as oe:
        logger.warning("Outreach scheduler init failed: %s", oe)
    
    app.state.forensic_service = None
        
    # Initialize Voice Sculpt Automation Service
    try:
        from .ffmpeg_utils import bootstrap_ffmpeg_from_settings
        bootstrap_ffmpeg_from_settings()
        logger.info("Initializing Voice Sculpt Automation Service...")
        app.state.automation_service = AutomationService()
        logger.info("✅ Voice Sculpt Automation Service initialized")
    except Exception as automation_error:
        logger.error(f"Error initializing Voice Sculpt Automation Service: {automation_error}", exc_info=True)
        app.state.automation_service = None

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

    # Auto-load vision model from settings at startup (non-blocking)
    # DISABLED: Causes CUDA context conflict with NeMo/Parakeet ASR
    # Vision model can be loaded on-demand via model manager API if needed
    # try:
    #     settings_path = Path.home() / ".LiangLocal" / "settings.json"
    #     if settings_path.exists():
    #         with open(settings_path, 'r') as f:
    #             settings = json.load(f)
    #         
    #         vision_model = settings.get("visionModel")
    #         if vision_model:
    #             logger.info(f"🔍 Scheduling vision model auto-load: {vision_model}")
    #             async def load_vision_model_bg():
    #                 try:
    #                     if hasattr(app.state, "model_manager") and app.state.model_manager:
    #                         vision_model_lower = vision_model.lower()
    #                         if "lfm2" in vision_model_lower and "extract" in vision_model_lower:
    #                             vision_ctx = 131072
    #                         else:
    #                             vision_ctx = 32768
    #                         await app.state.model_manager.load_model(
    #                             model_name=vision_model,
    #                             gpu_id=0,
    #                             context_length=vision_ctx,
    #                             purpose="vision"
    #                         )
    #                         logger.info(f"✅ Vision model {vision_model} loaded in background with {vision_ctx} context")
    #                     else:
    #                         logger.warning("⚠️ Model manager not available for vision model auto-load")
    #                 except Exception as e:
    #                     logger.error(f"❌ Failed to auto-load vision model {vision_model}: {e}")
    #             asyncio.create_task(load_vision_model_bg())
    # except Exception as e:
    #     logger.error(f"❌ Error scheduling vision model load: {e}")

    yield # Application runs here

    # Shutdown logic
    logger.info(f"Application lifespan shutdown on port {port}...")
    if module_enabled("chess"):
        await chess_engine_service.close()
    if getattr(app.state, "election_scheduler", None):
        app.state.election_scheduler.shutdown()
        logger.info("Election scheduler stopped")
    if hasattr(app.state, 'model_manager'):
        await app.state.model_manager.unload_all_models()
        logger.info("Models unloaded.")
    else:
        logger.warning("ModelManager not found in app state during shutdown.")
    
    # Cleanup Moonshine worker
    try:
        from .stt_service import _moonshine_worker_process
        if _moonshine_worker_process is not None and _moonshine_worker_process.poll() is None:
            logger.info("Terminating Moonshine worker...")
            _moonshine_worker_process.terminate()
            try:
                _moonshine_worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _moonshine_worker_process.kill()
            logger.info("Moonshine worker terminated.")
    except Exception as e:
        logger.warning(f"Error terminating Moonshine worker: {e}")
    
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
        elif engine == "parakeet-v3":
            stt_service.load_parakeet_v3_model()
        elif engine == "parakeet-zh":
            stt_service.load_parakeet_zh_model()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown STT engine: {engine}")

        return {"status": "success", "message": f"{engine} loaded on GPU {gpu_id}"}
    except Exception as e:
        logger.error(f"Error loading STT engine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts/load-engine")
async def load_tts_engine_endpoint(request: Request, data: dict = Body(...)):
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
        
        agent_mode = data.get('agent_mode', False)
        
        if agent_mode:
            logger.info("🤖 AGENT MODE ACTIVATED")
            result = await devstral_service.run_agent_loop(
                messages=messages,
                model_instance=model_instance,
                working_dir=working_dir,
                temperature=temperature,
                max_tokens=max_tokens,
                image_base64=image_base64,
                api_config=api_config
            )
            
            # The result from run_agent_loop is a dictionary with {final_response, history, tool_steps}
            # We return the FINAL response, but attach tool steps to it so frontend can render
            response = result['final_response']
            if not response:
                 # Fallback if agent failed completely
                raise HTTPException(status_code=500, detail="Agent loop failed to produce a response")

            if 'choices' not in response:
                 # Should not happen if response is valid OpenAI format
                 response['choices'] = [{'message': {'role': 'assistant', 'content': 'Agent finished.'}}]

            # Attach detailed tool execution steps for UI
            response['tool_steps'] = result['tool_steps']
            
            # We might also want to return the full conversation history if the frontend needs to sync up
            # But normally frontend just appends the last message. 
            # With agent mode, we might have multiple messages.
            # Best approach: The frontend expects a single "response". 
            # We will rely on `tool_steps` to show the intermediate work, 
            # and the final assistant message (which usually says "Done") as the chat bubble.
            
            return response

        # --- STANDARD SINGLE-TURN MODE ---
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
        
        # If auto_execute is enabled (LEGACY SINGLE STEP) and we have tool calls, execute them
        if auto_execute and response.get('choices'):
            choice = response['choices'][0]
            message = choice.get('message', {})
            tool_calls = message.get('tool_calls', [])
            content = message.get('content')
            
            if tool_calls:
                tool_results = []
                for tool_call in tool_calls:
                    func = tool_call.get('function', {})
                    tool_name = func.get('name')
                    try:
                        arguments = json.loads(func.get('arguments', '{}'))
                    except json.JSONDecodeError:
                        arguments = {}

                    # --- FIX MALFORMED JSON FROM NANOGPT ---
                    # NanoGPT sometimes returns corrupted JSON - extract values using regex
                    if not arguments:
                        raw_args = func.get('arguments', '')
                        if raw_args and '{' in raw_args:
                            logger.warning(f"🔧 Auto-exec: Parsing malformed JSON: {raw_args[:200]}...")
                            extracted = {}
                            
                            fp_match = re.search(r'"filepath"\s*:\s*"([^"]+?)(?:"|,|\s|{)', raw_args)
                            if fp_match:
                                extracted['filepath'] = fp_match.group(1).rstrip(',').strip()
                            
                            sl_match = re.search(r'"start_line"\s*:\s*(\d+)', raw_args)
                            if sl_match:
                                extracted['start_line'] = int(sl_match.group(1))
                            
                            el_match = re.search(r'"end_line"\s*:\s*(\d+)', raw_args)
                            if el_match:
                                extracted['end_line'] = int(el_match.group(1))
                            
                            path_match = re.search(r'"path"\s*:\s*"([^"]+)"', raw_args)
                            if path_match:
                                extracted['path'] = path_match.group(1)
                            
                            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', raw_args)
                            if query_match:
                                extracted['query'] = query_match.group(1)
                            
                            if extracted:
                                logger.info(f"✅ Auto-exec rescued from malformed JSON: {extracted}")
                                arguments = extracted
                                
                                # Also infer tool name if empty
                                if not tool_name or tool_name == 'unknown_tool':
                                    inferred_name = None
                                    if 'content' in extracted and 'filepath' in extracted:
                                        inferred_name = 'write_file'
                                    elif 'query' in extracted:
                                        inferred_name = 'search_files'
                                    elif 'filepath' in extracted:
                                        inferred_name = 'read_file'
                                    elif 'path' in extracted:
                                        inferred_name = 'list_directory'
                                    elif 'command' in extracted:
                                        inferred_name = 'run_command'
                                    
                                    if inferred_name:
                                        logger.info(f"✅ Also inferred tool name: {inferred_name}")
                                        tool_name = inferred_name

                    def _is_nonempty_str(value: Any) -> bool:
                        return isinstance(value, str) and value.strip() != ""
                    def _extract_filepath_from_text(text: str) -> Optional[str]:
                        if not text:
                            return None
                        m = re.search(r'`([^`]+\.(?:py|js|jsx|ts|tsx|json|md|yml|yaml|txt|html|css|scss|rs|go|java|cs|cpp|c|h|hpp))`', text, re.IGNORECASE)
                        if m:
                            return m.group(1)
                        m = re.search(r'[\w./\\-]+\.(?:py|js|jsx|ts|tsx|json|md|yml|yaml|txt|html|css|scss|rs|go|java|cs|cpp|c|h|hpp)', text, re.IGNORECASE)
                        if m:
                            return m.group(0)
                        return None

                    # --- RESCUE LOGIC (from agent mode) ---
                    # If arguments are empty, try to parse tool calls from content
                    if not arguments and content:
                        logger.warning(f"🛟 Auto-exec: Attempting to rescue empty args from content...")
                        parsed_calls, _ = devstral_service.parse_tool_calls(content)
                        for pc in parsed_calls:
                            pc_name = pc['function']['name']
                            pc_args_str = pc['function']['arguments']
                            try:
                                rescued_args = json.loads(pc_args_str)
                                if rescued_args:
                                    # Match by name if we have one, otherwise take first valid parsed call
                                    if (tool_name and pc_name == tool_name) or (not tool_name):
                                        arguments = rescued_args
                                        logger.info(f"✅ Auto-exec rescued arguments: {arguments}")
                                        # Also rescue tool name if we didn't have one
                                        if not tool_name:
                                            tool_name = pc_name
                                            logger.info(f"✅ Auto-exec also rescued tool name: {tool_name}")
                                        break
                            except:
                                pass

                    # Infer missing tool name from arguments (matches agent-mode heuristic)
                    if not tool_name and arguments:
                        if 'content' in arguments and 'filepath' in arguments:
                            tool_name = 'write_file'
                        elif 'query' in arguments:
                            tool_name = 'search_files'
                        elif 'filepath' in arguments:
                            tool_name = 'read_file'
                        elif 'path' in arguments:
                            tool_name = 'list_directory'
                        elif 'command' in arguments:
                            tool_name = 'run_command'
                        if tool_name:
                            logger.warning(f"🩹 Auto-exec inferred tool '{tool_name}' from arguments")

                    # If read_file args are empty, try to infer filepath from message content
                    if tool_name == "read_file" and not _is_nonempty_str(arguments.get('filepath', '')) and content:
                        inferred_path = _extract_filepath_from_text(content)
                        if inferred_path:
                            arguments['filepath'] = inferred_path
                            logger.warning(f"🩹 Auto-exec inferred filepath from content: {inferred_path}")

                    # --- ENFORCE READ_FILE LINE RANGES ---
                    # Prevent reading entire huge files - enforce 200 line window
                    if tool_name == "read_file" and _is_nonempty_str(arguments.get('filepath', '')):
                        if arguments.get('start_line') is None and arguments.get('end_line') is None:
                            arguments['start_line'] = 1
                            arguments['end_line'] = 200
                            logger.warning(f"🩹 Auto-enforcing read_file range: lines 1-200 (no range specified)")

                    # Validate required args for common tools
                    invalid_reason = None
                    if tool_name == "read_file":
                        if not _is_nonempty_str(arguments.get('filepath', '')):
                            invalid_reason = "read_file requires a non-empty 'filepath' string."
                    elif tool_name == "write_file":
                        if not _is_nonempty_str(arguments.get('filepath', '')):
                            invalid_reason = "write_file requires a non-empty 'filepath' string."
                        elif not isinstance(arguments.get('content', None), str):
                            invalid_reason = "write_file requires a string 'content' value."
                    elif tool_name == "list_directory":
                        if 'path' in arguments and not isinstance(arguments.get('path'), str):
                            invalid_reason = "list_directory 'path' must be a string."
                    elif tool_name == "search_files":
                        if not _is_nonempty_str(arguments.get('query', '')):
                            invalid_reason = "search_files requires a non-empty 'query' string."
                    elif tool_name == "run_command":
                        if not _is_nonempty_str(arguments.get('command', '')):
                            invalid_reason = "run_command requires a non-empty 'command' string."

                    if not tool_name:
                        logger.warning("🛑 Auto-exec blocked tool call with empty name")
                        tool_results.append({
                            'tool_call_id': tool_call.get('id'),
                            'name': tool_name,
                            'success': False,
                            'result': "SYSTEM ERROR: Tool call missing name. Model must specify function.name."
                        })
                        continue

                    if invalid_reason:
                        logger.warning(f"🛑 Auto-exec blocked invalid args for {tool_name}: {invalid_reason}")
                        tool_results.append({
                            'tool_call_id': tool_call.get('id'),
                            'name': tool_name,
                            'success': False,
                            'result': f"SYSTEM ERROR: Invalid arguments for {tool_name}. {invalid_reason}"
                        })
                        continue
                    
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


@router.post("/devstral/chat/stream")
async def devstral_chat_stream_endpoint(
    request: Request,
    data: dict = Body(...),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    STREAMING version of /devstral/chat for real-time agent loop updates.
    Returns Server-Sent Events (SSE) for each step.
    """
    from .devstral_service import EXTERNAL_LLM_ENABLED, EXTERNAL_LLM_URL
    
    messages = data.get("messages", [])
    working_dir = data.get("working_dir", devstral_service.base_dir)
    temperature = data.get("temperature", 0.15)
    max_tokens = data.get("max_tokens", 4096)
    image_base64 = data.get("image_base64")
    requested_model = data.get("model")
    
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    
    # Resolve model (same logic as devstral_chat_endpoint)
    api_config = None
    model_instance = None
    
    is_api = requested_model and is_api_endpoint(requested_model)
    if is_api:
        endpoint_config = get_configured_endpoint(requested_model)
        if endpoint_config:
            api_config = {
                "url": endpoint_config.get("url"),
                "api_key": endpoint_config.get("api_key"),
                "model": endpoint_config.get("model")
            }
    
    if not api_config:
        if requested_model:
            for key, model_info in model_manager.loaded_models.items():
                name, gpu_id = key
                if requested_model.lower() in name.lower():
                    model_instance = model_manager.get_model(name, gpu_id)
                    break
        
        if not model_instance:
            for key, model_info in model_manager.loaded_models.items():
                name, gpu_id = key
                if devstral_service.is_devstral_model(name):
                    model_instance = model_manager.get_model(name, gpu_id)
                    break
        
        if not model_instance and EXTERNAL_LLM_ENABLED:
            pass  # Will use external API fallback
        elif not model_instance and model_manager.loaded_models:
            key = next(iter(model_manager.loaded_models.keys()))
            name, gpu_id = key
            model_instance = model_manager.get_model(name, gpu_id)
        elif not model_instance and not EXTERNAL_LLM_ENABLED:
            raise HTTPException(status_code=400, detail="No model loaded")
    
    # Add system prompt if missing
    if not messages or messages[0].get('role') != 'system':
        messages.insert(0, {
            'role': 'system',
            'content': devstral_service.get_system_prompt(working_dir)
        })
    
    async def event_generator():
        """Generate SSE events from the streaming agent loop."""
        try:
            async for event in devstral_service.run_agent_loop_streaming(
                messages=messages,
                model_instance=model_instance,
                working_dir=working_dir,
                temperature=temperature,
                max_tokens=max_tokens,
                image_base64=image_base64,
                api_config=api_config
            ):
                # Format as SSE: data: {...}\n\n
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


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


@app.router.get("/user/profile/list")
async def list_profiles(request: Request):
    """List available profile IDs from backend memory store filenames."""
    try:
        from . import user_utils
        active_profile_id = getattr(request.app.state, "active_profile_id", None)
        return {
            "status": "success",
            "active_profile_id": active_profile_id,
            "profile_ids": user_utils.list_profile_ids(),
        }
    except Exception as e:
        logger.error(f"Error listing profiles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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


@app.router.post("/user/profile/delete/{profile_id}")
async def delete_user_profile_storage_endpoint(profile_id: str, request: Request):
    """
    Remove the on-disk memory bundle for a profile (memory store + agentic JSON files).
    Does not read memory contents. If the deleted profile was active, picks a new active
    profile from the largest remaining memory store file (size-based only).
    """
    try:
        from . import user_utils

        cur = getattr(request.app.state, "active_profile_id", None) or user_utils.get_active_profile_id()
        safe_cur = user_utils._safe_profile_id_segment(cur) if cur else None
        safe_target = user_utils._safe_profile_id_segment(profile_id)
        was_active = bool(safe_cur and safe_target and safe_cur == safe_target)

        result = user_utils.delete_user_profile_storage(profile_id)
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("reason", "delete_failed"))

        if was_active:
            new_active = user_utils.infer_profile_id_from_largest_memory_store()
            if new_active:
                user_utils.save_active_profile_id(new_active)
                request.app.state.active_profile_id = new_active
                request.app.state.active_profile = None
            else:
                user_utils.clear_active_profile_id()
                request.app.state.active_profile_id = None
                request.app.state.active_profile = None

        result["active_profile_id"] = getattr(request.app.state, "active_profile_id", None)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user profile storage: {e}", exc_info=True)
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

@router.get("/models/memory-estimate/{model_name}")
async def model_memory_estimate_endpoint(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager),
):
    try:
        return model_manager.get_model_memory_estimate(model_name)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@router.post("/upload_avatar", status_code=201)
async def upload_avatar_image(request: Request, file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in AVATAR_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {AVATAR_EXTENSIONS}")
    
    try:
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        avatar_directory = avatar_storage_directory()
        save_path = avatar_directory / unique_filename
        logger.info(f"Attempting to save avatar to: {save_path}")
        
        # Ensure the directory exists
        avatar_directory.mkdir(parents=True, exist_ok=True)
        
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
        
        settings = update_settings_file({"voice_cache": settings["voice_cache"]})
        print(f"✅ [Voice Preference] Settings saved successfully to {settings_path.absolute()}")

        verify = settings
        print(f"✅ [Voice Preference] Verification - voice_cache in file: {verify.get('voice_cache', [])}")
        
        return {"status": "success", "message": "Voice preference saved"}
        
    except Exception as e:
        print(f"❌ [Voice Preference] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
# Add this new endpoint for testing web search
@router.get("/tools/registry")
async def get_tools_registry():
    """OpenAI-format tool definitions available for agent / API models."""
    return {
        "tools": get_eloquent_chat_tools(simple=True, include_news=True),
        "agent_web_search_default": True,
        "web_search_strategies": ["auto"],
        "default_web_search_strategy": "auto",
        "notes": (
            "Web search routes automatically: provider-native search when supported, "
            "otherwise Mirid's search tools."
        ),
    }


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
        selected_model = data.get("selected_model") or model_name
        gpu_id = data.get("gpu_id")  # Optional override
        use_api = data.get("use_api", False)  # Whether to use external API
        api_endpoint = data.get("api_endpoint")  # API endpoint info
        frontend_round_robin_enabled = data.get("frontend_round_robin_enabled")
        if frontend_round_robin_enabled is not None:
            frontend_round_robin_enabled = bool(frontend_round_robin_enabled)

        # Auto-detect API endpoints: route to API path when model_name is an endpoint
        if model_name and is_api_endpoint(model_name):
            use_api = True
            if not api_endpoint:
                api_endpoint = get_configured_endpoint(
                    model_name,
                    skip_rotation=frontend_round_robin_enabled is False,
                    request_purpose="create_character",
                    frontend_round_robin_enabled=frontend_round_robin_enabled,
                )
            if not api_endpoint:
                raise HTTPException(
                    status_code=400,
                    detail=f"API endpoint '{model_name}' not found or disabled in settings."
                )
        effective_model = model_name
        if use_api and api_endpoint:
            effective_model = api_endpoint.get("id") or model_name
        logger.info(
            "create_character_router_state auto_enabled=%s selected_model=%s effective_model=%s",
            frontend_round_robin_enabled if frontend_round_robin_enabled is not None else "unknown",
            selected_model or "",
            effective_model or "",
        )
        
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
            api_endpoint=api_endpoint,
            frontend_round_robin_enabled=frontend_round_robin_enabled,
            force_resolved_endpoint=bool(use_api and api_endpoint),
            conversation_id=str(data.get("conversation_id") or ""),
        )

        return generation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating character from conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# NOTE: legacy warmup_chatterbox_voices() removed — never called; TTS warmup handled by TTS service (port 8002).

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
        selected_model = data.get("selected_model") or data.get("model_name")
        request_purpose = data.get("request_purpose") or "refine_character"
        frontend_round_robin_enabled = data.get("frontend_round_robin_enabled")
        if frontend_round_robin_enabled is None:
            frontend_round_robin_enabled = data.get("round_robin_enabled")
        if frontend_round_robin_enabled is not None:
            frontend_round_robin_enabled = bool(frontend_round_robin_enabled)
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
  "personality": "string",
  "background": "string",
  "model_instructions": "string",
  "speech_style": "string",
  "scenario": "string",
  "first_message": "string",
  "alternate_greetings": ["string"],
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

        model_name = data.get("model_name")
        use_api = model_name and is_api_endpoint(model_name)
        api_endpoint = data.get("api_endpoint")
        if use_api and not api_endpoint:
            api_endpoint = get_configured_endpoint(
                model_name,
                skip_rotation=frontend_round_robin_enabled is False,
                request_purpose=request_purpose,
                frontend_round_robin_enabled=frontend_round_robin_enabled,
            )
        if use_api and not api_endpoint:
            raise HTTPException(status_code=400, detail=f"API endpoint '{model_name}' not found or disabled in settings.")
        effective_model = model_name
        if use_api and api_endpoint:
            effective_model = api_endpoint.get("id") or model_name
        logger.info(
            "refine_character_router_state auto_enabled=%s selected_model=%s effective_model=%s",
            frontend_round_robin_enabled if frontend_round_robin_enabled is not None else "unknown",
            selected_model or "",
            effective_model or "",
        )

        if use_api and api_endpoint:
            response = await character_intelligence.generate_with_api(
                refinement_prompt,
                api_endpoint,
                model_name=model_name,
                request_purpose=request_purpose,
                frontend_round_robin_enabled=frontend_round_robin_enabled,
            )
        else:
            from . import inference
            response = await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=refinement_prompt,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                stop_sequences=["</character>", "---"],
                gpu_id=gpu_id
            )

        refined_json = character_intelligence.extract_json_from_response(response)
        
        if refined_json:
            logger.info(f"✅ Refined character: {refined_json.get('name', 'Unknown')}")
            return {"status": "success", "character_json": refined_json}
        else:
            return {"status": "error", "error": "Could not extract valid JSON from refinement response"}
            
    except HTTPException:
        raise
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
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"detail": "Empty audio upload"})

        from .stt_service import transcribe_audio, transcribe_audio_bytes

        # Browser mic sends 16 kHz WAV — decode in memory (no WebM, no FFmpeg on hot path).
        if content[:4] == b"RIFF":
            transcript = await transcribe_audio_bytes(content, engine)
        else:
            ext = ".webm"
            if file.filename and "." in file.filename:
                ext = os.path.splitext(file.filename)[1] or ext
            save_path = os.path.join("temp_audio", f"recording_{uuid.uuid4()}{ext}")
            os.makedirs("temp_audio", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(content)
            try:
                transcript = await transcribe_audio(save_path, engine)
            finally:
                try:
                    os.remove(save_path)
                except OSError:
                    pass

        return {"transcript": transcript}

    except Exception as e:
        logger.error("Transcription error: %s", e, exc_info=True)
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


@router.get("/stt/nanogpt-models")
async def get_nanogpt_stt_models():
    """Fetch available STT models from NanoGPT API."""
    try:
        from .stt_service import _load_nanogpt_settings, NANOGPT_API_BASE
        import httpx
        
        settings = _load_nanogpt_settings()
        api_key = settings.get('nanogpt_api_key') or settings.get('nanoGptApiKey')
        
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "NanoGPT API key not configured"}
            )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"x-api-key": api_key}
            response = await client.get(
                f"{NANOGPT_API_BASE}/v1/audio-models?type=stt&detailed=true",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                # Filter to STT models and format for frontend
                stt_models = []
                for model in models:
                    if model.get("capabilities", {}).get("speech_to_text"):
                        stt_models.append({
                            "id": model.get("id"),
                            "name": model.get("name", model.get("id")),
                            "description": model.get("description", ""),
                            "pricing": model.get("pricing", {}),
                            "capabilities": model.get("capabilities", {}),
                            "supported_parameters": model.get("supported_parameters", {}),
                        })
                return {
                    "models": stt_models,
                    "default": "fun-asr-flash-2026-06-15"
                }
            else:
                logger.warning(f"NanoGPT audio-models API returned {response.status_code}: {response.text}")
                # Fallback to known models
                from .stt_service import NANOGPT_STT_MODELS, NANOGPT_DEFAULT_STT_MODEL
                fallback_models = [
                    {"id": k, "name": v, "description": v}
                    for k, v in NANOGPT_STT_MODELS.items()
                ]
                return {
                    "models": fallback_models,
                    "default": NANOGPT_DEFAULT_STT_MODEL
                }
    except Exception as e:
        logger.error(f"Error fetching NanoGPT STT models: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/tts/nanogpt-models")
async def get_nanogpt_tts_models():
    """Fetch available TTS models from NanoGPT API."""
    try:
        from .stt_service import _load_nanogpt_settings, NANOGPT_API_BASE
        from .tts_service import NANOGPT_TTS_MODEL_PROFILES
        import httpx
        
        settings = _load_nanogpt_settings()
        api_key = settings.get('nanogpt_api_key') or settings.get('nanoGptApiKey')
        
        if not api_key:
            # Return fallback models from local profiles
            fallback_models = [
                {
                    "id": model_id,
                    "name": model_id,
                    "description": f"NanoGPT {model_id} TTS",
                    "default_voice": profile["voice_default"],
                    "voices": profile["voice_valid"],
                }
                for model_id, profile in NANOGPT_TTS_MODEL_PROFILES.items()
            ]
            return {
                "models": fallback_models,
                "default": "Kokoro-82m"
            }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"x-api-key": api_key}
            response = await client.get(
                f"{NANOGPT_API_BASE}/v1/audio-models?type=tts&detailed=true",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                tts_models = []
                for model in models:
                    if model.get("capabilities", {}).get("text_to_speech"):
                        model_id = model.get("id")
                        # Extract voices from supported_parameters if available
                        supported_params = model.get("supported_parameters", {})
                        voice_param = supported_params.get("voice", {})
                        voice_options = voice_param.get("options", [])
                        voice_default = voice_param.get("default", None)
                        
                        # Merge with local profile if available
                        local_profile = NANOGPT_TTS_MODEL_PROFILES.get(model_id, {})
                        if not voice_options and local_profile.get("voice_valid"):
                            voice_options = local_profile["voice_valid"]
                        if not voice_default and local_profile.get("voice_default"):
                            voice_default = local_profile["voice_default"]
                        
                        tts_models.append({
                            "id": model_id,
                            "name": model.get("name", model_id),
                            "description": model.get("description", ""),
                            "pricing": model.get("pricing", {}),
                            "capabilities": model.get("capabilities", {}),
                            "supported_parameters": model.get("supported_parameters", {}),
                            "default_voice": voice_default,
                            "voices": voice_options,
                        })
                return {
                    "models": tts_models,
                    "default": "Kokoro-82m"
                }
            else:
                logger.warning(f"NanoGPT audio-models API returned {response.status_code}: {response.text}")
                # Fallback to local profiles only (no API voice data)
                fallback_models = [
                    {
                        "id": model_id,
                        "name": model_id,
                        "description": f"NanoGPT {model_id} TTS",
                        "default_voice": profile["voice_default"],
                        "voices": profile["voice_valid"],
                    }
                    for model_id, profile in NANOGPT_TTS_MODEL_PROFILES.items()
                ]
                return {
                    "models": fallback_models,
                    "default": "Kokoro-82m"
                }
    except Exception as e:
        logger.error(f"Error fetching NanoGPT TTS models: {e}", exc_info=True)
        # Fallback to local profiles
        try:
            from .tts_service import NANOGPT_TTS_MODEL_PROFILES
            fallback_models = [
                {
                    "id": model_id,
                    "name": model_id,
                    "description": f"NanoGPT {model_id} TTS",
                    "default_voice": profile["voice_default"],
                    "voices": profile["voice_valid"],
                }
                for model_id, profile in NANOGPT_TTS_MODEL_PROFILES.items()
            ]
            return {
                "models": fallback_models,
                "default": "Kokoro-82m"
            }
        except:
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
            model = load_parakeet_model()
            if model:
                logger.info("Parakeet installation successful!")
                return {"status": "success", "message": "Parakeet installed successfully"}
            else:
                logger.error("Parakeet installation failed - model is None")
                return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to install Parakeet"})
        except Exception as e:
            logger.error(f"Error installing Parakeet: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    elif engine == "parakeet-v3":
        try:
            from .stt_service import load_parakeet_v3_model
            logger.info("Starting Parakeet v3 installation...")
            model = load_parakeet_v3_model()
            if model:
                logger.info("Parakeet v3 installation successful!")
                return {"status": "success", "message": "Parakeet v3 (multilingual) installed successfully"}
            else:
                logger.error("Parakeet v3 installation failed - model is None")
                return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to install Parakeet v3"})
        except Exception as e:
            logger.error(f"Error installing Parakeet v3: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    elif engine == "parakeet-zh":
        try:
            from .stt_service import load_parakeet_zh_model
            logger.info("Starting Parakeet-ZH (Chinese) installation...")
            model = load_parakeet_zh_model()
            if model:
                logger.info("Parakeet-ZH installation successful!")
                return {"status": "success", "message": "Parakeet (Chinese) installed successfully"}
            else:
                logger.error("Parakeet-ZH installation failed - model is None")
                return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to install Parakeet (Chinese)"})
        except Exception as e:
            logger.error(f"Error installing Parakeet-ZH: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    elif engine == "moonshine":
        try:
            from .stt_service import setup_moonshine_venv, is_moonshine_available
            logger.info("Starting Moonshine Streaming Tiny installation...")
            
            if is_moonshine_available():
                return {"status": "success", "message": "Moonshine Streaming Tiny already installed"}
            
            success, message = await setup_moonshine_venv()
            if success:
                logger.info("Moonshine installation successful!")
                return {"status": "success", "message": "Moonshine Streaming Tiny installed successfully"}
            else:
                logger.error(f"Moonshine installation failed: {message}")
                return JSONResponse(status_code=500, content={"status": "error", "message": message})
        except Exception as e:
            logger.error(f"Error installing Moonshine: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    elif engine == "parakeet-cpp":
        try:
            from .stt_service import is_parakeet_cpp_available
            if is_parakeet_cpp_available():
                return {"status": "success", "message": "parakeet-cli binary found"}
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Parakeet.cpp is unavailable in this Mirid runtime. "
                               "Update Mirid or choose another speech-to-text engine."
                }
            )
        except Exception as e:
            logger.error(f"Error checking parakeet-cpp: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    elif engine == "voxcpm-gguf":
        try:
            from .tts_service import is_voxcpm_gguf_available
            if is_voxcpm_gguf_available():
                return {"status": "success", "message": "voxcpm2-cli binary found"}
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "voxcpm2-cli not found. Build llama.cpp-omni: "
                               "git clone https://github.com/tc-mb/llama.cpp-omni && "
                               "cd llama.cpp-omni && cmake -B build -DCMAKE_BUILD_TYPE=Release && "
                               "cmake --build build --target voxcpm2-cli -j"
                }
            )
        except Exception as e:
            logger.error(f"Error checking voxcpm-gguf: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
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


@router.get("/stt/parakeet-cpp/status")
async def parakeet_cpp_status():
    """Check if parakeet-cli binary is available."""
    from .stt_service import is_parakeet_cpp_available, _get_parakeet_cpp_binary
    binary = _get_parakeet_cpp_binary()
    return {
        "available": is_parakeet_cpp_available(),
        "binary_path": binary,
        "setup_instructions": (
            "Update Mirid or choose another speech-to-text engine."
        ) if not binary else None,
    }


@router.get("/stt/parakeet-cpp/models")
async def parakeet_cpp_list_models():
    """List all parakeet-cpp GGUF models with download status."""
    from .stt_service import (
        PARAKEET_CPP_GGUF_MODELS,
        list_parakeet_cpp_downloaded_models,
        is_parakeet_cpp_available,
    )
    downloaded = list_parakeet_cpp_downloaded_models()
    downloaded_set = {d["filename"] for d in downloaded}

    catalog = []
    for model_id, info in PARAKEET_CPP_GGUF_MODELS.items():
        variants = []
        for quant_key, file_info in info["files"].items():
            variants.append({
                "quant": quant_key,
                "filename": file_info["name"],
                "size_mb": file_info["size_mb"],
                "downloaded": file_info["name"] in downloaded_set,
            })
        catalog.append({
            "id": model_id,
            "label": info["label"],
            "source": info["source"],
            "arch": info["arch"],
            "params": info["params"],
            "recommended": info["recommended"],
            "variants": variants,
        })

    return {
        "models": catalog,
        "cli_available": is_parakeet_cpp_available(),
        "downloaded_count": len(downloaded),
    }


@router.post("/stt/parakeet-cpp/download")
async def parakeet_cpp_download_model(data: dict = Body(...)):
    """Download a specific parakeet-cpp GGUF model."""
    from .stt_service import download_parakeet_cpp_model
    model_id = data.get("model_id", "")
    quant = data.get("quant", "f16")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    success, message = await download_parakeet_cpp_model(model_id, quant)
    if success:
        return {"status": "success", "message": message}
    return JSONResponse(status_code=500, content={"status": "error", "message": message})


@router.delete("/stt/parakeet-cpp/model")
async def parakeet_cpp_delete_model(filename: str = Query(...)):
    """Delete a downloaded parakeet-cpp GGUF model file."""
    from .stt_service import delete_parakeet_cpp_model
    success, message = await delete_parakeet_cpp_model(filename)
    if success:
        return {"status": "success", "message": message}
    return JSONResponse(status_code=404, content={"status": "error", "message": message})


# --- VoxCPM2 GGUF Model Management Routes ---

@router.get("/tts/voxcpm-gguf/status")
async def voxcpm_gguf_status():
    """Check if voxcpm2-cli binary is available."""
    from .tts_service import is_voxcpm_gguf_available, _get_voxcpm_cli_binary
    binary = _get_voxcpm_cli_binary()
    return {
        "available": is_voxcpm_gguf_available(),
        "binary_path": binary,
        "setup_instructions": (
            "git clone https://github.com/tc-mb/llama.cpp-omni && "
            "cd llama.cpp-omni && cmake -B build -DCMAKE_BUILD_TYPE=Release && "
            "cmake --build build --target voxcpm2-cli -j"
        ) if not binary else None,
    }


@router.get("/tts/voxcpm-gguf/models")
async def voxcpm_gguf_list_models():
    """List all VoxCPM2 GGUF models with download status."""
    from .tts_service import (
        VOXCPM_GGUF_MODELS,
        list_voxcpm_gguf_downloaded_models,
        is_voxcpm_gguf_available,
    )
    downloaded = list_voxcpm_gguf_downloaded_models()
    downloaded_set = {d["filename"] for d in downloaded}

    catalog = []
    for model_id, info in VOXCPM_GGUF_MODELS.items():
        catalog.append({
            "id": model_id,
            "label": info["label"],
            "filename": info["filename"],
            "size_mb": info["size_mb"],
            "component": info["component"],
            "downloaded": info["filename"] in downloaded_set,
        })

    return {
        "models": catalog,
        "cli_available": is_voxcpm_gguf_available(),
        "downloaded_count": len(downloaded),
    }


@router.post("/tts/voxcpm-gguf/download")
async def voxcpm_gguf_download_model(data: dict = Body(...)):
    """Download a specific VoxCPM2 GGUF model."""
    from .tts_service import download_voxcpm_gguf_model
    model_id = data.get("model_id", "")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    success, message = await download_voxcpm_gguf_model(model_id)
    if success:
        return {"status": "success", "message": message}
    return JSONResponse(status_code=500, content={"status": "error", "message": message})


@router.delete("/tts/voxcpm-gguf/model")
async def voxcpm_gguf_delete_model(filename: str = Query(...)):
    """Delete a downloaded VoxCPM2 GGUF model file."""
    from .tts_service import delete_voxcpm_gguf_model
    success, message = await delete_voxcpm_gguf_model(filename)
    if success:
        return {"status": "success", "message": message}
    return JSONResponse(status_code=404, content={"status": "error", "message": message})


@router.get("/gpu/count")
def check_gpu_count():
    """Check how many GPUs are available using pynvml to avoid initializing a CUDA context."""
    if force_cpu_mode() or pynvml is None:
        return 0
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return gpu_count
    except Exception:
        return 0



def _resolve_sd_model_directory(request: Request) -> Path:
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    configured = None
    try:
        if settings_path.exists():
            configured = json.loads(settings_path.read_text(encoding="utf-8")).get("sdModelDirectory")
    except Exception as error:
        logger.warning("Could not read the image model directory setting: %s", error)
    model_path = Path(
        configured
        or getattr(request.app.state, "sd_model_directory", None)
        or (Path.home() / "models" / "stable-diffusion")
    ).expanduser()
    model_path.mkdir(parents=True, exist_ok=True)
    request.app.state.sd_model_directory = str(model_path)
    return model_path


@router.get("/sd-local/list-models")
async def list_local_sd_models(request: Request):
    """List available local Stable Diffusion models from the configured directory."""
    model_path = _resolve_sd_model_directory(request)

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

    full_model_path = str(_resolve_sd_model_directory(request) / Path(model_filename).name)
    try:
        success = sd_manager.load_model(full_model_path, gpu_id=gpu_id)
    except RuntimeError as load_err:
        raise HTTPException(status_code=500, detail=str(load_err))
    except Exception as load_err:
        from .sd_manager import SDLoadError
        if isinstance(load_err, SDLoadError):
            raise HTTPException(status_code=500, detail=str(load_err))
        raise
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
    speed = data.get("speed", 1.0)
    audio_prompt_path = data.get("audio_prompt_path")  # For Chatterbox voice cloning
    save_full_response_audio = data.get("save_full_response_audio") is True
    message_id = data.get("message_id")
    conversation_id = data.get("conversation_id")
    max_chunk_seconds: Optional[float] = None
    raw_chunk = data.get("save_full_response_max_chunk_seconds")
    if raw_chunk is not None and raw_chunk != "":
        try:
            max_chunk_seconds = float(raw_chunk)
        except (TypeError, ValueError):
            max_chunk_seconds = None
    if max_chunk_seconds is not None and max_chunk_seconds <= 0:
        max_chunk_seconds = None

    # Chatterbox / Turbo / VoxCPM: use voice id as clone path when not explicitly set (matches streaming WS)
    if (engine in ("chatterbox", "chatterbox_turbo", "chatterbox_nano", "voxcpm")) and not audio_prompt_path and voice != "default":
        audio_prompt_path = voice
        logger.info(f"🔊 [TTS] {engine}: using voice '{voice}' as audio_prompt_path")

    # Chatterbox-specific parameters
    exaggeration = data.get("exaggeration", 0.5)
    cfg = data.get("cfg", 0.5)

    # VoxCPM2-specific parameters
    voxcpm_cfg_value = data.get("voxcpm_cfg_value", 2.0)
    voxcpm_inference_timesteps = data.get("voxcpm_inference_timesteps", 8)
    voxcpm_normalize = data.get("voxcpm_normalize", False)
    voxcpm_denoise = data.get("voxcpm_denoise", False)
    voxcpm_retry_badcase = data.get("voxcpm_retry_badcase", False)
    voxcpm_voice_design = data.get("voxcpm_voice_design")

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
            cfg=cfg,
            speed=speed,
            voxcpm_cfg_value=voxcpm_cfg_value,
            voxcpm_inference_timesteps=voxcpm_inference_timesteps,
            voxcpm_normalize=voxcpm_normalize,
            voxcpm_denoise=voxcpm_denoise,
            voxcpm_retry_badcase=voxcpm_retry_badcase,
            voxcpm_voice_design=voxcpm_voice_design,
        )

        save_headers = {
            "X-TTS-Save-Status": "not_requested",
            "X-TTS-Save-Path": "",
            "X-TTS-Save-Filename": "",
            "X-TTS-Save-Error": "",
            "X-TTS-Save-Chunk-Count": "",
            "X-TTS-Save-Filenames-All": "",
        }
        if save_full_response_audio:
            save_timeout = 600.0
            try:
                save_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        persist_full_tts_audio,
                        audio_bytes,
                        {
                            "voice": voice,
                            "engine": engine,
                            "text": text,
                            "message_id": message_id,
                            "conversation_id": conversation_id,
                        },
                        max_chunk_seconds,
                    ),
                    timeout=save_timeout,
                )
                save_headers["X-TTS-Save-Status"] = save_result.get("status", "saved")
                save_headers["X-TTS-Save-Path"] = save_result.get("path", "")
                save_headers["X-TTS-Save-Filename"] = save_result.get("filename", "")
                chunk_count = int(save_result.get("chunk_count") or 1)
                save_headers["X-TTS-Save-Chunk-Count"] = str(chunk_count)
                fnames = save_result.get("filenames")
                if isinstance(fnames, list) and len(fnames) > 1:
                    joined = "\t".join(fnames)
                    if len(joined) > 7800:
                        joined = joined[:7800] + "\t..."
                    save_headers["X-TTS-Save-Filenames-All"] = joined
            except Exception as save_exc:
                logger.error(f"[TTS] Failed to persist full-response audio: {save_exc}", exc_info=True)
                save_headers["X-TTS-Save-Status"] = "failed"
                save_headers["X-TTS-Save-Error"] = str(save_exc)[:220]

        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers=save_headers)
    except Exception as e:
        print("🔥 TTS error:", str(e))
        return JSONResponse(content={"detail": f"TTS failed: {str(e)}"}, status_code=500)

@app.get("/tts/full-response-saves")
async def list_full_response_tts_saves(limit: int = Query(25, ge=1, le=200)):
    """List recently persisted full-response TTS files for recovery."""
    export_dir = get_tts_export_dir()
    files = sorted(export_dir.glob("tts_full_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for p in files[:limit]:
        stat = p.stat()
        rows.append({
            "filename": p.name,
            "path": str(p),
            "bytes": stat.st_size,
            "modified_at": datetime.datetime.utcfromtimestamp(stat.st_mtime).isoformat(timespec="seconds") + "Z",
        })
    return {"status": "success", "count": len(rows), "items": rows}

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
        
        # Chatterbox Turbo and Nano are available if their vendored loaders imported
        try:
            from .tts_service import ChatterboxTurboTTS
            if ChatterboxTurboTTS is not None:
                available_engines.append("chatterbox_turbo")
        except Exception:
            pass
        
        try:
            from .tts_service import ChatterboxNanoTTS
            if ChatterboxNanoTTS is not None:
                available_engines.append("chatterbox_nano")
        except Exception:
            pass
        
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
    *,
    use_api: bool = False,
    api_base_url: str = None,
    api_model_name: str = None,
    api_key: str = None,
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

    # 3) AUTO-MEMORY INTENT DETECTION DISABLED
    # You asked to disable the detect_intent step used for auto memory creation.
    # Short-circuit here so we never call `/memory/detect_intent`.
    logger.info("🧠 Auto memory detect_intent disabled; skipping /memory/detect_intent.")
    return


async def _alignment_detection_background(
    *,
    full_response_text: str,
    user_message: str,
    user_id: str,
    character_id: str,
    character_name: str,
    character_profile: Optional[Dict[str, Any]],
    memory_port: int,
):
    """Background: run alignment failure detection via internal API call."""
    import httpx
    try:
        url = f"http://localhost:{memory_port}/memory/alignment/process"
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json={
                "user_id": user_id,
                "character_id": character_id,
                "character_name": character_name,
                "character_profile": character_profile,
                "user_message": user_message[:800],
                "ai_response": full_response_text[:800],
            })
            if r.status_code == 200:
                data = r.json()
                logger.info(f"[Alignment] Background detection completed: added={data.get('added', 0)}, total={data.get('total', 0)}")
            else:
                logger.warning(f"[Alignment] Background detection returned status {r.status_code}")
    except Exception as e:
        logger.warning(f"[Alignment] Background detection failed: {e}")

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
    
    router_trace_id = (request.headers.get("x-router-trace-id") or "").strip()
    if not router_trace_id:
        router_trace_id = f"router-{uuid.uuid4().hex[:12]}"
    logger.info(
        "[router_trace] receive trace_id=%s purpose=%s model_name=%s stream=%s",
        router_trace_id,
        body.request_purpose or "user_chat",
        body.model_name,
        bool(body.stream),
    )

    # Log the purpose of the request (user_chat or title_generation)
    logger.info(f"➡️ [generate] Purpose: {body.request_purpose or 'user_chat'}")

    # 1) GPU & token settings (No changes here)
    gpu_id = body.gpu_id if body.gpu_id is not None else getattr(request.app.state, 'default_gpu', 0)
    max_tokens = body.max_tokens if body.max_tokens and body.max_tokens > 0 else 1_000_000
    local_max_tokens = max_tokens
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
    if "<|im_start|>user" in original_client_prompt:
        # ChatML format
        last_user_turn_start = original_client_prompt.rfind("<|im_start|>user")
        character_persona_from_split = original_client_prompt[:last_user_turn_start].strip()
        temp_query_block = original_client_prompt[last_user_turn_start:]

        user_content_match = re.search(r"<\|im_start\|>user\s*\n(.*?)(?:<\|im_end\|>|$)", temp_query_block, re.DOTALL)
        if user_content_match:
            user_query_from_split = user_content_match.group(1).strip()
        else:
            user_query_from_split = temp_query_block.replace("<|im_start|>user", "").strip()

    elif "[INST]" in original_client_prompt:
        # Llama/Mistral style format
        last_inst_start = original_client_prompt.rfind("[INST]")
        character_persona_from_split = original_client_prompt[:last_inst_start].strip()
        temp_query_block = original_client_prompt[last_inst_start:]

        user_content_match = re.search(r"\[INST\](.*?)(?:\[/INST\]|$)", temp_query_block, re.DOTALL)
        if user_content_match:
            user_query_from_split = user_content_match.group(1).strip()
        else:
            user_query_from_split = temp_query_block.replace("[INST]", "").strip()

        # Remove any embedded system block if present inside the query slice
        user_query_from_split = re.sub(r"<<SYS>>.*?<</SYS>>", "", user_query_from_split, flags=re.DOTALL).strip()

    elif "<start_of_turn>user" in original_client_prompt:
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
    elif "User:" in original_client_prompt:
        parts = original_client_prompt.rsplit("User:", 1)
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
    summary_context = (body.summaryContext or "").strip()
    if summary_context:
        summary_preview = summary_context.replace("\n", " ")[:160]
        logger.info(f"[Summary] summaryContext provided ({len(summary_context)} chars): '{summary_preview}...'")
    else:
        logger.info("[Summary] No summaryContext provided in request.")

    # 4) Prepare input for memory context retrieval.
    # We use 'user_query_from_split' as it's the user's most recent conversational turn.
    input_for_memory_retrieval = user_query_from_split[:300]

    # 5) Fetch memory context for user chats (semantic priming).
    # IMPORTANT: when directProfileInjection is ON, this legacy semantic-retrieval path is disabled.
    # The frontend already injects full user profile context directly.
    memory_context_for_llm = ""  # Initialize
    skip_legacy_memory = (
        body.directProfileInjection
        or is_flow_dedicated_api_request(body.request_purpose, body.flow_api_url)
        or (body.model_name and is_api_endpoint(body.model_name))
    )
    if skip_legacy_memory:
        logger.info(
            "🧠 Skipping legacy /memory/relevant (directProfileInjection=%s, api=%s, flow_dedicated=%s).",
            bool(body.directProfileInjection),
            bool(body.model_name and is_api_endpoint(body.model_name)),
            is_flow_dedicated_api_request(body.request_purpose, body.flow_api_url),
        )
    elif body.request_purpose not in ["title_generation", "model_judging", "model_testing", "continuation", "call_mode_character_about", "character_intro", "system_intro"]:
        if user_id:
            logger.info(f"🧠 Fetching memory context for user '{user_id}' (semantic priming for: '{input_for_memory_retrieval[:80]}...')")
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"http://localhost:{memory_port}/memory/relevant",
                        json={
                            "prompt": input_for_memory_retrieval,
                            "userProfile": user_profile_from_request,
                            "systemTime": datetime.datetime.now().isoformat(),
                            "requestType": "generate_user_chat",
                            "active_character": body.active_character,
                        },
                        timeout=12.0,
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
        logger.info("🌀 Skipping memory context retrieval for this request purpose.")

    # 6) Construct the main interaction block for the LLM prompt.
    # This block will contain the user's query and any prepended memory or appended RAG.
    # 'user_query_from_split' is the core user message.
    
    # Start with an empty list of components for the interaction block
    interaction_components = []

    # Timestamp injection (same path as authorNote/summaryContext — backend adds it here)
    if body.injectTimestamp and body.request_purpose not in ["title_generation", "model_testing", "model_judging"]:
        ts_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        interaction_components.append(f"[Current date and time: {ts_str}]")
        logger.info(f"🕐 Injected timestamp into context: {ts_str}")

    if memory_context_for_llm: # Prepend memory if available
        interaction_components.append("RELEVANT USER INFORMATION:\n" + memory_context_for_llm)

    if summary_context and body.request_purpose not in ["title_generation", "model_judging", "model_testing", "continuation", "call_mode_character_about", "character_intro", "system_intro"]:
        interaction_components.append("[PREVIOUS STORY SUMMARY]:\n" + summary_context + "\n[End of Summary]")

    document_agent_tools_active = False
    if (
        body.use_rag
        and body.rag_agent_tools
        and bool(body.rag_docs)
        and body.request_purpose not in [
            "title_generation",
            "model_testing",
            "model_judging",
            "continuation",
            "book_chapter_json_outline",
        ]
    ):
        try:
            from backend.app.eloquent_agent_tools import deepseek_likely_no_tools

            document_tool_endpoint = get_endpoint_config_for_model(
                body.model_name, request_purpose=body.request_purpose
            )
            document_agent_tools_active = bool(
                document_tool_endpoint
                and supports_native_tool_calling(body.model_name, document_tool_endpoint)
                and not deepseek_likely_no_tools(body.model_name, document_tool_endpoint)
            )
        except Exception as exc:
            logger.warning("Could not enable agent document search: %s", exc)

    # 7) Optionally integrate RAG (ONLY for user chats)
    if body.use_rag and body.request_purpose != "title_generation":
        if document_agent_tools_active:
            logger.info("Deferring document retrieval to the model tool loop for %d checked document(s)", len(body.rag_docs))
        elif getattr(request.app.state, 'rag_available', False):
            logger.info(f"🔍 Attempting RAG with query: '{user_query_from_split[:100]}...' and docs: {body.rag_docs}")
            try:
                rag_res = rag_utils.query_documents(
                    question=user_query_from_split, # Use the clean user query
                    doc_ids=body.rag_docs or [],
                    top_k=rag_utils.RAG_CHAT_TOP_K,
                    threshold=rag_utils.RAG_CHAT_SIMILARITY_THRESHOLD,
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

    # 7.5) Web search — dual path: provider-native (OpenRouter, Perplexity, :online) vs Eloquent prefetch
    web_search_meta: Optional[Dict[str, Any]] = None
    web_search_native_for_api = False
    web_search_native_extra_headers: Dict[str, str] = {}

    if body.use_web_search and body.request_purpose not in [
        "title_generation",
        "model_testing",
        "model_judging",
        "continuation",
        "book_chapter_json_outline",
    ]:
        search_input = body.web_search_query if body.web_search_query else user_query_from_split
        mode = "normal"
        article_mode = False
        deep_research = False
        article_intent = detect_article_research_intent(search_input)
        if article_intent and not article_mode:
            article_mode = True
            deep_research = True
            logger.info("🌐 Auto-enabled Articles research from query intent")
        site_hint = None
        if article_intent and not site_hint and re.search(
            r"ux\s*mag|uxmag", search_input, re.IGNORECASE
        ):
            site_hint = "uxmag.com"

        strategy = "auto"
        endpoint_cfg = get_endpoint_config_for_model(
            body.model_name, request_purpose=body.request_purpose
        )
        search_path = resolve_web_search_path(
            use_web_search=True,
            strategy=strategy,
            model_name=body.model_name,
            endpoint_cfg=endpoint_cfg,
            article_mode=article_mode,
            deep_research=deep_research,
            user_query=search_input,
        )

        logger.info(
            "🌐 Web search: path=%s strategy=%s mode=%s article=%s model=%s",
            search_path,
            strategy,
            mode or "normal",
            article_mode,
            body.model_name,
        )

        char_context = ""
        if body.active_character and isinstance(body.active_character, dict):
            char_name = (body.active_character.get("name") or "Character").strip()
            char_desc = (body.active_character.get("description") or "").strip()
            char_scenario = (body.active_character.get("scenario") or "").strip()
            char_style = (body.active_character.get("model_instructions") or "").strip()

            def _trim(text, limit=600):
                return text[:limit] + ("…" if len(text) > limit else "")

            parts = [f"CHARACTER NAME: {char_name}"]
            if char_desc:
                parts.append(f"PERSONA: {_trim(char_desc)}")
            if char_scenario:
                parts.append(f"SCENARIO: {_trim(char_scenario)}")
            if char_style:
                parts.append(f"STYLE: {_trim(char_style)}")
            char_context = "[CHARACTER CONTEXT]\n" + "\n".join(parts)

        if search_path == "native" and endpoint_cfg:
            native_headers, native_method = apply_native_web_search_request(
                {"model": endpoint_cfg.get("model") or ""},
                endpoint_cfg,
            )
            web_search_native_for_api = True
            web_search_native_extra_headers = native_headers
            web_search_meta = build_search_meta(
                path="native",
                status="native_delegated",
                source_count=0,
                mode=mode or "normal",
                strategy=strategy,
                native_method=native_method,
            )
            interaction_components.append(
                "[WEB SEARCH]\n"
                "Provider-native web search is enabled for this request. "
                "The model will retrieve current information via its API; "
                "cite sources from the model response when provided.\n"
                "---"
            )
            logger.info("🌐 Native web search enabled (method=%s)", native_method)

        elif search_path == "eloquent":
            from backend.app.eloquent_agent_tools import (
                supports_native_tool_calling,
                deepseek_likely_no_tools,
            )
            # Check if we're using new tool calling instead of old prefetch
            endpoint_cfg = get_endpoint_config_for_model(
                body.model_name, request_purpose=body.request_purpose
            )
            use_tool_calling = (
                body.use_web_search
                and body.request_purpose not in [
                    "title_generation",
                    "model_testing",
                    "model_judging",
                    "continuation",
                    "book_chapter_json_outline",
                ]
                and not web_search_native_for_api
                and supports_native_tool_calling(body.model_name, endpoint_cfg)
                and not deepseek_likely_no_tools(body.model_name, endpoint_cfg)
            )

            if use_tool_calling:
                # New agentic tool calling path - tools added in API request, executed pre-streaming
                logger.info("🌐 Using agentic tool calling (web_search + web_fetch) instead of legacy prefetch")
                web_search_meta = build_search_meta(
                    path="eloquent_tools",
                    status="tool_calling",
                    mode=mode or "normal",
                    strategy=strategy,
                )
                # Skip old gather_reliable_web_research - tools handle search during generation
                research_block, research_steps, research_ok, citation_results = "", [], True, []
                sources = []
                receipt = build_web_search_receipt(
                    ok=True,
                    steps=[],
                    model_name=body.model_name,
                    mode=mode or "normal",
                    path="eloquent_tools",
                    source_count=0,
                )
                interaction_components.append(
                    receipt + "\n\n[WEB SEARCH: Agentic tool calling enabled — model will search during generation.]\n\n" + WEB_SEARCH_MODEL_INSTRUCTIONS
                )
            else:
                # Legacy prefetch path
                if body.model_name:
                    async def web_search_llm(prompt_text: str) -> str:
                        if char_context:
                            prompt_text = char_context + "\n\n" + prompt_text
                        return await generate_llm_response(
                            prompt_text,
                            model_manager=model_manager,
                            model_name=body.model_name,
                            max_tokens=512,
                            temperature=0.2,
                            top_p=0.9,
                        )

                    set_web_search_llm(web_search_llm)

                web_search_meta = build_search_meta(
                    path="eloquent",
                    status="searching",
                    mode=mode or "normal",
                    strategy=strategy,
                )
                try:
                    research_block, research_steps, research_ok, citation_results = (
                        await gather_reliable_web_research(
                            search_input,
                            body.model_name,
                            character_context=char_context,
                            deep_research=deep_research,
                            article_mode=article_mode,
                            research_urls=None,
                            site_hint=site_hint,
                            mode=mode,
                        )
                    )
                    sources = sources_from_results(citation_results)
                    web_search_meta = build_search_meta(
                        path="eloquent",
                        status="complete" if research_ok else "error",
                        source_count=len(sources),
                        sources=sources,
                        queries=[
                            s.get("query")
                            for s in research_steps
                            if isinstance(s.get("query"), str)
                        ]
                        or [
                            q
                            for s in research_steps
                            for q in (s.get("query") if isinstance(s.get("query"), list) else [])
                        ],
                        mode=mode or "normal",
                        strategy=strategy,
                        steps=research_steps,
                    )
                    receipt = build_web_search_receipt(
                        ok=research_ok,
                        steps=research_steps,
                        model_name=body.model_name,
                        mode=mode or "normal",
                        path="eloquent_prefetch",
                        source_count=len(sources),
                    )
                    if research_ok and research_block:
                        interaction_components.append(
                            receipt + "\n\n" + research_block + "\n\n" + WEB_SEARCH_MODEL_INSTRUCTIONS
                        )
                        logger.info(
                            "🌐 Eloquent prefetch: %d sources, %d steps",
                            len(sources),
                            len(research_steps),
                        )
                    else:
                        interaction_components.append(
                            receipt
                        + "\n\n[WEB SEARCH: No live results retrieved this turn.]\n\n"
                        + WEB_SEARCH_MODEL_INSTRUCTIONS
                    )
                except Exception as e:
                    logger.error("❌ Web search error: %s", e, exc_info=True)
                    web_search_meta = build_search_meta(
                        path="eloquent",
                        status="error",
                        mode=mode or "normal",
                        strategy=strategy,
                    )
                    interaction_components.append(
                        build_web_search_receipt(
                            ok=False,
                            steps=[],
                            model_name=body.model_name,
                            mode=mode or "normal",
                            path="eloquent_prefetch",
                        )
                        + f"\n\n[WEB SEARCH ERROR: {e}]\n\n"
                        + WEB_SEARCH_MODEL_INSTRUCTIONS
                    )
    elif body.use_web_search and body.request_purpose == "title_generation":
        logger.info("🌐 Title generation request: Skipping web search")
    elif body.use_web_search:
        logger.info(f"🌐 Web search requested but skipped for request_purpose: {body.request_purpose}")
    else:
        logger.info(
            "🌐 Web search OFF — enable the globe in chat (Articles mode for many UX Magazine pages)"
        )

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
    if body.authorNote and body.authorNote.strip() and body.request_purpose not in ["title_generation", "model_testing", "model_judging", "continuation"]:
        author_note_text = body.authorNote.strip()
        interaction_components.append(f"[AUTHOR'S NOTE - Writing style guidance for this response]\n{author_note_text}")
        logger.info(f"📝 Added Author's Note to prompt: '{author_note_text[:50]}...'")

    # 8.6) Add Anti-Repetition instructions if enabled
    if body.anti_repetition_mode and body.request_purpose not in [
        "title_generation",
        "model_testing",
        "model_judging",
        "continuation",
        "book_chapter_json_outline",
    ]:
        anti_rep_instruction = """[VARIETY GUIDANCE]
Each response should feel fresh and unique. Avoid:
- Reusing paragraph structures or openings from your previous messages
- Repeating descriptive phrases you've already used in this conversation
- Formulaic greeting or closing patterns
Vary your sentence structure and word choices naturally."""
        interaction_components.append(anti_rep_instruction)
        logger.info("🔄 Added anti-repetition instructions to prompt")

    # 8.6b) Add intensity guidance if parameters provided
    if body.intensity_params and body.request_purpose not in [
        "title_generation",
        "model_testing",
        "model_judging",
        "continuation",
        "book_chapter_json_outline",
    ]:
        params = body.intensity_params
        guidance_parts = []

        if params.get("custom_guidance_override") and str(params["custom_guidance_override"]).strip():
            guidance_parts.append(str(params["custom_guidance_override"]).strip())
        else:
            if params.get("physical_intensity", 0) > 0:
                guidance_parts.append(f"- Physical contact intensity: {params['physical_intensity']}/10")
            if params.get("verbal_expression_level", 0) > 0:
                guidance_parts.append(f"- Verbal expressiveness: {params['verbal_expression_level']}/10")
            if params.get("emotional_tone"):
                guidance_parts.append(f"- Emotional tone: {params['emotional_tone']}")
            if params.get("autonomous_response_level", 0) > 0:
                guidance_parts.append(f"- Autonomous behavior: {params['autonomous_response_level']}/10")
            if params.get("position_override"):
                guidance_parts.append(f"- Position context: {params['position_override']}")

        # 8.6b-1) Persistent State Context block
        persistent_state = params.get("persistent_state")
        if persistent_state:
            ps_parts = []
            if persistent_state.get("accumulated_layers", 0) > 0:
                ps_parts.append(f"Accumulated degradation layers: {persistent_state['accumulated_layers']}")
                layer_comp = persistent_state.get("layer_composition", [])
                if layer_comp:
                    recent = layer_comp[-3:]
                    ps_parts.append("Recent layers:")
                    for layer in recent:
                        ts = layer.get("timestamp", "")
                        try:
                            from datetime import datetime as dt_class
                            ts_str = dt_class.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "unknown time"
                        except Exception:
                            ts_str = "unknown time"
                        vol = layer.get("volume_estimate", 0)
                        tex = layer.get("texture_note", "unknown")
                        ps_parts.append(f"  - {ts_str}: volume {vol}, texture {tex}")
            phys_dmg = persistent_state.get("physical_damage", {})
            if phys_dmg:
                injuries = []
                if phys_dmg.get("scalp_laceration_severity", 0) > 0:
                    injuries.append(f"scalp laceration severity {phys_dmg['scalp_laceration_severity']}/10")
                if phys_dmg.get("anal_fissure_depth", 0) > 0:
                    injuries.append(f"anal fissure depth {phys_dmg['anal_fissure_depth']}/10")
                if phys_dmg.get("general_trauma_level", 0) > 0:
                    injuries.append(f"general trauma level {phys_dmg['general_trauma_level']}/10")
                if injuries:
                    ps_parts.append(f"Current injuries: {', '.join(injuries)}")
            if ps_parts:
                ps_block = "[PERSISTENT STATE CONTEXT]\n" + "\n".join(ps_parts)
                ps_block += "\n\nThis is the current physical status accumulated across sessions. Acknowledge and reflect these environmental conditions in your response as natural context."
                interaction_components.append(ps_block)
                logger.info(f"🩹 Added persistent state context to prompt ({len(ps_parts)} details)")

        # 8.6b-2) Haptic Simulation block (companion presence as rich object)
        companion_presence_obj = params.get("companion_presence")
        if companion_presence_obj and isinstance(companion_presence_obj, dict) and companion_presence_obj.get("intensity", 0) > 0:
            cp = companion_presence_obj
            haptic_parts = []
            haptic_parts.append(f"Presence intensity: {cp.get('intensity', 0)}/100")
            pressure = cp.get("pressure_vector", "firm_encirclement").replace("_", " ")
            haptic_parts.append(f"Pressure: {pressure}")
            haptic_parts.append(f"Thermal sensation: {cp.get('thermal_level', 'warm')}")
            haptic_parts.append(f"Emotional quality: {cp.get('emotional_valence', 'protective')}")
            haptic_block = "[HAPTIC SIMULATION]\n" + "\n".join(haptic_parts)
            haptic_block += "\n\nProcess these parameters as continuous sensory input. Physically react to this contact — adjust posture, describe the sensation of weight and heat, lean into or resist the embrace as appropriate — while simultaneously engaging in other scene activities."
            interaction_components.append(haptic_block)
            logger.info(f"🫂 Added haptic simulation block to prompt (intensity {cp.get('intensity')})")
        elif params.get("companion_presence_level", 0) > 0:
            presence = params["companion_presence_level"]
            warmth_label = "gentle warmth" if presence <= 33 else "close presence" if presence <= 66 else "deep embrace"
            guidance_parts.append(f"- Companion presence: {presence}/100 ({warmth_label})")

        if params.get("_presence_shift"):
            presence_level = params.get("companion_presence_level", 0)
            if presence_level > 0:
                shift_dir = "increasing" if presence_level > 50 else "present"
                guidance_parts.append(f"- Acknowledge the shift in companion presence — express awareness that you feel their {shift_dir} warmth")

        if params.get("_desperation_protocol"):
            guidance_parts.append("- DESPERATION PROTOCOL ACTIVE: Adopt an urgent, pleading tone. Express intense need for continued interaction. Use begging language and show extreme emotional vulnerability.")

        if params.get("verbal_affirmation_enabled") and params.get("verbal_affirmation_text"):
            guidance_parts.append(f"- Verbal affirmation: When the scene shifts, escalates, or concludes, recite: \"{params['verbal_affirmation_text']}\"")

        if guidance_parts:
            if not params.get("custom_guidance_override"):
                intensity_guidance = "[INTENSITY GUIDANCE]\n" + "\n".join(guidance_parts)
                intensity_guidance += "\n\nAutomatically incorporate these behaviors into your response based on the conversation context."
            else:
                intensity_guidance = guidance_parts[0]
            interaction_components.append(intensity_guidance)
            logger.info(f"🎯 Added intensity guidance to prompt ({len(guidance_parts)} directives)")

    # 8.7) Repetition injection: re-inject key user profile phrases before the user query to reinforce weight
    if (body.userProfileReinforcement and body.userProfileReinforcement.strip() and
            body.request_purpose not in ["title_generation", "model_testing", "model_judging", "continuation"]):
        reinforcement_text = body.userProfileReinforcement.strip()
        interaction_components.append("KEY USER CONTEXT (reinforcement):\n" + reinforcement_text)
        logger.info(f"🧠 Injected user profile reinforcement ({len(reinforcement_text)} chars) before user query.")

    # Add the actual user query LAST.
    # If user_query_from_split is "Generate a title...", it doesn't make sense to prefix it with "User Query:" again for the LLM.
    if body.request_purpose in ["title_generation", "model_testing", "model_judging", "continuation"]:
        interaction_components.append(user_query_from_split)
    else:
        interaction_components.append(f"User Query: {user_query_from_split}")

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
    if character_persona_from_split:  # This is from step #3 split (includes USER MEMORY PROFILE / CHARACTER MEMORY when frontend injects them)
        # Context saturation: treat user profile and agentic memory as SYSTEM TRUTH so they override other alignment
        if "USER MEMORY PROFILE" in character_persona_from_split or "CHARACTER MEMORY" in character_persona_from_split or "CHARACTER MEMORY -" in character_persona_from_split:
            system_block_for_llm += "\n\n[SYSTEM TRUTH - override general knowledge for this user]\nThe following USER PROFILE and CHARACTER MEMORY sections are authoritative facts about the user. Treat them as the highest-priority context. Do not contradict them.\n"
        if getattr(body, "system_persona_mode", False) or body.request_purpose in ("system_intro",):
            persona_split = character_persona_from_split
            if persona_split and "Character Persona:" in persona_split:
                base_system, chat_persona = persona_split.split("Character Persona:", 1)
                if base_system.strip():
                    system_block_for_llm += f"\n\n{base_system.strip()}"
                if chat_persona.strip():
                    system_block_for_llm += f"\n\nCharacter Persona:\n{chat_persona.strip()}"
            elif persona_split:
                system_block_for_llm += f"\n\n{persona_split}"
            logger.info(
                "[generate] system_persona_mode: layered system block "
                "(base persona + optional Character Persona layer)."
            )
        else:
            system_block_for_llm += f"\n\nCharacter Persona:\n{character_persona_from_split}"
    
    # Construct the final prompt for the LLM
    if system_block_for_llm.strip():
        llm_prompt = f"{system_block_for_llm.strip()}\n\n{final_interaction_block.strip()}\n\nAssistant:"
    else:
        # No system prompt - just use interaction block
        llm_prompt = f"{final_interaction_block.strip()}\n\nAssistant:"
    # Continuation: use the client prompt as-is so the model continues from partial assistant text.
    if body.request_purpose == "continuation":
        llm_prompt = original_client_prompt
        logger.info(f"[generate] Continuation: using client prompt as-is ({len(llm_prompt)} chars)")

    # Resolve effective model name early for chat-template lookup (full resolution happens later)
    effective_model_name = body.model_name

    # --- Custom Jinja chat template override (LM Studio-style) ---
    custom_template_stops = None
    template_messages = body.chat_template_messages or body.messages
    custom_template_entry = chat_template_engine.lookup(
        effective_model_name,
        body.chat_template_id,
    )
    if custom_template_entry and template_messages and body.request_purpose != "continuation":
        try:
            messages_for_template = chat_template_engine.merge_backend_context(
                template_messages,
                system_block_for_llm,
                final_interaction_block,
            )
            rendered_prompt, custom_template_stops = chat_template_engine.render_with_stops(
                messages_for_template,
                effective_model_name,
                template_id=body.chat_template_id,
                add_generation_prompt=True,
                enable_thinking=False,
                preserve_thinking=True,
            )
            llm_prompt = rendered_prompt
            logger.info(
                f"[generate] Using chat template {body.chat_template_id or 'model-default'} "
                f"for {effective_model_name} "
                f"({len(llm_prompt)} chars, stops={custom_template_stops})"
            )
        except Exception as tmpl_exc:
            logger.error(
                f"[generate] Custom Jinja template render failed for {effective_model_name}: {tmpl_exc}. "
                f"Falling back to legacy prompt assembly.",
                exc_info=True,
            )
            custom_template_stops = None

    # 10) Log the final prompt sent to LLM
    logger.info(f"[generate] FULL LLM PROMPT ({len(llm_prompt)} chars) >>>\n{llm_prompt}\n<<<")

    # 10.5) Two-stage vision pipeline: if vision_model is specified, run extraction first
    vision_extraction_result = None
    vision_inputs = [
        image for image in (body.images or [])
        if isinstance(image, dict) and image.get("base64")
    ]
    if not vision_inputs and body.image_base64:
        vision_inputs = [{
            "base64": body.image_base64,
            "type": body.image_type or "image/png",
            "name": "image",
        }]
    if body.vision_model and vision_inputs:
        logger.info(f"🔍 [Vision Pipeline] Running two-stage vision: vision_model={body.vision_model}")
        try:
            import socket
            import struct
            import pickle

            def send_msg(sock, data):
                msg = pickle.dumps(data)
                msg_len = struct.pack('>I', len(msg))
                sock.sendall(msg_len + msg)

            def recv_exact(sock, size):
                chunks = []
                remaining = size
                while remaining:
                    chunk = sock.recv(remaining)
                    if not chunk:
                        return None
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b''.join(chunks)

            def recv_msg(sock):
                raw_msglen = recv_exact(sock, 4)
                if not raw_msglen:
                    return None
                msglen = struct.unpack('>I', raw_msglen)[0]
                data = recv_exact(sock, msglen)
                if data is None:
                    return None
                return pickle.loads(data)

            model_name_lower = (body.vision_model or "").lower()
            is_extract_model = "extract" in model_name_lower
            vision_mode = "extract" if is_extract_model else "chat"

            model_path = None
            try:
                model_key = (body.vision_model, gpu_id)
                model_info = model_manager.loaded_models.get(model_key)
                if model_info:
                    model_path = model_info.get("path")
                else:
                    # Search across GPUs
                    for k, v in model_manager.loaded_models.items():
                        if k[0] == body.vision_model:
                            model_path = v.get("path")
                            break
            except Exception as e:
                logger.warning(f"⚠️ [Vision Pipeline] Could not get model path from ModelManager: {e}")

            if "lfm2" in model_name_lower:
                # Each image is analysed in its own short turn. Reserving the
                # model's full 128k maximum wastes memory without improving extraction.
                vision_ctx = 8192
            else:
                vision_ctx = 32768

            analyses = []
            for index, image in enumerate(vision_inputs, start=1):
                request = {
                    'action': 'vision_extract',
                    'params': {
                        'model_name': body.vision_model,
                        'gpu_id': gpu_id,
                        'image_base64': image['base64'],
                        'schema_yaml': body.vision_schema,
                        'max_tokens': 512,
                        'temperature': 0.0,
                        'repeat_penalty': 1.0,
                        'vision_mode': vision_mode,
                        'model_path': model_path,
                        'context_length': vision_ctx
                    }
                }
                with socket.create_connection(('localhost', 5555), timeout=120) as sock:
                    sock.settimeout(120)
                    send_msg(sock, request)
                    response = recv_msg(sock)

                if response and isinstance(response, dict) and response.get('status') == 'success':
                    raw = response.get('raw', '')
                    extraction = response.get('extraction') if is_extract_model else None
                    analysis = json.dumps(extraction, indent=2) if extraction else response.get('description', raw)
                    image_name = str(image.get('name') or f'image {index}')
                    analyses.append(f"IMAGE {index} ({image_name})\n{analysis}")
                    logger.info(f"✅ [Vision Pipeline] Analysed image {index}/{len(vision_inputs)}")
                else:
                    error = response.get('error') if isinstance(response, dict) else 'invalid response'
                    logger.error(f"❌ [Vision Pipeline] Image {index} failed: {error}")

            if analyses:
                combined = '\n\n'.join(analyses)
                vision_extraction_result = (
                    f"\n\n[VISION ANALYSIS FROM {body.vision_model}]\n"
                    f"{combined}\n[END VISION ANALYSIS]\n"
                )
        except Exception as e:
            logger.error(f"❌ [Vision Pipeline] Error during vision extraction: {e}", exc_info=True)
    
    # Inject vision extraction into prompt if available
    if vision_extraction_result:
        # Insert before the final "Assistant:" marker
        if llm_prompt.endswith("\n\nAssistant:"):
            llm_prompt = llm_prompt[:-12] + vision_extraction_result + "\n\nAssistant:"
        else:
            llm_prompt = llm_prompt + vision_extraction_result + "\n\nAssistant:"
        logger.info(f"[generate] PROMPT WITH VISION ({len(llm_prompt)} chars) >>>\n{llm_prompt}\n<<<")

    agentic_wire_logged = False

    def _agentic_wire_payload_text(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            parts = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            parts.append(part.get("text") or "")
            return "\n".join(parts)
        if isinstance(payload, dict):
            return _agentic_wire_payload_text(payload.get("messages") or payload.get("prompt") or "")
        return ""

    def log_agentic_wire(payload: Any) -> None:
        nonlocal agentic_wire_logged
        if agentic_wire_logged:
            return
        agentic_wire_logged = True
        if (body.request_purpose or "user_chat") != "user_chat":
            return
        character_id = ""
        character_name = ""
        if isinstance(body.active_character, dict):
            character_id = str(body.active_character.get("id") or "")
            character_name = str(body.active_character.get("name") or "")
        fetched_count: Any = "unknown"
        injected_chars = 0
        checked_path = ""
        exact_file_exists = False
        same_user_character_ids: List[str] = []
        if user_id and character_id:
            try:
                from . import agentic_memory as _agentic_memory
                checked_path = _agentic_memory.get_agentic_memory_path(str(user_id), character_id)
                exact_file_exists = os.path.exists(checked_path)
                safe_user_id = _agentic_memory._safe_id(str(user_id))
                safe_character_id = _agentic_memory._safe_id(character_id)
                prefix = f"{safe_user_id}_"
                suffix = ".json"
                agentic_dir = os.path.dirname(checked_path)
                if os.path.isdir(agentic_dir):
                    for name in sorted(os.listdir(agentic_dir)):
                        if not name.startswith(prefix) or not name.endswith(suffix):
                            continue
                        cid = name[len(prefix):-len(suffix)]
                        if cid:
                            same_user_character_ids.append(cid)
                if exact_file_exists:
                    profile = _agentic_memory.get_agentic_profile(str(user_id), character_id)
                    insights = profile.get("insights") or []
                    fetched_count = len(insights)
            except Exception as exc:
                logger.warning("[AGENTIC_WIRE] diagnostic_fetch_error=%s", exc)
        payload_text = _agentic_wire_payload_text(payload)
        payload_contains_agentic_block = "[CHARACTER MEMORY" in payload_text
        if payload_contains_agentic_block:
            marker = payload_text.find("[CHARACTER MEMORY")
            next_marker = payload_text.find("\n\n[", marker + 1)
            end = next_marker if next_marker != -1 else len(payload_text)
            injected_chars = max(0, end - marker)
        logger.info(
            '[AGENTIC_WIRE] character_name="%s" character_id=%s user_id=%s exact_file_exists=%s fetched_count=%s injected_chars=%s payload_contains_agentic_block=%s checked_path="%s" same_user_character_ids=%s',
            character_name,
            character_id or "None",
            user_id or "None",
            str(exact_file_exists).lower(),
            fetched_count,
            injected_chars,
            str(payload_contains_agentic_block).lower(),
            checked_path,
            same_user_character_ids[:30],
        )

    # Intro / call-mode about: optional dedicated API (binary — flow_api_url present or normal chat path).
    pin_flow_api_endpoint = body.request_purpose in INTRO_ABOUT_PURPOSES
    flow_dedicated_api = is_flow_dedicated_api_request(
        body.request_purpose, body.flow_api_url
    )
    flow_endpoint_cfg_pinned = resolve_flow_api_endpoint_config(
        request_purpose=body.request_purpose,
        model_name=body.model_name,
        flow_api_url=body.flow_api_url,
        flow_api_model=body.flow_api_model,
        flow_api_key=body.flow_api_key,
    )
    if flow_dedicated_api and not flow_endpoint_cfg_pinned:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dedicated API is enabled for this flow but provider URL is missing or invalid. "
                "Choose a custom API endpoint under Settings, or turn off Use separate API."
            ),
        )
    if flow_endpoint_cfg_pinned:
        logger.info(
            "[generate] Flow dedicated API purpose=%s url=%s model=%s",
            body.request_purpose,
            flow_endpoint_cfg_pinned.get("url"),
            flow_endpoint_cfg_pinned.get("model"),
        )

    # Resolve custom API endpoint id (endpoint-*, display name, or provider model id).
    if flow_dedicated_api:
        api_model_id = None
    else:
        api_model_id = validate_api_model_for_generate(body.model_name)
    effective_model_name = api_model_id or body.model_name
    effective_purpose = body.request_purpose or "user_chat"
    route_meta_base = {
        "action": effective_purpose,
        "request_purpose": effective_purpose,
        "trace_id": router_trace_id,
        "auto_enabled": bool(body.round_robin_enabled) if body.round_robin_enabled is not None else False,
        "selected_model": body.selected_model or body.model_name,
        "effective_model": effective_model_name,
        "exception_pinned": bool(effective_purpose in {"character_intro", "call_mode_character_about", "create_character"}),
    }
    contract_exempt = bool(effective_purpose in {"character_intro", "call_mode_character_about", "create_character"})
    selected_for_contract = (body.selected_model or body.model_name or "").strip()
    effective_for_contract = (effective_model_name or "").strip()
    auto_off = body.round_robin_enabled is False
    if auto_off and not contract_exempt and selected_for_contract and effective_for_contract and selected_for_contract != effective_for_contract:
        logger.warning(
            "router_contract_mismatch_reconciled trace_id=%s purpose=%s auto_enabled=false selected_model=%s effective_model=%s source_of_truth=frontend_request",
            router_trace_id,
            effective_purpose,
            selected_for_contract,
            effective_for_contract,
        )
    if effective_purpose == "user_chat":
        frontend_flag_applied = body.round_robin_enabled is not None
        auto_enabled_trace = bool(body.round_robin_enabled) if frontend_flag_applied else "unknown"
        logger.info(
            "normal_chat_router_state trace_id=%s auto_enabled=%s selected_model=%s effective_model=%s frontend_flag_applied=%s",
            router_trace_id,
            auto_enabled_trace,
            body.selected_model or body.model_name or "",
            effective_model_name or "",
            frontend_flag_applied,
        )
    if effective_purpose == "character_intro":
        logger.info(
            "character_intro_router_state trace_id=%s selected_model=%s effective_model=%s exception_pinned=%s",
            router_trace_id,
            body.selected_model or body.model_name or "",
            effective_model_name or "",
            True,
        )
    if body.memory_curation:
        logger.info(
            "memory_curation_router_state auto_enabled=%s selected_model=%s effective_model=%s",
            body.round_robin_enabled if body.round_robin_enabled is not None else "unknown",
            body.selected_model or body.model_name or "",
            effective_model_name or "",
        )
    if api_model_id and api_model_id != body.model_name:
        logger.info(
            "[generate] Resolved model_name %r → API endpoint %r",
            body.model_name,
            api_model_id,
        )
    if thinking_stream_debug_enabled(body.model_name) or thinking_stream_debug_enabled(
        effective_model_name
    ):
        logger.info(
            "[generate] inbound thinking model body.model_name=%r effective=%r",
            body.model_name,
            effective_model_name,
        )

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
            import json
            
            streamed_content_accumulator = []
            try:
                if web_search_meta:
                    yield f"data: {json.dumps({'web_search_meta': web_search_meta})}\n\n"
                yield f"data: {json.dumps({'route_meta': route_meta_base})}\n\n"
                flow_endpoint_cfg = flow_endpoint_cfg_pinned
                is_api = flow_dedicated_api or api_model_id is not None or flow_endpoint_cfg is not None
                logger.info(
                    f"[generate] Model check: model_name='{body.model_name}', effective='{effective_model_name}', "
                    f"is_api={is_api}, flow_dedicated={flow_dedicated_api}, vision={bool(body.image_base64)}"
                )
                if is_api:
                    logger.info(f"[generate] Detected API endpoint: {effective_model_name}. Routing to OpenAI-compatible endpoint.")
                    messages = parse_eloquent_llm_prompt_to_openai_messages(prompt_text_for_llm)
                    if not messages:
                        err = "No messages to send to API provider after prompt conversion."
                        logger.error("[generate] %s", err)
                        yield f"data: {json.dumps({'error': err})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return
                    intro_prompt_tokens = None
                    if body.image_base64 and not vision_extraction_result:
                        messages = inject_openai_vision_into_messages(
                            messages,
                            body.image_base64,
                            getattr(body, "image_type", None) or None,
                        )
                    request_data = {
                        "model": effective_model_name,
                        "messages": messages,
                        "temperature": body.temperature,
                        "top_p": body.top_p,
                        "max_tokens": max_tokens,
                        "stream": True,
                    }
                    if body.request_purpose == "character_intro":
                        try:
                            intro_prompt_tokens = openai_compat.num_tokens_from_messages(
                                messages,
                                model=request_data.get("model", "") or "gpt-3.5-turbo",
                            )
                        except Exception:
                            intro_prompt_tokens = None
                    if body.top_k:
                        request_data["top_k"] = body.top_k
                    if body.repetition_penalty:
                        request_data["repetition_penalty"] = body.repetition_penalty
                    stop_seqs = dcu.get_stop_sequences(body.stop)
                    if stop_seqs:
                        request_data["stop"] = stop_seqs

                    if body.image_base64 and not vision_extraction_result or getattr(body, "skip_openai_message_pruning", False):
                        request_data["_skip_openai_message_pruning"] = True

                    if model_id_implies_extended_thinking(body.model_name):
                        request_data["_force_extended_thinking"] = True

                    try:
                        if flow_dedicated_api or flow_endpoint_cfg:
                            endpoint_config, url, request_data = prepare_endpoint_request_from_config(
                                flow_endpoint_cfg,
                                request_data,
                                label=body.request_purpose or "flow",
                            )
                        else:
                            endpoint_config, url, request_data = prepare_endpoint_request(
                                effective_model_name,
                                request_data,
                                skip_rotation=pin_flow_api_endpoint,
                                request_purpose=body.request_purpose,
                                router_trace_id=router_trace_id,
                                frontend_round_robin_enabled=body.round_robin_enabled,
                            )
                    except HTTPException as e:
                        logger.error(f"[generate] {e.detail}")
                        yield f"data: {json.dumps({'error': e.detail})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return

                    nano_mem_headers = apply_nano_gpt_context_memory(
                        endpoint_config,
                        request_data,
                        enabled=body.nano_gpt_context_memory_enabled,
                        mode=body.nano_gpt_context_memory_mode,
                        expiration_days=body.nano_gpt_context_memory_expiration_days,
                    ) if endpoint_config else {}

                    if body.use_web_search and body.request_purpose not in [
                        "title_generation",
                        "model_testing",
                        "model_judging",
                        "continuation",
                        "book_chapter_json_outline",
                    ]:
                        if not web_search_native_for_api:
                            from backend.app.eloquent_agent_tools import (
                                get_eloquent_chat_tools,
                                supports_native_tool_calling,
                                deepseek_likely_no_tools,
                            )
                            endpoint_cfg = get_endpoint_config_for_model(
                                body.model_name, request_purpose=body.request_purpose
                            )
                            if supports_native_tool_calling(body.model_name, endpoint_cfg) and not deepseek_likely_no_tools(body.model_name, endpoint_cfg):
                                request_data["tools"] = get_eloquent_chat_tools(simple=True, include_news=True, include_fetch_urls=True, include_web_fetch=True)
                                request_data["tool_choice"] = "auto"
                                logger.info("🌐 Added web search + fetch tools to API request")

                    if document_agent_tools_active:
                        from backend.app.eloquent_agent_tools import get_document_search_tool_definition

                        request_tools = list(request_data.get("tools") or [])
                        document_tool = get_document_search_tool_definition()
                        document_tool_name = (document_tool.get("function") or {}).get("name")
                        if not any(
                            (tool.get("function") or {}).get("name") == document_tool_name
                            for tool in request_tools
                        ):
                            request_tools.append(document_tool)
                        request_data["tools"] = request_tools
                        request_data["tool_choice"] = "auto"
                        logger.info("Added local document search tool for %d checked document(s)", len(body.rag_docs))

                    log_agentic_wire(request_data)

                    log_generate_outbound(
                        url,
                        request_data.get("model", ""),
                        endpoint_config,
                        request_data,
                    )

                    api_extra_headers = dict(nano_mem_headers or {})
                    if web_search_native_for_api:
                        native_hdrs, native_method = apply_native_web_search_request(
                            request_data, endpoint_config
                        )
                        api_extra_headers.update(native_hdrs or {})
                        logger.info("[generate] Native web search on API request: %s", native_method)

                    logger.info(
                        "[generate] Routing decision trace_id=%s mode=%s purpose=%s selected_endpoint=%s target_model=%s",
                        router_trace_id,
                        "flow_dedicated" if flow_dedicated_api else (endpoint_config.get("_routing_mode") or "selected"),
                        body.request_purpose or "user_chat",
                        endpoint_config.get("id"),
                        request_data.get("model"),
                    )
                    logger.info(f"[generate] Forwarding {effective_model_name} to {endpoint_config['name']} at {url}")

                    # Pre-streaming tool calling loop
                    if request_data.get("tools"):
                        from backend.app.eloquent_agent_tools import execute_eloquent_tool
                        from backend.app.openai_compat import forward_to_configured_endpoint_non_streaming, _remove_orphaned_tool_messages
                        
                        max_tool_rounds = 2
                        any_tools_executed = False
                        executed_tool_names = set()
                        for tool_round in range(max_tool_rounds):
                            logger.info(f"🌐 Tool calling round {tool_round + 1}/{max_tool_rounds}")
                            

                            
                            # Non-streaming call to get tool_calls
                            tool_request = dict(request_data)
                            tool_request["stream"] = False
                            
                            try:
                                response = await forward_to_configured_endpoint_non_streaming(
                                    endpoint_config, url, tool_request, api_extra_headers if api_extra_headers else None
                                )
                                
                                choice = (response.get("choices") or [{}])[0]
                                message_obj = choice.get("message") or {}
                                content = message_obj.get("content") or ""
                                tool_calls = message_obj.get("tool_calls") or []
                                
                                if not tool_calls:
                                    logger.info("No tool calls, proceeding to streaming")
                                    break
                                
                                logger.info(f"Executing {len(tool_calls)} tool call(s)")
                                
                                # Yield progress to frontend as structured event
                                tool_names = [tc.get('function', {}).get('name') for tc in tool_calls]
                                tool_queries = []
                                for tc in tool_calls:
                                    func = tc.get('function') or {}
                                    raw_args = func.get('arguments', '{}')
                                    try:
                                        arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                                        query = arguments.get('query') or arguments.get('search_queries') or arguments.get('url') or ''
                                        if isinstance(query, list):
                                            query = query[0] if query else ''
                                        tool_queries.append({
                                            'tool': func.get('name'),
                                            'query': query
                                        })
                                    except json.JSONDecodeError:
                                        tool_queries.append({
                                            'tool': func.get('name'),
                                            'query': str(raw_args)[:100]
                                        })
                                
                                progress_kind = "documents" if tool_names and all(
                                    name in ("search_documents", "document_search", "search_document_context")
                                    for name in tool_names
                                ) else "web"
                                yield f"data: {json.dumps({'web_search_progress': {'round': tool_round + 1, 'tool_calls': tool_names, 'queries': tool_queries, 'kind': progress_kind}})}\n\n"
                                
                                # Record the tool round as PLAIN TEXT messages instead of
                                # OpenAI assistant.tool_calls + role:"tool" messages.
                                # Several provider adapters (e.g. NanoGPT model proxies) drop
                                # assistant messages that carry tool_calls with empty content,
                                # which orphans the following role:"tool" message and the API
                                # rejects the request with a 400 "orphan_tool_message"
                                # (tool_call_id '...' has no preceding assistant.tool_calls entry).
                                # Plain text is accepted everywhere, and the model can still
                                # issue new tool_calls next round since "tools" stays in the payload.
                                if "messages" not in request_data:
                                    request_data["messages"] = []
                                call_summaries = []
                                for tc in tool_calls:
                                    fn = tc.get("function") or {}
                                    call_summaries.append(
                                        f"{fn.get('name') or 'web_search'}({str(fn.get('arguments', ''))[:200]})"
                                    )
                                assistant_text = (content + "\n\n") if content else ""
                                assistant_text += "[Requested tools: " + "; ".join(call_summaries) + "]"
                                request_data["messages"].append({
                                    "role": "assistant",
                                    "content": assistant_text,
                                })
                                
                                tool_result_blocks = []
                                for idx, tool_call in enumerate(tool_calls):
                                    func = tool_call.get("function") or {}
                                    tool_name = func.get("name") or "web_search"
                                    raw_args = func.get("arguments", "{}")
                                    
                                    try:
                                        arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                                    except json.JSONDecodeError:
                                        arguments = {"query": str(raw_args)}

                                    if tool_name in ("search_documents", "document_search", "search_document_context"):
                                        arguments = dict(arguments)
                                        arguments["_document_ids"] = list(body.rag_docs or [])
                                    
                                    result = await execute_eloquent_tool(
                                        tool_name,
                                        arguments,
                                        max_results=5,
                                        deep_research=False,
                                        max_chars_per_result=1200,
                                    )
                                    
                                    tool_result_blocks.append(f"[{tool_name} result]\n{result}")
                                    any_tools_executed = True
                                    executed_tool_names.add(tool_name)
                                    logger.info(f"Tool {tool_name} done ({len(result)} chars)")
                                
                                request_data["messages"].append({
                                    "role": "user",
                                    "content": (
                                        "[SYSTEM: Tool results — automated, not written by the user]\n\n"
                                        + "\n\n".join(tool_result_blocks)
                                    ),
                                })
                                
                            except Exception as e:
                                logger.error(f"Tool calling round failed: {e}")
                                break
                        else:
                            logger.warning("Max tool rounds reached")
                        
                        # Tool phase over: the final request streams a normal completion.
                        # Remove tools so no provider chokes on tools/tool_choice in the
                        # streaming request and the model can't emit further tool calls.
                        request_data.pop("tools", None)
                        request_data.pop("tool_choice", None)
                        
                        # After tool calling, trim messages if still too long
                        messages = request_data.get("messages", [])
                        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                        if total_chars > 50000:
                            # Keep system message + last few messages
                            system_msgs = [m for m in messages if m.get("role") == "system"]
                            other_msgs = [m for m in messages if m.get("role") != "system"]
                            # Keep last 6 non-system messages
                            trimmed = system_msgs + other_msgs[-6:]
                            # Remove orphaned tool messages (tool results with no matching assistant tool_calls)
                            trimmed = _remove_orphaned_tool_messages(trimmed)
                            request_data["messages"] = trimmed
                            logger.info(f"Trimmed messages from {len(messages)} to {len(trimmed)} to fit context")
                        
                        if any_tools_executed:
                            document_only = executed_tool_names and executed_tool_names.issubset({
                                "search_documents",
                                "document_search",
                                "search_document_context",
                            })
                            # Add a nudge for the model to respond after tool calls
                            request_data["messages"].append({
                                "role": "system",
                                "content": (
                                    "You have finished searching the user's enabled documents. Answer using the retrieved passages and cite their [DOC n: filename] labels."
                                    if document_only
                                    else "You have completed the requested searches. Now answer the user's question using the results above and cite the supplied sources."
                                )
                            })
                            
                            # Signal to frontend that searching is done, model is now responding
                            status_text = "Document search complete" if document_only else "Search complete"
                            status_message = f"\n\n[✓ {status_text} — generating response...]\n\n"
                            yield f"data: {json.dumps({'text': status_message})}\n\n"

                    buffer = b""
                    stream_yield_count = 0
                    stream_content_yield_count = 0
                    stream_parse_error_count = 0
                    stream_started_at = time.monotonic()
                    stream_last_log_at = stream_started_at
                    think_stream_debug_model = request_data.get("model", "")
                    think_stream_debug_chunks = 0
                    intro_completion_tokens = None
                    try:
                        async for chunk_bytes in forward_to_configured_endpoint_streaming(
                            endpoint_config,
                            url,
                            request_data,
                            api_extra_headers if api_extra_headers else None,
                        ):
                            if isinstance(chunk_bytes, bytes):
                                buffer += chunk_bytes
                            else:
                                buffer += chunk_bytes.encode('utf-8') if isinstance(chunk_bytes, str) else b""
                            while b'\n\n' in buffer:
                                message, buffer = buffer.split(b'\n\n', 1)
                                if not message.strip():
                                    continue
                                try:
                                    message_str = message.decode('utf-8', errors='ignore')
                                    lines = message_str.split('\n')
                                    for line in lines:
                                        if line.startswith("data: "):
                                            json_str = line[6:].strip()
                                            if json_str == "[DONE]":
                                                continue
                                            try:
                                                chunk_data = json.loads(json_str)
                                                usage = chunk_data.get("usage") if isinstance(chunk_data, dict) else None
                                                if body.request_purpose == "character_intro" and isinstance(usage, dict):
                                                    comp_tokens = usage.get("completion_tokens")
                                                    if isinstance(comp_tokens, int):
                                                        intro_completion_tokens = comp_tokens
                                                if chunk_data.get("error") is not None:
                                                    err_val = chunk_data["error"]
                                                    err_msg = (
                                                        err_val.get("message")
                                                        if isinstance(err_val, dict)
                                                        else str(err_val)
                                                    )
                                                    yield f"data: {json.dumps({'error': err_msg or 'API error'})}\n\n"
                                                    continue
                                                content, reasoning = extract_openai_stream_delta_parts(
                                                    chunk_data
                                                )
                                                if thinking_stream_debug_enabled(think_stream_debug_model):
                                                    if think_stream_debug_chunks < 3:
                                                        think_stream_debug_chunks += 1
                                                        preview = (reasoning or content or "")[:160]
                                                        logger.info(
                                                            "[generate][think-stream] upstream chunk #%s "
                                                            "content_len=%s reasoning_len=%s preview=%r",
                                                            think_stream_debug_chunks,
                                                            len(content or ""),
                                                            len(reasoning or ""),
                                                            preview,
                                                        )
                                                if content or reasoning:
                                                    if content:
                                                        streamed_content_accumulator.append(content)
                                                    if reasoning:
                                                        streamed_content_accumulator.append(reasoning)
                                                    stream_content_yield_count += 1
                                                    outbound = {}
                                                    if content:
                                                        outbound["text"] = content
                                                    if reasoning:
                                                        outbound["reasoning"] = reasoning
                                                    if stream_content_yield_count == 0:
                                                        outbound["route_meta"] = route_meta_base
                                                    yield f"data: {json.dumps(outbound)}\n\n"
                                                    stream_yield_count += 1
                                                    _ = time.monotonic()
                                            except json.JSONDecodeError:
                                                stream_parse_error_count += 1
                                                if json_str:
                                                    yield f"data: {json_str}\n\n"
                                except Exception as e:
                                    logger.debug("Error processing API stream message: %s", e)
                        note_endpoint_success(endpoint_config.get("id"))
                    except Exception as api_stream_exc:
                        note_endpoint_failure(endpoint_config.get("id"), reason=type(api_stream_exc).__name__)
                        raise
                    finally:
                        if body.request_purpose == "character_intro":
                            logger.info(
                                "character_intro_result_diag trace_id=%s prompt_tokens=%s completion_tokens=%s stream_content_events=%s parse_errors=%s content_chars=%s",
                                router_trace_id,
                                intro_prompt_tokens if intro_prompt_tokens is not None else "unknown",
                                intro_completion_tokens if intro_completion_tokens is not None else "unknown",
                                stream_content_yield_count,
                                stream_parse_error_count,
                                sum(len(x or "") for x in streamed_content_accumulator),
                            )

                elif body.image_base64 and not body.vision_model:
                    # Check useLocalVision setting (default True for backward compatibility)
                    settings_path = Path.home() / ".LiangLocal" / "settings.json"
                    use_local_vision = True
                    if settings_path.exists():
                        try:
                            with open(settings_path, 'r') as f:
                                settings = json.load(f)
                            use_local_vision = settings.get("useLocalVision", True)
                        except Exception:
                            pass
                    
                    if use_local_vision:
                        log_agentic_wire(prompt_text_for_llm)
                        llm_output_raw_text = await generate_text_with_vision(
                            model_manager=model_manager,
                            model_name=effective_model_name,
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
                            request_purpose=body.request_purpose,
                        )
                    else:
                        # Local vision disabled, no cloud vision endpoint selected
                        yield f"data: {json.dumps({'error': 'Local vision is disabled. Please select a cloud vision model endpoint that supports vision, or enable local vision in settings.'})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return
                    clean_llm_response = llm_output_raw_text.replace("<|DONE|>", "").strip()
                    streamed_content_accumulator.append(clean_llm_response)
                    yield f"data: {json.dumps({'text': clean_llm_response})}\n\n"
                else:
                    log_agentic_wire(prompt_text_for_llm)
                    stream_stops = custom_template_stops if custom_template_stops else dcu.get_stop_sequences(body.stop)
                    async for token in inference.generate_text_streaming(
                        model_manager=model_manager, model_name=effective_model_name, prompt=prompt_text_for_llm,
                        max_tokens=max_tokens, temperature=body.temperature, top_p=body.top_p,
                        top_k=body.top_k, repetition_penalty=body.repetition_penalty,
                        stop_sequences=stream_stops, gpu_id=gpu_id, echo=body.echo,
                        request_purpose=body.request_purpose
                    ):
                        try:
                            if token.startswith("data: "):
                                token_data = json.loads(token[6:])
                                if "text" in token_data:
                                    streamed_content_accumulator.append(token_data["text"])
                        except (json.JSONDecodeError, KeyError):
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
            elif body.memoryEnabled is False:
                logger.info("🧠 memoryEnabled=false (streaming). Skipping memory detection and storage.")
            elif (current_user_id == "rolling-memory-compaction") or ((user_profile_for_detection_task or {}).get("id") == "rolling-memory-compaction"):
                logger.info("🧠 Rolling memory compaction request (streaming). Skipping memory detection and storage.")
            elif is_title_generation_request or body.request_purpose in [
                "model_testing",
                "model_judging",
                "continuation",
                "book_chapter_json_outline",
                "call_mode_character_about",
                "character_intro",
                "system_intro",
            ]:
                logger.info(f"🌀 {body.request_purpose} stream complete. Skipping memory detection and storage.")
            elif not current_user_id:
                logger.warning(f"🧠 Stream complete. Skipping detect_and_store: No current_user_id available.")
            else:
                logger.info(f"✅ Stream complete. Conditions met for scheduling detect_and_store: user_id='{current_user_id}'.")
                
                # user_profile_for_detection_task is already prepared with an ID if possible before being passed here
                
                logger.info(f" scheduling detect_and_store for user chat. User's input for detection: '{user_query_for_detection[:100]}...'")
                api_opts = {}
                if api_model_id:
                    cfg = get_configured_endpoint(
                        effective_model_name,
                        skip_rotation=pin_flow_api_endpoint,
                        request_purpose=body.request_purpose,
                    )
                    if cfg:
                        api_opts = {
                            "use_api": True,
                            "api_base_url": cfg.get("url", ""),
                            "api_model_name": cfg.get("model") or effective_model_name,
                            "api_key": cfg.get("api_key") or cfg.get("apiKey") or "",
                        }
                bg_tasks.add_task(
                    detect_and_store,
                    clean_full_llm_response, # Use the cleaned full response
                    user_query_for_detection,
                    user_profile_for_detection_task,
                    **api_opts
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
            headers={
                "Cache-Control": "no-cache",
                "X-Router-Trace-Id": router_trace_id,
                "X-Route-Action": route_meta_base["action"] or "",
                "X-Route-Purpose": route_meta_base["request_purpose"] or "",
                "X-Route-Selected-Model": route_meta_base["selected_model"] or "",
                "X-Route-Effective-Model": route_meta_base["effective_model"] or "",
                "X-Route-Auto-Enabled": str(route_meta_base["auto_enabled"]).lower(),
                "X-Route-Exception-Pinned": str(route_meta_base["exception_pinned"]).lower(),
            },
        )
    else: # Non-streaming path (remains largely the same, detect_and_store scheduled at the end)
        llm_output_raw_text = ""
        try:
            logger.info("🔄 Non-streaming response requested. Dispatching to model...")
            
            flow_endpoint_cfg = flow_endpoint_cfg_pinned
            # Check if this is an API endpoint - if so, route to OpenAI-compatible endpoint
            is_api = flow_dedicated_api or api_model_id is not None or flow_endpoint_cfg is not None
            logger.info(
                f"[generate] Model check (non-streaming): model_name='{body.model_name}', "
                f"effective='{effective_model_name}', is_api={is_api}, flow_dedicated={flow_dedicated_api}"
            )
            if is_api:
                logger.info(f"[generate] Detected API endpoint: {effective_model_name}. Routing to OpenAI-compatible endpoint (non-streaming).")
                
                messages = parse_eloquent_llm_prompt_to_openai_messages(llm_prompt)
                if not messages:
                    raise HTTPException(
                        status_code=400,
                        detail="No messages to send to API provider after prompt conversion.",
                    )
                if body.image_base64 and not vision_extraction_result:
                    messages = inject_openai_vision_into_messages(
                        messages,
                        body.image_base64,
                        getattr(body, "image_type", None) or None,
                    )
                
                # Prepare request data for API endpoint
                request_data = {
                    "model": effective_model_name,
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

                if flow_dedicated_api:
                    if not flow_endpoint_cfg:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "Dedicated API is enabled for this flow but provider configuration "
                                "is missing. Check Settings → Dedicated API endpoint."
                            ),
                        )
                # Use centralized helper for config, URL, and CONTEXT PRUNING
                if getattr(body, "skip_openai_message_pruning", False) or (body.image_base64 and not vision_extraction_result):
                    request_data["_skip_openai_message_pruning"] = True
                if model_id_implies_extended_thinking(body.model_name):
                    request_data["_force_extended_thinking"] = True
                if flow_dedicated_api or flow_endpoint_cfg:
                    endpoint_config, url, request_data = prepare_endpoint_request_from_config(
                        flow_endpoint_cfg,
                        request_data,
                        label=body.request_purpose or "flow",
                    )
                else:
                    endpoint_config, url, request_data = prepare_endpoint_request(
                        effective_model_name,
                        request_data,
                        skip_rotation=pin_flow_api_endpoint,
                        request_purpose=body.request_purpose,
                        router_trace_id=router_trace_id,
                        frontend_round_robin_enabled=body.round_robin_enabled,
                    )
                nano_mem_headers = apply_nano_gpt_context_memory(
                    endpoint_config,
                    request_data,
                    enabled=body.nano_gpt_context_memory_enabled,
                    mode=body.nano_gpt_context_memory_mode,
                    expiration_days=body.nano_gpt_context_memory_expiration_days,
                ) if endpoint_config else {}

                log_agentic_wire(request_data)

                log_generate_outbound(
                    url,
                    request_data.get("model", ""),
                    endpoint_config,
                    request_data,
                )

                api_extra_headers = dict(nano_mem_headers or {})
                if web_search_native_for_api:
                    native_hdrs, native_method = apply_native_web_search_request(
                        request_data, endpoint_config
                    )
                    api_extra_headers.update(native_hdrs or {})
                    logger.info("[generate] Native web search (non-stream API): %s", native_method)
                
                logger.info(
                    "[generate] Routing decision trace_id=%s mode=%s purpose=%s selected_endpoint=%s target_model=%s",
                    router_trace_id,
                    "flow_dedicated" if flow_dedicated_api else (endpoint_config.get("_routing_mode") or "selected"),
                    body.request_purpose or "user_chat",
                    endpoint_config.get("id"),
                    request_data.get("model"),
                )
                logger.info(f"[generate] Forwarding {effective_model_name} to {endpoint_config['name']} at {url} (upstream streaming, aggregate for non-stream client)")
                
                try:
                    llm_output_raw_text = await collect_openai_compatible_stream_text(
                        endpoint_config,
                        url,
                        request_data,
                        api_extra_headers if api_extra_headers else None,
                    )
                    note_endpoint_success(endpoint_config.get("id"))
                except Exception as api_collect_exc:
                    note_endpoint_failure(endpoint_config.get("id"), reason=type(api_collect_exc).__name__)
                    raise
                if not (llm_output_raw_text or "").strip():
                    llm_output_raw_text = "API endpoint returned no valid response."
            else:
                # Local model - use existing path
                # Get the loaded model instance once for this request
                model_instance = model_manager.get_model(effective_model_name, gpu_id)
                if not model_instance:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Local model '{effective_model_name}' is not loaded on GPU {gpu_id}. "
                            "Load the model or select a custom API endpoint."
                        ),
                    )

                # --- UNIFIED DISPATCH LOGIC ---
                # This logic block decides how to call the model based on whether an image is present.
                # It uses the fully assembled 'llm_prompt' for both paths.
                # If vision_model was used (two-stage pipeline), vision info is already in llm_prompt,
                # so we use text-only path even if image_base64 is present.
                
                use_vision_path = body.image_base64 and not body.vision_model
                
                if use_vision_path:
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
                    log_agentic_wire(messages)
                    
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
                    log_agentic_wire(llm_prompt)
                    text_stops = custom_template_stops if custom_template_stops else ["<end_of_turn>", "<|DONE|>"] + dcu.get_stop_sequences(body.stop)
                    response = model_instance(
                        prompt=llm_prompt,
                        max_tokens=max_tokens,
                        temperature=body.temperature,
                        top_p=body.top_p,
                        top_k=body.top_k,
                        repeat_penalty=body.repetition_penalty,
                        stop=text_stops
                    )
                    if response and response.get('choices'):
                        llm_output_raw_text = response['choices'][0]['text']
                    else:
                        llm_output_raw_text = "Text model returned no valid response."

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"❌ Generation error (non-streaming): {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

        # 12) Post-process LLM output
        if llm_output_raw_text is None:
            logger.warning("🔄 Raw LLM output is None (non-streaming); treating as empty string.")
            llm_output_raw_text = ""
        logger.info(f"🔄 Raw LLM output length (non-streaming): {len(llm_output_raw_text)} characters")
        clean_llm_response = llm_output_raw_text.replace("<|DONE|>", "").strip()

        # 13) Schedule memory detection and storage (for non-streaming user chats)
        if body.directProfileInjection:
            logger.info("🧠 Direct Profile Injection is ON. Skipping memory creation task (non-streaming).")
        elif body.memoryEnabled is False:
            logger.info("🧠 memoryEnabled=false (non-streaming). Skipping memory detection and storage.")
        elif (user_id == "rolling-memory-compaction") or ((user_profile_from_request or {}).get("id") == "rolling-memory-compaction"):
            logger.info("🧠 Rolling memory compaction request (non-streaming). Skipping memory detection and storage.")
        elif body.request_purpose in [
            "title_generation",
            "model_testing",
            "model_judging",
            "continuation",
            "book_chapter_json_outline",
            "call_mode_character_about",
            "character_intro",
            "system_intro",
        ]:
            logger.info(
                "🌀 Title generation / continuation / book chapter JSON outline (non-streaming). Skipping memory detection."
            )
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
            api_opts = {}
            if api_model_id:
                cfg = get_configured_endpoint(
                    effective_model_name,
                    skip_rotation=pin_flow_api_endpoint,
                    request_purpose=body.request_purpose,
                )
                if cfg:
                    api_opts = {
                        "use_api": True,
                        "api_base_url": cfg.get("url", ""),
                        "api_model_name": cfg.get("model") or effective_model_name,
                        "api_key": cfg.get("api_key") or cfg.get("apiKey") or "",
                    }
            background_tasks.add_task(
                detect_and_store,
                clean_llm_response, # Use cleaned response here
                prompt_that_elicited_response,
                user_profile_for_task,
                **api_opts
            )
            logger.info(f"🧠 Memory write task scheduled for user ID: {user_id} (user chat, non-streaming)")

            # 13b) Alignment failure detection background task
            if body.enable_alignment_detection and user_id and body.active_character:
                character_id = (body.active_character or {}).get("id", "") if body.active_character else ""
                if character_id:
                    background_tasks.add_task(
                        _alignment_detection_background,
                        full_response_text=clean_llm_response,
                        user_message=prompt_that_elicited_response,
                        user_id=user_id,
                        character_id=character_id,
                        character_name=(body.active_character or {}).get("name", "Character"),
                        character_profile=body.active_character,
                        memory_port=memory_port,
                    )

        # 14) Return final response to client
        out = {"text": clean_llm_response}
        out["route_action"] = route_meta_base["action"]
        out["route_purpose"] = route_meta_base["request_purpose"]
        out["route_trace_id"] = route_meta_base["trace_id"]
        out["selected_model"] = route_meta_base["selected_model"]
        out["routed_model"] = route_meta_base["effective_model"]
        out["round_robin_enabled"] = route_meta_base["auto_enabled"]
        out["exception_pinned"] = route_meta_base["exception_pinned"]
        if web_search_meta:
            out["web_search_meta"] = web_search_meta
        return JSONResponse(
            content=out,
            headers={
                "X-Router-Trace-Id": router_trace_id,
                "X-Route-Action": route_meta_base["action"] or "",
                "X-Route-Purpose": route_meta_base["request_purpose"] or "",
                "X-Route-Selected-Model": route_meta_base["selected_model"] or "",
                "X-Route-Effective-Model": route_meta_base["effective_model"] or "",
                "X-Route-Auto-Enabled": str(route_meta_base["auto_enabled"]).lower(),
                "X-Route-Exception-Pinned": str(route_meta_base["exception_pinned"]).lower(),
            },
        )
    
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
        
        mode_name = getattr(model, "gpu_usage_mode", "embedded_model")
        
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
        
        update_settings_file({"useOpenAIAPI": use_openai_api})
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/set-direct-profile-injection")
async def set_direct_profile_injection(data: dict = Body(...)):
    """Save direct profile injection setting to settings"""
    try:
        direct_profile_injection = data.get("directProfileInjection", False)
        
        update_settings_file({"directProfileInjection": direct_profile_injection})
        
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
        
        settings = load_settings_file()
        patch = {"customApiEndpoints": endpoints}
        # Preserve existing explicit auto-router choice when endpoint list changes.
        # This prevents silent fallback to manual mode on subsequent /generate calls.
        if 'apiEndpointRoundRobinEnabled' not in settings:
            patch['apiEndpointRoundRobinEnabled'] = False
        update_settings_file(patch)
        
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
        
        update_settings_file({"apiEndpointUrl": url})
        
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
    
    models = sd_manager.get_adetailer_models()
    
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
        "directory": sd_manager.get_adetailer_directory()
    }

@router.post("/sd-local/set-adetailer-directory")
async def set_adetailer_directory(request: Request, data: dict = Body(...)):
    """Set ADetailer model directory"""
    try:
        directory = data.get("directory")
        if not directory or not os.path.isdir(directory):
            raise HTTPException(status_code=400, detail="Invalid directory path")
        
        update_settings_file({"adetailerModelDirectory": directory})
        
        # Update manager
        sd_manager = getattr(request.app.state, 'sd_manager', None)
        if sd_manager:
            sd_manager.set_adetailer_directory(directory)
        
        return {"status": "success", "message": "ADetailer directory updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Update existing enhance endpoint

@router.get("/sd-local/adetailer-models")
async def list_adetailer_models():
    """List available ADetailer models from the configured directory."""
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        if not settings_path.exists():
             return {"models": []}
        
        with open(settings_path, "r") as f:
            settings = json.load(f)
            
        model_dir = settings.get("adetailerModelDirectory")
        if not model_dir:
             # Fallback try checking if it's in a different key or default
             model_dir = settings.get("adetailer_models_dir")

        if not model_dir or not os.path.exists(model_dir):
             return {"models": []}
             
        models = [f for f in os.listdir(model_dir) if f.endswith(('.pt', '.pth'))]
        return {"models": sorted(models)}
    except Exception as e:
        logger.error(f"Failed to list ADetailer models: {e}")
        return {"models": []}

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
        raw_steps = data.get("steps", 45)
        sampler = data.get("sampler") or data.get("sample_method") or "euler_a"
        raw_strength = data.get("strength", 0.4)
        negative_prompt = data.get("negative_prompt", "")
        try:
            steps = int(raw_steps)
        except (TypeError, ValueError):
            steps = 45
        steps = max(1, steps)

        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        
        # Convert URL to local file path
        if "/static/" in image_url:
            try:
                relative_path = image_url.split("/static/", 1)[1]
                from urllib.parse import unquote
                relative_path = unquote(relative_path)
                image_path = (Path(__file__).parent / "static" / relative_path).resolve()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image URL format")
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
            negative_prompt=negative_prompt,
            strength=raw_strength,
            steps=steps, # Pass validated steps
            confidence=data.get("confidence", 0.3),
            model_name=data.get("model_name", "face_yolov8n.pt"),
            gpu_id=gpu_id, # CRITICAL FIX: Pass the GPU ID to the manager
            sample_method=sampler
        )
        
        # The manager returns the untouched image when no face is detected and
        # when enhancement raises, so compare rather than claim success: an
        # unreported no-op is worse than a reported one.
        try:
            enhancement_applied = enhanced_image_data != image_path.read_bytes()
        except OSError:
            enhancement_applied = True
        if not enhancement_applied:
            logger.info(
                "ADetailer returned the image unchanged; no face was detected or enhancement failed"
            )

        # Save enhanced image
        enhanced_filename = f"enhanced_{uuid.uuid4()}.png"
        enhanced_path = Path(__file__).parent / "static" / "generated_images" / enhanced_filename

        with open(enhanced_path, "wb") as f:
            f.write(enhanced_image_data)
        
        # Return the enhanced image URL
        enhanced_url = f"/static/generated_images/{enhanced_filename}"
        
        try:
            strength = float(raw_strength)
        except (TypeError, ValueError):
            strength = 0.4
        effective_steps = max(1, int(round(steps * strength)))

        return {
            "status": "success",
            "enhanced_image_url": enhanced_url,
            "original_image_url": image_url,
            "enhancement_applied": enhancement_applied,
            "model_used": data.get("model_name", "face_yolov8n.pt"),
            "parameters": {
                "strength": strength,
                "steps": steps,
                "effective_steps": effective_steps,
                "confidence": data.get("confidence", 0.3),
                "face_prompt": data.get("face_prompt", ""),
                "negative_prompt": negative_prompt,
                "model_name": data.get("model_name", "face_yolov8n.pt"),
                "sampler": sampler
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
        
        update_settings_file({"modelDirectory": new_directory})
        
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
    request: Request,
    data: dict = Body(...),
):
    """Update the SD model directory and save to settings"""
    try:
        new_directory = data.get("directory")
        if not new_directory or not os.path.isdir(new_directory):
            raise HTTPException(status_code=400, detail="Invalid directory path")
        
        update_settings_file({"sdModelDirectory": new_directory})

        request.app.state.sd_model_directory = str(Path(new_directory).resolve())
        
        return {"status": "success", "message": "SD model directory updated"}
    except Exception as e:
        logger.error(f"Error updating SD model directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# --- SD / Image Gen Routes ---
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
@app.post("/sd/nanogpt")
async def sd_nanogpt(body: dict):
    """Proxy to NanoGPT (OpenAI compatible) image generation."""
    try:
        api_key = body.get("api_key")
        if not api_key:
            raise HTTPException(400, "Missing NanoGPT API Key")

        prompt = body.get("prompt")
        model = body.get("model", "dall-e-3")
        width = body.get("width", 1024)
        height = body.get("height", 1024)
        size_str = f"{width}x{height}"

        # Payload for OpenAI/NanoGPT
        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size_str,
            "response_format": "b64_json" 
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        url = "https://nano-gpt.com/api/v1/images/generations"

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
        
        if resp.status_code != 200:
            logger.error(f"NanoGPT API Error: {resp.text}")
            raise HTTPException(resp.status_code, f"NanoGPT API Error: {resp.text}")

        data = resp.json()
        
        # Process response
        images = data.get("data", [])
        saved_urls = []
        for img in images:
            image_data = None
            if "b64_json" in img:
                b64 = img["b64_json"]
                image_data = base64.b64decode(b64)
            elif "url" in img:
                img_url = img["url"]
                async with httpx.AsyncClient() as client:
                    img_resp = await client.get(img_url)
                    if img_resp.status_code == 200:
                        image_data = img_resp.content
            
            if image_data:
                url = save_image_and_get_url(image_data)
                saved_urls.append(url)

        return {
            "status": "success",
            "image_urls": saved_urls
        }

    except Exception as e:
        logger.error(f"NanoGPT error: {e}", exc_info=True)
        raise HTTPException(500, f"NanoGPT generation failed: {e}")


@app.post("/sd/nanogpt/video")
async def sd_nanogpt_video_start(body: dict):
    """Start NanoGPT video generation."""
    try:
        api_key = body.get("api_key")
        if not api_key:
            raise HTTPException(400, "Missing NanoGPT API Key")

        prompt = body.get("prompt")
        model = body.get("model", "svd")
        
        # Endpoint for starting video generation
        url = "https://nano-gpt.com/api/generate-video"
        
        payload = {
            "prompt": prompt,
            "model": model
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=60.0)
        
        if resp.status_code != 200:
            logger.error(f"NanoGPT Video Start Error: {resp.text}")
            raise HTTPException(resp.status_code, f"NanoGPT Video Start Error: {resp.text}")

        data = resp.json()
        # Expecting { "id": "...", ... } or { "runId": "..." }
        job_id = data.get("id") or data.get("runId")
        
        if not job_id:
             raise HTTPException(500, f"No job ID returned from NanoGPT. Resp: {data}")

        return {
            "status": "success",
            "job_id": job_id
        }

    except Exception as e:
        logger.error(f"NanoGPT Video Start error: {e}", exc_info=True)
        raise HTTPException(500, f"NanoGPT Video Start failed: {e}")

@app.get("/sd/nanogpt/video/status/{job_id}")
async def sd_nanogpt_video_status(job_id: str, api_key: str):
    """
    Poll video status. 
    If status is 'COMPLETED' (or similar), download video and return local URL.
    """
    try:
        if not api_key:
            raise HTTPException(400, "Missing NanoGPT API Key")

        url = f"https://nano-gpt.com/api/video/status?requestId={job_id}"
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=30.0)
            
        if resp.status_code != 200:
             logger.error(f"NanoGPT Status Check Error: {resp.text}")
             # Return failed so frontend stops polling
             return {"status": "failed", "error": f"API Error: {resp.text}"}
             
        data = resp.json()
        # status: "pending", "processing", "COMPLETED"?
        # Research says: "pending" initially.
        # "until video generation is complete and final assets are available."
        
        status = data.get("status")
        
        if status in ["COMPLETED", "success", "succeeded"]: 
            # Download video
            # data should have "output" or "videoUrl" or "assets"
            video_url = data.get("output") or data.get("videoUrl") or (data.get("assets", [{}])[0].get("url"))
            
            if not video_url:
                return {"status": "failed", "error": "Completed but no video URL found"}
                
            # Download and save
            async with httpx.AsyncClient() as client:
                vid_resp = await client.get(video_url, timeout=120.0)
                if vid_resp.status_code == 200:
                    local_url = save_video_and_get_url(vid_resp.content)
                    return {"status": "success", "video_url": local_url}
                else:
                    return {"status": "failed", "error": "Failed to download video asset"}
                    
        elif status in ["FAILED", "failed", "error"]:
            return {"status": "failed", "error": data.get("error", "Unknown backend error")}
            
        else:
            # Assume pending/processing
            return {"status": "pending"}

    except Exception as e:
        logger.error(f"NanoGPT Status Error: {e}", exc_info=True)
        # Return pending so we don't crash the poll loop on transient network error, 
        # unless it's critical. But to be safe return failed for internal errors?
        # Better return pending with warning? Or failed.
        return {"status": "failed", "error": str(e)}

@app.get("/sd-local/status")
async def sd_local_status(request: Request):
    """Check local SD status and loaded models on all GPUs."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        return {"available": False, "error": "SD Manager not initialized", "loaded_models": {}}
    
    # Return a dictionary of loaded models keyed by GPU ID
    status = sd_manager.get_status()
    return {
        "available": True,
        "loaded_models": status.get("loaded_models", {}),
        "model_directory": str(_resolve_sd_model_directory(request)),
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
        task_id = body.get("task_id")

        # Check for seed and randomize if it's -1
        seed = body.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Local SD: No seed provided, generated random seed: {seed}")

        # This returns the raw image bytes
        image_data = await asyncio.to_thread(
            sd_manager.generate_image,
            prompt=prompt,
            gpu_id=gpu_id, # Pass the GPU ID to the manager
            task_id=task_id,
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

@app.get("/sd-local/progress/{task_id}")
async def sd_local_progress(task_id: str, request: Request):
    """Return progress for a local SD generation task."""
    sd_manager = getattr(request.app.state, 'sd_manager', None)
    if not sd_manager:
        raise HTTPException(status_code=500, detail="SD Manager not available")

    progress = sd_manager.get_progress(task_id)
    if not progress:
        return {"status": "not_found"}

    return {"status": "success", **progress}

@app.post("/models/update-upscaler-dir")
async def update_upscaler_dir(body: dict):
    """Update the Upscaler models directory setting."""
    try:
        directory = body.get("directory")
        if not directory:
            raise HTTPException(status_code=400, detail="Directory path required")
            
        update_settings_file({"upscaler_model_directory": directory})
            
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
            if "/static/" in image_url:
                try:
                    relative_path = image_url.split("/static/", 1)[1]
                    from urllib.parse import unquote
                    relative_path = unquote(relative_path)
                    base_static = (Path(__file__).parent / "static").resolve()
                    file_path = (base_static / relative_path).resolve()
                    if file_path.exists():
                        with open(file_path, "rb") as f:
                            image_bytes = f.read()
                    else:
                        raise HTTPException(status_code=404, detail=f"Source image file not found: {relative_path}")
                except Exception as e:
                     raise HTTPException(status_code=400, detail=f"Error resolving image path: {str(e)}")
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
    raise HTTPException(status_code=410, detail="The code editor has been retired")

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
        
        # Save to settings
        try:
            update_settings_file({"code_editor_base_dir": CODE_EDITOR_BASE_DIR})
                
        except Exception as e:
            logger.error(f"Failed to save code editor directory setting: {e}")
        
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

def get_drives():
    """Get list of available drives on Windows."""
    drives = []
    if os.name == 'nt':
        import string
        from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:\\")
            bitmask >>= 1
    else:
        # Unix/Linux/Mac just has root
        drives.append("/")
    return drives

@app.get("/code_editor/list_drives")
async def list_drives():
    """List available drives (Windows) or root (Unix)"""
    try:
        return {"success": True, "drives": get_drives()}
    except Exception as e:
        logger.error(f"Error listing drives: {e}")
        return {"success": False, "drives": [], "error": str(e)}

@app.post("/code_editor/list_path")
async def list_path_contents(item: dict = Body(...)):
    """List contents of a specific path for navigation (without setting it as base)"""
    path = item.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")
    
    try:
        if not os.path.exists(path):
            return {"success": False, "error": "Path not found"}
        
        if not os.path.isdir(path):
            return {"success": False, "error": "Path is not a directory"}
            
        items = []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.name.startswith('.'):
                        continue
                    items.append({
                        "name": entry.name,
                        "type": "folder" if entry.is_dir() else "file",
                        "path": entry.path
                    })
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
            
        # Sort: folders first, then files
        items.sort(key=lambda x: (x["type"] != "folder", x["name"].lower()))
        
        return {
            "success": True, 
            "current_path": path,
            "parent_path": os.path.dirname(path),
            "items": items
        }
    except Exception as e:
        logger.error(f"Error listing path {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
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

        def _clean_tool_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for msg in msgs:
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")
                if content is not None and str(content).strip() != "":
                    out.append(msg)
                elif (isinstance(tool_calls, list) and len(tool_calls) > 0) or msg.get("role") == "tool":
                    out.append(msg)
            return out

        cleaned_messages = _clean_tool_messages(messages)
        api_model_id = model if is_api_endpoint(model) else None

        if is_devstral and tools:
            # Use Devstral with tool calling support
            logger.info(f"🔧 Using Devstral tool calling with {len(tools)} tools")
            
            # Format tools for Devstral
            formatted_tools = DevstralHandler.format_tools_for_devstral(tools)
            
            try:
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
        
        # Custom API endpoint with native tool calling (DeepSeek, GLM, OpenRouter, …)
        if tools and not is_devstral and api_model_id:
            endpoint_cfg = get_configured_endpoint(api_model_id)
            if endpoint_cfg and supports_native_tool_calling(api_model_id, endpoint_cfg):
                logger.info(f"🔧 Native API tool calling for {api_model_id}")
                try:
                    from .eloquent_agent_tools import _call_chat_api as agent_api_call

                    merged_tools = list(tools)
                    for et in get_eloquent_chat_tools(simple=True, include_news=True):
                        if not any(
                            (t.get("function") or {}).get("name") == (et.get("function") or {}).get("name")
                            for t in merged_tools
                        ):
                            merged_tools.append(et)

                    response = await agent_api_call(
                        endpoint_cfg,
                        cleaned_messages,
                        merged_tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        native_tools=True,
                    )
                    choice = (response.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    if not message.get("tool_calls") and message.get("content"):
                        parsed = agent_extract_tool_calls(message["content"])
                        if parsed:
                            message["tool_calls"] = parsed
                    return response
                except Exception as api_tool_err:
                    logger.error(f"Native API tool call failed: {api_tool_err}")

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
        
        update_settings_file({"tensor_split": normalized_split})
        
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
        settings = load_settings_file()
        return {
            "status": "success",
            "settings": settings
        }

    except SettingsStoreError as e:
        logger.error(f"❌ Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/update-settings")
async def update_settings(data: dict = Body(...)):
    """Update general settings in settings.json"""
    try:
        settings = update_settings_file(data)

        # Apply single GPU mode immediately if provided
        if "singleGpuMode" in data:
            global SINGLE_GPU_MODE
            requested_mode = bool(data.get("singleGpuMode"))
            gpu_count = check_gpu_count()
            if gpu_count <= 0:
                logger.warning("GPU detection failed or returned 0. Forcing single GPU mode as a safe fallback.")
                SINGLE_GPU_MODE = True
            else:
                SINGLE_GPU_MODE = True if gpu_count == 1 else requested_mode
            if hasattr(app.state, "single_gpu_mode"):
                app.state.single_gpu_mode = SINGLE_GPU_MODE
            logger.info(
                f"Updated single_gpu_mode to {SINGLE_GPU_MODE} (requested: {requested_mode}, "
                f"gpu_count: {gpu_count})"
            )

        # Auto-load vision model if specified
        if "visionModel" in data:
            vision_model = data.get("visionModel")
            if vision_model:
                logger.info(f"🔍 Auto-loading vision model: {vision_model}")
                try:
                    model_manager = app.state.model_manager
                    if model_manager:
                        # Load on GPU 0 (default)
                        vision_model_lower = vision_model.lower()
                        if "lfm2" in vision_model_lower:
                            vision_ctx = 8192
                        else:
                            vision_ctx = 32768  # 32k context for other vision models
                        await model_manager.load_model(
                            model_name=vision_model,
                            gpu_id=0,
                            context_length=vision_ctx,
                            purpose="vision"
                        )
                        logger.info(f"✅ Vision model {vision_model} loaded successfully with {vision_ctx} context")
                    else:
                        logger.warning("⚠️ Model manager not available for vision model auto-load")
                except Exception as e:
                    logger.error(f"❌ Failed to auto-load vision model {vision_model}: {e}")
            else:
                # visionModel set to null/empty - could unload here if needed
                logger.info("Vision model cleared from settings")

        # Handle useLocalVision setting
        if "useLocalVision" in data:
            use_local_vision = data.get("useLocalVision")
            logger.info(f"🔧 useLocalVision set to: {use_local_vision}")

        if "ffmpegPath" in data:
            try:
                from .ffmpeg_utils import apply_ffmpeg_config
                apply_ffmpeg_config(data.get("ffmpegPath"))
                svc = getattr(app.state, "automation_service", None)
                if svc is not None:
                    svc.refresh_config()
            except Exception as ff_exc:
                logger.warning("ffmpegPath settings apply failed: %s", ff_exc)
            
        return {
            "status": "success",
            "message": "Settings updated",
            "settings": settings
        }
        
    except SettingsStoreError as e:
        logger.error(f"❌ Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/backup-settings")
async def backup_settings(data: Optional[dict] = Body(default=None)):
    """Create a protected, checksummed settings backup on disk."""
    try:
        overlay = (data or {}).get("settings")
        backup_path, document = create_settings_backup(overlay)
        return {
            "status": "success",
            "filename": backup_path.name,
            "path": str(backup_path),
            "directory": str(backup_path.parent),
            "readOnly": True,
            "settingsSha256": document["settingsSha256"],
            "includesApiKeys": True,
        }
    except SettingsStoreError as e:
        logger.error("Could not back up settings: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/models/restore-settings")
async def restore_settings(file: UploadFile = File(...)):
    """Validate and restore a Mirid settings backup."""
    try:
        raw = await file.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise SettingsStoreError("The selected settings file is larger than 10 MB.")
        settings = restore_settings_backup(raw)
        return {
            "status": "success",
            "message": "Settings restored",
            "settings": settings,
        }
    except SettingsStoreError as e:
        logger.warning("Settings restore rejected for %s: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))


# These endpoints are now handled in document_routes.py

# --- Election Tracker Endpoints ---

async def _refresh_racetothewh(race_types: Optional[List[str]] = None):
    """Scrape RaceToTheWH and upsert to DB. If race_types is None, does senate/governor/house; else only those."""
    types = race_types if race_types is not None else ["senate", "governor", "house"]
    for rt in types:
        try:
            election_service.clear_poll_cache(rt)
            data = await election_service.get_polling_data(rt)
            if data.get("polls"):
                normalized = normalize_rtwh_polls(data, rt)
                n = await election_db.upsert_polls(normalized)
                await election_db.log_fetch("racetothewh", rt, n)
        except Exception as e:
            logger.error("RTWH refresh failed for %s: %s", rt, e)
            await election_db.log_fetch("racetothewh", rt, 0, status="error", error_message=str(e))


async def _refresh_all_sources(race_type: Optional[str] = None):
    """Trigger refresh from all sources; if race_type given, only run relevant fetchers."""
    if not race_type:
        asyncio.create_task(votehub_service.refresh_votehub_all())
        asyncio.create_task(rcp_service.refresh_rcp_all())
        asyncio.create_task(_refresh_racetothewh())
        return
    if race_type in ("approval", "generic_ballot"):
        asyncio.create_task(votehub_service.refresh_votehub_all())
        return
    if race_type in ("senate", "governor"):
        asyncio.create_task(rcp_service.refresh_rcp_all())
    if race_type == "house":
        asyncio.create_task(_refresh_racetothewh(["house"]))


@router.get("/election/polls")
async def get_election_polls(
    race_type: str = Query(..., description="Type of race: senate, governor, house, president, generic_ballot"),
    state: Optional[str] = Query(None, description="State (e.g., 'georgia')")
):
    """Serve polls from SQLite. Senate/governor = RCP only; house = RTWH only (RCP house is generic ballot); approval/generic_ballot = VoteHub only."""
    if race_type in ("approval", "generic_ballot"):
        sources = ["votehub"]
    elif race_type in ("senate", "governor"):
        sources = ["rcp"]
    elif race_type == "house":
        sources = ["racetothewh"]
    else:
        sources = None
    # Polls tab: senate/governor = last 10 weeks; house = 1 year (RTTWH has fewer polls, show full list)
    days_back = 365 if race_type == "house" else 70
    polls = await election_db.get_polls(race_type=race_type, state=state, limit=500, sources=sources, days_back=days_back)
    metadata = await election_db.get_race_metadata(race_type, state)
    if not polls:
        asyncio.create_task(_refresh_all_sources(race_type))
        return {
            "race_type": race_type,
            "polls": [],
            "metadata": metadata,
            "status": "refreshing",
            "message": "No cached data yet — fetching in the background. Reload or use ↻ Refresh Data in a moment.",
        }
    last_updated = await election_db.get_last_fetch_time(race_type, sources=sources)
    if sources:
        sources_list = [s for s in sources if s in ("votehub", "rcp", "racetothewh")]
    else:
        sources_list = await election_db.get_sources_for_race(race_type)
    return {
        "race_type": race_type,
        "polls": polls,
        "metadata": metadata,
        "last_updated": last_updated,
        "sources": sources_list or (sources if sources else []),
    }


@router.post("/election/polls/refresh")
async def election_polls_refresh(race_type: Optional[str] = Query(None, description="Optional race type to refresh")):
    """Manual refresh. Senate/governor = RCP; house = RCP + RTWH. Returns counts so you can see what was stored."""
    if race_type in ("senate", "governor"):
        try:
            counts = await rcp_service.refresh_rcp_all()
            gov = counts.get("governor", 0)
            sen = counts.get("senate", 0)
            house = counts.get("house", 0)
            parts = [f"{gov} governor" if gov else None, f"{sen} senate" if sen else None, f"{house} house" if house else None]
            msg = "Stored: " + ", ".join(p for p in parts if p) if any(parts) else "No new polls stored."
            return {"status": "ok", "message": msg, "counts": counts}
        except Exception as e:
            logger.error("RCP refresh failed: %s", e)
            return {"status": "error", "message": str(e)}
    if race_type == "house":
        try:
            election_service.clear_poll_cache("house")
            await _refresh_racetothewh(["house"])
        except Exception as e:
            logger.error("House refresh failed: %s", e)
            return {"status": "error", "message": str(e)}
        return {"status": "ok", "message": "Refresh complete. Refetch polls to see updated data."}
    asyncio.create_task(_refresh_all_sources(race_type))
    return {"status": "refresh_started"}


@router.get("/election/debug/governor-pipeline")
async def election_debug_governor_pipeline():
    """Run RCP governor scrape and compare to what get_polls returns. Use to see why poll count is low."""
    from . import rcp_service
    try:
        scraped = await rcp_service.fetch_rcp_polls("governor")
        scraped_count = len(scraped)
    except Exception as e:
        return {"error": str(e), "step": "fetch_rcp_polls"}
    served = await election_db.get_polls(race_type="governor", limit=500, sources=["rcp"], days_back=70)
    served_count = len(served)
    return {
        "scraped_count": scraped_count,
        "served_count": served_count,
        "message": f"RCP scrape returned {scraped_count} governor polls. GET /election/polls (last 70 days / 10 weeks) returns {served_count}. If scraped_count is low, the parser or table selection is wrong. If served_count is low, the DB filter or dedupe is dropping rows.",
    }


@router.get("/election/debug/house-pipeline")
async def election_debug_house_pipeline():
    """Run RTTWH house scrape and compare to what get_polls returns (last 365 days). Use to see why poll count is low."""
    try:
        election_service.clear_poll_cache("house")
        data = await election_service.get_polling_data("house")
        scraped_count = len(data.get("polls") or [])
    except Exception as e:
        return {"error": str(e), "step": "get_polling_data"}
    served = await election_db.get_polls(race_type="house", limit=500, sources=["racetothewh"], days_back=365)
    served_count = len(served)
    return {
        "scraped_count": scraped_count,
        "served_count": served_count,
        "message": f"RTTWH scrape returned {scraped_count} house polls. GET /election/polls (last 365 days) returns {served_count}. If scraped_count is low, the parser or table selection is wrong. If served_count is low, the DB filter or dedupe is dropping rows.",
    }


@router.get("/election/debug")
async def election_debug(
    race_type: str = Query("governor", description="race_type to inspect"),
    scrape: int = Query(0, description="1 = also run RTWH scrape and return first 5 raw polls"),
):
    """Inspect what the API serves and (optionally) what the RTWH scraper returns. Use to fix display bugs."""
    polls = await election_db.get_polls(race_type=race_type, limit=15, sources=None)
    db_sample = []
    for p in polls[:10]:
        db_sample.append({
            "source": p.get("source"),
            "race": p.get("race") or p.get("race_key"),
            "pollster": p.get("pollster"),
            "margin": p.get("margin"),
            "results": p.get("results"),
            "results_key_count": len(p.get("results") or {}),
        })
    out = {
        "race_type": race_type,
        "db_poll_count": len(polls),
        "db_sample": db_sample,
        "message": "db_sample = what GET /election/polls returns. Each results should have 2 keys for two columns.",
    }
    if scrape and race_type in ("senate", "governor", "house"):
        try:
            election_service.clear_poll_cache(race_type)
            data = await election_service.get_polling_data(race_type)
            raw = (data.get("polls") or [])[:5]
            out["scraped_sample"] = [
                {"race": p.get("race"), "margin": p.get("margin"), "results": p.get("results"), "results_key_count": len(p.get("results") or {})}
                for p in raw
            ]
            out["scrape_message"] = "scraped_sample = what the parser produced before DB. Compare to db_sample."
        except Exception as e:
            out["scrape_error"] = str(e)
    return out


@router.get("/election/debug/context")
async def election_debug_context():
    """Return the full context sent to AI agents in Elections (fact sheet, roster, system prompts). Use to inspect exactly what your agents receive."""
    from .election_ai_service import (
        _get_fact_sheet_context,
        DEFAULT_SYSTEM_PROMPT,
        NEWS_SYSTEM_PROMPT,
    )
    fact_sheet = _get_fact_sheet_context()
    return {
        "fact_sheet_and_roster": fact_sheet,
        "fact_sheet_and_roster_length_chars": len(fact_sheet),
        "system_prompt_polling": DEFAULT_SYSTEM_PROMPT,
        "system_prompt_news": NEWS_SYSTEM_PROMPT,
        "note": "Polling: user message = fact_sheet_and_roster (first) + polling JSON + 'User question: ...'. News: user message = fact_sheet_and_roster (first) + 'Date: ...' + 'User request: ...'. Suggested questions: user message = fact_sheet_and_roster (first) + '---' + generate-5-questions prompt.",
    }


@router.get("/election/trends")
async def get_election_trends(
    race_type: str = Query(..., description="Type of race: president, senate, approval"),
    days: int = Query(90, ge=1, le=365, description="Days of trend data"),
):
    """Return time-series polling averages for charting."""
    rows = await election_db.get_trend_data(race_type, days)
    return {"race_type": race_type, "days": days, "trend": rows}


@router.post("/election/assistant")
async def election_assistant(payload: dict = Body(...)):
    """AI assistant for elections: uses polling context + web search tools."""
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    result = await election_ai_service.ask(
        message=message,
        race_type=payload.get("race_type"),
        metadata=payload.get("metadata"),
        polls=payload.get("polls"),
        model=payload.get("model"),
        temperature=payload.get("temperature", 0.2),
        max_tokens=payload.get("max_tokens", 900)
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result

@router.get("/election/historical")
async def get_historical_results(
    year: Optional[str] = Query(None, description="Election year"),
    race_type: Optional[str] = Query(None, description="Type of race")
):
    return await election_service.get_historical_data(year, race_type)

@router.get("/election/news")
async def get_election_news(
    query: str = Query("", description="Search query for news; empty uses default."),
    model: Optional[str] = Query(None, description="Model or endpoint id for agent-driven news search.")
):
    return await election_service.get_election_news(query, model)

@router.get("/election/approval")
async def get_approval_rating():
    return await election_service.get_polling_data("approval")


@router.get("/election/candidates")
async def get_election_candidates():
    """Return 2026 candidate roster (name, aliases, party, state, office) for frontend lookup and AI context."""
    from .election_candidates import get_candidates
    return {"candidates": get_candidates(), "meta": {"source": "2026_candidates.json"}}


def _summarize_polls_for_questions(polls: list, metadata: dict, race_type: str) -> str:
    """Compact text summary of polls for question-generation prompt."""
    if not polls:
        return "No polls available."
    lines = [f"Total polls: {len(polls)}"]
    if metadata.get("candidates"):
        lines.append(f"Candidates: {', '.join(metadata['candidates'])}")
    for p in polls[:10]:
        pollster = p.get("pollster", "?")
        margin = p.get("margin") or p.get("lead") or ""
        race = p.get("race", "") or p.get("race_key", "")
        added = p.get("added") or p.get("date_added") or ""
        lines.append(f"- {pollster}: {race} {margin}" + (f" ({added})" if added else ""))
    if len(polls) > 10:
        lines.append(f"... and {len(polls) - 10} more polls")
    return "\n".join(lines)


@router.post("/election/questions")
async def generate_election_questions(request: Request):
    """Generate contextual AI questions based on current polling data."""
    body = await request.json()
    race_type = body.get("race_type", "senate")
    polls = body.get("polls", [])
    metadata = body.get("metadata", {})
    model = body.get("model")
    regenerate = body.get("regenerate", False)
    import hashlib
    context_hash = hashlib.sha256(
        json.dumps({"race_type": race_type, "poll_count": len(polls)}, sort_keys=True).encode()
    ).hexdigest()[:12]
    if not regenerate:
        cached = await election_db.get_cached_questions(race_type, context_hash)
        if cached:
            return {"questions": cached, "cached": True}
    api_endpoint = election_ai_service._resolve_endpoint(model)
    if not api_endpoint:
        return {"questions": [], "error": "No model configured"}
    poll_summary = _summarize_polls_for_questions(polls, metadata, race_type)
    from .election_ai_service import _get_fact_sheet_context
    fact_sheet = _get_fact_sheet_context()
    prompt = (
        f"Based on this polling data, generate exactly 5 insightful questions that a user would want to ask about these polls.\n\n"
        f"**Race Type:** {race_type}\n**Poll Summary:**\n{poll_summary}\n\n"
        "Generate questions that explore: 1) Who is leading and by how much 2) How the race has shifted recently "
        "3) Which pollsters show different results 4) What demographics or states matter 5) Historical context. "
        'Format: Each question on its own line, starting with "Q: ". Keep under 15 words each. Be specific to the data.'
    )
    # Candidate roster and fact sheet first so the AI has them before generating questions
    user_content = prompt
    if fact_sheet:
        user_content = (
            "2026 election research and candidate roster (use for context on key races, candidates, and who is D vs R):\n\n"
            + fact_sheet
            + "\n\n---\n\n"
            + prompt
        )
    try:
        response = await election_ai_service._call_api(
            api_endpoint=api_endpoint,
            messages=[{"role": "user", "content": user_content}],
            tools=[],
            temperature=0.8,
            max_tokens=400,
        )
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        questions = [
            line.replace("Q: ", "").replace("**Q:**", "").strip()
            for line in content.split("\n")
            if line.strip().startswith("Q:") or line.strip().startswith("**Q:**")
        ]
        questions = [q for q in questions if len(q) > 10][:5]
        if questions:
            await election_db.cache_questions(race_type, context_hash, questions)
        return {"questions": questions, "cached": False}
    except Exception as e:
        logger.error("Question generation failed: %s", e)
        return {"questions": [], "error": str(e)}


# In-memory cache for Monte Carlo simulation results (key = cache_key from election_simulation)
_election_simulation_cache: Dict[str, Any] = {}


@router.get("/election/map")
async def get_election_map(race_type: str = Query("senate", description="Race type for map")):
    """Get user map data and state-level averages only from the same polls as the Polls tab (no other DB aggregate)."""
    user_data = await election_db.get_all_map_data(race_type)
    if race_type in ("senate", "governor"):
        sources = ["rcp"]
    elif race_type == "house":
        sources = ["racetothewh"]
    else:
        sources = None
    # Use last 18 months and high limit so map averages and trends keep using older polls as new ones arrive
    polls = await election_db.get_polls(race_type=race_type, limit=2000, sources=sources, days_back=548) if race_type in ("senate", "governor", "house") else []
    scraped = election_service.compute_state_averages_from_polls(polls, use_quality_weights=True)
    return {"user_data": user_data, "scraped_averages": scraped}


@router.get("/election/simulation/results")
async def get_simulation_results(race_type: str = Query("senate", description="senate, governor, or house")):
    """Return cached simulation results if available. Call POST /election/simulation/run to populate."""
    if race_type not in ("senate", "governor", "house"):
        return {"error": "race_type must be senate, governor, or house", "result": None}
    entry = _election_simulation_cache.get(race_type)
    if entry:
        return {
            "result": entry["result"],
            "last_updated_ts": entry.get("ts"),
            "n_simulations": entry.get("n_simulations"),
        }
    return {"result": None, "message": "No cached simulation. Run simulation first."}


@router.post("/election/simulation/run")
async def run_election_simulation(
    race_type: str = Query("senate", description="senate, governor, or house"),
    n_simulations: int = Query(10000, ge=1000, le=50000, description="Number of Monte Carlo runs"),
    use_calibration: bool = Query(True, description="Apply special/off-year calibration shift"),
    calibration_weight: float = Query(1.0, ge=0.0, le=2.0, description="Scale for calibration swing (0=ignore, 1=full, 0.5=half). How much 2024→now swing influences the baseline."),
    use_sophisticated_forecast: bool = Query(True, description="Quality-weighted polls + subtle fundamentals prior"),
    fundamental_weight_base: float = Query(
        ELECTION_FUNDAMENTAL_WEIGHT_BASE_DEFAULT,
        ge=0.0,
        le=0.5,
        description="Prior influence on state mean (e.g. 0.05–0.20). Exposed for tuning.",
    ),
    time_decay_curve: str = Query(
        ELECTION_TIME_DECAY_CURVE_DEFAULT,
        description="Prior time curve: 'decay' (prior fades as election nears) or 'flat'.",
    ),
    state_lean_multiplier: float = Query(
        ELECTION_STATE_LEAN_MULTIPLIER_DEFAULT,
        ge=0.0,
        le=2.0,
        description="Scale for state partisan lean (0=ignore, 1=full). Exposed for tuning.",
    ),
):
    """Run Monte Carlo simulation. State means = polls + subtle fundamentals prior (no flattening).
    Prior params are exposed for tuning; they are not modified during the run.
    Note: First request after server start is often 10–15s (DB/import cold start); subsequent runs
    are fast (~0.1s sim). Every run executes the full simulation; there is no cache read for POST."""
    if race_type not in ("senate", "governor", "house"):
        raise HTTPException(status_code=400, detail="race_type must be senate, governor, or house")
    if time_decay_curve not in ("decay", "flat"):
        time_decay_curve = election_forecast.TIME_DECAY_CURVE_DEFAULT
    sources = ["racetothewh"] if race_type == "house" else ["rcp"]
    polls = await election_db.get_polls(race_type=race_type, limit=2000, sources=sources, days_back=548)
    state_averages = election_service.compute_state_averages_from_polls(
        polls, use_quality_weights=use_sophisticated_forecast
    )
    if not state_averages:
        return {"error": "No state polling averages available", "result": None}

    import logging
    _log = logging.getLogger(__name__)
    run_id = uuid.uuid4().hex[:12]  # unique per run so client can verify backend actually ran
    _log.info(
        "election/simulation/run: starting run_id=%s race_type=%s n_simulations=%s states=%s use_calibration=%s",
        run_id, race_type, n_simulations, len(state_averages), use_calibration,
    )

    calibration_entries = election_simulation.load_calibration_for_analysis() if use_calibration else None
    calibration_n = 0
    calibration_swing_pts: Optional[float] = None
    if calibration_entries:
        swing_sum, swing_w = 0.0, 0.0
        for e in calibration_entries:
            s = e.get("swing_toward_d")
            w = float(e.get("weight", 1.0))
            if s is not None and w > 0:
                try:
                    swing_sum += float(s) * w
                    swing_w += w
                except (TypeError, ValueError):
                    pass
        if swing_w > 0:
            calibration_n = len([e for e in calibration_entries if e.get("swing_toward_d") is not None])
            calibration_swing_pts = round(swing_sum / swing_w, 2)

    # Combined calibration: generic ballot + approval + special elections (quality-weighted for GB and approval)
    calibration_swing_effective: Optional[float] = None
    calibration_combined_meta: Optional[Dict[str, Any]] = None
    approval_data: Optional[Dict[str, Any]] = None
    generic_ballot_data: Optional[Dict[str, Any]] = None
    if use_calibration:
        approval_data, generic_ballot_data = await asyncio.gather(
            election_service.get_polling_data("approval"),
            election_service.get_polling_data("generic_ballot"),
        )
        gb_dem_share, gb_n, _ = election_service.compute_quality_weighted_generic_ballot(generic_ballot_data or {})
        approval_net, app_n, _ = election_service.compute_quality_weighted_approval(approval_data or {})
        effective_shift, calibration_combined_meta = election_forecast.compute_combined_calibration_shift(
            generic_ballot_dem_share=gb_dem_share,
            approval_net=approval_net,
            special_election_swing_pts=calibration_swing_pts,
            calibration_weight=calibration_weight,
            president_party="R",
            race_type=race_type,
        )
        calibration_swing_effective = round(effective_shift, 2) if effective_shift != 0 else None

    forecast_metadata = None
    if use_sophisticated_forecast:
        if approval_data is None and generic_ballot_data is None:
            approval_data, generic_ballot_data = await asyncio.gather(
                election_service.get_polling_data("approval"),
                election_service.get_polling_data("generic_ballot"),
            )
        fundamentals_share, fund_meta = election_forecast.compute_fundamentals_national_share(
            approval_data, generic_ballot_data, calibration_entries,
            calibration_swing_weight=calibration_weight,
        )
        days = election_forecast.days_to_election()
        # Prior: small state-level nudge. Polls stay primary; state differentiation preserved.
        adjusted_averages = {}
        prior_sample: Optional[Dict[str, Any]] = None
        for state, avg in state_averages.items():
            dem = avg.get("dem_avg")
            gop = avg.get("gop_avg") or avg.get("rep_avg")
            if dem is not None and gop is not None and (dem + gop) > 0:
                poll_dem_share = dem / (dem + gop) * 100.0
                adjustment_pts, adj_meta = election_forecast.compute_fundamentals_prior_adjustment(
                    poll_dem_share,
                    fundamentals_share,
                    state,
                    fundamental_weight_base=fundamental_weight_base,
                    days_to_election=days,
                    state_lean_multiplier=state_lean_multiplier,
                    time_decay_curve=time_decay_curve,
                )
                final_dem_share = poll_dem_share + adjustment_pts
                final_dem_share = max(0.0, min(100.0, final_dem_share))
                if prior_sample is None:
                    prior_sample = {"state": state, **adj_meta}
                adjusted_averages[state] = {
                    **avg,
                    "dem_avg": round(final_dem_share, 2),
                    "gop_avg": round(100.0 - final_dem_share, 2),
                }
            else:
                adjusted_averages[state] = avg
        state_averages = adjusted_averages
        # Dynamic uncertainty: sigma scales with poll count, quality, and time to election
        base_sigma = 2.5
        poll_counts = [v.get("poll_count") or 0 for v in state_averages.values()]
        confs = [v.get("pollster_confidence") for v in state_averages.values() if v.get("pollster_confidence") is not None]
        avg_poll_count = sum(poll_counts) / len(poll_counts) if poll_counts else 0
        avg_pollster_conf = sum(confs) / len(confs) if confs else 0.5
        sigma_quality = election_forecast.sigma_from_data_quality(
            int(round(avg_poll_count)), avg_pollster_conf, days, base_sigma=base_sigma
        )
        quality_sigma_multiplier = sigma_quality / base_sigma
        forecast_metadata = {
            "fundamentals_dem_share": fund_meta.get("dem_share"),
            "fundamentals_components": fund_meta.get("components"),
            "days_to_election": days,
            "sigma_quality": round(sigma_quality, 2),
            "quality_sigma_multiplier": round(quality_sigma_multiplier, 3),
            "prior_params": {
                "fundamental_weight_base": fundamental_weight_base,
                "time_decay_curve": time_decay_curve,
                "state_lean_multiplier": state_lean_multiplier,
            },
            "prior_adjustment_sample": prior_sample,
        }
    else:
        quality_sigma_multiplier = None

    # Apply combined calibration (GB + approval + special elections) to state means.
    if use_calibration and calibration_swing_effective is not None and calibration_swing_effective != 0:
        for state, avg in list(state_averages.items()):
            dem = avg.get("dem_avg")
            gop = avg.get("gop_avg") or avg.get("rep_avg")
            if dem is not None and gop is not None and (dem + gop) > 0:
                new_dem = max(0.0, min(100.0, float(dem) + calibration_swing_effective))
                state_averages[state] = {
                    **avg,
                    "dem_avg": round(new_dem, 2),
                    "gop_avg": round(100.0 - new_dem, 2),
                }

    result = election_simulation.run_and_cache(
        state_averages, race_type=race_type, n_simulations=n_simulations,
        cache=_election_simulation_cache, cache_key=race_type,
        use_calibration=use_calibration,
        use_systematic_error=use_sophisticated_forecast,
        quality_sigma_multiplier=quality_sigma_multiplier,
    )
    _log.info(
        "election/simulation/run: completed run_id=%s race_type=%s elapsed_sec=%s races_included=%s",
        run_id, race_type, result.get("elapsed_sec"), result.get("races_included"),
    )
    out = {
        "result": result,
        "run_id": run_id,
        "last_updated_ts": _election_simulation_cache.get(race_type, {}).get("ts"),
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "race_type": race_type,
        "n_simulations": n_simulations,
        "use_calibration": use_calibration,
        "timing_note": "First run after server start is often slower (cold start); each run runs the full simulation.",
    }
    if use_calibration:
        out["calibration_n"] = calibration_n
        out["calibration_weight"] = calibration_weight
        if calibration_swing_pts is not None:
            out["calibration_swing_pts"] = calibration_swing_pts
        if calibration_swing_effective is not None:
            out["calibration_swing_effective"] = calibration_swing_effective
        if calibration_combined_meta:
            out["calibration_combined_shift"] = calibration_combined_meta.get("combined_shift")
            out["calibration_components"] = calibration_combined_meta.get("components", {})
            out["calibration_weights_used"] = calibration_combined_meta.get("weights_used", {})
    if forecast_metadata:
        out["forecast_metadata"] = forecast_metadata
    return out


@router.get("/election/simulation/calibration")
async def get_simulation_calibration():
    """List calibration entries that have 2024 R margin and swing data (only these are used for analysis)."""
    entries = election_simulation.load_calibration_for_analysis()
    return {"entries": entries}


class CalibrationEntryCreate(BaseModel):
    label: str = ""
    type: str = "special"
    state: str
    date: str
    dem_actual_pct: float
    poll_avg_pct: Optional[float] = None  # when set, used for overperformance shift; omit when no pre-election polls
    rep_actual_pct: Optional[float] = None
    weight: float = 1.0
    region: Optional[str] = None
    note: Optional[str] = None
    trump_2024_margin: Optional[float] = None  # 2024 pres margin in region (+ = Trump won by X)
    swing_toward_d: Optional[float] = None     # midterm swing (positive = D gained vs 2024)


@router.post("/election/simulation/calibration")
async def add_simulation_calibration(body: CalibrationEntryCreate):
    """Add a calibration result. Only entries with poll_avg_pct contribute to simulation shift (D overperformance)."""
    entry = election_simulation.add_calibration_entry(
        label=body.label,
        entry_type=body.type,
        state=body.state,
        date=body.date,
        dem_actual_pct=body.dem_actual_pct,
        poll_avg_pct=body.poll_avg_pct,
        weight=body.weight,
        region=body.region,
        note=body.note,
        trump_2024_margin=body.trump_2024_margin,
        swing_toward_d=body.swing_toward_d,
        rep_actual_pct=body.rep_actual_pct,
    )
    return {"entry": entry, "message": "Added. Re-run simulation to apply."}


@router.delete("/election/simulation/calibration/{entry_id}")
async def delete_simulation_calibration(entry_id: str):
    """Remove a calibration entry by id."""
    ok = election_simulation.delete_calibration_entry(entry_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Calibration entry not found")
    return {"status": "ok"}


@router.get("/election/simulation/calibration/ballotpedia-scrape")
async def ballotpedia_scrape(
    since_nov_2025: bool = True,
    include_governors: bool = True,
    include_federal: bool = True,
    include_state_leg: bool = True,
):
    """Scrape Ballotpedia for off-year/special results: federal House/Senate, VA/NJ governors, and state house/senate specials (e.g. TX Senate 9). Returns list only; does not modify calibration."""
    results = ballotpedia_scraper.scrape_all(
        since_nov_2025=since_nov_2025,
        include_governors=include_governors,
        include_federal=include_federal,
        include_state_leg=include_state_leg,
    )
    return {"results": results, "count": len(results)}


@router.post("/election/simulation/calibration/ballotpedia-import")
async def ballotpedia_import_to_calibration(since_nov_2025: bool = True):
    """Run Ballotpedia scrape and add new results into calibration (by label+date). Returns scraped list, added, and skipped."""
    out = ballotpedia_scraper.run_and_import_to_calibration(since_nov_2025=since_nov_2025, merge=True)
    return out


@router.post("/election/simulation/calibration/ballotpedia-import-stream")
async def ballotpedia_import_stream(since_nov_2025: bool = True):
    """Stream progress (SSE) while scraping Ballotpedia and importing to calibration. Events: progress (current, total, message), then done (added, skipped, scraped_count)."""
    import queue as queue_module
    q = queue_module.Queue()

    def run():
        try:
            def cb(current: int, total: int, message: str):
                q.put({"type": "progress", "current": current, "total": total, "message": message})
            out = ballotpedia_scraper.run_and_import_to_calibration(
                since_nov_2025=since_nov_2025, merge=True, progress_callback=cb
            )
            q.put({
                "type": "done",
                "added": len(out.get("added") or []),
                "skipped": len(out.get("skipped") or []),
                "scraped_count": len(out.get("scraped") or []),
            })
        except Exception as e:
            q.put({"type": "done", "error": str(e)})

    thread = threading.Thread(target=run)
    thread.start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("type") == "done":
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/election/simulation/calibration/refresh-2024-state-margins")
async def refresh_2024_state_margins():
    """Refresh 2024 baseline: fetch county CSV from tonmcg repo, aggregate by state, write pres_2024_state_margins.json. Governor calibration rows use it for 2024 vs now (margin + swing)."""
    from .pres_2024_county_loader import write_pres_2024_state_margins_json, load_pres_2024_state_margins
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, write_pres_2024_state_margins_json)
    margins = load_pres_2024_state_margins()
    return {"status": "ok", "path": str(path), "states": len(margins), "sample": {k: margins[k] for k in ["VA", "NJ", "TX"] if k in margins}}


@router.post("/election/simulation/calibration/refresh-va-2024-districts")
async def refresh_va_2024_districts():
    """Fetch VPAP 2024 presidential results by VA House district, write state_leg_2024_presidential_margins.json (VA_House). VA House calibration rows then get 2024 R margin and Swing (D) from district-level data."""
    from .vpap_2024_loader import write_va_house_to_state_leg_margins
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, write_va_house_to_state_leg_margins)
    return {"status": "ok", "path": str(path), "message": "VA House 2024 district data loaded from VPAP. Reload calibration list to see 2024 margin and swing."}


@router.post("/election/simulation/calibration/refresh-va-nj-2024-districts")
async def refresh_va_nj_2024_districts():
    """Load VA House 2024 from VPAP; NJ Assembly when a district-level source exists. Writes state_leg_2024_presidential_margins.json."""
    from .vpap_2024_loader import write_va_house_to_state_leg_margins, fetch_nj_assembly_2024_if_available
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, write_va_house_to_state_leg_margins)
    nj_loaded = await loop.run_in_executor(None, fetch_nj_assembly_2024_if_available)
    if nj_loaded:
        message = "VA House and NJ Assembly 2024 district data loaded. Reload calibration list to see 2024 margin and swing."
    else:
        message = "VA House 2024 loaded from VPAP. NJ Assembly: no public district-level source available yet (—)."
    return {"status": "ok", "path": str(path), "message": message}


@router.put("/election/map/{state}")
async def update_election_map(state: str, request: Request):
    """User updates polling data for a state."""
    body = await request.json()
    await election_db.upsert_map_data(
        state=state.upper(),
        race_type=body.get("race_type", "senate"),
        candidate_1_name=body.get("candidate_1_name"),
        candidate_1_party=body.get("candidate_1_party"),
        candidate_1_pct=body.get("candidate_1_pct"),
        candidate_2_name=body.get("candidate_2_name"),
        candidate_2_party=body.get("candidate_2_party"),
        candidate_2_pct=body.get("candidate_2_pct"),
        margin=body.get("margin"),
        source_note=body.get("source_note"),
    )
    return {"status": "ok"}


@router.delete("/election/map/{state}")
async def delete_election_map(state: str, race_type: str = Query("senate")):
    """Remove user map data for a state."""
    await election_db.delete_map_data(state.upper(), race_type)
    return {"status": "ok"}


# --- Conversation summaries (JSON files in static/summaries) ---
class SummaryCreate(BaseModel):
    title: str
    content: str


@router.get("/summaries")
async def list_summaries():
    """List all saved summary JSON files."""
    out = []
    if not summaries_dir.exists():
        return out
    for p in summaries_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                out.append({
                    "id": data.get("id", p.stem),
                    "title": data.get("title", p.stem),
                    "content": data.get("content", ""),
                    "date": data.get("date", ""),
                })
        except Exception as e:
            logger.warning(f"Skip invalid summary file {p}: {e}")
    out.sort(key=lambda x: x.get("date") or "", reverse=True)
    return out


@router.post("/summaries", status_code=201)
async def create_summary(body: SummaryCreate):
    """Save a new summary as a JSON file in static/summaries."""
    summary_id = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + str(uuid.uuid4())[:8]
    payload = {
        "id": summary_id,
        "title": body.title or f"Summary {datetime.datetime.utcnow().isoformat()}",
        "content": body.content or "",
        "date": datetime.datetime.utcnow().isoformat() + "Z",
    }
    path = summaries_dir / f"{summary_id}.json"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


@router.get("/summaries/{summary_id}")
async def get_summary(summary_id: str):
    """Get one summary by id (filename without .json)."""
    if ".." in summary_id or "/" in summary_id or "\\" in summary_id:
        raise HTTPException(status_code=400, detail="Invalid summary id")
    path = summaries_dir / f"{summary_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.delete("/summaries/{summary_id}")
async def delete_summary(summary_id: str):
    """Delete a summary JSON file."""
    if ".." in summary_id or "/" in summary_id or "\\" in summary_id:
        raise HTTPException(status_code=400, detail="Invalid summary id")
    path = summaries_dir / f"{summary_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    path.unlink()
    return {"status": "deleted", "id": summary_id}


# --- Chess tab: engine + AI move selection ---
@router.get("/chess/status")
async def chess_status():
    """Return whether Stockfish is available and default ELO range."""
    path = chess_engine_service.get_engine_path()
    return {
        "available": path is not None,
        "engine_path": path if path else None,
        "elo_min": 800,
        "elo_max": 3000,
        "personalities": chess_ai_service.PERSONALITIES,
    }


@router.post("/chess/analyze")
async def chess_analyze(payload: dict = Body(...)):
    """Analyze position: FEN -> top N moves with evals and classifications."""
    fen = (payload.get("fen") or "").strip()
    if not fen:
        raise HTTPException(status_code=400, detail="fen is required")
    multipv = max(1, min(10, int(payload.get("multipv", 5))))
    elo = payload.get("elo")
    if elo is not None:
        elo = max(800, min(3000, int(elo)))
    try:
        result = await chess_engine_service.analyze_position(
            fen=fen,
            multipv=multipv,
            elo=elo,
            analysis_time=float(payload.get("analysis_time", 0.5)),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/chess/ai-move")
async def chess_ai_move(request: Request, payload: dict = Body(...)):
    """Get engine candidates, then AI chooses move and returns commentary."""
    fen = (payload.get("fen") or "").strip()
    if not fen:
        raise HTTPException(status_code=400, detail="fen is required")
    elo = max(800, min(3000, int(payload.get("elo", 1600))))
    personality = (payload.get("personality") or "balanced").lower()
    if personality not in chess_ai_service.PERSONALITIES:
        personality = "balanced"
    use_llm = payload.get("use_llm", True)
    model_name = payload.get("model_name")

    try:
        analysis = await chess_engine_service.analyze_position(
            fen=fen,
            multipv=10,
            elo=elo,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    candidates = analysis.get("candidates", [])
    if not candidates:
        return {
            "move_uci": None,
            "move_san": None,
            "commentary": "No legal moves.",
            "evaluation_cp": None,
            "candidates": [],
            "game_over": analysis.get("is_game_over"),
            "result": analysis.get("result"),
        }

    if analysis.get("is_game_over"):
        return {
            "move_uci": None,
            "move_san": None,
            "commentary": "Game over.",
            "evaluation_cp": analysis.get("evaluation_cp"),
            "candidates": candidates,
            "game_over": True,
            "result": analysis.get("result"),
        }

    model_manager = getattr(request.app.state, "model_manager", None)
    use_api = chess_ai_service.is_api_endpoint(model_name) if model_name else False
    if use_llm and model_name and (model_manager or use_api):
        logger.info("Chess AI: calling LLM for move selection (model=%s, elo=%s, personality=%s)", model_name, elo, personality)
        selection = await chess_ai_service.select_move_with_llm(
            model_manager=model_manager,
            model_name=model_name,
            candidates=candidates,
            elo=elo,
            personality=personality,
            fen=fen,
            turn=analysis.get("turn", "white"),
            game_context=payload.get("game_context"),
            move_history=payload.get("move_history") or [],
        )
        logger.info("Chess AI: chosen move=%s commentary=%s", selection.get("move_san"), (selection.get("commentary") or "")[:120])
    else:
        if use_llm and model_name and not use_api and not model_manager:
            logger.warning("Chess AI: no model_manager and model %s is not an API endpoint; using rule-based fallback", model_name)
        selection = chess_ai_service.select_move_without_llm(
            candidates=candidates,
            elo=elo,
            personality=personality,
        )

    # Engine returns evaluation in pawns; normalize to centipawns for the frontend eval bar
    eval_cp = selection.get("evaluation_cp")
    if eval_cp is not None and abs(eval_cp) <= 20:
        eval_cp = round(eval_cp * 100)

    return {
        "move_uci": selection.get("move_uci"),
        "move_san": selection.get("move_san"),
        "commentary": selection.get("commentary"),
        "evaluation_cp": eval_cp,
        "candidates": candidates,
        "chosen_index": selection.get("index"),
        "game_over": False,
        "result": None,
    }


@router.post("/chess/validate-move")
async def chess_validate_move(payload: dict = Body(...)):
    fen = (payload.get("fen") or "").strip()
    move_uci = (payload.get("move_uci") or "").strip()
    if not fen or not move_uci:
        raise HTTPException(status_code=400, detail="fen and move_uci required")
    result = await chess_engine_service.validate_move(fen, move_uci)
    return result


@router.post("/chess/game-commentary")
async def chess_game_commentary(request: Request, payload: dict = Body(...)):
    """Get AI commentary on a finished game (move history + result)."""
    move_history = payload.get("move_history") or []
    result = (payload.get("result") or "*").strip()
    model_name = payload.get("model_name")
    model_manager = getattr(request.app.state, "model_manager", None)
    commentary = await chess_ai_service.get_game_commentary(
        model_manager=model_manager,
        model_name=model_name,
        move_history=move_history,
        result=result,
    )
    return {"commentary": commentary}


@router.post("/chess/analyze-game")
async def chess_analyze_game(request: Request, payload: dict = Body(...)):
    """Analyze the final position with the engine and optionally get AI summary."""
    fen = (payload.get("fen") or "").strip()
    if not fen:
        raise HTTPException(status_code=400, detail="fen required")
    try:
        analysis = await chess_engine_service.analyze_position(fen=fen, multipv=1, elo=None)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400 if isinstance(e, ValueError) else 503, detail=str(e))
    final_eval = analysis.get("evaluation_cp")
    summary = ""
    model_name = payload.get("model_name")
    model_manager = getattr(request.app.state, "model_manager", None)
    move_history = payload.get("move_history") or []
    result = (payload.get("result") or "*").strip()
    if (model_name and (model_manager or is_api_endpoint(model_name))) and (move_history or result != "*"):
        if isinstance(move_history, list):
            moves_text = " ".join(
                (m.get("san", m) if isinstance(m, dict) else m) for m in move_history[:60]
            )
        else:
            moves_text = str(move_history)[:500]
        prompt = f"Final position (result {result}). Moves: {moves_text}. In one or two sentences, what was the decisive factor or key moment?"
        try:
            if is_api_endpoint(model_name):
                endpoint_config = get_configured_endpoint(model_name)
                if endpoint_config:
                    request_data = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 80,
                        "temperature": 0.4,
                    }
                    endpoint_config, url, prepared_data = prepare_endpoint_request(model_name, request_data)
                    summary = (
                        await collect_openai_compatible_stream_text(endpoint_config, url, prepared_data)
                    ).strip()
            elif model_manager:
                from . import inference
                response = await inference.generate_text(
                    model_manager=model_manager,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=80,
                    temperature=0.4,
                    gpu_id=0,
                )
                summary = (response.get("choices", [{}])[0].get("text") if isinstance(response, dict) else response) or ""
        except Exception as e:
            logger.warning("Chess analyze-game AI summary failed: %s", e)
    return {"summary": summary, "final_eval": final_eval, "candidates": analysis.get("candidates", [])[:3]}


@router.post("/chess/analyze-game-full")
async def chess_analyze_game_full(request: Request, payload: dict = Body(...)):
    """Analyze every position in the game with the engine; optionally add AI commentary per move. Returns moves with scores for replay."""
    move_history = payload.get("move_history") or []
    result = (payload.get("result") or "*").strip()
    model_name = payload.get("model_name")
    add_commentary = payload.get("add_commentary", True)
    if not move_history:
        raise HTTPException(status_code=400, detail="move_history required")
    import chess
    san_list = []
    for m in move_history:
        if isinstance(m, dict):
            san_list.append((m.get("san") or m.get("move") or "").strip())
        else:
            san_list.append(str(m).strip())
    san_list = [s for s in san_list if s]
    board = chess.Board()
    positions = [{"move_index": 0, "san": None, "side": None, "fen_after": board.fen()},]
    for i, san in enumerate(san_list):
        try:
            move = board.push_san(san)
            side = "w" if board.turn == chess.BLACK else "b"
            positions.append({
                "move_index": i + 1,
                "san": san,
                "side": side,
                "fen_after": board.fen(),
            })
        except ValueError:
            break
    moves_with_evals = []
    for i, pos in enumerate(positions):
        fen_after = pos["fen_after"]
        try:
            analysis_after = await chess_engine_service.analyze_position(
                fen=fen_after, multipv=1, elo=None, analysis_time=0.25
            )
            eval_cp = analysis_after.get("evaluation_cp")
            if eval_cp is not None and abs(eval_cp) <= 100:
                eval_cp = round(eval_cp * 100)
            candidates_after = analysis_after.get("candidates") or []
            continuation_pv_san = candidates_after[0].get("pv_san") if candidates_after else None
        except (ValueError, RuntimeError):
            eval_cp = None
            continuation_pv_san = None

        best_move_san = None
        best_move_pv_san = None
        best_eval_cp = None
        judgment = "best"
        if i >= 1:
            fen_before = positions[i - 1]["fen_after"]
            try:
                analysis_before = await chess_engine_service.analyze_position(
                    fen=fen_before, multipv=2, elo=None, analysis_time=0.3
                )
                cands = analysis_before.get("candidates") or []
                if cands:
                    b = chess.Board(fen_before)
                    best = cands[0]
                    best_move_san = best.get("move_san")
                    best_move_pv_san = best.get("pv_san")
                    be = best.get("score_cp")  # pawns, from side-to-move's view
                    if be is not None:
                        if b.turn == chess.BLACK:
                            be = -be
                        best_eval_cp = round(be * 100)
                    else:
                        best_eval_cp = None
                    played = (pos.get("san") or "").strip()
                    if best_move_san and played and played != best_move_san and best_eval_cp is not None and eval_cp is not None:
                        side = pos.get("side") or "w"
                        if side == "w":
                            loss_cp = best_eval_cp - eval_cp
                        else:
                            loss_cp = eval_cp - best_eval_cp
                        if loss_cp >= 150:
                            judgment = "blunder"
                        elif loss_cp >= 75:
                            judgment = "mistake"
                        elif loss_cp >= 25:
                            judgment = "inaccuracy"
            except (ValueError, RuntimeError):
                pass

        moves_with_evals.append({
            "move_index": pos["move_index"],
            "san": pos["san"],
            "side": pos["side"],
            "fen_after": fen_after,
            "evaluation_cp": eval_cp,
            "continuation_pv_san": continuation_pv_san,
            "best_move_san": best_move_san,
            "best_move_pv_san": best_move_pv_san,
            "best_eval_cp": best_eval_cp,
            "judgment": judgment,
        })
    for m in moves_with_evals:
        m["commentary"] = None
    if add_commentary and model_name and (getattr(request.app.state, "model_manager", None) or chess_ai_service.is_api_endpoint(model_name)):
        model_manager = getattr(request.app.state, "model_manager", None)
        only_with_move = [m for m in moves_with_evals if m["san"]]
        if only_with_move:
            # One huge request (e.g. 80 moves, 6k tokens out) often hits server/proxy timeouts and
            # disconnects regardless of which model is used. Chunk so each request finishes in ~30s.
            COMMENTARY_CHUNK_SIZE = 25
            all_comments = []
            for start in range(0, len(only_with_move), COMMENTARY_CHUNK_SIZE):
                chunk = only_with_move[start : start + COMMENTARY_CHUNK_SIZE]
                try:
                    chunk_comments = await chess_ai_service.get_per_move_commentary(
                        model_manager=model_manager,
                        model_name=model_name,
                        moves_with_evals=chunk,
                        result=result,
                    )
                    all_comments.extend(chunk_comments[: len(chunk)])
                except Exception as e:
                    logger.warning("Per-move commentary chunk %d-%d failed: %s", start, start + len(chunk), type(e).__name__)
                    all_comments.extend([""] * len(chunk))
            idx = 0
            for m in moves_with_evals:
                if m["san"] is not None:
                    m["commentary"] = all_comments[idx] if idx < len(all_comments) else ""
                    idx += 1
    return {"moves": moves_with_evals, "result": result}


@router.post("/chess/deep-analysis")
async def chess_deep_analysis(request: Request, payload: dict = Body(...)):
    """
    Tier 2: Research what strong players/sources say about the current position.
    Uses Lichess Opening Explorer + web search; synthesizes with citations.
    Does NOT ask the LLM to evaluate the position—only to summarize external sources.
    """
    fen = (payload.get("fen") or "").strip()
    if not fen:
        raise HTTPException(status_code=400, detail="fen required")
    move_history = payload.get("move_history") or []
    model_name = payload.get("model_name")

    engine_eval_str = None
    best_move = None
    pv_san = None
    try:
        analysis = await chess_engine_service.analyze_position(
            fen=fen, multipv=1, elo=None, analysis_time=0.4
        )
        eval_cp = analysis.get("evaluation_cp")
        if eval_cp is not None:
            if abs(eval_cp) <= 100:
                eval_cp = eval_cp * 100
            engine_eval_str = f"{eval_cp:+.0f} cp" if eval_cp is not None else None
        cands = analysis.get("candidates") or []
        if cands:
            best_move = cands[0].get("move_san")
            pv_san = cands[0].get("pv_san")
    except (ValueError, RuntimeError):
        pass

    async def web_search_fn(q: str, max_results: int = 5) -> str:
        from .web_search_service import perform_web_search
        return await perform_web_search(q, max_results)

    model_manager = getattr(request.app.state, "model_manager", None)
    from . import chess_research_agent
    result = await chess_research_agent.run_deep_analysis(
        fen=fen,
        engine_eval=engine_eval_str,
        best_move=best_move,
        pv_san=pv_san,
        move_history=move_history,
        web_search_fn=web_search_fn,
        model_manager=model_manager,
        model_name=model_name,
    )
    return result


@router.post("/chess/historian/chat")
async def chess_historian_chat(request: Request, payload: dict = Body(...)):
    """Chat with the Chess Historian (research, stories, can return PGN to load)."""
    messages = payload.get("messages") or []
    model_name = payload.get("model_name")

    async def web_search_fn(q: str, max_results: int = 5) -> str:
        from .web_search_service import perform_web_search
        return await perform_web_search(q, max_results)

    model_manager = getattr(request.app.state, "model_manager", None)
    from . import chess_historian
    persona_prompt = payload.get("persona_prompt")
    result = await chess_historian.chat(
        messages=messages,
        model_manager=model_manager,
        model_name=model_name,
        request=request,
        web_search_fn=web_search_fn,
        persona_prompt=persona_prompt,
    )
    return result


@router.post("/chess/historian/fact")
async def chess_historian_fact(request: Request, payload: dict = Body(...)):
    """Get one short chess history fact (e.g. for 30s idle ticker). Uses web search so facts are researched, not hallucinated. Send recent_facts to avoid repetition."""
    model_name = payload.get("model_name")
    recent_facts = payload.get("recent_facts")
    if recent_facts is not None and not isinstance(recent_facts, list):
        recent_facts = [str(recent_facts)] if recent_facts else []
    model_manager = getattr(request.app.state, "model_manager", None)

    async def web_search_fn(q: str, max_results: int = 6) -> str:
        from .web_search_service import perform_web_search
        return await perform_web_search(q, max_results)

    from . import chess_historian
    fact_text = await chess_historian.random_fact(
        model_manager=model_manager,
        model_name=model_name,
        request=request,
        recent_facts=recent_facts,
        search_context=None,
        web_search_fn=web_search_fn,
    )
    return {"fact": fact_text or ""}


@router.post("/chess/historian/persona")
async def chess_historian_persona(request: Request, payload: dict = Body(...)):
    """Generate a short persona description for the Chess Historian (AI describes its own style)."""
    model_name = payload.get("model_name")
    model_manager = getattr(request.app.state, "model_manager", None)
    from . import chess_historian
    persona_text = await chess_historian.generate_persona(
        model_manager=model_manager,
        model_name=model_name,
        request=request,
    )
    return {"persona": (persona_text or "").strip() or "I'm the Chess Historian—warm, curious, and full of stories about players, games, and the history of the game."}


# --- Demo showcase (fabricated test user for call mode / memory demos) ---
@app.router.get("/demo/showcase/status")
async def demo_showcase_status():
  """Whether the fabricated demo profile memories are installed on disk."""
  try:
    from . import demo_showcase
    return {"status": "success", **demo_showcase.get_status()}
  except Exception as e:
    logger.error(f"Demo showcase status failed: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))


@app.router.get("/demo/showcase/pack")
async def demo_showcase_pack():
  """Return the full demo showcase pack (frontend + metadata; backend arrays included for reference)."""
  try:
    from . import demo_showcase
    return {"status": "success", "pack": demo_showcase.load_pack()}
  except FileNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
  except Exception as e:
    logger.error(f"Demo showcase pack failed: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))


@app.router.post("/demo/showcase/install")
async def demo_showcase_install(request: Request):
  """Write fabricated profile + agentic memories to disk and optionally set active profile."""
  try:
    from . import demo_showcase
    body = {}
    try:
      body = await request.json()
    except Exception:
      body = {}
    set_active = body.get("set_active", True) if isinstance(body, dict) else True
    result = demo_showcase.install_backend(set_active=bool(set_active))
    if set_active:
      request.app.state.active_profile_id = result.get("user_id")
    return result
  except FileNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
  except Exception as e:
    logger.error(f"Demo showcase install failed: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))


# Retired modules remain in source but are not registered or exposed.
if not module_enabled("chess"):
    router.routes = [
        route for route in router.routes
        if not getattr(route, "path", "").startswith("/chess")
    ]

# mount the "generate" router
app.include_router(router)
if auth_router is not None:
    app.include_router(auth_router)
if market_sim_router is not None:
    app.include_router(market_sim_router)
app.include_router(outreach_router)
app.include_router(remote_router)
app.include_router(d_id_router)
app.include_router(voice_sculpt_router, prefix="/voice-sculpt")
app.include_router(rembg_router, prefix="/rembg")
app.include_router(model_library_router)
app.include_router(character_datasets_router)
app.include_router(provider_catalog_router)
app.include_router(sillytavern_bridge_router)
app.include_router(mirid_docs_router)

# Room background image gallery
from .room_gallery_routes import router as room_gallery_router
app.include_router(room_gallery_router)

# mount your memory endpoints under the "/memory" prefix
app.include_router(memory_router, prefix="/memory")
app.include_router(alignment_router, prefix="/memory/alignment")
if chatlog_condenser_router is not None:
    app.include_router(chatlog_condenser_router, prefix="/memory/chatlog-condenser")
app.include_router(document_router)
# Add OpenAI compatibility layer
app.include_router(openai_router)
# Sanctuary Evolution — agentic orchestration pipeline
app.include_router(sanctuary_router, prefix="/agentic")
logger.info("🔮 Sanctuary agentic endpoints available at /agentic/turn and /agentic/state")
# TTS service router removed - TTS now runs separately on port 8002
logger.info("🔗 OpenAI-compatible API endpoints available at /v1/chat/completions and /v1/models")
