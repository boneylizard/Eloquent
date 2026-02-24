# --- START: FORCE LLAMA_CPP IMPORT FIRST ---
# This MUST be the absolute first import to ensure llama.cpp claims the CUDA context
# before any other library (like nemo or torch) can.
import os
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    import site
    for site_pkg in site.getsitepackages():
        lib_path = os.path.join(site_pkg, "llama_cpp", "lib")
        if os.path.exists(lib_path):
            os.add_dll_directory(lib_path)

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError as e:
    # If this fails, the app cannot function with GGUF models.
    print(f"FATAL: Could not import llama_cpp. GGUF models will not be available. Error: {e}")
    LLAMA_CPP_AVAILABLE = False
# --- END: FORCE LLAMA_CPP IMPORT FIRST ---

import os
import logging
import asyncio
import glob
import threading
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import sys
import json
import re

import socket
import struct
import pickle
from fastapi import HTTPException
# --- GPU/SYSTEM UTILITY IMPORTS ---
# This block defines PYNVML_AVAILABLE for GPU detection without initializing torch.
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    PYNVML_AVAILABLE = False

# Set llama.cpp environment variables for performance
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["GGML_CUDA_DMMV_X"] = "32"
os.environ["GGML_CUDA_MMV_Y"] = "8"
os.environ["LLAMA_CUBLAS"] = "1"

# Secondary dependency check (ctransformers)
try:
    from ctransformers import AutoModelForCausalLM
    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    CTRANSFORMERS_AVAILABLE = False

# Function to get model directory from settings file
def get_model_directory():
    # Default location in user's home directory
    default_dir = str(Path.home() / "models" / "gguf")
    
    # Try to read from settings file
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    
    try:
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                if 'modelDirectory' in settings and os.path.isdir(settings['modelDirectory']):
                    return settings['modelDirectory']
    except Exception as e:
        logging.warning(f"Error reading model directory from settings: {e}")
    
    # If we can't read from settings or the directory doesn't exist,
    # try environment variable
    env_dir = os.environ.get("LiangLocal_MODEL_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    
    # Create default directory if it doesn't exist
    try:
        os.makedirs(Path.home() / "models" / "gguf", exist_ok=True)
    except:
        pass
    
    return default_dir

# Use the function to set MODEL_DIR
MODEL_DIR = get_model_directory()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DevstralHandler:
    """Handler for Devstral tool calling with proper template formatting"""
    
    @staticmethod
    def is_devstral_model(model_path: str) -> bool:
        """Check if the model is a Devstral model"""
        model_name = os.path.basename(model_path).lower()
        return "devstral" in model_name
    
    @staticmethod
    def format_tools_for_devstral(tools):
        """Format tools for Devstral's expected format"""
        if not tools:
            return None
        
        formatted_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    }
                })
        return formatted_tools

class GemmaVisionChatHandler(Llava15ChatHandler):
    """
    A custom chat handler that uses Llava's multimodal capabilities
    but formats the text prompt according to Gemma's specific chat template.
    """
    def render(self, messages: List[Dict[str, Any]]) -> str:
        prompt = ""
        system_message = ""
        if messages and messages[0].get("role") == "system":
            system_message = messages.pop(0).get("content", "") + "\n\n"

        for msg in messages:
            if msg.get("role") == "user":
                prompt += "<start_of_turn>user\n"
                content = msg.get("content", "")
                text_content = ""
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            text_content += part.get("text", "")
                        elif part.get("type") == "image_url":
                            # This placeholder is what the parent handler uses to inject vision tokens.
                            prompt += "<image>\n"
                
                prompt += (system_message + text_content).strip()
                prompt += "<end_of_turn>\n"
                system_message = "" # Only use system prompt once
            
            elif msg.get("role") == "assistant":
                prompt += f"<start_of_turn>model\n{msg.get('content', '')}<end_of_turn>\n"

        prompt += "<start_of_turn>model\n"
        logging.info(f"Final Rendered Prompt for Model:\n{prompt}")
        return prompt
class RemoteModelWrapper:
    """Optimized wrapper that forwards calls to the model service with connection pooling"""

    def __init__(self, model_name, gpu_id):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self._connection = None
        self._connection_lock = threading.Lock()
        # Spoof llama_cpp backend for compatibility
        self.__class__.__module__ = 'llama_cpp.llama'

    def _get_connection(self):
        """Get or create a persistent connection to the model service"""
        with self._connection_lock:
            if self._connection is None:
                try:
                    self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
                    self._connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    self._connection.settimeout(300)  # 5 minute timeout for model loading and long generations
                    self._connection.connect(('localhost', 5555))
                    logging.info(f"✅ [RemoteModelWrapper] Established persistent connection for {self.model_name}")
                except Exception as e:
                    logging.error(f"❌ [RemoteModelWrapper] Failed to create connection: {e}")
                    self._connection = None
                    raise
            return self._connection

    def _reset_connection(self):
        """Reset the connection if it's dead"""
        with self._connection_lock:
            if self._connection:
                try:
                    self._connection.close()
                except:
                    pass
                self._connection = None

    def _send_msg(self, sock, data):
        """Packs and sends a pickled message with error handling."""
        try:
            logging.info(f"🔄 [RemoteModelWrapper] Preparing to send message: {type(data)}")
            msg = pickle.dumps(data)
            msg_len = struct.pack('>I', len(msg))
            logging.info(f"🔄 [RemoteModelWrapper] Message serialized, size: {len(msg)} bytes")
            logging.info(f"🔄 [RemoteModelWrapper] Sending message to socket...")
            sock.sendall(msg_len + msg)
            logging.info(f"✅ [RemoteModelWrapper] Message sent successfully")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logging.error(f"❌ [RemoteModelWrapper] Socket error on send: {e}")
            raise

    def _recv_msg(self, sock):
        """Receives and unpacks a pickled message with error handling."""
        try:
            # Read the message length
            raw_msglen = sock.recv(4)
            if not raw_msglen:
                logging.warning(f"⚠️ [RemoteModelWrapper] No message length received, connection may be closed")
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]

            # Read the full message payload
            data = b''
            while len(data) < msglen:
                packet = sock.recv(msglen - len(data))
                if not packet:
                    logging.warning(f"⚠️ [RemoteModelWrapper] Connection broken while reading message payload")
                    return None # Connection broken
                data += packet
            
            result = pickle.loads(data)
            return result
        except (ConnectionResetError, struct.error, EOFError, OSError) as e:
            logging.error(f"❌ [RemoteModelWrapper] Socket error on receive: {e}")
            return None

    def _remote_call(self, action, **params):
        """Handles a single, non-streaming request-response with fresh connections."""
        conn = None
        try:
            logging.info(f"🔄 [RemoteModelWrapper] Starting remote call for action: {action}")
            # Create a fresh connection for each request since ModelService closes after each one
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.connect(('localhost', 5555))
            logging.info(f"✅ [RemoteModelWrapper] Fresh connection established for {action}")
            
            # Set appropriate timeout based on action type
            if action == 'load':
                conn.settimeout(600)  # 10 minutes for model loading
                logging.info(f"⏱️ [RemoteModelWrapper] Set 10-minute timeout for model loading")
            else:
                conn.settimeout(300)  # 5 minutes for other operations
                logging.info(f"⏱️ [RemoteModelWrapper] Set 5-minute timeout for {action}")
            
            request = {'action': action, 'params': params}
            logging.info(f"🔄 [RemoteModelWrapper] Request prepared: {request}")
            logging.info(f"🔄 [RemoteModelWrapper] About to send request...")
            self._send_msg(conn, request)
            logging.info(f"✅ [RemoteModelWrapper] Request sent, waiting for response...")
            response = self._recv_msg(conn)
            logging.info(f"✅ [RemoteModelWrapper] Response received: {type(response)}")
            return response
        except ConnectionRefusedError:
            logging.error("❌ [RemoteModelWrapper] Connection to model service at localhost:5555 was refused.")
            raise HTTPException(status_code=503, detail="Model service is unavailable.")
        except Exception as e:
            logging.error(f"❌ [RemoteModelWrapper] Error in remote call: {e}", exc_info=True)
            raise
        finally:
            # Always close the connection since ModelService closes after each request anyway
            if conn:
                try:
                    conn.close()
                    logging.info(f"✅ [RemoteModelWrapper] Connection closed for {action}")
                except Exception as e:
                    logging.warning(f"⚠️ [RemoteModelWrapper] Error closing connection: {e}")

    def _remote_stream(self, action, **params):
        """Yields responses from a streaming request with connection reuse."""
        conn = None
        try:
            conn = self._get_connection()
            request = {'action': action, 'params': params}
            self._send_msg(conn, request)
            
            while True:
                response = self._recv_msg(conn)
                # The service should send None or a specific sentinel to signal the end
                if response is None:
                    break
                if response and isinstance(response, dict) and "error" in response:
                    # Handle error response
                    yield response
                    break
                yield response
                    
        except ConnectionRefusedError:
            logging.error("❌ [RemoteModelWrapper] Connection to model service at localhost:5555 was refused.")
            yield {"error": "Model service is unavailable."}
        except Exception as e:
            logging.error(f"❌ [RemoteModelWrapper] Error in remote stream: {e}", exc_info=True)
            yield {"error": str(e)}
        finally:
            # Always reset connection after streaming
            if conn:
                self._reset_connection()
    def __call__(self, prompt=None, **kwargs):
        """Main entry point for model generation, dispatches to streaming or non-streaming call."""
        logging.info(f"🔄 [RemoteModelWrapper] __call__ invoked with prompt length: {len(prompt) if prompt else 0}")
        logging.info(f"🔄 [RemoteModelWrapper] Additional kwargs: {kwargs}")

        if prompt is not None:
            kwargs['prompt'] = prompt

        # Check for stream argument to decide which method to call
        if kwargs.get('stream'):
            logging.info(f"🔄 [RemoteModelWrapper] Calling _remote_stream for generate...")
            return self._remote_stream('generate', model_name=self.model_name, gpu_id=self.gpu_id, **kwargs)
        else:
            logging.info(f"🔄 [RemoteModelWrapper] Calling _remote_call for generate...")
            result = self._remote_call('generate', model_name=self.model_name, gpu_id=self.gpu_id, **kwargs)
            logging.info(f"✅ [RemoteModelWrapper] _remote_call completed, result type: {type(result)}")
            return result

    def create_completion(self, prompt=None, **kwargs):
        """llama-cpp compatibility alias. Forwards to __call__."""
        return self.__call__(prompt=prompt, **kwargs)
    
    def unload(self):
        """Tell the service to unload this model"""
        try:
            result = self._remote_call('unload', model_name=self.model_name, gpu_id=self.gpu_id)
            self._reset_connection()  # Close connection after unload
            return result
        except Exception as e:
            logging.error(f"❌ [RemoteModelWrapper] Error during unload: {e}")
            self._reset_connection()
            return {"error": str(e)}
    
    def embed(self, text: str):
        """Tell the service to generate embeddings for the given text."""
        return self._remote_call('embed', model_name=self.model_name, gpu_id=self.gpu_id, text=text)
    
    def __del__(self):
        """Cleanup when the wrapper is destroyed"""
        self._reset_connection()

class ModelManager:
    def __init__(self, gpu_usage_mode="split_services"):
        self.loaded_models: Dict[Tuple[str, int], Dict[str, Any]] = {}  # Stores model objects with composite key (model_name, gpu_id)
        self.lock = asyncio.Lock()
        #  NEW: Track which models serve which testing purposes
        self.model_purposes = {
            'test_model': None,           # {'name': 'model_name', 'gpu_id': 0}
            'primary_judge_model': None,  # {'name': 'model_name', 'gpu_id': 1}
            'secondary_judge_model': None, # {'name': 'model_name', 'gpu_id': 0}
            'test_model_a': None,         # {'name': 'model_name', 'gpu_id': 0}
            'test_model_b': None,         # {'name': 'model_name', 'gpu_id': 0}
            'forensic_embeddings': None,   # {'name': 'model_name', 'gpu_id': 0}
            'automation_interpreter': None # {'name': 'model_name', 'gpu_id': 1}
        }
        self.models_dir = Path(MODEL_DIR)
        self.has_gpu = self._detect_gpu()
        self.gpu_info = self._get_gpu_info()
        self.gpu_usage_mode = gpu_usage_mode  # Store the GPU usage mode
        
        # Set global environment variables
        os.environ["GGML_VERBOSE"] = "1"
        
        # Get GPU ID from environment if available (for multi-process mode)
        self.default_gpu_id = int(os.environ.get("GPU_ID", 0))
        logging.info(f"Default GPU ID set to: {self.default_gpu_id}")
        logging.info(f"GPU usage mode: {self.gpu_usage_mode}")
        
        
        # Set global environment variables
        os.environ["GGML_VERBOSE"] = "1"
        
        # Get GPU ID from environment if available (for multi-process mode)
        self.default_gpu_id = int(os.environ.get("GPU_ID", 0))
        logging.info(f"Default GPU ID set to: {self.default_gpu_id}")
        
    def _detect_gpu(self) -> bool:
        if PYNVML_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                logging.info(f"pynvml detected {gpu_count} GPUs.")
                return gpu_count > 0
            except pynvml.NVMLError as e:
                logging.warning(f"pynvml error detecting GPU: {e}")
                return False
        else:
            logging.warning("pynvml not available. Cannot reliably detect GPU without initializing PyTorch.")
            # Fallback to a less ideal check if needed, but pynvml is preferred
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False

            
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Gather information about available GPUs using pynvml."""
        if not PYNVML_AVAILABLE:
            logging.warning("pynvml not available, cannot get detailed GPU info.")
            return {"count": 0, "names": [], "cuda_version": None, "memory": []}

        gpu_info = {
            "count": 0,
            "names": [],
            "cuda_version": None,
            "memory": []
        }
        try:
            gpu_info["count"] = pynvml.nvmlDeviceGetCount()
            # Get CUDA version from the driver
            try:
                gpu_info["cuda_version"] = pynvml.nvmlSystemGetDriverVersion()
            except pynvml.NVMLError:
                gpu_info["cuda_version"] = "N/A"

            for i in range(gpu_info["count"]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_info["names"].append(pynvml.nvmlDeviceGetName(handle))
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info["memory"].append({
                    "free_mb": mem_info.free / (1024 * 1024),
                    "total_mb": mem_info.total / (1024 * 1024)
                })

            logging.info(f"Found {gpu_info['count']} GPUs via pynvml: {', '.join(gpu_info['names'])}")
        except pynvml.NVMLError as e:
            logging.error(f"Error getting GPU info from pynvml: {e}")
            # Reset to safe defaults on error
            return {"count": 0, "names": [], "cuda_version": None, "memory": []}

        return gpu_info

    def _load_tensor_split_settings(self) -> Optional[list]:
        """Load tensor split settings from settings.json - supports any number of GPUs"""
        try:
            settings_path = Path.home() / ".LiangLocal" / "settings.json"
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    tensor_split = settings.get('tensor_split')
                    if tensor_split and isinstance(tensor_split, list):
                        # Support any number of GPUs (2, 3, 4, etc.)
                        if len(tensor_split) >= 2:
                            # Validate that values sum to approximately 1.0
                            total = sum(tensor_split)
                            if abs(total - 1.0) < 0.01:
                                logging.info(f"✅ Loaded tensor_split from settings: {tensor_split} ({len(tensor_split)} GPUs)")
                                return tensor_split
                            else:
                                logging.warning(f"⚠️ Invalid tensor_split in settings (sum={total}), using default")
                        else:
                            logging.warning(f"⚠️ tensor_split must have at least 2 values, got {len(tensor_split)}")
        except Exception as e:
            logging.error(f"❌ Error loading tensor_split settings: {e}")
        return None

    def _get_gpu_params(self, gpu_id: int, context_length: int = 4096) -> Dict[str, Any]:
        """
        Return optimal parameters for GPU utilization, including tensor splitting.
        """
        logging.info(f"🔍 _get_gpu_params called with gpu_id={gpu_id}, context_length={context_length}")

        params = {
            "n_ctx": context_length,
            "n_batch": 8192,  # Increased from default for better GPU utilization
            "n_threads": 32,   # Increased for better CPU-GPU coordination
                                    "verbose": True,   # Restore verbose logging for debugging
            "seed": 42,
            "n_gpu_layers": -1, # Set to -1 to offload all possible layers
            "main_gpu": gpu_id,  # Set main_gpu initially
            "offload_kqv": True,
            "f16_kv": True,
            "use_mmap": True,
            "use_mlock": True,
            "low_vram": False,
            "rope_scaling": {"type": "yarn", "factor": 1.0},
            "use_cache": True,
            "logits_all": False,
            "embedding": False,
            "vocab_only": False,
            "flash_attn": True,
            "mul_mat_q": True,  # Enable matrix multiplication optimizations
            "fused_mlp": True,  # Enable fused MLP operations
            "fused_attention": True,  # Enable fused attention
        }

        effective_gpu_count = self.gpu_info.get("count", 0)

        # --- REVISED MULTI-GPU LOGIC WITH OPTIMIZED SPLIT ---
        if effective_gpu_count > 1 and self.gpu_usage_mode == "unified_model":
            logging.info(f"⭐ [Unified Mode] Configuring for {effective_gpu_count} GPUs with optimized tensor splitting.")
            
            # --- START: PERFORMANCE OPTIMIZATION ---
            # Load tensor split from settings or use default
            tensor_split = self._load_tensor_split_settings()
            
            if tensor_split is None:
                # Default: Equal split across all available GPUs
                # Supports 2, 3, 4, or more GPUs
                tensor_split = [1.0 / effective_gpu_count] * effective_gpu_count
                logging.info(f"⭐ Using default equal tensor_split across {effective_gpu_count} GPUs: {tensor_split}")
            elif len(tensor_split) != effective_gpu_count:
                # If user provided tensor_split but GPU count changed, warn and use equal split
                logging.warning(f"⚠️ tensor_split length ({len(tensor_split)}) doesn't match GPU count ({effective_gpu_count}). Using equal split.")
                tensor_split = [1.0 / effective_gpu_count] * effective_gpu_count
            
            params["tensor_split"] = tensor_split
            
            # Validate the tensor split values
            total_split = sum(tensor_split)
            if abs(total_split - 1.0) > 0.01:
                logging.warning(f"⚠️ [ModelManager] Tensor split values don't sum to 1.0: {tensor_split} = {total_split}")
                # Normalize to ensure they sum to 1.0
                normalized_split = [val / total_split for val in tensor_split]
                params["tensor_split"] = normalized_split
                logging.info(f"✅ [ModelManager] Normalized tensor_split to {normalized_split}")
                tensor_split = normalized_split
            else:
                logging.info(f"✅ [ModelManager] Tensor split validation passed: {tensor_split}")
            
            logging.info(f"⭐ [Unified Mode] Set tensor_split to {params['tensor_split']}")
            
            # Use row split mode for better multi-GPU performance
            params["split_mode"] = 2  # Row split mode for better performance
            logging.info(f"⭐ [Unified Mode] Enabled split_mode=2 for optimal layer distribution")
            
            # KV cache will be distributed based on tensor_split ratios
            logging.info(f"⭐ [Unified Mode] KV cache will be distributed according to tensor_split: {tensor_split}")
            # Find GPU with largest allocation
            max_gpu_idx = tensor_split.index(max(tensor_split))
            max_allocation = tensor_split[max_gpu_idx] * 100
            logging.info(f"✅ [Unified Mode] GPU {max_gpu_idx} ({max_allocation:.0f}%) has the largest allocation")
            
            # CRITICAL: Remove 'main_gpu' when using tensor_split for multi-GPU
            if 'main_gpu' in params:
                del params['main_gpu']
            
            # Add multi-GPU specific optimizations
            params.update({
                "n_batch": 16384,  # Larger batch size for multi-GPU
                "n_threads": 48,    # More threads for multi-GPU coordination
                "use_mmap": True,   # Memory mapping for faster loading
                "use_mlock": True,  # Lock memory to prevent swapping
                "low_vram": False,  # Disable low VRAM mode for better performance
                "flash_attn": True, # Enable flash attention if available
                "rope_scaling": {"type": "yarn", "factor": 1.0},
                "use_cache": True,  # Enable KV cache for faster generation
                "mul_mat_q": True,  # Enable matrix multiplication optimizations
                "fused_mlp": True,  # Enable fused MLP operations
                "fused_attention": True,  # Enable fused attention
            })
            
            logging.info(f"⭐ [Unified Mode] Applied multi-GPU performance optimizations.")

        else: # Covers single GPU and 'split_services' mode
            # For these cases, we assign the model to a specific GPU.
            params["main_gpu"] = gpu_id
            logging.info(f"⭐ [Single GPU / Split Mode] Assigning model exclusively to main_gpu: {gpu_id}")
        # --- END REVISED LOGIC ---

        # Set environment variables for optimal CUDA performance
        os.environ["GGML_CUDA_NO_PINNED"] = "0"
        os.environ["GGML_CUDA_DMMV_X"] = "32"
        os.environ["GGML_CUDA_MMV_Y"] = "8"
        os.environ["LLAMA_CUBLAS"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        
        return params

    def _ensure_model_purposes(self):
        """Ensure model_purposes dictionary exists (for backward compatibility)"""
        if not hasattr(self, 'model_purposes'):
            self.model_purposes = {
                'test_model': None,
                'primary_judge': None,
                'secondary_judge': None,
                'test_model_a': None,
                'test_model_b': None,
                'forensic_embeddings': None,
                'automation_interpreter': None
            }

    async def _load_with_llama_cpp(
        self,
        model_path: str,
        gpu_id: Optional[int] = None,
        n_ctx: Optional[int] = 4096,
        purpose: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a model using llama.cpp with Devstral-aware configuration.

        - In 'split_services' mode, load the model directly in this process,
          leveraging the existing CUDA_VISIBLE_DEVICES isolation.
        - In 'unified_model' mode, delegate to the remote model_service,
          which can see all GPUs for tensor splitting.

        Notes:
          - Detects Devstral/Unsloth GGUFs and enables llama.cpp's template auto-detection.
          - Propagates `is_devstral` and `purpose` to remote service for correct routing.
          - Returns a Llama instance (split mode) or a RemoteModelWrapper (unified mode).
        """
        try:
            # ---- Common params ---------------------------------------------------
            model_params = self._get_gpu_params(gpu_id or 0, context_length=n_ctx)
            mmproj_path = self._find_matching_mmproj(model_path)
            if mmproj_path:
                model_params["clip_model_path"] = str(mmproj_path)

            # Devstral detection (safe if DevstralHandler is a no-op for non-Devstral)
            try:
                is_devstral = DevstralHandler.is_devstral_model(model_path)
            except NameError:
                # If DevstralHandler isn't imported/available, default to False
                is_devstral = False

            if is_devstral:
                logging.info(f"🔧 Detected Devstral model: {os.path.basename(model_path)}")
                logging.info("🔧 Enabling llama.cpp template auto-detection (verbose=True)")
                # Let llama.cpp auto-detect the chat template from GGUF metadata
                model_params["verbose"] = True

            # Allow caller-supplied overrides via **kwargs (last write wins)
            if kwargs:
                logging.info(f"🔍 Applying kwargs overrides: {list(kwargs.keys())}")
                if 'embedding' in kwargs:
                    logging.info(f"🔍 CRITICAL: embedding parameter in kwargs = {kwargs['embedding']}")
                model_params.update(kwargs)
                if 'embedding' in model_params:
                    logging.info(f"🔍 After update: embedding parameter = {model_params['embedding']}")

            # ---- Split services: load locally -----------------------------------
            if self.gpu_usage_mode == "split_services":
                logging.info(f"✅ [Split Mode] Loading model DIRECTLY on GPU {gpu_id}.")
                
                # Log embedding parameter status
                if model_params.get('embedding'):
                    logging.info(f"🔍 Loading as EMBEDDING model (embedding=True)")
                else:
                    logging.info(f"🔍 Loading as TEXT GENERATION model (embedding=False/not set)")

                model = Llama(
                    model_path=model_path,
                    **model_params
                )

                # Tag instance for downstream logic
                try:
                    setattr(model, "_is_devstral", bool(is_devstral))
                    if purpose is not None:
                        setattr(model, "_purpose", purpose)
                except Exception:
                    # Non-fatal if Llama object disallows dynamic attributes
                    pass

                logging.info(f"✅ Model {os.path.basename(model_path)} loaded directly.")
                if is_devstral:
                    logging.info("🔧 Devstral tool-calling path is enabled (template auto-detect).")
                return model  # actual Llama instance

            # ---- Unified model: delegate to remote service -----------------------
            else:
                logging.info(f"🚀 [Unified Mode] Delegating model load to remote service for GPU {gpu_id}.")
                logging.info(f"🔍 [ModelManager] GPU usage mode: {self.gpu_usage_mode}")
                logging.info(f"🔍 [ModelManager] Effective GPU count: {self.gpu_info.get('count', 0)}")
                logging.info(f"🔍 [ModelManager] Model parameters being sent: {list(model_params.keys())}")
                if 'tensor_split' in model_params:
                    logging.info(f"🔍 [ModelManager] Tensor split: {model_params['tensor_split']}")

                wrapper = RemoteModelWrapper(os.path.basename(model_path), gpu_id)

                logging.info(
                    f"🚀 [ModelManager] Dispatching 'load' to ModelService for "
                    f"{os.path.basename(model_path)} on GPU {gpu_id}."
                )

                max_retries = 3
                retry_delay = 5

                model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
                if model_size_mb > 1000:
                    logging.info(
                        f"📊 [ModelManager] Loading large model: "
                        f"{os.path.basename(model_path)} ({model_size_mb:.1f} MB)"
                    )
                    logging.info("⏱️ [ModelManager] 70B+ models can take several minutes...")

                result = None
                for attempt in range(max_retries):
                    try:
                        logging.info(f"🔄 [ModelManager] Loading attempt {attempt + 1}/{max_retries}")
                        if model_size_mb > 1000:
                            logging.info("🔄 [ModelManager] Large model loading in progress...")
                            logging.info("⏱️ [ModelManager] Expect 5–10 minutes for some 70B models...")

                        # Validate model file before sending to service
                        if not os.path.exists(model_path):
                            raise FileNotFoundError(f"Model file not found: {model_path}")
                        
                        logging.info(f"🔍 [ModelManager] Model file validation passed: {model_path}")
                        logging.info(f"🔍 [ModelManager] Model file size: {model_size_mb:.1f} MB")

                        logging.info("🔄 [ModelManager] Calling ModelService with 10-minute timeout...")

                        # Include devstral/purpose signals so the service can configure tool-calling paths.
                        result = wrapper._remote_call(
                            'load',
                            model_name=os.path.basename(model_path),
                            model_path=str(model_path),
                            gpu_id=gpu_id,
                            context_length=n_ctx,
                            params=model_params,
                            gpu_usage_mode=self.gpu_usage_mode,
                            is_devstral=bool(is_devstral),
                            purpose=purpose,
                        )

                        logging.info(f"👍 [ModelManager] Response from ModelService: {result}")

                        if result and isinstance(result, dict) and "error" in result:
                            raise Exception(result["error"])

                        logging.info("✅ [ModelManager] Model loading completed successfully!")
                        break

                    except Exception as e:
                        if attempt < max_retries - 1:
                            logging.warning(
                                f"⚠️ [ModelManager] Attempt {attempt + 1} failed: {e}. "
                                f"Retrying in {retry_delay}s..."
                            )
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            wrapper._reset_connection()
                        else:
                            logging.error(
                                f"❌ [ModelManager] All loading attempts failed for "
                                f"{os.path.basename(model_path)}"
                            )
                            raise

                logging.info(f"✅ Model loaded via service: {result}")

                # Tag the wrapper so callers can branch behaviour without another lookup
                try:
                    setattr(wrapper, "_is_devstral", bool(is_devstral))
                    if purpose is not None:
                        setattr(wrapper, "_purpose", purpose)
                except Exception:
                    pass

                return wrapper

        except Exception as e:
            logging.exception(f"❌ Error in _load_with_llama_cpp: {e}")
            raise
    def _find_matching_mmproj(self, model_path: str) -> Optional[Path]:
        """
        Finds a matching mmproj file for a given model by parsing the model size
        (e.g., 4b, 12b, 27b) from the filenames, enforcing a clear naming convention.
        """
        model_dir = Path(model_path).parent
        model_name = Path(model_path).stem.lower()
        logging.info(f"🔍 Searching for mmproj to match model: {model_name}")

        # 1. Extract size (e.g., '4b', '27b') from the main model's filename.
        # This regex looks for a pattern like "-4b-" or "-27b-".
        model_size_match = re.search(r'-(\d+b)-', model_name)
        if not model_size_match:
            logging.warning(f"Could not determine model size from filename: {model_name}. Vision support will be disabled.")
            return None
        
        model_size = model_size_match.group(1)  # This will be a string like "4b" or "27b"
        logging.info(f"🔍 Determined model size to be: '{model_size}'")

        # 2. Find all potential mmproj files in the directory.
        mmproj_files = list(model_dir.glob("mmproj-*.gguf"))
        if not mmproj_files:
            logging.info("🔍 No mmproj files found in directory.")
            return None

        logging.info(f"🔍 Found potential mmproj files: {[f.name for f in mmproj_files]}")

        # 3. Iterate through the found mmproj files and find the one with the matching size.
        for mmproj_file in mmproj_files:
            mmproj_name = mmproj_file.name.lower()
            
            # Extract the size from the mmproj filename using the same pattern.
            mmproj_size_match = re.search(r'-(\d+b)-', mmproj_name)
            
            if mmproj_size_match:
                mmproj_size = mmproj_size_match.group(1)
                logging.info(f"🔍 Checking '{mmproj_name}' (size: {mmproj_size}) against model size '{model_size}'")
                
                # We have a match if the model name contains "gemma" and the sizes are identical.
                if "gemma" in model_name and "gemma" in mmproj_name and mmproj_size == model_size:
                    logging.info(f"🎯🎯🎯 Found exact size match for '{model_size}': {mmproj_file.name}")
                    return mmproj_file
        
        logging.error(f"❌ CRITICAL: Could not find a matching mmproj file for model size '{model_size}'. Make sure the correctly named file is in the model directory.")
        return None
    
    def _load_with_ctransformers(self, model_path: str, gpu_id: Optional[int] = None, n_ctx: Optional[int] = 4096, **kwargs):
        """Load a model using the ctransformers library"""
        try:
            logging.info(f"Loading with ctransformers: {model_path} on GPU {gpu_id}")
            
            # For ctransformers, handle GPU selection
            if gpu_id is not None and gpu_id >= 0:
                kwargs["device"] = f"cuda:{gpu_id}"
                
            # Ensure context length is passed correctly
            kwargs["context_length"] = n_ctx
                
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            logging.info(f"Successfully loaded model with ctransformers: {model_path} on GPU {gpu_id}")
            return model
        except Exception as e:
            logging.exception(f"Error loading model with ctransformers: {e}")
            raise

    async def load_model(
        self,
        model_name: str,
        gpu_id: Optional[int] = None,
        model_path: Optional[str] = None,
        context_length: Optional[int] = 4096,
        n_ctx: Optional[int] = None,  # Added for compatibility with existing code
        **kwargs
    ) -> None:
        print(f"DEBUG: Entered load_model for {model_name}", flush=True)
        logging.info(f"[DIAGNOSTIC] load_model called for: model='{model_name}', gpu_id={gpu_id}, context_length={context_length}, model_path={model_path}. Current loaded: {list(self.loaded_models.keys())}")
        """
        Load a model onto a specific GPU

        Args:
            model_name: Name of the model file
            gpu_id: GPU ID to load onto (0, 1, etc.), or None to use default
            model_path: Optional explicit path to model file
            context_length: The context length to use for this model (n_ctx)
            n_ctx: Alternative way to specify context length (for compatibility)
            **kwargs: Additional parameters to pass to the model loader
        """
        print(f"DEBUG: About to acquire lock", flush=True)
        async with self.lock:
            print(f"DEBUG: Lock acquired", flush=True)
            # Use n_ctx if provided, otherwise fallback to context_length
            if n_ctx is not None:
                context_length = n_ctx
                print(f"DEBUG: Using n_ctx={n_ctx} for context length", flush=True)

            # Try to extract from model name if context is missing or too low
            if context_length is None or context_length < 4096:
            # Smart filename inspection to guess context window
                if model_name:
                    lowered_name = str(model_name).lower()
                if "32k" in lowered_name:
                    context_length = 32768
                elif "16k" in lowered_name:
                    context_length = 16384
                elif "8k" in lowered_name:
                    context_length = 8192
                elif "4k" in lowered_name:
                    context_length = 4096
                elif "2k" in lowered_name:
                    context_length = 2048
                else:
                    context_length = 8192  # Safe fallback
                logging.warning(f"⚠️ Inferred context length {context_length} from model name '{model_name}'")

            # Final confirmation
            logging.info(f"⭐⭐⭐ FINAL CONTEXT LENGTH: {context_length}")
            
            # Use the default GPU ID if none specified
            requested_gpu_id = self.default_gpu_id if gpu_id is None else gpu_id
            
            # CRITICAL: When CUDA_VISIBLE_DEVICES is set, the backend only sees ONE GPU as device 0
            # So we need to use 0 for the actual model loading, but keep the original GPU ID for tracking
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices:
                # Backend is restricted to one GPU - always use device 0 for loading
                target_gpu_id = 0
                logging.info(f"🔒 CUDA_VISIBLE_DEVICES={cuda_visible_devices} detected. Using device 0 internally for loading (requested GPU {requested_gpu_id})")
            else:
                # Backend can see all GPUs - use the requested GPU ID directly
                target_gpu_id = requested_gpu_id
                logging.info(f"🔓 No CUDA_VISIBLE_DEVICES restriction. Using GPU {target_gpu_id} directly")

            # Create a composite key for this model (model_name, requested_gpu_id) to track original assignment
            model_key = (model_name, requested_gpu_id)

            if model_key in self.loaded_models:
                model_info = self.loaded_models[model_key]
                current_ctx = model_info.get("n_ctx")
                
                if current_ctx == context_length:
                    logging.info(f"Model {model_name} already loaded on GPU {target_gpu_id} with context length {context_length}.")
                    return
                else:
                    # Model is loaded but with a different context length, unload it first
                    logging.info(f"Model {model_name} already loaded on GPU {target_gpu_id} with context length {current_ctx}, unloading first.")
                    await self.unload_model(model_name, target_gpu_id)

            # If model_path not provided, search for it
            if not model_path:
                print(f"DEBUG: Searching for model file containing '{model_name}' within '{self.models_dir}'", flush=True)

                # Use a clean, simple search for any .gguf file recursively
                all_gguf_files = glob.glob(os.path.join(self.models_dir, "**", "*.gguf"), recursive=True)
                
                # --- NEW: Intelligent Search Logic ---
                # Clean the model name and split into keywords
                cleaned_name = model_name.lower().replace('/', '-').replace('_', '-')
                keywords = [word for word in cleaned_name.split('-') if word]

                logging.info(f"🔍 Searching for file with keywords: {keywords}")

                best_match = None
                highest_score = 0
                
                for file_path in all_gguf_files:
                    file_name_lower = os.path.basename(file_path).lower()
                    
                    # Score based on number of matching keywords
                    score = sum(1 for keyword in keywords if keyword in file_name_lower)
                    
                    # Boost score for the most important part of the name
                    core_model_name = keywords[-1]
                    if core_model_name in file_name_lower:
                        score += 5 # Add a significant boost

                    if score >= 2 and score > highest_score:
                        highest_score = score
                        best_match = file_path

                if not best_match:
                    # The error message now shows the cleaned keywords for easier debugging
                    raise FileNotFoundError(f"Could not find a GGUF file matching keywords: {keywords} in '{self.models_dir}'")

                model_path = best_match
            print(f"DEBUG: Found best model match: {model_path}", flush=True)

            logging.info(f"⭐ Loading model {model_name} from {model_path} on GPU {target_gpu_id} with context length {context_length}")

            try:
                # DEBUG: Check if embedding parameter was passed
                logging.info(f"🔍 [load_model] Received kwargs keys: {list(kwargs.keys())}")
                if 'embedding' in kwargs:
                    logging.info(f"🔍 [load_model] EMBEDDING PARAMETER DETECTED: embedding={kwargs['embedding']}")
                else:
                    logging.warning(f"🔍 [load_model] NO EMBEDDING PARAMETER in kwargs!")

                # Always force n_gpu_layers to a high value to ensure all layers go to GPU
                kwargs["n_gpu_layers"] = 999
                logging.info(f"⭐ Forcing ALL layers to GPU {target_gpu_id} with n_gpu_layers=999")

                if LLAMA_CPP_AVAILABLE:
                    # Create a copy of kwargs WITHOUT n_ctx to avoid duplicate parameter error
                    safe_kwargs = {k: v for k, v in kwargs.items() if k != 'n_ctx'}
                    model = await self._load_with_llama_cpp(model_path=str(model_path), gpu_id=target_gpu_id, n_ctx=context_length, **safe_kwargs)
                elif CTRANSFORMERS_AVAILABLE:
                    safe_kwargs = {k: v for k, v in kwargs.items() if k != 'n_ctx'}
                    model = self._load_with_ctransformers(str(model_path), target_gpu_id, n_ctx=context_length, **safe_kwargs)
                else:
                    raise ImportError("No compatible GGUF model loader found.")

                # Store the model with metadata using the composite key
                self.loaded_models[model_key] = {
                    "model": model,
                    "path": str(model_path),
                    "n_ctx": context_length  # Store n_ctx
                    # No need to store gpu_id here as it's part of the key
                }

                logging.info(f"✅ Model {model_name} loaded successfully on GPU {target_gpu_id}.")

                # Run a quick inference test to verify model works and measure speed
                # Skip text generation test for embedding models
                is_embedding_model = kwargs.get('embedding', False)
                try:
                    if LLAMA_CPP_AVAILABLE and not is_embedding_model:
                        import time
                        test_prompt = "Hello, this is a quick test."
                        logging.info(f"Running test inference with prompt: '{test_prompt}'")

                        start_time = time.time()
                        test_output = model(test_prompt, max_tokens=20)
                        end_time = time.time()

                        inference_time = end_time - start_time
                        
                        # Check if test_output is valid before accessing it
                        if test_output and isinstance(test_output, dict) and "choices" in test_output and test_output["choices"]:
                            output_text = test_output["choices"][0]["text"]
                            token_count = len(output_text.split())
                            tokens_per_second = token_count / inference_time if inference_time > 0 else 0

                            logging.info(f"⭐ Test inference successful: {output_text}")
                            logging.info(f"⭐ Generated {token_count} tokens in {inference_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
                        else:
                            logging.warning(f"Test inference returned invalid output: {test_output}")
                            logging.warning("This may indicate a connection or model issue")
                    elif is_embedding_model:
                        logging.info(f"✅ Embedding model loaded - skipping text generation test")
                except Exception as test_e:
                    logging.warning(f"Test inference failed: {test_e}")
                    logging.warning("This may indicate a connection or model issue")

            except Exception as e:
                logging.exception(f"❌ Error loading model {model_name} on GPU {target_gpu_id}: {e}")
                raise

    async def unload_model(self, model_name: str, gpu_id: int):
        logging.info(f"[DIAGNOSTIC] unload_model called for: model='{model_name}', gpu_id={gpu_id}. Current loaded: {list(self.loaded_models.keys())}")
        """Unload a specific model from a specific GPU"""
        async with self.lock:
            model_key = (model_name, gpu_id)
            
            if model_key in self.loaded_models:
                model_data = self.loaded_models.pop(model_key)

                # Tell the service to unload!
                if isinstance(model_data["model"], RemoteModelWrapper):
                    result = model_data["model"].unload()
                    logging.info(f"✅ [ModelManager] RemoteModelWrapper unload result: {result}")
                elif hasattr(model_data["model"], "shutdown"):
                    model_data["model"].shutdown()
                    logging.info(f"✅ [ModelManager] Subprocess for {model_name} shut down cleanly")
                else:
                    logging.info(f"✅ [ModelManager] Model {model_name} removed from memory (no special cleanup needed)")

                # Clear any purpose assignments for this model
                for purpose, purpose_info in self.model_purposes.items():
                    if (purpose_info and 
                        purpose_info['name'] == model_name and 
                        purpose_info['gpu_id'] == gpu_id):
                        self.model_purposes[purpose] = None
                        logging.info(f"✅ Cleared {purpose} assignment for {model_name}")
                
                logging.info(f"Model {model_name} unloaded from GPU {gpu_id}.")
                import gc
                gc.collect()
                
                # Try to clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logging.info(f"CUDA cache cleared for GPU {gpu_id}")
                except:
                    pass
            else:
                logging.warning(f"Model {model_name} on GPU {gpu_id} not found, nothing to unload.")

    async def unload_all_models(self):
        """Unload all models from all GPUs"""
        async with self.lock:
            keys_to_unload = list(self.loaded_models.keys())
            for model_key in keys_to_unload:
                model_name, gpu_id = model_key  # Unpack the composite key
                await self.unload_model(model_name, gpu_id)
            logging.info("All models unloaded")

    async def load_model_for_purpose(self, purpose: str, model_name: str, gpu_id: int, context_length: int = 4096):
        """Load a model for a specific testing purpose"""
        valid_purposes = ['test_model', 'primary_judge', 'secondary_judge', 'test_model_a', 'test_model_b', 'forensic_embeddings', 'automation_interpreter']
        if purpose not in valid_purposes:
            raise ValueError(f"Invalid purpose. Must be one of: {valid_purposes}")
        
        # Initialize model_purposes if it doesn't exist
        if not hasattr(self, 'model_purposes'):
            self.model_purposes = {
                'test_model_a': None,
                'test_model_b': None,
                'primary_judge': None,
                'secondary_judge': None,
                'test_model': None,
                'forensic_embeddings': None,
            }
        
        # Unload any existing model for this purpose first
        if self.model_purposes.get(purpose):
            old_model = self.model_purposes[purpose]
            logging.info(f"Unloading previous {purpose}: {old_model['name']} from GPU {old_model['gpu_id']}")
            await self.unload_model(old_model['name'], old_model['gpu_id'])
        
        # Load the new model (this will handle its own locking)
        # For embedding models, pass embedding=True to llama-cpp-python
        if purpose == 'forensic_embeddings':
            logging.info(f"🔍 Loading embedding model {model_name} with embedding=True")
            await self.load_model(model_name, gpu_id=gpu_id, context_length=context_length, embedding=True)
        else:
            await self.load_model(model_name, gpu_id=gpu_id, context_length=context_length)
        
        # Track the purpose assignment
        self.model_purposes[purpose] = {'name': model_name, 'gpu_id': gpu_id}
        logging.info(f"✅ Assigned {model_name} (GPU {gpu_id}) as {purpose}")

    async def unload_model_purpose(self, purpose: str):
        """Unload the model serving a specific purpose"""
        if self.model_purposes.get(purpose):
            model_info = self.model_purposes[purpose]
            await self.unload_model(model_info['name'], model_info['gpu_id'])
            self.model_purposes[purpose] = None
            logging.info(f"✅ Unloaded {purpose}")

    def get_models_by_purpose(self):
        """Get currently loaded models organized by their testing purpose"""
        # Initialize model_purposes if it doesn't exist
        if not hasattr(self, 'model_purposes'):
            self.model_purposes = {
                'test_model_a': None,
                'test_model_b': None,
                'primary_judge': None,
                'secondary_judge': None,
                'test_model': None,
                'forensic_embeddings': None,
                'automation_interpreter': None,
            }
        
        result = {}
        for purpose, model_info in self.model_purposes.items():
            if model_info:
                # Get additional details from loaded_models
                model_key = (model_info['name'], model_info['gpu_id'])
                model_data = self.loaded_models.get(model_key, {})
                
                result[purpose] = {
                    'name': model_info['name'],
                    'gpu_id': model_info['gpu_id'],
                    'context_length': model_data.get('n_ctx', 4096),
                    'is_loaded': model_key in self.loaded_models
                }
            else:
                result[purpose] = None
        
        return result

    async def update_context_length(self, model_name: str, new_context_length: int, gpu_id: Optional[int] = None):
        """
        Update the context length for a loaded model. This will require reloading the model.

        Args:
            model_name: The name of the loaded model.
            new_context_length: The new context length to set (n_ctx).
            gpu_id: The GPU ID where the model is loaded.
        """
        async with self.lock:
            # Use default GPU if not specified
            target_gpu_id = self.default_gpu_id if gpu_id is None else gpu_id
            model_key = (model_name, target_gpu_id)
            
            if model_key not in self.loaded_models:
                logging.warning(f"Model {model_name} on GPU {target_gpu_id} is not currently loaded.")
                return

            model_info = self.loaded_models[model_key]
            current_path = model_info["path"]

            logging.info(f"Updating context length for {model_name} on GPU {target_gpu_id} to {new_context_length}. Reloading model...")
            await self.unload_model(model_name, target_gpu_id)
            await self.load_model(model_name, gpu_id=target_gpu_id, model_path=current_path, context_length=new_context_length)
            logging.info(f"Context length for {model_name} on GPU {target_gpu_id} updated to {new_context_length}.")

    def list_available_models(self):
        """List all available GGUF models in the models directory"""
        try:
            model_files = [filename for filename in os.listdir(MODEL_DIR) if filename.endswith(".gguf")]
            return {"available_models": model_files}
        except Exception as e:
            logging.error(f"Error listing available models: {e}")
            return {"available_models": [], "error": str(e)}

    def get_loaded_models(self):
        """Return information about loaded models including which GPU they're on"""
        try:
            models_info = []
            for model_key, model_data in self.loaded_models.items():
                model_name, gpu_id = model_key  # Unpack the composite key
                context_length = model_data.get("n_ctx", 4096)
                gpu_name = "CPU"
                if gpu_id >= 0 and gpu_id < len(self.gpu_info["names"]):
                    gpu_name = self.gpu_info["names"][gpu_id]
                    
                models_info.append({
                    "name": model_name,
                    "gpu_id": gpu_id,
                    "gpu_name": gpu_name,
                    "context_length": context_length
                })
                
            return {"loaded_models": models_info}
        except Exception as e:
            logging.error(f"Error getting loaded models: {e}")
            return {"loaded_models": [], "error": str(e)}
    
    def get_loaded_models_on_gpu(self, gpu_id: int):
        """Return models loaded on a specific GPU"""
        try:
            models_on_gpu = {}
            for model_key, model_data in self.loaded_models.items():
                if isinstance(model_key, tuple) and len(model_key) == 2:
                    model_name, loaded_gpu_id = model_key
                    if loaded_gpu_id == gpu_id:
                        models_on_gpu[model_name] = model_data
            return models_on_gpu
        except Exception as e:
            logging.error(f"Error getting models on GPU {gpu_id}: {e}")
            return {}
        
    def get_model(self, model_name: str, gpu_id: int):
        """Get the actual model object for inference"""
        model_key = (model_name, gpu_id)
        
        model_data = self.loaded_models.get(model_key)
        if not model_data:
            raise ValueError(f"Model {model_name} on GPU {gpu_id} is not loaded")
            
        return model_data["model"]
        
    def get_system_info(self):
        """Returns system information including GPU details"""
        info = {
            "gpu_available": self.has_gpu,
            "gpu_count": self.gpu_info["count"],
            "gpu_names": self.gpu_info["names"],
            "cuda_version": self.gpu_info["cuda_version"],
            "loaded_models": self.get_loaded_models()["loaded_models"]
        }
        
        # Try to add current memory info for each GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = []
                for i in range(self.gpu_info["count"]):
                    mem_free, mem_total = torch.cuda.mem_get_info(i)
                    gpu_memory.append({
                        "gpu_id": i,
                        "free_mb": mem_free / (1024 * 1024),
                        "total_mb": mem_total / (1024 * 1024),
                        "used_mb": (mem_total - mem_free) / (1024 * 1024),
                        "used_percent": ((mem_total - mem_free) / mem_total) * 100
                    })
                info["gpu_memory"] = gpu_memory
        except:
            pass
        
        return info

        
    async def find_suitable_model(self, gpu_id=None, quiet=False):  # Still async def
        """
        Find the first available model loaded on the specified GPU.
        Ensures thread-safe access to the loaded_models dictionary.
        quiet: if True, use DEBUG for "no model" and diagnostic logs (avoids noise when no memory model is loaded).
        """
        if not quiet:
            logging.info(f"[DIAGNOSTIC] find_suitable_model called for GPU {gpu_id}. Checking state within lock.")

        # Force memory agent tasks to ALWAYS default to GPU 1 (the 4060 Ti)
        target_gpu_id = 1 if gpu_id is None else gpu_id
        if not quiet:
            logging.info(f"Forcing memory agent to look for a model on GPU {target_gpu_id}")
        suitable_model_name = None # Variable to hold the result

        # --- Acquire the lock before reading the shared dictionary ---
        async with self.lock:
            current_loaded_state_inside_lock = list(self.loaded_models.keys())
            if not quiet:
                logging.info(f"[DIAGNOSTIC] Inside lock for find_suitable_model (GPU {target_gpu_id}). Current loaded: {current_loaded_state_inside_lock}")

            # Make a temporary copy of keys to iterate over safely (optional but safer if modifying)
            current_keys = list(self.loaded_models.keys())

            # Simplified logic: Find the first model loaded on the target GPU
            for model_key in current_keys:
                # Ensure key is a tuple with 2 elements before unpacking
                if isinstance(model_key, tuple) and len(model_key) == 2:
                    model_name, loaded_gpu_id = model_key # Unpack key directly
                    if loaded_gpu_id == target_gpu_id:
                        logging.info(f"Found model on specified GPU: {model_name} on GPU {target_gpu_id}")
                        suitable_model_name = model_name
                        break # Found the first one, exit loop
                else:
                    # Log if key format is unexpected (helps debug state issues)
                    logging.warning(f"[DIAGNOSTIC] Unexpected key format in loaded_models: {model_key}")
        # --- Lock is released here ---

        # Logging and return happen outside the lock
        if suitable_model_name is None:
            if quiet:
                logging.debug(f"No model found loaded on GPU {target_gpu_id}")
            else:
                logging.warning(f"No model found loaded on GPU {target_gpu_id}")
            return None

        return suitable_model_name
    
    def generate(self, request):
        """
        Generate a response using the loaded model with performance monitoring.
        
        Args:
            request: The generation request object containing prompt or messages
            
        Returns:
            A response object with the generated content in a consistent format
        """
        start_time = time.time()
        logging.info("Starting generate method")
        
        # Determine memory port for this request
        try:
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            memory_port = 8000 if gpu_count <= 1 else 8001
        except:
            memory_port = 8001  # fallback to dual GPU behavior
        
        # Extract the prompt from different request formats
        prompt = None
        if hasattr(request, 'prompt'):
            prompt = request.prompt
        elif hasattr(request, 'messages') and len(request.messages) > 0:
            prompt = request.messages[-1].content
        else:
            # Try to handle dictionary format
            if isinstance(request, dict):
                prompt = request.get('prompt', '')
                if not prompt and 'messages' in request and len(request['messages']) > 0:
                    prompt = request['messages'][-1].get('content', '')
        
        if not prompt:
            logging.warning("Could not extract prompt from request")
            prompt = ""
            
        logging.info(f"Extracted prompt: {prompt[:50]}...")
        
        # Check for memory curation keywords in the prompt
        memory_keywords = ["clean up the memory", "remove duplicates", "curate memory"]
        should_curate = any(keyword in prompt.lower() for keyword in memory_keywords)
        
        # Execute memory curation if needed
        curation_result = ""
        if should_curate:
            logging.info("Memory curation keywords detected, calling curation endpoint")
            try:
                # Use synchronous request to avoid awaiting a dict
                response = requests.post(f"http://localhost:{memory_port}/memory/curate")
                if response.status_code == 200:
                    curation_result = "Memory curation complete. "
                    logging.info("Memory curation successful")
                else:
                    curation_result = f"Memory curation failed with status {response.status_code}. "
                    logging.warning(f"Memory curation failed: {response.status_code}")
            except Exception as e:
                curation_result = f"Memory curation error: {str(e)}. "
                logging.error(f"Error during memory curation: {e}")
            
            # For memory curation requests, we still want to respond with a proper message
            if hasattr(request, 'model_name'):
                model_name = request.model_name
            elif isinstance(request, dict) and 'model_name' in request:
                model_name = request.get('model_name')
            else:
                model_name = "default_model"
                
            # Return in a format matching what your frontend expects
            return {
                "text": f"🧠 {curation_result}I've cleaned up the memory system for you, user. All duplicates have been removed and the memory store now contains only relevant, high-quality memories.",
                "model": model_name,
                "finish_reason": "stop"
            }
        
        # Get the active model - assuming the model name is in the request or using default
        model_name = None
        if hasattr(request, 'model_name'):
            model_name = request.model_name
        elif hasattr(request, 'model'):
            model_name = request.model
        elif isinstance(request, dict):
            model_name = request.get('model_name', request.get('model'))
        
        # Extract gpu_id - this is now mandatory
        gpu_id = None
        if hasattr(request, 'gpu_id'):
            gpu_id = request.gpu_id
        elif isinstance(request, dict):
            gpu_id = request.get('gpu_id')
        
        if gpu_id is None:
            raise ValueError("gpu_id is required for model selection")

        # If no model specified, use the first loaded model
        if not model_name and self.loaded_models:
            # Find the first model loaded on the specified GPU
            for model_key in self.loaded_models.keys():
                if model_key[1] == gpu_id: # Check if the model is on the requested GPU
                    model_name = model_key[0]
                    logging.info(f"No model specified, using model: {model_name} on GPU {gpu_id}")
                    break
        
        if not model_name:
            raise ValueError(f"No model specified and no models loaded on GPU {gpu_id}")
        
        # Get the model object using the updated get_model method
        model = self.get_model(model_name, gpu_id)
        
        # Performance monitoring: Check if this is a unified mode model
        is_unified_mode = False
        if hasattr(model, 'gpu_usage_mode'):
            is_unified_mode = model.gpu_usage_mode == "unified_model"
        elif isinstance(model, RemoteModelWrapper):
            is_unified_mode = True
        
        # Adapt request for llama-cpp vs ctransformers
        try:
            # Set parameters suitable for the model
            max_tokens = 2048  # Default
            if hasattr(request, 'max_tokens'):
                max_tokens = request.max_tokens
            elif isinstance(request, dict):
                max_tokens = request.get('max_tokens', 2048)
            
            temperature = 0.7  # Default
            if hasattr(request, 'temperature'):
                temperature = request.temperature
            elif isinstance(request, dict):
                temperature = request.get('temperature', 0.7)
            
                # Generate response with llama-cpp
                if LLAMA_CPP_AVAILABLE:
                    logging.info(f"Generating with llama-cpp, max_tokens={max_tokens}, temp={temperature}")
                    
                    # Performance monitoring: Start timing
                    generation_start = time.time()
                    
                    try:
                        output = model(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=["User:", "Human:", "<|im_end|>"]
                        )
                        logging.info(f"✅ Model call completed successfully")
                    except Exception as model_error:
                        logging.error(f"❌ Model call failed: {model_error}")
                        raise Exception(f"Model generation failed: {model_error}")
                    
                    # Performance monitoring: Calculate metrics
                    generation_time = time.time() - generation_start
                if output and "choices" in output and output["choices"]:
                    generated_text = output["choices"][0]["text"]
                    # Rough token count estimation (1 token ≈ 4 characters)
                    estimated_tokens = len(generated_text) // 4
                    tokens_per_second = estimated_tokens / generation_time if generation_time > 0 else 0
                    
                    logging.info(f"🚀 [Performance] Generation completed in {generation_time:.2f}s")
                    logging.info(f"🚀 [Performance] Estimated tokens: {estimated_tokens}")
                    logging.info(f"🚀 [Performance] Speed: {tokens_per_second:.1f} tokens/second")
                    if is_unified_mode:
                        logging.info(f"🚀 [Performance] Unified mode performance: {tokens_per_second:.1f} tokens/second")
                
                # Check if output is valid before formatting
                if output and isinstance(output, dict) and "choices" in output and output["choices"]:
                    # Format the llama-cpp output to a standardized response
                    result = {
                        "text": output["choices"][0]["text"],
                        "model": model_name,
                        "finish_reason": output["choices"][0]["finish_reason"],
                        "gpu_id_used": gpu_id,  # Include the GPU ID in the result
                        "performance_metrics": {
                            "generation_time": generation_time,
                            "estimated_tokens": estimated_tokens if 'estimated_tokens' in locals() else 0,
                            "tokens_per_second": tokens_per_second if 'tokens_per_second' in locals() else 0,
                            "mode": "unified_model" if is_unified_mode else "split_services"
                        }
                    }
                else:
                    # Handle invalid output
                    logging.error(f"❌ Invalid output from model: {output}")
                    raise Exception(f"Model returned invalid output: {output}")
            # Generate response with ctransformers
            elif CTRANSFORMERS_AVAILABLE:
                logging.info(f"Generating with ctransformers, max_tokens={max_tokens}")
                
                # Performance monitoring: Start timing
                generation_start = time.time()
                
                output_text = model(prompt, max_new_tokens=max_tokens, temperature=temperature)
                
                # Performance monitoring: Calculate metrics
                generation_time = time.time() - generation_start
                estimated_tokens = len(output_text) // 4
                tokens_per_second = estimated_tokens / generation_time if generation_time > 0 else 0
                
                logging.info(f"🚀 [Performance] Generation completed in {generation_time:.2f}s")
                logging.info(f"🚀 [Performance] Speed: {tokens_per_second:.1f} tokens/second")
                
                result = {
                    "text": output_text,
                    "model": model_name,
                    "finish_reason": "length",
                    "gpu_id_used": gpu_id,  # Include the GPU ID in the result
                    "performance_metrics": {
                        "generation_time": generation_time,
                        "estimated_tokens": estimated_tokens,
                        "tokens_per_second": tokens_per_second,
                        "mode": "unified_model" if is_unified_mode else "split_services"
                    }
                }
            else:
                raise ImportError("No compatible model loader available")
                
            # If we did memory curation, prepend that info to the response
            if should_curate:
                logging.info(f"Adding curation result to output: {curation_result}")
                result["text"] = curation_result + result["text"]
            
            # Final performance logging
            total_time = time.time() - start_time
            logging.info(f"🚀 [Performance] Total request time: {total_time:.2f}s")
            logging.info(f"Generation complete, output: {result['text'][:50]}...")
            return result
            
        except Exception as e:
            logging.exception(f"Error during generation: {e}")
            raise
    @property
    def app(self):
        """Get the FastAPI app instance from the current request."""
        from fastapi import Request
        import inspect

        # Walk the call stack to find a FastAPI Request instance
        frame = inspect.currentframe()
        while frame:
            if frame.f_locals.get('self') and isinstance(frame.f_locals.get('self'), Request):
                request = frame.f_locals.get('self')
                return request.app
            frame = frame.f_back

        # Return None if no request found
        return None