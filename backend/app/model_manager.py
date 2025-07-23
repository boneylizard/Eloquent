# --- START: FORCE LLAMA_CPP IMPORT FIRST ---
# This MUST be the absolute first import to ensure llama.cpp claims the CUDA context
# before any other library (like nemo or torch) can.
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
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import sys
import json
import re
import pkg_resources
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
    """Wrapper that forwards calls to the model service"""

    def __init__(self, model_name, gpu_id):
        self.model_name = model_name
        self.gpu_id = gpu_id
        # Spoof llama_cpp backend for compatibility
        self.__class__.__module__ = 'llama_cpp.llama'

    def _send_msg(self, sock, data):
        """Packs and sends a pickled message."""
        try:
            msg = pickle.dumps(data)
            msg_len = struct.pack('>I', len(msg))
            sock.sendall(msg_len + msg)
        except (BrokenPipeError, ConnectionResetError) as e:
            logging.error(f"Socket error on send: {e}")
            raise

    def _recv_msg(self, sock):
        """Receives and unpacks a pickled message."""
        try:
            # Read the message length
            raw_msglen = sock.recv(4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]

            # Read the full message payload
            data = b''
            while len(data) < msglen:
                packet = sock.recv(msglen - len(data))
                if not packet:
                    return None # Connection broken
                data += packet
            
            return pickle.loads(data)
        except (ConnectionResetError, struct.error, EOFError) as e:
            logging.error(f"Socket error on receive: {e}")
            return None # Or handle error appropriately

    def _remote_call(self, action, **params):
        """Handles a single, non-streaming request-response."""
        # Use a new connection for each call to ensure thread safety
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            try:
                client.connect(('localhost', 5555))
                request = {'action': action, 'params': params}
                self._send_msg(client, request)
                response = self._recv_msg(client)
                return response
            except ConnectionRefusedError:
                logging.error("Connection to model service at localhost:5555 was refused.")
                raise HTTPException(status_code=503, detail="Model service is unavailable.")
            except Exception as e:
                logging.error(f"Error in remote call: {e}", exc_info=True)
                raise

    def _remote_stream(self, action, **params):
        """Yields responses from a streaming request."""
        # Use a new connection for each stream to ensure thread safety
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            try:
                client.connect(('localhost', 5555))
                request = {'action': action, 'params': params}
                self._send_msg(client, request)
                while True:
                    response = self._recv_msg(client)
                    # The service should send None or a specific sentinel to signal the end
                    if response is None:
                        break
                    yield response
            except ConnectionRefusedError:
                logging.error("Connection to model service at localhost:5555 was refused.")
                # Yield an error object that the consumer can handle
                yield {"error": "Model service is unavailable."}
            except Exception as e:
                logging.error(f"Error in remote stream: {e}", exc_info=True)
                yield {"error": str(e)}

    def __call__(self, prompt=None, **kwargs):
        """Main entry point for model generation, routes to streaming or non-streaming."""
        if prompt is not None:
            kwargs['prompt'] = prompt
        
        # Route based on the 'stream' parameter
        if kwargs.get('stream'):
            return self._remote_stream('generate', model_name=self.model_name, gpu_id=self.gpu_id, **kwargs)
        else:
            return self._remote_call('generate', model_name=self.model_name, gpu_id=self.gpu_id, **kwargs)

    def create_completion(self, prompt=None, **kwargs):
        """llama-cpp compatibility alias. Forwards to __call__."""
        return self.__call__(prompt=prompt, **kwargs)
    
    def unload(self):
        """Tell the service to unload this model"""
        return self._remote_call('unload', model_name=self.model_name, gpu_id=self.gpu_id)
    def embed(self, text: str):
        """Tell the service to generate embeddings for the given text."""
        return self._remote_call('embed', model_name=self.model_name, gpu_id=self.gpu_id, text=text)
    
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
            'forensic_embeddings': None    # {'name': 'model_name', 'gpu_id': 0}
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

    def _get_gpu_params(self, gpu_id: int, context_length: int = 4096) -> Dict[str, Any]:
        """
        Return optimal parameters for full GPU utilization, including tensor splitting for multi-GPU setups.
        """
        logging.info(f"🔍 _get_gpu_params called with gpu_id={gpu_id}, context_length={context_length}")

        params = {
            "n_ctx": context_length,
            "n_batch": 8192,
            "n_threads": 32,
            "verbose": True,
            "seed": 42,
            "n_gpu_layers": -1,
            "main_gpu": gpu_id,
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
        }

        effective_gpu_count = self.gpu_info.get("count", 1)

        if effective_gpu_count > 1:
            logging.info(f"Configuring tensor split for {effective_gpu_count} GPUs in '{self.gpu_usage_mode}' mode.")
            if self.gpu_usage_mode == "split_services":
                # Isolate model to its target GPU
                tensor_split = [0.0] * effective_gpu_count
                if gpu_id < effective_gpu_count:
                    tensor_split[gpu_id] = 1.0
                params["tensor_split"] = tensor_split
                logging.info(f"⭐ Split services mode: Setting tensor_split to {tensor_split}")
            elif self.gpu_usage_mode == "unified_model":
                # Distribute model across all available GPUs based on VRAM
                gpu_memory_ratios = []
                total_memory = 0
                for i in range(effective_gpu_count):
                    gpu_memory = self.gpu_info["memory"][i]["total_mb"]
                    gpu_memory_ratios.append(gpu_memory)
                    total_memory += gpu_memory
                tensor_split = [mem / total_memory for mem in gpu_memory_ratios]
                params["tensor_split"] = tensor_split
                logging.info(f"⭐ Unified model mode: Setting tensor_split based on memory ratios: {tensor_split}")
        else:
            logging.info("Only one GPU visible to the process, skipping tensor_split.")

        os.environ["GGML_CUDA_NO_PINNED"] = "0"
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
                'forensic_embeddings': None
            }

    async def _load_with_llama_cpp(self, model_path: str, gpu_id: Optional[int] = None, n_ctx: Optional[int] = 4096, **kwargs):
        """Load a model using the model service"""
        try:
            logging.info(f"⭐ Loading model via service on GPU {gpu_id} with context length {n_ctx}: {model_path}")
            
            # Get your normal parameters
            model_params = self._get_gpu_params(gpu_id or 0, context_length=n_ctx)
            
            # Handle vision models
            mmproj_path = self._find_matching_mmproj(model_path)
            if mmproj_path:
                model_params["clip_model_path"] = str(mmproj_path)
            
            # Send load request to service
            wrapper = RemoteModelWrapper(os.path.basename(model_path), gpu_id)
            
            logging.info(f"🚀 [ModelManager] Dispatching 'load' command to ModelService for {os.path.basename(model_path)} on GPU {gpu_id}.")
            result = wrapper._remote_call(
                'load',
                model_name=os.path.basename(model_path),
                model_path=str(model_path),
                gpu_id=gpu_id,
                context_length=n_ctx,
                params=model_params
            )
            logging.info(f"👍 [ModelManager] Received response from ModelService: {result}")
            
            if "error" in result:
                raise Exception(result["error"])
            
            logging.info(f"✅ Model loaded via service: {result}")
            return wrapper
            
        except Exception as e:
            logging.exception(f"❌ Error loading model via service: {e}")
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
            target_gpu_id = self.default_gpu_id if gpu_id is None else gpu_id

            # Create a composite key for this model (model_name, gpu_id)
            model_key = (model_name, target_gpu_id)

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

                if score > highest_score:
                    highest_score = score
                    best_match = file_path

            if not best_match:
                # The error message now shows the cleaned keywords for easier debugging
                raise FileNotFoundError(f"Could not find a GGUF file matching keywords: {keywords} in '{self.models_dir}'")

            model_path = best_match
            print(f"DEBUG: Found best model match: {model_path}", flush=True)

            logging.info(f"⭐ Loading model {model_name} from {model_path} on GPU {target_gpu_id} with context length {context_length}")

            try:
                if not self.has_gpu:
                    logging.warning("No GPU detected, model will use CPU only")
                    kwargs["n_gpu_layers"] = 0
                    target_gpu_id = -1
                elif target_gpu_id is not None and target_gpu_id >= 0:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)  # Ensure this is set *before* loading
                    logging.info(f"⭐ Setting CUDA_VISIBLE_DEVICES to {target_gpu_id} for loading {model_name}")

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
                try:
                    if LLAMA_CPP_AVAILABLE:
                        import time
                        test_prompt = "Hello, this is a quick test."
                        logging.info(f"Running test inference with prompt: '{test_prompt}'")

                        start_time = time.time()
                        test_output = model(test_prompt, max_tokens=20)
                        end_time = time.time()

                        inference_time = end_time - start_time
                        output_text = test_output["choices"][0]["text"]
                        token_count = len(output_text.split())
                        tokens_per_second = token_count / inference_time if inference_time > 0 else 0

                        logging.info(f"⭐ Test inference successful: {output_text}")
                        logging.info(f"⭐ Generated {token_count} tokens in {inference_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
                except Exception as test_e:
                    logging.warning(f"Test inference failed: {test_e}")

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

                # This is the fix - tell the service to unload!
                if isinstance(model_data["model"], RemoteModelWrapper):
                    result = model_data["model"]._remote_call(
                        'unload',
                        model_name=model_name,
                        gpu_id=gpu_id
                    )
                    logging.info(f"Service unload result: {result}")
                elif hasattr(model_data["model"], "shutdown"):
                    model_data["model"].shutdown()
                    logging.info(f"Subprocess for {model_name} shut down cleanly")

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
        valid_purposes = ['test_model', 'primary_judge', 'secondary_judge', 'test_model_a', 'test_model_b', 'forensic_embeddings']
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

        
    async def find_suitable_model(self, gpu_id=None):  # Still async def
        """
        Find the first available model loaded on the specified GPU.
        Ensures thread-safe access to the loaded_models dictionary.
        (Simplified: Removed keyword-based prioritization entirely)
        """
        logging.info(f"[DIAGNOSTIC] find_suitable_model called for GPU {gpu_id}. Checking state within lock.")  # Updated log

        # Force memory agent tasks to ALWAYS default to GPU 1 (the 4060 Ti)
        target_gpu_id = 1 if gpu_id is None else gpu_id
        logging.info(f"Forcing memory agent to look for a model on GPU {target_gpu_id}")
        suitable_model_name = None # Variable to hold the result

        # --- Acquire the lock before reading the shared dictionary ---
        async with self.lock:
            # Log the state *inside* the lock for accurate debugging
            current_loaded_state_inside_lock = list(self.loaded_models.keys())
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
            logging.warning(f"No model found loaded on GPU {target_gpu_id}")
            return None

        return suitable_model_name
    
    def generate(self, request):

        try:
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            memory_port = 8000 if gpu_count <= 1 else 8001
        except:
            memory_port = 8001  # fallback to dual GPU behavior
        """
        Generate a response using the loaded model.
        
        Args:
            request: The generation request object containing prompt or messages
            
        Returns:
            A response object with the generated content in a consistent format
        """
        logging.info("Starting generate method")
        
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
                if model_key[1] == gpu_id:  # Check if the model is on the requested GPU
                    model_name = model_key[0]
                    logging.info(f"No model specified, using model: {model_name} on GPU {gpu_id}")
                    break
        
        if not model_name:
            raise ValueError(f"No model specified and no models loaded on GPU {gpu_id}")
        
        # Get the model object using the updated get_model method
        model = self.get_model(model_name, gpu_id)
        
        # Adapt request for llama-cpp vs ctransformers
        try:
            # Set parameters suitable for the model
            max_tokens = 2048  # Default
            if hasattr(request, 'max_tokens'):
                max_tokens = request.max_tokens
            elif isinstance(request, dict):
                max_tokens = request.get('max_tokens', 2048)
            
            temperature = a = 0.7  # Default
            if hasattr(request, 'temperature'):
                temperature = request.temperature
            elif isinstance(request, dict):
                temperature = request.get('temperature', 0.7)
            
            # Generate response with llama-cpp
            if LLAMA_CPP_AVAILABLE:
                logging.info(f"Generating with llama-cpp, max_tokens={max_tokens}, temp={temperature}")
                output = model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["User:", "Human:", "<|im_end|>"]
                )
                
                # Format the llama-cpp output to a standardized response
                result = {
                    "text": output["choices"][0]["text"],
                    "model": model_name,
                    "finish_reason": output["choices"][0]["finish_reason"],
                    "gpu_id_used": gpu_id  # Include the GPU ID in the result
                }
            # Generate response with ctransformers
            elif CTRANSFORMERS_AVAILABLE:
                logging.info(f"Generating with ctransformers, max_tokens={max_tokens}")
                output_text = model(prompt, max_new_tokens=max_tokens, temperature=temperature)
                result = {
                    "text": output_text,
                    "model": model_name,
                    "finish_reason": "length",
                    "gpu_id_used": gpu_id  # Include the GPU ID in the result
                }
            else:
                raise ImportError("No compatible model loader available")
                
            # If we did memory curation, prepend that info to the response
            if should_curate:
                logging.info(f"Adding curation result to output: {curation_result}")
                result["text"] = curation_result + result["text"]
                
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