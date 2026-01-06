# backend/app/model_service.py
import os
import json
import logging
import socket
import struct
import pickle
import threading
import time
from typing import Dict, Any, Optional
from llama_cpp import Llama
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    """Optimized model service with connection pooling and performance improvements"""

    def __init__(self):
        self.models = {}
        self.connection_pool = {}  # Connection pool for better performance
        self.pool_lock = threading.Lock()
        self.max_pool_size = 10
        
    def _get_connection(self, client_id: str) -> Optional[socket.socket]:
        """Get a connection from the pool or create a new one"""
        with self.pool_lock:
            if client_id in self.connection_pool:
                conn = self.connection_pool[client_id]
                try:
                    # Test if connection is still alive
                    conn.send(b'ping')
                    return conn
                except:
                    # Remove dead connection
                    del self.connection_pool[client_id]
            
            # Create new connection
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                conn.settimeout(300)  # 5 minute timeout for large model loading
                
                # Set TCP keepalive parameters for long operations
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)  # Start keepalive after 60s
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)  # Send keepalive every 30s
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)     # Allow 5 failed keepalives
                
                return conn
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
                return None
    
    def _return_connection(self, client_id: str, conn: socket.socket):
        """Return a connection to the pool"""
        with self.pool_lock:
            if len(self.connection_pool) < self.max_pool_size:
                self.connection_pool[client_id] = conn
            else:
                conn.close()

    def load_model(self, model_name, model_path, gpu_id, context_length, params, gpu_usage_mode='split_services'):
        """Load a model with the correct environment and parameters for the selected GPU mode."""
        key = (model_name, gpu_id)

        # --- START: DEFINITIVE ENVIRONMENT AND PARAMETER FIX ---

        logging.info(f"--- ⚙️ Preparing to load '{model_name}' in '{gpu_usage_mode}' mode ---")

        # 1. Configure the environment for this process
        if gpu_usage_mode == "unified_model":
            # In Unified Mode, UNSET CUDA_VISIBLE_DEVICES so llama.cpp can see all GPUs.
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            
            # Enable maximum verbose logging for llama.cpp to see tensor splitting in real-time
            os.environ["GGML_VERBOSE"] = "1"
            os.environ["LLAMA_VERBOSE"] = "1"
            os.environ["GGML_CUDA_VERBOSE"] = "1"
            
            logging.info("✅ [Unified Mode] Environment configured for multi-GPU visibility.")
            logging.info("🔍 [Unified Mode] Enabled maximum verbose logging for real-time tensor split monitoring.")
        else: # split_services mode
            # In Split Mode, ISOLATE this process to a single GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logging.info(f"✅ [Split Mode] Environment isolated to GPU: {gpu_id}")

        # 2. Check if the model is already loaded correctly
        if key in self.models:
            if self.models[key].get('context_length') == context_length:
                logging.info(f"Model '{model_name}' is already loaded correctly.")
                return {"status": "already_loaded"}
            else:
                logging.info("Context length has changed. Unloading and reloading model.")
                self.unload_model(model_name, gpu_id)
        
        # 3. Finalize model parameters based on the mode
        if params is None:
            params = {}
        model_params = params.copy()
        
        # Check if embedding=True was explicitly passed, or auto-detect from model name
        is_embedding_model = model_params.get('embedding', False)
        if not is_embedding_model:
            # Auto-detect embedding models from name
            is_embedding_model = any(k in model_name.lower() for k in ["embed", "embedding", "gme", "gte", "bge", "jina", "nomic", "arctic", "mxbai", "e5", "frida", "inf-retriever", "sentence-t5"])
        
        if is_embedding_model:
            model_params['embedding'] = True
            logging.info(f"✅ Loading '{model_name}' as an embedding model.")

        # Add progress logging for large models
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
        if model_size_mb > 1000:  # Models larger than 1GB
            logging.info(f"📊 [ModelService] Loading large model: {model_name} ({model_size_mb:.1f} MB)")
            logging.info(f"⏱️ [ModelService] This may take several minutes for 70B+ models...")

        if gpu_usage_mode == "unified_model":
            # For unified mode, we rely on tensor_split and MUST remove main_gpu.
            if 'main_gpu' in model_params:
                del model_params['main_gpu']
            
            # Add performance optimizations for unified mode
            model_params.update({
                'n_batch': 8192,  # Larger batch size for better GPU utilization
                'n_threads': 32,   # More threads for better CPU-GPU coordination
                'use_mmap': True,   # Memory mapping for faster loading
                'use_mlock': True,  # Lock memory to prevent swapping
                'low_vram': False,  # Disable low VRAM mode for better performance
                'flash_attn': True, # Enable flash attention if available
                'rope_scaling': {"type": "yarn", "factor": 1.0},
                'use_cache': True,  # Enable KV cache for faster generation
                'verbose': True,    # Enable verbose logging to see tensor splitting in real-time
            })
            
            # Ensure we have basic required parameters
            if 'n_ctx' not in model_params:
                model_params['n_ctx'] = context_length
            if 'n_gpu_layers' not in model_params:
                model_params['n_gpu_layers'] = -1  # Use all available GPU layers
            
            # Validate tensor_split parameter
            if 'tensor_split' in model_params:
                tensor_split = model_params['tensor_split']
                if not isinstance(tensor_split, list) or len(tensor_split) != 2:
                    logging.warning(f"⚠️ [ModelService] Invalid tensor_split format: {tensor_split}")
                    logging.warning(f"⚠️ [ModelService] Expected list of 2 floats, got {type(tensor_split)} with length {len(tensor_split) if isinstance(tensor_split, list) else 'N/A'}")
                else:
                    total_split = sum(tensor_split)
                    if abs(total_split - 1.0) > 0.01:  # Allow small floating point errors
                        logging.warning(f"⚠️ [ModelService] Tensor split values don't sum to 1.0: {tensor_split} = {total_split}")
                        # Try to normalize the tensor split
                        normalized_split = [val / total_split for val in tensor_split]
                        model_params['tensor_split'] = normalized_split
                        logging.info(f"✅ [ModelService] Normalized tensor_split to {normalized_split}")
                    else:
                        logging.info(f"✅ [ModelService] Tensor split validation passed: {tensor_split}")
            
            logging.info("✅ [Unified Mode] Final parameters prepared for tensor splitting with performance optimizations.")
        else:
            # For split mode, we explicitly set the main_gpu.
            model_params['main_gpu'] = gpu_id
            
            # Ensure we have basic required parameters
            if 'n_ctx' not in model_params:
                model_params['n_ctx'] = context_length
            if 'n_gpu_layers' not in model_params:
                model_params['n_gpu_layers'] = -1  # Use all available GPU layers
                
            logging.info(f"✅ [Split Mode] Final parameters prepared for single GPU: {gpu_id}")

        # --- END: DEFINITIVE FIX ---

        try:
            logging.info(f"Attempting to instantiate Llama object for {model_name}...")
            logging.info(f"Using parameters: {json.dumps(model_params, indent=2, sort_keys=True)}")
            
            # Log environment variables that might affect llama.cpp
            logging.info(f"🔍 [ModelService] Environment check:")
            logging.info(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            logging.info(f"   PATH: {os.environ.get('PATH', 'Not set')[:100]}...")
            logging.info(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            
            # Check if model file exists and is accessible
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_file_size = os.path.getsize(model_path)
            logging.info(f"📁 [ModelService] Model file exists: {model_path}")
            logging.info(f"📊 [ModelService] Model file size: {model_file_size / (1024*1024):.1f} MB")
            
            # Check GPU memory availability if CUDA is available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    logging.info(f"🔍 [ModelService] CUDA available with {gpu_count} GPUs")
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        gpu_memory_free = torch.cuda.memory_reserved(i) / (1024**3)
                        logging.info(f"   GPU {i}: {gpu_name} - Total: {gpu_memory:.1f}GB, Reserved: {gpu_memory_free:.1f}GB")
                else:
                    logging.warning(f"⚠️ [ModelService] CUDA not available - this might cause issues with GPU models")
            except ImportError:
                logging.warning(f"⚠️ [ModelService] PyTorch not available - can't check GPU memory")
            except Exception as e:
                logging.warning(f"⚠️ [ModelService] Error checking GPU memory: {e}")
            
            # Add progress logging for large models
            if model_size_mb > 1000:
                logging.info(f"🔄 [ModelService] Loading large model - this may take several minutes...")
                logging.info(f"📊 [ModelService] Model size: {model_size_mb:.1f} MB")
                logging.info(f"⏱️ [ModelService] Starting model instantiation...")
            
            start_time = time.time()
            logging.info(f"🔄 [ModelService] About to call Llama() constructor...")
            logging.info(f"🔄 [ModelService] Model path: {model_path}")
            logging.info(f"🔄 [ModelService] Parameters: {model_params}")
            
            # Try to import llama_cpp to ensure it's available
            try:
                import llama_cpp
                logging.info(f"✅ [ModelService] llama_cpp imported successfully: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'version unknown'}")
            except ImportError as e:
                logging.error(f"❌ [ModelService] Failed to import llama_cpp: {e}")
                raise ImportError(f"llama_cpp not available: {e}")
            
            # Try to read a small portion of the model file to check if it's accessible
            try:
                with open(model_path, 'rb') as f:
                    # Read first 1024 bytes to check file accessibility
                    header = f.read(1024)
                    logging.info(f"✅ [ModelService] Model file is readable (read {len(header)} bytes)")
                    # Check if it looks like a GGUF file (should start with GGUF magic)
                    if header.startswith(b'GGUF'):
                        logging.info(f"✅ [ModelService] Model file appears to be a valid GGUF file")
                    else:
                        logging.warning(f"⚠️ [ModelService] Model file doesn't start with GGUF magic - might be corrupted")
            except Exception as e:
                logging.warning(f"⚠️ [ModelService] Could not read model file header: {e}")
            
            logging.info(f"🔄 [ModelService] Calling Llama constructor with {len(model_params)} parameters...")
            
            # Highlight embedding parameter
            if model_params.get('embedding'):
                logging.info(f"   🔍 EMBEDDING MODE: embedding=True")
            else:
                logging.info(f"   🔍 TEXT GENERATION MODE: embedding not set or False")
            
            # Log each parameter individually for debugging
            for param_name, param_value in model_params.items():
                if isinstance(param_value, (list, tuple)) and len(str(param_value)) > 100:
                    logging.info(f"   {param_name}: {type(param_value).__name__} with {len(param_value)} items")
                else:
                    logging.info(f"   {param_name}: {param_value}")
            
            try:
                model = Llama(
                    model_path=model_path,
                    **model_params
                )
            except Exception as llama_error:
                logging.error(f"❌ [ModelService] Llama constructor failed with error: {llama_error}")
                logging.error(f"❌ [ModelService] Error type: {type(llama_error).__name__}")
                
                # If tensor_split failed, try without it as a fallback
                if 'tensor_split' in model_params and "llama_context" in str(llama_error).lower():
                    logging.warning(f"⚠️ [ModelService] Tensor split failed, trying without tensor_split as fallback...")
                    fallback_params = model_params.copy()
                    del fallback_params['tensor_split']
                    
                    try:
                        logging.info(f"🔄 [ModelService] Retrying with fallback parameters (no tensor_split)...")
                        model = Llama(
                            model_path=model_path,
                            **fallback_params
                        )
                        logging.info(f"✅ [ModelService] Fallback loading successful without tensor_split!")
                    except Exception as fallback_error:
                        logging.error(f"❌ [ModelService] Fallback loading also failed: {fallback_error}")
                        # Re-raise the original error
                        raise llama_error
                else:
                    # Re-raise to be caught by the outer exception handler
                    raise
            
            logging.info(f"✅ [ModelService] Llama() constructor completed successfully!")
            load_time = time.time() - start_time
            
            if model_size_mb > 1000:
                logging.info(f"✅ [ModelService] Large model loaded successfully in {load_time:.1f} seconds")
            else:
                logging.info("✅ Llama object instantiated successfully.")

            self.models[key] = {
                'model': model,
                'context_length': context_length,
                'path': model_path,
                'gpu_usage_mode': gpu_usage_mode
            }

            logging.info(f"✅ [ModelService] Model loaded successfully, returning success response")
            return {"status": "success"}
        except Exception as e:
            logging.error(f"❌ [ModelService] Failed to load model '{model_name}': {e}")
            logging.error(f"❌ [ModelService] Exception type: {type(e).__name__}")
            logging.error(f"❌ [ModelService] Exception details: {str(e)}")
            
            # Log the full traceback for debugging
            import traceback
            logging.error(f"❌ [ModelService] Full traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logging.error(f"   {line}")
            
            # Check if it's a specific llama.cpp error
            error_msg = str(e)
            if "llama_context" in error_msg.lower():
                logging.error(f"❌ [ModelService] This appears to be a llama.cpp context creation error")
                logging.error(f"❌ [ModelService] Common causes: insufficient VRAM, corrupted model file, or parameter mismatch")
            
            return {"status": "error", "error": str(e)}

    def unload_model(self, model_name, gpu_id):
        """Unload a model and free VRAM"""
        key = (model_name, gpu_id)
        if key in self.models:
            # Delete the model object
            del self.models[key]['model']
            del self.models[key]
            
            # Force garbage collection
            gc.collect()
            
            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
                
            logging.info(f"Unloaded {model_name} from GPU {gpu_id}")
            return {"status": "unloaded"}
        return {"status": "not_loaded"}

    def generate(self, model_name, gpu_id, **kwargs):
        """Run inference with performance optimizations. Always acts as a generator."""
        key = (model_name, gpu_id)
        is_streaming = kwargs.get('stream', False)

        if key not in self.models:
            error_msg = {"error": "Model not loaded"}
            yield error_msg
            return
        
        try:
            model_info = self.models[key]
            model = model_info['model']
            
            logging.info(f"🔄 [ModelService] Running generation with kwargs: {kwargs}")
            
            # Create a mutable copy of the parameters
            generation_params = kwargs.copy()
            
            # Add default parameters if not present
            if 'temperature' not in generation_params:
                generation_params['temperature'] = 0.7
            if 'top_p' not in generation_params:
                generation_params['top_p'] = 0.9
            if 'top_k' not in generation_params:
                generation_params['top_k'] = 40
            if 'repeat_penalty' not in generation_params:
                generation_params['repeat_penalty'] = 1.1
            if 'max_tokens' not in generation_params or generation_params['max_tokens'] < 1:
                generation_params['max_tokens'] = 1024

            if is_streaming:
                logging.info(f"🔄 [ModelService] Using create_completion with stream=True for llama.cpp")
                completion_generator = model.create_completion(**generation_params)
                for chunk in completion_generator:
                    # Extract text from the chunk
                    if isinstance(chunk, dict) and 'choices' in chunk and chunk['choices']:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            # Send immediately without buffering
                            yield chunk
            else:
                logging.info(f"🔄 [ModelService] Using create_completion with stream=False for llama.cpp")
                # In non-streaming mode, we call the model and then yield the single, complete result.
                result = model.create_completion(**generation_params)
                yield result
                
        except Exception as e:
            logging.error(f"Generation error: {e}", exc_info=True)
            error_result = {"error": str(e)}
            yield error_result
    


    def embed(self, model_name, gpu_id, text):
        """Generate embeddings"""
        key = (model_name, gpu_id)
        if key not in self.models:
            return {"error": "Model not loaded for embedding"}
        
        try:
            # The embed method in llama-cpp-python returns the embeddings directly
            embedding_result = self.models[key]['model'].embed(text)
            return {"status": "success", "embedding": embedding_result}
        except Exception as e:
            logging.error(f"Embedding error: {e}")
            return {"error": str(e)}

def send_msg(sock, data):
    """Send a message with a length prefix"""
    try:
        msg = pickle.dumps(data)
        msg_len = struct.pack('>I', len(msg))
        sock.sendall(msg_len + msg)
    except Exception as e:
        logging.error(f"❌ [ModelService] Error sending message: {e}")
        raise

def recv_msg(sock):
    """Receive a message with a length prefix"""
    # Read message length
    raw_msglen = recv_all(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    
    # Read the message data
    data = recv_all(sock, msglen)
    if not data:
        return None
    
    return pickle.loads(data)

def recv_all(sock, n):
    """Helper to receive n bytes or return None if EOF is hit"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# Start the service
service = ModelService()
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
server.bind(('localhost', 5555))
server.listen(10)  # Increased backlog for better performance
logging.info("Model service started on port 5555 with performance optimizations")

while True:
    client, addr = server.accept()
    logging.info(f"New client connection from {addr}")
    
    try:
        data = recv_msg(client)
        if not data:
            client.close()
            continue

        action = data.get('action')
        params = data.get('params', {})
        
        logging.info(f"Processing {action} request from {addr}")
        logging.info(f"📋 Received params: {params}")

        if action == 'ping':
            logging.info(f"🏓 [ModelService] Responding to ping request")
            send_msg(client, {"status": "pong", "message": "ModelService is running", "timestamp": time.time()})
            client.close()

        elif action == 'load':
            logging.info(f"🔄 [ModelService] Processing load request for {params.get('model_name', 'unknown')}")
            
            # Extract parameters for logging
            model_name = params.get('model_name')
            model_path = params.get('model_path')
            gpu_id = params.get('gpu_id')
            context_length = params.get('context_length')
            model_params = params.get('params', {})  # Default to empty dict if None
            gpu_usage_mode = params.get('gpu_usage_mode', 'split_services')
            
            logging.info(f"📋 [ModelService] Extracted parameters:")
            logging.info(f"   model_name: {model_name}")
            logging.info(f"   model_path: {model_path}")
            logging.info(f"   gpu_id: {gpu_id}")
            logging.info(f"   context_length: {context_length}")
            logging.info(f"   gpu_usage_mode: {gpu_usage_mode}")
            logging.info(f"   model_params: {model_params}")
            
            # Validate required parameters
            if not model_name or not model_path or gpu_id is None or context_length is None:
                error_msg = f"Missing required parameters: model_name={model_name}, model_path={model_path}, gpu_id={gpu_id}, context_length={context_length}"
                logging.error(f"❌ [ModelService] {error_msg}")
                error_result = {"error": error_msg}
                send_msg(client, error_result)
                client.close()
                continue
            
            try:
                logging.info(f"🔄 [ModelService] Starting model load (this may take several minutes for 70B models)...")
                start_time = time.time()
                
                result = service.load_model(
                    model_name=model_name,
                    model_path=model_path,
                    gpu_id=gpu_id,
                    context_length=context_length,
                    params=model_params,
                    gpu_usage_mode=gpu_usage_mode
                )
                
                load_time = time.time() - start_time
                logging.info(f"✅ [ModelService] Load completed in {load_time:.1f}s, result: {result}")
                logging.info(f"🔄 [ModelService] Sending response back to client...")
                send_msg(client, result)
                logging.info(f"✅ [ModelService] Response sent, closing connection")
                client.close()
            except Exception as e:
                load_time = time.time() - start_time if 'start_time' in locals() else 0
                logging.error(f"❌ [ModelService] Error in load_model after {load_time:.1f}s: {e}")
                import traceback
                logging.error(f"❌ [ModelService] Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        logging.error(f"   {line}")
                error_result = {"error": str(e)}
                send_msg(client, error_result)
                client.close()

        elif action == 'generate':
            model_name = params.get('model_name')
            gpu_id = params.get('gpu_id')
            is_streaming = params.get('stream', False)
            
            # Create a dictionary of all other parameters, excluding the ones we pass explicitly.
            other_params = {k: v for k, v in params.items() if k not in ['model_name', 'gpu_id']}

            if is_streaming:
                logging.info(f"🔄 [ModelService] Starting STREAMING generation for {model_name}")
                try:
                    # In streaming mode, send each chunk as it arrives from the generator.
                    for chunk in service.generate(model_name=model_name, gpu_id=gpu_id, **other_params):
                        send_msg(client, chunk)
                    # Send a final 'None' message to signal the end of the stream.
                    send_msg(client, None)
                except Exception as e:
                    logging.error(f"❌ [ModelService] Streaming error: {e}", exc_info=True)
                    try:
                        send_msg(client, {"error": str(e)})
                    except: pass
                finally:
                    client.close()
            else:
                logging.info(f"🔄 [ModelService] Starting NON-STREAMING generation for {model_name}")
                try:
                    # In non-streaming mode, the result is still a generator. We must consume it.
                    # The final result from a non-streaming llama.cpp call is a single dictionary.
                    generator = service.generate(model_name=model_name, gpu_id=gpu_id, **other_params)
                    final_result = next(generator, None) # Get the single item from the generator.
                    
                    if final_result:
                        send_msg(client, final_result)
                    else:
                        send_msg(client, {"error": "Generation produced no output."})

                except Exception as e:
                    logging.error(f"❌ [ModelService] Non-streaming error: {e}", exc_info=True)
                    try:
                        send_msg(client, {"error": str(e)})
                    except: pass
                finally:
                    client.close()

        elif action == 'unload':
            logging.info(f"🔄 [ModelService] Processing unload request")
            model_name = params.get('model_name')
            gpu_id = params.get('gpu_id')
            
            if not model_name or gpu_id is None:
                error_msg = f"Missing required parameters for unload: model_name={model_name}, gpu_id={gpu_id}"
                logging.error(f"❌ [ModelService] {error_msg}")
                send_msg(client, {"error": error_msg})
                client.close()
                continue
            
            try:
                result = service.unload_model(model_name, gpu_id)
                logging.info(f"✅ [ModelService] Unload completed, result: {result}")
                send_msg(client, result)
                client.close()
            except Exception as e:
                logging.error(f"❌ [ModelService] Error in unload_model: {e}")
                send_msg(client, {"error": str(e)})
                client.close()

        elif action == 'embed':
            logging.info(f"🔄 [ModelService] Processing embed request")
            model_name = params.get('model_name')
            gpu_id = params.get('gpu_id')
            text = params.get('text')
            
            if not model_name or gpu_id is None or not text:
                error_msg = f"Missing required parameters for embed: model_name={model_name}, gpu_id={gpu_id}, text={bool(text)}"
                logging.error(f"❌ [ModelService] {error_msg}")
                send_msg(client, {"error": error_msg})
                client.close()
                continue
            
            try:
                result = service.embed(model_name, gpu_id, text)
                logging.info(f"✅ [ModelService] Embed completed, result type: {type(result)}")
                send_msg(client, result)
                client.close()
            except Exception as e:
                logging.error(f"❌ [ModelService] Error in embed: {e}")
                send_msg(client, {"error": str(e)})
                client.close()

        else:
            error_msg = f"Unknown action: {action}"
            logging.error(f"❌ [ModelService] {error_msg}")
            send_msg(client, {"error": error_msg})
            client.close()

    except Exception as e:
        logging.error(f"Error handling client request: {e}", exc_info=True)
        try:
            send_msg(client, {"error": str(e)})
        except:
            pass
        client.close()