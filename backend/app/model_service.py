# backend/app/model_service.py
import os
import socket
import pickle
import json
import logging
import struct

# Set CUDA environment BEFORE any imports
os.environ["CUDA_MODULE_LOADING"] = "EAGER"

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)

class ModelService:
    def __init__(self):
        self.models = {}
        
    def load_model(self, model_name, model_path, gpu_id, context_length, params):
        """Load a model with clean CUDA state"""
        key = (model_name, gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if key in self.models:
            if self.models[key]['context_length'] != context_length:
                del self.models[key]
            else:
                return {"status": "already_loaded"}


        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logging.info(f"Loading {model_name} on GPU {gpu_id} with context {context_length}")

        try:
            # Make sure we don't duplicate n_ctx
            model_params = params.copy()
            model_params['n_ctx'] = context_length

            # ADD THIS: Force model onto specific GPU using tensor_split
            model_params['main_gpu'] = gpu_id  # Set main GPU
            # --- ADD THESE TWO LINES ---
            # --- START OF MODIFICATION ---
            # Check for various embedding model name patterns
            is_embedding_model = any(keyword in model_name.lower() for keyword in ["embed", "gme", "gte", "qwen"])
            if is_embedding_model:
                model_params['embedding'] = True
                logging.info(f"✅ Loading {model_name} as embedding model.")
            # --- END OF MODIFICATION ---
            # If you have the tensor_split in params from model_manager, it should work
            # Otherwise add this:
            if 'tensor_split' not in model_params:
                gpu_count = 2  # Your GPU count
                tensor_split = [0.0] * gpu_count
                tensor_split[gpu_id] = 1.0
                model_params['tensor_split'] = tensor_split

            is_embedding_model = "embed" in model_name.lower() or "gme" in model_name.lower()

            # add embedding model specific params
            if is_embedding_model:
                model_params['embedding'] = True
                logging.info(f"Loading {model_name} as embedding model")

            logging.info(f"Attempting to instantiate Llama object for {model_name} on GPU {gpu_id}...")
            logging.info(f"Using parameters: {json.dumps(model_params, indent=2)}")
            
            model = Llama(
                model_path=model_path,
                **model_params
            )
            
            logging.info("✅ Llama object instantiated successfully.")

            self.models[key] = {
                'model': model,
                'context_length': context_length,
                'path': model_path
            }

            return {"status": "success"}
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return {"status": "error", "error": str(e)}

    def unload_model(self, model_name, gpu_id):
        """Unload a model and free VRAM"""
        key = (model_name, gpu_id)
        if key in self.models:
            # Delete the model object
            del self.models[key]['model']
            del self.models[key]
            
            # Force garbage collection
            import gc
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
        """Run inference"""
        key = (model_name, gpu_id)
        if key not in self.models:
            return {"error": "Model not loaded"}
        
        try:
            result = self.models[key]['model'](**kwargs)
            return result
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return {"error": str(e)}
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
    msg = pickle.dumps(data)
    msg_len = struct.pack('>I', len(msg))
    sock.sendall(msg_len + msg)

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
server.bind(('localhost', 5555))
server.listen(5)
logging.info("Model service started on port 5555")

while True:
    client, addr = server.accept()
    try:
        data = recv_msg(client)
        if not data:
            client.close()
            continue

        action = data.get('action')
        params = data.get('params', {})

        if action == 'load':
            result = service.load_model(**params)
            send_msg(client, result)

        elif action == 'generate':
            # Check if this is a streaming request
            if params.get('stream'):
                logging.info("Streaming request received. Iterating through generator.")
                try:
                    # The result is a generator. We must iterate over it.
                    stream_generator = service.generate(**params)
                    for token_chunk in stream_generator:
                        # Send each individual token back to the client
                        send_msg(client, token_chunk)
                    
                    # After the stream is finished, send a sentinel to signal the end.
                    # This tells the RemoteModelWrapper to stop listening.
                    send_msg(client, None) 
                    logging.info("Streaming complete. End sentinel sent.")

                except Exception as e:
                    logging.error(f"Error during stream generation: {e}", exc_info=True)
                    # Try to send an error message back if the stream fails mid-way
                    try:
                        send_msg(client, {"error": f"Stream error: {str(e)}"})
                        send_msg(client, None) # Also send the end sentinel
                    except:
                        pass # The client might have already disconnected
            else:
                # This is a standard, non-streaming request
                result = service.generate(**params)
                send_msg(client, result)

        elif action == 'unload':
            result = service.unload_model(
                model_name=params.get('model_name'),
                gpu_id=params.get('gpu_id')
            )
            send_msg(client, result)        
        elif action == 'embed':
            result = service.embed(**params)
            send_msg(client, result)
        else:
            result = {"error": "Unknown action"}
            send_msg(client, result)
            
    except Exception as e:
        logging.error(f"Fatal service loop error: {e}", exc_info=True)
        # Attempt to inform the client of the error if possible
        try:
            send_msg(client, {"error": str(e)})
        except:
            pass # Ignore errors if the socket is already closed
    finally:
        # Ensure the client connection is always closed
        client.close()