import os
import logging
import asyncio
import glob
from typing import Dict, Any
from pathlib import Path

# --- Dependency Checks ---
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from ctransformers import AutoModelForCausalLM
    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    CTRANSFORMERS_AVAILABLE = False

MODEL_DIR = r"C:\Users\bpfit\OneDrive\Desktop\LLM AI GGUFs"  # Your model directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.lock = asyncio.Lock()
        self.models_dir = Path(MODEL_DIR)
        self.has_gpu = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            logging.info(f"GPU detected: {gpu_available}")
            return gpu_available
        except ImportError:
            logging.warning("PyTorch not found, assuming no GPU.")
            return False

    async def _load_with_llama_cpp(self, model_path: str, **kwargs):
        try:
            logging.info(f"Attempting to load with llama-cpp: {model_path}")
            loop = asyncio.get_running_loop()

            kwargs["n_gpu_layers"] = -1
            kwargs["n_ctx"] = 4096
            kwargs["n_batch"] = 512
            kwargs["main_gpu"] = 1
            kwargs["low_vram"] = False
            kwargs["f16_kv"] = True
            kwargs["numa"] = False
            kwargs["n_threads"] = 1

            model = await loop.run_in_executor(None, lambda: Llama(
                model_path=str(model_path),
                **kwargs
            ))

            logging.info(f"Successfully loaded model with llama-cpp: {model_path}")
            return model
        except Exception as e:
            logging.exception(f"Error loading model with llama-cpp: {e}")
            raise

    def _load_with_ctransformers(self, model_path: str, **kwargs):
        try:
            logging.info(f"Attempting to load with ctransformers: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            logging.info(f"Successfully loaded model with ctransformers: {model_path}")
            return model
        except Exception as e:
            logging.exception(f"Error loading model with ctransformers: {e}")
            raise

    async def load_model(self, model_name: str, **kwargs) -> None:
        async with self.lock:
            if model_name in self.loaded_models:
                logging.info(f"Model {model_name} already loaded.")
                return

            model_files = glob.glob(os.path.join(self.models_dir, "**", model_name), recursive=True)
            if not model_files:
                raise FileNotFoundError(f"Model {model_name} not found in {self.models_dir}")

            model_path = model_files[0]
            logging.info(f"Loading model {model_name} from {model_path}")

            try:
                params = {
                    "n_ctx": 4096,
                    "n_batch": 512,
                    "verbose": True,
                }

                if self.has_gpu:
                    params["n_gpu_layers"] = 100
                    logging.info("GPU detected, setting n_gpu_layers=100")
                else:
                    params["n_gpu_layers"] = 0
                    logging.info("No GPU detected, setting n_gpu_layers=0")

                params.update(kwargs)
                logging.info(f"Model parameters: {params}")

                if LLAMA_CPP_AVAILABLE:
                    model = await self._load_with_llama_cpp(str(model_path), **params)
                elif CTRANSFORMERS_AVAILABLE:
                    model = self._load_with_ctransformers(str(model_path), **params)
                else:
                    raise ImportError("No compatible GGUF model loader found.")

                self.loaded_models[model_name] = model
                logging.info(f"Model {model_name} loaded successfully.")

            except Exception as e:
                logging.exception(f"Error loading model {model_name}: {e}")
                raise

    async def unload_model(self, model_name: str):
        async with self.lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                logging.info(f"Model {model_name} unloaded.")
                import gc
                gc.collect()

    async def unload_all_models(self):
        async with self.lock:
            for model_name in list(self.loaded_models.keys()):
                await self.unload_model(model_name)
            logging.info("All models unloaded")

    def list_available_models(self):
        model_files = [filename for filename in os.listdir(MODEL_DIR) if filename.endswith(".gguf")]
        return {"available_models": model_files}

    def get_loaded_models(self):
        return {"loaded_models": list(self.loaded_models.keys())}
