import logging
import json  # ADDED
import os    # ADDED
import threading
import time
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import asyncio
from stable_diffusion_cpp import StableDiffusion
import io
import random
import tempfile
import cv2
import numpy as np
from PIL import Image  # ADDED for Image type

logger = logging.getLogger(__name__)

class ADetailerProcessor:
    def __init__(self):
        self.detectors = {}  # Cache loaded models
        self.available_models = []
        self.model_directory = None
        self.initialized = False
        
    def set_model_directory(self, directory_path: str):
        """Set the directory where ADetailer models are stored"""
        self.model_directory = Path(directory_path)
        self.scan_available_models()
        
    def scan_available_models(self):
        """Scan directory for available .pt model files"""
        if not self.model_directory or not self.model_directory.exists():
            self.available_models = []
            return
            
        model_files = list(self.model_directory.glob("*.pt"))
        self.available_models = [f.name for f in model_files]
        logger.info(f"Found {len(self.available_models)} ADetailer models: {self.available_models}")
        
    def get_detector(self, model_name: str, gpu_id: int = 0):
        """Get or load a specific detector model on specified GPU"""
        if model_name not in self.detectors:
            if not self.model_directory:
                # Fallback to ultralytics auto-download
                model_path = model_name
            else:
                model_path = self.model_directory / model_name
                if not model_path.exists():
                    # Fallback to ultralytics auto-download
                    model_path = model_name
                else:
                    model_path = str(model_path)
                    
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                
                # Force YOLO to use the specified GPU
                device = f'cuda:{gpu_id}'
                model.to(device)
                
                self.detectors[model_name] = model
                logger.info(f"Loaded ADetailer model: {model_name} on GPU {gpu_id}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
                
        return self.detectors[model_name]
        
    def detect_faces(self, image: Image.Image, model_name: str = "face_yolov8n.pt", confidence: float = 0.3, gpu_id: int = 0):
        """Detect faces using specified model on specified GPU"""
        detector = self.get_detector(model_name, gpu_id=gpu_id)
        results = detector(image, conf=confidence, verbose=False)
        
        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    
        return boxes
    
    def create_mask_from_box(self, image_size: Tuple[int, int], box: Tuple[int, int, int, int], 
                             padding: int = 16, blur: int = 8, dilation: int = 0) -> Image.Image:
        """Create a mask from bounding box with optimized padding and blur - STABLE-DIFFUSION.CPP COMPATIBLE"""
        width, height = image_size
        x1, y1, x2, y2 = box

        # FIXED: Reduced padding to prevent halo effects (was 32, now 16)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # FIXED: Removed aggressive dilation (was 4, now 0)
        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # FIXED: Increased blur to prevent harsh seams (was 4, now 8)
        if blur > 0:
            blur_size = blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return Image.fromarray(mask, mode='L')

class SDManager:
    def __init__(self):
        self.loaded_models: Dict[int, StableDiffusion] = {}
        self.current_model_paths: Dict[int, str] = {}
        self.model_info: Dict[int, Dict[str, bool]] = {}
        self.adetailer_processor = ADetailerProcessor()
        self.progress_cache: Dict[str, Dict[str, Any]] = {}
        self.progress_lock = threading.Lock()
        self.model_locks: Dict[int, threading.Lock] = {}
        self.model_locks_lock = threading.Lock()

        # Set ADetailer model directory from settings
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        try:
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    adetailer_dir = settings.get('adetailerModelDirectory')
                    if adetailer_dir and os.path.isdir(adetailer_dir):
                        self.adetailer_processor.set_model_directory(adetailer_dir)
        except Exception as e:
            logger.warning(f"Could not load ADetailer model directory: {e}")        
        logger.info("SDManager initialized with ADetailer support")

    def _get_model_lock(self, gpu_id: int) -> threading.Lock:
        with self.model_locks_lock:
            lock = self.model_locks.get(gpu_id)
            if lock is None:
                lock = threading.Lock()
                self.model_locks[gpu_id] = lock
            return lock

    # Add the @property decorator to expose adetailer for backward compatibility
    @property
    def adetailer(self):
        """Backward compatibility property"""
        return self.adetailer_processor

    def init_progress(self, task_id: str, steps: Optional[int] = None, state: str = "starting") -> None:
        now = time.time()
        with self.progress_lock:
            self.progress_cache[task_id] = {
                "progress": 0.0,
                "state": state,
                "step": 0,
                "steps": steps,
                "started_at": now,
                "updated_at": now,
            }

    def update_progress(
        self,
        task_id: str,
        progress: Optional[float] = None,
        step: Optional[int] = None,
        steps: Optional[int] = None,
        state: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self.progress_lock:
            entry = self.progress_cache.get(task_id)
            if not entry:
                entry = {
                    "progress": 0.0,
                    "state": "starting",
                    "step": 0,
                    "steps": steps,
                    "started_at": now,
                    "updated_at": now,
                }
                self.progress_cache[task_id] = entry

            if steps is not None:
                entry["steps"] = steps
            if step is not None:
                entry["step"] = int(step)
            if progress is None and step is not None and entry.get("steps"):
                progress = ((step + 1) / float(entry["steps"])) * 100.0
            if progress is not None:
                entry["progress"] = max(0.0, min(100.0, float(progress)))
            if state:
                entry["state"] = state
            entry["updated_at"] = now

    def finish_progress(self, task_id: str, state: str = "complete") -> None:
        self.update_progress(task_id, progress=100.0, state=state)

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self.progress_lock:
            entry = self.progress_cache.get(task_id)
            return dict(entry) if entry else None

    def upscale_image(self, image_bytes: bytes, prompt: str = "", scale_factor: float = 2.0, strength: float = 0.2, gpu_id: int = 0) -> bytes:
        """
        Upscale an image by resizing it and running a weak img2img pass.
        """
        model_instance = self.loaded_models.get(gpu_id)
        if not model_instance:
            raise RuntimeError(f"No SD model loaded on GPU {gpu_id} for upscaling")

        try:
            # 1. Load the original image
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_width, original_height = original_image.size
            
            # 2. Resize (Upscale) using Lanczos
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Ensure dimensions are multiples of 64 (requirement for many SD models)
            new_width = (new_width // 64) * 64
            new_height = (new_height // 64) * 64
            # 3. Prepare prompt (if empty, use a generic enhancer)
            # Truncate prompt if too long and remove newlines to avoid CUDA errors
            if prompt:
                 # Sanitize: remove newlines and carriage returns, replace with comma
                 prompt = prompt.replace('\n', ', ').replace('\r', '')
                 # Reduce multiple spaces
                 import re
                 prompt = re.sub(r'\s+', ' ', prompt).strip()
                 
                 # Aggressive truncation to 100 chars for upscaling stability
                 # The "invalid configuration argument" CUDA error is likely triggered by
                 # prompts nearing the 77-token limit during img2img encoded batch processing.
                 if len(prompt) > 100:
                     logger.info(f"Aggressively truncating long prompt for upscale (len={len(prompt)})")
                     prompt = prompt[:100]
                     # Try to cut at the last comma to be cleaner and avoiding cutting words
                     last_comma = prompt.rfind(',')
                     if last_comma > 10: 
                         prompt = prompt[:last_comma]
                     else:
                         last_space = prompt.rfind(' ')
                         if last_space > 10:
                             prompt = prompt[:last_space]

            if not prompt:
                prompt = "highly detailed, high resolution, 8k, sharp focus, best quality"
            else:
                prompt = f"{prompt}, highly detailed, high resolution, 8k, sharp focus, best quality, masterpiece"

            # 4. Run img2img (weak strength to preserve structure but add detail)
            # Clean up PyTorch CUDA state
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info(f"Running img2img for upscale with strength {strength}")
            
            # Create a temporary file for the input image
            # We pass the ORIGINAL image and let img_to_img handle the resize to target width/height
            # This avoids potential issues with passing already-large images to the pipeline
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_input_file:
                original_image.save(temp_input_file, format="PNG")
                temp_input_path = temp_input_file.name

            try:
                upscaled_result = None

                model_lock = self._get_model_lock(gpu_id)
                with model_lock:
                    # Try generate_image_from_image if available (some wrappers have this specific method)
                    if hasattr(model_instance, 'generate_image_from_image'):
                        upscaled_result = model_instance.generate_image_from_image(
                            prompt=prompt,
                            image=temp_input_path,
                            strength=strength,
                            step_count=20, # method arg name might vary
                            cfg_scale=7.0,
                            sample_method="euler_a"
                        )
                    elif hasattr(model_instance, 'img_to_img'):
                        # Correct method name from inspection
                        upscaled_result = model_instance.img_to_img(
                            prompt=prompt,
                            image=temp_input_path, # Pass path string
                            strength=strength,
                            width=new_width,
                            height=new_height,
                            sample_steps=20,
                            cfg_scale=7.0,
                            sample_method="euler_a"
                        )
                    elif hasattr(model_instance, 'generate_image'):
                        upscaled_result = model_instance.generate_image(
                            prompt=prompt,
                            image=temp_input_path, # Pass path string
                            strength=strength,
                            width=new_width,
                            height=new_height,
                            sample_steps=20, 
                            cfg_scale=7.0,
                            sample_method="euler_a"
                        )
                    elif hasattr(model_instance, 'img2img'):
                        upscaled_result = model_instance.img2img(
                            prompt=prompt,
                            image=temp_input_path, # Pass path string
                            strength=strength,
                            width=new_width,
                            height=new_height,
                            sample_steps=20,
                            cfg_scale=7.0,
                            sample_method="euler_a"
                        )
                
                if upscaled_result and len(upscaled_result) > 0:
                    final_image = upscaled_result[0]
                    
                    # Convert to bytes
                    with io.BytesIO() as buffer:
                        final_image.save(buffer, format="PNG")
                        return buffer.getvalue()
                else:
                    raise RuntimeError("Upscaling returned no results (img2img pass failed)")
                    
            finally:
                # Cleanup temp file
                if os.path.exists(temp_input_path):
                    try:
                        os.unlink(temp_input_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp upscale input: {e}")

        except Exception as e:
            logger.error(f"Upscaling failed: {e}", exc_info=True)
            raise

    def enhance_image_with_adetailer(self, image_path: str, original_prompt: str = "",
                                     face_prompt: str = "", strength: float = 0.35,
                                     confidence: float = 0.3, model_name: str = "face_yolov8n_v2.pt", gpu_id: int = 0) -> bytes:
        """
        STABLE-DIFFUSION.CPP COMPATIBLE - Post-process image with ADetailer face enhancement on a specific GPU.
        """
        # CRITICAL FIX: Select the model for the requested GPU
        model_instance = self.loaded_models.get(gpu_id)
        if not model_instance:
            raise RuntimeError(f"No SD model loaded for ADetailer enhancement on GPU {gpu_id}")

        try:
            # Check if image_path is a path or raw bytes (for internal calls like after upscale)
            if isinstance(image_path, (bytes, bytearray)):
                original_image = Image.open(io.BytesIO(image_path))
            else:
                original_image = Image.open(image_path)
                
            logger.info(f"Loaded image for ADetailer enhancement: {original_image.size}")

            # CRITICAL FIX: Use the correct GPU ID for face detection
            face_boxes = self.adetailer_processor.detect_faces(original_image, model_name=model_name, confidence=confidence, gpu_id=gpu_id)

            if not face_boxes:
                logger.info("No faces detected, returning original image")
                # Return bytes of original
                with io.BytesIO() as buffer:
                    original_image.save(buffer, format="PNG")
                    return buffer.getvalue()

            logger.info(f"Detected {len(face_boxes)} faces for enhancement using {model_name} on GPU {gpu_id}")

            # Process each face
            enhanced_image = original_image.copy()
            model_lock = self._get_model_lock(gpu_id)

            for i, box in enumerate(face_boxes):
                logger.info(f"Processing face {i+1}/{len(face_boxes)}: {box}")

                # 1) Build the per-face prompt
                if face_prompt.strip():
                    enhance_prompt = f"{face_prompt.strip()}, ultra-sharp 8k facial texture, {original_prompt}"
                else:
                    enhance_prompt = f"ultra-sharp 8k facial texture, {original_prompt}"

                # 2) Create the mask
                mask = self.adetailer_processor.create_mask_from_box(
                    original_image.size,
                    box,
                    padding=12,
                    blur=24,
                    dilation=10
                )

                # 3) Inpaint
                w = (original_image.width  // 8) * 8
                h = (original_image.height // 8) * 8
                logger.info(f"Inpainting full frame at {w}×{h}")

                # Clean up PyTorch CUDA state before ggml inpainting
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                # Handle different API versions of stable-diffusion-cpp-python
                # Latest versions use generate_image with image param for img2img
                with model_lock:
                    if hasattr(model_instance, 'generate_image'):
                        enhanced_result = model_instance.generate_image(
                            prompt=enhance_prompt,
                            image=enhanced_image,
                            strength=0.30,
                            width=w,
                            height=h,
                            sample_steps=45,
                            cfg_scale=8.5,
                            sample_method="dpmpp2m"
                        )
                    elif hasattr(model_instance, 'img2img'):
                        enhanced_result = model_instance.img2img(
                            prompt=enhance_prompt,
                            image=enhanced_image,
                            mask_image=mask,
                            strength=0.30,
                            width=w,
                            height=h,
                            sample_steps=45,
                            cfg_scale=8.5,
                            sample_method="dpmpp2m"
                        )
                    elif hasattr(model_instance, 'img_to_img'):
                        enhanced_result = model_instance.img_to_img(
                            prompt=enhance_prompt,
                            image=enhanced_image,
                            mask_image=mask,
                            strength=0.30,
                            width=w,
                            height=h,
                            sample_steps=45,
                            cfg_scale=8.5,
                            sample_method="dpmpp2m"
                        )
                    else:
                        logger.error("StableDiffusion object has no generate_image or img2img method")
                        enhanced_result = None
                    
                if enhanced_result and len(enhanced_result) > 0:
                    enhanced_image = enhanced_result[0]
                    logger.info(f"Successfully enhanced face {i+1}")
                else:
                    logger.warning(f"No result from inpainting for face {i+1}")

            # Convert final image to bytes
            with io.BytesIO() as buffer:
                enhanced_image.save(buffer, format="PNG")
                return buffer.getvalue()

        except Exception as e:
            logger.error(f"ADetailer enhancement failed: {e}", exc_info=True)
            if 'original_image' in locals():
                with io.BytesIO() as buffer:
                    original_image.save(buffer, format="PNG")
                    return buffer.getvalue()
            # If we failed before loading, just re-read or fail
            if isinstance(image_path, str) and os.path.exists(image_path):
                 with open(image_path, 'rb') as f:
                    return f.read()
            return b"" # Should probably handle this better upstream

    def _clean_redundant_prompt_terms(self, prompt: str) -> str:
        """Remove redundant terms that might conflict in prompts - UNIVERSAL COMPATIBILITY"""
        terms = [term.strip() for term in prompt.split(',')]

        # Remove duplicates while preserving order
        seen = set()
        cleaned_terms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen and term.strip():
                seen.add(term_lower)
                cleaned_terms.append(term)

        return ', '.join(cleaned_terms)

    def is_adetailer_available(self) -> bool:
        """Check if ADetailer functionality is available"""
        try:
            from ultralytics import YOLO
            return True
        except ImportError:
            return False
        
    def _is_flux_model_path(self, model_path: str) -> bool:
        """Detect if the model path indicates a FLUX model"""
        model_name = Path(model_path).name.lower()
        return 'flux' in model_name
    
    def _is_sdxl_model_path(self, model_path: str) -> bool:
        """Detect if the model path indicates an SDXL model"""
        model_name = Path(model_path).name.lower()
        return any(keyword in model_name for keyword in ['juggernaut', 'juggernautXL', 'xl', 'sdxl', 'xl_base', 'xl_refiner', 'sd_xl', 'sd_xl_base', 'sd_xl_refiner', 'DiffusionXL', 'DiffusionXL_base', 'DiffusionXL_refiner', 'DiffusionXL'])

    def _get_flux_dependencies(self, model_path: str) -> Dict[str, str]:
        """Get expected paths for FLUX model dependencies"""
        model_dir = Path(model_path).parent
        
        return {
            'clip_l_path': str(model_dir / 'clip_l.safetensors'),
            't5xxl_path': str(model_dir / 't5xxl_fp16.safetensors'),
            'vae_path': str(model_dir / 'ae.safetensors')
        }
        
    def load_model(self, model_path: str, gpu_id: int = 0) -> bool:
        """Load a Stable Diffusion model on specified GPU"""
        try:
            logger.info(f"Loading SD model: {model_path} on GPU {gpu_id}")

            # Clean up PyTorch CUDA state before ggml takes over
            # This prevents CUDA context conflicts between PyTorch and ggml
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    logger.info("Cleaned up PyTorch CUDA state before SD load")
            except Exception as e:
                logger.warning(f"Could not clean PyTorch CUDA state: {e}")

            # Set CUDA device for this process
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Set CUDA_VISIBLE_DEVICES to {gpu_id}")

            is_flux = self._is_flux_model_path(model_path)
            is_sdxl = self._is_sdxl_model_path(model_path)
            
            # Store model info
            self.model_info[gpu_id] = {'is_flux': is_flux, 'is_sdxl': is_sdxl}

            if is_flux:
                logger.info("Detected FLUX model, using FLUX initialization")
                
                # Get dependency paths
                deps = self._get_flux_dependencies(model_path)
                logger.info(f"FLUX dependencies: {deps}")
                
                # Check if dependencies exist
                missing_deps = []
                for dep_name, dep_path in deps.items():
                    if not Path(dep_path).exists():
                        missing_deps.append(f"{dep_name}: {dep_path}")
                    else:
                        file_size = Path(dep_path).stat().st_size / (1024*1024)  # MB
                        logger.info(f"Found {dep_name}: {dep_path} ({file_size:.1f} MB)")
                
                if missing_deps:
                    logger.error(f"FLUX model dependencies missing: {missing_deps}")
                    logger.error("Required FLUX files:")
                    logger.error("- clip_l.safetensors (download from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors)")
                    logger.error("- t5xxl_fp16.safetensors (download from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)")
                    logger.error("- ae.safetensors (download from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors)")
                    return False
                
                logger.info("Initializing FLUX model with StableDiffusion library...")
                try:
                    model_instance = StableDiffusion(
                        diffusion_model_path=model_path,  # Use diffusion_model_path for FLUX
                        clip_l_path=deps['clip_l_path'],
                        t5xxl_path=deps['t5xxl_path'],
                        vae_path=deps['vae_path'],
                        vae_decode_only=True,  # FLUX optimization
                        wtype="default",
                        verbose=True, # Enable verbose for debugging
                    )
                    logger.info("FLUX StableDiffusion object created successfully")
                except Exception as flux_init_error:
                    logger.error(f"FLUX initialization failed: {type(flux_init_error).__name__}: {flux_init_error}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise flux_init_error
                
            elif is_sdxl:
                logger.info("Detected SDXL model, using SDXL initialization with CPU VAE")
                try:
                    model_instance = StableDiffusion(
                        model_path=model_path,
                        wtype="default",
                        verbose=True,
                        # Use keep_vae_on_cpu=True to ensure VAE is loaded on CPU
                        keep_vae_on_cpu=True,
                        vae_dtype="f16"  # Suggest using 16-bit float for VAE to save memory
                    )
                    logger.info("SDXL model initialized - checking if VAE is on CPU...")
                except TypeError as e:
                    logger.error(f"Parameter error (keep_vae_on_cpu may not be supported): {e}")
                    logger.info("Falling back to standard SDXL initialization...")
                    model_instance = StableDiffusion(
                        model_path=model_path,
                        wtype="default", 
                        verbose=True,
                    )
                if not model_instance:
                    logger.error("Failed to initialize SDXL model")
                    return False

                # Note: SDXL models do not require special handling like FLUX
            else:
                # Standard Stable Diffusion model
                logger.info("Detected standard SD model, using standard initialization")
                model_instance = StableDiffusion(
                    model_path=model_path,
                    wtype="default",
                    verbose=True,
                )
            
            self.loaded_models[gpu_id] = model_instance
            self.current_model_paths[gpu_id] = model_path
            
            model_type = 'FLUX' if is_flux else ('SDXL' if is_sdxl else 'SD')
            logger.info(f"{model_type} model loaded successfully on GPU {gpu_id}: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path} on GPU {gpu_id}: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full error traceback: {traceback.format_exc()}")
            self.loaded_models.pop(gpu_id, None)
            self.current_model_paths.pop(gpu_id, None)
            self.model_info.pop(gpu_id, None)
            return False
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.loaded_model is not None
    
    def generate_image(self, prompt: str, gpu_id: int = 0, task_id: Optional[str] = None, **kwargs) -> bytes:
        """Generate image from prompt on a specific GPU and return as bytes."""
        logger.info(f"Generating image with prompt: {prompt[:50]}... on GPU {gpu_id}")
        
        # CRITICAL FIX: Select the model for the requested GPU
        model_instance = self.loaded_models.get(gpu_id)
        if not model_instance:
            raise RuntimeError(f"No SD model loaded on GPU {gpu_id}")

        model_info = self.model_info.get(gpu_id, {})
        is_flux = model_info.get('is_flux', False)
        is_sdxl = model_info.get('is_sdxl', False)

        logger.info(f"Using model: {self.current_model_paths.get(gpu_id, 'N/A')}")
        logger.info(f"Model type: {'FLUX' if is_flux else ('SDXL' if is_sdxl else 'Standard SD')}")
        logger.info(f"Additional parameters: {kwargs}")
        
        # Extract parameters with FLUX-appropriate defaults
        width = kwargs.get('width', 512)
        height = kwargs.get('height', 512)
        negative_prompt = kwargs.get('negative_prompt', '')
        seed = kwargs.get('seed', random.randint(0, 2**32 - 1))
        
        if is_flux:
            # FLUX-specific settings
            steps = kwargs.get('steps', 4)
            cfg_scale = kwargs.get('cfg_scale', 1.0)
            sample_method = "euler"
            logger.info(f"Generating FLUX image: {steps} steps, cfg_scale={cfg_scale}")
        elif is_sdxl:
            # SDXL-specific settings
            steps = kwargs.get('steps', 20)
            cfg_scale = kwargs.get('cfg_scale', 7.0)
            sample_method = kwargs.get('sample_method', 'euler')
            logger.info(f"Generating SDXL image: {steps} steps, cfg_scale={cfg_scale}")
        else:
            # Standard SD settings
            steps = kwargs.get('steps', 20)
            cfg_scale = kwargs.get('cfg_scale', 7.0)
            sample_method = kwargs.get('sample_method', 'euler')
            logger.info(f"Generating Standard SD image: {steps} steps, cfg_scale={cfg_scale}")

        # Generate image using the appropriate parameters
        try:
            progress_callback = None
            if task_id:
                self.init_progress(task_id, steps=steps, state="starting")

                def progress_callback(*args, **callback_kwargs):
                    step = callback_kwargs.get("step")
                    step_count = callback_kwargs.get("step_count")
                    progress = callback_kwargs.get("progress")

                    if len(args) >= 2 and step is None and step_count is None:
                        step = args[0]
                        step_count = args[1]
                    elif len(args) >= 1 and progress is None and step is None:
                        progress = args[0]

                    if progress is not None and 0.0 <= float(progress) <= 1.0:
                        progress = float(progress) * 100.0

                    state = "sampling"
                    if step_count:
                        state = f"Sampling {int(step) + 1}/{int(step_count)}" if step is not None else state

                    self.update_progress(task_id, progress=progress, step=step, steps=step_count, state=state)
                    if step is not None and step_count:
                        logger.info(f"SD progress: {int(step) + 1}/{int(step_count)}")
                    return True

            # Clean up PyTorch CUDA state before ggml generation
            # This prevents CUDA context conflicts between PyTorch (TTS) and ggml (SD)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass  # Silently continue if PyTorch cleanup fails
            
            logger.info(f"Starting generation: {width}x{height}, {steps} steps, cfg={cfg_scale}, seed={seed}")
            
            # Handle different API versions of stable-diffusion-cpp-python
            # Latest versions use generate_image, older use txt2img or txt_to_img
            def call_with_progress(method, **call_kwargs):
                if progress_callback:
                    call_kwargs["progress_callback"] = progress_callback
                try:
                    return method(**call_kwargs)
                except TypeError as e:
                    if progress_callback and "progress_callback" in str(e):
                        call_kwargs.pop("progress_callback", None)
                        return method(**call_kwargs)
                    raise

            def supports_progress(method):
                try:
                    return "progress_callback" in inspect.signature(method).parameters
                except (TypeError, ValueError):
                    return False

            candidates = []
            if hasattr(model_instance, 'generate_image'):
                candidates.append(("generate_image", model_instance.generate_image))
            if hasattr(model_instance, 'txt_to_img'):
                candidates.append(("txt_to_img", model_instance.txt_to_img))
            if hasattr(model_instance, 'txt2img'):
                candidates.append(("txt2img", model_instance.txt2img))

            method_name = None
            method = None

            if candidates:
                method_name, method = candidates[0]

            if task_id and method and not supports_progress(method):
                logger.warning(f"Progress callback not supported by {method_name}; progress updates may be unavailable.")

            if not method:
                methods = [m for m in dir(model_instance) if not m.startswith('_')]
                raise RuntimeError(f"StableDiffusion object has no generate_image method. Available methods: {methods}")

            logger.info(f"Using SD method: {method_name}")

            model_lock = self._get_model_lock(gpu_id)
            with model_lock:
                images_list = call_with_progress(
                    method,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    cfg_scale=cfg_scale,
                    sample_steps=steps,
                    sample_method=sample_method,
                    seed=seed
                )
            
            logger.info("Generation completed successfully")
        
        except Exception as generation_error:
            logger.error(f"CRITICAL: Generation crashed - {type(generation_error).__name__}: {generation_error}")
            import traceback
            logger.error(f"Full generation traceback: {traceback.format_exc()}")
            if task_id:
                self.update_progress(task_id, state="error")
            
            # Try to get memory info if possible
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    logger.error(f"GPU memory at crash: {result.stdout.strip()}")
            except:
                pass
            
            raise RuntimeError(f"Image generation failed: {generation_error}")


        
        # Check if we got an image and take the first one
        if not images_list:
            raise RuntimeError("Image generation failed.")
            
        pil_image = images_list[0]
        
        # Convert the PIL Image object to raw bytes in PNG format
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            
        if task_id:
            self.finish_progress(task_id, state="complete")

        return image_bytes
