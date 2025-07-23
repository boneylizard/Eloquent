import logging
import json  # ADDED
import os    # ADDED
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
        
    def get_detector(self, model_name: str):
        """Get or load a specific detector model"""
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
                self.detectors[model_name] = YOLO(model_path)
                logger.info(f"Loaded ADetailer model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
                
        return self.detectors[model_name]
        
    def detect_faces(self, image: Image.Image, model_name: str = "face_yolov8n.pt", confidence: float = 0.3):
        """Detect faces using specified model"""
        detector = self.get_detector(model_name)
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
        self.loaded_model: Optional[StableDiffusion] = None
        self.current_model_path: Optional[str] = None
        self.is_flux_model: bool = False
        self.is_sdxl_model: bool = False
        self.adetailer_processor = ADetailerProcessor()

        # Set ADetailer model directory from settings
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        try:
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    adetailer_dir = settings.get('adetailerModelDirectory')
                    if adetailer_dir and os.path.isdir(adetailer_dir):
                        self.adetailer_processor.set_model_directory(adetailer_dir)  # FIXED: was self.adetailer
        except Exception as e:
            logger.warning(f"Could not load ADetailer model directory: {e}")        
        logger.info("SDManager initialized with ADetailer support")

    # Add the @property decorator to expose adetailer for backward compatibility
    @property
    def adetailer(self):
        """Backward compatibility property"""
        return self.adetailer_processor

    def enhance_image_with_adetailer(self, image_path: str, original_prompt: str = "", 
                                     face_prompt: str = "", strength: float = 0.35,  # FIXED: Reduced default
                                     confidence: float = 0.3, model_name: str = "face_yolov8n.pt") -> bytes:
        """
        STABLE-DIFFUSION.CPP COMPATIBLE - Post-process image with ADetailer face enhancement
        """
        if not self.loaded_model:
            raise RuntimeError("No SD model loaded for ADetailer enhancement")

        try:
            original_image = Image.open(image_path)
            logger.info(f"Loaded image for ADetailer enhancement: {original_image.size}")

            # Detect faces
            face_boxes = self.adetailer_processor.detect_faces(original_image, model_name=model_name, confidence=confidence)

            if not face_boxes:
                logger.info("No faces detected, returning original image")
                with open(image_path, 'rb') as f:
                    return f.read()

            logger.info(f"Detected {len(face_boxes)} faces for enhancement using {model_name}")

            # Process each face
            enhanced_image = original_image.copy()

            for i, box in enumerate(face_boxes):
                logger.info(f"Processing face {i+1}/{len(face_boxes)}: {box}")

                # 1) Build the per-face prompt – keep ultra-sharp phrasing
                if face_prompt.strip():
                    enhance_prompt = f"{face_prompt.strip()}, ultra-sharp 8k facial texture, {original_prompt}"
                else:
                    enhance_prompt = f"ultra-sharp 8k facial texture, {original_prompt}"

                # 2) Mask: tighter padding so effects stay on the face
                mask = self.adetailer_processor.create_mask_from_box(
                    original_image.size,
                    box,
                    padding=12,    # was 16 – stronger focus
                    blur=24,       # heavier feather for safety
                    dilation=10
                )

                # 3) Full-frame inpaint with explicit size
                w = (original_image.width  // 8) * 8
                h = (original_image.height // 8) * 8
                logger.info(f"Inpainting full frame at {w}×{h}")

                enhanced_result = self.loaded_model.img_to_img(
                    prompt=enhance_prompt,
                    image=enhanced_image,
                    mask_image=mask,
                    strength=0.34,          # bolder redraw, still below “plastic”
                    width=w,
                    height=h,
                    sample_steps=60,        # already maxed
                    cfg_scale=8.5,          # extra pop in colour / texture
                    sample_method="dpmpp2m"
                )
                if enhanced_result and len(enhanced_result) > 0:
                    enhanced_image = enhanced_result[0]   # replace whole canvas
                    logger.info(f"Successfully enhanced face {i+1}")
                else:
                    logger.warning(f"No result from inpainting for face {i+1}")

            # Convert final image to bytes
            with io.BytesIO() as buffer:
                enhanced_image.save(buffer, format="PNG")
                return buffer.getvalue()

        except Exception as e:
            logger.error(f"ADetailer enhancement failed: {e}", exc_info=True)
            with open(image_path, 'rb') as f:
                return f.read()

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
        
    def load_model(self, model_path: str) -> bool:
        """Load a Stable Diffusion or FLUX model"""
        try:
            logger.info(f"Loading SD model: {model_path}")
            
            self.is_flux_model = self._is_flux_model_path(model_path)
            self.is_sdxl_model = self._is_sdxl_model_path(model_path)

            if self.is_flux_model:
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
                    self.loaded_model = StableDiffusion(
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
                
            elif self.is_sdxl_model:
                logger.info("Detected SDXL model, using SDXL initialization with CPU VAE")
                try:
                    self.loaded_model = StableDiffusion(
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
                    self.loaded_model = StableDiffusion(
                        model_path=model_path,
                        wtype="default", 
                        verbose=True,
                    )
                if not self.loaded_model:
                    logger.error("Failed to initialize SDXL model")
                    return False

                # Note: SDXL models do not require special handling like FLUX
            else:
                # Standard Stable Diffusion model
                logger.info("Detected standard SD model, using standard initialization")
                self.loaded_model = StableDiffusion(
                    model_path=model_path,
                    wtype="default",
                    verbose=True,
                )
            
            self.current_model_path = model_path
            model_type = 'FLUX' if self.is_flux_model else ('SDXL' if self.is_sdxl_model else 'SD')
            logger.info(f"{model_type} model loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full error traceback: {traceback.format_exc()}")
            self.loaded_model = None
            self.current_model_path = None
            self.is_flux_model = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.loaded_model is not None
    
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image from prompt and return as bytes."""
        logger.info(f"Generating image with prompt: {prompt[:50]}...")  # Log first 50 chars of prompt
        logger.info(f"Using model: {self.current_model_path if self.current_model_path else 'No model loaded'}")
        logger.info(f"Model type: {'FLUX' if self.is_flux_model else ('SDXL' if self.is_sdxl_model else 'Standard SD')}")
        logger.info(f"Additional parameters: {kwargs}")
        # Ensure a model is loaded
        if not self.loaded_model:
            raise RuntimeError("No SD model loaded")
        
        # Extract parameters with FLUX-appropriate defaults
        width = kwargs.get('width', 512)
        height = kwargs.get('height', 512)
        negative_prompt = kwargs.get('negative_prompt', '')
        seed = kwargs.get('seed', random.randint(0, 2**32 - 1))
        
        if self.is_flux_model:
            # FLUX-specific settings
            steps = kwargs.get('steps', 4)
            cfg_scale = kwargs.get('cfg_scale', 1.0)
            sample_method = "euler"
            logger.info(f"Generating FLUX image: {steps} steps, cfg_scale={cfg_scale}")
        elif self.is_sdxl_model:
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
            logger.info(f"Starting generation: {width}x{height}, {steps} steps, cfg={cfg_scale}, seed={seed}")
            model_type = 'FLUX' if self.is_flux_model else ('SDXL' if self.is_sdxl_model else 'Standard SD')
            logger.info(f"Model type: {model_type}")

            
            images_list = self.loaded_model.txt_to_img(
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
            
        return image_bytes