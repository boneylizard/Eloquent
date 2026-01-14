import os
import logging
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import spandrel

logger = logging.getLogger(__name__)

class UpscaleManager:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.current_model = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure directory exists
        if not self.models_dir.exists():
            logger.warning(f"Upscaler directory not found: {self.models_dir}")
        else:
            self.scan_models()

    def scan_models(self):
        """Scan the models directory for .pth files."""
        if not self.models_dir.exists():
            return []
            
        self.models = {}
        for file in self.models_dir.glob("*.pth"):
            self.models[file.stem] = file
        
        logger.info(f"Found {len(self.models)} upscaler models in {self.models_dir}")
        return list(self.models.keys())

    def load_model(self, model_name: str):
        """Load a specific model by name."""
        if model_name == self.current_model_name and self.current_model is not None:
            return self.current_model
            
        if model_name not in self.models:
            # Try finding partially matching name
            matches = [k for k in self.models.keys() if model_name.lower() in k.lower()]
            if matches:
                 model_name = matches[0]
            else:
                raise ValueError(f"Model {model_name} not found")

        model_path = self.models[model_name]
        logger.info(f"Loading upscaler: {model_name} from {model_path}")
        
        try:
            # Load model architecture using spandrel
            model_descriptor = spandrel.ModelLoader().load_from_file(str(model_path))
            self.current_model = model_descriptor.model.to(self.device).eval()
            self.current_model_name = model_name
            return self.current_model
        except Exception as e:
            logger.error(f"Failed to load upscaler {model_name}: {e}")
            raise

    def upscale(self, image: Image.Image, model_name: str = None, scale_factor: float = None) -> Image.Image:
        """
        Upscale an image.
        If scale_factor is provided, it might resize the output if the model's native scale matches or differs.
        Most ESRGAN models are fixed 4x.
        """
        if not self.models:
            self.scan_models()
            if not self.models:
                raise ValueError("No upscaler models found. Please configure the Upscaler Model Directory.")

        # Default to first available if not specified
        if not model_name:
            if self.current_model_name:
                model_name = self.current_model_name
            else:
                model_name = next(iter(self.models.keys()))

        model = self.load_model(model_name)
        
        # Preprocess image
        # Convert PIL to Tensor
        img_np = np.array(image.convert("RGB"))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                output = model(img_tensor)
            
            # Postprocess
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            output_img = Image.fromarray(output)
            
            # Handle requested scale factor vs model native scale
            # If user asks for 2x but optimal model is 4x, we might downscale the result?
            # actually spandrel models have a .scale attribute usually (in the descriptor, not the model itself directly usually accessible easily without wrapper)
            # For now, we return the raw model output. The frontend can resize if needed, or we can resize here.
            
            # If specific scale requested and result is different, resize result
            if scale_factor:
                target_w = int(image.width * scale_factor)
                target_h = int(image.height * scale_factor)
                
                if output_img.width != target_w or output_img.height != target_h:
                    logger.info(f"Resizing upscaler output from {output_img.size} to {target_w}x{target_h}")
                    output_img = output_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            return output_img

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                raise ValueError("CUDA out of memory during upscaling")
            raise e
