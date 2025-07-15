import aiohttp
import asyncio
import base64
import os
import logging
import uuid
import time
from typing import Dict, Any, Optional

# Configuration
AUTOMATIC1111_URL = os.environ.get("AUTOMATIC1111_URL", "http://127.0.0.1:7860")
SAVE_PATH = os.environ.get("IMAGES_SAVE_PATH", "data/images")

# Ensure the images directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

async def check_auto1111_status() -> bool:
    """Check if Automatic1111 is running and available"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{AUTOMATIC1111_URL}/sdapi/v1/sd-models", timeout=5) as response:
                if response.status == 200:
                    return True
                else:
                    logging.warning(f"Automatic1111 responded with status code {response.status}")
                    return False
    except Exception as e:
        logging.warning(f"Automatic1111 check failed: {str(e)}")
        return False

async def get_auto1111_models() -> list:
    """Get list of available models in Automatic1111"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{AUTOMATIC1111_URL}/sdapi/v1/sd-models") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"Failed to get models. Status: {response.status}")
                    return []
    except Exception as e:
        logging.error(f"Error getting Automatic1111 models: {str(e)}")
        return []

async def get_auto1111_samplers() -> list:
    """Get list of available samplers in Automatic1111"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{AUTOMATIC1111_URL}/sdapi/v1/samplers") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"Failed to get samplers. Status: {response.status}")
                    return []
    except Exception as e:
        logging.error(f"Error getting Automatic1111 samplers: {str(e)}")
        return []

async def set_auto1111_model(model_name: str) -> bool:
    """Set the active model in Automatic1111"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "sd_model_checkpoint": model_name
            }
            async with session.post(
                f"{AUTOMATIC1111_URL}/sdapi/v1/options", 
                json=payload
            ) as response:
                return response.status == 200
    except Exception as e:
        logging.error(f"Error setting Automatic1111 model: {str(e)}")
        return False

async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    guidance_scale: float = 7.0,
    sampler: str = "Euler a",
    seed: int = -1,
    model: Optional[str] = None
) -> str:
    """Generate an image using Automatic1111 API"""
    try:
        # Switch model if specified
        if model:
            await set_auto1111_model(model)
        
        # Prepare payload
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": guidance_scale,
            "sampler_name": sampler,
            "seed": seed if seed >= 0 else int(time.time() * 1000) % 4294967295,
            "batch_size": 1,
            "save_images": True
        }
        
        async with aiohttp.ClientSession() as session:
            # Generate the image
            async with session.post(
                f"{AUTOMATIC1111_URL}/sdapi/v1/txt2img",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Image generation failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                
                # Handle the image data
                for i, image_data_b64 in enumerate(response_data['images']):
                    # Save the image
                    image_data = base64.b64decode(image_data_b64)
                    timestamp = int(time.time())
                    file_uuid = uuid.uuid4().hex[:8]
                    filename = f"{timestamp}_{file_uuid}.png"
                    file_path = os.path.join(SAVE_PATH, filename)
                    
                    with open(file_path, "wb") as image_file:
                        image_file.write(image_data)
                    
                    logging.info(f"Image saved to {file_path}")
                    
                    # For now, just return the first image
                    if i == 0:
                        return file_path
                
                raise Exception("No images were generated")
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        raise e