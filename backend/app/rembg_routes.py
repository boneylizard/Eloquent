from fastapi import APIRouter
from typing import Optional
import requests
from PIL import Image
import rembg
import io
import logging
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

def get_local_file_path(image_url: str) -> Optional[Path]:
    if '/generated_images/' in image_url:
        parts = image_url.split('/')
        if 'generated_images' in parts:
            idx = parts.index('generated_images')
            filename = parts[idx + 1]
            return Path('static') / 'generated_images' / filename
    return None

def save_image_and_get_url(image_data: bytes) -> str:
    generated_images_dir = Path('static') / 'generated_images'
    generated_images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    save_path = generated_images_dir / filename
    with open(save_path, "wb") as f:
        f.write(image_data)
    return f"/static/generated_images/{filename}"

@router.post("/remove-background")
async def remove_background(body: dict):
    """
    Remove background from an image and return the processed image URL with bounding box.

    Input: { "image_url": str, "padding": int (default 10) }
    Output: { "image_url": str, "bounding_box": { "x1": int, "y1": int, "x2": int, "y2": int } }
    """
    image_url = body.get("image_url")
    padding = body.get("padding", 10)

    if not image_url:
        raise ValueError("image_url is required")

    try:
        local_path = get_local_file_path(image_url)
        if local_path and local_path.exists():
            img = Image.open(local_path)
        else:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))

        output = rembg.remove(img)
        bbox = output.getbbox()

        bbox_dict = None
        if bbox:
            left, top, right, bottom = bbox
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(output.width, right + padding)
            bottom = min(output.height, bottom + padding)
            bbox_dict = {
                "x1": int(left),
                "y1": int(top),
                "x2": int(right),
                "y2": int(bottom)
            }
        else:
            bbox_dict = {
                "x1": 0,
                "y1": 0,
                "x2": output.width,
                "y2": output.height
            }

        img_bytes = io.BytesIO()
        output.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        final_url = save_image_and_get_url(img_bytes.getvalue())

        return {
            "image_url": final_url,
            "bounding_box": bbox_dict,
            "width": output.width,
            "height": output.height
        }

    except Exception as e:
        logger.error(f"Failed to remove background for {image_url}: {e}", exc_info=True)
        raise e

@router.get("/status")
async def status():
    try:
        import rembg
        return {"available": True}
    except ImportError:
        return {"available": False}