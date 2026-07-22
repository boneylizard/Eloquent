import json
import shutil
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
GALLERY_DIR = STATIC_DIR / "room_gallery"
GENERATED_DIR = STATIC_DIR / "generated_images"
THUMBNAILS_DIR = GALLERY_DIR / "thumbnails"
MANIFEST_FILE = GALLERY_DIR / "gallery_manifest.json"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MANIFEST = {
    "version": 1,
    "categories": [
        {"id": "builtin_apocalyptic", "name": "Apocalyptic", "builtin": True, "sort_order": 0},
        {"id": "builtin_cyberpunk",  "name": "Cyberpunk",  "builtin": True, "sort_order": 1},
        {"id": "builtin_fantasy",    "name": "Fantasy",    "builtin": True, "sort_order": 2},
        {"id": "builtin_horror",     "name": "Horror",     "builtin": True, "sort_order": 3},
        {"id": "builtin_modern",     "name": "Modern",     "builtin": True, "sort_order": 4},
        {"id": "builtin_scifi",      "name": "SciFi",      "builtin": True, "sort_order": 5},
    ],
    "images": []
}

def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        return dict(DEFAULT_MANIFEST)
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load gallery manifest: {e}")
        return dict(DEFAULT_MANIFEST)

def save_manifest(manifest: dict) -> None:
    tmp = MANIFEST_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        tmp.replace(MANIFEST_FILE)
    except IOError as e:
        logger.error(f"Failed to save gallery manifest: {e}")
        raise HTTPException(status_code=500, detail="Failed to save gallery manifest")

def generate_thumbnail(source_path: Path) -> str:
    thumb_filename = f"{source_path.stem}_thumb.jpg"
    thumb_path = THUMBNAILS_DIR / thumb_filename
    if thumb_path.exists():
        return f"/static/room_gallery/thumbnails/{thumb_filename}"
    try:
        img = Image.open(source_path)
        img.thumbnail((256, 256))
        img = img.convert("RGB")
        img.save(thumb_path, "JPEG", quality=85)
    except Exception as e:
        logger.warning(f"Thumbnail generation failed for {source_path.name}: {e}")
        return None
    return f"/static/room_gallery/thumbnails/{thumb_filename}"

ALLOWED_GEN_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

@router.get("/room-gallery/images")
async def list_images(search: str = "", category_id: str = "", source: str = ""):
    manifest = load_manifest()
    images = list(manifest.get("images", []))

    if search:
        q = search.lower()
        images = [
            i for i in images
            if q in (i.get("display_name") or "").lower()
            or any(q in (t or "").lower() for t in i.get("tags") or [])
        ]
    if category_id:
        images = [i for i in images if i.get("category_id") == category_id]
    if source:
        images = [i for i in images if i.get("source") == source]

    return {"images": images, "total": len(images)}

@router.get("/room-gallery/categories")
async def list_categories():
    manifest = load_manifest()
    images = manifest.get("images", [])
    total = len(images)

    categories = [{"id": "all", "name": "All", "count": total, "builtin": True}]
    for cat in manifest.get("categories", []):
        count = sum(1 for img in images if img.get("category_id") == cat["id"])
        categories.append({**cat, "count": count})

    return {"categories": categories}

@router.post("/room-gallery/save-from-generation")
async def save_from_generation(body: dict):
    image_url = (body.get("image_url") or "").strip()
    if not image_url:
        raise HTTPException(400, "image_url is required")

    if not image_url.startswith("/static/generated_images/"):
        raise HTTPException(400, "Only /static/generated_images/* paths are accepted")

    filename = Path(image_url).name
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_GEN_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    source_path = GENERATED_DIR / filename
    if not source_path.exists():
        raise HTTPException(404, f"Source image not found on disk: {filename}")

    new_filename = f"generated_{uuid.uuid4().hex}{ext}"
    dest_path = GALLERY_DIR / new_filename
    shutil.copy2(source_path, dest_path)

    thumb_url = generate_thumbnail(dest_path)

    params = body.get("parameters") or {}
    display_name = body.get("display_name")
    if not display_name:
        display_name = (params.get("prompt") or "Untitled")[:80]

    entry = {
        "id": f"img_{uuid.uuid4().hex[:16]}",
        "filename": new_filename,
        "path": f"/static/room_gallery/{new_filename}",
        "thumbnail_path": thumb_url,
        "display_name": display_name,
        "category_id": body.get("category_id"),
        "tags": body.get("tags") or [],
        "source": "generated",
        "width": params.get("width"),
        "height": params.get("height"),
        "file_size": dest_path.stat().st_size,
        "format": ext.lstrip("."),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "favorite": False,
        "generation_params": params,
    }

    manifest = load_manifest()
    manifest.setdefault("images", []).append(entry)
    save_manifest(manifest)

    return {"status": "success", "image": entry}
