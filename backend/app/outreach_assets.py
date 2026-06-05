"""
Per-rule image pools for scheduled outreach notifications.
"""
from __future__ import annotations

import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ASSETS_ROOT = Path(__file__).resolve().parent.parent / "data" / "outreach_assets"
_STATIC_OUTREACH = Path(__file__).resolve().parent / "static" / "outreach_runtime"
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def prune_orphan_asset_dirs(active_rule_ids: List[str]) -> None:
    if not _ASSETS_ROOT.is_dir():
        return
    active = {"".join(c if c.isalnum() or c in "-_" else "_" for c in (rid or "")) for rid in active_rule_ids}
    for child in _ASSETS_ROOT.iterdir():
        if child.is_dir() and child.name not in active:
            shutil.rmtree(child, ignore_errors=True)


def assets_dir_for_rule(rule_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (rule_id or "unknown"))
    return _ASSETS_ROOT / safe


def list_rule_image_files(rule_id: str) -> List[Path]:
    folder = assets_dir_for_rule(rule_id)
    if not folder.is_dir():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIXES and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.name.lower())


def rule_image_count(rule_id: str) -> int:
    return len(list_rule_image_files(rule_id))


def clear_rule_images(rule_id: str) -> None:
    folder = assets_dir_for_rule(rule_id)
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def save_uploaded_images(rule_id: str, filenames_and_bytes: List[Tuple[str, bytes]], replace: bool = True) -> int:
    folder = assets_dir_for_rule(rule_id)
    if replace and folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for name, data in filenames_and_bytes:
        if not data:
            continue
        ext = Path(name).suffix.lower()
        if ext not in _ALLOWED_SUFFIXES:
            continue
        dest = folder / f"{uuid.uuid4().hex[:12]}{ext}"
        dest.write_bytes(data)
        saved += 1
    return saved


def pick_random_image_path(rule_id: str) -> Optional[Path]:
    files = list_rule_image_files(rule_id)
    if not files:
        return None
    return random.choice(files)


def publish_runtime_image(rule_id: str, source: Path) -> str:
    """
    Copy a pool image into static/outreach_runtime for HTTP serving.
    Returns URL path starting with /static/...
    """
    _STATIC_OUTREACH.mkdir(parents=True, exist_ok=True)
    ext = source.suffix.lower() or ".jpg"
    dest_name = f"{rule_id[:24]}_{uuid.uuid4().hex[:10]}{ext}"
    dest = _STATIC_OUTREACH / dest_name
    shutil.copy2(source, dest)
    return f"/static/outreach_runtime/{dest_name}"


def build_image_message(
    rule_id: str,
    character: Dict[str, Any],
    *,
    base_url: str = "http://127.0.0.1:8000",
) -> Optional[Dict[str, Any]]:
    src = pick_random_image_path(rule_id)
    if not src:
        return None
    rel = publish_runtime_image(rule_id, src)
    full = f"{base_url.rstrip('/')}{rel}" if rel.startswith("/") else rel
    char_id = character.get("id")
    return {
        "id": f"img-{uuid.uuid4().hex[:12]}",
        "role": "bot",
        "type": "image",
        "content": "Shared a photo with you.",
        "imagePath": full,
        "prompt": "Outreach attachment",
        "characterId": char_id,
        "characterName": character.get("name"),
        "avatar": character.get("avatar"),
        "isScheduledOutreach": True,
        "outreachAttachment": True,
    }
