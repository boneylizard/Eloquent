"""Persisted D-ID avatar/background presets (URLs only)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal

from .runtime_paths import data_path

Kind = Literal["avatar", "background"]


def _path() -> Path:
    p = data_path("d_id_saved_assets.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_assets() -> List[Dict[str, Any]]:
    p = _path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_assets(items: List[Dict[str, Any]]) -> None:
    _path().write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_assets(kind: Kind | None = None) -> List[Dict[str, Any]]:
    items = load_assets()
    if kind:
        items = [x for x in items if x.get("kind") == kind]
    return sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)


def add_asset(*, kind: Kind, label: str, url: str) -> Dict[str, Any]:
    items = load_assets()
    entry = {
        "id": uuid.uuid4().hex[:16],
        "kind": kind,
        "label": (label or "").strip() or "Untitled",
        "url": url.strip(),
        "created_at": int(time.time()),
    }
    items.insert(0, entry)
    save_assets(items)
    return entry


def delete_asset(asset_id: str) -> bool:
    items = load_assets()
    new = [x for x in items if x.get("id") != asset_id]
    if len(new) == len(items):
        return False
    save_assets(new)
    return True
