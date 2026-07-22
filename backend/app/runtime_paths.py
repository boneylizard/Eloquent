from __future__ import annotations

import os
import sys
from pathlib import Path


def runtime_data_root() -> Path:
    configured = os.environ.get("MIRID_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "ai.mirid.desktop" / "data"
        return Path.home() / ".mirid" / "data"

    return Path(__file__).resolve().parent.parent / "data"


def data_path(*parts: str) -> Path:
    return runtime_data_root().joinpath(*parts)
