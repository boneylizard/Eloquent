import json
import os
from pathlib import Path
from typing import Any


MODULE_DEFAULTS = {
    "chess": False,
    "elections": False,
    "market": False,
    "chatlog_condenser": False,
    "voice": True,
}

RETIRED_MODULES = {"chess", "chatlog_condenser", "code_editor", "forensics", "market", "watch"}


def _settings() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in (
        Path.home() / ".LiangLocal" / "settings.json",
        Path.home() / ".mirid" / "settings.json",
    ):
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                merged.update(value)
        except (OSError, ValueError):
            continue
    return merged


def module_enabled(module_id: str) -> bool:
    if module_id in RETIRED_MODULES:
        return False

    available = os.environ.get("MIRID_AVAILABLE_MODULES")
    if available is not None:
        available_modules = {value.strip() for value in available.split(",") if value.strip()}
        if module_id not in available_modules:
            return False

    enabled_from_env = {
        value.strip()
        for value in os.environ.get("MIRID_ENABLED_MODULES", "").split(",")
        if value.strip()
    }
    if module_id in enabled_from_env:
        return True

    configured = _settings().get("modules", {})
    if isinstance(configured, dict) and isinstance(configured.get(module_id), bool):
        return configured[module_id]

    return MODULE_DEFAULTS.get(module_id, True)
