"""
Profile Manager — stores and serves agentic pipeline profiles.

Each profile is a JSON file containing:
  - id, name, description
  - labels: display names for all UI-facing strings
  - prompts: the 4 prompt templates (analysis, somatic, directive, planning)
  - display_config: gauge/chip color and label configuration

Profiles live at: backend/app/user_memories/sanctuary/_profiles/<id>.json
The _default.json profile mirrors the original hardcoded sanctuary values.
"""

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sanctuary.profile_manager")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_USER_MEMORY_DIR = os.path.join(_CURRENT_DIR, "..", "user_memories")
_PROFILES_DIR = os.path.join(_USER_MEMORY_DIR, "sanctuary", "_profiles")

try:
    os.makedirs(_PROFILES_DIR, exist_ok=True)
except Exception as exc:
    logger.warning("profile_manager: could not create profiles dir: %s", exc)
    _PROFILES_DIR = os.path.join(os.getcwd(), "app", "user_memories", "sanctuary", "_profiles")
    os.makedirs(_PROFILES_DIR, exist_ok=True)

_DEFAULT_LABELS = {
    "LABEL_TACTILE_OUTREACH": "tactile_outreach",
    "LABEL_CHARACTER_SIGNAL": "character_signal",
    "LABEL_TACTILE_DISPLAY": "Tactile Outreach",
    "LABEL_SIGNAL_DISPLAY": "Character Signal",
    "LABEL_POSE": "pose",
    "LABEL_GESTURE": "gesture_narrative",
    "LABEL_PROXIMITY": "proximity",
    "LABEL_COVERT_ACTION": "covert_action",
    "LABEL_VOICE_THIS_TURN": "voice_this_turn",
    "dashboard_lubrication": "Lubrication",
    "dashboard_pupils": "Pupils",
    "dashboard_position": "Position",
    "dashboard_breath": "Breath",
    "dashboard_tension": "Tension",
    "heat_gauge": "Heat Index",
    "dominance_gauge": "Dominance",
    "trap_gauge": "Trap Progress",
    "posture_label": "Posture",
}

_DEFAULT_GAUGE_CONFIG = {
    "heat": {
        "label": "Heat Index",
        "min": 0.0,
        "max": 1.0,
        "color": "rgba(255, 130, 100, 0.8)",
        "unit": "",
    },
    "dominance": {
        "label": "Dominance",
        "min": 0.0,
        "max": 1.0,
        "color": "rgba(120, 180, 255, 0.8)",
        "unit": "",
    },
    "trap": {
        "label": "Trap Progress",
        "min": 0.0,
        "max": 1.0,
        "color": "rgba(255, 120, 200, 0.8)",
        "unit": "",
    },
}

_DEFAULT_DASHBOARD_CHIPS = {
    "lubrication_level": {
        "label": "Lubrication",
        "color": "rgba(255, 120, 200, 0.8)",
        "type": "enum",
        "enum_values": ["dry", "damp", "slick", "drenched"],
    },
    "pupil_dilation": {
        "label": "Pupils",
        "color": "rgba(120, 200, 255, 0.8)",
        "type": "percentage",
    },
    "spatial_position": {
        "label": "Position",
        "color": "rgba(100, 220, 150, 0.8)",
        "type": "enum",
        "enum_values": ["across_room", "nearby", "touching", "entangled", "pinned"],
    },
    "breath_rate": {
        "label": "Breath",
        "color": "rgba(255, 220, 120, 0.8)",
        "type": "enum",
        "enum_values": ["slow", "steady", "quick", "ragged"],
    },
    "muscle_tension": {
        "label": "Tension",
        "color": "rgba(255, 130, 100, 0.8)",
        "type": "percentage",
    },
}

_DEFAULT_PROFILE = {
    "id": "_default",
    "name": "Default Sanctuary",
    "description": "The original sanctuary agentic pipeline with classic BDSM-themed dynamics",
    "labels": _DEFAULT_LABELS,
    "display_config": {
        "gauges": _DEFAULT_GAUGE_CONFIG,
        "dashboard_chips": _DEFAULT_DASHBOARD_CHIPS,
    },
    "prompts": {
        "contextual_analysis": None,
        "somatic_generation": None,
        "directive_block_template": None,
        "planning_generation": None,
        "planning_injection_template": None,
    },
}


def _safe_id(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return "unknown"
    return "".join(c for c in raw if c.isalnum() or c in ("-", "_")) or "unknown"


def _profile_path(profile_id: str) -> str:
    return os.path.join(_PROFILES_DIR, f"{_safe_id(profile_id)}.json")


def get_default_profile() -> Dict[str, Any]:
    """Return a deep copy of the default profile."""
    return json.loads(json.dumps(_DEFAULT_PROFILE))


def load_profile(profile_id: str) -> Dict[str, Any]:
    """Load a profile by ID. Falls back to _default if not found."""
    if not profile_id or profile_id == "_default":
        return get_default_profile()

    path = _profile_path(profile_id)
    if not os.path.exists(path):
        logger.warning("profile_manager: profile '%s' not found at %s, returning default", profile_id, path)
        return get_default_profile()

    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        profile["id"] = profile_id
        return profile
    except Exception as exc:
        logger.error("profile_manager: failed to load profile '%s': %s", profile_id, exc)
        return get_default_profile()


def save_profile(profile_id: str, data: Dict[str, Any]) -> bool:
    """Save a profile. profile_id must not be '_default'."""
    safe_id = _safe_id(profile_id)
    if safe_id == "_default":
        logger.warning("profile_manager: cannot overwrite _default profile")
        return False

    path = _profile_path(safe_id)
    try:
        to_save = dict(data)
        to_save["id"] = safe_id
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        logger.info("profile_manager: saved profile '%s' to %s", safe_id, path)
        return True
    except Exception as exc:
        logger.error("profile_manager: failed to save profile '%s': %s", safe_id, exc)
        return False


def delete_profile(profile_id: str) -> bool:
    """Delete a custom profile. Cannot delete _default."""
    safe_id = _safe_id(profile_id)
    if safe_id == "_default":
        logger.warning("profile_manager: cannot delete _default profile")
        return False

    path = _profile_path(safe_id)
    if not os.path.exists(path):
        logger.warning("profile_manager: profile '%s' not found for deletion", safe_id)
        return False

    try:
        os.remove(path)
        logger.info("profile_manager: deleted profile '%s'", safe_id)
        return True
    except Exception as exc:
        logger.error("profile_manager: failed to delete profile '%s': %s", safe_id, exc)
        return False


def list_profiles() -> List[Dict[str, Any]]:
    """List all available profiles (including _default)."""
    profiles = []
    default = get_default_profile()
    profiles.append({
        "id": default["id"],
        "name": default["name"],
        "description": default["description"],
    })

    if not os.path.exists(_PROFILES_DIR):
        return profiles

    for fname in sorted(os.listdir(_PROFILES_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(_PROFILES_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profiles.append({
                "id": data.get("id", fname[:-5]),
                "name": data.get("name", fname[:-5]),
                "description": data.get("description", ""),
            })
        except Exception as exc:
            logger.warning("profile_manager: failed to read %s: %s", fname, exc)

    return profiles


def resolve_profile_labels(profile: Dict[str, Any]) -> Dict[str, str]:
    """Get the labels dict from a profile, falling back to defaults."""
    labels = dict(_DEFAULT_LABELS)
    profile_labels = profile.get("labels", {})
    if isinstance(profile_labels, dict):
        labels.update(profile_labels)
    return labels


def resolve_profile_prompt(profile: Dict[str, Any], prompt_key: str) -> Optional[str]:
    """Get a specific prompt template from a profile. Falls back to None (meaning use hardcoded)."""
    prompts = profile.get("prompts", {})
    if not isinstance(prompts, dict):
        return None
    return prompts.get(prompt_key)
