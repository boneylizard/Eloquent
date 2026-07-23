# user_utils.py
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .settings_store import update_settings as update_settings_file

logger = logging.getLogger("user_utils")

_MEMORY_STORE_SUFFIX = "_memory_store.json"


def _safe_profile_id_segment(raw: Optional[str]) -> Optional[str]:
    """Sanitize profile/user id for on-disk filenames (matches memory_intelligence rules)."""
    if not raw or not isinstance(raw, str):
        return None
    safe = "".join(c for c in raw if c.isalnum() or c in ("-", "_"))
    return safe or None


def infer_profile_id_from_largest_memory_store() -> Optional[str]:
    """
    Pick profile_id from the largest *_memory_store.json in user_memories.
    Uses file size only; does not read memory JSON contents.
    """
    try:
        profiles_dir = get_profiles_directory()
        candidates: List[tuple] = []
        for name in os.listdir(profiles_dir):
            full_path = profiles_dir / name
            if full_path.is_dir():
                continue
            if not name.endswith(_MEMORY_STORE_SUFFIX):
                continue
            inferred_id = name[: -len(_MEMORY_STORE_SUFFIX)]
            if not inferred_id:
                continue
            try:
                size = full_path.stat().st_size
            except OSError:
                size = 0
            candidates.append((size, inferred_id))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception as e:
        logger.error(f"Error inferring profile id from memory stores: {e}")
        return None


# Define a function to find the user profiles directory
def get_profiles_directory():
    # Base directory is where this module exists
    base_dir = Path(__file__).parent
    profiles_dir = base_dir / "user_memories"  # CHANGED from "user_profiles" to "user_memories"
    profiles_dir.mkdir(exist_ok=True)
    return profiles_dir


def list_profile_ids() -> List[str]:
    """
    List available profile IDs based on backend memory store filenames.
    This does NOT read/parse any user memory contents.
    """
    try:
        profiles_dir = get_profiles_directory()
        suffix = _MEMORY_STORE_SUFFIX
        ids: List[str] = []
        for name in os.listdir(profiles_dir):
            p = profiles_dir / name
            if p.is_dir():
                continue
            if not name.endswith(suffix):
                continue
            pid = name[:-len(suffix)]
            if pid:
                ids.append(pid)
        # Stable ordering for UI
        ids.sort()
        return ids
    except Exception as e:
        logger.error(f"Error listing profile IDs: {e}")
        return []

# Function to get the active profile ID from settings
def get_active_profile_id():
    """
    Return the active profile ID from the settings file if present.
    If not set, try to infer it from existing user memory files in backend/app/user_memories
    (filenames like '<user_id>_memory_store.json'), persist it to settings, and return it.
    """
    try:
        settings_file = Path.home() / ".LiangLocal" / "settings.json"
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                active_id = settings.get("activeProfileId")
                if active_id:
                    return active_id
    except Exception as e:
        logger.error(f"Error reading active profile ID: {e}")

    # Try to infer from existing memory store files if no activeProfileId is configured.
    inferred_id = infer_profile_id_from_largest_memory_store()
    if inferred_id:
        logger.info(f"Inferred active profile ID from largest memory store: {inferred_id}")
        try:
            save_active_profile_id(inferred_id)
        except Exception as se:
            logger.error(f"Failed to persist inferred active profile ID: {se}")
        return inferred_id

    # Return None if nothing could be determined
    return None

# Function to load a profile by ID, respecting format preferences
def load_profile(profile_id: str = None):
    if not profile_id:
        profile_id = get_active_profile_id()
        if not profile_id:
            logger.warning("No active profile ID found")
            return None
    
    profiles_dir = get_profiles_directory()
    
    # Try loading profile in different formats
    for ext in ['.json', '.yaml', '.yml', '.toml']:
        profile_path = profiles_dir / f"{profile_id}{ext}"
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    if ext == '.json':
                        return json.load(f)
                    elif ext in ['.yaml', '.yml']:
                        return yaml.safe_load(f)
                    elif ext == '.toml':
                        import toml
                        return toml.load(f)
            except Exception as e:
                logger.error(f"Error loading profile {profile_path}: {e}")
    
    # If we get here, no profile was found
    logger.warning(f"No profile found for ID: {profile_id}")
    return None

# Function to get the directory containing user memory files
def get_memory_directory():
    return get_profiles_directory() / "memories"

# Ensure memory directory exists
def ensure_memory_directory():
    memory_dir = get_memory_directory()
    memory_dir.mkdir(exist_ok=True)
    return memory_dir
def clear_active_profile_id() -> bool:
    """Remove activeProfileId from settings (no memory file reads)."""
    try:
        update_settings_file({"activeProfileId": None})
        logger.info("Cleared activeProfileId from settings")
        return True
    except Exception as e:
        logger.error(f"Error clearing active profile ID: {e}")
        return False


def delete_user_profile_storage(profile_id: str) -> Dict[str, Any]:
    """
    Delete on-disk artifacts for a profile: main memory store, optional legacy
    profile metadata files in the same folder, and agentic JSON files for this user.
    Uses filenames and os.remove only — does not read memory JSON contents.
    """
    safe = _safe_profile_id_segment(profile_id)
    if not safe:
        return {"status": "error", "reason": "invalid_profile_id"}

    from . import agentic_memory

    profiles_dir = get_profiles_directory()
    removed: List[str] = []

    mem = profiles_dir / f"{safe}{_MEMORY_STORE_SUFFIX}"
    tmp = profiles_dir / f"{safe}{_MEMORY_STORE_SUFFIX}.tmp"
    for path in (mem, tmp):
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(str(path))
        except OSError as e:
            return {"status": "error", "reason": f"remove_failed:{path.name}:{e}"}

    # Legacy optional profile document: {id}.json / yaml / toml (not the memory_store)
    for ext in (".json", ".yaml", ".yml", ".toml"):
        side = profiles_dir / f"{safe}{ext}"
        if side == mem:
            continue
        try:
            if side.exists() and side.is_file():
                side.unlink()
                removed.append(str(side))
        except OSError as e:
            logger.warning(f"Could not remove legacy profile file {side}: {e}")

    agentic_removed = 0
    try:
        agentic_removed = agentic_memory.delete_all_agentic_files_for_user(profile_id)
    except Exception as e:
        logger.warning(f"Agentic file cleanup for {safe!r}: {e}")

    had_store = str(mem) in removed
    return {
        "status": "success",
        "safe_profile_id": safe,
        "removed_memory_store": had_store,
        "removed_paths": removed,
        "agentic_files_removed": agentic_removed,
        "nothing_removed": not had_store and agentic_removed == 0,
    }


def save_active_profile_id(profile_id):
    """Save active profile ID to settings file."""
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        logger.info(f"📝 Saving active profile ID: {profile_id} to {settings_path}")
        update_settings_file({"activeProfileId": profile_id})
        logger.info(f"✅ Successfully saved active profile ID to settings")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving active profile ID: {e}")
        logger.exception(e)  # Print full traceback
        return False
