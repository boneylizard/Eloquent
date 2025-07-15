# user_utils.py
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("user_utils")

# Define a function to find the user profiles directory
def get_profiles_directory():
    # Base directory is where this module exists
    base_dir = Path(__file__).parent
    profiles_dir = base_dir / "user_memories"  # CHANGED from "user_profiles" to "user_memories"
    profiles_dir.mkdir(exist_ok=True)
    return profiles_dir

# Function to get the active profile ID from settings
def get_active_profile_id():
    try:
        settings_file = Path.home() / ".LiangLocal" / "settings.json"
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                return settings.get("activeProfileId")
    except Exception as e:
        logger.error(f"Error reading active profile ID: {e}")
    
    # Return None if not found or error
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
def save_active_profile_id(profile_id):
    """Save active profile ID to settings file."""
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        settings_dir = settings_path.parent
        
        logger.info(f"📝 Saving active profile ID: {profile_id} to {settings_path}")
        
        settings_dir.mkdir(exist_ok=True)
        
        # Read existing settings
        settings = {}
        if settings_path.exists():
            logger.info(f"📖 Reading existing settings from {settings_path}")
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                logger.info(f"📖 Current settings: {settings}")
        else:
            logger.info(f"📝 Settings file doesn't exist, creating new one")
        
        # Update with new profile ID
        previous_id = settings.get('activeProfileId')
        settings['activeProfileId'] = profile_id
        
        logger.info(f"🔄 Updating activeProfileId: {previous_id} -> {profile_id}")
        
        # Write back
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
            
        logger.info(f"✅ Successfully saved active profile ID to settings")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving active profile ID: {e}")
        logger.exception(e)  # Print full traceback
        return False