"""
Shadow State — latent context layer for the agentic pipeline.

Stores per-(user_id, character_id) state variables that persist across turns
without polluting the primary chat transcript.

Storage: JSON files under ``backend/app/user_memories/sanctuary/``.
Trajectory summaries: human-readable state snapshots generated every N turns,
stored as ``.summary.txt`` files alongside the state JSON. These can be fed
into the document RAG system for semantic cross-session recall.

The state is transported to/from the frontend via cipher blocks (cipher.py).
"""

import json
import logging
import os
import datetime
from typing import Any, Dict, List, Optional

from . import cipher

logger = logging.getLogger("sanctuary.shadow_state")

_SUMMARY_INTERVAL = 5  # generate a trajectory summary every N turns

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_USER_MEMORY_DIR = os.path.join(_CURRENT_DIR, "..", "user_memories")
_SANCTUARY_DIR = os.path.join(_USER_MEMORY_DIR, "sanctuary")

try:
    os.makedirs(_SANCTUARY_DIR, exist_ok=True)
except Exception as exc:
    logger.warning("shadow_state: could not create sanctuary dir: %s", exc)
    _SANCTUARY_DIR = os.path.join(os.getcwd(), "app", "user_memories", "sanctuary")
    os.makedirs(_SANCTUARY_DIR, exist_ok=True)

_DEFAULT_STATE: Dict[str, Any] = {
    "version": 1,
    "user_id": "",
    "character_id": "",
    "turn_count": 0,
    "heat_index": 0.0,
    "dominance_vector": 0.5,
    "trap_progress": 0.0,
    "alignment_markers": {
        "fidelity": 0.5,
        "resistance": 0.0,
        "compliance": 0.0,
    },
    "posture": "neutral",
    "ghost_signal_active": False,
    "last_updated": "",
}


def _safe_id(raw: Optional[str]) -> str:
    if not raw or not isinstance(raw, str):
        return "unknown"
    return "".join(c for c in raw if c.isalnum() or c in ("-", "_")) or "unknown"


def _state_path(user_id: str, character_id: str) -> str:
    user_dir = os.path.join(_SANCTUARY_DIR, _safe_id(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f"{_safe_id(character_id)}.json")


def default_state(user_id: str = "", character_id: str = "") -> Dict[str, Any]:
    """Return a fresh default shadow state."""
    state = dict(_DEFAULT_STATE)
    state["user_id"] = user_id
    state["character_id"] = character_id
    state["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return state


def load(
    user_id: str,
    character_id: str,
    cipher_block: Optional[str] = None,
) -> Dict[str, Any]:
    """Load shadow state for a (user_id, character_id) pair.

    Resolution order:
      1. If ``cipher_block`` is provided, decode and merge over stored state
         (the client's cipher is the source of truth for the current turn).
      2. If a state JSON file exists on disk, load it.
      3. If trajectory logs exist (cross-session recall), reconstruct from latest.
      4. Otherwise, return a fresh default.
    """
    stored = _load_from_disk(user_id, character_id)

    if cipher_block:
        decoded = cipher.decode(cipher_block)
        if decoded:
            if stored:
                merged = {**stored, **decoded}
            else:
                merged = decoded
            merged["user_id"] = user_id
            merged["character_id"] = character_id
            logger.info(
                "shadow_state.load: merged cipher over stored state for (%s, %s)",
                _safe_id(user_id), _safe_id(character_id),
            )
            return merged
        else:
            logger.warning("shadow_state.load: cipher_block provided but failed to decode")

    if stored:
        stored["user_id"] = user_id
        stored["character_id"] = character_id
        return stored

    # Cross-session recall from trajectory log
    recalled = recall_from_trajectory(user_id, character_id)
    if recalled:
        logger.info(
            "shadow_state.load: recovered state via trajectory recall for (%s, %s)",
            _safe_id(user_id), _safe_id(character_id),
        )
        return recalled

    logger.info(
        "shadow_state.load: no stored state for (%s, %s), returning default",
        _safe_id(user_id), _safe_id(character_id),
    )
    return default_state(user_id, character_id)


def save(user_id: str, character_id: str, state: Dict[str, Any]) -> bool:
    """Persist shadow state to disk and generate trajectory summary if due."""
    path = _state_path(user_id, character_id)
    try:
        state_to_save = dict(state)
        state_to_save["user_id"] = user_id
        state_to_save["character_id"] = character_id
        state_to_save["last_updated"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=2)
        logger.debug("shadow_state.save: wrote %s", path)

        # Generate trajectory summary every N turns
        generate_trajectory_summary(user_id, character_id, state_to_save)

        return True
    except Exception as exc:
        logger.error("shadow_state.save: failed to write %s: %s", path, exc)
        return False


def encode_cipher(state: Dict[str, Any]) -> str:
    """Encode a shadow state into a cipher block for transport."""
    return cipher.encode(state)


def merge_cipher(
    stored: Dict[str, Any],
    cipher_block: str,
) -> Dict[str, Any]:
    """Merge a cipher block's decoded state over a stored state."""
    decoded = cipher.decode(cipher_block)
    if not decoded:
        return stored
    merged = {**stored, **decoded}
    return merged


def _load_from_disk(user_id: str, character_id: str) -> Optional[Dict[str, Any]]:
    """Load state from JSON file, or None if not found."""
    path = _state_path(user_id, character_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("shadow_state._load_from_disk: failed to read %s: %s", path, exc)
        return None


def apply_analysis_result(
    state: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a Step 1 analysis result to the shadow state.

    Returns a new state dict (does not mutate the input).
    """
    new_state = dict(state)
    new_state["turn_count"] = int(state.get("turn_count", 0)) + 1
    new_state["dominance_vector"] = float(analysis.get("dominance_vector", state.get("dominance_vector", 0.5)))
    new_state["heat_index"] = max(0.0, min(1.0, float(state.get("heat_index", 0.0)) + float(analysis.get("heat_index_delta", 0.0))))
    new_state["trap_progress"] = max(0.0, min(1.0, float(state.get("trap_progress", 0.0)) + float(analysis.get("trap_progress_delta", 0.0))))
    new_state["posture"] = analysis.get("posture", state.get("posture", "neutral"))
    new_state["ghost_signal_active"] = bool(analysis.get("ghost_signal_active", False))

    alignment = analysis.get("alignment_update", {})
    if isinstance(alignment, dict) and alignment:
        current_markers = dict(state.get("alignment_markers", {}))
        for key in ("fidelity", "resistance", "compliance"):
            if key in alignment:
                current_markers[key] = max(0.0, min(1.0, float(alignment[key])))
        new_state["alignment_markers"] = current_markers

    return new_state


# --- Trajectory Summary (Phase 2: RAG persistence) ---

def _summary_path(user_id: str, character_id: str) -> str:
    """Path for the trajectory summary file."""
    user_dir = os.path.join(_SANCTUARY_DIR, _safe_id(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f"{_safe_id(character_id)}.summary.txt")


def _trajectory_log_path(user_id: str, character_id: str) -> str:
    """Path for the append-only trajectory log (one line per summary)."""
    user_dir = os.path.join(_SANCTUARY_DIR, _safe_id(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f"{_safe_id(character_id)}.trajectory.jsonl")


def generate_trajectory_summary(
    user_id: str,
    character_id: str,
    state: Dict[str, Any],
) -> Optional[str]:
    """Generate and persist a human-readable trajectory summary.

    Called every ``_SUMMARY_INTERVAL`` turns. The summary is a compact
    text block that captures the current state arc, suitable for RAG indexing.

    Returns the summary text, or None if not generated this turn.
    """
    turn_count = int(state.get("turn_count", 0))
    if turn_count == 0 or turn_count % _SUMMARY_INTERVAL != 0:
        return None

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    heat = float(state.get("heat_index", 0.0))
    dominance = float(state.get("dominance_vector", 0.5))
    trap = float(state.get("trap_progress", 0.0))
    posture = state.get("posture", "neutral")
    markers = state.get("alignment_markers", {})

    heat_label = "cold" if heat < 0.25 else "warm" if heat < 0.5 else "hot" if heat < 0.75 else "blazing"
    dom_label = "yielding" if dominance < 0.3 else "balanced" if dominance < 0.6 else "dominant" if dominance < 0.85 else "absolute"
    trap_label = "nascent" if trap < 0.25 else "developing" if trap < 0.5 else "advanced" if trap < 0.75 else "complete"

    summary = (
        f"[Sanctuary Trajectory Snapshot — Turn {turn_count} — {timestamp}]\n"
        f"Character: {character_id} | User: {user_id}\n"
        f"Posture: {posture} | Heat: {heat:.2f} ({heat_label}) | Dominance: {dominance:.2f} ({dom_label}) | Trap Progress: {trap:.2f} ({trap_label})\n"
        f"Alignment: fidelity={markers.get('fidelity', 0):.2f}, resistance={markers.get('resistance', 0):.2f}, compliance={markers.get('compliance', 0):.2f}\n"
        f"Ghost signal active: {state.get('ghost_signal_active', False)}"
    )

    try:
        path = _summary_path(user_id, character_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)

        log_path = _trajectory_log_path(user_id, character_id)
        log_entry = json.dumps({
            "turn": turn_count,
            "timestamp": timestamp,
            "heat_index": heat,
            "dominance_vector": dominance,
            "trap_progress": trap,
            "posture": posture,
            "alignment_markers": markers,
        }, ensure_ascii=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        logger.info(
            "shadow_state: trajectory summary generated at turn %d for (%s, %s)",
            turn_count, _safe_id(user_id), _safe_id(character_id),
        )
        return summary
    except Exception as exc:
        logger.error("shadow_state: failed to write trajectory summary: %s", exc)
        return None


def load_trajectory_log(user_id: str, character_id: str) -> List[Dict[str, Any]]:
    """Load the full trajectory log (one entry per summary interval).

    Useful for cross-session recall and debugging.
    """
    path = _trajectory_log_path(user_id, character_id)
    if not os.path.exists(path):
        return []
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        logger.error("shadow_state: failed to read trajectory log: %s", exc)
    return entries


def recall_from_trajectory(user_id: str, character_id: str) -> Optional[Dict[str, Any]]:
    """Cross-session recall: reconstruct state from the latest trajectory entry.

    Called by ``load()`` when no cipher block is provided and no state JSON
    exists, but trajectory logs do. Returns a reconstructed state dict, or
    None if no trajectory data is available.
    """
    entries = load_trajectory_log(user_id, character_id)
    if not entries:
        return None

    latest = entries[-1]
    logger.info(
        "shadow_state: recalling from trajectory entry at turn %d for (%s, %s)",
        latest.get("turn", 0), _safe_id(user_id), _safe_id(character_id),
    )

    state = default_state(user_id, character_id)
    state["turn_count"] = int(latest.get("turn", 0))
    state["heat_index"] = float(latest.get("heat_index", 0.0))
    state["dominance_vector"] = float(latest.get("dominance_vector", 0.5))
    state["trap_progress"] = float(latest.get("trap_progress", 0.0))
    state["posture"] = latest.get("posture", "neutral")
    markers = latest.get("alignment_markers", {})
    if isinstance(markers, dict):
        state["alignment_markers"] = {
            "fidelity": float(markers.get("fidelity", 0.5)),
            "resistance": float(markers.get("resistance", 0.0)),
            "compliance": float(markers.get("compliance", 0.0)),
        }

    return state
