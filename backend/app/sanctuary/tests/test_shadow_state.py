"""Tests for the shadow_state module."""

import os
import tempfile

from backend.app.sanctuary import cipher
from backend.app.sanctuary import shadow_state


def test_default_state():
    state = shadow_state.default_state("user1", "char1")
    assert state["version"] == 1
    assert state["turn_count"] == 0
    assert state["heat_index"] == 0.0
    assert state["dominance_vector"] == 0.5
    assert state["posture"] == "neutral"
    assert state["user_id"] == "user1"
    assert state["character_id"] == "char1"


def test_load_save_roundtrip():
    user_id = "test_user_save"
    char_id = "test_char_save"

    state = shadow_state.default_state(user_id, char_id)
    state["heat_index"] = 0.65
    state["dominance_vector"] = 0.8
    state["posture"] = "dominant"
    state["turn_count"] = 7

    assert shadow_state.save(user_id, char_id, state)

    loaded = shadow_state.load(user_id, char_id)
    assert loaded["heat_index"] == 0.65
    assert loaded["dominance_vector"] == 0.8
    assert loaded["posture"] == "dominant"
    assert loaded["turn_count"] == 7


def test_load_with_cipher_merge():
    user_id = "test_user_cipher"
    char_id = "test_char_cipher"

    stored = shadow_state.default_state(user_id, char_id)
    stored["heat_index"] = 0.3
    stored["dominance_vector"] = 0.4
    shadow_state.save(user_id, char_id, stored)

    cipher_state = {
        "version": 1,
        "turn_count": 5,
        "heat_index": 0.7,
        "dominance_vector": 0.9,
        "posture": "predatory",
    }
    block = cipher.encode(cipher_state)

    loaded = shadow_state.load(user_id, char_id, cipher_block=block)
    assert abs(loaded["heat_index"] - 0.7) < 0.01  # hex encoding has small float error
    assert abs(loaded["dominance_vector"] - 0.9) < 0.01
    assert loaded["posture"] == "predatory"
    assert loaded["turn_count"] == 5


def test_load_no_stored_no_cipher_returns_default():
    user_id = "test_user_fresh"
    char_id = "test_char_fresh"
    loaded = shadow_state.load(user_id, char_id)
    assert loaded["turn_count"] == 0
    assert loaded["heat_index"] == 0.0
    assert loaded["posture"] == "neutral"


def test_apply_analysis_result():
    state = shadow_state.default_state("u", "c")
    state["turn_count"] = 3
    state["heat_index"] = 0.4
    state["dominance_vector"] = 0.5
    state["trap_progress"] = 0.2

    analysis = {
        "posture": "dominant",
        "dominance_vector": 0.8,
        "heat_index_delta": 0.15,
        "trap_progress_delta": 0.05,
        "ghost_signal_active": True,
        "alignment_update": {"fidelity": 0.6, "resistance": 0.1, "compliance": 0.7},
        "reasoning": "test",
    }

    new_state = shadow_state.apply_analysis_result(state, analysis)
    assert new_state["turn_count"] == 4
    assert abs(new_state["heat_index"] - 0.55) < 0.01
    assert new_state["dominance_vector"] == 0.8
    assert abs(new_state["trap_progress"] - 0.25) < 0.01
    assert new_state["posture"] == "dominant"
    assert new_state["ghost_signal_active"] is True
    assert new_state["alignment_markers"]["fidelity"] == 0.6
    assert new_state["alignment_markers"]["compliance"] == 0.7

    # Original state should not be mutated
    assert state["turn_count"] == 3


def test_apply_analysis_clamps_values():
    state = shadow_state.default_state("u", "c")
    state["heat_index"] = 0.95
    state["trap_progress"] = 0.98

    analysis = {
        "posture": "predatory",
        "dominance_vector": 1.0,
        "heat_index_delta": 0.3,
        "trap_progress_delta": 0.3,
        "ghost_signal_active": False,
        "alignment_update": {},
        "reasoning": "",
    }

    new_state = shadow_state.apply_analysis_result(state, analysis)
    assert new_state["heat_index"] == 1.0
    assert new_state["trap_progress"] == 1.0


def test_safe_id():
    assert shadow_state._safe_id("user-123_abc") == "user-123_abc"
    assert shadow_state._safe_id("user with spaces!") == "userwithspaces"
    assert shadow_state._safe_id(None) == "unknown"
    assert shadow_state._safe_id("") == "unknown"
