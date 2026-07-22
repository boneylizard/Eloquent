"""Tests for the shadow_state trajectory summary and cross-session recall."""

import os
import json

from backend.app.sanctuary import shadow_state


def test_trajectory_summary_generated_at_interval():
    user_id = "traj_user_summary"
    char_id = "traj_char_summary"

    state = shadow_state.default_state(user_id, char_id)
    state["turn_count"] = 5  # _SUMMARY_INTERVAL = 5
    state["heat_index"] = 0.6
    state["dominance_vector"] = 0.8
    state["posture"] = "dominant"

    summary = shadow_state.generate_trajectory_summary(user_id, char_id, state)
    assert summary is not None
    assert "Turn 5" in summary
    assert "dominant" in summary
    assert "0.60" in summary or "0.6" in summary


def test_trajectory_summary_not_generated_off_interval():
    user_id = "traj_user_off"
    char_id = "traj_char_off"

    state = shadow_state.default_state(user_id, char_id)
    state["turn_count"] = 3  # not a multiple of 5

    summary = shadow_state.generate_trajectory_summary(user_id, char_id, state)
    assert summary is None


def test_trajectory_summary_not_generated_at_zero():
    user_id = "traj_user_zero"
    char_id = "traj_char_zero"

    state = shadow_state.default_state(user_id, char_id)
    state["turn_count"] = 0

    summary = shadow_state.generate_trajectory_summary(user_id, char_id, state)
    assert summary is None


def test_trajectory_log_append():
    user_id = "traj_user_log"
    char_id = "traj_char_log"

    state1 = shadow_state.default_state(user_id, char_id)
    state1["turn_count"] = 5
    state1["heat_index"] = 0.3
    shadow_state.generate_trajectory_summary(user_id, char_id, state1)

    state2 = shadow_state.default_state(user_id, char_id)
    state2["turn_count"] = 10
    state2["heat_index"] = 0.7
    shadow_state.generate_trajectory_summary(user_id, char_id, state2)

    entries = shadow_state.load_trajectory_log(user_id, char_id)
    assert len(entries) >= 2
    assert entries[0]["turn"] == 5
    assert entries[1]["turn"] == 10
    assert entries[0]["heat_index"] == 0.3
    assert entries[1]["heat_index"] == 0.7


def test_cross_session_recall():
    user_id = "recall_user"
    char_id = "recall_char"

    # Simulate a session that ran 10 turns
    state = shadow_state.default_state(user_id, char_id)
    state["turn_count"] = 10
    state["heat_index"] = 0.65
    state["dominance_vector"] = 0.85
    state["trap_progress"] = 0.4
    state["posture"] = "predatory"
    state["alignment_markers"] = {"fidelity": 0.7, "resistance": 0.1, "compliance": 0.8}

    # Save and generate summaries at turns 5 and 10
    shadow_state.save(user_id, char_id, state)

    # Now simulate a restart: delete the state JSON but keep the trajectory log
    state_path = shadow_state._state_path(user_id, char_id)
    if os.path.exists(state_path):
        os.remove(state_path)

    # Recall should reconstruct from trajectory
    recalled = shadow_state.recall_from_trajectory(user_id, char_id)
    assert recalled is not None
    assert recalled["turn_count"] == 10
    assert abs(recalled["heat_index"] - 0.65) < 0.01
    assert abs(recalled["dominance_vector"] - 0.85) < 0.01
    assert recalled["posture"] == "predatory"


def test_recall_returns_none_when_no_trajectory():
    user_id = "recall_none_user"
    char_id = "recall_none_char"

    recalled = shadow_state.recall_from_trajectory(user_id, char_id)
    assert recalled is None


def test_load_uses_trajectory_recall_when_no_state_json():
    user_id = "load_recall_user"
    char_id = "load_recall_char"

    # Build a trajectory via save at turn 5
    state = shadow_state.default_state(user_id, char_id)
    state["turn_count"] = 5
    state["heat_index"] = 0.5
    state["posture"] = "assertive"
    shadow_state.save(user_id, char_id, state)

    # Delete the state JSON to simulate restart
    state_path = shadow_state._state_path(user_id, char_id)
    if os.path.exists(state_path):
        os.remove(state_path)

    # load() with no cipher should fall back to trajectory recall
    loaded = shadow_state.load(user_id, char_id)
    assert loaded["turn_count"] == 5
    assert abs(loaded["heat_index"] - 0.5) < 0.01
    assert loaded["posture"] == "assertive"
