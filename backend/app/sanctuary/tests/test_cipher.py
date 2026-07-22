"""Tests for the cipher module."""

import json

from backend.app.sanctuary import cipher


def test_encode_decode_roundtrip():
    state = {
        "version": 1,
        "turn_count": 42,
        "heat_index": 0.75,
        "dominance_vector": 0.6,
        "trap_progress": 0.3,
        "posture": "dominant",
        "ghost_signal_active": True,
    }
    block = cipher.encode(state)
    assert block.startswith("⟦CIPHER:v1:")
    assert block.endswith("⟧")

    decoded = cipher.decode(block)
    assert decoded is not None
    assert decoded["version"] == 1
    assert decoded["turn_count"] == 42
    assert abs(decoded["heat_index"] - 0.75) < 0.01
    assert decoded["dominance_vector"] == 0.6
    assert decoded["trap_progress"] == 0.3
    assert decoded["posture"] == "dominant"
    assert decoded["ghost_signal_active"] is True


def test_encode_decode_minimal():
    state = {"version": 1, "turn_count": 0, "heat_index": 0.0}
    block = cipher.encode(state)
    decoded = cipher.decode(block)
    assert decoded is not None
    assert decoded["turn_count"] == 0
    assert decoded["heat_index"] == 0.0


def test_decode_invalid_returns_none():
    assert cipher.decode("") is None
    assert cipher.decode("not a cipher block") is None
    assert cipher.decode("⟦CIPHER:v2:zz:zz⟧") is None  # version 2 doesn't match v(\d+) pattern correctly — actually it does, let me use truly malformed
    assert cipher.decode("⟦NOTCIPHER:v1:abc:def⟧") is None  # wrong prefix
    assert cipher.decode("CIPHER:v1:abc:def") is None  # missing brackets


def test_strip_from_text():
    state = {"version": 1, "turn_count": 5, "heat_index": 0.5}
    block = cipher.encode(state)
    text = f"He steps closer.{block}The door clicks shut."
    stripped = cipher.strip_from_text(text)
    assert block not in stripped
    assert "He steps closer." in stripped
    assert "The door clicks shut." in stripped


def test_find_in_text():
    state = {"version": 1, "turn_count": 1, "heat_index": 0.1}
    block = cipher.encode(state)
    text = f"Some prose.{block}More prose."
    found = cipher.find_in_text(text)
    assert found is not None
    assert found == block


def test_find_in_text_none():
    assert cipher.find_in_text("just regular text") is None


def test_heat_index_precision():
    for heat in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        state = {"version": 1, "turn_count": 0, "heat_index": heat}
        block = cipher.encode(state)
        decoded = cipher.decode(block)
        assert abs(decoded["heat_index"] - heat) < 0.02, f"heat_index {heat} not preserved"
