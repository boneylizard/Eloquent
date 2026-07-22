"""Tests for the pipeline SSE event formatting."""

import json

from backend.app.sanctuary import pipeline


def test_sse_event_format():
    event = pipeline._sse_event("analysis", {"posture": "dominant", "dominance_vector": 0.8})
    assert event.startswith("event: analysis\n")
    assert "data: " in event
    assert event.endswith("\n\n")

    # Parse the data line
    lines = event.strip().split("\n")
    data_line = [l for l in lines if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["posture"] == "dominant"
    assert payload["dominance_vector"] == 0.8


def test_sse_event_text():
    event = pipeline._sse_event("text", {"delta": "He steps closer."})
    assert "event: text" in event
    lines = event.strip().split("\n")
    data_line = [l for l in lines if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["delta"] == "He steps closer."


def test_sse_event_done():
    event = pipeline._sse_event("done", {"turn_id": "abc123", "error": False})
    assert "event: done" in event
    lines = event.strip().split("\n")
    data_line = [l for l in lines if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["turn_id"] == "abc123"
    assert payload["error"] is False


def test_sse_event_cipher():
    event = pipeline._sse_event("cipher", {"block": "⟦CIPHER:v1:abc:def⟧", "phase": "final"})
    assert "event: cipher" in event
    lines = event.strip().split("\n")
    data_line = [l for l in lines if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["phase"] == "final"


def test_sse_event_somatic():
    somatic_data = {
        "dashboard": {"lubrication_level": "damp"},
        "posture_label": "coiled tension",
    }
    event = pipeline._sse_event("somatic", somatic_data)
    assert "event: somatic" in event
    lines = event.strip().split("\n")
    data_line = [l for l in lines if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["dashboard"]["lubrication_level"] == "damp"
    assert payload["posture_label"] == "coiled tension"


def test_sse_event_error():
    event = pipeline._sse_event("error", {"step": "analysis", "detail": "model timeout"})
    assert "event: error" in event
    lines = event.strip().split("\n")
    data_line = [l for l in lines if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["step"] == "analysis"
