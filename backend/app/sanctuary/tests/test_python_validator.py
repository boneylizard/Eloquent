"""Tests for the python_validator module."""

from backend.app.sanctuary import python_validator


def test_empty_code_is_valid():
    result = python_validator.validate("")
    assert result["valid"] is True
    assert result["drives"] == []


def test_valid_grip():
    result = python_validator.validate("grip('wrist', 0.7)")
    assert result["valid"] is True
    assert len(result["drives"]) == 1
    assert result["drives"][0]["action"] == "grip"
    assert result["drives"][0]["args"]["target"] == "wrist"
    assert result["drives"][0]["args"]["intensity"] == 0.7


def test_valid_shock():
    result = python_validator.validate("shock(0.5, 0.8)")
    assert result["valid"] is True
    assert result["drives"][0]["action"] == "shock"


def test_valid_freeze():
    result = python_validator.validate("freeze(0.6)")
    assert result["valid"] is True
    assert result["drives"][0]["action"] == "freeze"


def test_valid_theme():
    result = python_validator.validate("theme('crimson')")
    assert result["valid"] is True
    assert result["drives"][0]["args"]["palette"] == "crimson"


def test_multiple_drives():
    result = python_validator.validate("grip('throat', 0.8)\ntheme('void')")
    assert result["valid"] is True
    assert len(result["drives"]) == 2


def test_unknown_function_rejected():
    result = python_validator.validate("exec('malicious code')")
    assert result["valid"] is False
    assert "Unknown drive function" in result["errors"][0]


def test_import_rejected():
    result = python_validator.validate("import os\nos.system('rm -rf /')")
    assert result["valid"] is False


def test_assignment_rejected():
    result = python_validator.validate("x = 1")
    assert result["valid"] is False
    assert "Disallowed statement" in result["errors"][0]


def test_syntax_error_rejected():
    result = python_validator.validate("grip(")
    assert result["valid"] is False
    assert "Syntax error" in result["errors"][0]


def test_wrong_arg_count_rejected():
    result = python_validator.validate("grip('wrist')")
    assert result["valid"] is False
    assert "expects 2 arg(s)" in result["errors"][0]


def test_invalid_theme_palette_rejected():
    result = python_validator.validate("theme('nonexistent')")
    assert result["valid"] is False
    assert "unknown palette" in result["errors"][0]


def test_drives_to_hijack_descriptor_grip():
    drives = [{"action": "grip", "args": {"target": "wrist", "intensity": 0.8}}]
    desc = python_validator.drives_to_hijack_descriptor(drives)
    assert desc["lock"]["input_locked"] is True
    assert desc["lock"]["duration_ms"] > 0


def test_drives_to_hijack_descriptor_shock():
    drives = [{"action": "shock", "args": {"intensity": 0.6, "duration": 0.4}}]
    desc = python_validator.drives_to_hijack_descriptor(drives)
    assert desc["shake"]["intensity"] == 0.6
    assert desc["glitch"]["intensity"] > 0


def test_drives_to_hijack_descriptor_freeze():
    drives = [{"action": "freeze", "args": {"duration": 0.5}}]
    desc = python_validator.drives_to_hijack_descriptor(drives)
    assert desc["lock"]["input_locked"] is True
    assert desc["lock"]["scroll_locked"] is True


def test_drives_to_hijack_descriptor_theme():
    drives = [{"action": "theme", "args": {"palette": "ice"}}]
    desc = python_validator.drives_to_hijack_descriptor(drives)
    assert desc["theme_shift"] == "ice"
