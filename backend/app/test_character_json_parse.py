"""Unit tests for character JSON parse/repair."""

from __future__ import annotations

import json

import pytest

from . import character_json_parse as cjp


VALID_CHARACTER = {
    "name": "Mira",
    "description": "A sharp-witted archivist.",
    "model_instructions": "Speak in dry metaphors.",
    "scenario": "A candlelit archive.",
    "first_message": "You found the wrong shelf again.",
    "example_dialogue": [{"role": "user", "content": "Hello"}, {"role": "character", "content": "Hmm."}],
    "loreEntries": [{"content": "Keeps a raven.", "keywords": ["raven"]}],
}


def test_parse_character_json_valid():
    raw = json.dumps(VALID_CHARACTER)
    character, partial, salvaged, err = cjp.parse_character_json(raw)
    assert err is None
    assert salvaged is False
    assert partial is None
    assert character["name"] == "Mira"


def test_parse_character_json_with_fence():
    raw = "```json\n" + json.dumps(VALID_CHARACTER) + "\n```"
    character, partial, salvaged, err = cjp.parse_character_json(raw)
    assert character is not None
    assert err is None


def test_parse_character_json_truncated_repair():
    broken = (
        '{"name": "Kai", "description": "A wandering smith", '
        '"model_instructions": "Blunt and practical'
    )
    character, partial, salvaged, err = cjp.parse_character_json(broken)
    recovered = character or partial
    assert recovered is not None
    assert recovered.get("name") == "Kai"
    assert recovered.get("description")


def test_parse_character_json_garbage():
    character, partial, salvaged, err = cjp.parse_character_json("not json at all")
    assert character is None
    assert partial is None
    assert err is not None


def test_character_json_is_usable():
    assert cjp.character_json_is_usable(VALID_CHARACTER) is True
    assert cjp.character_json_is_usable({"name": "X"}, partial_ok=True) is True
    assert cjp.character_json_is_usable({"name": "X"}, partial_ok=False) is False


def test_salvage_character_fields():
    blob = '{"name": "Elena", "description": "Quiet healer"'
    salvaged = cjp.salvage_character_fields(blob)
    assert salvaged["name"] == "Elena"
    assert salvaged.get("_salvaged") is True


def test_build_character_repair_prompt_includes_schema():
    prompt = cjp.build_character_repair_prompt('{"name": "broken')
    assert "name" in prompt
    assert "BROKEN_JSON" in prompt
