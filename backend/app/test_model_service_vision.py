from .vision_support import build_vision_completion_options, build_vision_messages, parse_json_object


def test_extract_prompt_uses_yaml_in_system_message():
    messages = build_vision_messages("abc", "animal: The animal shown", "extract")

    assert messages[0]["role"] == "system"
    assert "animal: The animal shown" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ]


def test_extract_uses_greedy_json_constrained_decoding():
    options = build_vision_completion_options([], "extract", 512, 0.8, 1.3)

    assert options["temperature"] == 0.0
    assert options["top_k"] == 0
    assert options["repeat_penalty"] == 1.0
    assert options["response_format"] == {"type": "json_object"}


def test_json_parser_recovers_fenced_or_prefixed_objects():
    assert parse_json_object('```json\n{"kind":"photo"}\n```') == {"kind": "photo"}
    assert parse_json_object('Result: {"kind":"screenshot"} trailing') == {"kind": "screenshot"}
