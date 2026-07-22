import json

from . import persona_realignment


def test_output_contract_requires_apply_ready_model_instructions():
    output_spec = persona_realignment.REALIGNMENT_OUTPUT_JSON_SPEC

    assert 'revised_model_instructions' in output_spec
    assert 'required and must not be empty' in output_spec


def test_parse_realignment_response_preserves_review_fields():
    raw = json.dumps({
        'delta_vs_current_instructions': ['Preserve established details.'],
        'revised_character_instructions': 'Full rewritten character prompt.',
        'revised_model_instructions': 'Do not repeat questions already answered.',
        'revised_user_profile_memories': [],
    })

    parsed = persona_realignment.parse_realignment_response(raw)

    assert parsed['revised_model_instructions'] == 'Do not repeat questions already answered.'
    assert parsed['delta_vs_current_instructions'] == ['Preserve established details.']
