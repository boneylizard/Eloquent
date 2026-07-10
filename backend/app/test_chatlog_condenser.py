"""
Chatlog condenser unit tests + reconstruction-fidelity eval harness structure.

Run: pytest backend/app/test_chatlog_condenser.py -v
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from pydantic import ValidationError

from . import chatlog_condenser as cc
from . import chatlog_condenser_prompt as prompts
from .chatlog_condenser_routes import CondenseChatlogRequest


SAMPLE_LOG = """**User:** I think we should use event sourcing for the audit trail.

**Assistant:** Event sourcing helps replay, but it adds operational complexity. What's your consistency requirement?

**User:** Strong consistency on the ledger, eventually consistent elsewhere.

**Assistant:** Then don't event-source the ledger itself — use an append-only table with snapshots. I overstated event sourcing earlier.

**User:** That correction matters. So ledger = OLTP table, domain events elsewhere?

**Assistant:** Yes. Thread shift: we were talking deployment; now implementation. Keep the dialectical record.
"""


def test_condense_request_accepts_defaults():
    req = CondenseChatlogRequest.model_validate(
        {
            "text": "**User:** hi",
            "model_name": "endpoint-1",
        }
    )
    assert req.chunk_target_tokens == cc.DEFAULT_CHUNK_TARGET_TOKENS
    assert req.target_ratio == 0.4
    assert req.overlap_turns == 5
    assert req.include_full_log_context is True


def test_condense_request_accepts_large_budgets():
    req = CondenseChatlogRequest.model_validate(
        {
            "text": "x",
            "model_name": "m",
            "chunk_target_tokens": 48000,
            "target_ratio": 0.05,
            "overlap_turns": 50,
        }
    )
    assert req.chunk_target_tokens == 48000
    assert req.target_ratio == 0.05
    assert req.overlap_turns == 50


def test_condense_request_rejects_non_positive():
    with pytest.raises(ValidationError):
        CondenseChatlogRequest.model_validate(
            {
                "text": "x",
                "model_name": "m",
                "chunk_target_tokens": 0,
            }
        )
    with pytest.raises(ValidationError):
        CondenseChatlogRequest.model_validate(
            {
                "text": "x",
                "model_name": "m",
                "target_ratio": 0,
            }
        )
    with pytest.raises(ValidationError):
        CondenseChatlogRequest.model_validate(
            {
                "text": "x",
                "model_name": "m",
                "overlap_turns": -1,
            }
        )


def test_normalize_endpoint_model_id():
    assert cc.normalize_endpoint_model_id("endpoint-endpoint-123") == "endpoint-123"
    assert cc.normalize_endpoint_model_id("endpoint-123") == "endpoint-123"


def test_openai_compat_remote_protocol_error_is_retryable():
    pytest.importorskip("tiktoken")
    httpx = pytest.importorskip("httpx")
    from .openai_compat import _openai_compat_is_transient_upstream_for_retry

    err = httpx.RemoteProtocolError(
        "Server disconnected without sending a response."
    )
    assert _openai_compat_is_transient_upstream_for_retry(
        err, include_read_write_timeout=True
    )


def test_condenser_llm_error_is_transient():
    assert cc._condenser_llm_error_is_transient(
        HTTPException(
            status_code=502,
            detail=(
                "Cannot connect to deepseek at https://example.com: "
                "RemoteProtocolError: Server disconnected"
            ),
        )
    )
    assert not cc._condenser_llm_error_is_transient(
        HTTPException(status_code=400, detail="bad request")
    )
    assert not cc._condenser_llm_error_is_transient(
        HTTPException(status_code=502, detail="Remote API error: quota exceeded")
    )


def test_format_condenser_api_error_config_vs_provider():
    cfg = cc._format_condenser_api_error(
        HTTPException(status_code=404, detail="endpoint not configured"),
        endpoint_name="nano-gpt",
    )
    assert "configuration" in cfg.lower()
    assert "nano-gpt" in cfg

    provider = cc._format_condenser_api_error(
        HTTPException(
            status_code=502,
            detail="Cannot connect: RemoteProtocolError: Server disconnected",
        ),
        endpoint_name="deepseek",
    )
    assert cc._CONDENSER_API_DISCONNECT_MSG.split(".")[0] in provider
    assert "RemoteProtocolError" in provider


def test_call_llm_api_uses_stream_aggregate_not_non_streaming():
    import inspect

    source = inspect.getsource(cc.call_llm)
    assert "collect_openai_compatible_stream_text" in source
    assert "forward_to_configured_endpoint_non_streaming" not in source
    assert '"stream": True' in source


def test_parse_chatlog_markdown_speakers():
    turns = cc.parse_chatlog(SAMPLE_LOG)
    assert len(turns) >= 4
    speakers = {t.speaker for t in turns}
    assert "User" in speakers
    assert "Assistant" in speakers
    assert any("correction" in t.content.lower() for t in turns)


def test_parse_chatlog_role_prefix():
    text = "User: First point.\n\nAssistant: Pushback here.\n\nUser: Refinement."
    turns = cc.parse_chatlog(text)
    assert len(turns) == 3
    assert turns[1].speaker == "Assistant"


def test_chunk_turns_respects_user_chunk_budget():
    turns = [
        cc.Turn(speaker="A", content="x" * 400, index=i) for i in range(20)
    ]
    small = cc.chunk_turns_with_overlap(turns, chunk_target_tokens=2000, overlap_turns=0)
    large = cc.chunk_turns_with_overlap(turns, chunk_target_tokens=48000, overlap_turns=0)
    assert len(small) > len(large)


def test_estimate_condenser_llm_passes():
    assert cc.estimate_condenser_llm_passes(1, run_eval=False) == 2
    assert cc.estimate_condenser_llm_passes(2, run_eval=False) == 5
    assert cc.estimate_condenser_llm_passes(2, run_eval=True) == 7


def test_chunk_turns_with_overlap_produces_multiple_chunks():
    turns = [
        cc.Turn(speaker="A", content=f"paragraph {i} " + ("x" * 500), index=i)
        for i in range(30)
    ]
    chunks = cc.chunk_turns_with_overlap(
        turns, chunk_target_tokens=2000, overlap_turns=3
    )
    assert len(chunks) > 1
    assert chunks[1].overlap_turn_count > 0


def test_merge_skeletons_preserves_moves():
    sk1 = {"moves": [{"id": "m1", "type": "claim"}], "open_threads": ["t1"]}
    sk2 = {"moves": [{"id": "m2", "type": "correction"}], "open_threads": ["t1", "t2"]}
    merged = cc.merge_skeletons([sk1, sk2])
    assert len(merged["moves"]) == 2
    assert "t1" in merged["open_threads"]
    assert "t2" in merged["open_threads"]


def test_parse_json_object_with_fence():
    raw = 'Here:\n```json\n{"moves": [], "open_threads": []}\n```'
    obj = cc.parse_json_object(raw)
    assert "moves" in obj


def test_repair_truncated_json_blob_closes_brackets():
    broken = '{"chunk_id":"c1","moves":[{"id":"m1","type":"claim","speakers":["A"],"gist":"ok"'
    repaired = cc.repair_truncated_json_blob(broken)
    parsed = json.loads(repaired)
    assert parsed["moves"][0]["id"] == "m1"


def test_salvage_moves_from_truncated_skeleton():
    broken = (
        '{"chunk_id":"c1","moves":['
        '{"id":"m1","type":"claim","speakers":["A"],"gist":"first"},'
        '{"id":"m2","type":"pushback","speakers":["B"],"gist":"second"},'
        '{"id":"m3","type":"correction","speakers":["A"],"gist":"incompl'
    )
    moves = cc.salvage_moves_from_blob(broken)
    assert len(moves) >= 2
    sk = cc.salvage_skeleton_object(broken, chunk_id="c1")
    assert sk["chunk_id"] == "c1"
    assert len(sk["moves"]) >= 2


def test_parse_json_object_never_raises_on_garbage():
    obj = cc.parse_json_object("not json at all", context="skeleton", chunk_id="c9")
    assert obj.get("chunk_id") == "c9"
    assert obj.get("_degraded") is True
    assert obj.get("moves") == []


def test_minimal_skeleton_shape():
    sk = cc.minimal_skeleton("chunk_2")
    assert sk["chunk_id"] == "chunk_2"
    assert sk["moves"] == []
    assert sk["_degraded"] is True


def test_output_tokens_for_chunk_at_least_default():
    chunk = cc.ChunkSpec(chunk_id="chunk_1", turns=[cc.Turn(speaker="A", content="hi", index=0)])
    assert cc.output_tokens_for_chunk(chunk) >= cc.DEFAULT_MAX_OUTPUT_TOKENS


def test_condenser_module_has_no_8192_output_cap():
    import inspect

    source = inspect.getsource(cc)
    assert "max_tokens=8192" not in source
    assert "max_tokens = 8192" not in source


def test_json_closing_suffix_unterminated_string():
    blob = '{"moves": [{"id": "m1", "gist": "half'
    suffix = cc.json_closing_suffix(blob)
    assert suffix.startswith('"')
    repaired = cc.repair_truncated_json_blob(blob)
    parsed = json.loads(repaired)
    assert parsed["moves"][0]["id"] == "m1"


def test_repair_truncated_json_trim_last_move():
    blob = (
        '{"chunk_id": "chunk_1", "moves": ['
        '{"id": "m1", "type": "claim", "speakers": ["A"], "gist": "ok"},'
        '{"id": "m2", "type": "claim", "speakers": ["B"], "gist": "cut off mid'
    )
    obj = cc.parse_json_object(blob, context="test", chunk_id="chunk_1")
    assert len(obj["moves"]) >= 1
    assert obj["moves"][0]["id"] == "m1"


def test_salvage_moves_drops_incomplete_trailing_move():
    blob = (
        '{"chunk_id": "c1", "moves": ['
        '{"id": "m1", "type": "correction", "speakers": ["U"], "gist": "first"},'
        '{"id": "m2", "type": "claim", "speakers": ["A"], "gist": "broken'
    )
    moves = cc.salvage_moves_from_blob(blob)
    assert len(moves) == 1
    assert moves[0]["type"] == "correction"


def test_skeleton_max_tokens_scales_with_chunk():
    small = cc.ChunkSpec("c1", [cc.Turn("A", "hi", 0)])
    big = cc.ChunkSpec(
        "c2",
        [cc.Turn("A", "x" * 20000, i) for i in range(40)],
    )
    assert cc.skeleton_max_tokens_for_chunk(small) >= cc.SKELETON_MIN_OUTPUT_TOKENS
    assert cc.skeleton_max_tokens_for_chunk(big) <= cc.SKELETON_MAX_OUTPUT_TOKENS
    assert cc.skeleton_max_tokens_for_chunk(big) >= cc.skeleton_max_tokens_for_chunk(small)


def test_parse_json_object_optional_never_raises():
    obj, err = cc.parse_json_object_optional(
        "not json at all", context="skeleton"
    )
    assert err is None
    assert isinstance(obj, dict)


def test_score_eval_results_fidelity():
    results = [
        {"probe_id": "p1", "pass": True},
        {"probe_id": "p2", "pass": False, "note": "missing correction"},
    ]
    summary = cc.score_eval_results(results)
    assert summary["total"] == 2
    assert summary["pass_count"] == 1
    assert summary["fidelity_score"] == 0.5
    assert "missing_correction" in summary["failure_modes"]


def test_reconstruction_eval_harness_structure():
    """Documents the eval contract without calling an LLM."""
    probes = {
        "probes": [
            {
                "id": "p1",
                "question": "What did the assistant correct about event sourcing?",
                "structural_focus": "correction move",
                "oracle_hints": ["overstated", "append-only"],
            }
        ]
    }
    condensed = "**Assistant:** I overstated event sourcing; use append-only ledger with snapshots."
    payload = {
        "probes": probes["probes"],
        "condensed_chars": len(condensed),
        "oracle_check": all(
            h.lower() in condensed.lower() for h in probes["probes"][0]["oracle_hints"]
        ),
    }
    assert payload["oracle_check"] is True


@pytest.mark.parametrize(
    "snippet,expected_in",
    [
        ("NOT summarization", ("SKELETON_SYSTEM", "RENDER_SYSTEM")),
        ("bullet highlights", ("SKELETON_SYSTEM",)),
        ("bullet takeaways", ("RENDER_SYSTEM",)),
        ("speaker turns", ("RENDER_SYSTEM",)),
        ("cross_chunk_callbacks", ("SKELETON_SYSTEM",)),
        ("GLOBAL READ, LOCAL WRITE", ("SKELETON_SYSTEM", "RENDER_SYSTEM")),
        ("FULL_CHATLOG", ("STITCH_SYSTEM",)),
    ],
)
def test_prompts_forbid_flattening(snippet, expected_in):
    for attr in expected_in:
        assert snippet in getattr(prompts, attr)


def test_build_skeleton_user_message_includes_full_log():
    full = "**User:** opener\n\n**Assistant:** reply"
    segment = "**User:** segment only"
    msg = prompts.build_skeleton_user_message(
        chunk_id="chunk_2",
        segment_transcript=segment,
        prior_context='{"moves":[]}',
        full_chatlog=full,
        overlap_turn_count=3,
        include_full_log_context=True,
    )
    assert "FULL_CHATLOG:" in msg
    assert full in msg
    assert "SEGMENT_TO_PROCESS" in msg
    assert segment in msg
    assert "PRIOR_SKELETON_TAIL" in msg
    assert "overlap prior segment" in msg
    assert prompts.GLOBAL_LOCAL_INSTRUCTION.split("\n")[0] in msg


def test_build_skeleton_user_message_segment_only_when_disabled():
    msg = prompts.build_skeleton_user_message(
        chunk_id="chunk_1",
        segment_transcript="**User:** only",
        full_chatlog="**User:** full\n\n**Assistant:** ignored",
        include_full_log_context=False,
    )
    assert "FULL_CHATLOG:" not in msg
    assert "SEGMENT_TO_PROCESS" in msg
    assert "**User:** only" in msg
    assert "ignored" not in msg


def test_build_render_user_message_includes_full_log():
    msg = prompts.build_render_user_message(
        skeleton_json='{"moves":[]}',
        segment_transcript="**User:** seg",
        full_chatlog="**User:** full",
        chunk_id="chunk_1",
        include_full_log_context=True,
    )
    assert "FULL_CHATLOG:" in msg
    assert "SEGMENT_TO_PROCESS" in msg
    assert "STRUCTURAL_SKELETON" in msg
    assert "SEGMENT_TO_PROCESS only" in msg


def test_build_stitch_user_message_includes_full_log():
    msg = prompts.build_stitch_user_message(
        prior_skeleton_tail="{}",
        segments="--- SEGMENT 1 ---\ndraft",
        full_chatlog="**User:** entire log",
        include_full_log_context=True,
    )
    assert "FULL_CHATLOG:" in msg
    assert "**User:** entire log" in msg


def test_estimate_per_call_context_tokens_with_full_log():
    est = cc.estimate_per_call_context_tokens(
        full_chatlog_md="x" * 4000,
        segment_md="y" * 800,
        prior_context="z" * 200,
        include_full_log_context=True,
    )
    assert est == cc.estimate_tokens("x" * 4000) + cc.estimate_tokens("y" * 800) + cc.estimate_tokens("z" * 200)


def test_context_size_warning_when_large():
    big = "a" * (cc.FULL_CONTEXT_WARN_TOKENS_EST * cc.CHARS_PER_TOKEN + 1000)
    est = cc.estimate_per_call_context_tokens(
        full_chatlog_md=big,
        segment_md="seg",
        include_full_log_context=True,
    )
    warn = cc.context_size_warning(est)
    assert warn is not None
    assert "exceeds" in warn.lower()


def test_context_size_warning_none_when_small():
    assert cc.context_size_warning(1000) is None
