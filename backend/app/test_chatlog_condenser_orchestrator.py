"""
Tests for autonomous chatlog condenser orchestrator (mock LLM, no network).

Run: pytest backend/app/test_chatlog_condenser_orchestrator.py -v
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from .chatlog_condenser_orchestrator import (
    CONDENSER_RUNS_DIR,
    OrchestratorSettings,
    OrchestratorStore,
    build_orchestrator_messages,
    build_segment_user_message,
    detect_no_advance,
    load_run_from_disk,
    merge_step_output,
    parse_progress_marker,
    persist_run_state,
    write_step_checkpoint,
)
from .chatlog_condenser_session import format_progress_marker


@pytest.fixture
def store() -> OrchestratorStore:
    return OrchestratorStore()


SAMPLE_LOG = (
    "**User:** one\n\n**Assistant:** two\n\n"
    "**User:** three\n\n**Assistant:** four\n\n"
    "**User:** five\n\n**Assistant:** six"
)


def test_create_run_parses_turns(store):
    async def _run():
        run = await store.create(
            original_log=SAMPLE_LOG,
            endpoint_ids=["endpoint-a", "endpoint-b"],
            settings=OrchestratorSettings(chunk_turns=2, auto_run=False),
        )
        assert run.total_turns == 6
        assert run.cursor_turn == -1
        assert run.status == "idle"
        assert run.endpoint_ids == ["endpoint-a", "endpoint-b"]

    asyncio.run(_run())


def test_cursor_advance_on_marker(store):
    async def _run():
        run = await store.create(
            original_log=SAMPLE_LOG,
            endpoint_ids=["endpoint-a"],
            settings=OrchestratorSettings(auto_run=False),
        )
        marker = merge_step_output(
            run, "**User:** one tight\n[CONDENSED THROUGH: turn index 1]"
        )
        assert marker == 1
        assert run.cursor_turn == 1
        assert "one tight" in run.partial_condensed

    asyncio.run(_run())


def test_detect_no_advance():
    class R:
        cursor_turn = 2

    assert detect_no_advance(R(), 2) is True
    assert detect_no_advance(R(), 1) is True
    assert detect_no_advance(R(), 3) is False
    assert detect_no_advance(R(), None) is True


def test_failover_index_increment(store):
    calls: list[str] = []

    async def mock_llm(*, model_name, system, user, max_tokens, temperature):
        calls.append(model_name)
        if model_name == "endpoint-a":
            raise HTTPException(status_code=502, detail="server disconnected")
        return "**User:** x\n[CONDENSED THROUGH: turn index 0]"

    async def _run():
        store.set_llm_runner(mock_llm)
        run = await store.create(
            original_log="**User:** hi\n\n**Assistant:** bye",
            endpoint_ids=["endpoint-a", "endpoint-b"],
            settings=OrchestratorSettings(
                auto_run=False, max_retries_per_step=2
            ),
        )
        run.status = "running"
        ok = await store.execute_step(run.run_id)
        assert ok is True
        assert run.current_endpoint_index == 1
        assert calls == ["endpoint-a", "endpoint-b"]
        assert run.cursor_turn == 0

    asyncio.run(_run())


def test_wrangler_then_complete(store):
    step = {"n": 0}

    async def mock_llm(*, model_name, system, user, max_tokens, temperature):
        step["n"] += 1
        if step["n"] == 1:
            return "Repeated fluff with no marker"
        if step["n"] == 2:
            return "**User:** one\n[CONDENSED THROUGH: turn index 0]"
        return "**User:** three\n[CONDENSED THROUGH: turn index 2]"

    async def _run():
        store.set_llm_runner(mock_llm)
        run = await store.create(
            original_log=SAMPLE_LOG,
            endpoint_ids=["endpoint-a"],
            settings=OrchestratorSettings(auto_run=False, chunk_turns=3),
        )
        run.status = "running"
        assert await store.execute_step(run.run_id) is True
        assert run.cursor_turn == 0
        assert run.wrangler_nudges >= 1

        run.status = "running"
        assert await store.execute_step(run.run_id) is True
        assert run.cursor_turn == 2

    asyncio.run(_run())


def test_status_transitions_pause_resume(store):
    async def _run():
        run = await store.create(
            original_log=SAMPLE_LOG,
            endpoint_ids=["endpoint-a"],
            settings=OrchestratorSettings(auto_run=False),
        )
        await store.start_run(run)
        assert run.status == "running"

        paused = await store.pause(run.run_id)
        assert paused.status == "paused"

        resumed = await store.resume(run.run_id)
        assert resumed.status == "running"

        stopped = await store.stop(run.run_id)
        assert stopped.status == "stopped"
        assert stopped.partial_condensed or stopped.cursor_turn >= -1

    asyncio.run(_run())


def test_rotate_every_step_alternates_endpoints(store):
    calls: list[str] = []

    async def mock_llm(*, model_name, system, user, max_tokens, temperature):
        calls.append(model_name)
        idx = len(calls) - 1
        return f"chunk {idx}\n[CONDENSED THROUGH: turn index {idx}]"

    async def _run():
        store.set_llm_runner(mock_llm)
        run = await store.create(
            original_log=SAMPLE_LOG,
            endpoint_ids=["endpoint-a", "endpoint-b"],
            settings=OrchestratorSettings(
                auto_run=False,
                alternate_apis_every_step=True,
                api_routing_mode="rotate_every_step",
            ),
        )
        run.status = "running"
        await store.execute_step(run.run_id)
        assert calls[0] == "endpoint-a"
        await store.execute_step(run.run_id)
        assert calls[1] == "endpoint-b"
        await store.execute_step(run.run_id)
        assert calls[2] == "endpoint-a"

    asyncio.run(_run())


def test_checkpoint_persisted(store, tmp_path, monkeypatch):
    from . import chatlog_condenser_orchestrator as orch_mod

    async def mock_llm(*, model_name, system, user, max_tokens, temperature):
        return "**User:** one\n[CONDENSED THROUGH: turn index 0]"

    async def _run():
        monkeypatch.setattr(orch_mod, "CONDENSER_RUNS_DIR", tmp_path)
        store.set_llm_runner(mock_llm)
        run = await store.create(
            original_log="**User:** hi\n\n**Assistant:** bye",
            endpoint_ids=["endpoint-a"],
            settings=OrchestratorSettings(auto_run=False),
        )
        run.status = "running"
        await store.execute_step(run.run_id)
        ckpt = tmp_path / run.run_id / "checkpoint_1.json"
        draft = tmp_path / run.run_id / "draft_1.md"
        assert ckpt.exists()
        assert draft.exists()
        run.status = "stopped"
        persist_run_state(run)
        reloaded = load_run_from_disk(run.run_id)
        assert reloaded is not None
        assert reloaded.cursor_turn == 0

    asyncio.run(_run())


def test_supervisor_instruction_in_user_message(store):
    async def _run():
        run = await store.create(
            original_log=SAMPLE_LOG,
            endpoint_ids=["endpoint-a"],
            settings=OrchestratorSettings(auto_run=False),
        )
        await store.set_supervisor_instruction(run.run_id, "skip to turn 50")
        run = await store.require(run.run_id)
        user = build_segment_user_message(run)
        assert "SUPERVISOR" in user
        assert "skip to turn 50" in user

    asyncio.run(_run())


def test_failover_does_not_rotate_on_success(store):
    calls: list[str] = []
    step = {"n": 0}

    async def mock_llm(*, model_name, system, user, max_tokens, temperature):
        calls.append(model_name)
        idx = step["n"]
        step["n"] += 1
        return f"**User:** x\n[CONDENSED THROUGH: turn index {idx}]"

    async def _run():
        store.set_llm_runner(mock_llm)
        run = await store.create(
            original_log="**User:** hi\n\n**Assistant:** bye",
            endpoint_ids=["endpoint-a", "endpoint-b"],
            settings=OrchestratorSettings(
                auto_run=False, use_global_round_robin=False
            ),
        )
        run.status = "running"
        await store.execute_step(run.run_id)
        await store.execute_step(run.run_id)
        assert calls == ["endpoint-a", "endpoint-a"]

    asyncio.run(_run())


def test_completed_when_cursor_at_last_turn(store):
    async def _run():
        run = await store.create(
            original_log="**User:** only\n\n**Assistant:** one",
            endpoint_ids=["endpoint-a"],
            settings=OrchestratorSettings(auto_run=False),
        )
        merge_step_output(
            run, "**User:** only\n[CONDENSED THROUGH: turn index 1]"
        )
        assert run.is_complete()
        run.status = "completed"
        assert run.status == "completed"

    asyncio.run(_run())


def test_build_messages_include_progress(store):
    async def _run():
        run = await store.create(
            original_log=SAMPLE_LOG,
            endpoint_ids=["endpoint-a"],
            settings=OrchestratorSettings(chunk_turns=5),
        )
        merge_step_output(run, "draft\n[CONDENSED THROUGH: turn index 1]")
        msgs = build_orchestrator_messages(run)
        assert msgs[0]["role"] == "system"
        assert "SEQUENTIAL PROGRESS" in msgs[0]["content"]
        assert "ORIGINAL_CHATLOG" in msgs[0]["content"]
        user = build_segment_user_message(run)
        assert "Continue" in user or "continue" in user.lower()

    asyncio.run(_run())


def test_parse_progress_marker_reexport():
    assert parse_progress_marker("[CONDENSED THROUGH: turn index 4]") == 4
    assert format_progress_marker(4) == "[CONDENSED THROUGH: turn index 4]"
