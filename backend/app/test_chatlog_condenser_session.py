"""

Tests for agentic chatlog condenser session store (no LLM calls).



Run: pytest backend/app/test_chatlog_condenser_session.py -v

"""



from __future__ import annotations



import asyncio



import pytest



from .chatlog_condenser_session import (

    CondenserSession,

    CondenserSessionStore,

    SessionMessage,

    SessionSettings,

    _append_stream_piece,

    build_agent_messages,

    build_llm_messages,

    build_progress_block,

    condensed_tail_excerpt,

    format_progress_marker,

    parse_progress_marker,

    session_store,

)





@pytest.fixture

def store() -> CondenserSessionStore:

    return CondenserSessionStore()





def test_create_and_get_session(store):

    settings = SessionSettings(model_name="endpoint-test")



    async def _run():

        session = await store.create(

            original_log="**User:** hello\n\n**Assistant:** hi",

            settings=settings,

        )

        fetched = await store.get(session.session_id)

        assert fetched is not None

        assert fetched.original_log.startswith("**User:**")

        assert fetched.settings.model_name == "endpoint-test"

        assert fetched.messages == []

        assert fetched.total_turn_count == 2

        assert fetched.last_condensed_turn_index == -1



    asyncio.run(_run())





def test_append_messages_and_partial_condensed(store):

    async def _run():

        session = await store.create(

            original_log="log",

            settings=SessionSettings(model_name="m"),

        )

        store.append_user_message(session, "condense please")

        assert len(session.messages) == 1

        assert session.messages[0].role == "user"



        store.append_assistant_message(session, "**User:** x\n[CONDENSED THROUGH: turn index 0]")

        assert "**User:** x" in session.partial_condensed

        assert session.last_condensed_turn_index == 0

        assert session.messages[1].role == "assistant"



    asyncio.run(_run())





def test_reset_clears_run_state(store):

    async def _run():

        session = await store.create(

            original_log="log",

            settings=SessionSettings(model_name="m"),

        )

        store.append_user_message(session, "go")

        store.append_assistant_message(session, "draft v1\n[CONDENSED THROUGH: turn index 1]")

        await store.reset(session.session_id)

        assert session.messages == []

        assert session.partial_condensed == ""

        assert session.last_condensed_turn_index == -1

        assert session.status == "active"



    asyncio.run(_run())


def test_cancel_preserves_original_log_and_messages(store):
    """POST /cancel must not wipe source log or run history (Stop semantics)."""

    async def _run():
        session = await store.create(
            original_log="**User:** keep me",
            settings=SessionSettings(model_name="m"),
        )
        store.append_user_message(session, "condense")
        store.append_assistant_message(session, "draft line one")
        session.status = "streaming"
        session.partial_condensed = "draft line one"

        cleared = await store.clear_streaming(session.session_id)
        pub = cleared.to_public_dict()

        assert cleared.original_log.startswith("**User:** keep me")
        assert pub["original_log"].startswith("**User:** keep me")
        assert len(cleared.messages) == 2
        assert cleared.partial_condensed == "draft line one"
        assert cleared.status == "active"

    asyncio.run(_run())


def test_append_assistant_skips_empty_content(store):
    async def _run():
        session = await store.create(
            original_log="log",
            settings=SessionSettings(model_name="m"),
        )
        session.partial_condensed = "existing draft"
        store.append_assistant_message(session, "   ")
        assert len(session.messages) == 0
        assert session.partial_condensed == "existing draft"

    asyncio.run(_run())


def test_clear_streaming_unlocks_stuck_session(store):

    async def _run():

        session = await store.create(

            original_log="log",

            settings=SessionSettings(model_name="m"),

        )

        session.status = "streaming"

        cleared = await store.clear_streaming(session.session_id)

        assert cleared.status == "active"



    asyncio.run(_run())


def test_reset_clears_streaming_flag(store):

    async def _run():

        session = await store.create(

            original_log="log",

            settings=SessionSettings(model_name="m"),

        )

        session.status = "streaming"

        await store.reset(session.session_id)

        assert session.status == "active"



    asyncio.run(_run())


def test_append_stream_piece_dedupes_cumulative_chunks():

    collected: list[str] = []

    assert _append_stream_piece(collected, "Hello") == "Hello"

    assert _append_stream_piece(collected, "Hello world") == " world"

    assert "".join(collected) == "Hello world"

    assert _append_stream_piece(collected, "Hello world") == ""

    assert _append_stream_piece(collected, "world") == ""





def test_require_missing_raises(store):

    from fastapi import HTTPException



    async def _run():

        with pytest.raises(HTTPException) as exc:

            await store.require("00000000-0000-0000-0000-000000000000")

        assert exc.value.status_code == 404



    asyncio.run(_run())





def test_parse_progress_marker_formats():

    assert parse_progress_marker("[CONDENSED THROUGH: turn index 12]") == 12

    assert parse_progress_marker("draft\n[CONDENSED THROUGH: turn 5]") == 5

    assert parse_progress_marker("<!-- progress: through turn 7 of 20 -->") == 7

    assert parse_progress_marker("no marker here") is None





def test_append_accumulates_and_advances_turn_index(store):

    async def _run():

        log = "**A:** one\n\n**B:** two\n\n**C:** three\n\n**D:** four"

        session = await store.create(

            original_log=log,

            settings=SessionSettings(model_name="m"),

        )

        store.append_assistant_message(

            session, "**A:** one tight\n[CONDENSED THROUGH: turn index 1]"

        )

        assert session.last_condensed_turn_index == 1

        first_len = len(session.partial_condensed)



        store.append_assistant_message(

            session, "**C:** three tight\n[CONDENSED THROUGH: turn index 3]"

        )

        assert session.last_condensed_turn_index == 3

        assert len(session.partial_condensed) > first_len

        assert "**A:** one tight" in session.partial_condensed

        assert "**C:** three tight" in session.partial_condensed



    asyncio.run(_run())





def test_build_agent_messages_excludes_assistant_history():

    session = CondenserSession(

        session_id="s1",

        original_log="**A:** one\n\n**B:** two",

        settings=SessionSettings(model_name="m"),

        messages=[

            SessionMessage(role="user", content="start"),

            SessionMessage(role="assistant", content="**A:** one tight\n[CONDENSED THROUGH: turn index 0]"),

        ],

        partial_condensed="**A:** one tight",

        total_turn_count=2,

        last_condensed_turn_index=0,

        parsed_turns=[],

    )

    msgs = build_agent_messages(session)

    assert msgs[0]["role"] == "system"

    assert "**A:** one" in msgs[0]["content"]

    assert "SEQUENTIAL PROGRESS" in msgs[0]["content"]

    assert "CONDENSED_SO_FAR" in msgs[0]["content"]

    assert all(m["role"] != "assistant" for m in msgs[1:])

    assert msgs[1]["role"] == "user"

    assert "Continue from turn 1" in msgs[1]["content"]





def test_build_llm_messages_alias():

    session = CondenserSession(

        session_id="s1",

        original_log="**A:** one",

        settings=SessionSettings(model_name="m"),

    )

    assert build_llm_messages(session) == build_agent_messages(session)





def test_build_progress_block_segment_start():

    block = build_progress_block(

        last_turn_index=4,

        total_turns=10,

        segment_start_turn=5,

    )

    assert "through turn index: 4" in block

    assert "Continue from turn index: 5" in block

    assert "turns 5 onward" in block





def test_condensed_tail_excerpt_truncates():

    long = "x" * 5000

    tail = condensed_tail_excerpt(long, max_chars=100)

    assert tail.startswith("…")

    assert len(tail) < 200





def test_format_progress_marker():

    assert format_progress_marker(9) == "[CONDENSED THROUGH: turn index 9]"





def test_session_store_singleton_isolated_from_tests():

    """Document that routes use module singleton; tests use fresh store fixture."""

    assert session_store is not None

