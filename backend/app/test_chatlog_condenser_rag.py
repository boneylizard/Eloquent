"""
Tests for chatlog condenser RAG supplement (mocked rag_utils, no FAISS required).

Run: pytest backend/app/test_chatlog_condenser_rag.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from .chatlog_condenser_orchestrator import (
    OrchestratorRun,
    OrchestratorSettings,
    build_orchestrator_messages,
)
from .chatlog_condenser_rag import (
    dedupe_rag_chunks,
    query_rag_for_step,
)


SAMPLE_LOG = "**User:** one\n\n**Assistant:** two\n\n**User:** three\n\n**Assistant:** four"


def test_dedupe_rag_chunks_skips_partial_overlap():
    chunks = [
        {"chunk": "already condensed claim about X", "document": {"id": "d1", "filename": "t.txt"}},
        {"chunk": "new distant callback about Y", "document": {"id": "d1", "filename": "t.txt"}},
    ]
    partial = "already condensed claim about X in dense form"
    out = dedupe_rag_chunks(chunks, partial_condensed=partial)
    assert len(out) == 1
    assert "callback" in out[0]["chunk"]


@patch("backend.app.rag_utils.rag_processor")
@patch("backend.app.rag_utils.query_documents")
@patch("backend.app.rag_utils.is_rag_available")
def test_query_rag_for_step_returns_supplement_block(
    mock_available, mock_query_docs, mock_processor
):
    mock_available.return_value = True
    mock_query_docs.return_value = {
        "status": "success",
        "chunks": [
            {
                "chunk": "User revised the definition of epistemic closure.",
                "document": {"id": "doc1", "filename": "transcript.md"},
            }
        ],
    }
    mock_processor.format_for_prompt.return_value = (
        "RELEVANT DOCUMENT SECTIONS:\n\n[DOC 1] transcript.md\n"
        "User revised the definition of epistemic closure.\n"
    )

    block = query_rag_for_step(
        doc_ids=["doc1"],
        segment_start=0,
        segment_end=3,
        partial_condensed="",
    )
    assert "RAG_SUPPLEMENT" in block
    assert "epistemic closure" in block


@patch("backend.app.rag_utils.query_documents")
@patch("backend.app.rag_utils.is_rag_available")
def test_query_rag_empty_continues(mock_available, mock_query_docs):
    mock_available.return_value = True
    mock_query_docs.return_value = {
        "status": "success",
        "chunks": [],
        "formatted_context": "",
    }

    block = query_rag_for_step(
        doc_ids=["doc1"],
        segment_start=0,
        segment_end=1,
    )
    assert block == ""


@patch("backend.app.chatlog_condenser_orchestrator.query_rag_for_step")
def test_build_orchestrator_messages_injects_rag_supplement(mock_query):
    mock_query.return_value = (
        "RAG_SUPPLEMENT (retrieved document chunks — cross-reference only):\n"
        "- sample retrieved line\n"
    )
    run = OrchestratorRun(
        run_id="test",
        original_log=SAMPLE_LOG,
        endpoint_ids=["endpoint-a"],
        settings=OrchestratorSettings(
            chunk_turns=2,
            use_rag=True,
            rag_doc_ids=["doc-transcript"],
            include_full_log_context=True,
        ),
        total_turns=4,
        cursor_turn=-1,
    )
    msgs = build_orchestrator_messages(run)
    assert len(msgs) == 2
    combined = msgs[0]["content"] + msgs[1]["content"]
    assert "RAG_SUPPLEMENT" in combined
    assert "ORIGINAL_CHATLOG" in msgs[0]["content"]
    mock_query.assert_called_once()
    call_kw = mock_query.call_args.kwargs
    assert call_kw["doc_ids"] == ["doc-transcript"]
    assert call_kw["segment_start"] == 0


@patch("backend.app.chatlog_condenser_orchestrator.query_rag_for_step")
def test_build_orchestrator_rag_in_user_when_full_log_off(mock_query):
    mock_query.return_value = "RAG_SUPPLEMENT (retrieved document chunks — cross-reference only):\nbody"
    run = OrchestratorRun(
        run_id="test",
        original_log=SAMPLE_LOG,
        endpoint_ids=["endpoint-a"],
        settings=OrchestratorSettings(
            use_rag=True,
            rag_doc_ids=["doc1"],
            include_full_log_context=False,
        ),
        total_turns=4,
        cursor_turn=1,
    )
    msgs = build_orchestrator_messages(run)
    assert "RAG_SUPPLEMENT" in msgs[1]["content"]
    assert "omitted from system" in msgs[0]["content"]


@patch("backend.app.chatlog_condenser_orchestrator.query_rag_for_step")
def test_build_orchestrator_failsafe_query_on_wrangler(mock_query):
    mock_query.return_value = ""
    run = OrchestratorRun(
        run_id="test",
        original_log=SAMPLE_LOG,
        endpoint_ids=["endpoint-a"],
        settings=OrchestratorSettings(use_rag=True, rag_doc_ids=["d1"]),
        total_turns=4,
        cursor_turn=0,
    )
    build_orchestrator_messages(run, wrangler=True)
    assert mock_query.call_args.kwargs.get("failsafe") is True
