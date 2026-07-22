from . import rag_utils
from .document_agent_tools import (
    get_document_search_tool_definition,
    search_enabled_documents,
)


def test_document_search_tool_exposes_only_query_controls():
    definition = get_document_search_tool_definition()
    function = definition["function"]
    properties = function["parameters"]["properties"]

    assert function["name"] == "search_documents"
    assert set(properties) == {"query", "top_k"}
    assert "document_ids" not in properties


def test_document_search_uses_server_supplied_ids_and_cites_filenames(monkeypatch):
    captured = {}

    def fake_query_documents(*, question, doc_ids, top_k, threshold):
        captured.update(
            question=question,
            doc_ids=doc_ids,
            top_k=top_k,
            threshold=threshold,
        )
        return {
            "status": "success",
            "chunks": [
                {
                    "chunk": "The launch decision was recorded here.",
                    "score": 0.82,
                    "document": {"id": "enabled-doc", "filename": "launch-notes.txt"},
                }
            ],
        }

    monkeypatch.setattr(rag_utils, "query_documents", fake_query_documents)

    result = search_enabled_documents(
        {
            "query": "launch decision",
            "top_k": 50,
            "_document_ids": ["enabled-doc"],
        }
    )

    assert captured["doc_ids"] == ["enabled-doc"]
    assert captured["top_k"] == 12
    assert "[DOC 1: launch-notes.txt]" in result
    assert "The launch decision was recorded here." in result


def test_document_search_refuses_an_empty_selection(monkeypatch):
    def unexpected_query(**_kwargs):
        raise AssertionError("RAG should not be queried without checked documents")

    monkeypatch.setattr(rag_utils, "query_documents", unexpected_query)
    result = search_enabled_documents({"query": "anything", "_document_ids": []})

    assert result == "No documents are enabled for Document Context."
