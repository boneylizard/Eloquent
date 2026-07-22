import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_document_search_tool_definition() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the local documents the user explicitly enabled in Mirid. "
                "Use this when the answer may depend on those files. Results include source filenames; "
                "cite the supplied [DOC n: filename] labels in the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused semantic search query for the enabled documents",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12,
                        "default": 6,
                        "description": "Maximum number of relevant passages to return",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def search_enabled_documents(arguments: Any) -> str:
    try:
        from . import rag_utils

        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip().startswith("{") else {"query": arguments}
        query = arguments.get("query") or arguments.get("q") or ""
        if not query:
            return "Error: query required."
        document_ids = list(arguments.get("_document_ids") or [])
        if not document_ids:
            return "No documents are enabled for Document Context."
        top_k = max(1, min(int(arguments.get("top_k") or 6), 12))
        data = rag_utils.query_documents(
            question=query,
            doc_ids=document_ids,
            top_k=top_k,
            threshold=rag_utils.RAG_CHAT_SIMILARITY_THRESHOLD,
        )
        if data.get("status") != "success":
            return f"Document search failed: {data.get('error') or 'unknown error'}"
        chunks = data.get("chunks") or []
        lines = [
            f"LOCAL DOCUMENT SEARCH: {len(chunks)} relevant passage(s)",
            f"Query: {query}",
            "Cite useful evidence with the exact [DOC n: filename] label shown below.",
            "",
        ]
        for index, chunk in enumerate(chunks, 1):
            document = chunk.get("document") or {}
            filename = document.get("filename") or "unknown file"
            lines.append(f"[DOC {index}: {filename}]")
            lines.append(str(chunk.get("chunk") or ""))
            lines.append("")
        return "\n".join(lines)[:80000]
    except Exception as exc:
        logger.exception("document search tool failed")
        return f"Document search failed: {exc}"
