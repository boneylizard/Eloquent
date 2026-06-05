"""API routes for Transcript Corpus semantic search."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import transcript_corpus as tc

logger = logging.getLogger("transcript_corpus_routes")

corpus_router = APIRouter(tags=["transcript-corpus"], prefix="/corpus")


class IndexRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute or ~ path to folder of .txt files")
    corpus_name: Optional[str] = None
    corpus_id: Optional[str] = None
    recursive: bool = True
    background: bool = True


class SearchRequest(BaseModel):
    corpus_id: str
    query: str
    top_k: int = Field(25, ge=1, le=200, description="Results per page (limit)")
    offset: int = Field(0, ge=0, description="Pagination offset into filtered matches")
    min_score: float = Field(0.15, ge=0.0, le=1.0)
    min_first_person: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_first_person: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_file_contains: Optional[str] = None
    keyword: Optional[str] = Field(
        None,
        description="Optional keyword filter (must appear in chunk text)",
    )


@corpus_router.get("/status")
async def corpus_status() -> Dict[str, Any]:
    return {
        "vector_search_available": tc.is_vector_search_available(),
        "embedding_model": tc.EMBEDDING_MODEL_NAME if tc.HAVE_VECTOR_SEARCH else None,
        "data_dir": str(tc.CORPUS_ROOT),
    }


@corpus_router.get("/list")
async def corpus_list() -> Dict[str, Any]:
    corpora = tc.list_corpora()
    return {"corpora": corpora, "count": len(corpora)}


@corpus_router.get("/job/{job_id}")
async def corpus_job_status(job_id: str) -> Dict[str, Any]:
    job = tc.get_index_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@corpus_router.post("/index")
async def corpus_index(body: IndexRequest) -> Dict[str, Any]:
    logger.info(
        "POST /corpus/index folder=%r recursive=%s background=%s",
        body.folder_path,
        body.recursive,
        body.background,
    )
    if not tc.is_vector_search_available():
        logger.warning("POST /corpus/index rejected: vector search dependencies missing")
        raise HTTPException(
            status_code=503,
            detail="Install sentence-transformers and faiss-cpu for transcript search",
        )
    try:
        if body.background:
            job_id = tc.start_index_job(
                body.folder_path,
                corpus_name=body.corpus_name,
                corpus_id=body.corpus_id,
                recursive=body.recursive,
            )
            logger.info("POST /corpus/index started background job %s", job_id)
            return {"ok": True, "job_id": job_id, "status": "queued"}
        result = tc.index_folder(
            body.folder_path,
            corpus_name=body.corpus_name,
            corpus_id=body.corpus_id,
            recursive=body.recursive,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Corpus index failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@corpus_router.post("/search")
async def corpus_search(body: SearchRequest) -> Dict[str, Any]:
    if not tc.is_vector_search_available():
        raise HTTPException(status_code=503, detail="Vector search not available")
    try:
        return tc.search_corpus(
            body.corpus_id,
            body.query,
            top_k=body.top_k,
            offset=body.offset,
            min_score=body.min_score,
            min_first_person=body.min_first_person,
            max_first_person=body.max_first_person,
            source_file_contains=body.source_file_contains,
            keyword=body.keyword,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Corpus search failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@corpus_router.delete("/{corpus_id}")
async def corpus_delete(corpus_id: str) -> Dict[str, Any]:
    if not tc.delete_corpus(corpus_id):
        raise HTTPException(status_code=404, detail="Corpus not found")
    return {"ok": True, "deleted": corpus_id}
