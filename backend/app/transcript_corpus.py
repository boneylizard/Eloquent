"""
Transcript Corpus — semantic search over a folder of .txt files.

Separate from chat document RAG: indexes arbitrary folders on disk, persists
per-corpus FAISS indexes under backend/data/transcript_corpora/.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("transcript_corpus")

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "data" / "transcript_corpora"
REGISTRY_FILE = CORPUS_ROOT / "registry.json"
JOBS_DIR = CORPUS_ROOT / "jobs"

FIRST_PERSON_RE = re.compile(
    r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    HAVE_VECTOR_SEARCH = True
except ImportError:
    HAVE_VECTOR_SEARCH = False
    faiss = None  # type: ignore
    np = None  # type: ignore
    SentenceTransformer = None  # type: ignore

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder: Optional[Any] = None
_embedder_lock = threading.Lock()

_index_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Log to console + default logging (visible in backend terminal / log file)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_person_ratio(text: str) -> float:
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    hits = len(FIRST_PERSON_RE.findall(text))
    return round(hits / len(words), 4)


def _get_embedder():
    global _embedder
    if not HAVE_VECTOR_SEARCH:
        return None
    with _embedder_lock:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Loaded transcript corpus embedding model: %s", EMBEDDING_MODEL_NAME)
        return _embedder


def _chunk_text(text: str, target_words: int = 120) -> List[str]:
    """Sentence-aware chunking (same spirit as rag_utils fallback)."""
    if not text or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        wc = len(sentence.split())
        if count + wc > target_words and current:
            chunks.append(" ".join(current))
            current = []
            count = 0
        current.append(sentence)
        count += wc
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def _read_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _load_registry() -> List[Dict[str, Any]]:
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        return []
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load corpus registry: %s", e)
        return []


def _save_registry(entries: List[Dict[str, Any]]) -> None:
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _corpus_dir(corpus_id: str) -> Path:
    return CORPUS_ROOT / corpus_id


def is_vector_search_available() -> bool:
    return HAVE_VECTOR_SEARCH and _get_embedder() is not None


@dataclass
class CorpusChunk:
    chunk_id: str
    text: str
    source_file: str
    source_path: str
    char_start: int
    char_end: int
    first_person_ratio: float
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "first_person_ratio": self.first_person_ratio,
            "word_count": self.word_count,
        }


def list_corpora() -> List[Dict[str, Any]]:
    return _load_registry()


def get_corpus_meta(corpus_id: str) -> Optional[Dict[str, Any]]:
    for entry in _load_registry():
        if entry.get("id") == corpus_id:
            return entry
    return None


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def get_index_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        if job_id in _index_jobs:
            return dict(_index_jobs[job_id])
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read job file %s: %s", path, e)
        return None


def _set_job(job_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    with _jobs_lock:
        cur = dict(_index_jobs.get(job_id) or {})
        if not cur:
            path = _job_path(job_id)
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        cur = json.load(f)
                except (json.JSONDecodeError, OSError):
                    cur = {}
        cur.update(patch)
        cur["job_id"] = job_id
        cur["updated_at"] = _utc_now()
        _index_jobs[job_id] = cur
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_job_path(job_id), "w", encoding="utf-8") as f:
                json.dump(cur, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("Failed to persist job %s: %s", job_id, e)
        phase = cur.get("phase") or cur.get("status")
        logger.info(
            "Corpus job %s: %s — %s",
            job_id,
            phase,
            cur.get("message") or cur.get("current_file") or "",
        )
        return cur


def _scan_txt_files(folder: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.txt" if recursive else "*.txt"
    files = sorted(folder.glob(pattern))
    return [p for p in files if p.is_file()]


def _build_chunks_for_file(
    file_path: Path,
    folder: Path,
    corpus_id: str,
    file_index: int,
) -> List[CorpusChunk]:
    text = _read_txt(file_path)
    rel = str(file_path.relative_to(folder)).replace("\\", "/")
    chunks_out: List[CorpusChunk] = []
    offset = 0
    for chunk_idx, chunk_text in enumerate(_chunk_text(text)):
        start = text.find(chunk_text, offset)
        if start < 0:
            start = offset
        end = start + len(chunk_text)
        offset = end
        chunk_id = f"{corpus_id}_{file_index:04d}_{chunk_idx:04d}"
        chunks_out.append(
            CorpusChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_file=rel,
                source_path=str(file_path.resolve()),
                char_start=start,
                char_end=end,
                first_person_ratio=first_person_ratio(chunk_text),
                word_count=len(chunk_text.split()),
            )
        )
    return chunks_out


def index_folder(
    folder_path: str,
    *,
    corpus_name: Optional[str] = None,
    corpus_id: Optional[str] = None,
    recursive: bool = True,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not is_vector_search_available():
        raise RuntimeError(
            "Vector search dependencies missing. Install: pip install sentence-transformers faiss-cpu"
        )

    folder = Path(folder_path).expanduser().resolve()
    logger.info("Corpus index_folder: path=%s recursive=%s job=%s", folder, recursive, job_id)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if job_id:
        _set_job(
            job_id,
            {
                "status": "running",
                "phase": "scanning",
                "message": f"Scanning for .txt files in {folder}…",
            },
        )

    txt_files = _scan_txt_files(folder, recursive)
    logger.info("Corpus index: found %d .txt files in %s", len(txt_files), folder)
    if not txt_files:
        raise ValueError(f"No .txt files found in {folder}")

    corpus_id = corpus_id or str(uuid.uuid4())[:12]
    name = (corpus_name or folder.name).strip() or corpus_id
    cdir = _corpus_dir(corpus_id)
    cdir.mkdir(parents=True, exist_ok=True)

    if job_id:
        _set_job(
            job_id,
            {
                "status": "running",
                "phase": "chunking",
                "corpus_id": corpus_id,
                "folder_path": str(folder),
                "files_total": len(txt_files),
                "files_done": 0,
                "chunks_indexed": 0,
                "message": f"Chunking {len(txt_files)} files…",
                "started_at": _utc_now(),
            },
        )

    all_chunks: List[CorpusChunk] = []
    for fi, fp in enumerate(txt_files):
        logger.info("Corpus index chunking file %d/%d: %s", fi + 1, len(txt_files), fp.name)
        all_chunks.extend(_build_chunks_for_file(fp, folder, corpus_id, fi))
        if job_id:
            _set_job(
                job_id,
                {
                    "phase": "chunking",
                    "files_done": fi + 1,
                    "chunks_indexed": len(all_chunks),
                    "current_file": fp.name,
                    "message": f"Chunking file {fi + 1}/{len(txt_files)}: {fp.name}",
                },
            )

    if not all_chunks:
        raise ValueError("No text chunks could be created from the files")

    texts = [c.text for c in all_chunks]
    if job_id:
        _set_job(
            job_id,
            {
                "phase": "loading_model",
                "message": "Loading embedding model (first run may take 1–2 minutes)…",
                "chunks_indexed": len(all_chunks),
            },
        )
    logger.info("Corpus index: loading embedding model for %d chunks", len(texts))
    model = _get_embedder()

    if job_id:
        _set_job(
            job_id,
            {
                "phase": "embedding",
                "embed_total": len(texts),
                "embed_done": 0,
                "message": f"Embedding {len(texts)} chunks (this can take several minutes)…",
            },
        )
    logger.info("Corpus index: embedding %d chunks…", len(texts))
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if job_id:
        _set_job(job_id, {"embed_done": len(texts), "phase": "embedding", "message": "Embeddings complete, saving index…"})
    faiss.normalize_L2(embeddings)

    if job_id:
        _set_job(job_id, {"phase": "saving", "message": "Writing FAISS index and metadata…"})
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(cdir / "index.faiss"))
    with open(cdir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")

    meta = {
        "id": corpus_id,
        "name": name,
        "source_folder": str(folder),
        "recursive": recursive,
        "file_count": len(txt_files),
        "chunk_count": len(all_chunks),
        "indexed_at": _utc_now(),
        "embedding_model": EMBEDDING_MODEL_NAME,
    }
    with open(cdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    registry = _load_registry()
    registry = [e for e in registry if e.get("id") != corpus_id]
    registry.append(meta)
    _save_registry(registry)

    result = {"ok": True, **meta}
    logger.info(
        "Corpus index complete: id=%s name=%s files=%d chunks=%d",
        corpus_id,
        name,
        len(txt_files),
        len(all_chunks),
    )
    if job_id:
        _set_job(
            job_id,
            {
                "status": "completed",
                "phase": "done",
                "message": f"Done — {len(all_chunks)} chunks from {len(txt_files)} files",
                "finished_at": _utc_now(),
                **result,
            },
        )
    return result


def start_index_job(
    folder_path: str,
    *,
    corpus_name: Optional[str] = None,
    corpus_id: Optional[str] = None,
    recursive: bool = True,
) -> str:
    job_id = str(uuid.uuid4())[:12]
    logger.info("Corpus index job queued: %s folder=%s", job_id, folder_path)
    _set_job(
        job_id,
        {
            "status": "queued",
            "phase": "queued",
            "folder_path": folder_path,
            "message": "Waiting to start…",
        },
    )

    def _run():
        try:
            _set_job(job_id, {"status": "running", "phase": "starting", "message": "Index job started"})
            index_folder(
                folder_path,
                corpus_name=corpus_name,
                corpus_id=corpus_id,
                recursive=recursive,
                job_id=job_id,
            )
        except Exception as e:
            logger.exception("Corpus index job failed")
            _set_job(
                job_id,
                {
                    "status": "failed",
                    "phase": "failed",
                    "error": str(e),
                    "message": str(e),
                    "finished_at": _utc_now(),
                },
            )

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def _load_chunks(corpus_id: str) -> List[Dict[str, Any]]:
    path = _corpus_dir(corpus_id) / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Corpus chunks not found: {corpus_id}")
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _load_faiss(corpus_id: str):
    path = _corpus_dir(corpus_id) / "index.faiss"
    if not path.exists():
        raise FileNotFoundError(f"Corpus index not found: {corpus_id}")
    return faiss.read_index(str(path))


def search_corpus(
    corpus_id: str,
    query: str,
    *,
    top_k: int = 25,
    offset: int = 0,
    min_score: float = 0.15,
    min_first_person: Optional[float] = None,
    max_first_person: Optional[float] = None,
    source_file_contains: Optional[str] = None,
    keyword: Optional[str] = None,
) -> Dict[str, Any]:
    if not is_vector_search_available():
        raise RuntimeError("Vector search not available")

    meta = get_corpus_meta(corpus_id)
    if not meta:
        raise KeyError(f"Unknown corpus: {corpus_id}")

    q = (query or "").strip()
    if not q:
        raise ValueError("Query is required")

    chunks = _load_chunks(corpus_id)
    index = _load_faiss(corpus_id)

    model = _get_embedder()
    q_emb = model.encode([q], convert_to_numpy=True)[0]
    q_emb = q_emb / np.linalg.norm(q_emb)
    q_emb = np.expand_dims(q_emb.astype("float32"), axis=0)

    # Search the whole index so result count reflects relevance, not a fixed page size.
    search_k = len(chunks)
    scores, indices = index.search(q_emb, search_k)

    kw_lower = (keyword or "").strip().lower()
    file_filter = (source_file_contains or "").strip().lower()
    limit = max(1, min(int(top_k), 200))
    offset = max(0, int(offset))

    all_matches: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        if float(score) < min_score:
            continue
        ch = chunks[int(idx)]
        fp_ratio = float(ch.get("first_person_ratio", 0))
        if min_first_person is not None and fp_ratio < min_first_person:
            continue
        if max_first_person is not None and fp_ratio > max_first_person:
            continue
        if file_filter and file_filter not in ch.get("source_file", "").lower():
            continue
        text = ch.get("text", "")
        if kw_lower and kw_lower not in text.lower():
            continue
        all_matches.append(
            {
                "score": round(float(score), 4),
                "chunk_id": ch.get("chunk_id"),
                "text": text,
                "source_file": ch.get("source_file"),
                "source_path": ch.get("source_path"),
                "char_start": ch.get("char_start"),
                "char_end": ch.get("char_end"),
                "first_person_ratio": fp_ratio,
                "word_count": ch.get("word_count"),
            }
        )

    total_matches = len(all_matches)
    page = all_matches[offset : offset + limit]
    has_more = (offset + limit) < total_matches
    scores_on_page = [r["score"] for r in page]

    return {
        "corpus_id": corpus_id,
        "corpus_name": meta.get("name"),
        "query": q,
        "result_count": len(page),
        "total_matches": total_matches,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "next_offset": offset + limit if has_more else None,
        "min_score": min_score,
        "score_range": {
            "page_min": min(scores_on_page) if scores_on_page else None,
            "page_max": max(scores_on_page) if scores_on_page else None,
            "best_overall": all_matches[0]["score"] if all_matches else None,
            "weakest_on_page": min(scores_on_page) if scores_on_page else None,
        },
        "results": page,
    }


def delete_corpus(corpus_id: str) -> bool:
    import shutil

    registry = _load_registry()
    new_reg = [e for e in registry if e.get("id") != corpus_id]
    if len(new_reg) == len(registry):
        return False
    _save_registry(new_reg)
    cdir = _corpus_dir(corpus_id)
    if cdir.exists():
        shutil.rmtree(cdir, ignore_errors=True)
    return True
