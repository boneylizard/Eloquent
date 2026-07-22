import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .runtime_paths import data_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_utils")

# Constants - use relative path based on this file's location
DOCUMENT_STORE_DIR = data_path("documents")
DOCUMENT_META_FILE = DOCUMENT_STORE_DIR / "document_meta.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Small but effective model

# Chonkie chunking settings
CHUNKING_STRATEGY = "semantic"  # Options: "semantic", "sentence", "token"
# Bigger chunks = more text per hit when RAG returns a chunk (re-process / re-index docs after changing).
MAX_CHUNK_SIZE = 256  # Target tokens per chunk (embedding tokenizer, not your chat model)
MIN_CHUNK_SIZE = 32   # Minimum tokens per chunk
SIMILARITY_THRESHOLD = 0.6  # For semantic chunking - higher = more aggressive splitting

# --- User chat (/generate) when "document context" is ON: knobs for how much gets pasted in ---
# RAG_CHAT_TOP_K: max number of document chunks appended (raise = more context, longer prompt).
# RAG_CHAT_SIMILARITY_THRESHOLD: 0–1 cosine similarity; LOWER = keep weaker matches (more noise).
RAG_CHAT_TOP_K = int(os.environ.get("ELOQUENT_RAG_CHAT_TOP_K", "48"))
RAG_CHAT_SIMILARITY_THRESHOLD = float(os.environ.get("ELOQUENT_RAG_CHAT_THRESHOLD", "0.22"))

# Make sure the directory exists
DOCUMENT_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("RAG functionality limited: sentence-transformers not installed")

# Try to import torch for device detection
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Try to import FAISS for vector search
try:
    import faiss
    import numpy as np
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    logger.warning("RAG functionality limited: faiss-cpu not installed")

# Try to import Chonkie for smart chunking
try:
    from chonkie import SemanticChunker, SentenceChunker, TokenChunker
    HAVE_CHONKIE = True
    logger.info("Chonkie chunking library available")
except ImportError:
    HAVE_CHONKIE = False
    logger.warning("Chonkie not installed - using fallback chunking. Install with: pip install chonkie")

class RAGProcessor:
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.documents = []
        self.document_chunks = []
        self.chunk_to_doc_mapping = {}
        self.chunker = None
        self.device = "cpu"
        # Cached L2-normalised embeddings, one row per chunk in self.document_chunks.
        # Kept in sync by _build_index so deletions can rebuild the FAISS index
        # without re-encoding every remaining chunk from scratch.
        self._embeddings = None
        
        # Initialize embedding model
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                # Respect explicit env preference; otherwise auto-detect.
                # Broken CUDA drivers can throw at encode time, so we fall back to CPU on errors.
                device_pref = os.environ.get("RAG_EMBEDDING_DEVICE", "auto").lower()
                if device_pref == "auto":
                    device_pref = "cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu"
                self.device = device_pref
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
                logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME} on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
                self.device = "cpu"
        
        # Initialize Chonkie chunker
        if HAVE_CHONKIE and self.embedding_model:
            try:
                self._init_chunker()
            except Exception as e:
                logger.error(f"Failed to initialize Chonkie chunker: {e}")
                self.chunker = None
    
    def _init_chunker(self):
        """Initialize the Chonkie chunker based on strategy."""
        if not HAVE_CHONKIE:
            return
            
        try:
            if CHUNKING_STRATEGY == "semantic":
                # Semantic chunking groups text by meaning
                self.chunker = SemanticChunker(
                    embedding_model=EMBEDDING_MODEL_NAME,
                    chunk_size=MAX_CHUNK_SIZE,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                )
                logger.info(f"Initialized SemanticChunker (max {MAX_CHUNK_SIZE} tokens, threshold {SIMILARITY_THRESHOLD})")
            elif CHUNKING_STRATEGY == "sentence":
                # Sentence-based chunking keeps sentences together
                self.chunker = SentenceChunker(
                    chunk_size=MAX_CHUNK_SIZE,
                    chunk_overlap=MIN_CHUNK_SIZE // 2,
                )
                logger.info(f"Initialized SentenceChunker (max {MAX_CHUNK_SIZE} tokens)")
            else:
                # Token-based chunking (fallback)
                self.chunker = TokenChunker(
                    chunk_size=MAX_CHUNK_SIZE,
                    chunk_overlap=MIN_CHUNK_SIZE // 2,
                )
                logger.info(f"Initialized TokenChunker (max {MAX_CHUNK_SIZE} tokens)")
        except Exception as e:
            logger.error(f"Chunker initialization failed: {e}")
            self.chunker = None
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available."""
        return HAVE_SENTENCE_TRANSFORMERS and HAVE_FAISS and self.embedding_model is not None
    
    def _encode_safe(self, texts, **kwargs):
        """Encode texts, automatically falling back to CPU on CUDA errors."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not available")
        
        try:
            return self.embedding_model.encode(texts, device=self.device, **kwargs)
        except Exception as e:
            err_lower = str(e).lower()
            if self.device != "cpu" and ("cuda" in err_lower or "accelerator" in err_lower):
                logger.warning(f"Embedding failed on {self.device}: {e}. Falling back to CPU.")
                try:
                    self.embedding_model = self.embedding_model.to("cpu")
                except Exception as move_err:
                    logger.warning(f"Could not move model to CPU: {move_err}")
                self.device = "cpu"
                return self.embedding_model.encode(texts, device="cpu", **kwargs)
            raise
    
    def load_documents(self) -> bool:
        """Load document metadata and prepare for RAG."""
        if not self.is_available():
            print("RAG not available, skipping document loading")
            return False
            
        try:
            # Log the document directory and metadata path
            print(f"Loading documents from directory: {DOCUMENT_STORE_DIR}")
            
            # Load document metadata (create empty file if missing so RAG is still "available")
            if not DOCUMENT_META_FILE.exists():
                print(f"No document metadata file found at {DOCUMENT_META_FILE}; initializing empty RAG index.")
                self.documents = []
                self.document_chunks = []
                self.chunk_to_doc_mapping = {}
                self._embeddings = None
                self.faiss_index = None
                DOCUMENT_META_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(DOCUMENT_META_FILE, "w", encoding="utf-8") as f:
                    json.dump([], f)
                return True

            with open(DOCUMENT_META_FILE, "r", encoding="utf-8") as f:
                try:
                    self.documents = json.load(f)
                    print(f"Loaded {len(self.documents)} documents from metadata")
                except json.JSONDecodeError as e:
                    print(f"JSON error in metadata file: {e}")
                    return False
                
            if not self.documents:
                print("No documents found in metadata, but allowing RAG system to initialize for conversation storage")
                # Initialize empty but working RAG system
                self.documents = []
                self.document_chunks = []
                self.chunk_to_doc_mapping = {}
                self._embeddings = None
                self.faiss_index = None
                return True

            # Reset chunks data
            self.document_chunks = []
            self.chunk_to_doc_mapping = {}
            self._embeddings = None
            
            # Process each document
            successful_docs = 0
            for doc_index, doc in enumerate(self.documents):
                # Log the document being processed
                print(f"Processing document {doc_index+1}/{len(self.documents)}: {doc.get('filename', 'unknown')}")
                doc_id = doc.get('id', 'unknown')
                
                # Get the document content
                text_file_path = DOCUMENT_STORE_DIR / doc.get("text_filename", "")
                if not text_file_path.exists():
                    print(f"Text content not found for document: {doc.get('filename', 'unknown')} at {text_file_path}")
                    continue
                    
                try:
                    with open(text_file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        print(f"Read {len(content)} characters from {text_file_path}")
                except UnicodeDecodeError:
                    print(f"Unicode error reading {text_file_path}, trying with latin-1 encoding")
                    try:
                        with open(text_file_path, "r", encoding="latin-1") as f:
                            content = f.read()
                        print(f"Successfully read file with latin-1 encoding")
                    except Exception as read_err:
                        print(f"Failed to read document {doc_id}: {read_err}")
                        continue
                except Exception as e:
                    print(f"Error reading document {doc_id}: {e}")
                    continue
                
                # Chunk the document
                chunks = self._chunk_text(content)
                print(f"Created {len(chunks)} chunks from document")
                
                # Store chunks with document mapping
                chunk_count = 0
                for chunk_text in chunks:
                    if not chunk_text.strip():  # Skip empty chunks
                        continue
                    chunk_index = len(self.document_chunks)
                    self.document_chunks.append(chunk_text)
                    self.chunk_to_doc_mapping[chunk_index] = doc_index
                    chunk_count += 1
                
                print(f"Added {chunk_count} non-empty chunks from document {doc_id}")
                successful_docs += 1
            
            # Build the FAISS index
            if self.document_chunks:
                print(f"Building FAISS index with {len(self.document_chunks)} chunks from {successful_docs} documents")
                self._build_index()
                print("FAISS index built successfully")
                return True
            else:
                print("No document chunks extracted, index not built")
                return False
                    
        except Exception as e:
            print(f"Error loading documents for RAG: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into semantically meaningful chunks using Chonkie."""
        if not text or not text.strip():
            return []
        
        # Use Chonkie if available
        if self.chunker:
            try:
                chonkie_chunks = self.chunker.chunk(text)
                # Extract text from Chunk objects
                chunks = [chunk.text for chunk in chonkie_chunks if chunk.text.strip()]
                
                if chunks:
                    avg_len = sum(len(c.split()) for c in chunks) / len(chunks)
                    logger.info(f"Chonkie created {len(chunks)} chunks (avg ~{avg_len:.0f} words each)")
                    return chunks
            except Exception as e:
                logger.warning(f"Chonkie chunking failed, using fallback: {e}")
        
        # Fallback: sentence-based chunking without Chonkie
        return self._fallback_chunk_text(text)
    
    def _fallback_chunk_text(self, text: str) -> List[str]:
        """Fallback chunking when Chonkie is not available - sentence-based."""
        import re
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        target_words = 80  # ~100-150 words per chunk instead of 300+
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            word_count = len(sentence.split())
            
            # If adding this sentence would exceed target, save current chunk
            if current_word_count + word_count > target_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            current_chunk.append(sentence)
            current_word_count += word_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        if chunks:
            avg_len = sum(len(c.split()) for c in chunks) / len(chunks)
            logger.info(f"Fallback created {len(chunks)} chunks (avg ~{avg_len:.0f} words each)")
        
        return chunks
    
    def _build_index(self) -> None:
        """Build a FAISS index for the document chunks.

        Encodes every chunk fresh and caches the L2-normalised result in
        self._embeddings so subsequent deletions can rebuild the index without
        re-encoding (see _rebuild_index_from_cache).
        """
        if not self.document_chunks:
            print("No chunks to index")
            self._embeddings = None
            self.faiss_index = None
            return

        try:
            # Check if we have the dependencies
            if not HAVE_FAISS:
                print("FAISS not available, cannot build index")
                return

            # Get embeddings for all chunks
            print(f"Generating embeddings for {len(self.document_chunks)} chunks...")
            embeddings = self._encode_safe(
                self.document_chunks,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Normalize embeddings for cosine similarity
            print("Normalizing embeddings...")
            faiss.normalize_L2(embeddings)

            # Cache so deletions can rebuild without re-encoding
            self._embeddings = embeddings

            # Create and populate index
            vector_dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity
            self.faiss_index.add(embeddings)

            print(f"FAISS index built with {len(self.document_chunks)} chunks")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            import traceback
            traceback.print_exc()
            self.faiss_index = None
            self._embeddings = None

    def _rebuild_index_from_cache(self) -> None:
        """Rebuild the FAISS index from cached embeddings without re-encoding.

        self._embeddings has already been reduced (by remove_document) to only
        the rows that still correspond to self.document_chunks, so we simply
        wrap the surviving matrix in a fresh IndexFlatIP.
        """
        if not HAVE_FAISS:
            print("FAISS not available, cannot rebuild index")
            self.faiss_index = None
            return

        if not self.document_chunks or self._embeddings is None:
            self.faiss_index = None
            return

        # Sanity: embeddings matrix should match chunk count
        if self._embeddings.shape[0] != len(self.document_chunks):
            print(
                f"Embedding cache out of sync ({self._embeddings.shape[0]} rows "
                f"vs {len(self.document_chunks)} chunks) — falling back to full rebuild"
            )
            self._build_index()
            return

        embeddings = np.ascontiguousarray(self._embeddings)
        vector_dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(vector_dimension)
        self.faiss_index.add(embeddings)
        print(f"FAISS index rebuilt from cache with {len(self.document_chunks)} chunks (no re-encoding)")
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the in-memory RAG index without re-reading
        all files from disk and without re-encoding the surviving chunks.

        Surviving chunks, their doc mapping, and cached embeddings are rebuilt
        as contiguous arrays so the FAISS index stays consistent after the
        deletion.
        """
        doc_index = None
        for i, doc in enumerate(self.documents):
            if doc.get("id") == document_id:
                doc_index = i
                break

        if doc_index is None:
            return False

        # Remove the document from the metadata list
        self.documents.pop(doc_index)

        # Walk the existing parallel structures once, keeping only chunks that
        # did NOT belong to the removed document, and shifting the recorded
        # doc index for every chunk belonging to documents that came after it.
        old_chunks = self.document_chunks
        old_mapping = self.chunk_to_doc_mapping
        old_embeddings = self._embeddings

        new_chunks = []
        new_mapping = {}
        new_rows = []

        for old_idx, chunk_text in enumerate(old_chunks):
            old_doc_idx = old_mapping.get(old_idx)

            # Chunk belonged to the deleted document — drop it.
            if old_doc_idx == doc_index:
                continue

            # Shift doc index down by one for documents after the removed one.
            if old_doc_idx is not None and old_doc_idx > doc_index:
                old_doc_idx -= 1

            new_idx = len(new_chunks)
            new_chunks.append(chunk_text)
            new_mapping[new_idx] = old_doc_idx

            if old_embeddings is not None and old_idx < old_embeddings.shape[0]:
                new_rows.append(old_embeddings[old_idx])

        self.document_chunks = new_chunks
        self.chunk_to_doc_mapping = new_mapping
        if new_rows:
            self._embeddings = np.ascontiguousarray(np.stack(new_rows))
        else:
            self._embeddings = None

        # Rebuild FAISS index from cached embeddings (no file I/O, no re-encoding)
        if self.document_chunks:
            self._rebuild_index_from_cache()
        else:
            self.faiss_index = None
            self._embeddings = None

        logger.info(f"Removed document {document_id} from RAG index without full refresh or re-encoding")
        return True

    def query(self, question: str, doc_ids: Optional[List[str]] = None, top_k: int = 30, threshold: float = 0.05) -> List[Dict[str, Any]]:
        threshold = float(threshold)  # Ensure threshold is a float
        print(f"[DEBUG] threshold is {threshold} (type: {type(threshold)})")
        """
        Query the RAG system for relevant document chunks.
        
        Args:
            question (str): The query text
            doc_ids: None = search all indexed docs; non-empty list = only those IDs;
                empty list = no documents (UI: context enabled, nothing selected).
            top_k (int): Number of top results to return
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            List[Dict]: List of relevant chunks with metadata
        """
        if not self.is_available() or not self.faiss_index:
            print("RAG system not available for query")
            return []

        # Explicit empty selection (e.g. document context ON but no files checked): must not
        # fall through as falsy and skip filtering — that would search the entire index.
        if isinstance(doc_ids, list) and len(doc_ids) == 0:
            print("RAG Query: empty doc_ids list — returning no chunks")
            return []

        try:
            print(f"RAG Query: '{question[:50]}...' with {len(doc_ids or [])} doc IDs")
            print(f"Available documents: {len(self.documents)}")
            if doc_ids:
                print(f"Filtering to document IDs: {doc_ids}")
                # Log if any requested docs aren't found
                for doc_id in doc_ids:
                    if not any(d.get('id') == doc_id for d in self.documents):
                        print(f"Requested document ID not found: {doc_id}")
            
            # Get embedding for the query
            print("Generating query embedding...")
            query_embedding = self._encode_safe([question])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
            query_embedding = np.expand_dims(query_embedding, axis=0)  # Add batch dimension
            
            # Search the index
            print(f"Searching FAISS index with {len(self.document_chunks)} chunks")
            search_k = min(top_k * 2, len(self.document_chunks))  # Get more and filter later
            scores, indices = self.faiss_index.search(query_embedding, search_k)
            
            # Log search results
            print(f"FAISS returned {len(indices[0])} results with top scores: {scores[0][:5]}")
            
            # Extract and format results
            results = []
            for score, chunk_idx in zip(scores[0], indices[0]):
                print(f"→ Chunk {chunk_idx} scored {score:.4f}")

                if score < threshold:
                    print("   ✖ Skipped: below threshold")
                    continue

                doc_idx = self.chunk_to_doc_mapping.get(int(chunk_idx))
                if doc_idx is None:
                    print(f"   ✖ Skipped: no doc mapping for chunk {chunk_idx}")
                    continue

                try:
                    doc = self.documents[doc_idx]
                except IndexError:
                    print(f"   ✖ Skipped: invalid document index {doc_idx} for chunk {chunk_idx}")
                    continue

                if doc_ids is not None and doc["id"] not in doc_ids:
                    print(f"   ✖ Skipped: doc ID {doc['id']} not in {doc_ids}")
                    continue

                print(f"   ✔ Accepted chunk from '{doc['filename']}' (doc ID: {doc['id']})")

                results.append({
                    "chunk": self.document_chunks[chunk_idx],
                    "score": float(score),
                    "document": {
                        "id": doc["id"],
                        "filename": doc["filename"],
                        "file_type": doc.get("file_type", "unknown")
                    }
                })

                if len(results) >= top_k:
                    break
            
            print(f"Returning {len(results)} relevant chunks")
            return results
                
        except Exception as e:
            print(f"Error during RAG query: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def format_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for inclusion in the prompt."""
        if not chunks:
            return ""
        
        formatted_text = "RELEVANT DOCUMENT SECTIONS:\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            doc_info = chunk["document"]
            formatted_text += f"[DOC {i}] {doc_info['filename']}\n"
            formatted_text += f"{chunk['chunk']}\n\n"
        
        formatted_text += "Please use the above document sections to inform your response.\n"
        return formatted_text


# Singleton instance
rag_processor = RAGProcessor()


# API Functions
def is_rag_available() -> bool:
    """Check if RAG functionality is available."""
    return rag_processor.is_available()


def refresh_rag_index() -> bool:
    """Refresh the RAG index with current documents."""
    return rag_processor.load_documents()
def store_conversation_chunk(speaker, content, topic, conversation_id):
    """Store a conversation fragment in RAG for later retrieval."""
    chunk_text = f"[{speaker}]: {content}"
    
    # Add to document chunks with special metadata
    chunk_index = len(rag_processor.document_chunks)
    rag_processor.document_chunks.append(chunk_text)
    
    # Store in a special "conversation" document
    conv_doc = {
        "id": f"conv_{conversation_id}_{chunk_index}",
        "filename": f"Conversation_{topic}",
        "file_type": "conversation",
        "speaker": speaker,
        "topic": topic
    }
    
    # Rebuild index (or add incrementally)
    rag_processor._build_index()

def query_documents(question: str, doc_ids: Optional[List[str]] = None, top_k: int = 30, threshold: float = 0.05) -> Dict[str, Any]:
    threshold = float(threshold)  # Ensure threshold is a float
    print(f"[DEBUG] threshold is {threshold} (type: {type(threshold)})")
    """
    Query documents for relevant chunks.
    
    Args:
        question (str): The query text
        doc_ids: None = all docs; non-empty list = filter; [] = no docs
        top_k (int): Number of top results to return
        threshold (float): Similarity threshold (0-1)
    
    Returns:
        Dict: Query results and context
    """
    print(f"query_documents called with question: '{question[:50]}...', doc_ids: {doc_ids}")
    
    if not is_rag_available():
        print("RAG functionality not available for query_documents")
        return {
            "status": "error",
            "error": "RAG functionality not available",
            "chunks": [],
            "formatted_context": ""
        }
    
    # Check if documents are loaded
    if not rag_processor.documents:
        print("No documents loaded in RAG processor")
        # Try to refresh the index
        print("Attempting to refresh RAG index")
        success = refresh_rag_index()
        if not success:
            print("Failed to refresh RAG index")
            return {
                "status": "error",
                "error": "No documents available",
                "chunks": [],
                "formatted_context": ""
            }
    
    # Query the RAG system
    print("Querying RAG system for relevant chunks")
    chunks = rag_processor.query(
        question=question,
        doc_ids=doc_ids,
        top_k=top_k,
        threshold=threshold
    )
    
    # Format for prompt inclusion
    formatted_context = rag_processor.format_for_prompt(chunks)
    print(f"Returning {len(chunks)} chunks with formatted context ({len(formatted_context)} chars)")
    
    # Log a preview of the formatted context
    if formatted_context:
        context_preview = formatted_context[:200].replace('\n', ' ') + '...'
        print(f"Context preview: {context_preview}")
    
    return {
        "status": "success",
        "chunks": chunks,
        "formatted_context": formatted_context
    }


def initialize_rag_system() -> bool:
    """Initialize the RAG system on server startup."""
    print("Initializing RAG system...")
    if is_rag_available():
        print("RAG available, refreshing index")
        return refresh_rag_index()
    else:
        print("RAG system not available - missing dependencies")
        return False
