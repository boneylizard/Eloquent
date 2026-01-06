import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_utils")

# Constants - use relative path based on this file's location
DOCUMENT_STORE_DIR = Path(__file__).parent / "static" / "documents"
DOCUMENT_META_FILE = DOCUMENT_STORE_DIR / "document_meta.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Small but effective model
CHUNK_SIZE = 300  # Token size for chunking documents
CHUNK_OVERLAP = 50  # Overlap between chunks

# Make sure the directory exists
DOCUMENT_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("RAG functionality limited: sentence-transformers not installed")

# Try to import FAISS for vector search
try:
    import faiss
    import numpy as np
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    logger.warning("RAG functionality limited: faiss-cpu not installed")

class RAGProcessor:
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.documents = []
        self.document_chunks = []
        self.chunk_to_doc_mapping = {}
        
        # Initialize if dependencies are available
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available."""
        return HAVE_SENTENCE_TRANSFORMERS and HAVE_FAISS and self.embedding_model is not None
    
    def load_documents(self) -> bool:
        """Load document metadata and prepare for RAG."""
        if not self.is_available():
            print("RAG not available, skipping document loading")
            return False
            
        try:
            # Log the document directory and metadata path
            print(f"Loading documents from directory: {DOCUMENT_STORE_DIR}")
            
            # Load document metadata
            if not DOCUMENT_META_FILE.exists():
                print(f"No document metadata file found at {DOCUMENT_META_FILE}")
                return False
                
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
                return True

            # Reset chunks data
            self.document_chunks = []
            self.chunk_to_doc_mapping = {}
            
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
        """Split text into chunks suitable for embedding."""
        if not text:
            return []
        
        # Simple character-based chunking
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word.split())  # Approximating token size
            if current_size + word_size > CHUNK_SIZE:
                # Start a new chunk if current is full
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Handle overlap
                overlap_start = max(0, len(current_chunk) - CHUNK_OVERLAP)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(w.split()) for w in current_chunk)
            
            current_chunk.append(word)
            current_size += word_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _build_index(self) -> None:
        """Build a FAISS index for the document chunks."""
        if not self.document_chunks:
            print("No chunks to index")
            return
        
        try:
            # Check if we have the dependencies
            if not HAVE_FAISS:
                print("FAISS not available, cannot build index")
                return
                
            # Get embeddings for all chunks
            print(f"Generating embeddings for {len(self.document_chunks)} chunks...")
            embeddings = self.embedding_model.encode(
                self.document_chunks, 
                convert_to_numpy=True, 
                show_progress_bar=False
            )
            
            # Normalize embeddings for cosine similarity
            print("Normalizing embeddings...")
            faiss.normalize_L2(embeddings)
            
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
    
    def query(self, question: str, doc_ids: Optional[List[str]] = None, top_k: int = 5, threshold: float = 0.05) -> List[Dict[str, Any]]:
        threshold = float(threshold)  # Ensure threshold is a float
        print(f"[DEBUG] threshold is {threshold} (type: {type(threshold)})")
        """
        Query the RAG system for relevant document chunks.
        
        Args:
            question (str): The query text
            doc_ids (List[str], optional): Filter to specific document IDs
            top_k (int): Number of top results to return
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            List[Dict]: List of relevant chunks with metadata
        """
        if not self.is_available() or not self.faiss_index:
            print("RAG system not available for query")
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
            query_embedding = self.embedding_model.encode([question])[0]
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

                if doc_ids and doc["id"] not in doc_ids:
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

def query_documents(question: str, doc_ids: Optional[List[str]] = None, top_k: int = 5, threshold: float = 0.05) -> Dict[str, Any]:
    threshold = float(threshold)  # Ensure threshold is a float
    print(f"[DEBUG] threshold is {threshold} (type: {type(threshold)})")
    """
    Query documents for relevant chunks.
    
    Args:
        question (str): The query text
        doc_ids (List[str], optional): Filter to specific document IDs
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