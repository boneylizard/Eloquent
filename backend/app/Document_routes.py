import os
import json
import uuid
import logging
import shutil
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse

# Import third-party document processing libraries
try:
    import fitz  # PyMuPDF for PDF processing
except ImportError:
    fitz = None

try:
    import docx  # python-docx for Word documents
except ImportError:
    docx = None

try:
    import pandas as pd  # pandas for CSV processing
except ImportError:
    pd = None

# Configure logging
logger = logging.getLogger("document_routes")

# Router setup
document_router = APIRouter(tags=["document"])

# Constants - change these to use the static directory
DOCUMENT_STORE_DIR = Path("C:/Users/bpfit/LiangLocal/backend/app/static/documents")
DOCUMENT_META_FILE = DOCUMENT_STORE_DIR / "document_meta.json"

# Make sure the directory exists
DOCUMENT_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Import RAG utilities for index refreshing
try:
    from . import rag_utils
    HAVE_RAG = True
except ImportError:
    HAVE_RAG = False
    logger.warning("RAG utilities not available, index will not be refreshed automatically")

# Helper functions
def load_document_metadata() -> List[Dict[str, Any]]:
    """Load document metadata from the metadata file."""
    if not DOCUMENT_META_FILE.exists():
        return []
    
    try:
        with open(DOCUMENT_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading document metadata: {e}")
        return []

def save_document_metadata(documents: List[Dict[str, Any]]) -> None:
    """Save document metadata to the metadata file."""
    try:
        with open(DOCUMENT_META_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(f"Error saving document metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to save document metadata")

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text content from a PDF file."""
    if fitz is None:
        logger.warning("PyMuPDF not installed, returning empty PDF content")
        return ""
    
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text content from a Word document."""
    if docx is None:
        logger.warning("python-docx not installed, returning empty Word content")
        return ""
    
    try:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting text from Word document: {e}")
        return ""

def extract_text_from_csv(file_path: Path) -> str:
    """Extract text content from a CSV file."""
    if pd is None:
        logger.warning("pandas not installed, returning empty CSV content")
        return ""
    
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        logger.error(f"Error extracting text from CSV: {e}")
        return ""

def get_file_extension(filename: str) -> str:
    """Get the lowercase file extension from a filename."""
    return Path(filename).suffix.lower()

def extract_document_text(file_path: Path, file_extension: str) -> str:
    """Extract text from a document based on its file extension."""
    if file_extension in ['.pdf']:
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_extension in ['.csv']:
        return extract_text_from_csv(file_path)
    elif file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading text file with latin-1 encoding: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""
    else:
        logger.warning(f"Unsupported file extension: {file_extension}")
        return ""

def refresh_rag_index():
    """Refresh the RAG index after document changes."""
    if HAVE_RAG:
        try:
            logger.info("Refreshing RAG index after document changes")
            refresh_success = rag_utils.refresh_rag_index()
            if refresh_success:
                logger.info("RAG index refreshed successfully")
            else:
                logger.warning("Failed to refresh RAG index")
            return refresh_success
        except Exception as e:
            logger.error(f"Error refreshing RAG index: {e}")
            return False
    return False

# Routes
@document_router.get("/document/list")
async def list_documents():
    """List all available documents."""
    documents = load_document_metadata()
    return {"status": "success", "file_list": documents}
@document_router.post("/conversation/store")
async def store_conversation_chunk(request: Request):
    try:
        data = await request.json()
        speaker = data.get("speaker")
        content = data.get("content") 
        topic = data.get("topic", "analysis")
        conversation_id = data.get("conversation_id", "default")
        
        if not speaker or not content:
            raise HTTPException(status_code=400, detail="Speaker and content required")
        
        # Create conversation chunk text
        chunk_text = f"[{speaker}]: {content}"
        
        # Add to RAG processor's document chunks
        if HAVE_RAG and rag_utils.rag_processor.is_available():
            chunk_index = len(rag_utils.rag_processor.document_chunks)
            rag_utils.rag_processor.document_chunks.append(chunk_text)
            rag_utils.rag_processor.chunk_to_doc_mapping[chunk_index] = len(rag_utils.rag_processor.documents)
            
            # Add fake "document" for this conversation
            conv_doc = {
                "id": f"conv_{conversation_id}_{chunk_index}",
                "filename": f"Conversation_{topic}",
                "file_type": "conversation",
                "speaker": speaker,
                "topic": topic
            }
            rag_utils.rag_processor.documents.append(conv_doc)
            
            # Rebuild index
            rag_utils.rag_processor._build_index()
            
            return {"status": "success", "message": "Conversation chunk stored"}
        else:
            return {"status": "error", "message": "RAG not available"}
            
    except Exception as e:
        logger.error(f"Error storing conversation chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@document_router.post("/document/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a new document."""
    try:
        # Generate a unique file ID and safe filename
        file_id = str(uuid.uuid4())
        original_filename = file.filename
        file_extension = get_file_extension(original_filename)
        
        # Check supported file types
        supported_extensions = [
            '.pdf', '.doc', '.docx', '.txt', '.md', 
            '.csv', '.json', '.py', '.js', '.html', '.css'
        ]
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported types: {', '.join(supported_extensions)}"
            )
        
        # Create a safe storage filename
        storage_filename = f"{file_id}{file_extension}"
        file_path = DOCUMENT_STORE_DIR / storage_filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract document text based on file type
        document_text = extract_document_text(file_path, file_extension)
        
        # Save text content alongside the document
        text_file_path = DOCUMENT_STORE_DIR / f"{file_id}.txt"
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(document_text)
        
        # Add metadata
        document_info = {
            "id": file_id,
            "filename": original_filename,
            "storage_filename": storage_filename,
            "text_filename": f"{file_id}.txt",
            "file_type": file_extension[1:],  # Remove the dot
            "size_bytes": os.path.getsize(file_path),
            "upload_date": datetime.datetime.now().isoformat(),
            "content_length": len(document_text)
        }
        
        # Update metadata store
        documents = load_document_metadata()
        documents.append(document_info)
        save_document_metadata(documents)
        
        # Refresh the RAG index after adding a new document
        refresh_rag_index()
        
        return {
            "status": "success",
            "message": f"Document '{original_filename}' uploaded successfully",
            "document": document_info
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
    finally:
        await file.close()

@document_router.delete("/document/delete/{document_id}")
async def delete_document(document_id: str):
    """Delete a document by ID."""
    documents = load_document_metadata()
    
    # Find the document
    document = next((doc for doc in documents if doc["id"] == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete the document files
        storage_file = DOCUMENT_STORE_DIR / document["storage_filename"]
        text_file = DOCUMENT_STORE_DIR / document["text_filename"]
        
        if storage_file.exists():
            storage_file.unlink()
        
        if text_file.exists():
            text_file.unlink()
        
        # Update metadata
        documents = [doc for doc in documents if doc["id"] != document_id]
        save_document_metadata(documents)
        
        # Refresh the RAG index after deleting a document
        refresh_rag_index()
        
        return {
            "status": "success",
            "message": f"Document '{document['filename']}' deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@document_router.get("/document/content/{document_id}")
async def get_document_content(document_id: str):
    """Get the text content of a document by ID."""
    documents = load_document_metadata()
    
    # Find the document
    document = next((doc for doc in documents if doc["id"] == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Get the text content
        text_file_path = DOCUMENT_STORE_DIR / document["text_filename"]
        
        if not text_file_path.exists():
            raise HTTPException(status_code=404, detail="Document text content not found")
        
        with open(text_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {
            "status": "success",
            "document": {
                "id": document["id"],
                "filename": document["filename"],
                "content": content
            }
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting document content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving document content: {str(e)}")

@document_router.post("/refresh-rag-index")
async def refresh_rag_endpoint():
    """Manually refresh the RAG index."""
    if HAVE_RAG:
        success = refresh_rag_index()
        if success:
            return {"status": "success", "message": "RAG index refreshed successfully"}
        else:
            return {"status": "error", "message": "Failed to refresh RAG index"}
    else:
        return {"status": "error", "message": "RAG utilities not available"}

# Export the router
router = document_router