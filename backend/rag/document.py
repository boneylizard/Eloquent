import os
import asyncio
import json
import logging
import mimetypes
from typing import List, Dict, Any

import docx
import csv
import numpy as np
import faiss

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Use llama-cpp-python embedding model
from llama_cpp import Llama

LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "")  # Set via environment variable or settings
embedding_model = None

# Initialize embedding model
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = Llama(model_path=LLAMA_MODEL_PATH, embedding=True)
    return embedding_model

# Text extraction functions
def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Generic text extraction
def extract_text(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_text_from_docx(file_path)
    elif mime_type == 'text/plain':
        return open(file_path, encoding='utf-8').read()
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

# Simple text chunking
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Generate embeddings using llama-cpp-python
def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    embeddings = [model.embed(chunk) for chunk in chunks]
    return embeddings

# FAISS indexing
def create_faiss_index(embeddings: List[List[float]], doc_id: str) -> None:
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    index_path = os.path.join(PROCESSED_DIR, f"{doc_id}.index")
    faiss.write_index(index, index_path)

# Document processing
async def process_document(file_path: str) -> dict:
    doc_id = os.path.basename(file_path)

    text = extract_text(file_path)
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

    embeddings = await asyncio.to_thread(generate_faiss_embeddings, chunks)
    create_faiss_index(embeddings, doc_id)

    doc_info = {
        "id": doc_id,
        "chunk_count": len(chunks)
    }

    json_path = os.path.join(PROCESSED_DIR, f"{doc_id}.json")
    with open(json_path, 'w') as json_file:
        json.dump(doc_info, json_file)

    logging.info(f"Processed document {doc_id} with {len(chunks)} chunks")
    return doc_info
