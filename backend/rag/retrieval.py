import os
import json
import logging
import asyncio
from typing import List, Dict, Any
import faiss
import numpy as np

# Import from document module to reuse embedding model
from rag.document import get_embedding_model, PROCESSED_DIR

async def retrieve_context(
    query: str, 
    doc_ids: List[str], 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant document chunks based on the query
    """
    try:
        # Get embedding model
        model = get_embedding_model()
        
        # Generate query embedding
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: model.encode([query])[0]
        )
        
        all_results = []
        
        # Process each document
        for doc_id in doc_ids:
            try:
                # Load document info
                doc_info_path = os.path.join(PROCESSED_DIR, f"{doc_id}.json")
                if not os.path.exists(doc_info_path):
                    logging.warning(f"Document info not found for {doc_id}")
                    continue
                
                with open(doc_info_path, "r") as f:
                    doc_info = json.load(f)
                
                # Load FAISS index
                index_path = os.path.join(PROCESSED_DIR, f"{doc_id}.index")
                if not os.path.exists(index_path):
                    logging.warning(f"FAISS index not found for {doc_id}")
                    continue
                
                index = await loop.run_in_executor(
                    None,
                    lambda: faiss.read_index(index_path)
                )
                
                # Search the index
                query_embedding_reshaped = np.array([query_embedding]).astype('float32')
                D, I = await loop.run_in_executor(
                    None,
                    lambda: index.search(query_embedding_reshaped, min(top_k, len(doc_info["chunks"])))
                )
                
                # Get the chunks
                for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                    if idx < len(doc_info["chunks"]):
                        chunk = doc_info["chunks"][idx]
                        all_results.append({
                            "doc_id": doc_id,
                            "filename": doc_info["filename"],
                            "chunk_id": chunk["id"],
                            "text": chunk["text"],
                            "score": float(1.0 / (1.0 + distance))  # Convert distance to similarity score
                        })
            except Exception as e:
                logging.error(f"Error retrieving from document {doc_id}: {str(e)}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
        
    except Exception as e:
        logging.error(f"Error in retrieve_context: {str(e)}")
        raise e

async def search_documents(
    query: str,
    doc_ids: List[str] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Search across all processed documents or a subset of documents
    """
    try:
        # If no doc_ids provided, search all processed documents
        if not doc_ids:
            doc_ids = []
            for filename in os.listdir(PROCESSED_DIR):
                if filename.endswith(".json"):
                    doc_ids.append(filename.replace(".json", ""))
        
        results = await retrieve_context(query, doc_ids, top_k)
        
        # Group results by document
        grouped_results = {}
        for result in results:
            doc_id = result["doc_id"]
            if doc_id not in grouped_results:
                grouped_results[doc_id] = {
                    "doc_id": doc_id,
                    "filename": result["filename"],
                    "chunks": []
                }
            grouped_results[doc_id]["chunks"].append({
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "score": result["score"]
            })
        
        return {
            "query": query,
            "total_results": len(results),
            "documents": list(grouped_results.values())
        }
        
    except Exception as e:
        logging.error(f"Error in search_documents: {str(e)}")
        raise e