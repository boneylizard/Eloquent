from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
import os
import httpx

# Create router
memory_router = APIRouter(prefix="/memory", tags=["memory"])

# Models
class Memory(BaseModel):
    content: str
    category: str = "other"  
    importance: float = 0.5
    type: str = "auto"
    tags: List[str] = []

class MemoryObservationRequest(BaseModel):
    user_message: str
    ai_response: str
    model_name: Optional[str] = None

class MemoryEnhancementRequest(BaseModel):
    prompt: str

# Memory storage - temporary for prototype
MEMORIES = []

# Memory categories that match your frontend
MEMORY_CATEGORIES = [
    'personal_info',
    'preferences',
    'interests',
    'facts',
    'skills',
    'opinions',
    'experiences',
    'other'
]

@memory_router.post("/observe")
async def observe_conversation(request: MemoryObservationRequest):
    """
    Process a conversation and extract memories using GPU1.
    This would be called after each user-AI interaction.
    """
    # Log the request
    logger.info(f"🧠 [Memory Agent] Observing conversation")
    logger.info(f"User: {request.user_message[:100]}...")
    logger.info(f"AI: {request.ai_response[:100]}...")

    # Define the input format for the GPU1 model
    model_input = {
        "user_message": request.user_message,
        "ai_response": request.ai_response
    }

    try:
        # Call the GPU1 model for memory extraction
        # Replace 'call_gpu1_model' with your actual function to call the model
        model_output = await call_gpu1_model(model_input, gpu_id=1)

        # Process the model output to extract memories
        extracted_memories = process_model_output(model_output)

        # Store the extracted memories
        memories_added = 0
        for memory_data in extracted_memories:
            # Create a Memory object
            memory = Memory(
                content=memory_data["content"],
                category=memory_data["category"],
                importance=memory_data["importance"],
                type="auto",
                tags=[memory_data["category"]]
            )
            MEMORIES.append(memory.dict())
            memories_added += 1

        return {
            "status": "success",
            "memories_added": memories_added,
            "memories": [m.dict() for m in memories_added]
        }

    except Exception as e:
        logger.error(f"❌ Error during memory extraction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }
    
def process_model_output(model_output):
    """
    Processes the output from the GPU1 model to extract memory objects.

    Args:
        model_output: The output from the GPU1 model.

    Returns:
        A list of dictionaries, where each dictionary represents a memory.
    """
    extracted_memories = []
    for memory_data in model_output:
        extracted_memories.append({
            "content": memory_data["content"],
            "category": memory_data["category"],
            "importance": memory_data["importance"]
        })
    return extracted_memories

@memory_router.post("/enhance")
async def enhance_prompt(request: MemoryEnhancementRequest):
    """
    Find relevant memories to enhance a prompt before sending to GPU0.
    In a full implementation, this would use GPU1 for intelligent memory retrieval.
    """
    prompt = request.prompt
    
    # In a real implementation, we'd use GPU1 to find relevant memories
    # For now, use simple keyword matching
    relevant_memories = []
    for memory in MEMORIES:
        # Simple relevance scoring
        prompt_words = set(prompt.lower().split())
        memory_words = set(memory["content"].lower().split())
        overlap = prompt_words.intersection(memory_words)
        
        if overlap:
            relevant_memories.append(memory)
            if len(relevant_memories) >= 3:  # Limit to 3 most relevant memories
                break
    
    # Format memories for injection
    if relevant_memories:
        memory_context = "\n".join([f"- {m['content']}" for m in relevant_memories])
        return {
            "has_memories": True,
            "memory_context": f"<user_context>\n{memory_context}\n</user_context>",
            "memories": relevant_memories
        }
    
    return {"has_memories": False, "memory_context": "", "memories": []}

@memory_router.get("/list")
async def list_memories():
    """Return all stored memories"""
    return {"memories": MEMORIES}

@memory_router.post("/add")
async def add_memory(memory: Memory):
    """Manually add a memory"""
    memory_dict = memory.dict()
    MEMORIES.append(memory_dict)
    return {"status": "success", "memory": memory_dict}

@memory_router.delete("/delete/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory by its ID (would be implemented in real version)"""
    # In your real implementation, you'd delete from your storage
    return {"status": "success", "message": f"Memory {memory_id} deleted"}

@memory_router.post("/generate")
async def generate_memory_with_model(request: MemoryObservationRequest):
    """
    Use GPU1 to analyze conversation and generate memories.
    This demonstrates using the secondary model for memory operations.
    """
    # This would call your GPU1 model with a prompt like:
    # "Analyze this conversation and extract facts about the user that should be remembered"
    
    # For prototype, we'll just simulate this:
    return {
        "status": "success",
        "message": "This endpoint would use GPU1 to analyze and generate memories"
    }