# memory_routes.py - Backend routes for memory operations

from fastapi import APIRouter, Depends, HTTPException, Request, Body, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
import logging
import traceback
import re
from . import memory_intelligence
from . import agentic_memory
from . import persona_realignment
from . import memory_curator_prompt
from . import preview_prompt_save
from . import ethics_review_bundle
from . import inference
from .model_manager import ModelManager
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from . import memory_intelligence
import logging
import datetime
from .memory_intelligence import process_completed_exchange
from sentence_transformers import util
import torch
from .memory_intelligence import similarity_model


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_routes")

async def get_model_manager_from_state(request: Request):
    # This safely accesses the model manager from app.state
    yield request.app.state.model_manager

memory_router = APIRouter(tags=["memory"])

# === Model Classes ===

# Define request model for the new endpoint
class MemoryDetectRequest(BaseModel):
    original_prompt: str
    response_text: str
    model_name: Optional[str] = None
    user_name: Optional[str] = None
    user_id: Optional[str] = None
    gpu_id: Optional[int] = None
    single_gpu_mode: Optional[bool] = None

class MemoryRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    userProfile: Optional[Dict[str, Any]] = None
    systemTime: Optional[str] = None
    requestType: Optional[str] = None
    active_character: Optional[Dict[str, Any]] = None


class ObservationRequest(BaseModel):
    user_message: str
    ai_response: str
    user_name: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    userProfile: Optional[Dict[str, Any]] = None
    systemTime: Optional[str] = None
    conversationId: Optional[str] = None
    memory_creation_settings: Optional[dict] = Field(default_factory=dict)


class Memory(BaseModel):
    content: str
    category: str = "other"
    importance: float = 0.7
    type: str = "auto"
    tags: List[str] = []
    user_id: Optional[str] = None

# === Model Classes ===

class ContextualMemoryRequest(BaseModel):
    conversation_snippet: str
    category: str = "contextual"
    importance: float = 0.6
    user_name: Optional[str] = None


@memory_router.post("/detect_keywords")
async def detect_lore_keywords(
    data: dict = Body(...)
):
    """
    Returns lore content from the character data sent by frontend.
    """
    logger = logging.getLogger("memory_routes")
    
    # VERY VISIBLE LOGGING
    print("=" * 60)
    print("🌍 BACKEND LORE ENDPOINT HIT!")
    print("=" * 60)
    
    message = data.get('message', '')
    active_character = data.get('activeCharacter')
    
    print(f"🌍 BACKEND: Message = '{message}'")
    print(f"🌍 BACKEND: Character = {active_character is not None}")
    
    if not message:
        print("🌍 BACKEND: No message - returning empty")
        return {"lore_content": "", "keywords_found": []}
    
    if not active_character:
        print("🌍 BACKEND: No character - returning empty")
        return {"lore_content": "", "keywords_found": []}
    
    lore_entries = active_character.get('loreEntries', [])
    print(f"🌍 BACKEND: Found {len(lore_entries)} lore entries")
    
    if not lore_entries:
        return {"lore_content": "", "keywords_found": []}
    
    message_lower = message.lower()
    lore_content_parts = []
    keywords_found = []
    
    for entry in lore_entries:
        keywords = entry.get('keywords', [])
        content = entry.get('content', '')
        for keyword in keywords:
            if keyword.strip().lower() in message_lower:
                print(f"🌍 BACKEND: Added lore content: {content[:50]}...")
                lore_content_parts.append(content)
                keywords_found.extend(keywords)
                break
    
    print(f"🌍 BACKEND: Returning {len(lore_content_parts)} triggered lore entries")
    
    return {
        "lore_content": "\n\n".join(lore_content_parts),
        "keywords_found": list(set(keywords_found))
    }


def get_user_id_from_request(request: Request, client_supplied_id: Optional[str] = None) -> Optional[str]:
    """Consistently get user ID with proper fallbacks."""
    user_id = client_supplied_id
    if not user_id:
        try:
            from . import user_utils
            user_id = user_utils.get_active_profile_id()
        except Exception:
            pass
    if not user_id:
        try:
            user_id = request.app.state.active_profile_id
        except Exception:
            pass
    return str(user_id) if user_id else None


@memory_router.post("/detect_intent")
async def detect_memory_intent_api(
    request_obj: Request,
    detect_request: MemoryDetectRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    """
    Detect if a user message contains personal information worth remembering.
    """
    logger = logging.getLogger("memory_routes")
    
    user_id = detect_request.user_id
    user_name = detect_request.user_name
    
    if not user_id:
        logger.info("🧠 [detect_intent] Using default test user ID as fallback")
        user_id = "default_test_user"
    
    logger.info(f"🧠 [detect_intent] Using user_id: {user_id}")
    
    if not user_id and not user_name:
        logger.warning("🧠 [detect_intent] Missing user_id or user_name for memory intent detection")
        raise HTTPException(status_code=400, detail="user_id or user_name is required")
    
    single_gpu_mode = detect_request.single_gpu_mode or False
    gpu_id = detect_request.gpu_id or 0
    
    logger.info(f"🧠 API /detect_intent called on GPU {gpu_id}")
    
    try:
        logger.info(f"Received detect_request: prompt='{detect_request.original_prompt[:50]}...', response='{detect_request.response_text[:50]}...'")
        
        model_name = detect_request.model_name
        if not model_name:
            model_name = await model_manager.find_suitable_model(gpu_id=gpu_id)
        
        if not model_name:
            logger.warning(f"API /detect_intent: No model on GPU {gpu_id}; skipping (returning MEMORY_DETECTED: NO).")
            return {"status": "success", "detection_result": "MEMORY_DETECTED: NO"}
        
        logger.info(f"Using model '{model_name}' for intent detection.")
        
        detection_prompt = f"""Analyze this user message for personal information:

"{detect_request.original_prompt}"

You must respond in EXACTLY this format:

MEMORY_DETECTED: YES
MEMORY_CONTENT: This is where you summarize the key personal information to remember. Just parse the message and extract relevant details.
MEMORY_CATEGORY: personal_info (example value, adjust as needed)
MEMORY_IMPORTANCE: 0.8
"""
        
        logger.info(f"Sending detection prompt to model '{model_name}'")
        result_text = await inference.generate_text(
            model_manager=model_manager,
            model_name=model_name,
            prompt=detection_prompt,
            max_tokens=256,
            temperature=0.2,
            gpu_id=gpu_id,
        )
        
        logger.info(f"API /detect_intent: Inference successful. Result preview: {result_text[:80]}...")
        logger.info(f"Raw detection result TEXT from model '{model_name}':\n---RESULT START---\n{result_text}\n---RESULT END---")
        
        if "MEMORY_DETECTED: NO" in result_text:
            logger.info("Memory detection result: NO memory to store")
            return {"status": "success", "detection_result": "MEMORY_DETECTED: NO"}
        elif "MEMORY_DETECTED: YES" in result_text:
            logger.info("Memory detection result: YES - memory to store")
            
            content_match = re.search(r"MEMORY_CONTENT: (.*?)(?:\n|$)", result_text)
            category_match = re.search(r"MEMORY_CATEGORY: (.*?)(?:\n|$)", result_text)
            importance_match = re.search(r"MEMORY_IMPORTANCE: (.*?)(?:\n|$)", result_text)
            
            if content_match:
                memory_content = content_match.group(1).strip()
                memory_category = category_match.group(1).strip() if category_match else "personal_info"
                memory_importance = float(importance_match.group(1).strip()) if importance_match else 0.8
                
                return {
                    "status": "success",
                    "detection_result": "MEMORY_DETECTED: YES",
                    "memory_content": memory_content,
                    "memory_category": memory_category,
                    "memory_importance": memory_importance,
                }
            else:
                logger.warning("Memory detection returned YES but missing required fields")
                return {"status": "success", "detection_result": "MEMORY_DETECTED: YES", "raw": result_text}
        else:
            logger.warning(f"Memory detection returned unclear result: {result_text[:100]}")
            return {"status": "success", "detection_result": "Memory detection returned invalid result", "raw": result_text}
    
    except Exception as e:
        logger.error(f"API /detect_intent: Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error during memory intent detection: {str(e)}")


@memory_router.post("/relevant")
async def get_relevant_memories(
    request_obj: Request,
    request: MemoryRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    """
    Fetches candidate memories from the specific user's memory_store.json,
    then uses an LLM on GPU 1 to refine and select the most relevant context.
    Uses user_id derived from userProfile, prioritizing 'id'.
    """
    logger = logging.getLogger("memory_routes")
    
    logger.info(f"🧠 Received prompt for LLM-powered memory retrieval: {request.prompt[:100]}...")
    
    user_profile = request.userProfile or {}
    user_id = (
        user_profile.get("id") or
        user_profile.get("userId") or
        user_profile.get("user_id")
    )
    if user_id:
        user_id = str(user_id)
    
    logger.info(f"Extracted user_id: '{user_id}' for memory retrieval.")
    
    if not user_id:
        logger.warning("Cannot retrieve relevant memories without a valid 'id', 'userId', or 'user_id' in userProfile.")
        return {
            "status": "user_id_missing",
            "context": "",
            "message": "Cannot retrieve relevant memories without a valid user ID in userProfile."
        }
    
    try:
        single_gpu_mode = getattr(request_obj.app.state, "single_gpu_mode", False)
        gpu_id = 0 if single_gpu_mode else 1
        
        logger.info(f"Using GPU {gpu_id} for memory refinement (single_gpu_mode: {single_gpu_mode})")
        
        memories = memory_intelligence.get_all_memories_for_user(user_id)
        logger.info(f"🧠 [relevant] Loaded {len(memories)} memories for user '{user_id}'")
        
        if not memories:
            logger.info(f"🧠 [relevant] No memories in store for user '{user_id}'")
        
    except ValueError as ve:
        logger.error(f"Error fetching backend memories for user '{user_id}': {ve}")
        return {
            "status": "error",
            "context": "",
            "message": f"Failed to access memory store for user '{user_id}': {str(ve)}"
        }
    
    # Get character lore if available
    lore_content = ""
    if request.active_character:
        try:
            lore_entries = request.active_character.get("loreEntries", [])
            triggered = []
            prompt_lower = request.prompt.lower()
            for entry in lore_entries:
                keywords = entry.get("keywords", [])
                for keyword in keywords:
                    if keyword.strip().lower() in prompt_lower:
                        triggered.append(entry.get("content", ""))
                        break
            if triggered:
                lore_content = "WORLD KNOWLEDGE:\n" + "\n\n".join(triggered)
                logger.info(f"🧠 {len(triggered)} relevant lore entries for active character")
        except Exception as e:
            logger.error(f"Error getting character lore: {e}")
    
    if not memories and not lore_content:
        return {
            "status": "character_lore_only" if lore_content else "backend_user_store_empty",
            "context": lore_content,
        }
    
    # Filter invalid items
    valid_memories = [m for m in memories if isinstance(m, dict) and m.get("content")]
    invalid_count = len(memories) - len(valid_memories)
    if invalid_count > 0:
        logger.info(f"Filtered out {invalid_count} invalid items from candidate memories before LLM refinement.")
    
    if not valid_memories and not lore_content:
        return {
            "status": "character_lore_only_after_scoring" if lore_content else "no_relevant_memories_after_scoring",
            "context": lore_content,
        }
    
    # Use LLM to refine relevant memories
    try:
        refined = await memory_intelligence.refine_memories_with_llm(
            model_manager=model_manager,
            prompt=request.prompt,
            memories=valid_memories,
            gpu_id=gpu_id,
            lore_content=lore_content,
        )
        
        return {
            "status": "refined_context",
            "context": refined,
            "refinement_method": "llm_refined_gpu1_with_lore" if lore_content else "llm_refined_gpu1",
        }
    except Exception as e:
        logger.error(f"Unknown LLM refinement error: {e}")
        logger.warning(f"LLM refinement failed for user '{user_id}': {e}")
        
        # Fallback: simple formatting
        try:
            simple_context = memory_intelligence.format_memories_simple(valid_memories)
            if lore_content:
                simple_context = lore_content + "\n\n" + simple_context
            return {
                "status": "partial_success",
                "context": simple_context,
                "message": f"LLM refinement failed but using simple formatting: {str(e)}"
            }
        except Exception as e2:
            return {
                "status": "error",
                "context": "",
                "message": f"❌ Unexpected Error in /relevant endpoint for user '{user_id}': {str(e2)}"
            }


@memory_router.post("/memory/create")
async def create_memory(memory_data: dict = Body(...)):
    """
    Manually create a memory from a dictionary payload for a specific user.
    Requires user_id within the memory_data dictionary.
    """
    logger = logging.getLogger("memory_routes")
    
    user_id = memory_data.get("user_id")
    logger.info(f"POST /memory/create: Attempting to create memory manually for user '{user_id}'")
    
    if not user_id:
        logger.warning("Cannot create memory manually without a user_id in the payload.")
        raise HTTPException(status_code=400, detail="user_id is required in the request body dictionary to create a memory.")
    
    content = memory_data.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory content is required in payload for /memory/create.")
    
    if not content:
        raise HTTPException(status_code=400, detail="Memory content cannot be empty for /memory/create.")
    
    try:
        result = memory_intelligence.add_memory_to_store(user_id, memory_data)
        if result:
            logger.info(f"Successfully created memory via /memory/create for user '{user_id}'")
            return {"status": "success", "message": f"Memory created successfully for user '{user_id}'."}
        else:
            logger.warning(f"Manual memory creation via /memory/create failed for user '{user_id}' (duplicate or save error).")
            raise HTTPException(status_code=409, detail="Memory might be a duplicate or failed to save.")
    except ValueError as ve:
        logger.error(f"❌ Value Error creating memory via /memory/create for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory storage: {str(ve)}")
    except Exception as e:
        logger.error(f"Error creating memory via /memory/create for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@memory_router.post("/observe")
async def observe_conversation(
    request: Request,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    """
    Process a conversation and extract memories.
    Enhanced to be more selective about what gets stored.
    """
    logger = logging.getLogger("memory_routes")
    
    body = await request.json()
    user_name = body.get("user_name")
    user_message = body.get("user_message", "")
    ai_response = body.get("ai_response", "")
    
    if not user_name:
        logger.warning("🧠 [observe] Missing user_name for memory observation")
        raise HTTPException(status_code=400, detail="Missing user_name for memory observation")
    
    logger.info(f"Received observation request for user '{user_name}': user_message='{user_message[:50]}...', ai_response='{ai_response[:50]}...'")
    
    try:
        detection_result = body.get("detection_result", "")
        
        if "MEMORY_DETECTED: YES" in detection_result:
            logger.info(f"🧠 [observe] Memory intent detected for user '{user_name}'")
            
            content_match = re.search(r"MEMORY_CONTENT: (.*?)(?:\n|$)", detection_result)
            category_match = re.search(r"MEMORY_CATEGORY: (.*?)(?:\n|$)", detection_result)
            importance_match = re.search(r"MEMORY_IMPORTANCE: (.*?)(?:\n|$)", detection_result)
            
            if content_match:
                memory_content = content_match.group(1).strip()
                memory_category = category_match.group(1).strip() if category_match else "personal_info"
                memory_importance = float(importance_match.group(1).strip()) if importance_match else 0.8
                
                memory_intelligence.add_memory_to_store(
                    user_name,
                    {
                        "content": memory_content,
                        "category": memory_category,
                        "importance": memory_importance,
                        "type": "auto",
                    }
                )
                logger.info(f"🧠 [observe] Adding detected memory: {memory_content[:80]}")
                return {"status": "success", "message": "Memory detected and stored"}
            else:
                logger.warning("🧠 [observe] Memory intent detected but no content extracted")
                return {"status": "success", "message": "Memory detected but content extraction failed"}
        else:
            logger.info(f"🧠 [observe] No memory intent detected for user '{user_name}'")
            return {
                "status": "success",
                "message": "No memory intent detected",
                "suggestion": "analyze_conversations",
                "note": "No explicit memory but scheduled deeper analysis"
            }
    
    except Exception as e:
        logger.error(f"🧠 [observe] Error during memory observation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during observation: {str(e)}")


@memory_router.get("/get_all")
async def get_all_backend_memories(user_id: str = Query(...)):
    """
    Returns all memories currently stored in the backend for the SPECIFIED user.
    Intended for frontend synchronization based on the active user profile.
    Requires user_id as a query parameter.
    """
    if not user_id:
        logger.warning("get_all endpoint called without a user_id query parameter.")
        raise HTTPException(status_code=400, detail="user_id query parameter is required.")
    
    try:
        memories = memory_intelligence.get_all_memories_for_user(user_id)
        return {"status": "success", "memories": memories, "count": len(memories)}
    except ValueError as ve:
        logger.error(f"❌ Value Error getting memories for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory retrieval: {str(ve)}")
    except Exception as e:
        logger.error(f"Error in /get_all endpoint for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve backend memories for user '{user_id}': {str(e)}")


@memory_router.get("/list")
async def list_memories(user_id: str = Query(...)):
    """
    Return all stored memories for a specific user.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes")
    logger.info(f"GET /list: Attempting to list memories for user '{user_id}'")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required to list memories.")
    
    try:
        memories = memory_intelligence.get_all_memories_for_user(user_id)
        logger.info(f"Found {len(memories)} memories for user '{user_id}'")
        return {"status": "success", "memories": memories, "count": len(memories)}
    except ValueError as ve:
        logger.error(f"❌ Value Error listing memories for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory listing: {str(ve)}")
    except Exception as e:
        logger.error(f"Error listing memories for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list memories for user '{user_id}': {str(e)}")


@memory_router.post("/add")
async def add_memory(request: Request, memory: Memory):
    """
    Manually add a memory for a specific user.
    Tries multiple methods to determine user_id with fallbacks.
    """
    logger = logging.getLogger("memory_routes")
    
    user_id = None
    
    # Try multiple sources for user_id
    if memory.user_id:
        user_id = memory.user_id
        logger.info(f"Using user_id from userProfile: {user_id}")
    
    if not user_id:
        try:
            user_id = request.app.state.active_profile_id
            logger.info(f"Using active_profile_id from app state: {user_id}")
        except Exception:
            pass
    
    if not user_id:
        try:
            from . import user_utils
            user_id = user_utils.get_active_profile_id()
            logger.info(f"Fallback to user_utils.get_active_profile_id(): {user_id}")
        except Exception as e:
            logger.error(f"Error accessing user_utils for fallback user_id: {e}")
    
    if not user_id:
        logger.error("Cannot add memory - no user_id found in any source")
        raise HTTPException(status_code=400, detail="Could not determine user_id from any source")
    
    try:
        result = memory_intelligence.add_memory_to_store(user_id, memory.dict())
        if result:
            logger.info(f"Added new memory for user '{user_id}'")
            return {"status": "success", "message": f"Memory added for user '{user_id}'."}
        else:
            logger.warning(f"Memory add failed for user '{user_id}' (duplicate or save error)")
            raise HTTPException(status_code=409, detail="Memory might be a duplicate or failed to save")
    except ValueError as ve:
        logger.error(f"❌ Value Error adding memory for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory storage: {str(ve)}")
    except Exception as e:
        logger.error(f"Error adding memory for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@memory_router.post("/sync")
async def sync_client_memories(user_id: str = Query(...), memories: List[Dict[str, Any]] = Body(...)):
    """
    Sync memories from client to server for a specific user.
    Adds memories from the list that are not already present (exact or semantic match)
    in the user's backend store.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes")
    
    logger.info(f"POST /sync: Attempting to sync {len(memories)} memories from client for user '{user_id}'")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required to sync memories.")
    
    try:
        existing_memories = memory_intelligence.get_all_memories_for_user(user_id)
        existing_contents = {m.get("content", "").strip().lower() for m in existing_memories if isinstance(m, dict)}
        
        new_memories = []
        skipped = 0
        similarity_threshold = 0.85
        
        for mem in memories:
            if not isinstance(mem, dict) or not mem.get("content"):
                logger.warning(f"Skipping invalid sync content for user '{user_id}'")
                skipped += 1
                continue
            
            content = mem["content"].strip()
            if content.lower() in existing_contents:
                skipped += 1
                continue
            
            new_memories.append(mem)
        
        if new_memories:
            result = memory_intelligence.add_memories_batch(user_id, new_memories)
            if result:
                logger.info(f"Successfully synced and saved {len(new_memories)} new memories for user '{user_id}'. Skipped {skipped}.")
                return {
                    "status": "success",
                    "added": len(new_memories),
                    "skipped": skipped,
                    "message": f"Synced {len(new_memories)} new memories for user '{user_id}'."
                }
            else:
                logger.warning(f"Failed to save synced memories for user '{user_id}'")
                raise HTTPException(status_code=500, detail=f"Added {len(new_memories)} memories but failed to save the store for user '{user_id}'.")
        else:
            logger.info(f"Sync completed for user '{user_id}'. No new unique memories added. Skipped {skipped}.")
            return {
                "status": "success",
                "added": 0,
                "skipped": skipped,
                "message": f"No new memories to sync for user '{user_id}'."
            }
    except ValueError as ve:
        logger.error(f"❌ Value Error syncing memories for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory sync: {str(ve)}")
    except Exception as e:
        logger.error(f"Error syncing memories for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during sync: {str(e)}")


@memory_router.delete("/clear")
async def clear_memories(user_id: str = Query(...)):
    """
    Clear all memories for a specific user.
    Requires user_id as a query parameter.
    (Effectively calls purge_memory_store for the user).
    """
    logger = logging.getLogger("memory_routes")
    logger.info(f"DELETE /clear: Attempting to clear memories for user '{user_id}'")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required to clear memories.")
    
    try:
        result = memory_intelligence.purge_memory_store(user_id)
        if result:
            logger.info(f"Successfully cleared memories for user '{user_id}'")
            return {"status": "success", "message": "Memory store completely cleared for this user."}
        else:
            logger.warning(f"Unknown error during memory clear")
            raise HTTPException(status_code=500, detail=f"Failed to clear memories for user '{user_id}'.")
    except ValueError as ve:
        logger.error(f"❌ Value Error clearing memories for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory clear: {str(ve)}")
    except Exception as e:
        logger.error(f"Error clearing memories for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory store: {str(e)}")


@memory_router.post("/purge")
async def purge_memory_endpoint(user_id: str = Query(...)):
    """
    Completely purge all memories from a specific user's memory store.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes")
    logger.info(f"POST /purge: Attempting to purge memories for user '{user_id}'")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required to purge memories.")
    
    try:
        result = memory_intelligence.purge_memory_store(user_id)
        if result:
            logger.info(f"Memory store purged successfully for user '{user_id}'")
            return {"status": "success", "message": "Memory store completely purged for this user."}
        else:
            logger.warning(f"Unknown error during memory purge")
            raise HTTPException(status_code=500, detail=f"Failed to purge memories for user '{user_id}'.")
    except ValueError as ve:
        logger.error(f"❌ Value Error purging memories for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory purge: {str(ve)}")
    except Exception as e:
        logger.error(f"Error during memory purge for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to purge memory store: {str(e)}")


@memory_router.post("/curate")
async def curate_memory_endpoint(user_id: str = Query(...)):
    """
    Run semantic memory curation to remove duplicates for a specific user.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes")
    logger.info(f"POST /curate: Attempting to curate memories for user '{user_id}'")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required to curate memories.")
    
    try:
        result = memory_intelligence.curate_memory_store(user_id)
        if result:
            logger.info(f"Memory curation process completed for user '{user_id}' with status: {result}")
            return {"status": "success", "message": f"Memory curation process finished for user '{user_id}'."}
        else:
            logger.warning(f"Unknown error during memory curation")
            raise HTTPException(status_code=500, detail=f"Memory curation failed for user '{user_id}'.")
    except ValueError as ve:
        logger.error(f"❌ Value Error curating memories for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory curation: {str(ve)}")
    except Exception as e:
        logger.error(f"Error during memory curation endpoint for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@memory_router.post("/model-based-extraction")
async def model_based_extraction(
    request_obj: Request,
    request: MemoryRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    """
    Manually trigger model-based memory extraction for a conversation.
    """
    logger = logging.getLogger("memory_routes")
    
    user_name = request.userProfile.get("id") if request.userProfile else None
    user_id = user_name
    
    logger.info(f"POST /model-based-extraction: Triggered for user '{user_id}'")
    
    if not user_name and not user_id:
        raise HTTPException(status_code=400, detail="Cannot perform model-based extraction without user_name/user_id.")
    
    if not user_name:
        raise HTTPException(status_code=400, detail="user_name (acting as user_id) is required for model-based extraction.")
    
    try:
        single_gpu_mode = getattr(request_obj.app.state, "single_gpu_mode", False)
        gpu_id = 0 if single_gpu_mode else 1
        
        logger.info(f"Using GPU {gpu_id} for memory extraction (single_gpu_mode: {single_gpu_mode})")
        
        model_name = await model_manager.find_suitable_model(gpu_id=gpu_id)
        if model_name:
            logger.info(f"Found suitable model '{model_name}' for extraction.")
        else:
            logger.warning(f"No suitable model found on GPU {gpu_id} for memory extraction.")
            raise HTTPException(status_code=503, detail=f"No model available on GPU {gpu_id} for memory extraction")
        
        extraction_result = await memory_intelligence.model_based_memory_extraction(
            model_manager=model_manager,
            model_name=model_name,
            prompt=request.prompt,
            conversation_history=request.conversation_history,
            gpu_id=gpu_id,
        )
        
        if extraction_result:
            storage_result = memory_intelligence.add_memory_to_store(user_id, extraction_result)
            logger.info(f"Storage result after model extraction for user '{user_id}': {storage_result}")
            return {"status": "success", "result": extraction_result, "stored": storage_result}
        else:
            logger.info(f"No memories extracted by model for user '{user_id}'")
            return {"status": "success", "result": "no_memories", "message": f"No memories extracted by model for user '{user_id}'."}
    
    except ValueError as ve:
        logger.error(f"❌ Value Error during model extraction for user '{user_id}': {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory storage: {str(ve)}")
    except Exception as e:
        logger.error(f"Error during model-based extraction for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# === Agentic Memory Request Models ===

class AgenticProcessRequest(BaseModel):
    user_id: str
    character_id: str
    character_name: Optional[str] = None
    character_profile: Optional[Dict[str, Any]] = None
    user_message: str
    ai_response: str
    use_api: Optional[bool] = None
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None

class AgenticDeleteInsightsRequest(BaseModel):
    user_id: str
    character_id: str
    insight_ids: List[str] = []

class AgenticUpdateInsightRequest(BaseModel):
    user_id: str
    character_id: str
    insight_id: str
    content: Optional[str] = None
    category: Optional[str] = None
    importance: Optional[float] = None

class AgenticCopyToCharacterRequest(BaseModel):
    """Optional: copy per-character agentic memories to another character (same user_id)."""
    user_id: str
    source_character_id: str
    target_character_id: str
    mode: str = "merge"  # 'merge' appends with content dedupe; 'replace' overwrites the target file.


@memory_router.get("/agentic/list")
async def list_agentic_profiles(user_id: str = Query(...)):
    """All agentic profiles for a user (Settings → memories UI)."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        profiles = agentic_memory.list_agentic_profiles_for_user(user_id)
        total_insights = sum(int(p.get("count") or 0) for p in profiles)
        logger.info(
            "[Agentic Memory] GET /agentic/list user_id=%r -> %s profile(s), %s insight(s)",
            user_id,
            len(profiles),
            total_insights,
        )
        return {"status": "success", "profiles": profiles, "profile_count": len(profiles), "total_insights": total_insights}
    except Exception as e:
        logger.error(f"agentic_memory list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.get("/agentic")
async def get_agentic_memory(
    user_id: str = Query(...),
    character_id: str = Query(...),
):
    """
    Get the agentic memory profile for (user_id, character_id).
    Returns insights list and optional formatted context for injection.
    """
    logger.info(f"[Agentic Memory] GET /agentic user_id={user_id!r} character_id={character_id!r}")
    if not user_id or not character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    try:
        profile = agentic_memory.get_agentic_profile(user_id, character_id)
        formatted = agentic_memory.format_agentic_context(profile["insights"])
        count = len(profile["insights"])
        logger.info(f"[Agentic Memory] GET /agentic -> {count} insights, formatted_context={len(formatted)} chars")
        return {
            "status": "success",
            "insights": profile["insights"],
            "formatted_context": formatted,
            "count": count,
        }
    except Exception as e:
        logger.error(f"agentic_memory get error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/agentic/delete_insights")
async def agentic_delete_insights(body: AgenticDeleteInsightsRequest):
    """Settings UI: remove one or more agentic insights by id."""
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    try:
        n = agentic_memory.delete_agentic_insights(
            body.user_id, body.character_id, body.insight_ids or []
        )
        return {"status": "success", "removed": n}
    except Exception as e:
        logger.error(f"agentic delete_insights error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/agentic/update_insight")
async def agentic_update_insight(body: AgenticUpdateInsightRequest):
    """Settings UI: edit a single agentic insight."""
    if not body.user_id or not body.character_id or not body.insight_id:
        raise HTTPException(status_code=400, detail="user_id, character_id, and insight_id are required")
    try:
        ok = agentic_memory.update_agentic_insight(
            body.user_id,
            body.character_id,
            body.insight_id,
            content=body.content,
            category=body.category,
            importance=body.importance,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Insight not found")
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"agentic update_insight error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/agentic/copy_to_character")
async def agentic_copy_to_character(body: AgenticCopyToCharacterRequest):
    """
    Copy agentic memories from source_character_id to target_character_id for the same user.
    merge = append (dedupe by content); replace = overwrite target file from source.
    """
    if not body.user_id or not body.source_character_id or not body.target_character_id:
        raise HTTPException(
            status_code=400,
            detail="user_id, source_character_id, and target_character_id are required",
        )
    try:
        result = agentic_memory.copy_agentic_profile_to_character(
            body.user_id,
            body.source_character_id,
            body.target_character_id,
            mode=body.mode,
        )
        return {"status": "success", **result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"agentic copy_to_character error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/agentic/process")
async def process_agentic_memory(
    request: Request,
    body: AgenticProcessRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    """
    Run the agentic memory agent on a user/bot exchange and append new insights
    to the character-specific profile. Optional; only call when character has
    agenticMemoryEnabled.
    """
    logger.warning("[Agentic Memory] REQUEST RECEIVED at /memory/agentic/process — if you never see this, the frontend is not calling this URL.")
    logger.info(
        "[Agentic Memory] POST /agentic/process user_id=%r character_id=%r char_name=%r character_profile=%s",
        body.user_id,
        body.character_id,
        body.character_name,
        "yes" if body.character_profile else "no",
    )
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    single_gpu_mode = getattr(request.app.state, "single_gpu_mode", False)
    gpu_id = 0 if single_gpu_mode else 0
    try:
        profile = agentic_memory.get_agentic_profile(body.user_id, body.character_id)
        use_api = body.use_api and body.api_base_url and body.model_name
        new_insights = await agentic_memory.run_agentic_agent(
            model_manager=model_manager,
            user_message=body.user_message,
            ai_response=body.ai_response,
            character_name=body.character_name or "Character",
            existing_insights=profile["insights"],
            character_profile=body.character_profile,
            gpu_id=gpu_id,
            single_gpu_mode=single_gpu_mode,
            api_base_url=body.api_base_url if use_api else None,
            api_model_name=body.model_name if use_api else None,
        )
        if body.character_profile:
            logger.info(
                "[Agentic Memory] POST /agentic/process character_profile synced for %r",
                body.character_id,
            )
        if not new_insights:
            store_path = agentic_memory.get_agentic_memory_path(body.user_id, body.character_id)
            prof_after = agentic_memory.get_agentic_profile(body.user_id, body.character_id)
            total = len(prof_after.get("insights") or [])
            logger.info(
                "[Agentic Memory] POST /agentic/process -> no new insights (agent returned 0); total=%s file=%s",
                total,
                store_path,
            )
            return {"status": "success", "added": 0, "message": "No new insights", "total": total}
        added = agentic_memory.add_agentic_insights(
            body.user_id, body.character_id, new_insights
        )
        store_path = agentic_memory.get_agentic_memory_path(body.user_id, body.character_id)
        prof_after = agentic_memory.get_agentic_profile(body.user_id, body.character_id)
        total = len(prof_after.get("insights") or [])
        logger.info(
            "[Agentic Memory] POST /agentic/process -> added %s insight(s); total=%s file=%s",
            added,
            total,
            store_path,
        )
        return {
            "status": "success",
            "added": added,
            "total": total,
            "store_path": store_path,
            "message": f"Added {added} insight(s) to character memory.",
            "insights": new_insights[:10],
        }
    except Exception as e:
        logger.error(f"[Agentic Memory] POST /agentic/process error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class AgenticCleanupRequest(BaseModel):
    user_id: str
    character_id: str
    character_name: Optional[str] = None
    character_profile: Optional[Dict[str, Any]] = None
    use_api: Optional[bool] = None
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None


@memory_router.post("/agentic/cleanup")
async def cleanup_agentic_memory(
    request: Request,
    body: AgenticCleanupRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    """
    LLM-assisted duplicate pruning, then deterministic dedupe / trim for agentic insights.
    """
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    single_gpu_mode = getattr(request.app.state, "single_gpu_mode", False)
    gpu_id = getattr(request.app.state, "default_gpu", 0)
    try:
        profile = agentic_memory.get_agentic_profile(body.user_id, body.character_id)
        insights = profile.get("insights") or []
        use_api = body.use_api and body.api_base_url and body.model_name
        
        remove_ids = await agentic_memory.run_agentic_cleanup_agent(
            model_manager=model_manager,
            insights=insights,
            character_name=body.character_name or "Character",
            character_profile=body.character_profile,
            gpu_id=gpu_id,
            api_base_url=body.api_base_url if use_api else None,
            api_model_name=body.model_name if use_api else None,
        )
        
        if remove_ids:
            agentic_memory.delete_agentic_insights(body.user_id, body.character_id, remove_ids)
        
        # Also run deterministic cleanup
        cleanup_result = agentic_memory.cleanup_agentic_profile(body.user_id, body.character_id)
        
        total_removed = len(remove_ids) + cleanup_result.get("removed", 0)
        prof_after = agentic_memory.get_agentic_profile(body.user_id, body.character_id)
        total = len(prof_after.get("insights") or [])
        
        return {
            "status": "success",
            "llm_removed": len(remove_ids),
            "dedupe_removed": cleanup_result.get("removed", 0),
            "total_removed": total_removed,
            "remaining": total,
        }
    except Exception as e:
        logger.error(f"[Agentic Memory] POST /agentic/cleanup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === Persona Realignment ===

class PersonaRealignmentPromptPackRequest(BaseModel):
    user_id: str
    character_id: str
    character_name: Optional[str] = None
    user_display_name: Optional[str] = None
    character_card: Optional[Dict[str, Any]] = None
    current_character_instructions: Optional[str] = None
    rolling_packs: Optional[List[str]] = None
    transcripts: Optional[List[str]] = None
    include_backend_memories: bool = True
    agentic_mode: Optional[str] = None
    agentic_max_chars: Optional[int] = None
    agentic_rag_query: Optional[str] = None
    ethics_framing: Optional[str] = Field(None, description="Optional authoritative research ethics, institutional or committee oversight, and study purpose for this run. When provided, it is embedded first in the DATA bundle as binding framing — not casual chat preferences.")
    extra_notes: Optional[str] = None
    also_rewrite_user_profile: bool = False
    user_profile_rewrite_mode: Optional[str] = None
    save_preview_to_disk: bool = False
    attach_ethics_review_bundle: bool = False
    reviewer_character_id: Optional[str] = None
    reviewer_character_name: Optional[str] = None
    reviewer_character_instructions: Optional[str] = Field(None, description="Full system prompt text from an optional Eloquent 'evaluator' character; leads the analyst preamble when set.")
    backend_memory_max_items: Optional[int] = Field(None, description="Cap rows from saved profile memories included in this pack (not chat pruning—server-side list cap).")
    indexed_profile_memory_max_items: Optional[int] = Field(None, description="Indexed JSON rows for profile rewrite section.")
    agentic_meta_max_chars: Optional[int] = Field(None, description="Max chars for agentic file meta JSON block.")
    example_dialogue_max_chars: Optional[int] = Field(None, description="Max chars for embedded example dialogue on the character card.")
    agentic_rag_top_k: Optional[int] = Field(None, description="When agentic_mode=rag, max insights considered before char budget fills.")


class PersonaRealignmentParseRequest(BaseModel):
    raw_text: str
    character_id: Optional[str] = None
    user_id: Optional[str] = None


@memory_router.post("/persona_realignment/prompt_pack")
async def persona_realignment_prompt_pack(body: PersonaRealignmentPromptPackRequest):
    """Build the full persona realignment prompt pack for a character."""
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    
    try:
        # Fetch backend memories
        backend_memories = []
        if body.include_backend_memories:
            try:
                all_mems = memory_intelligence.get_all_memories_for_user(body.user_id)
                if body.backend_memory_max_items:
                    all_mems = all_mems[:body.backend_memory_max_items]
                backend_memories = all_mems
            except Exception as e:
                logger.warning(f"Failed to fetch backend memories for realignment: {e}")
        
        # Fetch agentic insights
        agentic_insights = []
        agentic_meta = {}
        try:
            profile = agentic_memory.get_agentic_profile(body.user_id, body.character_id)
            agentic_insights = profile.get("insights") or []
            agentic_meta = profile.get("meta") or {}
        except Exception as e:
            logger.warning(f"Failed to fetch agentic profile for realignment: {e}")
        
        # Build prompt pack
        pack = persona_realignment.build_full_analyst_prompt(
            user_id=body.user_id,
            character_id=body.character_id,
            character_name=body.character_name,
            backend_memories=backend_memories,
            agentic_insights=agentic_insights,
            agentic_meta=agentic_meta,
            character_card=body.character_card,
            current_character_instructions=body.current_character_instructions,
            rolling_packs=body.rolling_packs,
            transcripts=body.transcripts,
            agentic_mode=body.agentic_mode,
            agentic_max_chars=body.agentic_max_chars,
            agentic_rag_query=body.agentic_rag_query,
            user_display_name=body.user_display_name,
            extra_notes=body.extra_notes,
            also_rewrite_user_profile=body.also_rewrite_user_profile,
            user_profile_rewrite_mode=body.user_profile_rewrite_mode,
            reviewer_character_name=body.reviewer_character_name,
            reviewer_character_instructions=body.reviewer_character_instructions,
            backend_memory_max_items=body.backend_memory_max_items,
            indexed_profile_memory_max_items=body.indexed_profile_memory_max_items,
            agentic_meta_max_chars=body.agentic_meta_max_chars,
            example_dialogue_max_chars=body.example_dialogue_max_chars,
            agentic_rag_top_k=body.agentic_rag_top_k,
        )
        
        # Append ethics review bundle if requested
        if body.attach_ethics_review_bundle:
            try:
                ethics_parts = ethics_review_bundle.get_bundle_parts()
                if ethics_parts:
                    pack["ethics_review_bundle_parts"] = "\n\n---\n\n## APPENDIX: Ethics / framing source excerpts (local repo)\n\n" + ethics_parts
            except Exception as e:
                logger.warning(f"ethics_review_bundle append failed: {e}")
        
        # Save preview if requested
        preview_saved = None
        if body.save_preview_to_disk:
            try:
                preview_saved = preview_prompt_save.save_preview("persona_realignment", pack)
            except Exception as e:
                logger.warning(f"preview_prompt_save persona_realignment failed: {e}")
        
        return {
            "status": "success",
            "analyst_preamble": pack.get("analyst_preamble", ""),
            "task_and_data": pack.get("task_and_data", ""),
            "combined": pack.get("combined", ""),
            "output_spec": pack.get("output_spec", ""),
            "ethics_review_bundle_parts": pack.get("ethics_review_bundle_parts", ""),
            "preview_saved": preview_saved,
        }
    except Exception as e:
        logger.error(f"persona_realignment prompt_pack error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/persona_realignment/parse_response")
async def persona_realignment_parse_response(body: PersonaRealignmentParseRequest):
    """Parse the LLM response from a persona realignment run."""
    try:
        result = persona_realignment.parse_realignment_response(
            body.raw_text,
        )
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"persona_realignment parse_response error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.get("/ethics_review/manifest")
async def get_ethics_review_manifest():
    """Which repo paths are embedded when attach_ethics_review_bundle is used (no file payload).
    Code excerpts are added only when "Include code excerpts" is enabled in Ethics review. Otherwise the model does not see these files."""
    try:
        manifest = ethics_review_bundle.get_manifest()
        return {"status": "success", "manifest": manifest}
    except Exception as e:
        logger.error(f"ethics_review_manifest GET failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.get("/ethics_review/bundle")
async def get_ethics_review_bundle():
    """Whitelisted local excerpts for reviewer models (copy into chat or use via attach_ethics_review_bundle)."""
    try:
        bundle = ethics_review_bundle.get_bundle_parts()
        return {"status": "success", "bundle": bundle}
    except Exception as e:
        logger.error(f"ethics_review_bundle GET failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# === Memory Curator ===

class MemoryCuratorPromptRequest(BaseModel):
    user_display_name: Optional[str] = None
    user_profile_summary: Optional[str] = None
    curator_character_name: Optional[str] = None
    curator_character_card: Optional[Dict[str, Any]] = None
    extra_notes: Optional[str] = None
    save_preview_to_disk: bool = False
    target_character_id: Optional[str] = None
    target_character_name: Optional[str] = None


class CuratorParseRequest(BaseModel):
    raw_response: str
    mode: str = "profile"  # "profile" or "agentic"


class CuratorApplyProfileRequest(BaseModel):
    user_id: str
    memories: List[Dict[str, Any]]


class CuratorApplyAgenticRequest(BaseModel):
    user_id: str
    character_id: str
    insights: List[Dict[str, Any]]


@memory_router.post("/curator/prompt_pack")
async def curator_prompt_pack(body: MemoryCuratorPromptRequest):
    """Build a memory curator prompt pack."""
    if not body.user_display_name and not body.user_profile_summary:
        raise HTTPException(status_code=400, detail="user_id is required")
    if body.target_character_id is None and body.curator_character_card is None:
        pass  # profile mode doesn't need character_id
    
    try:
        mode = "agentic" if body.target_character_id else "profile"
        if mode == "agentic" and not body.target_character_id:
            raise HTTPException(status_code=400, detail="target_character_id is required for agentic mode")
        if mode not in ("profile", "agentic"):
            raise HTTPException(status_code=400, detail="mode must be profile or agentic")
        
        pack = memory_curator_prompt.build_curator_prompt(
            user_display_name=body.user_display_name,
            user_profile_summary=body.user_profile_summary,
            curator_character_name=body.curator_character_name,
            curator_character_card=body.curator_character_card,
            extra_notes=body.extra_notes,
            target_character_id=body.target_character_id,
            target_character_name=body.target_character_name,
            mode=mode,
        )
        
        preview_saved = None
        if body.save_preview_to_disk:
            try:
                preview_saved = preview_prompt_save.save_preview("curator", pack)
            except Exception as e:
                logger.warning(f"preview_prompt_save curator failed: {e}")
        
        return {"status": "success", "prompt_pack": pack, "preview_saved": preview_saved}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"curator prompt_pack error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/curator/parse_response")
async def curator_parse_response(body: CuratorParseRequest):
    """Parse the LLM response from a curator run."""
    try:
        # Try to extract JSON from the response
        text = body.raw_response.strip()
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                start = text.find("[")
                end = text.rfind("]") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        
        obj_start = text.find("{")
        if obj_start >= 0:
            obj_end = text.rfind("}") + 1
            raw = text[obj_start:obj_end]
            data = json.loads(raw)
        else:
            raise ValueError("Could not parse a JSON object from the response")
        
        if body.mode == "profile":
            if "memories" not in data:
                raise ValueError('Parsed JSON must include a "memories" array for profile mode')
            return {"status": "success", "parsed": data}
        elif body.mode == "agentic":
            if "insights" not in data:
                raise ValueError('Parsed JSON must include an "insights" array for agentic mode')
            return {"status": "success", "parsed": data}
        else:
            return {"status": "success", "parsed": data}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in response: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"curator parse_response error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/curator/apply_profile")
async def curator_apply_profile(body: CuratorApplyProfileRequest):
    """Apply curated profile memories."""
    if not body.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        # Normalize and save
        valid_memories = [m for m in body.memories if isinstance(m, dict) and m.get("content", "").strip()]
        if not valid_memories:
            raise HTTPException(status_code=400, detail="No valid memories to save after normalization")
        
        result = memory_intelligence.replace_memory_store(body.user_id, valid_memories)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to save memory store")
        return {"status": "success", "applied": len(valid_memories)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"curator apply_profile error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("/curator/apply_agentic")
async def curator_apply_agentic(body: CuratorApplyAgenticRequest):
    """Apply curated agentic insights."""
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    try:
        valid_insights = [i for i in body.insights if isinstance(i, dict) and i.get("content", "").strip()]
        if not valid_insights:
            raise HTTPException(status_code=400, detail="No valid insights to save after normalization")
        
        agentic_memory.save_agentic_profile(body.user_id, body.character_id, valid_insights)
        return {"status": "success", "applied": len(valid_insights)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"curator apply_agentic error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.get("/preview/list")
async def list_memory_preview_prompts(limit: int = Query(default=20)):
    """List saved preview prompts."""
    try:
        previews = preview_prompt_save.list_previews(limit=limit)
        return {"status": "success", "previews": previews}
    except Exception as e:
        logger.error(f"preview list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
