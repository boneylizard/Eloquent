# memory_routes.py - Backend routes for memory operations

from fastapi import APIRouter, Depends, HTTPException, Request, Body, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
import logging
import traceback
from . import memory_intelligence
from . import inference  # Import the inference module
from .model_manager import ModelManager
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from . import memory_intelligence # Assuming memory_intelligence.py is in the same directory
import logging # Example
import datetime # Example
from .memory_intelligence import process_completed_exchange # Example
from sentence_transformers import util # Example
import torch # Example
from .memory_intelligence import similarity_model # Example
import re


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
    model_name: Optional[str] = None # Optional: specify model, otherwise find suitable one
    user_name: Optional[str] = None # Optional: specify user name for personalization
    user_id: Optional[str] = None    # ← add this field
    gpu_id: Optional[int] = None # Optional: specify GPU ID for processing
    single_gpu_mode: Optional[bool] = None # Optional: specify single GPU mode

class MemoryRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    userProfile: Optional[Dict[str, Any]] = None
    systemTime: Optional[str] = None
    requestType: Optional[str] = None
    active_character: Optional[Dict[str, Any]] = None # <-- Corrected field name


class ObservationRequest(BaseModel):
    user_message: str
    ai_response: str
    user_name: Optional[str] = None # <-- Keep this addition
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
    user_id: Optional[str] = None # Optional user ID for personalization

# === Model Classes ===
# ... (other models) ...

class ContextualMemoryRequest(BaseModel):
    conversation_snippet: str
    category: str = "contextual" # Add category field with a default
    importance: float = 0.6     # Add importance field with a default
    user_name: Optional[str] = None # Keep optional user_name

# character_lore = {``
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
    logger.info("🌍 BACKEND LORE ENDPOINT HIT!")
    
    try:
        message = data.get('message', '')
        active_character = data.get('activeCharacter')
        
        print(f"🌍 BACKEND: Message = '{message}'")
        print(f"🌍 BACKEND: Character = {active_character.get('name') if active_character else 'None'}")
        
        if not message:
            print("🌍 BACKEND: No message - returning empty")
            return {"status": "success", "lore_triggered": []}
        
        if not active_character:
            print("🌍 BACKEND: No character - returning empty")
            return {"status": "success", "lore_triggered": []}
        
        lore_entries = active_character.get('loreEntries', [])
        print(f"🌍 BACKEND: Found {len(lore_entries)} lore entries")
        
        triggered_lore = []
        for entry in lore_entries:
            if isinstance(entry, dict) and 'content' in entry:
                content = entry.get('content', '').strip()
                if content:
                    triggered_lore.append(content)
                    print(f"🌍 BACKEND: Added lore content: {content[:50]}...")
        
        print(f"🌍 BACKEND: Returning {len(triggered_lore)} lore entries")
        print("=" * 60)
        
        return {
            "status": "success",
            "lore_triggered": triggered_lore
        }
        
    except Exception as e:
        print(f"🌍 BACKEND ERROR: {e}")
        logger.error(f"🌍 BACKEND ERROR: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "lore_triggered": []}
# --- Helper function to get user ID consistently ---
def get_user_id_from_request(request, client_supplied_id=None):
    """Consistently get user ID with proper fallbacks."""
    # 1. Try client-supplied ID first
    if client_supplied_id:
        return client_supplied_id
        
    # 2. Try the active profile in app state
    active_profile_id = getattr(request.app.state, "active_profile_id", None)
    if active_profile_id:
        return active_profile_id
        
    # 3. If all else fails, return None (endpoint should handle this)
    return None

# CORRECTED /detect_intent endpoint
@memory_router.post("/detect_intent")
async def detect_memory_intent_api(
    request_obj: Request,
    detect_request: MemoryDetectRequest,
    model_manager: ModelManager = Depends(get_model_manager_from_state)
):
    logger = logging.getLogger("memory_routes")
    
    # Get user ID consistently
    user_id = get_user_id_from_request(
        request_obj,
        detect_request.user_id or detect_request.user_name
    )
    
    if not user_id:
        logger.warning("🧠 [detect_intent] Using default test user ID as fallback")
        user_id = "default_test_user"  # Last-resort fallback
        
    logger.info(f"🧠 [detect_intent] Using user_id: {user_id}")
    """
    Receives prompt/response text and uses a local model to detect memory intent.
    """
    logger = logging.getLogger("memory_routes")

    user_id = detect_request.user_id or detect_request.user_name  
    if not user_id:
        logger.warning("🧠 [detect_intent] Missing user_id or user_name for memory intent detection")
        raise HTTPException(status_code=400, detail="user_id or user_name is required")
    
    # Correctly access app state from request_obj
    single_gpu_mode = getattr(request_obj.app.state, "single_gpu_mode", False)
    target_gpu_id = 0 if single_gpu_mode else 1  # Use GPU 0 (3090) for peripheral memory operations

    
    
    logger.info(f"🧠 API /detect_intent called on GPU {target_gpu_id} instance.")
    logger.debug(f"Received detect_request: prompt='{detect_request.original_prompt[:50]}...', response='{detect_request.response_text[:50]}...'")

    try:
        # Find suitable model on this instance (GPU 1)
        # Use provided name or find the first one available on GPU 1
        model_to_use = detect_request.model_name
        if not model_to_use:
            # Use await if find_suitable_model is async
            model_to_use = await model_manager.find_suitable_model(gpu_id=target_gpu_id)

        logger.info(f"Using model '{model_to_use}' on GPU {target_gpu_id} for intent detection.")

        if not model_to_use:
            logger.error(f"API /detect_intent: No suitable model found on GPU {target_gpu_id}.")
            raise HTTPException(status_code=500, detail=f"No suitable memory detection model loaded on GPU {target_gpu_id}")

        # Improved memory detection prompt that's more selective
        memory_detection_prompt = f"""Analyze this user message for personal information:

"{detect_request.original_prompt}"

You must respond in EXACTLY this format:

MEMORY_DETECTED: YES
MEMORY_CONTENT: This is where you summarize the key personal information to remember. Just parse the message and extract relevant details.
MEMORY_CATEGORY: personal_info (example value, adjust as needed)
MEMORY_IMPORTANCE: 0.8 (example value, adjust as needed)

OR if no personal information:

MEMORY_DETECTED: NO

Your response:"""

        logger.debug(f"Sending detection prompt to model '{model_to_use}'")
        
        # Generate the intent detection
        detection_result_text = await inference.generate_text(
            model_manager=model_manager,
            model_name=model_to_use,
            prompt=memory_detection_prompt,
            max_tokens=250,
            temperature=0.1, # Very low temperature for consistency
            repetition_penalty=1.1,
            gpu_id=target_gpu_id
        )

        logger.info(f"API /detect_intent: Inference successful. Result preview: {str(detection_result_text)[:100]}...")
        logger.debug(f"Raw detection result TEXT from model '{model_to_use}':\n---RESULT START---\n{detection_result_text}\n---RESULT END---")
        
        # Enhanced parsing logic to ensure consistent output
        if detection_result_text and isinstance(detection_result_text, str):
            # Normalize detection result - sometimes models add extra verbiage
            if "MEMORY_DETECTED: NO" in detection_result_text:
                logger.info("Memory detection result: NO memory to store")
                # Force standardized NO response
                detection_result_text = "MEMORY_DETECTED: NO"
            elif "MEMORY_DETECTED: YES" in detection_result_text:
                logger.info("Memory detection result: YES - memory to store")
                # Ensure we have the core components
                if not all(x in detection_result_text for x in ["MEMORY_CONTENT:", "MEMORY_CATEGORY:", "MEMORY_IMPORTANCE:"]):
                    logger.warning("Memory detection returned YES but missing required fields")
                    # Could fix this here by extracting what is available and formatting it properly
            else:
                logger.warning(f"Memory detection returned unclear result: {detection_result_text[:50]}...")
                # Default to NO if unclear
                detection_result_text = "MEMORY_DETECTED: NO"
        else:
            logger.warning("Memory detection returned invalid result")
            detection_result_text = "MEMORY_DETECTED: NO"
            
        # Return the standardized result
        return {"status": "success", "detection_result": detection_result_text}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions from downstream (like inference)
        raise http_exc
    except Exception as e:
        logger.error(f"API /detect_intent: Error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during memory intent detection: {str(e)}")

# Finally, let's fix the route handler in memory_routes.py for the /relevant endpoint
@memory_router.post("/relevant")
async def get_relevant_memories(
    request_obj: Request,  # Add this parameter to access app state
    request: MemoryRequest, # This is the Pydantic model
    model_manager: ModelManager = Depends(get_model_manager_from_state)
):
    """
    Fetches candidate memories from the specific user's memory_store.json,
    then uses an LLM on GPU 1 to refine and select the most relevant context.
    Uses user_id derived from userProfile, prioritizing 'id'.
    """
    
    try:
        logger.info(f"🧠 Received prompt for LLM-powered memory retrieval: {request.prompt[:100]}...")
        original_prompt = request.prompt
        user_profile_data = request.userProfile or {}

        # --- Derive user_id (Corrected & Safer) ---
        user_id_value = user_profile_data.get("id") or \
                        user_profile_data.get("userId") or \
                        user_profile_data.get("user_id")

        user_id = str(user_id_value) if user_id_value is not None else None

        logger.info(f"Extracted user_id: '{user_id}' for memory retrieval.")

        if not user_id:
            logger.error("Cannot retrieve relevant memories without a valid 'id', 'userId', or 'user_id' in userProfile.")
            # Return empty results if no user_id can be determined
            return {
                "status": "success", # Return success status but indicate no memories found
                "reason": "user_id_missing",
                "memories": [],
                "formatted_memories": "",
                "memory_count": 0,
                "retrieval_source": "none"
            }

        # Load what's on disk
        backend_memories = memory_intelligence.get_memory_store(user_id=user_id)
        all_candidate_memories = []
        
        # Access single_gpu_mode from app state using request_obj instead of request
        single_gpu_mode = getattr(request_obj.app.state, "single_gpu_mode", False)
        gpu_for_refinement = 0 if single_gpu_mode else 1
        logger.info(f"Using GPU {gpu_for_refinement} for memory refinement (single_gpu_mode: {single_gpu_mode})")

        try:
            if backend_memories:
                logger.info(f"🧠 [relevant] Loaded {len(backend_memories)} memories for user '{user_id}'")
                all_candidate_memories.extend(backend_memories)
            else:
                logger.info(f"🧠 [relevant] No memories in store for user '{user_id}'")
        except ValueError as ve:
            logger.error(f"Error fetching backend memories for user '{user_id}': {ve}", exc_info=True)
            return {
                "status": "error",
                "error": f"Failed to access memory store for user '{user_id}': {ve}",
                "memories": [],
                "formatted_memories": "",
                "memory_count": 0
            }
        except Exception as backend_err:
            logger.error(f"Error fetching backend memories for user '{user_id}': {backend_err}", exc_info=True)
            # Continue to try LLM refinement even if there's an error

        # --- Check for lore entries if active_character is present ---
        character_lore_context = ""
        if request.active_character:
            try:
                triggered_lore = memory_intelligence.get_triggered_character_lore(
                    original_prompt, request.active_character
                )
                if triggered_lore and len(triggered_lore) > 0:
                    character_lore_context = "WORLD KNOWLEDGE:\n" + "\n".join([f"• {lore}" for lore in triggered_lore])
                    logger.info(f"Found {len(triggered_lore)} relevant lore entries for active character")
            except Exception as lore_err:
                logger.error(f"Error getting character lore: {lore_err}", exc_info=True)

        # If there's nothing to refine, return early
        if not all_candidate_memories:
            # Only return lore context if we have it
            if character_lore_context:
                return {
                    "status": "success",
                    "memories": [],
                    "formatted_memories": character_lore_context,
                    "memory_count": 0,
                    "retrieval_source": "character_lore_only"
                }
            return {
                "status": "success",
                "memories": [],
                "formatted_memories": "",
                "memory_count": 0,
                "retrieval_source": "backend_user_store_empty"
            }

        # Filter out any non-dictionary items before refinement
        valid_candidate_memories = [mem for mem in all_candidate_memories if isinstance(mem, dict)]
        if len(valid_candidate_memories) != len(all_candidate_memories):
            logger.warning(f"Filtered out {len(all_candidate_memories) - len(valid_candidate_memories)} invalid items from candidate memories before LLM refinement.")

        # Use analyze_for_relevant_memories to pre-filter memories before LLM refinement
        relevant_memories = await memory_intelligence.analyze_for_relevant_memories(
            model_manager=model_manager,
            prompt=original_prompt,
            memories=valid_candidate_memories,
            gpu_id=gpu_for_refinement,
            user_id=user_id,
            single_gpu_mode=single_gpu_mode
        )

        # If we have no relevant memories after scoring, return early
        if not relevant_memories:
            # Only return lore context if we have it
            if character_lore_context:
                return {
                    "status": "success",
                    "memories": [],
                    "formatted_memories": character_lore_context,
                    "memory_count": 0,
                    "retrieval_source": "character_lore_only_after_scoring"
                }
            return {
                "status": "success",
                "memories": [],
                "formatted_memories": "",
                "memory_count": 0,
                "retrieval_source": "no_relevant_memories_after_scoring"
            }

        # Now use the LLM to refine the already filtered relevant memories
        refined_context_result = await memory_intelligence.get_llm_refined_context(
            model_manager=model_manager,
            original_prompt=original_prompt,
            candidate_memories=relevant_memories,
            gpu_id=gpu_for_refinement,
            single_gpu_mode=single_gpu_mode,
        )

        # Process refinement result
        if refined_context_result.get("status") == "success":
            refined_context = refined_context_result.get("refined_context", "")
            
            # Combine with character lore if available
            final_context = refined_context
            if character_lore_context and refined_context:
                final_context = f"{refined_context}\n\n{character_lore_context}"
            elif character_lore_context:
                final_context = character_lore_context
            
            # Count memory items by simple heuristic (bullet points)
            memory_count = final_context.count("•")
            if memory_count == 0 and len(final_context) > 20:
                # If no bullet points but text exists, estimate based on newlines
                memory_count = len([line for line in final_context.split('\n') if line.strip()])
            
            return {
                "status": "success", 
                "memories": relevant_memories, 
                "formatted_memories": final_context, 
                "memory_count": memory_count, 
                "retrieval_source": "llm_refined_gpu1_with_lore"
            }
        else:
            error_detail = refined_context_result.get('error', 'Unknown LLM refinement error')
            logger.error(f"LLM refinement failed for user '{user_id}': {error_detail}")
            
            # Fallback to simple formatting if LLM refinement fails
            simple_formatted = memory_intelligence.format_memories_for_context(relevant_memories)
            
            # Combine with character lore if available
            if character_lore_context and simple_formatted:
                simple_formatted = f"{simple_formatted}\n\n{character_lore_context}"
            elif character_lore_context:
                simple_formatted = character_lore_context
                
            return {
                "status": "partial_success", 
                "error": f"LLM refinement failed but using simple formatting: {error_detail}",
                "memories": relevant_memories, 
                "formatted_memories": simple_formatted, 
                "memory_count": len(relevant_memories)
            }

    except Exception as e:
        # Log user_id if available, otherwise log 'unknown'
        user_id_for_log = user_id if 'user_id' in locals() and user_id else 'unknown'
        logger.error(f"❌ Unexpected Error in /relevant endpoint for user '{user_id_for_log}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during memory retrieval: {str(e)}")

    
# --- Replace the existing create_memory function with this ---
# Note: This endpoint seems similar to /add but takes a raw dict.
# Ensure frontend sends user_id within this dictionary.
@memory_router.post("/memory/create")
async def create_memory(memory_data: dict):
    """
    Manually create a memory from a dictionary payload for a specific user.
    Requires user_id within the memory_data dictionary.
    """
    logger = logging.getLogger("memory_routes") # Get logger instance
    user_id = memory_data.get("user_id") # Extract user_id from the dictionary body

    logger.info(f"POST /memory/create: Attempting to create memory manually for user '{user_id}'.") # Log user_id

    if not user_id:
        logger.error("Cannot create memory manually without a user_id in the payload.")
        raise HTTPException(status_code=400, detail="user_id is required in the request body dictionary to create a memory.")

    try:
        # Validate required fields
        if 'content' not in memory_data:
            logger.error("Memory content is required in payload for /memory/create.")
            raise HTTPException(status_code=422, detail="Memory content is required")

        content = memory_data.get('content', '').strip()
        if not content:
            logger.error("Memory content cannot be empty for /memory/create.")
            raise HTTPException(status_code=422, detail="Memory content cannot be empty")

        # Validate the content using the intelligence module function
        if not memory_intelligence.is_valid_memory_content(content):
            logger.warning(f"Invalid manual memory content via /memory/create for user '{user_id}': {content[:50]}...")
            raise HTTPException(status_code=422, detail="Invalid memory content provided.")

        # Create a proper memory object for storage (excluding user_id from the object itself)
        memory_to_store = {
            "content": content,
            "category": memory_data.get('category', 'manual'), # Default category
            "importance": float(memory_data.get('importance', 0.8)), # Default importance
            "type": "manual", # Indicate manual creation
            "created": datetime.datetime.now().isoformat(), # Set creation time
            "accessed": 0 # Initialize access count
            # Add other fields like 'tags' if provided in memory_data
        }
        if 'tags' in memory_data and isinstance(memory_data['tags'], list):
             memory_to_store['tags'] = memory_data['tags']


        # Store the memory FOR THE SPECIFIC USER
        # store_memories now takes user_id
        success = await memory_intelligence.store_memories(
            memories=[memory_to_store], # Pass as a list
            user_id=user_id # <-- PASS user_id
        )

        if success:
            logger.info(f"Successfully created memory via /memory/create for user '{user_id}'.") # Log user_id
            return {"status": "success", "memory": memory_to_store} # Return the created memory object
        else:
            # This could mean duplicate or save failure
            logger.warning(f"Manual memory creation via /memory/create failed for user '{user_id}' (duplicate or save error).") # Log user_id
            raise HTTPException(status_code=409, detail="Memory might be a duplicate or failed to save.")

    except ValueError as ve: # Catch potential errors from store_memories if user_id is invalid
         logger.error(f"❌ Value Error creating memory via /memory/create for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory storage: {str(ve)}")
    except HTTPException as http_exc:
        # Re-raise specific HTTP exceptions (like 400, 409, 422)
        raise http_exc
    except Exception as e:
        logger.error(f"Error creating memory via /memory/create for user '{user_id}': {e}", exc_info=True) # Log user_id
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# --- Replace the existing observe_conversation function with this ---
@memory_router.post("/observe")
async def observe_conversation(
    request: ObservationRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager_from_state)
):
    """
    Process a conversation and extract memories.
    Enhanced to be more selective about what gets stored.
    """
    user_id = request.user_name
    if not user_id:
        logger.warning("🧠 [observe] Missing user_name for memory observation")
        raise HTTPException(400, "Missing user_name for memory observation")

    logger.debug(
        f"Received observation request for user '{user_id}': "
        f"user_message='{request.user_message[:50]}...', "
        f"ai_response='{request.ai_response[:50]}...'"
    )
    
    # First, check for explicit memory intent
    try:
        # Call memory detection to check intent
        detection_result = await detect_memory_intent_api(
            MemoryDetectRequest(
                original_prompt=request.user_message,
                response_text=request.ai_response,
                user_name=user_id
            ),
            model_manager
        )
        
        detection_text = detection_result.get("detection_result", "")
        
        # Check if memory intent was detected
        if "MEMORY_DETECTED: YES" in detection_text:
            logger.info(f"🧠 [observe] Memory intent detected for user '{user_id}'")
            
            # Extract memory details
            memory_content = None
            content_match = re.search(r"MEMORY_CONTENT: (.*?)(?:\n|$)", detection_text, re.DOTALL)
            if content_match:
                memory_content = content_match.group(1).strip()
            
            category_match = re.search(r"MEMORY_CATEGORY: (.*?)(?:\n|$)", detection_text)
            category = category_match.group(1).strip() if category_match else "personal_info"
            
            importance_match = re.search(r"MEMORY_IMPORTANCE: (.*?)(?:\n|$)", detection_text)
            try:
                importance = max(0.1, min(1.0, float(importance_match.group(1).strip()))) if importance_match else 0.8
            except:
                importance = 0.8
            
            if memory_content:
                # Create memory manually
                memory = Memory(
                    content=memory_content,
                    category=category,
                    importance=importance,
                    type="detected",
                    user_id=user_id
                )
                
                # Store memory immediately
                memory_dict = memory.dict()
                if "created" not in memory_dict:
                    memory_dict["created"] = datetime.datetime.now().isoformat()
                
                logger.info(f"🧠 [observe] Adding detected memory: {memory_content[:50]}...")
                
                # Call store_memories directly
                await memory_intelligence.store_memories([memory_dict], user_id=user_id)
                
                return {
                    "status": "success",
                    "message": "Memory detected and stored",
                    "memory": memory_dict
                }
            else:
                logger.warning("🧠 [observe] Memory intent detected but no content extracted")
        else:
            logger.info(f"🧠 [observe] No memory intent detected for user '{user_id}'")
            
            # Add option to schedule a deeper analysis in background
            if request.memory_creation_settings.get("analyze_conversations", False):
                background_tasks.add_task(
                    process_completed_exchange,
                    model_manager,
                    request.user_message,
                    request.ai_response,
                    user_id,
                    gpu_id=1
                )
                return {
                    "status": "scheduled",
                    "message": "No explicit memory but scheduled deeper analysis"
                }
            else:
                # Return 'skipped' to indicate we didn't need memory here
                return {
                    "status": "skipped",
                    "message": "No memory intent detected"
                }
    except Exception as e:
        logger.error(f"🧠 [observe] Error during memory observation: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error during observation: {str(e)}"
        }

# Add this new endpoint definition

@memory_router.get("/get_all")
async def get_all_backend_memories(
    # Make user_id a required query parameter. Frontend must send e.g., /get_all?user_id=bernard
    user_id: str = Query(...)
):
    """
    Returns all memories currently stored in the backend for the SPECIFIED user.
    Intended for frontend synchronization based on the active user profile.
    Requires user_id as a query parameter.
    """
    logger.info(f"Request received for /get_all backend memories for user '{user_id}'.") # Log user_id

    # Basic validation (FastAPI's Query(...) already makes it required)
    if not user_id:
         # This case might not be reachable if Query(...) is used, but good practice
         logger.error("get_all endpoint called without a user_id query parameter.")
         raise HTTPException(status_code=400, detail="user_id query parameter is required.")

    try:
        # Use the existing function from memory_intelligence, passing the user_id
        all_memories = memory_intelligence.get_memory_store(user_id=user_id) # <-- PASS user_id
        logger.info(f"Returning {len(all_memories)} memories from backend store for user '{user_id}'.") # Log user_id
        return {"status": "success", "memories": all_memories}

    except ValueError as ve: # Catch potential errors from get_memory_store if user_id is invalid
         logger.error(f"❌ Value Error getting memories for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory retrieval: {str(ve)}")
    except Exception as e:
        logger.error(f"Error in /get_all endpoint for user '{user_id}': {e}", exc_info=True) # Log user_id
        # Raise HTTPException so the frontend knows something went wrong
        raise HTTPException(status_code=500, detail=f"Failed to retrieve backend memories for user '{user_id}'.")
    
# --- Replace the existing list_memories function with this ---
@memory_router.get("/list")
async def list_memories(
    # Add user_id as a required query parameter
    user_id: str = Query(...)
):
    """
    Return all stored memories for a specific user.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes") # Get logger instance
    logger.info(f"GET /list: Attempting to list memories for user '{user_id}'.") # Log user_id

    if not user_id:
        # Should be caught by Query(...), but good practice
        logger.error("Cannot list memories without a user_id.")
        raise HTTPException(status_code=400, detail="user_id query parameter is required to list memories.")

    try:
        # Get memories from the store FOR THE SPECIFIC USER
        memories = memory_intelligence.get_memory_store(user_id=user_id) # <-- PASS user_id
        logger.info(f"Found {len(memories)} memories for user '{user_id}'.") # Log user_id
        return {"status": "success", "memories": memories, "count": len(memories)}

    except ValueError as ve: # Catch potential errors from get_memory_store if user_id is invalid
         logger.error(f"❌ Value Error listing memories for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory listing: {str(ve)}")
    except Exception as e:
        logger.error(f"Error listing memories for user '{user_id}': {e}", exc_info=True) # Log user_id
        # Return 500 for unexpected errors during retrieval
        raise HTTPException(status_code=500, detail=f"Failed to list memories for user '{user_id}': {str(e)}")

# --- Replace the existing add_memory function with this ---
@memory_router.post("/add")
async def add_memory(
    request: Request, # Add the request parameter to access app state
    memory: Memory
): 
    """
    Manually add a memory for a specific user.
    Tries multiple methods to determine user_id with fallbacks.
    """
    logger = logging.getLogger("memory_routes")
    
    # 1. First try to get user_id from the Memory model itself
    user_id = memory.user_id
    
    # 2. Then try to extract from userProfile if available
    if not user_id and hasattr(memory, 'userProfile') and memory.userProfile:
        profile = memory.userProfile
        profile_id = (
            profile.get("id") or 
            profile.get("userId") or 
            profile.get("user_id")
        )
        if profile_id:
            user_id = str(profile_id)
            logger.info(f"Using user_id from userProfile: {user_id}")
    
    # 3. If still not found, try app state's active profile
    if not user_id and request and hasattr(request.app.state, "active_profile_id"):
        user_id = request.app.state.active_profile_id
        if user_id:
            logger.info(f"Using active_profile_id from app state: {user_id}")
    
    # 4. Last resort: try user_utils (for when request context isn't available)
    if not user_id:
        try:
            from . import user_utils
            user_id = user_utils.get_active_profile_id()
            if user_id:
                logger.info(f"Fallback to user_utils.get_active_profile_id(): {user_id}")
        except Exception as e:
            logger.error(f"Error accessing user_utils for fallback user_id: {e}")
    
    # Final check - if we still don't have a user_id, reject the request
    if not user_id:
        logger.error("Cannot add memory - no user_id found in any source")
        raise HTTPException(status_code=400, detail="Could not determine user_id from any source")

    try:
        # Format the memory dictionary from the Pydantic model
        memory_dict_to_store = memory.dict(exclude={"user_id"})

        # Add created timestamp if not present
        if "created" not in memory_dict_to_store or not memory_dict_to_store["created"]:
            memory_dict_to_store["created"] = datetime.datetime.now().isoformat()

        # Add accessed count if not present
        if "accessed" not in memory_dict_to_store:
            memory_dict_to_store["accessed"] = 0

        memory_content = memory_dict_to_store.get("content", "").strip()

        # Validate memory content
        if not memory_intelligence.is_valid_memory_content(memory_content):
            logger.warning(f"Invalid memory content for user '{user_id}': {memory_content[:50]}...")
            raise HTTPException(status_code=422, detail="Invalid memory content provided")

        # Store the memory
        storage_success = await memory_intelligence.store_memories(
            memories=[memory_dict_to_store],
            user_id=user_id
        )

        if storage_success:
            logger.info(f"Added new memory for user '{user_id}': {memory_content[:50]}...")
            return {"status": "success", "memory": memory_dict_to_store}
        else:
            logger.warning(f"Memory add failed for user '{user_id}' (duplicate or save error)")
            raise HTTPException(status_code=409, detail="Memory might be a duplicate or failed to save")

    except ValueError as ve:
        logger.error(f"❌ Value Error adding memory for user '{user_id}': {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid user_id for memory storage: {str(ve)}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error adding memory for user '{user_id}': {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- Replace the existing sync_client_memories function with this ---
@memory_router.post("/sync")
async def sync_client_memories(
    # Add user_id as a required query parameter
    user_id: str = Query(...),
    # Expect a list of memory objects in the body
    memories: List[Dict[str, Any]] = Body(...)
):
    """
    Sync memories from client to server for a specific user.
    Adds memories from the list that are not already present (exact or semantic match)
    in the user's backend store.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes") # Get logger instance
    logger.info(f"POST /sync: Attempting to sync {len(memories)} memories from client for user '{user_id}'.") # Log user_id

    if not user_id:
        # Should be caught by Query(...), but belt-and-suspenders
        logger.error("Cannot sync memories without a user_id.")
        raise HTTPException(status_code=400, detail="user_id query parameter is required to sync memories.")

    try:
        # Get existing memories for the specific user
        all_memories = memory_intelligence.get_memory_store(user_id=user_id)
        # Create a set of existing contents for faster lookup
        existing_contents = {m.get("content", "").strip().lower() for m in all_memories if m.get("content")}

        added_count = 0
        skipped_count = 0
        newly_added_memories = [] # Keep track of memories actually added in this sync

        for memory in memories:
            # Skip invalid memories or memories missing content
            if not isinstance(memory, dict) or "content" not in memory:
                skipped_count += 1
                continue

            content = memory.get("content", "").strip()
            if not content:
                skipped_count += 1
                continue

            # Validate memory content before checking duplicates
            if not memory_intelligence.is_valid_memory_content(content):
                logger.debug(f"Skipping invalid sync content for user '{user_id}': {content[:50]}...")
                skipped_count += 1
                continue

            # Check for exact duplicates first (case-insensitive)
            normalized_content = content.lower()
            if normalized_content in existing_contents:
                skipped_count += 1
                continue # Skip exact duplicate

            # Check for semantic similarity against existing memories
            # (Note: This can be slow if the existing store is large. Consider optimizing if needed)
            is_semantic_duplicate = False
            # Maybe only check against recent N memories for performance?
            # For now, check against all loaded memories for the user
            for existing_memory in all_memories:
                if memory_intelligence.does_it_basically_mean_the_same_thing(
                    existing_memory.get("content", ""),
                    content,
                    threshold=memory_intelligence.CONFIG["similarity_threshold"]
                ):
                    is_semantic_duplicate = True
                    # logger.debug(f"Skipping semantic duplicate during sync for user '{user_id}': {content[:50]}...")
                    break

            if is_semantic_duplicate:
                skipped_count += 1
                continue

            # If not duplicate, prepare to add it
            # Add timestamps if missing (client should ideally provide)
            if "created" not in memory or not memory["created"]:
                memory["created"] = datetime.datetime.now().isoformat()
            if "accessed" not in memory:
                memory["accessed"] = 0
            # Ensure type consistency if needed
            memory["importance"] = float(memory.get('importance', 0.7))

            # Add to the list of memories to be saved and update lookup sets
            newly_added_memories.append(memory)
            existing_contents.add(normalized_content) # Add to exact match set
            added_count += 1

        # Only save if new memories were identified
        if added_count > 0:
            # Combine existing and newly added memories
            all_memories.extend(newly_added_memories)
            save_success = memory_intelligence.save_memory_store(all_memories, user_id=user_id) # <-- PASS user_id
            if save_success:
                logger.info(f"Successfully synced and saved {added_count} new memories for user '{user_id}'. Skipped {skipped_count}.") # Log user_id
            else:
                logger.error(f"Failed to save synced memories for user '{user_id}'.") # Log user_id
                # Return an error if save failed
                raise HTTPException(status_code=500, detail=f"Added {added_count} memories but failed to save the store for user '{user_id}'.")
        else:
            logger.info(f"Sync completed for user '{user_id}'. No new unique memories added. Skipped {skipped_count}.") # Log user_id

        # Return success status, including counts
        return {
            "status": "success",
            "added_count": added_count,
            "skipped_count": skipped_count,
            "total_memories_after_sync": len(all_memories) # Reflects count *after* potential adds
        }

    except ValueError as ve: # Catch potential errors from get/save store if user_id is invalid
         logger.error(f"❌ Value Error syncing memories for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory sync: {str(ve)}")
    except Exception as e:
        logger.error(f"Error syncing memories for user '{user_id}': {e}", exc_info=True) # Log user_id
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error during sync: {str(e)}")

# --- Replace the existing clear_memories function with this ---
@memory_router.delete("/clear") # Changed to DELETE method as it's a destructive action
async def clear_memories(
    # Add user_id as a required query parameter
    user_id: str = Query(...)
):
    """
    Clear all memories for a specific user.
    Requires user_id as a query parameter.
    (Effectively calls purge_memory_store for the user).
    """
    logger = logging.getLogger("memory_routes") # Get logger instance
    logger.info(f"DELETE /clear: Attempting to clear memories for user '{user_id}'.") # Log user_id

    if not user_id:
        # Should be caught by Query(...), but good practice
        logger.error("Cannot clear memories without a user_id.")
        raise HTTPException(status_code=400, detail="user_id query parameter is required to clear memories.")

    try:
        # Get current memory count for reporting (optional, but nice)
        # Note: This adds an extra read operation before the purge/write
        current_memories = memory_intelligence.get_memory_store(user_id=user_id)
        memory_count = len(current_memories)

        # Call the purge function from memory_intelligence for the specific user
        result = memory_intelligence.purge_memory_store(user_id=user_id) # <-- PASS user_id

        if result.get("status") == "success":
            logger.info(f"Successfully cleared {memory_count} memories for user '{user_id}'") # Log user_id
            # Return 200 OK with details
            return {
                "status": "success",
                "cleared_memories": memory_count, # Report count before clearing
                "message": "Memory store completely cleared for this user."
            }
        else:
            # If purge_memory_store failed (e.g., save error)
            error_reason = result.get("reason", "Unknown error during memory clear")
            logger.error(f"Failed to clear memories for user '{user_id}': {error_reason}") # Log user_id
            # Return 500 as the operation failed server-side
            raise HTTPException(status_code=500, detail=f"Failed to clear memory store: {error_reason}")

    except ValueError as ve: # Catch potential errors from get/purge store if user_id is invalid
         logger.error(f"❌ Value Error clearing memories for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory clear: {str(ve)}")
    except Exception as e:
        logger.error(f"Error clearing memories for user '{user_id}': {e}", exc_info=True) # Log user_id
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- Replace the existing purge_memory_endpoint function with this ---
@memory_router.post("/purge") # Keeping POST method as original
async def purge_memory_endpoint(
    # Add user_id as a required query parameter
    user_id: str = Query(...)
):
    """
    Completely purge all memories from a specific user's memory store.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes") # Get logger instance
    logger.info(f"POST /purge: Attempting to purge memories for user '{user_id}'.") # Log user_id

    if not user_id:
        # Should be caught by Query(...), but good practice
        logger.error("Cannot purge memories without a user_id.")
        raise HTTPException(status_code=400, detail="user_id query parameter is required to purge memories.")

    try:
        # Call the purge function from memory_intelligence for the specific user
        result = memory_intelligence.purge_memory_store(user_id=user_id) # <-- PASS user_id

        if result.get("status") == "success":
            logger.info(f"Memory store purged successfully for user '{user_id}'.") # Log user_id
            # Return success status
            return {
                "status": "success",
                "message": "Memory store completely purged for this user.",
                "details": result # Include details from the purge function if any
            }
        else:
            # If purge_memory_store failed
            error_reason = result.get("reason", "Unknown error during memory purge")
            logger.error(f"Failed to purge memories for user '{user_id}': {error_reason}") # Log user_id
            # Return 500 as the operation failed server-side
            raise HTTPException(status_code=500, detail=f"Failed to purge memory store: {error_reason}")

    except ValueError as ve: # Catch potential errors from purge store if user_id is invalid
         logger.error(f"❌ Value Error purging memories for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory purge: {str(ve)}")
    except Exception as e:
        logger.error(f"Error during memory purge for user '{user_id}': {e}", exc_info=True) # Log user_id
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
# --- Replace the existing curate_memory_endpoint function with this ---
@memory_router.post("/curate") # Keeping POST method as original
async def curate_memory_endpoint(
     # Add user_id as a required query parameter
    user_id: str = Query(...)
):
    """
    Run semantic memory curation to remove duplicates for a specific user.
    Requires user_id as a query parameter.
    """
    logger = logging.getLogger("memory_routes") # Get logger instance
    logger.info(f"POST /curate: Attempting to curate memories for user '{user_id}'.") # Log user_id

    if not user_id:
        # Should be caught by Query(...), but good practice
        logger.error("Cannot curate memories without a user_id.")
        raise HTTPException(status_code=400, detail="user_id query parameter is required to curate memories.")

    try:
        # Call the curation function from memory_intelligence for the specific user
        # We previously updated curate_memory_store to accept user_id
        result = memory_intelligence.curate_memory_store(user_id=user_id) # <-- PASS user_id

        # Check the status returned by the curation function
        if result.get("status") in ["success", "partial_success", "skipped"]: # Treat skipped as success from API perspective
            logger.info(f"Memory curation process completed for user '{user_id}' with status: {result.get('status')}") # Log user_id
            # Return success status along with details from the result
            return {
                "status": "success", # Report overall success to the API caller
                "message": f"Memory curation process finished for user '{user_id}'.",
                "details": result # Include the detailed result from the curation function
            }
        else:
            # If curate_memory_store failed (e.g., save error, model error)
            error_reason = result.get("reason", "Unknown error during memory curation")
            logger.error(f"Memory curation failed for user '{user_id}': {error_reason}") # Log user_id
            # Return 500 as the operation failed server-side
            raise HTTPException(status_code=500, detail=f"Memory curation failed: {error_reason}")

    except ValueError as ve: # Catch potential errors from curate store if user_id is invalid
         logger.error(f"❌ Value Error curating memories for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory curation: {str(ve)}")
    except Exception as e:
        logger.error(f"Error during memory curation endpoint for user '{user_id}': {e}", exc_info=True) # Log user_id
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# CORRECTED /model-based-extraction endpoint
@memory_router.post("/model-based-extraction")
async def model_based_extraction(
    request_obj: Request,  # FastAPI Request object for app state
    request: ObservationRequest,  # Pydantic model
    model_manager: ModelManager = Depends(get_model_manager_from_state)
):
    """
    Manually trigger model-based memory extraction for a conversation.
    """
    user_id = request.user_name
    logger = logging.getLogger("memory_routes")

    logger.info(f"POST /model-based-extraction: Triggered for user '{user_id}'.")

    if not user_id:
        logger.error("Cannot perform model-based extraction without user_name/user_id.")
        raise HTTPException(status_code=400, detail="user_name (acting as user_id) is required for model-based extraction.")

    try:
        # Correctly access app state from request_obj
        single_gpu_mode = getattr(request_obj.app.state, "single_gpu_mode", False)
        target_gpu_id = 0 if single_gpu_mode else 0  # Use GPU 0 (3090) for peripheral memory operations
        
        logger.info(f"Using GPU {target_gpu_id} for memory extraction (single_gpu_mode: {single_gpu_mode})")

        model_name = None
        # This logic might be better inside memory_intelligence if reused often
        async with model_manager.lock: # Use lock for safe access
            for mkey, minfo in model_manager.loaded_models.items():
                m_name, m_gpu = mkey
                if m_gpu == target_gpu_id:
                    model_name = m_name
                    logger.info(f"Found suitable model '{model_name}' on GPU {target_gpu_id} for extraction.")
                    break

        if not model_name:
            logger.error(f"No suitable model found on GPU {target_gpu_id} for memory extraction.")
            raise HTTPException(status_code=500, detail=f"No suitable model found on GPU {target_gpu_id} for memory extraction")

        # Extract memories using model-based approach
        # Assuming model_based_memory_creation takes gpu_id but not user_id directly
        memories = await memory_intelligence.model_based_memory_creation(
            model_manager=model_manager,
            model_name=model_name,
            user_message=request.user_message,
            ai_response=request.ai_response,
            gpu_id=target_gpu_id,
            single_gpu_mode=single_gpu_mode
        )

        # Store the extracted memories FOR THE SPECIFIC USER
        storage_success = False
        if memories and len(memories) > 0:
            # Pass user_id to store_memories
            storage_success = await memory_intelligence.store_memories(
                memories=memories,
                user_id=user_id # <-- PASS user_id
            )
            logger.info(f"Storage result after model extraction for user '{user_id}': {storage_success}") # Log user_id
        else:
            logger.info(f"No memories extracted by model for user '{user_id}'.") # Log user_id

        return {
            "status": "success" if len(memories) > 0 else "no_memories",
            "model_used": model_name,
            "memories_created": len(memories),
            "storage_success": storage_success,
            "memories": memories # Return the extracted memories
        }

    except ValueError as ve: # Catch potential errors from store_memories if user_id is invalid
         logger.error(f"❌ Value Error during model extraction for user '{user_id}': {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid user_id for memory storage: {str(ve)}")
    except HTTPException as http_exc:
        # Re-raise specific HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error during model-based extraction for user '{user_id}': {e}", exc_info=True) # Log user_id
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# This router will be imported in main.py
router = memory_router
