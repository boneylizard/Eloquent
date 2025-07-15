# memory_intelligence.py - Memory system for semantic storage, retrieval, and curation

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer, util
import torch
import json
import datetime
import re
import os
import logging # Keep import for getLogger
import traceback
# Assuming inference.py is in the same directory structure allowing this import
from . import inference
# Assuming ModelManager class definition is accessible for type hinting
from .model_manager import ModelManager
import time

# --- Configure logging ---
# --- COMMENTED OUT basicConfig to rely on Uvicorn/FastAPI logging ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_intelligence") # Use getLogger to potentially get Uvicorn/FastAPI configured logger

# Initialize the similarity model with fallback chain
logger.info("🧠 [Startup] Loading similarity model (may download on first run)...")
try:
    similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    logger.info("✅ Successfully loaded SentenceTransformer 'all-mpnet-base-v2' (best quality)")
except Exception as e:
    logger.warning(f"Failed to load all-mpnet-base-v2: {e}")
    try:
        similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        logger.info("✅ Successfully loaded SentenceTransformer 'all-MiniLM-L12-v2' (fallback)")
    except Exception as e2:
        logger.error(f"Failed to load fallback model all-MiniLM-L12-v2: {e2}")
        try:
            similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✅ Successfully loaded SentenceTransformer 'all-MiniLM-L6-v2' (last resort)")
        except Exception as e3:
            logger.error(f"❌ Failed to load any similarity model: {e3}")
            raise RuntimeError("Could not load any sentence transformer model. Please check your internet connection and sentence-transformers installation.")

# --- Define the path to the memory store file ---
try:
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Keep the original path for potential fallback or default behavior if needed
    # MEMORY_STORE_PATH = os.path.join(_CURRENT_DIR, "memory_store.json")
    USER_MEMORY_DIR = os.path.join(_CURRENT_DIR, "user_memories") # <-- ADD THIS
    # Ensure the directory exists
    os.makedirs(USER_MEMORY_DIR, exist_ok=True) # <-- ADD THIS
    logger.info(f"User memory directory set to: {USER_MEMORY_DIR}") # <-- ADD THIS
except NameError:
    # (Keep your existing fallback logic if needed, but add USER_MEMORY_DIR)
    logger.warning("__file__ not defined, using CWD-based path for memory store.")
    _APP_DIR = os.path.join(os.getcwd(), "app")
    # MEMORY_STORE_PATH = os.path.join(_APP_DIR, "memory_store.json")
    USER_MEMORY_DIR = os.path.join(_APP_DIR, "user_memories") # <-- ADD THIS (adjusted path)
    os.makedirs(USER_MEMORY_DIR, exist_ok=True) # <-- ADD THIS
    logger.info(f"User memory directory (fallback): {USER_MEMORY_DIR}") # <-- ADD THIS


# === Model Classes ===
# ... (rest of the models remain the same) ...
class MemoryRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Dict[str, Any]]] = None

class MemoryObservationRequest(BaseModel):
    user_message: str
    ai_response: str
    conversation_history: Optional[List[Dict[str, Any]]] = None

class Memory(BaseModel):
    content: str
    category: str = "general"
    importance: float = 0.7
    type: str = "auto"
    tags: List[str] = []
    created: Optional[str] = None
    accessed: Optional[int] = 0
    last_accessed: Optional[str] = None
    user_id: Optional[str] = None # <-- ADD THIS
    user_name: Optional[str] = None # <-- ADD THIS

# === Memory System Configuration ===
# ... (CONFIG remains the same) ...
CONFIG = {
    "max_tokens": 2048,
    "temperature": 0.3,
    "min_memories": 5,
    "similarity_threshold": 0.7,
    "debug_logging": True,
    "memory_extraction_temperature": 0.1,
    "memory_quality_threshold": 0.55,
    "banned_prefixes": [],
    "memory_content_min_length": 10,
    "memory_deduplication_window": 1000,
}


def get_user_memory_path(user_id: Optional[str] = None) -> str:
    """
    Generates the file path for a specific user's memory store.
    If user_id is not provided, attempts to determine it through fallbacks.
    """
    logger = logging.getLogger("memory_intelligence")
    
    # If user_id is provided, use it directly
    if user_id and isinstance(user_id, str):
        logger.debug(f"🧠 [Path] Using provided user_id={user_id!r}")
    # Otherwise try fallbacks
    else:
        logger.debug(f"🧠 [Path] No valid user_id provided, trying fallbacks")
        try:
            # Try importing user_utils to get active profile
            from . import user_utils
            fallback_id = user_utils.get_active_profile_id()
            if fallback_id:
                user_id = fallback_id
                logger.info(f"🧠 [Path] Using active profile ID from user_utils: {user_id}")
            else:
                logger.warning("🧠 [Path] No active profile ID found via user_utils")
        except Exception as e:
            logger.error(f"🧠 [Path] Error getting fallback user_id: {e}")
    
    # Final check after fallbacks
    if not user_id or not isinstance(user_id, str):
        raise ValueError("Could not determine valid user_id for memory path through any method")
    
    # Sanitize: keep only alphanumerics, dash or underscore
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
    if not safe_user_id:
        raise ValueError(f"Could not create safe filename from user_id: {user_id}")
    
    path = os.path.join(USER_MEMORY_DIR, f"{safe_user_id}_memory_store.json")
    logger.debug(f"🧠 [Path] Memory path is {path}")
    return path

# character_lore_entries = [
#     {"content": "Lore about character A", "keywords": ["A", "character A"]},
def get_triggered_character_lore(prompt: str, active_character: Optional[Dict[str, Any]]) -> List[str]:
    """
    Scans the prompt for keywords defined in the active character's lore entries
    and returns the content of the triggered entries.

    Args:
        prompt: The incoming user message text.
        active_character: The character object loaded from AppContext state,
                          expected to have a 'loreEntries' key containing a list like:
                          [{ content: str, keywords: List[str] }, ...]

    Returns:
        A list of unique lore content strings that were triggered by keywords in the prompt.
    """
    if not active_character or not isinstance(active_character, dict):
        logger.debug("[Lore Retrieval] No active character provided.")
        return []
    if not prompt or not isinstance(prompt, str):
        logger.debug("[Lore Retrieval] No prompt string provided.")
        return []

    # Safely get lore entries, default to empty list
    lore_entries = active_character.get('loreEntries', [])
    if not isinstance(lore_entries, list) or not lore_entries:
         # logger.debug(f"[Lore Retrieval] No lore entries found for character {active_character.get('name', 'Unknown')}.")
         return []

    triggered_lore_content = set() # Use a set to automatically handle duplicates
    prompt_lower = prompt.lower() # Convert prompt to lowercase once for efficiency

    # logger.debug(f"[Lore Retrieval] Scanning prompt for triggers from {len(lore_entries)} lore entries...")

    for entry in lore_entries:
        if not isinstance(entry, dict) or 'content' not in entry or 'keywords' not in entry:
            # logger.warning(f"[Lore Retrieval] Skipping invalid lore entry format: {entry}")
            continue

        content = entry.get('content', '').strip()
        keywords = entry.get('keywords', [])

        # Ensure keywords is a list and content exists
        if not content or not isinstance(keywords, list) or not keywords:
            continue

        # Check if any keyword exists in the prompt (case-insensitive)
        for keyword in keywords:
            if not keyword or not isinstance(keyword, str) or not keyword.strip():
                continue # Skip empty keywords

            # Perform case-insensitive check
            if keyword.strip().lower() in prompt_lower:
                # logger.info(f"[Lore Retrieval] Keyword '{keyword}' triggered lore: '{content[:50]}...'")
                triggered_lore_content.add(content)
                break # Move to the next lore entry once one keyword matches

    if triggered_lore_content:
        logger.info(f"[Lore Retrieval] Found {len(triggered_lore_content)} unique triggered lore entries for character {active_character.get('name', 'Unknown')}.")
        return list(triggered_lore_content) # Convert set back to list
    else:
        # logger.debug(f"[Lore Retrieval] No keywords triggered for character {active_character.get('name', 'Unknown')}.")
        return []

# <<<< End of new function >>>>
# === Core Memory Functions ===
def is_valid_memory_content(content: str) -> bool:
    """Check if memory content is valid with stricter validation to prevent memory pollution."""
    if not content or len(content) < CONFIG["memory_content_min_length"]:
        logger.debug(f"❌ Rejected memory - too short: '{content}'")
        return False
    
    normalized = content.lower().strip()
    logger.debug(f"Checking memory content: '{content}'")
    
    # Check for explicit memory intent patterns first - these should override other rejections
    memory_intent_patterns = [
        r"please remember", r"remember that", r"don't forget",
        r"make note", r"keep in mind", r"for future reference"
    ]
    
    for pattern in memory_intent_patterns:
        if re.search(pattern, normalized):
            logger.info(f"✅ Explicit memory request detected: '{content[:50]}...'")
            return True
    
    # --- Instructional prompt rejection patterns ---
    instructional_patterns = [
        # Writing/generation instructions
        r"^write", r"^create", r"^generate", r"^compose", r"^draft",
        r"^summarize", r"^analyze", r"^explain", r"^help me",
        
        # Roleplaying prompts
        r"^roleplay", r"^pretend", r"^act as", r"^you are", r"^you're now",
        
        # Commands
        r"^can you", r"^could you", r"^would you", r"^please", 
        r"^I need you to", r"^I want you to",
        
        # First-person expressions that aren't worth remembering
        r"^I think", r"^I feel", r"^I believe", r"^in my opinion"
    ]
    
    for pattern in instructional_patterns:
        if re.search(pattern, normalized):
            logger.info(f"❌ Rejected instructional prompt: '{content[:50]}...'")
            return False
    
    # Existing banned prefix checks
    for prefix in CONFIG["banned_prefixes"]:
        if normalized.startswith(prefix.lower()):
            logger.info(f"❌ Rejected memory with banned prefix: '{prefix}' in '{content[:50]}...'")
            return False
    
    # Reject questions unless they're prefixed with an explicit memory request
    if content.endswith('?') and len(content.split()) > 3:
        logger.info(f"❌ Rejected question memory: '{content[:50]}...'")
        return False
    
    # Reject meta-memory discussions
    meta_memory_terms = [
        "memory system", "meta-memory", "memory context",
        "memory extraction", "memory creation", "memory storage",
        "memory logic", "duplicate detection"
    ]
    
    for term in meta_memory_terms:
        if term in normalized:
            logger.info(f"❌ Rejected meta-memory discussion: '{term}' in '{content[:50]}...'")
            logger.error(f"META-MEMORY REJECTED: term '{term}' found in '{normalized}'")
            return False
    
    # Reject incomplete thoughts ending with ellipsis
    if content.endswith(("...", "…")) and len(content) < 30:
        logger.info(f"❌ Rejected memory with trailing ellipsis: '{content}'")
        return False
    
    # If we've made it through all the filters, the content is valid
    logger.debug(f"✅ Valid memory content: '{content[:50]}...'")
    return True

# ... (does_it_basically_mean_the_same_thing remains the same) ...
def does_it_basically_mean_the_same_thing(text1, text2, threshold=None):
    """Checks if two texts basically mean the same thing with improved debugging."""
    if threshold is None: threshold = CONFIG["similarity_threshold"]
    if not text1 or not text2: return False
    if text1.lower().strip() == text2.lower().strip():
        logger.info(f"✅ Exact match detected between texts")
        return True
    try:
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())
        if not chars1 or not chars2: return False # Avoid division by zero
        char_overlap = len(chars1.intersection(chars2)) / max(len(chars1), len(chars2))
        if char_overlap < 0.5:
            logger.debug(f"Character similarity too low ({char_overlap:.2f}) - texts likely different")
            return False
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2: return False # Avoid division by zero
        word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        if word_overlap > 0.8:
            logger.info(f"⚠️ High word overlap ({word_overlap:.2f}) - texts likely similar")
            logger.info(f"  Text 1: '{text1[:50]}...'")
            logger.info(f"  Text 2: '{text2[:50]}...'")
            return True
        embedding1 = similarity_model.encode(text1, convert_to_tensor=True)
        embedding2 = similarity_model.encode(text2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
        similarity = cosine_scores[0][0].item()
        log_level = logging.INFO if similarity > threshold * 0.3 else logging.DEBUG
        logger.log(log_level, f"Similarity between texts: {similarity:.4f} (threshold: {threshold:.4f})")
        if similarity > threshold:
            logger.info(f"⚠️ Semantic duplicate detected with similarity {similarity:.4f}:")
            logger.info(f"  Text 1: '{text1[:50]}...'")
            logger.info(f"  Text 2: '{text2[:50]}...'")
        return similarity > threshold
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        logger.error(f"  Text 1: '{text1[:50]}'")
        logger.error(f"  Text 2: '{text2[:50]}'")
        return text1.lower().strip() == text2.lower().strip()

# ... (get_memory_store remains the same - includes BOM fix) ...
def get_memory_store(user_id: str) -> List[Dict[str, Any]]:
    """
    Read the memory store from disk for a specific user.
    If the file doesn't exist, create it as an empty list.
    Always returns a list of memories.
    """
    memory_path = get_user_memory_path(user_id)
    logger.info(f"🧠 [I/O] Loading memory store from {memory_path}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    # If file doesn't exist, initialize it
    if not os.path.exists(memory_path):
        try:
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info(f"🧠 [I/O] Created new empty memory store at {memory_path}")
        except Exception as e:
            logger.error(f"🧠 [I/O] Failed to create memory store at {memory_path}: {e}", exc_info=True)
            return []

    # Now read and parse
    try:
        start = time.time()
        with open(memory_path, 'r', encoding='utf-8-sig') as f:
            raw = f.read().strip()
            if not raw:
                # Empty file—treat as no memories
                logger.info(f"🧠 [I/O] Memory store empty at {memory_path}")
                return []
            data = json.loads(raw)
        elapsed = time.time() - start

        # Data may be either a list of memories or an object with a "memories" key
        if isinstance(data, list):
            memories = data
        elif isinstance(data, dict) and "memories" in data and isinstance(data["memories"], list):
            memories = data["memories"]
        else:
            logger.warning(f"🧠 [I/O] Unexpected memory store format at {memory_path}, resetting to list")
            memories = []

        logger.info(f"🧠 [I/O] Read {len(memories)} memories in {elapsed:.2f}s")
        return memories

    except json.JSONDecodeError as jde:
        logger.error(f"🧠 [I/O] JSON decode error in {memory_path}: {jde}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"🧠 [I/O] Error reading memory store at {memory_path}: {e}", exc_info=True)
        return []

# ... (save_memory_store remains the same - includes atomic write) ...
def save_memory_store(memories, user_id: str) -> bool:
    """
    Save the memory list for a specific user to disk, using atomic write.
    """
    # Validate user_id
    if not user_id or not isinstance(user_id, str):
        logger.warning("save_memory_store called without a valid user_id; skipping save.")
        return False

    try:
        # 1) Determine the file path
        memory_store_path = get_user_memory_path(user_id)
        logger.info(f"🧠 [I/O] Saving {len(memories)} memories to {memory_store_path}")

        # 2) Ensure directory exists
        os.makedirs(os.path.dirname(memory_store_path), exist_ok=True)

        # 3) Prepare data to save
        data_to_save = memories if isinstance(memories, list) else []

        # 4) Write atomically
        temp_path = memory_store_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        # On Windows, use os.replace; on others, os.rename works too
        if os.path.exists(memory_store_path):
            os.replace(temp_path, memory_store_path)
        else:
            os.rename(temp_path, memory_store_path)

        logger.info(f"✅ Successfully saved {len(data_to_save)} memories for user '{user_id}' to {memory_store_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Error saving memory store to {memory_store_path}: {e}", exc_info=True)
        return False

# --- ADD THIS NEW FUNCTION ---
async def delete_memory_for_user(user_id: str, content_to_delete: str) -> bool:
    logger.info(f"🧠 [Delete] Attempting to delete memory from {user_id!r}: {content_to_delete[:50]!r}")
    # …
    """
    Deletes a specific memory based on its content from a specific user's memory store.

    Args:
        user_id: The identifier for the user whose memory store to modify.
        content_to_delete: The exact content string of the memory to delete.

    Returns:
        True if the memory was found and the store was successfully saved, False otherwise.
    """
    logger = logging.getLogger(__name__) # Get logger instance
    logger.info(f"Attempting to delete memory for user '{user_id}' with content: {content_to_delete[:50]}...")

    if not user_id or not content_to_delete:
        logger.error("delete_memory_for_user called with missing user_id or content.")
        return False

    try:
        # 1. Load the current memory store for the user
        all_memories = get_memory_store(user_id=user_id)

        # 2. Find and filter out the memory to delete
        memory_found = False
        updated_memories = []
        for memory in all_memories:
            # Compare content exactly (strip whitespace for robustness)
            if memory.get("content", "").strip() == content_to_delete.strip():
                memory_found = True
                logger.debug(f"Found memory to delete for user '{user_id}'.")
            else:
                updated_memories.append(memory)

        # 3. If memory wasn't found, return False
        if not memory_found:
            logger.warning(f"Memory content not found for deletion for user '{user_id}'.")
            return False
            if not memory_found:
                logger.warn("🧠 [Delete] No matching memory found; nothing to delete")
                return False
            logger.info("🧠 [Delete] Successfully deleted memory")
            return True

        # 4. Save the updated memory store (which excludes the deleted memory)
        save_success = save_memory_store(updated_memories, user_id=user_id)

        if save_success:
            logger.info(f"Successfully deleted memory and saved store for user '{user_id}'.")
            return True
        else:
            logger.error(f"Failed to save updated memory store after deletion for user '{user_id}'.")
            return False

    except ValueError as ve: # Catch potential errors from get/save store if user_id is invalid
         logger.error(f"❌ Value Error during memory deletion process for user '{user_id}': {ve}", exc_info=True)
         return False # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error during memory deletion for user '{user_id}': {e}", exc_info=True)
        return False # Indicate failure

# === Memory Retrieval System ===

# Now let's fix the get_llm_refined_context function that processes the retrieved memories
async def get_llm_refined_context(
    model_manager: ModelManager,
    original_prompt: str,
    candidate_memories: List[Dict[str, Any]],
    gpu_id: int = None,  # Changed from hardcoded 1 to allow flexibility
    single_gpu_mode: bool = False,
) -> Dict[str, Any]:
    """Uses an LLM on the specified GPU to refine and select the most relevant context."""
    # Add this near the beginning of the function
    if gpu_id is None:
        gpu_id = 0 if single_gpu_mode else 1
    logger = logging.getLogger("memory_intelligence")  # Use the same logger as above
    logger.info(f"🧠 [Refine] Sending {len(candidate_memories)} candidates to LLM for final pruning")
    t0 = time.time()
    
    if not candidate_memories:
        logger.warning("🧠 [Refine] No candidate memories provided for LLM refinement.")
        return {"status": "no_candidates", "refined_context": ""}

    # Format the candidate memories for the prompt with better numbering and structure
    formatted_candidates = ""
    for i, mem in enumerate(candidate_memories):
        content = mem.get('content', 'N/A')
        category = mem.get('category', 'N/A')
        importance = mem.get('importance', 'N/A')
        formatted_candidates += f"[{i+1}] ({category}, importance: {importance:.1f}): {content}\n\n"
    
    # Improved refinement prompt with clearer instructions
    refinement_prompt = f"""
You are an expert AI assistant specializing in text analysis. Your task is to analyze a list of text-based memories and select the most relevant ones to provide rich context for another AI.

**PRIMARY GOAL: CONTEXT QUANTITY**
Your main goal is to select a high quantity of relevant memories to ensure the other AI has the richest possible context. It is better to include a memory that is only partially relevant than to provide too few.

**CRITICAL INSTRUCTIONS:**
1.  **SELECT 5-7 MEMORIES:** You MUST select a minimum of 5, and up to 7, of the most relevant memories from the candidate list. If there are fewer than 5 relevant memories, select all that are relevant.
2.  **EXTRACT FULL CONTENT:** You MUST extract the full text content of the memories you select.
3.  **DO NOT USE NUMBERS:** Do not include the original candidate numbers (e.g., "[1]", "[2]") in your output.
4.  **FORMAT AS LIST:** Present the output as a simple list, with each memory on a new line preceded by a bullet point (e.g., "- ...").
5.  **NO COMMENTARY:** Output ONLY the bulleted list of memories.
6.  **EMPTY RESPONSE:** If, and only if, NONE of the candidates are relevant, respond with the exact text: NO_RELEVANT_CONTEXT

TOPIC OF USER'S QUERY:
\"\"\"{original_prompt}\"\"\"

CANDIDATE MEMORIES:
---
{formatted_candidates.strip()}
---

REFINED CONTEXT FOR AI (Your output MUST be a bulleted list of 5-7 memory contents or NO_RELEVANT_CONTEXT):
"""

    try:
        # Find a suitable model on the memory GPU
        model_name = await model_manager.find_suitable_model(gpu_id=gpu_id)
        if not model_name:
            logger.error(f"🧠 [Refine] No suitable model found on GPU {gpu_id} for context refinement.")
            return {"status": "error", "error": f"No model found on GPU {gpu_id}"}

        logger.info(f"🧠 [Refine] Using model '{model_name}' on GPU {gpu_id} for refinement.")
        
        # Generate the refined context
        llm_response_text = await inference.generate_text(
            model_manager=model_manager,
            model_name=model_name,
            prompt=refinement_prompt,
            max_tokens=1500,  # Increased token limit for more detailed context
            temperature=0.2,  # Keep low temperature for deterministic output
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            stop_sequences=["USER QUERY:", "CANDIDATE MEMORIES:", "---"],
            gpu_id=gpu_id
        )
        # --- RESILIENCY FALLBACK (from Claude's suggestion) ---
        # If the model returns reference numbers instead of content, extract manually.
        if llm_response_text and re.search(r'\[\d+\]', llm_response_text):
            logger.warning(f"🧠 [Refine] LLM returned reference numbers ('{llm_response_text}'). Extracting content manually.")
            
            # Find all numbers within square brackets
            numbers = re.findall(r'\[(\d+)\]', llm_response_text)
            
            extracted_content = []
            for num_str in numbers:
                try:
                    # Convert to a 0-based index
                    idx = int(num_str) - 1
                    if 0 <= idx < len(candidate_memories):
                        content = candidate_memories[idx].get('content', '')
                        if content:
                            # Prepending with a bullet point to match desired format
                            extracted_content.append(f"- {content}")
                except (ValueError, IndexError):
                    continue
            
            # If we successfully extracted content, overwrite the bad response
            if extracted_content:
                llm_response_text = "\n".join(extracted_content)
                logger.info(f"🧠 [Refine] Manually extracted {len(extracted_content)} memories.")

        # Check if we got a valid response (this check now runs on the potentially fixed text)
        if (
            not llm_response_text
            or not isinstance(llm_response_text, str)
            or llm_response_text.strip() == "NO_RELEVANT_CONTEXT"
        ):
            logger.info("🧠 [Refine] LLM indicated no relevant context found or returned non-string.")
            return {"status": "success", "refined_context": ""}

        refined_context = llm_response_text.strip()
        duration = time.time() - t0
        logger.info(
            f"🧠 [Refine] LLM generated context in {duration:.2f}s (length: {len(refined_context)}). "
            f"Preview: {refined_context[:100]}..."
        )
        
        # CRITICAL: Save the raw refined context to DEBUG log for troubleshooting
        logger.debug(f"🧠 [Refine] FULL REFINED CONTEXT:\n{refined_context}")
        
        return {"status": "success", "refined_context": refined_context}

    except Exception as e:
        logger.error(f"🧠 [Refine] Error during LLM context refinement: {e}", exc_info=True)
        return {"status": "error", "error": f"LLM call failed: {str(e)}"}

# First, let's fix the analyze_for_relevant_memories function in memory_intelligence.py
# This function is returning empty results based on your logs

async def analyze_for_relevant_memories(model_manager, prompt, memories=None, gpu_id=None, single_gpu_mode=False, user_id: Optional[str] = None):
    """Find memories relevant to the current prompt"""
    if gpu_id is None:
        gpu_id = 0 if single_gpu_mode else 1
    try:
        logger.info(f"🧠 [Retrieval] Finding relevant memories for prompt: {prompt[:60]!r}")
        
        # Load memories from disk if not provided
        all_memories = []
        if memories and len(memories) > 0:
            all_memories.extend(memories)
            logger.info(f"Using {len(memories)} provided memories")
        else:
            if not user_id:
                logger.warning("🧠 [Retrieval] No user_id provided for memory loading")
                return []
            
            disk_memories = get_memory_store(user_id=user_id)
            if disk_memories and len(disk_memories) > 0:
                all_memories.extend(disk_memories)
                logger.info(f"🧠 [Retrieval] Loaded {len(disk_memories)} memories for user '{user_id}' from disk")
            else:
                logger.warning(f"🧠 [Retrieval] No memories found for user '{user_id}' on disk")
        
        if not all_memories:
            logger.info("🧠 [Retrieval] No memories available for retrieval")
            return []
        
        logger.info(f"🧠 [Retrieval] Scoring {len(all_memories)} candidate memories")
        
        # Get embeddings for the prompt
        prompt_embedding = similarity_model.encode(prompt, convert_to_tensor=True)
        
        # Score memories based on relevance to prompt
        scored_memories = []
        for memory in all_memories:
            memory_content = memory.get('content', '')
 
            
            # Skip invalid memories
            if not memory_content:
                continue
                
            # Get embedding for memory
            try:
                memory_embedding = similarity_model.encode(memory_content, convert_to_tensor=True)
                
                # Calculate semantic similarity
                cosine_scores = util.pytorch_cos_sim(prompt_embedding, memory_embedding)
                semantic_score = cosine_scores[0][0].item()
                
                # Calculate base score
                score = semantic_score
                
                # Add the scored memory
                scored_memories.append((memory, score))
            except Exception as e:
                logger.error(f"🧠 [Retrieval] Error scoring memory: {e}", exc_info=True)
                continue
        
        # Sort memories by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # ADD DEBUG HERE - right after the sort, before filtering
        logger.info("🧠 [DROP DEAD DEBUG] Checking scores for problematic memories:")
        for i, (memory, score) in enumerate(scored_memories[:10]):
            content = memory.get('content', '')
            if 'Drop Dead Fred' in content or 'AI safeguards' in content:
                logger.warning(f"🧠 [DROP DEAD] Position {i+1}: Score {score:.4f} - '{content[:60]}...'")
        
        # Set minimum threshold for relevance
        min_score_threshold = 0.01
        filtered_scored_memories = [(mem, score) for mem, score in scored_memories if score > min_score_threshold]
        
        # Take top memories
        top_count = min(CONFIG["min_memories"], len(filtered_scored_memories))
        relevant_memories = [mem[0] for mem in filtered_scored_memories[:top_count]]
        
        logger.info(f"🧠 [Retrieval] Found {len(relevant_memories)} relevant memories above threshold {min_score_threshold}")
        
        # Update access counts and times
        for memory in relevant_memories:
            memory['last_accessed'] = datetime.datetime.now().isoformat()
            memory['accessed'] = memory.get('accessed', 0) + 1
            
        # Save the updated memory store
        if len(relevant_memories) > 0:
            save_memory_store(all_memories, user_id=user_id)
        
        return relevant_memories
            
    except Exception as e:
        logger.error(f"🧠 [Retrieval] Error in memory retrieval: {e}", exc_info=True)
        return []

# === Memory Creation System ===

async def analyze_for_memory_creation(model_manager, user_message, ai_response, gpu_id=None, single_gpu_mode=False):
    if gpu_id is None:
        gpu_id = 0 if single_gpu_mode else 1
    """Analyze conversation and create memories with better error handling and opt-in logic"""
    try:
        logger.info(f"🧠 Analyzing conversation for memory creation")
        logger.debug(f"User message: {user_message[:100]}...")
        logger.debug(f"AI response: {ai_response[:100]}...")
        
        # Check for explicit memory intent first
        explicit_memory_patterns = [
            r"remember that (.*?)(?:\.|$)",
            r"please remember (.*?)(?:\.|$)",
            r"don't forget (.*?)(?:\.|$)",
            r"make a note (.*?)(?:\.|$)",
            r"for future reference (.*?)(?:\.|$)",
        ]

        # Only proceed if there's an explicit memory request
        explicit_content = None
        has_explicit_intent = False
        for pattern in explicit_memory_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                explicit_content = match.group(1).strip()
                has_explicit_intent = True
                logger.info(f"🧠 [Create] Detected explicit memory request: {explicit_content[:50]}...")
                break

        # If we have explicit content, create a high-importance memory directly
        if has_explicit_intent and explicit_content and len(explicit_content) > 5:
            explicit_memory = {
                "content": explicit_content,
                "category": "explicit_memory",
                "importance": 0.9,  # High importance for explicit requests
                "created": datetime.datetime.now().isoformat(),
                "accessed": 0
            }
            logger.info(f"🧠 [Create] Created explicit memory: {explicit_content}")
            return [explicit_memory]  # Return just the explicit memory

        # If no explicit request, skip memory creation entirely unless you want additional
        # pattern-based and model-based extraction as a fallback
        # For opt-in only, uncomment the next two lines:
        logger.info(f"🧠 [Create] No explicit memory request - skipping memory creation")
        return []  # Return empty list for opt-in only
        
        # If you want to continue with pattern and model extraction as a fallback,
        # comment out the two lines above and uncomment everything below:
        
        """
        # Combine hybrid approach: pattern-based + model-based extraction
        pattern_memories = pattern_based_memory_creation(user_message, ai_response)
        logger.info(f"Pattern-based approach found {len(pattern_memories)} potential memories")
        
        # If we have a model manager and a small model on GPU 1, use it for memory extraction
        model_memories = []
        if model_manager:
            try:
                model_name = model_manager.find_suitable_model(gpu_id=gpu_id)
                if model_name:
                    logger.info(f"Using model {model_name} for memory extraction")
                    model_memories = await model_based_memory_creation(
                        model_manager, 
                        model_name, 
                        user_message, 
                        ai_response, 
                        gpu_id
                    )
                    logger.info(f"Model-based approach found {len(model_memories)} potential memories")
                else:
                    logger.warning("No suitable model found for memory extraction")
            except Exception as e:
                logger.error(f"Error in model-based memory extraction: {e}")
                traceback.print_exc()
                
        # Rest of original function continues here...
        """
            
    except Exception as e:
        logger.error(f"Error in memory creation: {e}")
        traceback.print_exc()
        return []

def pattern_based_memory_creation(user_message, ai_response):
    """Extract memories based on patterns in the conversation"""
    logger.info("Using pattern-based memory creation approach")
    
    # Combine user message and AI response for context
    combined_text = f"User: {user_message}\nAI: {ai_response}"
    
    # Define categories and pattern matching
    extractors = [
        {
            "category": "preferences",
            "patterns": [
                r"(?:I|The user) prefer(?:s)? (.*?)(?:\.|$)",
                r"(?:I|The user) like(?:s)? (.*?)(?:\.|$)",
                r"(?:I|The user) want(?:s)? (.*?)(?:\.|$)",
                r"(?:I|The user) need(?:s)? (.*?)(?:\.|$)",
                r"(?:I|The user) would rather (.*?)(?:\.|$)",
                r"(?:I'm|The user is) looking for (.*?)(?:\.|$)"
            ],
            "importance": 0.8
        },
        {
            "category": "projects",
            "patterns": [
                r"(?:I'm|The user is) (?:building|developing|creating|working on) (.*?)(?:\.|$)",
                r"(?:my|The user's) project(?:s)? (?:called|named)? (.*?)(?:\.|$)",
                r"(?:I|The user) (?:built|developed|created|designed) (.*?)(?:\.|$)"
            ],
            "importance": 0.9
        },
        {
            "category": "skills",
            "patterns": [
                r"(?:I|The user) (?:am|is) (?:skilled|experienced|knowledgeable) in (.*?)(?:\.|$)",
                r"(?:my|The user's) background in (.*?)(?:\.|$)",
                r"(?:I|The user) (?:studied|learned|mastered) (.*?)(?:\.|$)",
                r"(?:I|The user) have expertise in (.*?)(?:\.|$)"
            ],
            "importance": 0.75
        },
        {
            "category": "values",
            "patterns": [
                r"(?:I|The user) (?:believe|value|prioritize) (.*?)(?:\.|$)",
                r"(?:I|The user) (?:think|feel) that (.*?)(?:\.|$)",
                r"(?:it's|it is) important (?:to me|for the user) that (.*?)(?:\.|$)"
            ],
            "importance": 0.85
        },
        {
            "category": "personal_info",
            "patterns": [
                r"(?:I am|The user is) (?:a) (.*?)(?:\.|$)",
                r"(?:I|The user) identif(?:y|ies) as (.*?)(?:\.|$)"
            ],
            "importance": 0.9
        },
        # New patterns for facts and entities
        {
            "category": "facts",
            "patterns": [
                r"(?:I|The user) (?:discovered|found|learned) that (.*?)(?:\.|$)",
                r"(?:I|The user) realized (.*?)(?:\.|$)",
                r"(?:It's|It is) (?:true|confirmed) that (.*?)(?:\.|$)"
            ],
            "importance": 0.7
        },
        {
            "category": "entities",
            "patterns": [
                r"(?:I|The user) (?:has|have|own) (?:a|an) (.*?)(?:\.|$)",
                r"(?:My|The user's) (.*?) is (?:called|named) (.*?)(?:\.|$)"
            ],
            "importance": 0.75
        }
    ]
    
    # Extract information using all patterns
    extracted_memories = []
    
    # Check for each extractor
    for extractor in extractors:
        category = extractor["category"]
        patterns = extractor["patterns"]
        importance = extractor["importance"]
        
        for pattern in patterns:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            
            for match in matches:
                extracted_text = match.group(1).strip()
                
                # Only use if we have a substantial match
                if len(extracted_text) > 5:
                    # Format the memory content based on analysis
                    if category in ["preferences", "values", "skills"]:
                        if not extracted_text.lower().startswith("the user"):
                            memory_content = f"The user {extracted_text}"
                        else:
                            memory_content = extracted_text
                    else:
                        memory_content = extracted_text
                    
                    # Run final validation
                    if is_valid_memory_content(memory_content):
                        # Only add if not a duplicate
                        if not any(memory_content.lower() == m.get("content", "").lower() for m in extracted_memories):
                            memory = {
                                "content": memory_content,
                                "category": category,
                                "importance": importance,
                                "created": datetime.datetime.now().isoformat(),
                                "accessed": 0
                            }
                            extracted_memories.append(memory)
    
    # If no specific memories extracted, create a better fallback memory
    if len(extracted_memories) == 0 and len(user_message) > 10:
        # Extract up to 500 characters for important information
        if len(user_message) > 500:
            shortened_content = user_message[:497] + "..."
        else:
            shortened_content = user_message
        
        # Create a proper memory without the "discussed topic" prefix
        fallback_memory = {
            "content": shortened_content,
            "category": "personal_info",  # Better category than "conversations"
            "importance": 0.7,  # Higher importance
            "created": datetime.datetime.now().isoformat(),
            "accessed": 0
        }
        extracted_memories.append(fallback_memory)

    logger.info(f"Pattern-based approach created {len(extracted_memories)} memories from conversation")
    return extracted_memories

async def model_based_memory_creation(model_manager, model_name, user_message, ai_response, gpu_id=None, single_gpu_mode=False):
    if gpu_id is None:
        gpu_id = 0 if single_gpu_mode else 1
    """Extract memories using a language model"""
    logger.info(f"Using model-based memory creation with model {model_name} on GPU {gpu_id}")
    
    try:
        # Combine user message and AI response
        combined_text = f"User: {user_message}\nAI: {ai_response}"
        
        # Create memory extraction prompt with clear indentation
        extraction_prompt = f"""
TASK: Extract important information from this conversation that should be remembered for future interactions.

RULES:
- Extract facts, preferences, or important context about the user
- Do not extract questions, requests, or general conversation
- Keep extractions brief and specific
- Only extract if genuinely important for future reference

RESPOND WITH EXACTLY THIS FORMAT:
MEMORY_DETECTED: YES or NO
MEMORY_CONTENT: [the specific information to remember]
MEMORY_CATEGORY: [personal_info, preferences, projects, or facts]
MEMORY_IMPORTANCE: [0.1 to 1.0]

CONVERSATION:
{combined_text}

ANALYSIS:
"""
        
        # Call the model
        result = model_manager.generate({
            "model_name": model_name,
            "prompt": extraction_prompt,
            "max_tokens": 1024,
            "temperature": CONFIG["memory_extraction_temperature"],
            "gpu_id": gpu_id
        })
        
        # Process the model output to extract JSON
        try:
            # Find JSON array in the response
            text_output = result["text"] if isinstance(result, dict) else str(result)
            start_idx = text_output.find('[')
            end_idx = text_output.rfind(']') + 1

            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text_output[start_idx:end_idx]
                # Validate JSON format
                memories = json.loads(json_str)
            
                
                # Validate and format memories
                valid_memories = []
                memory_contents = set()
                for memory in memories:
                    if not isinstance(memory, dict) or 'content' not in memory:
                        continue
                    
                    content = memory['content'].strip()
                    
                    # Skip invalid memories
                    if not is_valid_memory_content(content):
                        continue
                    
                    # Format and validate the memory
                    valid_memory = {
                        "content": content,
                        "category": memory.get('category', 'general'),
                        "importance": float(memory.get('importance', 0.7)),
                        "created": datetime.datetime.now().isoformat(),
                        "accessed": 0
                    }
                    
                    # Add details if present
                    if 'details' in memory:
                        valid_memory['details'] = memory['details']
                    
                    valid_memories.append(valid_memory)
                    memory_contents.add(content)
                
                logger.info(f"Model-based approach extracted {len(valid_memories)} valid memories")
                return valid_memories
            else:
                logger.warning("No valid JSON array found in model output")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in model output: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Error in model-based memory creation: {e}")
        traceback.print_exc()
        return []

# Add this function to app/memory_intelligence.py

async def model_based_memory_creation_from_snippet(
    model_manager,
    conversation_snippet: str,
    gpu_id: int = None, # Default to GPU 1 for memory tasks
    model_name: Optional[str] = None, # Optionally specify model
    single_gpu_mode: bool = False, # For single GPU setups
):
    if gpu_id is None:
        gpu_id = 0 if single_gpu_mode else 1
    logger = logging.getLogger("memory_intelligence")  # Use the same logger as above
    """
    Extracts memories from a specific conversation snippet using a language model.
    Designed to be triggered contextually.

    Args:
        model_manager: The ModelManager instance.
        conversation_snippet: The specific text snippet identified for memory creation.
        gpu_id: The GPU ID to use for the memory extraction model.
        model_name: Optional name of the model to use on the specified GPU.

    Returns:
        A list of valid memory dictionaries extracted from the snippet.
    """
    logger.info(f"🧠 Using model on GPU {gpu_id} for memory extraction from snippet.")
    logger.debug(f"Snippet received: {conversation_snippet[:150]}...") # Log beginning of snippet

    try:
        # If no specific model name provided, find a suitable one on the target GPU
        if not model_name:
            model_name = model_manager.find_suitable_model(gpu_id=gpu_id)
            if not model_name:
                logger.error(f"❌ No suitable model found on GPU {gpu_id} for memory extraction.")
                return []
            logger.info(f"Using auto-selected model '{model_name}' on GPU {gpu_id}.")
        else:
             # Ensure the chosen model is loaded on the correct GPU
             model_key = f"{model_name}"
             if model_key not in model_manager.loaded_models or model_manager.loaded_models[model_key].get("gpu_id") != gpu_id:
                 logger.warning(f"Model '{model_name}' not loaded on GPU {gpu_id}. Attempting to load.")
                 try:
                     await model_manager.load_model(model_name, gpu_id=gpu_id)
                 except Exception as load_err:
                      logger.error(f"❌ Failed to load model '{model_name}' on GPU {gpu_id}: {load_err}")
                      return []



        # (Using a similar prompt structure as your existing model_based_memory_creation)
        extraction_prompt = f"""
MEMORY EXTRACTION FROM SNIPPET TASK:

Analyze the following conversation snippet and extract key pieces of factual information or context that should be remembered for future interactions. Focus on details that provide lasting value. Avoid generic statements or conversational filler.

EXTRACTION GUIDELINES:
1. Focus on extracting reusable facts, concepts, or key context from the snippet relevant for future recall.
2. Formulate memories as clear, concise, standalone statements.
3. Assign an appropriate category (e.g., 'facts', 'preferences', 'project_details', 'technical_info', 'personal_info', 'concepts').
4. Assign an importance score (0.1-1.0) based on perceived future utility.
5. Prioritize clarity, specificity, and relevance to the provided snippet.
6. AVOID: Meta-instructions (reminders), conversational filler, generic observations, vague statements.
7. AVOID: Inferring personal user attributes or preferences; only capture them if explicitly stated as a fact within the snippet.
8. Output ONLY a valid JSON array of memory objects. Each object must have "content", "category", and "importance" keys. Output an empty array [] if no relevant memories are found.

CONVERSATION SNIPPET:
\"\"\"
{conversation_snippet}
\"\"\"

EXTRACTED MEMORIES (JSON array):
"""

        # Call the model using the model_manager's generate method
        # Ensure generate can handle being called with a specific gpu_id if needed,
        # or relies on the model being pre-loaded onto the correct GPU.
        # Using a low temperature for more deterministic extraction.
        result = model_manager.generate({
            "model_name": model_name,
            "prompt": extraction_prompt,
            "max_tokens": 512, # Adjust as needed, depends on expected memory detail
            "temperature": CONFIG.get("memory_extraction_temperature", 0.2), # Use configured or low temp
            "repetition_penalty": 1.1, # 🧠 Added for Cogito thinking mode
            "gpu_id": gpu_id # Pass GPU ID if generate method supports it
            # Add other parameters like top_p, top_k if your generate method uses them
        })

        # Process the model's output text to get the JSON array
        extracted_memories_raw = []
        if isinstance(result, dict) and 'text' in result:
            model_output_text = result['text']
            try:
                # Attempt to find and parse JSON array within the output
                json_start = model_output_text.find('[')
                json_end = model_output_text.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = model_output_text[json_start:json_end]
                    extracted_memories_raw = json.loads(json_str)
                    if not isinstance(extracted_memories_raw, list):
                        extracted_memories_raw = [] # Ensure it's a list
                else:
                    logger.warning("No JSON array found in model output for snippet extraction.")

            except json.JSONDecodeError as json_e:
                logger.error(f"JSON parsing error during snippet extraction: {json_e}")
                logger.debug(f"Model output was: {model_output_text}")
                extracted_memories_raw = []
        else:
             logger.warning(f"Unexpected result format from model_manager.generate: {type(result)}")
             extracted_memories_raw = []


        # Validate and format the extracted memories
        valid_memories = []
        if isinstance(extracted_memories_raw, list):
            for memory_data in extracted_memories_raw:
                if isinstance(memory_data, dict) and 'content' in memory_data:
                    content = memory_data['content'].strip()
                    if is_valid_memory_content(content): # Use your existing validation
                        valid_memory = {
                            "content": content,
                            "category": memory_data.get('category', 'contextual'), # Default category
                            "importance": float(memory_data.get('importance', 0.6)), # Default importance
                            "type": "contextual_auto", # Indicate it came from this process
                            "created": datetime.datetime.now().isoformat(),
                            "accessed": 0
                        }
                        # Add optional details if provided by model
                        if 'details' in memory_data:
                             valid_memory['details'] = memory_data['details']

                        valid_memories.append(valid_memory)
                    else:
                        logger.debug(f"Skipping invalid memory content from snippet: {content[:50]}...")
                else:
                    logger.warning(f"Skipping invalid memory data format: {memory_data}")

        logger.info(f"✅ Extracted {len(valid_memories)} valid memories from snippet using GPU {gpu_id}.")
        return valid_memories

    except Exception as e:
        logger.error(f"❌ Error in model-based memory creation from snippet: {e}", exc_info=True)
        return [] # Return empty list on error

async def store_memories(memories, user_id: Optional[str] = None):
    """Store newly created memories with absolutely foolproof duplicate checking"""
    if not memories or len(memories) == 0:
        logger.info("No new memories to store")
        return False

    try:
        logger.info(f"Processing {len(memories)} potential memories for storage")
        
        # Load existing memories
        existing_memories = get_memory_store(user_id=user_id)
        logger.info(f"Loaded {len(existing_memories)} existing memories from store")
        
        # Create a set of ALL existing memory contents for exact duplicate checking
        existing_contents = set()
        for memory in existing_memories:
            if 'content' in memory and memory['content']:
                normalized = memory['content'].strip().lower()
                existing_contents.add(normalized)
        
        # Add timestamps if missing
        for memory in memories:
            if 'created' not in memory:
                memory['created'] = datetime.datetime.now().isoformat()
            if 'accessed' not in memory:
                memory['accessed'] = 0
        
        # Process and enhance new memories
        enhanced_memories = []
        for memory in memories:
            if not memory or 'content' not in memory:
                logger.warning(f"Skipping memory without content")
                continue
            
            content = memory.get('content', '').strip()
            if not content:
                continue
            
            if content and len(content) > 0 and not content[0].isupper():
                content = content[0].upper() + content[1:]
            if content and not content.endswith(('.', '!', '?', ':', ';')):
                content += '.'
            memory['content'] = content

            if not is_valid_memory_content(content):
                logger.warning(f"Skipping invalid memory content: {content[:50]}...")
                continue
            
            if user_id and 'user' not in memory:
                # Add user ID to memory if not already present
                memory['user'] = user_id
                logger.debug(f"Added user tag '{user_id}' to memory")
            
            enhanced_memories.append(memory)
        
        # Add new memories with enhanced duplicate checking
        memories_added = 0
        for memory in enhanced_memories:
            content = memory.get('content', '').strip()
            normalized_content = content.lower()
            if normalized_content in existing_contents:
                logger.info(f"Skipping exact duplicate: {content[:50]}...")
                continue

            is_semantic_duplicate = False
            recent_memories = sorted(
                existing_memories,
                key=lambda x: x.get('created', '2000-01-01'),
                reverse=True
            )[:CONFIG["memory_deduplication_window"]]

            for existing_memory in recent_memories:
                if does_it_basically_mean_the_same_thing(
                    existing_memory.get('content', ''),
                    content,
                    threshold=CONFIG["similarity_threshold"]
                ):
                    is_semantic_duplicate = True
                    logger.info(f"Skipping semantic duplicate: {content[:50]}...")
                    logger.info(f"Similar to existing: {existing_memory.get('content')[:50]}...")
                    break

            if is_semantic_duplicate:
                continue

            existing_memories.append(memory)
            existing_contents.add(normalized_content)
            memories_added += 1
            logger.info(f"✅ Added new memory: {content[:50]}...")

        if memories_added > 0:
            if save_memory_store(existing_memories, user_id=user_id):
                # Save the updated memory store
                logger.info(f"Added {memories_added} new memories for user '{user_id}'. Store now has {len(existing_memories)} memories.")
                return True
            else:
                logger.warning("❌ Failed to save updated memory store.")
                return False
        else:
            logger.info("No new unique memories to add")
            return True  # ✅ This is now correct

    except Exception as e:
        logger.error(f"Error storing memories: {e}")
        traceback.print_exc()
        return False


def curate_memory_store(user_id: Optional[str] = None, similarity_threshold=0.7):
    """
    Curates a user's memory store by removing invalid entries and semantic duplicates,
    then enhancing and saving the result.
    """
    logger.info(f"🧹 Starting memory curation for user: {user_id}")

    try:
        memories = get_memory_store(user_id=user_id)
        if not memories:
            logger.warning(f"❌ No memories found for user '{user_id}'")
            return {"status": "error", "reason": "no_memories"}

        original_count = len(memories)
        logger.info(f"📊 Found {original_count} memories to process")

        valid_memories = []
        invalid_count = 0
        memory_contents = set()

        for memory in memories:
            content = memory.get('content', '').strip()

            if not content or len(content) < CONFIG["memory_content_min_length"]:
                invalid_count += 1
                continue

            if not is_valid_memory_content(content):
                invalid_count += 1
                continue

            normalized_content = content.lower()
            if normalized_content in memory_contents:
                logger.info(f"Skipping exact duplicate: {content[:50]}...")
                continue

            memory_contents.add(normalized_content)
            memory['content'] = content
            valid_memories.append(memory)

        logger.info(f"🧹 Removed {invalid_count} invalid memories, {len(valid_memories)} remaining")

        if len(valid_memories) < 2:
            save_memory_store(valid_memories, user_id=user_id)
            logger.info(f"✅ Saved {len(valid_memories)} valid memories (not enough for deduplication)")
            return {
                "status": "partial_success",
                "original_count": original_count,
                "invalid_removed": invalid_count,
                "remaining_count": len(valid_memories)
            }

        model = similarity_model

        contents = [mem["content"] for mem in valid_memories]

        # Process embeddings in batches to avoid OOM errors
        batch_size = 64
        num_batches = (len(contents) + batch_size - 1) // batch_size
        all_embeddings = None

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(contents))

            batch_contents = contents[start_idx:end_idx]
            batch_embeddings = model.encode(batch_contents, convert_to_tensor=True)

            if all_embeddings is None:
                all_embeddings = batch_embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, batch_embeddings), dim=0)

            logger.info(f"Processed batch {batch_idx+1}/{num_batches}")

        sim_matrix = util.pytorch_cos_sim(all_embeddings, all_embeddings)

        to_delete = set()
        for i in range(len(valid_memories)):
            if i in to_delete:
                continue

            for j in range(i + 1, len(valid_memories)):
                if j in to_delete:
                    continue

                similarity = sim_matrix[i][j].item()
                if similarity > similarity_threshold:
                    mem_i, mem_j = valid_memories[i], valid_memories[j]
                    score_i, score_j = 0, 0

                    score_i += min(len(mem_i.get('content', '')) / 50, 3)
                    score_j += min(len(mem_j.get('content', '')) / 50, 3)

                    score_i += float(mem_i.get('importance', 0.5)) * 5
                    score_j += float(mem_j.get('importance', 0.5)) * 5

                    score_i += min(mem_i.get('accessed', 0), 5)
                    score_j += min(mem_j.get('accessed', 0), 5)

                    if mem_i.get('created', '') > mem_j.get('created', ''):
                        score_i += 1
                    else:
                        score_j += 1

                    if mem_i.get('content', '').endswith(('.', '!', '?')):
                        score_i += 2
                    if mem_j.get('content', '').endswith(('.', '!', '?')):
                        score_j += 2

                    if mem_i.get('details'):
                        score_i += 1
                    if mem_j.get('details'):
                        score_j += 1

                    if score_i >= score_j:
                        to_delete.add(j)
                    else:
                        to_delete.add(i)
                        break

        curated_memories = [mem for idx, mem in enumerate(valid_memories) if idx not in to_delete]

        for memory in curated_memories:
            content = memory.get('content', '').strip()
            if content and len(content) > 0:
                content = content[0].upper() + content[1:]
            if content and not content.endswith(('.', '!', '?')):
                content += '.'
            memory['content'] = content

            if 'created' not in memory:
                memory['created'] = datetime.datetime.now().isoformat()
            if 'accessed' not in memory:
                memory['accessed'] = 0

        save_memory_store(curated_memories, user_id=user_id)

        logger.info(f"✅ Memory curation complete for user '{user_id}': Removed {len(to_delete)} duplicates, {invalid_count} invalid memories. {len(curated_memories)} memories remain.")

        return {
            "status": "success",
            "original_count": original_count,
            "invalid_removed": invalid_count,
            "duplicates_removed": len(to_delete),
            "remaining_count": len(curated_memories)
        }

    except Exception as e:
        logger.error(f"❌ Error during memory curation for user '{user_id}': {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}
    

# Corrected function in src/memory_intelligence.py
def purge_memory_store(user_id: Optional[str] = None): # <-- Changed parameter to user_id
    """
    Completely purge all memories from a specific user's memory store.
    Requires user_id.
    """
    logger = logging.getLogger(__name__) # Use logger
    if not user_id:
        logger.error("❌ purge_memory_store called without a user_id.")
        return {"status": "error", "reason": "user_id_missing"}

    try:
        memory_file_path = get_user_memory_path(user_id=user_id) # <-- Get path using user_id
        logger.warning(f"🧹 Attempting to PURGE memory store for user '{user_id}' at: {memory_file_path}")

        # Check if the file exists before trying to remove/overwrite
        if os.path.exists(memory_file_path):
            # Overwrite with an empty list [] using atomic write principles
            # (Reusing the save_memory_store logic but with an empty list)
            save_success = save_memory_store([], user_id=user_id) # Use save_memory_store for atomic write

            if save_success:
                logger.info(f"✅ Memory store purged successfully for user '{user_id}' at {memory_file_path}")
                return {"status": "success", "message": "Memory store completely purged for this user."}
            else:
                logger.error(f"❌ Failed to save empty store during purge for user '{user_id}' at {memory_file_path}.")
                return {"status": "error", "reason": "failed_to_save_empty_store"}
        else:
            logger.info(f"⚠️ Memory file did not exist for user '{user_id}' at {memory_file_path}. No purge needed, ensuring directory exists.")
            # Ensure the directory exists even if the file didn't
            os.makedirs(os.path.dirname(memory_file_path), exist_ok=True)
            # Optionally, create an empty file now if desired, using save_memory_store
            save_memory_store([], user_id=user_id)
            return {"status": "success", "message": "Memory store file did not exist. Created empty store."}

    except ValueError as ve: # Catch error from get_user_memory_path if user_id is invalid
        logger.error(f"❌ Value Error determining path for purge (user_id: '{user_id}'): {ve}")
        return {"status": "error", "reason": f"Invalid user_id for purge: {str(ve)}"}
    except Exception as e:
        logger.error(f"❌ Unexpected error purging memory store for user '{user_id}': {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

async def process_incoming_message_with_userprofile(model_manager, message, userProfile=None, history=None, user_id: Optional[str] = None, gpu_id=None, single_gpu_mode=False):
    if gpu_id is None:
        gpu_id = 1 if single_gpu_mode else 0
    """
    Process memory retrieval using GPU 1 but with memories from userProfile
    instead of the backend memory store. Now supports user-specific profiles.

    Args:
        model_manager: Model manager instance
        message: User message text
        userProfile: User profile containing memories from localStorage
        history: Optional conversation history
        user_id: Optional unique identifier for the user
        gpu_id: GPU ID for memory operations (typically GPU 1)

    Returns:
        Dict with relevant memories
    """
    try:
        logger.info(f"🧠 [BRIDGE] Using GPU {gpu_id} to process memories from userProfile (user_id={user_id})")

        # Extract memories from userProfile if available
        profile_memories = []
        if userProfile and isinstance(userProfile, dict) and 'memories' in userProfile:
            profile_memories = userProfile.get('memories', [])
            logger.info(f"Found {len(profile_memories)} memories in userProfile for user '{user_id}'")
        else:
            logger.warning(f"No memories found in userProfile for user '{user_id}'")

        # If we have no memories, return empty results
        if not profile_memories:
            return {
                "status": "success",
                "memories": [],
                "formatted_memories": "",
                "memory_count": 0
            }

        # Use the existing analysis logic but with profile_memories instead of from disk
        prompt_embedding = similarity_model.encode(message, convert_to_tensor=True)

        # Score memories based on relevance to prompt
        scored_memories = []
        for memory in profile_memories:
            memory_content = memory.get('content', '')


            if not memory_content:
                continue

            memory_embedding = similarity_model.encode(memory_content, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(prompt_embedding, memory_embedding)
            semantic_score = cosine_scores[0][0].item()

            score = semantic_score

            scored_memories.append((memory, score))
            logger.info(f"🧠 [APPEND DEBUG] Added to scored_memories (total now: {len(scored_memories)})")

        # Sort and select top relevant memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_count = min(CONFIG["min_memories"], len(scored_memories))
        relevant_memories = [mem[0] for mem in scored_memories[:top_count]]

        # Format for context injection
        formatted_memories = format_memories_for_context(relevant_memories)

        logger.info(f"✅ Retrieved {len(relevant_memories)} relevant memories for user '{user_id}'")
        return {
            "status": "success",
            "memories": relevant_memories,
            "formatted_memories": formatted_memories,
            "memory_count": len(relevant_memories),
            "retrieval_source": "userProfile_gpu"
        }

    except Exception as e:
        logger.error(f"❌ Error processing userProfile memories for user '{user_id}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "memories": [],
            "formatted_memories": "",
            "memory_count": 0
        }


# Let's also fix the memory formatting function to ensure it produces usable context
def format_memories_for_context(memories, max_memories=5, max_chars=1500):
    """
    Format memories for injection into context with better readability and organization.
    Increased max_chars to allow more comprehensive memory context.
    """
    if not memories:
        return ""
        
    # Sort memories by importance and accessed count
    sorted_memories = sorted(
        memories,
        key=lambda x: (float(x.get('importance', 0.5)) * 0.7 + min(x.get('accessed', 0) * 0.1, 0.3)),
        reverse=True
    )
    
    # Take top memories
    top_memories = sorted_memories[:max_memories]
    
    # Group memories by category
    memory_by_category = {}
    for memory in top_memories:
        category = memory.get('category', 'general')
        if category not in memory_by_category:
            memory_by_category[category] = []
        memory_by_category[category].append(memory)
    
    # Format the output with categories
    formatted = "RELEVANT USER INFORMATION:\n"
    
    # Count total characters to stay within limit
    total_chars = len(formatted)
    
    for category, mem_list in sorted(memory_by_category.items()):
        # Format category header with better readability
        category_display = category.replace('_', ' ').title()
        category_header = f"\n{category_display}:\n"
        total_chars += len(category_header)
        
        if total_chars > max_chars:
            break
            
        formatted += category_header
        
        # Add memories for this category
        for memory in mem_list:
            content = memory.get('content', '')
            importance = memory.get('importance', 0.5)
            
            # Format importance as stars
            stars = "★" * max(1, min(5, round(importance * 5)))
            
            memory_line = f"• {content}\n"
            total_chars += len(memory_line)
            
            if total_chars > max_chars:
                # Trim the string if we're exceeding the limit
                formatted = formatted[:max_chars - 3] + "..."
                break
                
            formatted += memory_line
    
    return formatted

# === Memory Router Integration ===

async def process_incoming_message(model_manager, message, history=None, gpu_id=None, single_gpu_mode=False):
    if gpu_id is None:
        gpu_id = 0 if single_gpu_mode else 1
    """
    Process an incoming user message for memory operations.
    This is a helper function for the memory_router to decide what to do with a message.
    
    Args:
        model_manager: Model manager instance
        message: User message text
        history: Optional conversation history
        gpu_id: GPU ID for memory operations (typically GPU 1)
        
    Returns:
        Dict with memory retrieval results
    """
    try:
        # Step 1: Analyze for relevant memories
        relevant_memories = await analyze_for_relevant_memories(model_manager, message, gpu_id=gpu_id)
        
        # Step 2: Format memories for context injection
        formatted_memories = format_memories_for_context(relevant_memories)
        
        # Return results
        return {
            "status": "success",
            "memories": relevant_memories,
            "formatted_memories": formatted_memories,
            "memory_count": len(relevant_memories)
        }
    except Exception as e:
        logger.error(f"Error processing incoming message for memory: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "memories": [],
            "formatted_memories": "",
            "memory_count": 0
        }

async def process_completed_exchange(model_manager, user_message, ai_response, gpu_id=None, single_gpu_mode=False, user_name=None, prefer_multiple_memories=True, max_memory_length=500, recommended_memory_count=5, split_long_memories=True, conversation_history=None, userProfile=None, systemTime=None, requestType=None):
    if gpu_id is None:
        gpu_id = 0 if single_gpu_mode else 1
    """
    Process a completed exchange between user and AI for memory creation.
    This is a helper function for memory_router to create memories after a response.
    
    Args:
        model_manager: Model manager instance
        user_message: User message text
        ai_response: AI response text
        gpu_id: GPU ID for memory operations (typically GPU 1)
        user_name: Optional user name for personalization
        
    Returns:
        Dict with memory creation results
    """
    try:
        # Step 1: Analyze and create memories
        new_memories = await analyze_for_memory_creation(model_manager, user_message, ai_response, gpu_id=gpu_id)
        
        # Step 2: Store memories
        if new_memories and len(new_memories) > 0:
            storage_success = await store_memories(new_memories)
        else:
            storage_success = False
        
        # Return results
        return {
            "status": "success" if storage_success else "no_memories",
            "memories_created": len(new_memories),
            "storage_success": storage_success
        }
    except Exception as e:
        logger.error(f"Error processing completed exchange for memory: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "memories_created": 0,
            "storage_success": False
        }