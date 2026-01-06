# character_intelligence.py - Character detection and auto-generation system

import logging
import json
import datetime
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import util
import torch

# Use the same embedding model as memory system
logger = logging.getLogger("character_intelligence")
try:
    from .memory_intelligence import similarity_model
    logger.info("✅ Using shared similarity_model from memory_intelligence")
except ImportError:
    logger.error("❌ Could not import similarity_model from memory_intelligence")
    similarity_model = None

# Character concept embeddings - these define what we're looking for
CHARACTER_CONCEPTS = [
    "character personality traits temperament behavior",
    "physical appearance looks description features",
    "character background story history origins",
    "speaking style dialogue voice manner",
    "character goals motivations desires dreams",
    "character relationships family friends",
    "character occupation job role profession",
    "character setting environment world location"
]

# Pre-compute concept embeddings at module load
concept_embeddings = None
if similarity_model:
    try:
        concept_embeddings = similarity_model.encode(CHARACTER_CONCEPTS, convert_to_tensor=True)
        logger.info(f"✅ Pre-computed {len(CHARACTER_CONCEPTS)} character concept embeddings")
    except Exception as e:
        logger.error(f"❌ Failed to compute concept embeddings: {e}")

# def analyze_character_readiness(messages: List[Dict[str, Any]], lookback_count: int = 25) -> Dict[str, Any]:
#     """
#     Analyze recent conversation messages for character information.
#     
#     Args:
#         messages: List of message objects with 'content' and 'role'
#         lookback_count: How many recent messages to analyze
#         
#     Returns:
#         Dict with readiness score, detected elements, and character suggestions
#     """
#     if not similarity_model or concept_embeddings is None:
#         logger.error("❌ Similarity model not available for character analysis")
#         return {"status": "error", "error": "Embedding model not available"}
#     
#     if not messages:
#         return {"status": "success", "readiness_score": 0, "detected_elements": []}
#     
#     try:
#         # Get recent messages, excluding system messages
#         recent_messages = []
#         for msg in reversed(messages[-lookback_count:]):
#             if msg.get('role') not in ['system'] and msg.get('content'):
#                 recent_messages.append(msg['content'])
#         
#         if not recent_messages:
#             return {"status": "success", "readiness_score": 0, "detected_elements": []}
#         
#         # Combine messages into analysis text
#         conversation_text = " ".join(recent_messages)
#         logger.info(f"🔍 Analyzing {len(recent_messages)} messages ({len(conversation_text)} chars)")
#         
#         # Split into chunks for better analysis
#         chunks = split_into_chunks(conversation_text, max_length=500)
#         
#         detected_elements = []
#         concept_scores = [0.0] * len(CHARACTER_CONCEPTS)
#         
#         # Analyze each chunk against character concepts
#         for chunk in chunks:
#             chunk_embedding = similarity_model.encode(chunk, convert_to_tensor=True)
#             similarities = util.pytorch_cos_sim(chunk_embedding, concept_embeddings)[0]
#             
#             for i, (concept, score) in enumerate(zip(CHARACTER_CONCEPTS, similarities)):
#                 concept_scores[i] = max(concept_scores[i], score.item())
#                 
#                 # If similarity is high enough, record this as detected
#                 if score.item() > 0.3:  # Threshold for detection
#                     detected_elements.append({
#                         "concept": concept.split()[1],  # Get the main concept word
#                         "score": score.item(),
#                         "text_sample": chunk[:100] + "..." if len(chunk) > 100 else chunk
#                     })
#         
#         # Calculate overall readiness score
#         readiness_score = calculate_readiness_score(concept_scores, detected_elements)
#         
#         # Detect potential character name(s)
#         suggested_names = extract_potential_character_names(conversation_text)
#         
#         result = {
#             "status": "success",
#             "readiness_score": readiness_score,
#             "detected_elements": detected_elements,
#             "concept_scores": dict(zip(CHARACTER_CONCEPTS, concept_scores)),
#             "suggested_names": suggested_names,
#             "analysis_summary": generate_analysis_summary(concept_scores, detected_elements)
#         }
#         
#         logger.info(f"🎯 Character readiness: {readiness_score:.1f}% ({len(detected_elements)} elements)")
#         return result
#         
#     except Exception as e:
#         logger.error(f"❌ Error in character readiness analysis: {e}", exc_info=True)
#         return {"status": "error", "error": str(e)}


def split_into_chunks(text: str, max_length: int = 500) -> List[str]:
    """Split text into chunks for better embedding analysis."""
    if len(text) <= max_length:
        return [text]
    
    # Try to split on sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def calculate_readiness_score(concept_scores: List[float], detected_elements: List[Dict]) -> float:
    """Calculate overall character readiness score (0-100)."""
    if not concept_scores:
        return 0.0
    
    # Base score from concept detection
    avg_score = sum(concept_scores) / len(concept_scores)
    base_score = min(avg_score * 100, 80)  # Cap at 80% from embeddings alone
    
    # Bonus for variety of detected elements
    unique_concepts = len(set(elem["concept"] for elem in detected_elements))
    variety_bonus = min(unique_concepts * 5, 20)  # Up to 20% bonus
    
    # Bonus for high-confidence detections
    high_confidence = sum(1 for elem in detected_elements if elem["score"] > 0.5)
    confidence_bonus = min(high_confidence * 3, 15)  # Up to 15% bonus
    
    total_score = min(base_score + variety_bonus + confidence_bonus, 100)
    return round(total_score, 1)

def extract_potential_character_names(text: str) -> List[str]:
    """Extract potential character names from conversation."""
    # Look for patterns like "character named X" or repeated proper nouns
    name_patterns = [
        r"character named (\w+)",
        r"character called (\w+)",
        r"named (\w+)",
        r"called (\w+)"
    ]
    
    names = []
    for pattern in name_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            if name and len(name) > 2 and name not in names:
                names.append(name)
    
    return names[:3]  # Return up to 3 potential names

def generate_analysis_summary(concept_scores: List[float], detected_elements: List[Dict]) -> str:
    """Generate a human-readable summary of the analysis."""
    if not detected_elements:
        return "No character information detected in recent conversation."
    
    detected_concepts = list(set(elem["concept"] for elem in detected_elements))
    
    if len(detected_concepts) >= 4:
        return f"Rich character information detected: {', '.join(detected_concepts[:4])}{'...' if len(detected_concepts) > 4 else ''}"
    elif len(detected_concepts) >= 2:
        return f"Some character details found: {', '.join(detected_concepts)}"
    else:
        return f"Basic character information detected: {detected_concepts[0] if detected_concepts else 'general'}"

async def generate_character_json(
    model_manager, 
    messages: List[Dict[str, Any]], 
    character_analysis: Dict[str, Any],
    model_name: str = None,
    gpu_id: int = None,
    single_gpu_mode: bool = False,
    use_api: bool = False,
    api_endpoint: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Use an LLM to generate character JSON based on conversation analysis.
    Supports both local models and external APIs.
    """
    logger.info(f"🎨 Generating character JSON (use_api={use_api})")
    
    try:
        # Prepare conversation context
        recent_messages = messages[-15:] if len(messages) > 15 else messages
        conversation_context = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in recent_messages 
            if msg.get('content')
        ])
        
        # Build generation prompt
        generation_prompt = build_character_generation_prompt(
            conversation_context, 
            character_analysis
        )
        
        if use_api and api_endpoint:
            # Use external API (OpenAI-compatible)
            response = await generate_with_api(generation_prompt, api_endpoint)
        else:
            # Generate character JSON using local inference module
            from . import inference
            response = await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=generation_prompt,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                stop_sequences=["</character>", "---"],
                gpu_id=gpu_id
            )
        
        # Extract and parse JSON from response
        character_json = extract_json_from_response(response)
        
        if character_json:
            logger.info(f"✅ Generated character JSON for: {character_json.get('name', 'Unnamed')}")
            return {"status": "success", "character_json": character_json}
        else:
            return {"status": "error", "error": "Could not extract valid JSON from model response"}
            
    except Exception as e:
        logger.error(f"❌ Error generating character JSON: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


async def generate_with_api(prompt: str, api_endpoint: Dict[str, Any]) -> str:
    """Generate text using an external OpenAI-compatible API."""
    import httpx
    
    url = api_endpoint.get("url", "").rstrip("/")
    # Handle various URL formats
    if url.endswith("/chat/completions"):
        pass  # Already complete
    elif url.endswith("/v1"):
        url = f"{url}/chat/completions"
    elif "/v1" in url and not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    else:
        url = f"{url}/v1/chat/completions"
    
    api_key = api_endpoint.get("api_key", "")
    model = api_endpoint.get("model", "")
    
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    logger.info(f"🌐 Calling API: {url} with model {model}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # Extract content from OpenAI-style response
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError("Invalid API response format")

def build_character_generation_prompt(conversation_context: str, analysis: Dict[str, Any]) -> str:
    """
    Builds a robust, creative-fill prompt for generating a character JSON from a conversation.

    This prompt instructs the model to act as a profiler but gives it explicit
    permission to creatively improvise fields if specific details are missing,
    based on the overall character pattern.

    Args:
        conversation_context: The string containing the full conversation history.
        analysis: A dictionary containing pre-computed analysis, like suggested names
                  and detected character elements.

    Returns:
        A string containing the complete prompt for the LLM.
    """
    
    # --- 1. Extract and Format Analysis Hints ---
    suggested_names = analysis.get("suggested_names", [])
    name_hint = f"A likely name for the character is: {suggested_names[0]}." if suggested_names else "No specific name was suggested; determine the name from the conversation."

    detected_elements_list = [elem.get('concept', 'unknown') for elem in analysis.get('detected_elements', [])]
    elements_hint = f"Key concepts detected in the conversation are: {', '.join(detected_elements_list)}." if detected_elements_list else "No specific elements were pre-analyzed."

    # --- 2. Construct the Core Prompt with New Creative Instructions ---
    prompt = f"""System:
You are a highly advanced AI with a specialization in creative character profiling. Your purpose is to analyze a provided conversation and synthesize the information into a complete and compelling character profile in JSON format.

**YOUR TASK AND RULES:**
1.  Read the entire "Provided Conversation History" and use the "Analysis Hints" to understand the core of the character.
2.  Populate the fields of the JSON object described below. Your primary source of information is the conversation.
3.  **CREATIVE IMPROVISATION RULE: If specific details for a field (like 'scenario' or 'first_message') are missing from the conversation, you MUST use your understanding of the character's established persona to creatively improvise and fill in that field. The improvised content MUST be consistent with the character's overall pattern and personality.**
4.  Do not leave fields blank unless it is absolutely impossible to create consistent content.
5.  Your final output MUST be ONLY the JSON object and nothing else. Do not include any commentary before or after the JSON block.

**REQUIRED JSON OUTPUT STRUCTURE:**
* `"name"`: (String) The character's full name.
* `"description"`: (String) A brief, one-sentence summary of the character.
* `"model_instructions"`: (String) Detailed instructions for an AI on how to accurately roleplay this character, including their speaking style, motivations, and key behaviors.
* `"scenario"`: (String) The setting and context where interactions with this character typically take place.
* `"first_message"`: (String) A sample opening greeting from the character that captures their voice and personality.
* `"example_dialogue"`: (Array of Objects) A list containing one or more user/character exchanges. Each object must have a "role" (`"user"` or `"character"`) and "content" (the message).
* `"loreEntries"`: (Array of Objects) A list of relevant background facts. Each object must have "content" (the fact) and "keywords" (an array of strings that trigger this lore).

---
**BEGIN ANALYSIS**
---

**Provided Conversation History:**
{conversation_context}

**Analysis Hints:**
- {name_hint}
- {elements_hint}

**Generated Character JSON:**
```json
"""

    return prompt


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract and validate JSON from model response."""
    try:
        # Log the raw response for debugging
        logger.debug(f"Raw model response: {response}")
        # Try to find JSON block
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            logger.warning("No valid JSON found in model response")
            return None
        
        json_str = response[json_start:json_end]
        logger.debug(f"Extracted JSON string: {json_str}")
        character_data = json.loads(json_str)
        logger.debug(f"Parsed JSON data: {character_data}")
        # Validate required fields
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in character_data:
                logger.warning(f"Missing required field: {field}")
                return None
        
        # Set defaults for optional fields
        defaults = {
            "model_instructions": "",
            "scenario": "",
            "first_message": f"Hello! I'm {character_data.get('name', 'a character')}.",
            "example_dialogue": [],
            "loreEntries": []
        }
        
        for key, default_value in defaults.items():
            if key not in character_data:
                character_data[key] = default_value
        
        return character_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return None