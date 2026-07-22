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

MAX_CHARACTER_GEN_ATTEMPTS = 3
CHARACTER_API_MAX_ATTEMPTS = 3
CHARACTER_API_MAX_TOKENS = 16384
_CHARACTER_API_TIMEOUT_MSG = (
    "The character API stopped responding before the model finished (read timeout). "
    "Try again, use a faster model, or shorten the conversation context."
)


class CharacterApiError(Exception):
    """Remote API failure for character generation (after retries)."""


def _character_api_error_is_transient(exc: BaseException) -> bool:
    import httpx
    from fastapi import HTTPException

    from .openai_compat import _openai_compat_is_transient_upstream_for_retry

    if isinstance(exc, httpx.RequestError):
        return _openai_compat_is_transient_upstream_for_retry(
            exc, include_read_write_timeout=True
        )
    if isinstance(exc, HTTPException) and exc.status_code == 502:
        detail = str(exc.detail or "").lower()
        return any(
            marker in detail
            for marker in (
                "readtimeout",
                "writetimeout",
                "remoteprotocolerror",
                "server disconnected",
                "cannot connect",
            )
        )
    return False


def _format_character_api_error(
    exc: BaseException,
    *,
    endpoint_name: str = "API",
) -> str:
    import httpx

    if isinstance(exc, httpx.ReadTimeout):
        return _CHARACTER_API_TIMEOUT_MSG
    if isinstance(exc, CharacterApiError):
        return str(exc)
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        detail = str(exc.detail or "").strip()
        if exc.status_code == 502 and _character_api_error_is_transient(exc):
            tech = f" — {detail}" if detail else ""
            return f"{_CHARACTER_API_TIMEOUT_MSG}{tech}"
        return detail or f"API error from {endpoint_name} (HTTP {exc.status_code})"
    return str(exc) or f"API error from {endpoint_name}"


def _resolve_character_endpoint_model_id(
    model_name: Optional[str],
    api_endpoint: Optional[Dict[str, Any]],
) -> str:
    if model_name and str(model_name).startswith("endpoint-"):
        return str(model_name)
    if api_endpoint:
        ep_id = api_endpoint.get("id") or ""
        if str(ep_id).startswith("endpoint-"):
            return str(ep_id)
    return ""


def _build_character_api_request_data(
    prompt: str,
    *,
    endpoint_model_id: str,
    configured_model: str = "",
) -> Dict[str, Any]:
    model_field = endpoint_model_id or configured_model or "gpt-3.5-turbo"
    return {
        "model": model_field,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": CHARACTER_API_MAX_TOKENS,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": True,
        "_skip_openai_message_pruning": True,
        "_max_stream_attempts": CHARACTER_API_MAX_ATTEMPTS,
    }


async def _generate_character_llm_text(
    *,
    prompt: str,
    model_manager,
    model_name: str,
    gpu_id: int,
    use_api: bool,
    api_endpoint: Optional[Dict[str, Any]],
    frontend_round_robin_enabled: Optional[bool] = None,
    force_resolved_endpoint: bool = False,
) -> str:
    if use_api and api_endpoint:
        return await generate_with_api(
            prompt,
            api_endpoint,
            model_name=model_name,
            frontend_round_robin_enabled=frontend_round_robin_enabled,
            force_resolved_endpoint=force_resolved_endpoint,
        )
    from . import inference

    return await inference.generate_text(
        model_manager=model_manager,
        model_name=model_name,
        prompt=prompt,
        max_tokens=2048,
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        stop_sequences=["</character>", "---"],
        gpu_id=gpu_id,
    )


async def generate_character_json(
    model_manager, 
    messages: List[Dict[str, Any]], 
    character_analysis: Dict[str, Any],
    model_name: str = None,
    gpu_id: int = None,
    single_gpu_mode: bool = False,
    use_api: bool = False,
    api_endpoint: Dict[str, Any] = None,
    frontend_round_robin_enabled: Optional[bool] = None,
    force_resolved_endpoint: bool = False,
    conversation_id: str = "",
) -> Dict[str, Any]:
    """
    Use an LLM to generate character JSON based on conversation analysis.
    Supports both local models and external APIs. Retries with JSON repair on failure.
    """
    from . import character_json_parse as cjp

    logger.info(f"🎨 Generating character JSON (use_api={use_api})")

    try:
        recent_messages = messages[-15:] if len(messages) > 15 else messages
        conversation_context = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in recent_messages
            if msg.get('content')
        ])

        generation_prompt = build_character_generation_prompt(
            conversation_context,
            character_analysis,
        )

        last_raw = ""
        last_error: Optional[str] = None
        last_partial: Optional[Dict[str, Any]] = None
        backup_paths: List[str] = []

        for attempt in range(MAX_CHARACTER_GEN_ATTEMPTS):
            prompt = (
                generation_prompt
                if attempt == 0
                else cjp.build_character_repair_prompt(last_raw)
            )

            response = await _generate_character_llm_text(
                prompt=prompt,
                model_manager=model_manager,
                model_name=model_name,
                gpu_id=gpu_id,
                use_api=use_api,
                api_endpoint=api_endpoint,
                frontend_round_robin_enabled=frontend_round_robin_enabled,
                force_resolved_endpoint=force_resolved_endpoint,
            )
            last_raw = response or ""
            backup_paths.append(
                cjp.save_character_generation_backup(
                    last_raw,
                    attempt=attempt,
                    conversation_id=conversation_id,
                )
            )

            character_json, partial, salvaged, parse_error = cjp.parse_character_json(
                last_raw
            )
            last_error = parse_error
            if partial:
                last_partial = partial

            if character_json:
                logger.info(
                    "✅ Generated character JSON for: %s (attempt %d)",
                    character_json.get("name", "Unnamed"),
                    attempt + 1,
                )
                return {
                    "status": "success",
                    "character_json": character_json,
                    "attempts": attempt + 1,
                    "backup_paths": backup_paths,
                }

            if attempt < MAX_CHARACTER_GEN_ATTEMPTS - 1:
                logger.warning(
                    "Character JSON parse failed (attempt %d): %s — retrying",
                    attempt + 1,
                    parse_error,
                )

        excerpt = (last_raw or "")[:2000]
        if last_partial and cjp.character_json_is_usable(last_partial, partial_ok=True):
            logger.warning(
                "Returning salvaged partial character after %d attempts",
                MAX_CHARACTER_GEN_ATTEMPTS,
            )
            return {
                "status": "partial",
                "character_json": last_partial,
                "partial_character_json": last_partial,
                "salvaged": True,
                "error": last_error or "Incomplete JSON; partial fields recovered",
                "raw_response_excerpt": excerpt,
                "backup_paths": backup_paths,
                "attempts": MAX_CHARACTER_GEN_ATTEMPTS,
            }

        return {
            "status": "error",
            "error": last_error or "Could not extract valid JSON from model response",
            "raw_response_excerpt": excerpt,
            "backup_paths": backup_paths,
            "attempts": MAX_CHARACTER_GEN_ATTEMPTS,
        }

    except CharacterApiError as e:
        logger.error("❌ Character API failed after retries: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "error_type": "api_timeout",
        }
    except Exception as e:
        logger.error(f"❌ Error generating character JSON: {e}", exc_info=True)
        return {
            "status": "error",
            "error": _format_character_api_error(e),
            "error_type": "api_error" if "timeout" in str(e).lower() else "internal",
        }


async def generate_with_api(
    prompt: str,
    api_endpoint: Dict[str, Any],
    *,
    model_name: Optional[str] = None,
    request_purpose: Optional[str] = "create_character",
    frontend_round_robin_enabled: Optional[bool] = None,
    force_resolved_endpoint: bool = False,
) -> str:
    """Generate text via configured OpenAI-compatible endpoint (streaming, long read timeout)."""
    import asyncio

    import httpx
    from fastapi import HTTPException

    from .openai_compat import (
        approx_openai_messages_payload_chars,
        collect_openai_compatible_stream_text,
        prepare_endpoint_request,
    )

    endpoint_model_id = _resolve_character_endpoint_model_id(model_name, api_endpoint)
    endpoint_name = api_endpoint.get("name") or endpoint_model_id or "API"
    request_data = _build_character_api_request_data(
        prompt,
        endpoint_model_id=endpoint_model_id,
        configured_model=api_endpoint.get("model") or "",
    )

    if force_resolved_endpoint and api_endpoint:
        base_url = (api_endpoint.get("url") or "").rstrip("/")
        if not base_url:
            raise CharacterApiError("API endpoint URL is missing")
        if base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"
        endpoint_config = {
            "url": base_url,
            "api_key": api_endpoint.get("api_key") or api_endpoint.get("apiKey") or "",
            "name": endpoint_name,
            "model": api_endpoint.get("model", ""),
        }
        prepared = dict(request_data)
        if prepared.get("model", "").startswith("endpoint-"):
            configured = (api_endpoint.get("model") or "").strip()
            prepared["model"] = configured or "gpt-3.5-turbo"
    elif endpoint_model_id:
        endpoint_config, url, prepared = prepare_endpoint_request(
            endpoint_model_id,
            request_data,
            request_purpose=request_purpose or "create_character",
            frontend_round_robin_enabled=frontend_round_robin_enabled,
        )
    else:
        base_url = (api_endpoint.get("url") or "").rstrip("/")
        if not base_url:
            raise CharacterApiError("API endpoint URL is missing")
        if base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"
        endpoint_config = {
            "url": base_url,
            "api_key": api_endpoint.get("api_key", ""),
            "name": endpoint_name,
            "model": api_endpoint.get("model", ""),
        }
        prepared = dict(request_data)
        if prepared.get("model", "").startswith("endpoint-"):
            configured = (api_endpoint.get("model") or "").strip()
            prepared["model"] = configured or "gpt-3.5-turbo"

    msg_chars = approx_openai_messages_payload_chars(prepared.get("messages", []))
    log_fn = logger.warning if msg_chars >= 80_000 else logger.info
    log_fn(
        "Character API request: ~%s prompt chars, max_tokens=%s, upstream=stream_aggregate, endpoint=%s",
        msg_chars,
        prepared.get("max_tokens"),
        endpoint_name,
    )

    last_exc: Optional[BaseException] = None
    for attempt in range(CHARACTER_API_MAX_ATTEMPTS):
        try:
            text_out = await collect_openai_compatible_stream_text(
                endpoint_config, url, prepared
            )
            if not (text_out or "").strip():
                raise CharacterApiError(
                    f"{endpoint_name} returned an empty response. Try again or pick another model."
                )
            return text_out
        except CharacterApiError:
            raise
        except HTTPException as e:
            last_exc = e
            if (
                _character_api_error_is_transient(e)
                and attempt < CHARACTER_API_MAX_ATTEMPTS - 1
            ):
                delay = 2.0 * (attempt + 1)
                logger.warning(
                    "Character API transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    CHARACTER_API_MAX_ATTEMPTS,
                    delay,
                    e.detail,
                )
                await asyncio.sleep(delay)
                continue
            raise CharacterApiError(
                _format_character_api_error(e, endpoint_name=endpoint_name)
            ) from e
        except httpx.RequestError as e:
            last_exc = e
            if (
                _character_api_error_is_transient(e)
                and attempt < CHARACTER_API_MAX_ATTEMPTS - 1
            ):
                delay = 2.0 * (attempt + 1)
                logger.warning(
                    "Character API connection error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    CHARACTER_API_MAX_ATTEMPTS,
                    delay,
                    type(e).__name__,
                )
                await asyncio.sleep(delay)
                continue
            raise CharacterApiError(
                _format_character_api_error(e, endpoint_name=endpoint_name)
            ) from e

    if last_exc is not None:
        raise CharacterApiError(
            _format_character_api_error(last_exc, endpoint_name=endpoint_name)
        ) from last_exc
    raise CharacterApiError(_CHARACTER_API_TIMEOUT_MSG)

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
* `"personality"`: (String) Their temperament, contradictions, desires, fears, boundaries, and relationship habits.
* `"background"`: (String) The history and formative experiences that matter during roleplay.
* `"model_instructions"`: (String) Detailed instructions for an AI on how to accurately roleplay this character, including their speaking style, motivations, and key behaviors.
* `"speech_style"`: (String) Concrete guidance for vocabulary, rhythm, tone, verbal habits, and use of action prose.
* `"scenario"`: (String) The setting and context where interactions with this character typically take place.
* `"first_message"`: (String) A sample opening greeting from the character that captures their voice and personality.
* `"alternate_greetings"`: (Array of Strings) Optional alternate opening messages for different starting scenes.
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
    """Extract and validate JSON from model response (with local repair/salvage)."""
    from . import character_json_parse as cjp

    character_json, partial, _salvaged, _err = cjp.parse_character_json(response or "")
    if character_json:
        return character_json
    if partial and cjp.character_json_is_usable(partial, partial_ok=True):
        return partial
    return None
