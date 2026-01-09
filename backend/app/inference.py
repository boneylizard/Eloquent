import asyncio
import json
import logging
import time
import os
import re  # <-- Added missing import for regex
from typing import List, AsyncGenerator, Dict, Any, Optional, Union
from fastapi import HTTPException # <-- Added missing import
import random

def _normalize_completion(c: Any) -> Dict[str, Any]:
    """
    Normalize various completion return shapes to:
      {"choices":[{"text": "..."}], "usage":{...}} minimally.
    Accepts dict, JSON-string, plain string, bytes.
    """
    if c is None:
        return {"choices": []}

    # bytes → str
    if isinstance(c, (bytes, bytearray)):
        try:
            c = c.decode("utf-8", errors="ignore")
        except Exception:
            return {"choices": []}

    # JSON-string → dict, or plain string → wrap
    if isinstance(c, str):
        c_str = c.strip()
        if c_str.startswith("{") or c_str.startswith("["):
            try:
                c = json.loads(c_str)
            except Exception:
                # fall through and wrap as text
                pass
        if isinstance(c, str):
            return {"choices": [{"text": c}]}

    # Already a dict but maybe not OpenAI shape
    if isinstance(c, dict):
        if "choices" in c and isinstance(c["choices"], list):
            return c
        # Llama.cpp / custom services sometimes return {'content': '...'} or {'text': '...'}
        if "content" in c and isinstance(c["content"], str):
            return {"choices": [{"text": c["content"]}], **{k:v for k,v in c.items() if k not in ("content",)}}
        if "text" in c and isinstance(c["text"], str):
            return {"choices": [{"text": c["text"]}], **{k:v for k,v in c.items() if k not in ("text",)}}

    # Unknown shape → empty (avoid TypeError)
    return {"choices": []}

def _extract_text_from_chunk(c: Any) -> str:
    """
    Given any streaming chunk shape (dict / str / bytes) return ONLY the text/content.
    Handles OpenAI-style streaming deltas, llama.cpp/ctransformers shapes, and plain strings.
    Returns '' on anything non-textual.
    """
    try:
        # bytes → str
        if isinstance(c, (bytes, bytearray)):
            c = c.decode("utf-8", errors="ignore")

        # JSON string → dict
        if isinstance(c, str):
            s = c.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    c = json.loads(s)
                except Exception:
                    # if it's a plain string chunk, return it as-is
                    return c
            else:
                # plain text chunk
                return c

        # dict-like chunks
        if isinstance(c, dict):
            # OpenAI Chat stream: {"choices":[{"delta":{"content":"..."}}], ...}
            if "choices" in c and isinstance(c["choices"], list) and c["choices"]:
                ch = c["choices"][0]
                if isinstance(ch, dict):
                    delta = ch.get("delta")
                    if isinstance(delta, dict):
                        return delta.get("content", "") or ""
                    # Some providers use {"choices":[{"text":"..."}]}
                    if "text" in ch and isinstance(ch["text"], str):
                        return ch["text"]
                    # Some non-streaming shapes
                    msg = ch.get("message")
                    if isinstance(msg, dict):
                        return msg.get("content", "") or ""

            # Non-OpenAI variants
            if "text" in c and isinstance(c["text"], str):
                return c["text"]
            if "content" in c and isinstance(c["content"], str):
                return c["content"]

        # Anything else → nothing
        return ""
    except Exception:
        return ""   
# Assume ModelManager is imported correctly if needed for type hinting,
# but it's primarily passed as an argument.
# from .model_manager import ModelManager
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__) # Use logger

def process_model_output(text: str) -> List[Dict[str, Any]]:
    """
    Process the model output to extract memories.
    Attempts to parse JSON from the text or extract structured data.

    Args:
        text: The raw text output from the model

    Returns:
        List of memory objects extracted
    """
    # Find JSON-like content in the text
    try:
        # Try to find the start of a JSON array
        start_idx = text.find('[')
        if start_idx == -1:
            logger.warning("No JSON array found in model output for memory extraction")
            return []

        # Find the matching closing bracket robustly
        open_brackets = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '[':
                open_brackets += 1
            elif text[i] == ']':
                open_brackets -= 1
                if open_brackets == 0:
                    end_idx = i + 1
                    break

        if end_idx == -1:
            logger.warning("No matching closing bracket found in model output for memory extraction")
            return []

        json_str = text[start_idx:end_idx]
        memories = json.loads(json_str)

        # Validate the structure of each memory
        valid_memories = []
        if not isinstance(memories, list):
            logger.warning(f"Parsed JSON is not a list: {type(memories)}")
            return []

        for memory in memories:
            if isinstance(memory, dict) and all(key in memory for key in ["content", "category", "importance"]):
                # Basic validation passed
                valid_memories.append(memory)
            else:
                logger.warning(f"Invalid memory structure skipped: {memory}")

        return valid_memories

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error during memory extraction: {e}")
        logger.debug(f"Problematic text snippet: {text[start_idx:end_idx if end_idx > start_idx else start_idx+200]}")
        return []
    except Exception as e:
        logger.error(f"Error processing model output for memories: {e}", exc_info=True)
        return []

async def generate_text(
    model_manager,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    stop_sequences: List[str] = None,
    gpu_id: int = 0,
    extract_memories: bool = False,
    skip_local_check: bool = False,
    echo: bool = False,
    **kwargs
) -> Union[str, List[Dict[str, Any]]]:
    """
    Generate text synchronously from a model.
    If skip_local_check is True, assumes model is loaded and directly tries to get it.
    """
    # Safety check: Prevent loading API endpoints as local models
    if model_name and model_name.startswith('endpoint-'):
        error_msg = f"Model '{model_name}' is an API endpoint and cannot be loaded as a local model. This should have been routed to the OpenAI-compatible endpoint."
        logger.error(f"[inference] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    # --- AUTO-SIZE max_tokens from the loaded model's n_ctx ---
    if max_tokens < 0:
        model_key = (model_name, gpu_id)
        model_info = model_manager.loaded_models.get(model_key, {})
        # Default to 4096 if not found
        full_ctx = model_info.get("n_ctx", 4096)
        safety_margin = 50
        # Ensure we leave room for prompt and any stop tokens
        max_tokens = max(full_ctx - safety_margin, 1)
        logger.info(f"[inference] Autosized max_tokens to {max_tokens} based on n_ctx={full_ctx}")

    if stop_sequences is None:
        stop_sequences = []

    try:  # Outer try block
        target_gpu_id = gpu_id
        model_key = (model_name, target_gpu_id)
        model = None  # Define model variable

        # --- CORRECTED LOGIC ---
        # For testing requests, the model MUST already be loaded. We just get it.
        # If it's not loaded, get_model will raise a ValueError, which is the correct behavior.
        if kwargs.get('request_purpose') in ["model_testing", "model_judging"]:
            logger.info(f"[inference] Purpose '{kwargs.get('request_purpose')}': Directly getting pre-loaded model '{model_name}' on GPU {target_gpu_id}.")
            try:
                model = model_manager.get_model(model_name, target_gpu_id)
                logger.info(f"[inference] Successfully retrieved pre-loaded model for purpose '{kwargs.get('request_purpose')}'.")
            except ValueError as e:
                logger.error(f"[inference] CRITICAL: Model '{model_name}' for purpose '{kwargs.get('request_purpose')}' was NOT pre-loaded on GPU {target_gpu_id}. Error: {e}")
                # Re-raise as HTTPException so the frontend gets a meaningful error
                raise HTTPException(status_code=409, detail=f"Model '{model_name}' required for testing was not pre-loaded on GPU {target_gpu_id}.") from e

        # For all other requests (e.g., normal chat), use the existing check-and-load logic.
        else:
            # --- Perform normal local state check and load/reload if needed ---
            intended_n_ctx = kwargs.get('n_ctx')
            is_loaded = model_key in model_manager.loaded_models
            current_context = model_manager.loaded_models.get(model_key, {}).get("n_ctx")
            context_matches = (intended_n_ctx is None) or (current_context is not None and current_context >= intended_n_ctx)
            needs_action = not is_loaded or not context_matches

            if needs_action:
                load_reason = "not loaded" if not is_loaded else f"context mismatch (current {current_context}, intended {intended_n_ctx})"
                logger.info(f"[inference] Model {model_name} on GPU {target_gpu_id} {load_reason}. Loading/Reloading...")
                try:
                    await model_manager.load_model(model_name, gpu_id=target_gpu_id, n_ctx=intended_n_ctx)
                    model = model_manager.get_model(model_name, target_gpu_id)
                except Exception as load_err:
                    logger.error(f"[inference] Error during load/reload: {load_err}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to load/reload model {model_name}.") from load_err
            else:
                logger.info(f"[inference] Model {model_name} already loaded on GPU {target_gpu_id} with context {current_context}.")
                model = model_manager.get_model(model_name, target_gpu_id)
        # --- End of Corrected Logic Block ---

        # Final check if model object exists after either path
        if model is None:
            logger.error(f"[inference] CRITICAL: Model object is None after load/check logic for model '{model_name}' on GPU {target_gpu_id}.")
            raise ValueError(f"Failed to obtain model object for {model_name} GPU {target_gpu_id}")


        # --- Generation Logic ---
        start_time = time.time()
        logger.debug(f"[inference] FULL PROMPT >>>\n{prompt}\n<<<")
        #anti_prefix_instruction = "\n\nIMPORTANT: Begin your response with the actual content. Do not start with phrases like 'to indicate the end of the response', 'to signal completion', 'to ensure the AI does not continue', 'and include all relevant information', or any similar meta-commentary. Start directly with your answer."
        #prompt = prompt + anti_prefix_instruction
        #logger.info("[inference] Added anti-prefix instruction to prompt")
        token_count = 0
        backend = model.__class__.__module__.split('.')[0] if model else "unknown"
        logger.info(f"[inference] Using backend: {backend} for model {model_name}")
        generated_text = ""
        
        if extract_memories: # Handle memory extraction request (legacy?)
            logger.warning("[inference] extract_memories=True path invoked in generate_text.")
            memory_extraction_prompt = f"Analyze the following text and extract key information for memory. Format as JSON array.\nText:\n{prompt}"
            if 'llama_cpp' in backend:
                completion = model.create_completion(
                    memory_extraction_prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, repeat_penalty=repetition_penalty,
                    stop=stop_sequences or [], echo=False,
                    seed=random.randint(0, 2_147_483_647)
                )
                text_output = completion["choices"][0]["text"]
                logger.debug(f"[inference] RAW MODEL OUTPUT >>>\n{text_output}\n<<<")
                token_count = completion["usage"]["completion_tokens"]
            elif 'ctransformers' in backend:
                 text_output = model(
                    memory_extraction_prompt, max_new_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                    stop=stop_sequences or []
                )
                 token_count = len(text_output.split()) # Rough estimate
            else:
                raise ValueError(f"Unknown backend for memory extraction: {backend}")
            extracted_memories = process_model_output(text_output) # Assuming process_model_output is defined
            end_time = time.time(); time_taken = end_time - start_time
            logger.info(f"[inference] Extracted {len(extracted_memories)} memories using {token_count} tokens in {time_taken:.2f}s on GPU {target_gpu_id}")
            return extracted_memories
        else: # Standard text generation
            if 'llama_cpp' in backend:
                completion = model.create_completion(
                    prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, repeat_penalty=repetition_penalty,
                    stop=stop_sequences or [],
                    seed=random.randint(0, 2_147_483_647)
                )
                generated_text = completion["choices"][0]["text"]
                logger.debug(f"[inference] RAW MODEL OUTPUT >>>\n{generated_text}\n<<<")
                token_count = completion["usage"]["completion_tokens"]
            elif 'ctransformers' in backend:
                generated_text = model(
                    prompt, max_new_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                    stop=stop_sequences or [], echo=False
                )
                token_count = len(generated_text.split()) # Rough estimate
            else:
                raise ValueError(f"Unknown backend for generation: {backend}")

            end_time = time.time()
            time_taken = end_time - start_time
            tokens_per_second = token_count / time_taken if time_taken > 0 else 0
            logger.info(f"[inference] Generated {token_count} tokens in {time_taken:.2f}s ({tokens_per_second:.2f} tokens/s) on GPU {target_gpu_id}")
            return generated_text

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except ValueError as ve: # Catch ValueErrors (e.g., from get_model or final check)
        logger.error(f"[inference] ValueError during generate_text execution: {ve}", exc_info=True)
        # Important: Re-wrap the ValueError in HTTPException for the background task handler
        raise HTTPException(status_code=500, detail=f"Generation failed due to value error: {str(ve)}") from ve
    except Exception as e:
        logger.error(f"[inference] Error during generate_text execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}") from e



async def generate_text_streaming(
    model_manager,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    stop_sequences: List[str] = None,
    gpu_id: int = 0,
    skip_local_check: bool = False,
    extract_memories: bool = False,
    echo: bool = False,
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Stream text generation from a model.
    Ensures the correct model instance and context length are used on the specified GPU.
    """
    start_time = time.time()
    emitted_tokens = 0
    
    if stop_sequences is None:
        stop_sequences = []
    
    # Safety check: Prevent loading API endpoints as local models
    if model_name and model_name.startswith('endpoint-'):
        error_msg = f"Model '{model_name}' is an API endpoint and cannot be loaded as a local model. This should have been routed to the OpenAI-compatible endpoint."
        logger.error(f"[inference] {error_msg}")
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        yield "data: [DONE]\n\n"
        return
        
    # Set up key variables for model loading
    target_gpu_id = gpu_id
    model_key = (model_name, target_gpu_id)
    model = None
    
    # --- AUTO-SIZE max_tokens from the loaded model's n_ctx ---
    if max_tokens < 0:
        model_info = model_manager.loaded_models.get(model_key, {})
        full_ctx = model_info.get("n_ctx", 4096)
        safety_margin = 50
        max_tokens = max(full_ctx - safety_margin, 1)
        logger.info(f"[inference] Autosized max_tokens to {max_tokens} based on n_ctx={full_ctx}")

    try:
        # --- CORRECTED LOGIC ---
        intended_n_ctx = kwargs.get('n_ctx')

        # For testing requests, the model MUST already be loaded. We just get it.
        if kwargs.get('request_purpose') in ["model_testing", "model_judging"]:
            logger.info(f"[inference] Purpose '{kwargs.get('request_purpose')}': Directly getting pre-loaded model '{model_name}' on GPU {target_gpu_id}.")
            try:
                model = model_manager.get_model(model_name, target_gpu_id)
                logger.info(f"[inference] Successfully retrieved pre-loaded model for streaming purpose '{kwargs.get('request_purpose')}'.")
            except ValueError as e:
                error_msg = f"Model '{model_name}' required for testing was not pre-loaded on GPU {target_gpu_id}."
                logger.error(f"[inference] CRITICAL: {error_msg} Error: {e}")
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

        # For all other requests (e.g., normal chat), use the existing check-and-load logic.
        else:
            is_loaded = model_key in model_manager.loaded_models
            current_context = model_manager.loaded_models.get(model_key, {}).get("n_ctx")
            context_matches = (intended_n_ctx is None) or (current_context is not None and current_context >= intended_n_ctx)
            needs_loading = not is_loaded or not context_matches

            if needs_loading:
                # Double-check: API endpoints should never reach this point
                if model_name and model_name.startswith('endpoint-'):
                    error_msg = f"CRITICAL: Attempted to load API endpoint '{model_name}' as local model. This should have been caught earlier."
                    logger.error(f"[inference] {error_msg}")
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                reason = "not loaded" if not is_loaded else f"context mismatch (current={current_context}, intended={intended_n_ctx})"
                logger.info(f"[inference] Model {model_name} on GPU {target_gpu_id} {reason}, loading now for streaming.")
                try:
                    await model_manager.load_model(model_name, gpu_id=target_gpu_id, n_ctx=intended_n_ctx)
                    model = model_manager.get_model(model_name, target_gpu_id)
                except Exception as e:
                    error_msg = f"Failed to load model {model_name} on GPU {target_gpu_id}: {e}"
                    logger.error(f"[inference] {error_msg}")
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            else:
                logger.info(f"[inference] Using already loaded model {model_name} on GPU {target_gpu_id} for streaming.")
                model = model_manager.get_model(model_name, target_gpu_id)

        # Final safety check
        if model is None:
            error_msg = f"Failed to get model object for {model_name} after all checks"
            logger.error(f"[inference] {error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Get backend type for generation method
        backend = model.__class__.__module__.split('.')[0] if model else "unknown"
        logger.info(f"[inference] Using backend: {backend} for streaming model {model_name}")
        logger.debug(f"[inference] PROMPT >>>\n{prompt[:200]}... (length: {len(prompt)})\n<<<")

        # --- Memory Extraction (if requested) ---
        if extract_memories and 'llama_cpp' in backend:
            logger.info("[inference] Memory extraction requested during streaming")
            memory_extraction_prompt = f"Analyze the following text and extract key information for memory. Format as JSON array.\nText:\n{prompt}"
            memory_tokens = ""
            
            # Use non-streaming for memory extraction (ModelService handles this)
            memory_result = model.create_completion(
                memory_extraction_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                repeat_penalty=repetition_penalty, stop=stop_sequences or []
            )
            
            # Process the non-streaming result
            completion_norm = _normalize_completion(memory_result)
            choices = completion_norm.get("choices") or []
            first = choices[0] if choices else {}
            memory_tokens = first.get("text") or first.get("delta", {}).get("content") or ""
            
            # Process the complete memory text
            extracted_memories = process_model_output(memory_tokens)
            logger.info(f"[inference] Extracted {len(extracted_memories)} memories from prompt")
            
            # Send memories as a special event type
            yield f"event: memory\ndata: {json.dumps({'memories': extracted_memories})}\n\n"
        
        # --- Main Text Generation (Streaming) ---
        def _extract_text(cobj):
            cobj = _normalize_completion(cobj)
            choices = cobj.get("choices") or []
            if not choices:
                return ""
            first = choices[0] or {}
            # Support both llama.cpp text and OpenAI-style delta content
            return first.get("text") or (first.get("delta") or {}).get("content") or ""
        
        if 'llama_cpp' in backend:
            completion_generator = model.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stop=stop_sequences or [],
                stream=True,
                echo=echo,
                seed=random.randint(0, 2_147_483_647),
            )

            # IMPORTANT: do NOT iterate the dict; always extract the text
            for completion in completion_generator:
                piece = _extract_text_from_chunk(completion)
                if not piece:
                    await asyncio.sleep(0)
                    continue

                emitted_tokens += 1
                # Emit SSE JSON line
                yield f"data: {json.dumps({'text': piece})}\n\n"
                await asyncio.sleep(0)

        elif 'ctransformers' in backend:
            logger.warning("[inference] Using token-by-token approach for ctransformers streaming")
            try:
                generated_tokens = model.generate(
                    model.tokenize(prompt),
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_tokens,
                    stop_sequences=model.tokenize(stop_sequences) if stop_sequences else None,
                    reset=True,
                )

                for token_id in generated_tokens:
                    token = model.detokenize([token_id])
                    if not token:
                        await asyncio.sleep(0)
                        continue
                    emitted_tokens += 1
                    yield f"data: {json.dumps({'text': token})}\n\n"
                    await asyncio.sleep(0)
            except Exception as ctransformers_err:
                logger.error(f"[inference] ctransformers streaming error: {ctransformers_err}")
                # Emit SSE error line instead of raw text
                yield f"data: {json.dumps({'error': str(ctransformers_err)})}\n\n"
                yield "data: [DONE]\n\n"
                return

        else:
            error_msg = f"Unsupported backend for streaming: {backend}"
            logger.error(f"[inference] {error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Signal end of stream to the frontend
        yield "data: [DONE]\n\n"
        return

    except HTTPException as http_exc:
        logger.error(f"[inference] HTTPException during streaming: {http_exc.detail}")
        yield f"data: {json.dumps({'error': http_exc.detail})}\n\n"
        yield "data: [DONE]\n\n"

    except ValueError as val_err:
        logger.error(f"[inference] ValueError during streaming: {val_err}")
        yield f"data: {json.dumps({'error': str(val_err)})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"[inference] Unexpected error during streaming: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': f'Error during generation: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Guaranteed to execute for metrics
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_per_second = emitted_tokens / time_taken if time_taken > 0 and emitted_tokens > 0 else 0
        logger.info(f"[inference] Streaming completed: {emitted_tokens} tokens in {time_taken:.2f}s ({tokens_per_second:.2f} tokens/s)")