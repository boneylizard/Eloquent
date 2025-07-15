import asyncio
import json
import logging
import time
import os
import re  # <-- Added missing import for regex
from typing import List, AsyncGenerator, Dict, Any, Optional, Union
from fastapi import HTTPException # <-- Added missing import
import random
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

                # NEW CODE - Filter for instruction leakage - MOVED HERE
                if False and generated_text and isinstance(generated_text, str):
                    # Check for common instruction leakage patterns
                    instruction_patterns = [
                        "and include all relevant information in the answer",
                        "and include all information in your response",
                        "be concise and clear",
                        "please provide a detailed explanation",
                        "please summarize the key points",
                        "please provide a comprehensive answer",
                        "please provide a detailed response",
                        "please provide a complete answer",
                        "please provide a thorough explanation",
                        "remember to address all parts of the question",
                        "please provide a complete response",
                        "please provide a detailed summary",
                        "please provide a thorough summary",
                        "please provide a comprehensive summary",
                        "please provide a complete explanation",
                        "please provide a thorough answer",
                        "please provide a detailed analysis",
                        "please provide a comprehensive analysis",
                        "please provide a complete analysis",
                        "please provide a thorough analysis",
                        "please provide a detailed overview",
                        "please provide a comprehensive overview",
                        "please provide a complete overview",
                        "please provide a thorough overview",
                        "please provide a detailed breakdown",
                        "and include all relevant details",
                        "and include all relevant context",
                        "and include all relevant examples",
                        "and include all relevant data",
                        "and include all relevant information",
                        "and include all relevant facts",
                        "and include all relevant evidence",
                        "and include all relevant arguments",
                        "and include all relevant points",
                        "and include all relevant insights",
                        "and include all relevant perspectives",
                        "and include all relevant considerations",
                        "and include all relevant implications",
                        "and include all relevant conclusions",
                        "and include all relevant recommendations",
                        "and include all relevant suggestions",
                        "and include all relevant observations",
                        "and include all relevant interpretations",
                        "and include all relevant analyses",
                        "and include all relevant evaluations",
                        "and include all relevant assessments",
                        "and include all relevant judgments",
                        "and include all relevant critiques",
                        "and include all relevant reviews",
                        "and include all relevant feedback",
                        "and include all relevant responses",
                        "and include all relevant reactions",
                        "and include all relevant comments",
                        "and include all relevant notes",
                        "and include all relevant annotations",
                        "and include all relevant references",
                        "to include all relevant information",
                        "to include all relevant context",
                        "to include all relevant examples",
                        "to include all relevant data",
                        "to include all relevant information",
                        "to include all relevant facts",
                        "to include all relevant evidence",
                        "to include all relevant arguments",
                        "to include all relevant points",
                        "to include all relevant insights",
                        "to include all relevant perspectives",
                        "to include all relevant considerations",
                        "to include all relevant implications",
                        "to include all relevant conclusions",
                        "to include all relevant recommendations",
                        "to include all relevant suggestions",
                        "to include all relevant observations",
                        "to include all relevant interpretations",
                        "to include all relevant analyses",
                        "to include all relevant evaluations",
                        "to include all relevant assessments",
                        "to include all relevant judgments",
                        "to indicate the end of the response",
                        "to indicate the end of the answer",
                        "to indicate the end of the text",
                        "to indicate the end of the output",
                        "to indicate the end of the message",
                        "to indicate the end of the conversation",
                        "to indicate the end of the discussion",
                        "to indicate the end of the explanation",
                        "to indicate the end of the summary",
                        "to indicate the end of the analysis",
                        "to indicate the end of the overview",
                        "to indicate the end of the breakdown",
                        "to indicate the end of the response",
                        "to indicate the end of the answer",
                        "to indicate the end of the text",
                        "to indicate the end of the output",
                        "to indicate the end of the message",
                        "to indicate the end of the conversation",
                        "to indicate the end of the discussion",
                        "to indicate the end of the explanation",
                        "to indicate the end of the summary",
                        "to indicate the end of the analysis",
                        "to indicate the end of the overview",
                        "to indicate the end of the breakdown",
                        "to indicate the end of the response",
                        "to indicate the end of the answer",
                        "to indicate the end of the text",
                        "to indicate the end of the output",
                        "to indicate the end of the message",
                        "to indicate the end of the conversation",
                        "to indicate the end of the discussion",
                        "to indicate the end of the explanation",
                        "to indicate the end of the summary",
                        "to indicate the end of the analysis",
                        "to indicate the end of the overview",
                        "to indicate the end of the breakdown",
                        "be sure to include all relevant information",
                        "be sure to include all relevant context",
                        "be sure to include all relevant examples",
                        "be sure to include all relevant data",
                        "be sure to include all relevant information",
                        "be sure to include all relevant facts",
                        "be sure to include all relevant evidence",
                        "be sure to include all relevant arguments",
                        "be sure to include all relevant points",
                        "be sure to include all relevant insights",
                        "be sure to include all relevant perspectives",
                        "be sure to include all relevant considerations",
                        "be sure to include all relevant implications",
                        "be sure to include all relevant conclusions",
                        "be sure to include all relevant recommendations",
                        "be sure to include all relevant suggestions",
                        "be sure to include all relevant observations",
                        "be sure to include all relevant interpretations",
                        "be sure to include all relevant analyses",
                    ]
                    
                    # Check if the text starts with any of these patterns
                    for pattern in instruction_patterns:
                        if generated_text.lower().startswith(pattern.lower()):
                            # Split at the first period and take everything after
                            parts = generated_text.split('.', 1)
                            if len(parts) > 1 and parts[1].strip():
                                logger.info(f"[inference] Filtered instruction leakage: {pattern}")
                                generated_text = parts[1].strip()
                                break
                    
                    # 2. Then check regex patterns for broader coverage
                    regex_patterns = [
                        # Pattern for "and include/to include all relevant..."
                        r"^(?:and|to|be sure to) include all (?:relevant|important|necessary|detailed|complete).*?\.(\s|$)",
                        
                        # Pattern for "please provide a..."
                        r"^please provide a (?:detailed|comprehensive|complete|thorough).*?\.(\s|$)",
                        
                        # Pattern for "please summarize..." and similar
                        r"^please (?:summarize|explain|describe|list|discuss|analyze|evaluate|assess).*?\.(\s|$)",
                        
                        # Pattern for "remember to address..."
                        r"^remember to (?:address|include|consider|mention|discuss).*?\.(\s|$)",
                        
                        # Pattern for "be concise..."
                        r"^be (?:concise|clear|detailed|thorough|comprehensive).*?\.(\s|$)",
                        
                        # Pattern for "to indicate the end..."
                        r"^to (?:indicate|signal|mark|show|signify) the end.*?\.(\s|$)"
                    ]
                    
                    # Check if the text starts with any of these patterns
                    for pattern in regex_patterns:
                        match = re.search(pattern, generated_text, re.IGNORECASE)
                        if match:
                            # Get the matched text
                            matched_prefix = match.group(0)
                            logger.info(f"[inference] Filtered pattern instruction: {matched_prefix}")
                            
                            # Remove the matched prefix
                            generated_text = generated_text[len(matched_prefix):].strip()
                            
                            # If we're left with nothing, break
                            if not generated_text:
                                break
                                
                            # If the instruction was followed by another sentence, keep checking
                            if re.match(r"^[A-Z]", generated_text):
                                continue
                            break
                
                token_count = completion["usage"]["completion_tokens"]
            elif 'ctransformers' in backend:
                 generated_text = model(
                    prompt, max_new_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                    stop=stop_sequences or [], echo=False
                 )
                 
                 # NEW CODE - Filter for instruction leakage - MOVED HERE TOO
                 if False and generated_text and isinstance(generated_text, str):
                    # Check for common instruction leakage patterns
                    instruction_patterns = [
                        "and include all relevant information in the answer",
                        "and include all information in your response",
                        "be concise and clear",
                        "please provide a detailed explanation",
                        "please summarize the key points",
                        "please provide a comprehensive answer",
                        "please provide a detailed response",
                        "please provide a complete answer",
                        "please provide a thorough explanation",
                        "remember to address all parts of the question",
                        "please provide a complete response",
                        "please provide a detailed summary",
                        "please provide a thorough summary",
                        "please provide a comprehensive summary",
                        "please provide a complete explanation",
                        "please provide a thorough answer",
                        "please provide a detailed analysis",
                        "please provide a comprehensive analysis",
                        "please provide a complete analysis",
                        "please provide a thorough analysis",
                        "please provide a detailed overview",
                        "please provide a comprehensive overview",
                        "please provide a complete overview",
                        "please provide a thorough overview",
                        "please provide a detailed breakdown",
                        "and include all relevant details",
                        "and include all relevant context",
                        "and include all relevant examples",
                        "and include all relevant data",
                        "and include all relevant information",
                        "and include all relevant facts",
                        "and include all relevant evidence",
                        "and include all relevant arguments",
                        "and include all relevant points",
                        "and include all relevant insights",
                        "and include all relevant perspectives",
                        "and include all relevant considerations",
                        "and include all relevant implications",
                        "and include all relevant conclusions",
                        "and include all relevant recommendations",
                        "and include all relevant suggestions",
                        "and include all relevant observations",
                        "and include all relevant interpretations",
                        "and include all relevant analyses",
                        "and include all relevant evaluations",
                        "and include all relevant assessments",
                        "and include all relevant judgments",
                        "and include all relevant critiques",
                        "and include all relevant reviews",
                        "and include all relevant feedback",
                        "and include all relevant responses",
                        "and include all relevant reactions",
                        "and include all relevant comments",
                        "and include all relevant notes",
                        "and include all relevant annotations",
                        "and include all relevant references",
                        "to include all relevant information",
                        "to include all relevant context",
                        "to include all relevant examples",
                        "to include all relevant data",
                        "to include all relevant information",
                        "to include all relevant facts",
                        "to include all relevant evidence",
                        "to include all relevant arguments",
                        "to include all relevant points",
                        "to include all relevant insights",
                        "to include all relevant perspectives",
                        "to include all relevant considerations",
                        "to include all relevant implications",
                        "to include all relevant conclusions",
                        "to include all relevant recommendations",
                        "to include all relevant suggestions",
                        "to include all relevant observations",
                        "to include all relevant interpretations",
                        "to include all relevant analyses",
                        "to include all relevant evaluations",
                        "to include all relevant assessments",
                        "to include all relevant judgments",
                        "to indicate the end of the response",
                        "to indicate the end of the answer",
                        "to indicate the end of the text",
                        "to indicate the end of the output",
                        "to indicate the end of the message",
                        "to indicate the end of the conversation",
                        "to indicate the end of the discussion",
                        "to indicate the end of the explanation",
                        "to indicate the end of the summary",
                        "to indicate the end of the analysis",
                        "to indicate the end of the overview",
                        "to indicate the end of the breakdown",
                        "to indicate the end of the response",
                        "to indicate the end of the answer",
                        "to indicate the end of the text",
                        "to indicate the end of the output",
                        "to indicate the end of the message",
                        "to indicate the end of the conversation",
                        "to indicate the end of the discussion",
                        "to indicate the end of the explanation",
                        "to indicate the end of the summary",
                        "to indicate the end of the analysis",
                        "to indicate the end of the overview",
                        "to indicate the end of the breakdown",
                        "to indicate the end of the response",
                        "to indicate the end of the answer",
                        "to indicate the end of the text",
                        "to indicate the end of the output",
                        "to indicate the end of the message",
                        "to indicate the end of the conversation",
                        "to indicate the end of the discussion",
                        "to indicate the end of the explanation",
                        "to indicate the end of the summary",
                        "to indicate the end of the analysis",
                        "to indicate the end of the overview",
                        "to indicate the end of the breakdown",
                        "be sure to include all relevant information",
                        "be sure to include all relevant context",
                        "be sure to include all relevant examples",
                        "be sure to include all relevant data",
                        "be sure to include all relevant information",
                        "be sure to include all relevant facts",
                        "be sure to include all relevant evidence",
                        "be sure to include all relevant arguments",
                        "be sure to include all relevant points",
                        "be sure to include all relevant insights",
                        "be sure to include all relevant perspectives",
                        "be sure to include all relevant considerations",
                        "be sure to include all relevant implications",
                        "be sure to include all relevant conclusions",
                        "be sure to include all relevant recommendations",
                        "be sure to include all relevant suggestions",
                        "be sure to include all relevant observations",
                        "be sure to include all relevant interpretations",
                        "be sure to include all relevant analyses",
                    ]
                    
                    # Check if the text starts with any of these patterns
                    for pattern in instruction_patterns:
                        if generated_text.lower().startswith(pattern.lower()):
                            # Split at the first period and take everything after
                            parts = generated_text.split('.', 1)
                            if len(parts) > 1 and parts[1].strip():
                                logger.info(f"[inference] Filtered instruction leakage: {pattern}")
                                generated_text = parts[1].strip()
                                break
                    
                    # 2. Then check regex patterns for broader coverage
                    regex_patterns = [
                        # Pattern for "and include/to include all relevant..."
                        r"^(?:and|to|be sure to) include all (?:relevant|important|necessary|detailed|complete).*?\.(\s|$)",
                        
                        # Pattern for "please provide a..."
                        r"^please provide a (?:detailed|comprehensive|complete|thorough).*?\.(\s|$)",
                        
                        # Pattern for "please summarize..." and similar
                        r"^please (?:summarize|explain|describe|list|discuss|analyze|evaluate|assess).*?\.(\s|$)",
                        
                        # Pattern for "remember to address..."
                        r"^remember to (?:address|include|consider|mention|discuss).*?\.(\s|$)",
                        
                        # Pattern for "be concise..."
                        r"^be (?:concise|clear|detailed|thorough|comprehensive).*?\.(\s|$)",
                        
                        # Pattern for "to indicate the end..."
                        r"^to (?:indicate|signal|mark|show|signify) the end.*?\.(\s|$)"
                    ]
                    
                    # Check if the text starts with any of these patterns
                    for pattern in regex_patterns:
                        match = re.search(pattern, generated_text, re.IGNORECASE)
                        if match:
                            # Get the matched text
                            matched_prefix = match.group(0)
                            logger.info(f"[inference] Filtered pattern instruction: {matched_prefix}")
                            
                            # Remove the matched prefix
                            generated_text = generated_text[len(matched_prefix):].strip()
                            
                            # If we're left with nothing, break
                            if not generated_text:
                                break
                                
                            # If the instruction was followed by another sentence, keep checking
                            if re.match(r"^[A-Z]", generated_text):
                                continue
                            break
                 
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

    Args:
        model_manager: The model manager instance
        model_name: Name of the model to use
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate (-1 for auto-sizing)
        temperature: Controls randomness, higher = more random
        top_p: Cumulative probability cutoff for token selection
        top_k: Number of highest probability tokens to consider
        repetition_penalty: Penalty for repeating tokens
        stop_sequences: List of sequences at which to stop generation
        gpu_id: Which GPU to use
        skip_local_check: If True, assume model is already loaded
        extract_memories: If True, extract memories from prompt
        echo: Whether to echo the prompt in the output
        **kwargs: Additional parameters like n_ctx

    Yields:
        Generated text tokens one by one
    """
    start_time = time.time()
    token_count = 0
    
    if stop_sequences is None:
        stop_sequences = []
        
    # Set up key variables for model loading
    target_gpu_id = gpu_id
    model_key = (model_name, target_gpu_id)
    model = None
    
    # --- AUTO-SIZE max_tokens from the loaded model's n_ctx ---
    # Do this early, so both memory extraction and streaming use the same value
    if max_tokens < 0:
        model_info = model_manager.loaded_models.get(model_key, {})
        # Default to 4096 if not found
        full_ctx = model_info.get("n_ctx", 4096)
        safety_margin = 50
        # Ensure we leave room for prompt and any stop tokens
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
                yield f"Error: {error_msg}"
                return

        # For all other requests (e.g., normal chat), use the existing check-and-load logic.
        else:
            is_loaded = model_key in model_manager.loaded_models
            current_context = model_manager.loaded_models.get(model_key, {}).get("n_ctx")
            context_matches = (intended_n_ctx is None) or (current_context is not None and current_context >= intended_n_ctx)
            needs_loading = not is_loaded or not context_matches

            if needs_loading:
                # Model needs loading or reloading
                reason = "not loaded" if not is_loaded else f"context mismatch (current={current_context}, intended={intended_n_ctx})"
                logger.info(f"[inference] Model {model_name} on GPU {target_gpu_id} {reason}, loading now for streaming.")
                try:
                    await model_manager.load_model(model_name, gpu_id=target_gpu_id, n_ctx=intended_n_ctx)
                    model = model_manager.get_model(model_name, target_gpu_id)
                except Exception as e:
                    error_msg = f"Failed to load model {model_name} on GPU {target_gpu_id}: {e}"
                    logger.error(f"[inference] {error_msg}")
                    yield f"Error: {error_msg}"
                    return
            else:
                # Model already loaded with correct context
                logger.info(f"[inference] Using already loaded model {model_name} on GPU {target_gpu_id} for streaming.")
                model = model_manager.get_model(model_name, target_gpu_id)

        # Final safety check
        if model is None:
            error_msg = f"Failed to get model object for {model_name} after all checks"
            logger.error(f"[inference] {error_msg}")
            yield f"Error: {error_msg}"
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
            
            # Use streaming to collect all tokens for memory extraction
            memory_generator = model.create_completion(
                memory_extraction_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                repeat_penalty=repetition_penalty, stop=stop_sequences or [], stream=True
            )
            
            # FIX: Use regular for loop instead of async for
            for completion in memory_generator:
                if completion and "choices" in completion and len(completion["choices"]) > 0:
                    token = completion["choices"][0].get("text", "")
                    memory_tokens += token
                    # Let other tasks run
                    await asyncio.sleep(0)
            
            # Process the complete memory text
            extracted_memories = process_model_output(memory_tokens)
            logger.info(f"[inference] Extracted {len(extracted_memories)} memories from prompt")
            
            # Send memories as a special event type
            yield f"event: memory\ndata: {json.dumps({'memories': extracted_memories})}\n\n"
        
        # --- Main Text Generation Streaming ---
        if 'llama_cpp' in backend:
            # Create the completion generator
            completion_generator = model.create_completion(
                prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                repeat_penalty=repetition_penalty, stop=stop_sequences or [], stream=True, echo=echo,
                seed=random.randint(0, 2_147_483_647)
            )
            # FIX: Use regular for loop instead of async for
            for completion in completion_generator:
                if completion and "choices" in completion and len(completion["choices"]) > 0:
                    token = completion["choices"][0].get("text", "")
                    if token:
                        token_count += 1
                        yield token
                        # Use sleep(0) to yield control without delay
                        await asyncio.sleep(0)
                else:
                    # Handle unexpected format better
                    logger.warning(f"[inference] Unexpected completion format: {completion}")
                    if token_count == 0:
                        # If this is the first token, send an error
                        yield "Error: Invalid response format from model"
                        return

        elif 'ctransformers' in backend:
            logger.warning("[inference] Using token-by-token approach for ctransformers streaming")
            try:
                # Synchronous approach, still blocks the event loop
                generated_tokens = model.generate(
                    model.tokenize(prompt), top_k=top_k, top_p=top_p, temperature=temperature,
                    repetition_penalty=repetition_penalty, max_new_tokens=max_tokens,
                    stop_sequences=model.tokenize(stop_sequences) if stop_sequences else None, reset=True
                )
                
                for token_id in generated_tokens:
                    token = model.detokenize([token_id])
                    token_count += 1
                    yield token
                    # Yield control but don't add unnecessary delay
                    await asyncio.sleep(0)
            except Exception as ctransformers_err:
                logger.error(f"[inference] ctransformers streaming error: {ctransformers_err}")
                yield f"Error: {str(ctransformers_err)}"
                return
        else:
            error_msg = f"Unsupported backend for streaming: {backend}"
            logger.error(f"[inference] {error_msg}")
            yield f"Error: {error_msg}"
            return

    except HTTPException as http_exc:
        logger.error(f"[inference] HTTPException during streaming: {http_exc.detail}")
        yield f"Error: {http_exc.detail}"
    except ValueError as val_err:
        logger.error(f"[inference] ValueError during streaming: {val_err}")
        yield f"Error: {str(val_err)}"
    except Exception as e:
        logger.error(f"[inference] Unexpected error during streaming: {e}", exc_info=True)
        yield f"Error during generation: {str(e)}"
    finally:
        # Guaranteed to execute for metrics
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_per_second = token_count / time_taken if time_taken > 0 and token_count > 0 else 0
        logger.info(f"[inference] Streaming completed: {token_count} tokens in {time_taken:.2f}s ({tokens_per_second:.2f} tokens/s)")