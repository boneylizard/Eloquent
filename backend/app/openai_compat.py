"""
OpenAI API Compatibility Layer for Eloquent
Provides standard OpenAI endpoints that route through Eloquent's existing inference system.
"""

import json
import time
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import logging
import httpx
from .model_manager import ModelManager
from . import inference
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatibility"])

# === Dependency to get model manager ===
def get_model_manager(request: Request) -> ModelManager:
    """Get the ModelManager instance from request app state"""
    return getattr(request.app.state, 'model_manager', None)

# === OpenAI API Models ===

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0, le=1, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(2048, ge=1, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    
    # Additional parameters that map to Eloquent's system
    top_k: Optional[int] = Field(40, ge=1, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.1, ge=0, description="Repetition penalty")
    gpu_id: Optional[int] = Field(None, description="GPU ID to use (Eloquent-specific)")

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "eloquent"

# === Helper Functions ===
def is_api_endpoint(model_name: str) -> bool:
    """Check if the model name is an API endpoint"""
    return model_name and model_name.startswith('endpoint-')

def get_configured_endpoint(model_id: str = None):
    """Read custom API endpoints from settings.json and find the specified one."""
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        if not settings_path.exists():
            return None

        with open(settings_path, 'r') as f:
            settings = json.load(f)

        custom_endpoints = settings.get('customApiEndpoints', [])

        # If a model_id is provided, find that specific endpoint
        if model_id:
            for endpoint in custom_endpoints:
                if endpoint.get('id') == model_id and endpoint.get('enabled', False):
                    return {
                        'url': endpoint.get('url', '').rstrip('/'),
                        'api_key': endpoint.get('apiKey', ''),
                        'name': endpoint.get('name', 'Custom Endpoint'),
                        'model': endpoint.get('model', '')  # Model name to send to the API
                    }
            # If the specific endpoint is not found or disabled, it's an error
            return None

        # Fallback for old behavior: find the first enabled endpoint
        for endpoint in custom_endpoints:
            if endpoint.get('enabled', False):
                return {
                    'url': endpoint.get('url', '').rstrip('/'),
                    'api_key': endpoint.get('apiKey', ''),
                    'name': endpoint.get('name', 'Custom Endpoint'),
                    'model': endpoint.get('model', '')  # Model name to send to the API
                }
        return None
    except Exception as e:
        logger.error(f"Error reading settings: {e}")
        return None
def _prepare_endpoint_request(model_name: str, request_data: dict):
    """Prepare endpoint config and URL - returns (endpoint_config, url, request_data) or raises HTTPException"""
    endpoint_config = get_configured_endpoint(model_name)
    
    if not endpoint_config:
        raise HTTPException(
            status_code=400, 
            detail="No custom API endpoints configured. Please add one in Settings → LLM Settings → Custom API Endpoints"
        )
    
    base_url = endpoint_config['url']
    # Avoid double /v1 if the base URL already ends with /v1
    if base_url.endswith('/v1'):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"
    
    # Don't send internal endpoint ID as model name - use configured model or a default
    if request_data.get('model', '').startswith('endpoint-'):
        configured_model = endpoint_config.get('model', '').strip()
        if configured_model:
            request_data['model'] = configured_model
        else:
            # Use a generic default that most OpenAI-compatible APIs accept
            request_data['model'] = 'gpt-3.5-turbo'
    
    logger.info(f"[OpenAI Compat] Forwarding {model_name} to {endpoint_config['name']} at {url}")
    logger.info(f"[OpenAI Compat] Request data: {request_data}")
    
    return endpoint_config, url, request_data


async def forward_to_configured_endpoint_streaming(endpoint_config: dict, url: str, request_data: dict):
    """Forward OpenAI streaming request to the configured custom endpoint.
    
    Note: endpoint_config, url, and request_data should be prepared by _prepare_endpoint_request()
    before calling this generator to ensure errors are raised before streaming starts.
    """
    # Build headers similar to what SillyTavern sends
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    
    api_key = endpoint_config.get('api_key', '')
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # Chub.ai also accepts CH-API-Key header
        if api_key.startswith('CHK-'):
            headers["CH-API-Key"] = api_key
    
    base_url = endpoint_config['url']
    
    logger.info(f"[OpenAI Compat] Making request to {url}")
    logger.info(f"[OpenAI Compat] Headers (redacted): {list(headers.keys())}")
    logger.info(f"[OpenAI Compat] Request body keys: {list(request_data.keys())}")
    
    try:
        # Note: verify=False is for debugging SSL issues - remove in production if not needed
        async with httpx.AsyncClient(timeout=150.0, follow_redirects=True, verify=True) as client:
            logger.info(f"[OpenAI Compat] Initiating POST to {url}...")
            async with client.stream("POST", url, headers=headers, json=request_data) as response:
                logger.info(f"[OpenAI Compat] Got response status: {response.status_code}")
                if response.status_code != 200:
                    error_text = await response.aread()
                    # Yield error as SSE event instead of raising (can't raise mid-stream)
                    error_msg = f"Remote API error ({response.status_code}): {error_text.decode()}"
                    logger.error(f"[OpenAI Compat] {error_msg}")
                    error_event = {
                        "error": {
                            "message": error_msg,
                            "type": "api_error",
                            "code": response.status_code
                        }
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                async for chunk in response.aiter_raw():
                    yield chunk
                    
    except httpx.RequestError as e:
        logger.error(f"[OpenAI Compat] Connection error to {url}: {type(e).__name__}: {e}", exc_info=True)
        # Yield error as SSE event
        error_event = {
            "error": {
                "message": f"Cannot connect to {endpoint_config['name']} at {base_url}: {type(e).__name__}: {str(e)}",
                "type": "connection_error",
                "code": 502
            }
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"[OpenAI Compat] Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        error_event = {
            "error": {
                "message": f"Unexpected error: {type(e).__name__}: {str(e)}",
                "type": "unknown_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        yield "data: [DONE]\n\n"

async def forward_to_configured_endpoint_non_streaming(endpoint_config: dict, url: str, request_data: dict):
    """Forward OpenAI non-streaming request to the configured custom endpoint.
    
    Note: endpoint_config, url, and request_data should be prepared by _prepare_endpoint_request()
    """
    # Build headers similar to what SillyTavern sends
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    
    api_key = endpoint_config.get('api_key', '')
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # Chub.ai also accepts CH-API-Key header
        if api_key.startswith('CHK-'):
            headers["CH-API-Key"] = api_key
    
    base_url = endpoint_config['url']
    
    logger.info(f"[OpenAI Compat] Making non-streaming request to {url}")
    
    try:
        async with httpx.AsyncClient(timeout=150.0, follow_redirects=True) as client:
            response = await client.post(url, headers=headers, json=request_data)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Remote API error from {base_url}: {response.text}"
                )
            return response.json()
                    
    except httpx.RequestError as e:
        logger.error(f"[OpenAI Compat] Connection error to {url}: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=502, 
            detail=f"Cannot connect to {endpoint_config['name']} at {base_url}: {type(e).__name__}: {str(e)}"
        )
def convert_messages_to_prompt(messages: List[ChatMessage], model_name: str) -> str:
    """Convert OpenAI messages to Eloquent prompt format"""
    # Extract system message if present
    system_msg = "You are a helpful assistant."
    user_messages = []
    
    for msg in messages:
        if msg.role == "system":
            system_msg = msg.content
        else:
            user_messages.append({"role": msg.role, "content": msg.content})
    
    # This is a simple conversion - you might want to use your formatPrompt function
    # For now, we'll build a basic chat format
    prompt = f"{system_msg}\n\n"
    
    for msg in user_messages:
        if msg["role"] == "user":
            prompt += f"Human: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    
    prompt += "Assistant:"
    return prompt

def create_openai_chunk(chunk_id: str, model: str, content: str = "", finish_reason: str = None) -> str:
    """Create OpenAI-compatible streaming chunk"""
    chunk_data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(chunk_data)}\n\n"

def get_api_endpoint_url():
    """Read API endpoint URL from settings.json"""
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            return settings.get('apiEndpointUrl')
    except:
        pass
    return None

async def stream_eloquent_to_openai(inference_module, model_manager, params: dict, chunk_id: str, model: str):
    """Convert Eloquent streaming response to OpenAI format using direct inference"""
    try:
        # Send initial chunk
        yield create_openai_chunk(chunk_id, model, "", None)
        
        # Use your existing inference module directly
        async for token in inference_module.generate_text_streaming(
            model_manager=model_manager,
            model_name=params["model_name"],
            prompt=params["prompt"],
            max_tokens=params.get("max_tokens", 2048),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 40),
            repetition_penalty=params.get("repetition_penalty", 1.1),
            stop_sequences=params.get("stop_sequences", []),
            gpu_id=params.get("gpu_id", 0)
        ):
            if token:  # Only send non-empty tokens
                yield create_openai_chunk(chunk_id, model, token, None)
        
        # Send final chunk with finish_reason
        yield create_openai_chunk(chunk_id, model, "", "stop")
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        # Send error chunk
        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk", 
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "error"
            }],
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

# === API Endpoints ===

@router.get("/models")
async def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    """List available models - uses Eloquent's model manager directly"""
    try:
        if not model_manager:
            raise HTTPException(status_code=500, detail="Model manager not available")
        
        # Get available models from your model manager
        available_models = model_manager.list_available_models()
        
        # Convert to OpenAI format
        openai_models = []
        if isinstance(available_models, list):
            for model in available_models:
                openai_models.append(ModelInfo(
                    id=model if isinstance(model, str) else model.get("name", "unknown"),
                    created=int(time.time()),
                ))
        elif isinstance(available_models, dict):
            # Handle different response formats from your model manager
            model_list = available_models.get("available_models", available_models.get("models", []))
            for model in model_list:
                openai_models.append(ModelInfo(
                    id=model if isinstance(model, str) else model.get("name", "unknown"),
                    created=int(time.time()),
                ))
        
        return {"object": "list", "data": openai_models}
    
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        # Return a basic response if we can't fetch the list
        return {
            "object": "list", 
            "data": [ModelInfo(id="default", created=int(time.time()))]
        }

@router.post("/chat/completions")
async def chat_completions(raw_request: Request, model_manager: ModelManager = Depends(get_model_manager)):
    """OpenAI-compatible chat completions endpoint with raw request logging."""

    # --- START DEBUG LOG ---
    # Log the raw request body to see exactly what the frontend is sending
    try:
        request_body_json = await raw_request.json()
        logger.info(f"🚨 [/v1/chat/completions] RAW REQUEST BODY:\n{json.dumps(request_body_json, indent=2)}")
    except json.JSONDecodeError:
        request_body_raw = await raw_request.body()
        logger.error(f"🚨 [/v1/chat/completions] FAILED TO PARSE JSON. RAW BODY:\n{request_body_raw.decode('utf-8')}")
        raise HTTPException(status_code=400, detail="Invalid JSON received.")
    # --- END DEBUG LOG ---

    try:
        # Manually validate the received JSON using the Pydantic model
        request = ChatCompletionRequest.parse_obj(request_body_json)
    except Exception as e:
        # If validation fails, we now know why from the log above.
        logger.error(f"🚨 Pydantic validation failed: {e}")
        # The 422 error will be raised automatically by FastAPI here, which is what we see.
        # We are re-raising just to be explicit.
        raise HTTPException(status_code=422, detail=f"Unprocessable Entity: {e}")

    # The rest of your original function logic remains the same...
    try:
        if not model_manager:
            raise HTTPException(status_code=500, detail="Model manager not available")

        if is_api_endpoint(request.model):
            logger.info(f"[OpenAI Compat] Detected API endpoint: {request.model}")

            request_data = {
                "model": request.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "stream": request.stream,
            }

            if request.stop: request_data["stop"] = request.stop
            if request.top_k: request_data["top_k"] = request.top_k
            if request.repetition_penalty: request_data["repetition_penalty"] = request.repetition_penalty

            # Prepare endpoint config BEFORE starting stream - this ensures errors are raised
            # before the response starts, so CORS headers are properly applied
            endpoint_config, url, request_data = _prepare_endpoint_request(request.model, request_data)

            if request.stream:
                return StreamingResponse(
                    forward_to_configured_endpoint_streaming(endpoint_config, url, request_data),
                    media_type="text/event-stream"
                )
            else:
                result = await forward_to_configured_endpoint_non_streaming(endpoint_config, url, request_data)
                return result

        else:
            prompt = convert_messages_to_prompt(request.messages, request.model)
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

            if request.stream:
                inference_params = {
                    "model_name": request.model, "prompt": prompt, "max_tokens": request.max_tokens,
                    "temperature": request.temperature, "top_p": request.top_p, "top_k": request.top_k,
                    "repetition_penalty": request.repetition_penalty, "gpu_id": request.gpu_id or 0,
                    "stop_sequences": request.stop or []
                }
                return StreamingResponse(
                    stream_eloquent_to_openai(inference, model_manager, inference_params, chunk_id, request.model),
                    media_type="text/event-stream"
                )
            else:
                generated_text = await inference.generate_text(
                    model_manager=model_manager, model_name=request.model, prompt=prompt,
                    max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p,
                    top_k=request.top_k, repetition_penalty=request.repetition_penalty,
                    stop_sequences=request.stop or [], gpu_id=request.gpu_id or 0
                )
                return ChatCompletionResponse(
                    id=chunk_id, created=int(time.time()), model=request.model,
                    choices=[{ "index": 0, "message": { "role": "assistant", "content": generated_text }, "finish_reason": "stop" }],
                    usage={ "prompt_tokens": len(prompt.split()), "completion_tokens": len(generated_text.split()), "total_tokens": len(prompt.split()) + len(generated_text.split()) }
                )

    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# === Custom Endpoints Management ===
@router.post("/sync-custom-endpoints")
async def sync_custom_endpoints(request: Request, endpoints_data: dict):
    """Sync custom API endpoints from frontend settings"""
    try:
        custom_endpoints = endpoints_data.get("customApiEndpoints", [])
        
        # Store in app state so get_configured_endpoint can access it
        request.app.state.custom_api_endpoints = custom_endpoints
        
        logger.info(f"[OpenAI Compat] Synced {len(custom_endpoints)} custom endpoints")
        
        return {"status": "success", "message": f"Synced {len(custom_endpoints)} endpoints"}
    except Exception as e:
        logger.error(f"Error syncing custom endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# Optional: Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for OpenAI compatibility layer"""
    return {"status": "ok", "service": "eloquent-openai-compat"}