"""OpenAI-compatible routes for Mirid's local and configured remote models."""

import json
import os
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import logging
import httpx
import tiktoken # Added for accurate token counting
try:
    import tiktoken_ext.openai_public
    if tiktoken.registry.ENCODING_CONSTRUCTORS is None:
        tiktoken.registry.ENCODING_CONSTRUCTORS = {}
    tiktoken.registry.ENCODING_CONSTRUCTORS.update(tiktoken_ext.openai_public.ENCODING_CONSTRUCTORS)
except ImportError:
    pass
from .model_manager import ModelManager
from . import inference
from .settings_store import update_settings as update_settings_file
from pathlib import Path

logger = logging.getLogger(__name__)

# Slow remotes (e.g. nano-gpt) may buffer long before first byte; allow long gaps between chunks.
REMOTE_HTTPC_TIMEOUT = httpx.Timeout(connect=60.0, read=3600.0, write=120.0, pool=60.0)

# Short-lived in-memory endpoint health guard.
# If an endpoint repeatedly errors, we briefly cool it down so auto-rotation can move on.
_ENDPOINT_FAILURE_STATE: Dict[str, Dict[str, Any]] = {}
_ENDPOINT_FAILURE_THRESHOLD = 2
_ENDPOINT_FAILURE_WINDOW_SECONDS = 120.0
_ENDPOINT_COOLDOWN_SECONDS = 180.0
_ENDPOINT_ROTATION_STATE: Dict[str, Dict[str, str]] = {}

router = APIRouter(prefix="/v1", tags=["OpenAI Compatibility"])

MIRID_APP_URL = os.getenv("MIRID_APP_URL", "https://mirid.ai").strip() or "https://mirid.ai"
MIRID_APP_TITLE = os.getenv("MIRID_APP_TITLE", "Mirid").strip() or "Mirid"

# === Dependency to get model manager ===
def get_model_manager(request: Request) -> ModelManager:
    """Get the ModelManager instance from request app state"""
    return getattr(request.app.state, 'model_manager', None)


def get_provider_attribution_headers(endpoint_config: Optional[dict]) -> Dict[str, str]:
    """Return documented provider attribution headers without exposing user data."""
    raw_url = str((endpoint_config or {}).get("url") or "").strip()
    try:
        hostname = (urlparse(raw_url).hostname or "").lower()
    except ValueError:
        hostname = ""
    if hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai"):
        return {
            "HTTP-Referer": MIRID_APP_URL,
            "X-OpenRouter-Title": MIRID_APP_TITLE,
            "X-OpenRouter-Categories": "roleplay,general-chat",
        }
    return {}

# === OpenAI API Models ===

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: Any = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0, le=1, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(2048, ge=1, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    
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
    owned_by: str = "mirid"

# === Helper Functions ===

def strip_thinking_model_suffix(model: str) -> Tuple[str, bool]:
    """Remove Eloquent :thinking / -thinking suffix before provider or endpoint lookup."""
    m = (model or "").strip()
    if not m:
        return "", False
    low = m.lower()
    if low.endswith(":thinking"):
        return m[: -len(":thinking")].rstrip(), True
    if low.endswith("-thinking"):
        return m[: -len("-thinking")].rstrip(), True
    return m, False


def normalize_endpoint_model_id(model_name: Optional[str]) -> str:
    """Fix duplicated prefix from UI (endpoint-endpoint-* → endpoint-*)."""
    if not model_name:
        return ""
    raw = str(model_name).strip()
    raw, _ = strip_thinking_model_suffix(raw)
    # Strip human labels/comments accidentally persisted with model ids, e.g. "foo #2".
    raw = re.sub(r"\s+#\d+\s*$", "", raw).strip()
    if raw.startswith("endpoint-endpoint-"):
        return "endpoint-" + raw[len("endpoint-endpoint-") :]
    return raw


def _endpoint_match_tokens(endpoint: dict) -> set[str]:
    """Canonical matching tokens for endpoint identity checks."""
    tokens: set[str] = set()
    for value in (
        endpoint.get("id"),
        endpoint.get("name"),
        endpoint.get("model"),
    ):
        token = normalize_endpoint_model_id(value)
        if token:
            tokens.add(token)
    return tokens


def _load_custom_api_endpoints() -> List[dict]:
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        if not settings_path.exists():
            return []
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        endpoints = settings.get("customApiEndpoints", [])
        return endpoints if isinstance(endpoints, list) else []
    except Exception as e:
        logger.error("Error reading custom API endpoints from settings: %s", e)
        return []


def find_custom_api_endpoint(model_name: Optional[str]) -> tuple[Optional[dict], str]:
    """
    Match model_name to a custom API endpoint record.
    Returns (record, status) where status is 'ok' | 'disabled' | 'missing' | 'none'.
    """
    if not model_name or not isinstance(model_name, str):
        return None, "none"
    raw = normalize_endpoint_model_id(model_name)
    if not raw:
        return None, "none"

    for endpoint in _load_custom_api_endpoints():
        eid = (endpoint.get("id") or "").strip()
        ename = (endpoint.get("name") or "").strip()
        emodel = (endpoint.get("model") or "").strip()
        if raw not in (eid, ename, emodel):
            continue
        if endpoint.get("enabled", False):
            return endpoint, "ok"
        return endpoint, "disabled"

    if raw.startswith("endpoint-"):
        return None, "missing"
    return None, "none"


def resolve_api_endpoint_id(model_name: Optional[str]) -> Optional[str]:
    """Canonical enabled endpoint-* id, or None if not a configured API model."""
    record, status = find_custom_api_endpoint(model_name)
    if status == "ok" and record:
        return (record.get("id") or "").strip() or None
    return None


def validate_api_model_for_generate(model_name: Optional[str]) -> Optional[str]:
    """
    Resolve API model_name for /generate.
    Returns canonical endpoint id, None for local GGUF models, or raises HTTP 400.
    """
    record, status = find_custom_api_endpoint(model_name)
    if status == "ok" and record:
        eid = (record.get("id") or "").strip()
        if not (record.get("url") or "").strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"API endpoint '{eid or model_name}' has no provider URL. "
                    "Set URL under Settings → Custom API Endpoints."
                ),
            )
        return eid or None
    if status == "disabled" and record:
        eid = (record.get("id") or model_name or "").strip()
        raise HTTPException(
            status_code=400,
            detail=(
                f"API endpoint '{eid}' is disabled. "
                "Enable it under Settings → Custom API Endpoints."
            ),
        )
    if status == "missing":
        raw = normalize_endpoint_model_id(model_name)
        raise HTTPException(
            status_code=400,
            detail=(
                f"API endpoint '{raw}' was not found in Settings → Custom API Endpoints. "
                "Re-select your API model or add the endpoint again."
            ),
        )
    return None


def log_generate_outbound(
    url: str,
    model: str,
    endpoint_config: Optional[dict] = None,
    request_data: Optional[dict] = None,
) -> None:
    """Single-line marker immediately before provider HTTP (easy to grep in logs)."""
    name = (endpoint_config or {}).get("name") or (endpoint_config or {}).get("id") or "custom"
    thinking_payload = None
    injected = False
    if request_data:
        thinking_payload = request_data.get("thinking") or request_data.get("reasoning")
        injected = bool(thinking_payload)
    logger.info(
        "[generate] model=%s thinking_injected=%s thinking_payload=%s url=%s endpoint=%s",
        model,
        injected,
        thinking_payload,
        url,
        name,
    )


def is_api_endpoint(model_name: str) -> bool:
    """True when model_name maps to an enabled custom API endpoint."""
    if not model_name:
        return False
    if not isinstance(model_name, str):
        logger.warning(f"[is_api_endpoint] model_name is not a string: {type(model_name)} = {model_name}")
        return False
    result = resolve_api_endpoint_id(model_name) is not None
    logger.debug(f"[is_api_endpoint] Checking '{model_name}': {result}")
    return result


FLOW_API_REQUEST_PURPOSES = frozenset({
    "character_intro",
    "system_intro",
    "call_mode_character_about",
})

INTRO_ABOUT_PURPOSES = FLOW_API_REQUEST_PURPOSES | frozenset({
    "about_character",
})


def is_flow_dedicated_api_request(
    request_purpose: Optional[str],
    flow_api_url: Optional[str] = None,
) -> bool:
    """True when the client enabled Settings → Dedicated API (sends flow_api_url)."""
    return request_purpose in FLOW_API_REQUEST_PURPOSES and bool(
        (flow_api_url or "").strip()
    )


def get_configured_endpoint(
    model_id: str = None,
    *,
    rotation_candidate_ids: Optional[List[str]] = None,
    rotation_cursor_key: Optional[str] = None,
    skip_rotation: bool = False,
    request_purpose: Optional[str] = None,
    router_trace_id: Optional[str] = None,
    frontend_round_robin_enabled: Optional[bool] = None,
):
    """Read custom API endpoints from settings.json and find the specified one.

    When ``apiEndpointRoundRobinEnabled`` is on and at least two rotate-enabled
    endpoints exist, each call advances a persisted cursor. Pass
    ``rotation_candidate_ids`` (e.g. orchestrator #1/#2 selection) to limit
    rotation to that subset; ``rotation_cursor_key`` scopes the cursor (default
    ``__manual_rotation__`` for global rotation, or ``__orch_<run_id>__``).

    Flow intro / call-mode about purposes (``FLOW_API_REQUEST_PURPOSES``) never
    participate in round-robin — callers cannot override this via ``skip_rotation``.
    """
    if request_purpose in INTRO_ABOUT_PURPOSES:
        skip_rotation = True
    try:
        settings_path = Path.home() / ".LiangLocal" / "settings.json"
        if not settings_path.exists():
            return None

        with open(settings_path, 'r') as f:
            settings = json.load(f)

        custom_endpoints = settings.get('customApiEndpoints', [])
        backend_round_robin_enabled = bool(settings.get('apiEndpointRoundRobinEnabled', False))
        frontend_auto_flag = (
            bool(frontend_round_robin_enabled)
            if frontend_round_robin_enabled is not None
            else None
        )
        purpose = request_purpose or "user_chat"
        effective_round_robin_enabled = (
            bool(frontend_round_robin_enabled)
            if frontend_round_robin_enabled is not None
            else backend_round_robin_enabled
        )
        if (
            purpose == "user_chat"
            and frontend_auto_flag is not None
            and frontend_auto_flag != backend_round_robin_enabled
        ):
            logger.warning(
                "router_state_mismatch_reconciled trace_id=%s purpose=%s frontend_auto_flag=%s backend_auto_flag=%s source_of_truth=frontend_request",
                router_trace_id or "",
                purpose,
                frontend_auto_flag,
                backend_round_robin_enabled,
            )
        route_mode = "manual"
        selected_in_candidates = None
        correction_action = None
        override_ignored = False
        override_reason = None

        def _fmt(endpoint: dict) -> dict:
            native_flag = endpoint.get('supports_native_search')
            if native_flag is None:
                native_flag = endpoint.get('supportsNativeSearch')
            return {
                'id': endpoint.get('id', ''),
                'url': endpoint.get('url', '').rstrip('/'),
                'api_key': endpoint.get('apiKey', ''),
                'name': endpoint.get('name', 'Custom Endpoint'),
                'model': endpoint.get('model', ''),  # Model name to send to the API
                'context_window': endpoint.get('context_window') or endpoint.get('contextWindow'),
                'supports_native_search': native_flag,
                '_routing_mode': route_mode,
            }

        def _persist_round_robin_cursor(updated_settings: dict):
            try:
                update_settings_file({
                    "apiEndpointRoundRobinCursor": updated_settings.get(
                        "apiEndpointRoundRobinCursor",
                        {},
                    )
                })
            except Exception as write_err:
                logger.warning(f"Failed to persist API round-robin cursor: {write_err}")

        def _rotate_among(candidates: list, cursor_key: str):
            if len(candidates) < 2:
                return None
            cursor_map = settings.get('apiEndpointRoundRobinCursor') or {}
            if not isinstance(cursor_map, dict):
                cursor_map = {}
            idx_raw = cursor_map.get(cursor_key, 0)
            try:
                idx = int(idx_raw)
            except Exception:
                idx = 0
            idx = idx % len(candidates)
            candidate_ids = [str(ep.get("id") or "").strip() for ep in candidates]
            pool_signature = "|".join(candidate_ids)
            last_state = _ENDPOINT_ROTATION_STATE.get(cursor_key) or {}
            last_selected = str(last_state.get("last_selected") or "")
            chosen_idx = idx
            next_selected_reason = "cursor"
            # Guarantee no repeated endpoint when 2+ healthy candidates are available.
            if (
                last_selected
                and candidate_ids[chosen_idx] == last_selected
                and len(candidates) >= 2
            ):
                chosen_idx = (chosen_idx + 1) % len(candidates)
                next_selected_reason = "avoid_repeat"
            chosen = candidates[chosen_idx]
            cursor_map[cursor_key] = (chosen_idx + 1) % len(candidates)
            settings['apiEndpointRoundRobinCursor'] = cursor_map
            _persist_round_robin_cursor(settings)
            _ENDPOINT_ROTATION_STATE[cursor_key] = {
                "last_selected": str(chosen.get("id") or ""),
                "pool_signature": pool_signature,
            }
            return _fmt(chosen), {
                "last_selected": last_selected,
                "next_selected_reason": next_selected_reason,
            }

        def _is_endpoint_healthy(endpoint: dict) -> bool:
            endpoint_id = str(endpoint.get("id") or "").strip()
            if not endpoint_id:
                return True
            state = _ENDPOINT_FAILURE_STATE.get(endpoint_id) or {}
            cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
            return cooldown_until <= time.monotonic()

        def _healthy_candidates(candidates: List[dict]) -> List[dict]:
            healthy = [ep for ep in candidates if _is_endpoint_healthy(ep)]
            return healthy or candidates

        def _build_rotation_candidates() -> List[dict]:
            id_filter = {normalize_endpoint_model_id(v) for v in (rotation_candidate_ids or []) if normalize_endpoint_model_id(v)}
            return [
                ep for ep in custom_endpoints
                if ep.get('enabled', False)
                and ep.get('rotate_enabled', True)
                and (not id_filter or normalize_endpoint_model_id(ep.get("id")) in id_filter)
            ]

        def _log_router_decision(
            *,
            selected_endpoint: Optional[dict],
            candidates: Optional[List[dict]] = None,
            last_selected: str = "",
            next_selected_reason: str = "",
        ) -> None:
            nonlocal selected_in_candidates, correction_action, route_mode, override_ignored, override_reason
            candidate_ids = [
                normalize_endpoint_model_id(ep.get("id"))
                for ep in (candidates or [])
                if normalize_endpoint_model_id(ep.get("id"))
            ]
            selected_id = normalize_endpoint_model_id((selected_endpoint or {}).get("id"))
            if candidates is None:
                selected_in_candidates = bool(selected_id)
            else:
                selected_in_candidates = selected_id in set(candidate_ids)
            auto_enabled = bool(effective_round_robin_enabled and not skip_rotation)
            msg = (
                "[API Router] decision trace_id=%s mode=%s purpose=%s auto_enabled=%s candidate_count=%d "
                "candidate_ids=%s selected_endpoint=%s last_selected=%s next_selected_reason=%s "
                "selected_in_candidates=%s override_ignored=%s override_reason=%s"
            )
            logger.info(
                msg,
                router_trace_id or "",
                route_mode,
                purpose,
                auto_enabled,
                len(candidate_ids),
                candidate_ids,
                selected_id or "",
                normalize_endpoint_model_id(last_selected),
                next_selected_reason or "",
                selected_in_candidates,
                override_ignored,
                override_reason or "",
            )
            if candidates is not None and not selected_in_candidates and correction_action:
                logger.warning(
                    "[API Router] correction action=%s selected_endpoint=%s candidates=%s",
                    correction_action,
                    selected_id or "",
                    candidate_ids,
                )

        # If a model_id is provided, find that specific endpoint (id, display name, or provider model)
        if model_id:
            canonical_id = resolve_api_endpoint_id(model_id)
            lookup_id = canonical_id or normalize_endpoint_model_id(model_id)
            requested_endpoint = None
            for endpoint in custom_endpoints:
                if not endpoint.get('enabled', False):
                    continue
                if lookup_id and lookup_id in _endpoint_match_tokens(endpoint):
                    requested_endpoint = endpoint
                    break

            if effective_round_robin_enabled and not skip_rotation:
                route_mode = "auto"
                candidates = _build_rotation_candidates()
                candidate_ids = {
                    normalize_endpoint_model_id(ep.get("id"))
                    for ep in candidates
                    if normalize_endpoint_model_id(ep.get("id"))
                }
                # user_chat contract: explicit endpoint selection is never honored while auto-routing.
                if purpose == "user_chat" and requested_endpoint:
                    override_ignored = True
                    req_id = normalize_endpoint_model_id(requested_endpoint.get("id"))
                    if req_id and req_id not in candidate_ids:
                        override_reason = "requested_endpoint_not_in_rotate_candidates"
                    else:
                        override_reason = "manual_override_disallowed_for_user_chat_auto"
                if not candidates:
                    if requested_endpoint:
                        route_mode = "manual_fallback"
                        correction_action = "empty_pool_used_requested_endpoint"
                        override_ignored = False
                        override_reason = None
                        selected = _fmt(requested_endpoint)
                        _log_router_decision(selected_endpoint=selected, candidates=None)
                        return selected
                    correction_action = "no_eligible_rotate_candidates"
                    _log_router_decision(selected_endpoint=None, candidates=[])
                    if purpose == "user_chat":
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "Auto API routing has no included endpoints and no individual model was selected. "
                                "Select a model or include an endpoint in rotation."
                            ),
                        )
                    return None

                chosen_pool = _healthy_candidates(candidates)
                cursor_key = rotation_cursor_key or "__manual_rotation__"
                if len(chosen_pool) >= 2:
                    rotated = _rotate_among(chosen_pool, cursor_key)
                    if rotated:
                        selected_endpoint, rotate_meta = rotated
                        # Strict invariant: selected endpoint must be from eligible candidates.
                        # If explicit requested endpoint was outside auto list, we corrected to rotated candidate.
                        if requested_endpoint and not override_ignored:
                            req_id = normalize_endpoint_model_id(requested_endpoint.get("id"))
                            if req_id not in candidate_ids:
                                correction_action = "replaced_out_of_list_requested_endpoint"
                        _log_router_decision(
                            selected_endpoint=selected_endpoint,
                            candidates=candidates,
                            last_selected=rotate_meta.get("last_selected", ""),
                            next_selected_reason=rotate_meta.get("next_selected_reason", ""),
                        )
                        return selected_endpoint
                # Single candidate pool: choose it deterministically.
                selected = _fmt(chosen_pool[0])
                selected_raw_id = str(chosen_pool[0].get("id") or "")
                last_single = ""
                if len(candidates) >= 2:
                    last_state = _ENDPOINT_ROTATION_STATE.get(cursor_key) or {}
                    last_single = str(last_state.get("last_selected") or "")
                    _ENDPOINT_ROTATION_STATE[cursor_key] = {
                        "last_selected": selected_raw_id,
                        "pool_signature": "|".join(str(ep.get("id") or "") for ep in chosen_pool),
                    }
                if requested_endpoint and not override_ignored:
                    req_id = normalize_endpoint_model_id(requested_endpoint.get("id"))
                    if req_id not in candidate_ids:
                        correction_action = "replaced_out_of_list_requested_endpoint"
                _log_router_decision(
                    selected_endpoint=selected,
                    candidates=candidates,
                    last_selected=last_single,
                    next_selected_reason="single_healthy_candidate",
                )
                return selected

            if requested_endpoint:
                route_mode = "pinned" if skip_rotation else "manual"
                selected = _fmt(requested_endpoint)
                _log_router_decision(selected_endpoint=selected, candidates=None)
                return selected
            # If the specific endpoint is not found or disabled, it's an error
            return None

        # Fallback for old behavior: find the first enabled endpoint
        if effective_round_robin_enabled and not skip_rotation:
            route_mode = "auto"
            candidates = _build_rotation_candidates()
            if not candidates:
                correction_action = "no_eligible_rotate_candidates"
                _log_router_decision(selected_endpoint=None, candidates=[])
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Auto API routing is enabled for user_chat but no eligible rotate candidates exist "
                        "(enabled && rotate_enabled). Enable at least one rotate-enabled endpoint."
                    ),
                )
            chosen_pool = _healthy_candidates(candidates)
            cursor_key = rotation_cursor_key or "__manual_rotation__"
            if len(chosen_pool) >= 2:
                rotated = _rotate_among(chosen_pool, cursor_key)
                if rotated:
                    selected_endpoint, rotate_meta = rotated
                    _log_router_decision(
                        selected_endpoint=selected_endpoint,
                        candidates=candidates,
                        last_selected=rotate_meta.get("last_selected", ""),
                        next_selected_reason=rotate_meta.get("next_selected_reason", ""),
                    )
                    return selected_endpoint
            selected = _fmt(chosen_pool[0])
            _log_router_decision(
                selected_endpoint=selected,
                candidates=candidates,
                next_selected_reason="single_healthy_candidate",
            )
            return selected

        for endpoint in custom_endpoints:
            if endpoint.get('enabled', False):
                selected = _fmt(endpoint)
                route_mode = "manual"
                _log_router_decision(selected_endpoint=selected, candidates=None)
                return selected
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading settings: {e}")
        return None


def note_endpoint_failure(endpoint_id: Optional[str], reason: str = "upstream_error") -> None:
    """Record endpoint failure; repeated failures trigger temporary cooldown."""
    endpoint = str(endpoint_id or "").strip()
    if not endpoint:
        return
    now = time.monotonic()
    state = _ENDPOINT_FAILURE_STATE.get(endpoint) or {}
    last_at = float(state.get("last_failure_at", 0.0) or 0.0)
    if now - last_at > _ENDPOINT_FAILURE_WINDOW_SECONDS:
        state["failures"] = 0
    failures = int(state.get("failures", 0) or 0) + 1
    state["failures"] = failures
    state["last_failure_at"] = now
    if failures >= _ENDPOINT_FAILURE_THRESHOLD:
        state["cooldown_until"] = now + _ENDPOINT_COOLDOWN_SECONDS
    _ENDPOINT_FAILURE_STATE[endpoint] = state
    logger.warning(
        "[API Router] mark_failure endpoint=%s failures=%d cooldown_until=%s reason=%s",
        endpoint,
        failures,
        round(float(state.get("cooldown_until", 0.0) or 0.0), 2),
        reason,
    )


def note_endpoint_success(endpoint_id: Optional[str]) -> None:
    endpoint = str(endpoint_id or "").strip()
    if not endpoint:
        return
    if endpoint in _ENDPOINT_FAILURE_STATE:
        _ENDPOINT_FAILURE_STATE.pop(endpoint, None)
        logger.info("[API Router] clear_failure endpoint=%s", endpoint)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Return the number of tokens used by a list of messages.
    Adapted from OpenAI cookbook."""
    encoding = None
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for newer models if unknown
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as enc_error:
            # In a frozen/broken environment tiktoken may fail to discover its
            # encoding plugins entirely. Never let token counting kill a
            # request; fall back to a rough chars/4 heuristic below.
            logger.warning(
                "tiktoken unavailable (%s); using heuristic token estimate", enc_error
            )
    except Exception as enc_error:
        logger.warning(
            "tiktoken unavailable (%s); using heuristic token estimate", enc_error
        )

    tokens_per_message = 3 # every message follows <|start|>{role/name}\n{content}\n
    tokens_per_name = 1
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            text = str(value)
            if encoding is not None:
                num_tokens += len(encoding.encode(text))
            else:
                # Heuristic: ~4 characters per token (OpenAI rule of thumb).
                num_tokens += max(1, len(text) // 4)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def prune_messages(messages: List[dict], max_input_tokens: int, model_name: str = "gpt-3.5-turbo"):
    """
    Prunes the message history to fit within max_input_tokens.
    Strategies:
    1. ALWAYS keep the System message (if present at index 0).
    2. ALWAYS keep the newest user message verbatim (or the final message if no user exists).
    3. Remove messages from the beginning of the history (after system) until it fits.
    """
    if not messages:
        return [], 0, 0
        
    current_tokens = num_tokens_from_messages(messages, model_name)
    original_tokens = current_tokens
    
    # If we are already under the limit, return early!
    if current_tokens <= max_input_tokens:
        return messages, original_tokens, current_tokens
        
    # Identification
    has_system = messages[0]['role'] == 'system'
    system_msg = messages[0] if has_system else None
    
    # Preserve the newest user message verbatim when present.
    # Fallback to the very last message if there is no user turn.
    last_user_index = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_index = i
            break
    pinned_index = last_user_index if last_user_index is not None else len(messages) - 1
    last_msg = messages[pinned_index]
    
    # Build prunable indices in chronological order.
    prunable_indices = []
    for idx in range(len(messages)):
        if has_system and idx == 0:
            continue
        if idx == pinned_index:
            continue
        prunable_indices.append(idx)

    dropped = set()

    while prunable_indices:
        candidate_history = [msg for idx, msg in enumerate(messages) if idx not in dropped]
        
        token_count = num_tokens_from_messages(candidate_history, model_name)
        
        if token_count <= max_input_tokens:
            candidate_history = _remove_orphaned_tool_messages(candidate_history)
            return candidate_history, original_tokens, token_count
            
        # If still too big, drop the oldest remaining prunable message.
        removed_idx = prunable_indices.pop(0)
        dropped.add(removed_idx)

        # If we dropped an assistant message with tool_calls, also remove any
        # tool result messages referencing those tool_call_ids to avoid orphaned
        # tool messages that would cause API errors.
        removed_msg = messages[removed_idx]
        if removed_msg.get("tool_calls"):
            tool_call_ids = {tc.get("id") for tc in removed_msg["tool_calls"] if tc.get("id")}
            for pi in list(prunable_indices):
                msg = messages[pi]
                if msg.get("role") == "tool" and msg.get("tool_call_id") in tool_call_ids:
                    dropped.add(pi)
                    prunable_indices.remove(pi)
        
    # Worst case: only system + last message
    final_fallback = []
    if system_msg:
        final_fallback.append(system_msg)
    final_fallback.append(last_msg)
    
    # Ensure no orphaned tool messages in fallback
    final_fallback = _remove_orphaned_tool_messages(final_fallback)
    
    final_count = num_tokens_from_messages(final_fallback, model_name)
    
    # NEW: Safety check - if system + last message still exceeds the limit, 
    # we MUST truncate the system message.
    if final_count > max_input_tokens and has_system:
        logger.warning(f"[OpenAI Compat] System message + last prompt ({final_count}) still exceeds limit ({max_input_tokens}). Truncating system message.")
        
        # Calculate how much we need to cut from the system message
        last_msg_tokens = num_tokens_from_messages([last_msg], model_name)
        allowed_system_tokens = max_input_tokens - last_msg_tokens - 50 # padding
        
        if allowed_system_tokens > 100:
            encoding = None
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                except Exception as enc_error:
                    logger.warning(
                        "tiktoken unavailable (%s); truncating system message by characters", enc_error
                    )

            if encoding is not None:
                system_tokens = encoding.encode(system_msg['content'])
                # Truncate from the middle of the system prompt (often lore/tracker info is there)
                half = allowed_system_tokens // 2
                truncated_content = (
                    encoding.decode(system_tokens[:half]) +
                    "\n... [Truncated for Context] ...\n" +
                    encoding.decode(system_tokens[-(half):])
                )
            else:
                # Heuristic character-based truncation (~4 chars per token).
                content = system_msg['content']
                half_chars = (allowed_system_tokens * 4) // 2
                truncated_content = (
                    content[:half_chars] +
                    "\n... [Truncated for Context] ...\n" +
                    content[-half_chars:]
                )
            final_fallback[0]['content'] = truncated_content
            final_count = num_tokens_from_messages(final_fallback, model_name)
            logger.info(f"[OpenAI Compat] System message truncated. New total: {final_count}")

    return final_fallback, original_tokens, final_count


def _remove_orphaned_tool_messages(messages):
    """Remove tool result messages that have no matching assistant tool_calls entry."""
    assistant_tool_call_ids = set()
    result = []
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id"):
                    assistant_tool_call_ids.add(tc["id"])
        if msg.get("role") == "tool" and msg.get("tool_call_id") and msg["tool_call_id"] not in assistant_tool_call_ids:
            continue
        result.append(msg)
    return result


def parse_eloquent_llm_prompt_to_openai_messages(llm_prompt: str) -> List[Dict[str, Any]]:
    """
    Convert the assembled /generate prompt string into OpenAI-style chat messages.
    Used for custom API endpoints (OpenAI-compatible), including Moonshot / Kimi and NanoGPT proxies.
    """
    messages: List[Dict[str, Any]] = []
    segments = re.findall(
        r"<start_of_turn>(user|model)\n(.*?)(?:<end_of_turn>|$)", llm_prompt, re.DOTALL
    )
    if segments:
        for role, content in segments:
            messages.append(
                {
                    "role": "assistant" if role == "model" else "user",
                    "content": content.strip(),
                }
            )
        system_part = llm_prompt.split("<start_of_turn>")[0].strip()
        if system_part:
            messages.insert(0, {"role": "system", "content": system_part})
        return messages

    if "Character Persona:" in llm_prompt:
        parts = llm_prompt.split("Character Persona:", 1)
        if parts[0].strip():
            messages.append({"role": "system", "content": parts[0].strip()})
        if len(parts) > 1:
            persona_and_user = parts[1]
            if "User Query:" in persona_and_user:
                persona, user_query = persona_and_user.split("User Query:", 1)
                if persona.strip():
                    messages.append(
                        {"role": "system", "content": f"Character Persona:\n{persona.strip()}"}
                    )
                messages.append({"role": "user", "content": user_query.strip()})
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": persona_and_user.replace("Assistant:", "").strip(),
                    }
                )
        return messages

    if "User Query:" in llm_prompt:
        parts = llm_prompt.split("User Query:", 1)
        if parts[0].strip():
            messages.append({"role": "system", "content": parts[0].strip()})
        messages.append({"role": "user", "content": parts[1].replace("Assistant:", "").strip()})
        return messages

    clean_prompt = llm_prompt.replace("Assistant:", "").strip()
    messages.append({"role": "user", "content": clean_prompt})
    return messages


def build_vision_data_url(image_base64: str, image_type: Optional[str] = None) -> str:
    """Return a data: URI suitable for OpenAI-style image_url (Kimi / Moonshot / NanoGPT)."""
    raw = (image_base64 or "").strip()
    if raw.startswith("data:"):
        return raw
    it = (image_type or "png").strip().lower().removeprefix("image/")
    if it == "jpg":
        it = "jpeg"
    if it not in ("jpeg", "png", "webp", "gif"):
        it = "png"
    mime = f"image/{it}"
    return f"data:{mime};base64,{raw}"


def inject_openai_vision_into_messages(
    messages: List[Dict[str, Any]],
    image_base64: str,
    image_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Attach a vision image to the last user turn using OpenAI multimodal content parts.
    Moonshot Kimi and other OpenAI-compatible vision APIs expect this shape.
    """
    if not image_base64:
        return list(messages)
    out: List[Dict[str, Any]] = []
    for m in messages:
        out.append(dict(m))
    url = build_vision_data_url(image_base64, image_type)
    image_part = {"type": "image_url", "image_url": {"url": url}}

    last_user = -1
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            last_user = i
            break

    if last_user < 0:
        out.append({"role": "user", "content": [{"type": "text", "text": ""}, image_part]})
        return out

    c = out[last_user].get("content")
    if isinstance(c, str):
        out[last_user]["content"] = [{"type": "text", "text": c}, image_part]
    elif isinstance(c, list):
        new_parts: List[Dict[str, Any]] = []
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "image_url":
                existing = (p.get("image_url") or {}).get("url")
                if existing == url:
                    return out
            new_parts.append(dict(p))
        new_parts.append(image_part)
        out[last_user]["content"] = new_parts
    else:
        out[last_user]["content"] = [{"type": "text", "text": str(c)}, image_part]
    return out


def approx_openai_messages_payload_chars(messages: List[Dict[str, Any]]) -> int:
    """Rough character count for logging (supports string or multimodal list content)."""
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            total += sum(len(str(p)) for p in c)
        else:
            total += len(str(c))
    return total


# Provider-native web search is applied in web_search_routing.apply_native_web_search_request()
# from main.py when chat web search path resolves to "native".


def apply_nano_gpt_context_memory(
    endpoint_config: dict,
    request_data: dict,
    *,
    enabled: bool,
    mode: str,
    expiration_days: int,
) -> Dict[str, str]:
    """
    NanoGPT Context Memory (docs.nano-gpt.com): header `memory: true` or model suffix `:memory-<days>`.
    Only applies when the configured endpoint URL is nano-gpt.com.
    Mutates request_data['model'] when mode is suffix.
    Returns extra HTTP headers to merge for header mode.
    """
    url = (endpoint_config.get("url") or "").lower()
    if not enabled or "nano-gpt.com" not in url:
        return {}
    try:
        days = max(1, min(365, int(expiration_days)))
    except (TypeError, ValueError):
        days = 30
    mode_n = (mode or "header").strip().lower()
    if mode_n == "suffix":
        m = request_data.get("model") or ""
        if ":memory" in m:
            return {}
        request_data["model"] = f"{m}:memory-{days}"
        return {}
    return {"memory": "true", "memory_expiration_days": str(days)}


def thinking_stream_debug_enabled(model: str) -> bool:
    """Gated dev logging for :thinking models or ELOQUENT_THINKING_STREAM_DEBUG=1."""
    flag = os.environ.get("ELOQUENT_THINKING_STREAM_DEBUG", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    return ":thinking" in (model or "").lower()


def model_id_implies_extended_thinking(model: str) -> bool:
    """True only when the requested model explicitly asks us to enable provider thinking."""
    m = (model or "").lower()
    if not m:
        return False
    if ":thinking" in m or m.endswith("-thinking"):
        return True
    return False


def _thinking_budget_tokens(request_data: dict) -> int:
    """NanoGPT: budget_tokens must be >= 1024 and < max_tokens."""
    try:
        max_tokens = int(request_data.get("max_tokens") or 4096)
    except (TypeError, ValueError):
        max_tokens = 4096
    budget = min(8192, max(1024, max_tokens // 2))
    if budget >= max_tokens:
        budget = max(1024, max_tokens - 512)
    if budget >= max_tokens:
        budget = 1024
    return budget


def apply_extended_thinking_request(
    endpoint_config: dict,
    request_data: dict,
    *,
    force_thinking: bool = False,
) -> bool:
    """
    Enable provider reasoning streams for thinking-capable models.

    NanoGPT (docs.nano-gpt.com extended thinking): ``thinking: { type, budget_tokens }``.
    OpenRouter GLM/Z.ai (reasoning tokens guide): ``reasoning: { enabled: true, max_tokens }``.
    Skips when caller already set ``thinking`` or ``reasoning``.
    Returns True when thinking/reasoning is present on the outbound body.
    """
    if request_data.get("thinking") or request_data.get("reasoning"):
        return True
    model = (request_data.get("model") or "").strip()
    url = (endpoint_config.get("url") or "").lower()
    if not force_thinking and not model_id_implies_extended_thinking(model):
        if "nano-gpt.com" in url or "nanogpt" in url or "openrouter.ai" in url:
            request_data.setdefault("reasoning_effort", "none")
        return False

    budget = _thinking_budget_tokens(request_data)

    if "nano-gpt.com" in url or "nanogpt" in url:
        request_data["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if thinking_stream_debug_enabled(model):
            logger.info(
                "[OpenAI Compat] extended thinking enabled (NanoGPT) model=%s budget_tokens=%s",
                model,
                budget,
            )
        return True

    if "openrouter.ai" in url:
        request_data["reasoning"] = {"enabled": True, "max_tokens": budget}
        if thinking_stream_debug_enabled(model):
            logger.info(
                "[OpenAI Compat] extended thinking enabled (OpenRouter) model=%s max_tokens=%s",
                model,
                budget,
            )
        return True

    request_data["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if thinking_stream_debug_enabled(model):
        logger.info(
            "[OpenAI Compat] extended thinking enabled (generic thinking=) model=%s budget_tokens=%s",
            model,
            budget,
        )
    return True


def _prepare_outbound_provider_model(
    model_name: str,
    request_data: dict,
    endpoint_config: dict,
) -> bool:
    """
    Resolve endpoint-* → configured provider model; strip :thinking suffix for upstream.
    Returns whether extended thinking should be forced on the outbound request.
    """
    force_thinking = bool(request_data.pop("_force_extended_thinking", False))
    force_thinking = force_thinking or model_id_implies_extended_thinking(model_name) or model_id_implies_extended_thinking(
        request_data.get("model", "")
    )
    url = (endpoint_config.get("url") or "").lower()
    preserve_thinking_suffix = "nano-gpt.com" in url or "nanogpt" in url

    req_model = (request_data.get("model") or "").strip()
    if req_model.startswith("endpoint-") or not req_model:
        configured_model = (endpoint_config.get("model") or "").strip()
        if configured_model:
            if preserve_thinking_suffix:
                request_data["model"] = configured_model
                _, had_suffix = strip_thinking_model_suffix(configured_model)
                if had_suffix:
                    force_thinking = False
            else:
                provider_model, had_suffix = strip_thinking_model_suffix(configured_model)
                request_data["model"] = provider_model or configured_model
                force_thinking = force_thinking or had_suffix
        elif not req_model:
            request_data["model"] = "gpt-3.5-turbo"

    if preserve_thinking_suffix:
        _, had_suffix = strip_thinking_model_suffix(request_data.get("model", ""))
        if had_suffix:
            return False

    provider_model, had_suffix = strip_thinking_model_suffix(request_data.get("model", ""))
    if had_suffix:
        request_data["model"] = provider_model
        force_thinking = True

    return force_thinking


_REASONING_FIELD_NAMES = (
    "reasoning", "reasoning_content", "thinking", "reasoning_text",
    "reason", "think", "internal_monologue", "chain_of_thought",
    "thought", "thought_process",
)


def _extract_reasoning_from_dict(d: dict) -> str:
    """
    Scan a dict for reasoning/thinking content under known field names only.

    Providers sometimes stamp serving metadata (deployment ids, fingerprints)
    onto stream chunks, so unknown fields are never treated as reasoning.
    """
    if not isinstance(d, dict):
        return ""

    for field in _REASONING_FIELD_NAMES:
        val = d.get(field)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list) and val:
            parts: List[str] = []
            for item in val:
                if isinstance(item, dict):
                    piece = item.get("text") or item.get("content") or ""
                    if piece:
                        parts.append(str(piece))
                elif isinstance(item, str) and item:
                    parts.append(item)
            if parts:
                return "".join(parts)

    return ""


class ThinkingStreamParser:
    """
    Stateful parser for thinking model streams.
    Handles <think>...</think> tags and untagged reasoning.
    """
    
    def __init__(self):
        self.buffer = ""
        self.in_think_block = False
        self.reasoning_collected = []
        self.content_collected = []
        self.think_tag_detected = False
    
    def process_chunk(self, chunk_data: dict) -> Tuple[str, str]:
        """
        Process a single chunk and return (content, reasoning).
        Maintains state across chunks.
        """
        if not chunk_data or not isinstance(chunk_data, dict):
            return "", ""
        
        # Extract raw content from chunk
        choices = chunk_data.get("choices")
        if not choices or not isinstance(choices, list):
            return "", ""
        first_choice = choices[0] if isinstance(choices[0], dict) else None
        if not isinstance(first_choice, dict):
            return "", ""
        
        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            return "", ""
        
        token = str(delta.get("content") or "")
        if not token:
            return "", ""
        
        # Add token to buffer
        self.buffer += token
        
        content_out = ""
        reasoning_out = ""
        
        if not self.in_think_block:
            # Check for <think> tag
            if "<think>" in self.buffer:
                self.think_tag_detected = True
                pre, _, post = self.buffer.partition("<think>")
                if pre:
                    content_out = pre
                self.buffer = post
                self.in_think_block = True
            elif len(self.buffer) > 7:
                # Safe to yield if we know a <think> tag isn't partially forming
                content_out = self.buffer[:-7]
                self.buffer = self.buffer[-7:]
        else:
            # Inside think block, look for </think>
            if "</think>" in self.buffer:
                _, _, post = self.buffer.partition("</think>")
                reasoning_out = self.buffer[:-9]  # Everything before </think>
                self.buffer = post
                self.in_think_block = False
            else:
                # Keep buffering, but yield reasoning if buffer is large
                if len(self.buffer) > 100:
                    reasoning_out = self.buffer[:-8]
                    self.buffer = self.buffer[-8:]
        
        return content_out, reasoning_out
    
    def flush(self) -> Tuple[str, str]:
        """Flush any remaining buffered content."""
        content_out = ""
        reasoning_out = ""
        
        if self.buffer:
            if self.in_think_block:
                reasoning_out = self.buffer
            else:
                content_out = self.buffer
            self.buffer = ""
        
        return content_out, reasoning_out


def extract_openai_stream_delta_parts(chunk_data: dict) -> Tuple[str, str]:
    """
    Split upstream OpenAI-style stream JSON into visible content vs reasoning.

    Supports NanoGPT ``delta.reasoning``, ``delta.thinking``, legacy
    ``reasoning_content``, OpenRouter ``delta.reasoning_details[]``, top-level
    ``reasoning``/``text``, and ``choices[0].message.reasoning``.
    """
    if not chunk_data or not isinstance(chunk_data, dict):
        return "", ""

    # ── Top-level fields (NanoGPT, some proxies emit flat JSON) ──
    top_text = chunk_data.get("text")
    top_reasoning = _extract_reasoning_from_dict(chunk_data)
    if top_text is not None:
        return str(top_text or ""), top_reasoning

    # If top-level has reasoning but no text, still capture it
    if top_reasoning:
        return "", top_reasoning

    # ── choices-based format (OpenAI, OpenRouter, etc.) ──
    choices = chunk_data.get("choices")
    if not choices or not isinstance(choices, list):
        return "", ""
    first_choice = choices[0] if isinstance(choices[0], dict) else None
    if not isinstance(first_choice, dict):
        return "", ""

    choice_reasoning = _extract_reasoning_from_dict(first_choice)

    # ── delta format (streaming) ──
    delta = first_choice.get("delta")
    if isinstance(delta, dict):
        content = str(delta.get("content") or "")
        reasoning = _extract_reasoning_from_dict(delta) or choice_reasoning
        # Also check reasoning_details array (OpenRouter)
        if not reasoning:
            details = delta.get("reasoning_details")
            if isinstance(details, list):
                parts2: List[str] = []
                for item in details:
                    if not isinstance(item, dict):
                        continue
                    piece = item.get("text") or item.get("content") or ""
                    if piece:
                        parts2.append(str(piece))
                reasoning = "".join(parts2)
        if content or reasoning:
            return content, reasoning

    if choice_reasoning:
        return "", choice_reasoning

    # ── message format (non-delta / chat-completion shape) ──
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        reasoning = _extract_reasoning_from_dict(message)
        if isinstance(content, str) and content:
            return content, reasoning
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        text_parts.append(str(text))
            if text_parts:
                return "".join(text_parts), reasoning
        # content may be None but reasoning present
        if reasoning:
            return "", reasoning

    # ── legacy text field ──
    choice_text = first_choice.get("text")
    if choice_text:
        return str(choice_text), ""

    return "", ""


def resolve_flow_api_endpoint_config(
    *,
    request_purpose: Optional[str],
    model_name: str = "",
    flow_api_url: Optional[str] = None,
    flow_api_model: Optional[str] = None,
    flow_api_key: Optional[str] = None,
) -> Optional[dict]:
    """Dedicated intro/about API: use only request-body provider fields (no settings lookup)."""
    if not is_flow_dedicated_api_request(request_purpose, flow_api_url):
        return None

    url = (flow_api_url or "").strip().rstrip("/")
    if not url:
        return None

    model = (flow_api_model or "").strip() or "gpt-3.5-turbo"
    key = (flow_api_key or "").strip()
    return {
        "url": url,
        "model": model,
        "api_key": key,
        "name": f"flow-{request_purpose}",
    }


def _build_nanogpt_url(base_url: str, request_data: dict, endpoint_config: dict = None) -> str:
    """Build NanoGPT URL - use /api/v1thinking/ for thinking models to merge reasoning into content."""
    is_nanogpt = "nano-gpt.com" in base_url.lower() or "nanogpt" in base_url.lower()
    is_subscription_route = "/api/subscription/v1" in base_url.lower()
    # Check both request_data model and endpoint_config model
    model_name = request_data.get("model", "") or (endpoint_config.get("model", "") if endpoint_config else "")
    is_thinking_model = model_id_implies_extended_thinking(model_name) or request_data.get("thinking")
    
    if is_nanogpt and is_thinking_model and not is_subscription_route:
        # Use /api/v1thinking/ endpoint - reasoning merges into content stream
        if base_url.endswith("/v1"):
            return f"{base_url}thinking/chat/completions"
        elif "/api/v1" in base_url:
            return base_url.replace("/api/v1", "/api/v1thinking") + "/chat/completions"
        else:
            return f"{base_url}/api/v1thinking/chat/completions"
    
    # Standard endpoint
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    else:
        return f"{base_url}/v1/chat/completions"


def prepare_endpoint_request_from_config(endpoint_config: dict, request_data: dict, label: str = "custom"):
    """Like prepare_endpoint_request but uses a pre-built endpoint_config dict (flow overrides)."""
    if not endpoint_config:
        raise HTTPException(status_code=400, detail="No endpoint configuration for API request.")

    base_url = endpoint_config["url"]
    url = _build_nanogpt_url(base_url, request_data, endpoint_config)

    force_thinking = _prepare_outbound_provider_model("", request_data, endpoint_config)
    if request_data.pop("_skip_openai_message_pruning", False):
        logger.info(
            "[OpenAI Compat] skip_openai_message_pruning (%s): %s messages",
            label,
            len(request_data.get("messages", [])),
        )
        request_data["max_tokens"] = min(request_data.get("max_tokens", 16384) or 16384, 32768)
        apply_extended_thinking_request(endpoint_config, request_data, force_thinking=force_thinking)
        log_generate_outbound(url, request_data.get("model", ""), endpoint_config, request_data)
        return endpoint_config, url, request_data

    default_limit = 1_000_000
    configured_limit = endpoint_config.get("context_window")
    if configured_limit:
        try:
            CONTEXT_WINDOW_LIMIT = int(configured_limit)
        except (ValueError, TypeError):
            CONTEXT_WINDOW_LIMIT = default_limit
    else:
        CONTEXT_WINDOW_LIMIT = default_limit

    SAFETY_MARGIN = 1000
    requested_gen_tokens = min(request_data.get("max_tokens", 16384) or 16384, 32768)
    request_data["max_tokens"] = requested_gen_tokens
    max_input_tokens = CONTEXT_WINDOW_LIMIT - requested_gen_tokens - SAFETY_MARGIN
    if max_input_tokens < 1000:
        max_input_tokens = 1000

    messages = request_data.get("messages", [])
    pruned_messages, original_count, new_count = prune_messages(
        messages,
        max_input_tokens=max_input_tokens,
        model_name=request_data["model"],
    )
    request_data["messages"] = pruned_messages
    if original_count > new_count:
        logger.warning(
            "[OpenAI Compat] Flow %s pruned context %s → %s tokens",
            label,
            original_count,
            new_count,
        )
    logger.info("[OpenAI Compat] Forwarding flow %s to %s at %s", label, endpoint_config.get("name"), url)
    apply_extended_thinking_request(endpoint_config, request_data, force_thinking=force_thinking)
    log_generate_outbound(url, request_data.get("model", ""), endpoint_config, request_data)
    return endpoint_config, url, request_data


def prepare_endpoint_request(
    model_name: str,
    request_data: dict,
    *,
    skip_rotation: bool = False,
    request_purpose: Optional[str] = None,
    router_trace_id: Optional[str] = None,
    frontend_round_robin_enabled: Optional[bool] = None,
):
    """Prepare endpoint config and URL - returns (endpoint_config, url, request_data) or raises HTTPException"""
    endpoint_config = get_configured_endpoint(
        model_name,
        skip_rotation=skip_rotation,
        request_purpose=request_purpose,
        router_trace_id=router_trace_id,
        frontend_round_robin_enabled=frontend_round_robin_enabled,
    )
    
    if not endpoint_config:
        raw = normalize_endpoint_model_id(model_name)
        if raw.startswith("endpoint-"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"API endpoint '{raw}' not found or is disabled. "
                    "Check Settings → Custom API Endpoints."
                ),
            )
        raise HTTPException(
            status_code=400,
            detail="No custom API endpoints configured. Please add one in Settings → LLM Settings → Custom API Endpoints",
        )

    if not (endpoint_config.get("url") or "").strip():
        raise HTTPException(
            status_code=400,
            detail=(
                f"API endpoint '{endpoint_config.get('name') or model_name}' has no provider URL configured."
            ),
        )
    
    base_url = endpoint_config['url']
    url = _build_nanogpt_url(base_url, request_data, endpoint_config)
    
    force_thinking = _prepare_outbound_provider_model(model_name, request_data, endpoint_config)

    if request_data.pop("_skip_openai_message_pruning", False):
        logger.info("[OpenAI Compat] skip_openai_message_pruning: sending messages without local token pruning (caller accepts upstream limits).")
        msg_count = len(request_data.get("messages", []))
        char_count = approx_openai_messages_payload_chars(request_data.get("messages", []))
        logger.info(f"[OpenAI Compat] Outgoing API Payload (unpruned): {msg_count} messages, ~{char_count} chars")
        logger.info(f"[OpenAI Compat] Forwarding {model_name} to {endpoint_config['name']} at {url}")
        request_data["max_tokens"] = min(request_data.get("max_tokens", 16384) or 16384, 32768)
        apply_extended_thinking_request(endpoint_config, request_data, force_thinking=force_thinking)
        log_generate_outbound(url, request_data.get("model", ""), endpoint_config, request_data)
        return endpoint_config, url, request_data
    
    # --- CONTEXT PRUNING LOGIC ---
    # Smart context limit management to prevent upstream "input too long" errors.
    # We allow the limit to be configured in the endpoint settings (settings.json),
    # defaulting to 32768 as a higher-but-reasonable baseline for long-form chats.
    default_limit = 1_000_000
    configured_limit = endpoint_config.get('context_window')
    
    if configured_limit:
        try:
            CONTEXT_WINDOW_LIMIT = int(configured_limit)
        except (ValueError, TypeError):
            CONTEXT_WINDOW_LIMIT = default_limit
    else:
        CONTEXT_WINDOW_LIMIT = default_limit

    SAFETY_MARGIN = 1000 # Increased safety margin to account for tokenizer mismatch (tiktoken vs Llama)
    
    requested_gen_tokens = min(request_data.get('max_tokens', 16384) or 16384, 32768)
    request_data['max_tokens'] = requested_gen_tokens
    # The budget for INPUT tokens is Total - Output - Safety
    max_input_tokens = CONTEXT_WINDOW_LIMIT - requested_gen_tokens - SAFETY_MARGIN
    
    # Ensure sane minimum
    if max_input_tokens < 1000:
        max_input_tokens = 1000 # Force at least 1k input context, risking truncation error over garbage output
    
    messages = request_data.get('messages', [])
    
    # Prune messages to fit budget
    pruned_messages, original_count, new_count = prune_messages(
        messages, 
        max_input_tokens=max_input_tokens,
        model_name=request_data['model']
    )
    
    request_data['messages'] = pruned_messages
    
    if original_count > new_count:
        logger.warning(f"[OpenAI Compat] Pruned context from {original_count} to {new_count} tokens (Limit: {max_input_tokens}). Removed {len(messages) - len(pruned_messages)} messages.")
    else:
        logger.info(f"[OpenAI Compat] Context OK: {original_count} tokens (Limit: {max_input_tokens})")

    # Log payload size for debugging context issues
    msg_count = len(request_data.get('messages', []))
    char_count = approx_openai_messages_payload_chars(request_data.get("messages", []))
    logger.info(f"[OpenAI Compat] Outgoing API Payload: {msg_count} messages, ~{char_count} chars")
    logger.info(f"[OpenAI Compat] Forwarding {model_name} to {endpoint_config['name']} at {url}")
    apply_extended_thinking_request(endpoint_config, request_data, force_thinking=force_thinking)
    log_generate_outbound(url, request_data.get("model", ""), endpoint_config, request_data)
    
    return endpoint_config, url, request_data


def _resolve_redirect_url(base: str, location: str) -> str:
    """Resolve a possibly relative Location header to an absolute URL."""
    from urllib.parse import urljoin
    return urljoin(base.rstrip("/") + "/", location)


def _openai_compat_is_transient_upstream_for_retry(
    exc: BaseException,
    *,
    include_read_write_timeout: bool = False,
) -> bool:
    """Brief WAN / DNS / router blips — safe to retry a fresh request."""
    kinds: tuple = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
    # Upstream may drop mid-request on large bodies without sending a response.
    kinds = kinds + (httpx.RemoteProtocolError,)
    try:
        import httpcore

        if httpcore.RemoteProtocolError not in kinds:
            kinds = kinds + (httpcore.RemoteProtocolError,)
    except ImportError:
        pass
    if include_read_write_timeout:
        kinds = kinds + (httpx.ReadTimeout, httpx.WriteTimeout)
    return isinstance(exc, kinds)


async def forward_to_configured_endpoint_streaming(
    endpoint_config: dict,
    url: str,
    request_data: dict,
    extra_headers: Optional[Dict[str, str]] = None,
):
    """Forward OpenAI streaming request to the configured custom endpoint.
    
    Note: endpoint_config, url, and request_data should be prepared by _prepare_endpoint_request()
    before calling this generator to ensure errors are raised before streaming starts.
    """
    # Build headers similar to what SillyTavern sends
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        **get_provider_attribution_headers(endpoint_config),
    }
    
    api_key = endpoint_config.get('api_key', '')
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # Chub.ai also accepts CH-API-Key header
        if api_key.startswith('CHK-'):
            headers["CH-API-Key"] = api_key

    if extra_headers:
        headers.update(extra_headers)
    
    base_url = endpoint_config['url']
    
    log_generate_outbound(url, request_data.get("model", ""), endpoint_config, request_data)
    logger.info(f"[OpenAI Compat] Making request to {url}")
    logger.info(f"[OpenAI Compat] Headers (redacted): {list(headers.keys())}")
    logger.info(f"[OpenAI Compat] Request body keys: {list(request_data.keys())}")

    max_stream_attempts = request_data.get("_max_stream_attempts")
    if max_stream_attempts is not None:
        try:
            max_stream_attempts = max(1, int(max_stream_attempts))
        except (TypeError, ValueError):
            max_stream_attempts = 1
    else:
        max_stream_attempts = 5

    for attempt in range(1, max_stream_attempts + 1):
        try:
            # follow_redirects=False so we can re-POST on 301/302 (default follow would change POST→GET and cause 405)
            async with httpx.AsyncClient(timeout=REMOTE_HTTPC_TIMEOUT, follow_redirects=False, verify=True) as client:
                logger.info(f"[OpenAI Compat] Initiating POST to {url}...")
                async with client.stream("POST", url, headers=headers, json=request_data) as response:
                    logger.info(f"[OpenAI Compat] Got response status: {response.status_code}")
                    if response.status_code in (301, 302, 307, 308):
                        location = response.headers.get("location")
                        await response.aread()  # consume body before closing
                        if location:
                            redirect_url = _resolve_redirect_url(url, location)
                            logger.info(f"[OpenAI Compat] Following redirect (preserving POST): {url} -> {redirect_url}")
                            # Re-POST to redirect URL (one hop only)
                            async with client.stream("POST", redirect_url, headers=headers, json=request_data) as redir_response:
                                if redir_response.status_code != 200:
                                    err = await redir_response.aread()
                                    error_msg = f"Remote API error ({redir_response.status_code}): {err.decode()}"
                                    logger.error(f"[OpenAI Compat] {error_msg}")
                                    yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error', 'code': redir_response.status_code}})}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return
                                chunk_count = 0
                                buffer = b""
                                async for chunk_bytes in redir_response.aiter_raw():
                                    if isinstance(chunk_bytes, bytes):
                                        buffer += chunk_bytes
                                    else:
                                        buffer += chunk_bytes.encode('utf-8') if isinstance(chunk_bytes, str) else b""
                                    while b'\n\n' in buffer:
                                        message, buffer = buffer.split(b'\n\n', 1)
                                        if not message.strip():
                                            continue
                                        chunk_count += 1
                                        if chunk_count % 50 == 0:
                                            logger.debug(f"[OpenAI Compat] Received {chunk_count} chunks from remote...")
                                        try:
                                            message_str = message.decode('utf-8', errors='ignore')
                                            if chunk_count == 1:
                                                logger.info(f"[OpenAI Compat] First chunk received: {message_str[:200]}...")
                                            lines = message_str.split('\n')
                                            for line in lines:
                                                if line.startswith('data: '):
                                                    json_str = line[6:].strip()
                                                    if json_str == '[DONE]':
                                                        yield "data: [DONE]\n\n"
                                                        continue
                                                    if json_str:
                                                        upstream_data = json.loads(json_str)
                                                        if "error" in str(upstream_data).lower() and '"message":' in str(upstream_data):
                                                            logger.warning(f"[OpenAI Compat] Detected error in successful stream: {json_str[:200]}")
                                                        enriched = dict(upstream_data)
                                                        enriched["raw"] = upstream_data
                                                        yield f"data: {json.dumps(enriched)}\n\n"
                                        except json.JSONDecodeError:
                                            yield f"{message.decode('utf-8', errors='ignore')}\n\n"
                                        except Exception as e:
                                            logger.debug(f"[OpenAI Compat] Stream parse error: {e}")
                                            yield f"{message.decode('utf-8', errors='ignore')}\n\n"
                                logger.info(f"[OpenAI Compat] Stream completed successfully. Total chunks: {chunk_count}")
                        else:
                            error_msg = f"Redirect with no Location header ({response.status_code})"
                            logger.error(f"[OpenAI Compat] {error_msg}")
                            yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error', 'code': response.status_code}})}\n\n"
                            yield "data: [DONE]\n\n"
                        return
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = f"Remote API error ({response.status_code}): {error_text.decode()}"
                        if response.status_code == 405:
                            logger.error(f"[OpenAI Compat] 405 Method Not Allowed. Location header: {response.headers.get('location')}. If the API worked in a fresh chat, the server may be redirecting long requests and the client was sending GET after redirect.")
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

                    chunk_count = 0
                    buffer = b""
                    async for chunk_bytes in response.aiter_raw():
                        if isinstance(chunk_bytes, bytes):
                            buffer += chunk_bytes
                        else:
                            buffer += chunk_bytes.encode('utf-8') if isinstance(chunk_bytes, str) else b""
                        while b'\n\n' in buffer:
                            message, buffer = buffer.split(b'\n\n', 1)
                            if not message.strip():
                                continue
                            chunk_count += 1
                            if chunk_count % 50 == 0:
                                logger.debug(f"[OpenAI Compat] Received {chunk_count} chunks from remote...")
                            try:
                                message_str = message.decode('utf-8', errors='ignore')
                                if chunk_count == 1:
                                    logger.info(f"[OpenAI Compat] First chunk received: {message_str[:200]}...")
                                lines = message_str.split('\n')
                                for line in lines:
                                    if line.startswith('data: '):
                                        json_str = line[6:].strip()
                                        if json_str == '[DONE]':
                                            yield "data: [DONE]\n\n"
                                            continue
                                        if json_str:
                                            upstream_data = json.loads(json_str)
                                            if "error" in str(upstream_data).lower() and '"message":' in str(upstream_data):
                                                logger.warning(f"[OpenAI Compat] Detected error in successful stream: {json_str[:200]}")
                                            # Re-emit with ALL original fields PLUS raw field
                                            enriched = dict(upstream_data)
                                            enriched["raw"] = upstream_data
                                            yield f"data: {json.dumps(enriched)}\n\n"
                            except json.JSONDecodeError:
                                # Not JSON, pass through as-is
                                yield f"{message.decode('utf-8', errors='ignore')}\n\n"
                            except Exception as e:
                                logger.debug(f"[OpenAI Compat] Stream parse error: {e}")
                                yield f"{message.decode('utf-8', errors='ignore')}\n\n"
                    logger.info(f"[OpenAI Compat] Stream completed successfully. Total chunks: {chunk_count}")
            return

        except httpx.RequestError as e:
            if _openai_compat_is_transient_upstream_for_retry(e) and attempt < max_stream_attempts:
                delay = min(20.0, 1.25 * (2 ** (attempt - 1)))
                logger.warning(
                    "[OpenAI Compat] Transient upstream error (stream attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    max_stream_attempts,
                    delay,
                    type(e).__name__,
                )
                await asyncio.sleep(delay)
                continue
            logger.warning("[OpenAI Compat] Connection error to %s: %s", url, type(e).__name__)
            error_event = {
                "error": {
                    "message": f"Cannot connect to {endpoint_config['name']} at {base_url}: {type(e).__name__}: {str(e)}",
                    "type": "connection_error",
                    "code": 502,
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            logger.error(f"[OpenAI Compat] Unexpected error: {type(e).__name__}: {e}", exc_info=True)
            error_event = {
                "error": {
                    "message": f"Unexpected error: {type(e).__name__}: {str(e)}",
                    "type": "unknown_error",
                    "code": 500,
                },
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"
            return


async def collect_openai_compatible_stream_text(
    endpoint_config: dict,
    url: str,
    request_data: dict,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Upstream OpenAI-style chat with stream=True; aggregate delta content.
    Avoids remotes that buffer the entire completion before sending a non-streaming body (short read timeouts).
    """
    rd = dict(request_data)
    rd["stream"] = True
    parts: List[str] = []
    buffer = b""
    logger.info("[OpenAI Compat] collect_openai_compatible_stream_text: upstream streaming aggregate")
    async for chunk_bytes in forward_to_configured_endpoint_streaming(
        endpoint_config, url, rd, extra_headers
    ):
        if isinstance(chunk_bytes, bytes):
            buffer += chunk_bytes
        else:
            buffer += chunk_bytes.encode("utf-8") if isinstance(chunk_bytes, str) else b""
        while b"\n\n" in buffer:
            message, buffer = buffer.split(b"\n\n", 1)
            if not message.strip():
                continue
            try:
                message_str = message.decode("utf-8", errors="ignore")
                for line in message_str.split("\n"):
                    if not line.startswith("data: "):
                        continue
                    json_str = line[6:].strip()
                    if json_str == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
                    err = chunk_data.get("error")
                    if isinstance(err, dict):
                        msg = err.get("message", str(err))
                        code = int(err.get("code", 502) or 502)
                        raise HTTPException(status_code=min(code, 599), detail=msg)
                    content, reasoning = extract_openai_stream_delta_parts(chunk_data)
                    if reasoning:
                        parts.append(reasoning)
                    if content:
                        parts.append(content)
            except HTTPException:
                raise
            except Exception as e:
                logger.debug("[OpenAI Compat] stream line parse skip: %s", e)
    return "".join(parts)


async def forward_to_configured_endpoint_non_streaming(
    endpoint_config: dict,
    url: str,
    request_data: dict,
    extra_headers: Optional[Dict[str, str]] = None,
):
    """Forward OpenAI non-streaming request to the configured custom endpoint.
    
    Note: endpoint_config, url, and request_data should be prepared by _prepare_endpoint_request()
    """
    # Build headers similar to what SillyTavern sends
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        **get_provider_attribution_headers(endpoint_config),
    }
    
    api_key = endpoint_config.get('api_key', '')
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # Chub.ai also accepts CH-API-Key header
        if api_key.startswith('CHK-'):
            headers["CH-API-Key"] = api_key

    if extra_headers:
        headers.update(extra_headers)
    
    base_url = endpoint_config['url']
    
    logger.info(f"[OpenAI Compat] Making non-streaming request to {url}")
    
    # Log payload size
    msg_count = len(request_data.get('messages', []))
    char_count = approx_openai_messages_payload_chars(request_data.get("messages", []))
    logger.info(f"[OpenAI Compat] Outgoing API Payload: {msg_count} messages, ~{char_count} chars")
    
    last_error = None
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=REMOTE_HTTPC_TIMEOUT, follow_redirects=False) as client:
                response = await client.post(url, headers=headers, json=request_data)
                if response.status_code in (301, 302, 307, 308):
                    location = response.headers.get("location")
                    if location:
                        url = _resolve_redirect_url(url, location)
                        logger.info(f"[OpenAI Compat] Following redirect (preserving POST) -> {url}")
                        response = await client.post(url, headers=headers, json=request_data)
                if response.status_code != 200:
                    if response.status_code == 405:
                        logger.error("[OpenAI Compat] 405 Method Not Allowed. If the API worked in a fresh chat, the server may be redirecting and the client was sending GET after redirect.")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Remote API error from {base_url}: {response.text}"
                    )
                return response.json()
        except httpx.RequestError as e:
            last_error = e
            if not _openai_compat_is_transient_upstream_for_retry(e, include_read_write_timeout=True):
                logger.warning("[OpenAI Compat] Non-retryable connection error: %s", type(e).__name__)
                raise HTTPException(
                    status_code=502,
                    detail=f"Cannot connect to {endpoint_config['name']} at {base_url}: {type(e).__name__}: {str(e)}"
                ) from e
            if attempt < max_attempts - 1:
                delay = min(20.0, 1.25 * (2 ** attempt))
                logger.warning(
                    "[OpenAI Compat] Connection error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_attempts,
                    delay,
                    type(e).__name__,
                )
                await asyncio.sleep(delay)
            else:
                logger.warning(
                    "[OpenAI Compat] Connection failed after %d attempts: %s",
                    max_attempts,
                    type(last_error).__name__,
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"Cannot connect to {endpoint_config['name']} at {base_url}: {type(last_error).__name__}: {str(last_error)}"
                ) from last_error


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

def _chat_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def _local_chat_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    return [
        {"role": message.role, "content": _chat_content_text(message.content)}
        for message in messages
    ]


def _available_local_model_names(model_manager: ModelManager) -> List[str]:
    available = model_manager.list_available_models()
    if isinstance(available, dict):
        available = available.get("available_models", [])
    return [str(name) for name in (available or [])]


def _resolve_local_model(model_manager: ModelManager, requested_model: str, requested_gpu: Optional[int]):
    loaded = model_manager.get_loaded_models().get("loaded_models", [])
    if requested_model in {"default", "mirid", "local"}:
        if not loaded:
            raise HTTPException(status_code=409, detail="No local GGUF model is loaded in Mirid.")
        selected = loaded[0]
        return selected["name"], int(selected["gpu_id"])

    for model in loaded:
        if model.get("name") == requested_model and (
            requested_gpu is None or int(model.get("gpu_id", 0)) == requested_gpu
        ):
            return requested_model, int(model.get("gpu_id", 0))

    if requested_model not in _available_local_model_names(model_manager):
        raise HTTPException(status_code=404, detail=f"Model not found: {requested_model}")
    return requested_model, int(requested_gpu or 0)


async def _get_local_chat_model(model_manager: ModelManager, model_name: str, gpu_id: int):
    try:
        return model_manager.get_model(model_name, gpu_id)
    except ValueError:
        await model_manager.load_model(model_name, gpu_id=gpu_id)
        return model_manager.get_model(model_name, gpu_id)


def _local_chat_kwargs(request: ChatCompletionRequest) -> Dict[str, Any]:
    stop = request.stop
    if isinstance(stop, str):
        stop = [stop]
    return {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "repeat_penalty": request.repetition_penalty,
        "max_tokens": request.max_tokens,
        "stop": stop or [],
    }


async def _stream_local_chat(model, request: ChatCompletionRequest, model_name: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    yield create_openai_chunk(chunk_id, model_name, "", None)
    try:
        chunks = model.create_chat_completion(
            messages=_local_chat_messages(request.messages),
            stream=True,
            **_local_chat_kwargs(request),
        )
        for chunk in chunks:
            if isinstance(chunk, dict) and chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
            choice = choices[0] if choices else {}
            content = choice.get("text", "") or (choice.get("delta") or {}).get("content", "")
            if content:
                yield create_openai_chunk(chunk_id, model_name, content, None)
            await asyncio.sleep(0)
        yield create_openai_chunk(chunk_id, model_name, "", "stop")
    except Exception as error:
        logger.exception("Local OpenAI-compatible stream failed for %s", model_name)
        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": str(error), "type": "server_error"},
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def _complete_local_chat(model, request: ChatCompletionRequest, model_name: str):
    result = await asyncio.to_thread(
        model.create_chat_completion,
        messages=_local_chat_messages(request.messages),
        stream=False,
        **_local_chat_kwargs(request),
    )
    if not isinstance(result, dict) or result.get("error"):
        error = result.get("error") if isinstance(result, dict) else "Local generation failed."
        raise HTTPException(status_code=500, detail=str(error))
    choices = result.get("choices", [])
    choice = choices[0] if choices else {}
    content = (choice.get("message") or {}).get("content") or choice.get("text", "")
    return {
        "id": result.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": result.get("created") or int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": choice.get("finish_reason") or "stop",
        }],
        "usage": result.get("usage"),
    }

@router.get("/models")
async def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    """List local GGUF models and enabled configured endpoints."""
    try:
        if not model_manager:
            raise HTTPException(status_code=500, detail="Model manager not available")

        openai_models = [
            ModelInfo(id=model_name, created=int(time.time()), owned_by="mirid-local")
            for model_name in _available_local_model_names(model_manager)
        ]
        for endpoint in _load_custom_api_endpoints():
            endpoint_id = str(endpoint.get("id") or "").strip()
            if endpoint.get("enabled") and endpoint_id:
                openai_models.append(
                    ModelInfo(id=endpoint_id, created=int(time.time()), owned_by="mirid-provider")
                )
        return {"object": "list", "data": openai_models}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"object": "list", "data": []}


@router.post("/chat/completions")
async def serve_chat_completions(raw_request: Request, model_manager: ModelManager = Depends(get_model_manager)):
    """Serve local GGUF models or proxy an explicitly selected configured endpoint."""
    try:
        request_payload = await raw_request.json()
        request = (
            ChatCompletionRequest.model_validate(request_payload)
            if hasattr(ChatCompletionRequest, "model_validate")
            else ChatCompletionRequest.parse_obj(request_payload)
        )
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="Invalid JSON received.") from error
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Unprocessable Entity: {error}") from error

    logger.info(
        "[/v1/chat/completions] model=%s stream=%s messages=%d",
        request.model,
        request.stream,
        len(request.messages),
    )
    try:
        if request.model and is_api_endpoint(request.model):
            request_data = (
                request.model_dump()
                if hasattr(request, "model_dump")
                else request.dict()
            )
            request_data["messages"] = _local_chat_messages(request.messages)
            endpoint_config, url, request_data = prepare_endpoint_request(request.model, request_data)
            if request.stream:
                return StreamingResponse(
                    forward_to_configured_endpoint_streaming(endpoint_config, url, request_data),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return await forward_to_configured_endpoint_non_streaming(endpoint_config, url, request_data)

        model_name, gpu_id = _resolve_local_model(model_manager, request.model, request.gpu_id)
        model = await _get_local_chat_model(model_manager, model_name, gpu_id)
        if request.stream:
            return StreamingResponse(
                _stream_local_chat(model, request, model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await _complete_local_chat(model, request, model_name)
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Error in chat completions: %s", error, exc_info=True)
        raise HTTPException(status_code=500, detail=str(error)) from error

async def _legacy_chat_completions(raw_request: Request, model_manager: ModelManager = Depends(get_model_manager)):
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
        # ALWAYS resolve through API endpoints — never fall through to local GGUF.
        # If the model name is empty or not a recognized API endpoint, pick one
        # from the round-robin pool so every request routes to a remote API.
        endpoint_config = None
        url = None
        effective_model = request.model

        if request.model and is_api_endpoint(request.model):
            logger.info(f"[OpenAI Compat] Detected API endpoint: {request.model}")
        else:
            # Empty or unknown model name — grab the first available API endpoint
            fallback_endpoint = get_configured_endpoint(
                model_id=None,
                request_purpose="user_chat",
                frontend_round_robin_enabled=True,
            )
            if fallback_endpoint:
                effective_model = fallback_endpoint.get("model") or request.model or ""
                logger.info(
                    "[OpenAI Compat] Model '%s' not a known API endpoint — forced to round-robin endpoint '%s' (%s)",
                    request.model, fallback_endpoint.get("id"), effective_model,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No API endpoints configured. Add one in Settings → Custom API Endpoints.",
                )

        request_data = {
            "model": effective_model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }

        if request.stop: request_data["stop"] = request.stop
        if request.top_k: request_data["top_k"] = request.top_k
        if request.repetition_penalty: request_data["repetition_penalty"] = request.repetition_penalty

        endpoint_config, url, request_data = prepare_endpoint_request(effective_model, request_data)

        if request.stream:
            return StreamingResponse(
                forward_to_configured_endpoint_streaming(endpoint_config, url, request_data),
                media_type="text/event-stream"
            )
        else:
            result = await forward_to_configured_endpoint_non_streaming(endpoint_config, url, request_data)
            return result

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
    return {"status": "ok", "service": "mirid-openai-compat"}
