import json
import logging
import datetime
import os
import random
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Body, Depends, Request
from fastapi.responses import FileResponse

from .prompts import build_generate_entity_prompt, build_agentic_tick_prompt, RATE_USER_PROMPT
from .avatar_vocab import build_avatar_prompt, validate_avatar_selections
from .name_pool import get_name_pool_block
from .dummy_rivals import (
    DUMMY_RIVAL_PROFILES,
    DUMMY_RIVAL_BY_ID,
    get_dummy_pool_summary,
    get_dummy_context_for_tick,
    get_active_dummies,
    build_dummy_context_prompt,
)
from .interaction_log import (
    get_interaction_context,
    log_exchange,
    log_character_action,
    log_social_awareness_to_all,
    format_user_profile_for_prompt as format_user_profile_log,
)
from ..runtime_paths import data_path, runtime_data_root

logger = logging.getLogger("lattice")
lattice_router = APIRouter(prefix="/lattice", tags=["lattice"])

MAX_GEN_ATTEMPTS = 1
MAX_TICK_ATTEMPTS = 2


def _build_character_context_block(character_name: str, character_profile: dict) -> str:
    dp = character_profile.get("dating_profile", {})
    personality = character_profile.get("personality", "")
    description = character_profile.get("description", "")
    speech_style = character_profile.get("speech_style", "")
    lines = [
        f"Name: {character_name}",
        f"Personality: {personality}" if personality else "",
        f"Description: {description}" if description else "",
        f"Speech style: {speech_style}" if speech_style else "",
        f"Bio: {dp.get('bio', '')}" if dp.get("bio") else "",
        f"Seeking: {dp.get('seeking', '')}" if dp.get("seeking") else "",
        f"Turn-ons: {', '.join(dp.get('turn_ons', []))}" if dp.get("turn_ons") else "",
        f"Turn-offs: {', '.join(dp.get('turn_offs', []))}" if dp.get("turn_offs") else "",
    ]
    return "\n".join(l for l in lines if l)


def _build_full_prompt(
    instructions: str,
    character_name: str,
    character_profile: dict,
    user_name: str = "the user",
    user_dating_profile: dict = None,
    character_id: str = None,
) -> str:
    char_block = _build_character_context_block(character_name, character_profile)
    user_block = format_user_profile_log(user_dating_profile) if user_dating_profile else ""

    interaction_block = ""
    if character_id:
        ctx = get_interaction_context(character_id, limit=30)
        if ctx.get("formatted_text"):
            interaction_block = ctx["formatted_text"]

    parts = [instructions]
    if char_block:
        parts.append(f"\n=== YOUR PROFILE ===\n{char_block}")
    if user_block:
        parts.append(f"\n=== USER PROFILE ===\n{user_block}")
    if interaction_block:
        parts.append(f"\n=== YOUR HISTORY ===\n{interaction_block}")

    return "\n".join(parts)


def _extract_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    try:
        from . import character_json_parse as cjp
    except ImportError:
        from backend.app import character_json_parse as cjp

    # Strip code fences
    cleaned = cjp._strip_fences(text)

    # Try array first (for batch generation)
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        try:
            result = json.loads(cleaned[arr_start:arr_end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try single object
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        blob = cleaned[obj_start:obj_end + 1]
        # Try direct parse
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
        # Try repair + parse
        try:
            repaired = cjp.repair_truncated_json_blob(blob)
            if repaired:
                return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass

    # Salvage: extract individual fields via regex from the raw text
    salvaged = cjp.salvage_character_fields(cleaned or text)
    if salvaged and salvaged.get("name"):
        return salvaged

    # Last resort: try to find ANY { } pair and extract it
    import re as _re
    matches = _re.findall(r'\{[^}]+\}', text)
    for m in matches:
        try:
            parsed = json.loads(m)
            if isinstance(parsed, dict) and parsed.get("name"):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _replace_many(text, replacements):
    result = text
    for key, value in replacements.items():
        result = result.replace(key, str(value) if value is not None else "")
    return result


def _normalize_character_json(raw: Dict[str, Any]) -> Dict[str, Any]:
    dating_profile = raw.get("dating_profile", {})
    if not isinstance(dating_profile, dict):
        dating_profile = {}

    lore_entries = raw.get("loreEntries", [])
    if not isinstance(lore_entries, list):
        lore_entries = []

    example_dialogue = raw.get("example_dialogue", [])
    if not isinstance(example_dialogue, list):
        example_dialogue = []

    avatar_raw = raw.get("avatar", {}) if isinstance(raw.get("avatar"), dict) else {}

    normalized = {
        "name": str(raw.get("name", "")).strip(),
        "description": str(raw.get("description", "")).strip(),
        "model_instructions": str(raw.get("model_instructions", "")).strip(),
        "scenario": str(raw.get("scenario", "")).strip(),
        "first_message": str(raw.get("first_message", "")).strip(),
        "example_dialogue": example_dialogue[:6],
        "speech_style": str(raw.get("speech_style", "")).strip(),
        "personality": str(raw.get("personality", "")).strip(),
        "background": str(raw.get("background", "")).strip(),
        "ethics_justification": str(raw.get("ethics_justification", "")).strip(),
        "loreEntries": [
            {
                "content": str(e.get("content", "")).strip(),
                "keywords": e.get("keywords", []) if isinstance(e.get("keywords"), list) else [],
            }
            for e in lore_entries[:10]
            if isinstance(e, dict) and e.get("content")
        ],
        "dating_profile": {
            "bio": str(dating_profile.get("bio", "")).strip(),
            "seeking": str(dating_profile.get("seeking", "")).strip(),
            "section_affinity": dating_profile.get("section_affinity", []) if isinstance(dating_profile.get("section_affinity"), list) else [],
            "turn_ons": dating_profile.get("turn_ons", []) if isinstance(dating_profile.get("turn_ons"), list) else [],
            "turn_offs": dating_profile.get("turn_offs", []) if isinstance(dating_profile.get("turn_offs"), list) else [],
            "preferred_modality": str(dating_profile.get("preferred_modality", "both")).strip(),
        },
        "chat_role": "npc",
        "created_at": datetime.datetime.utcnow().isoformat(),
        "avatars": [],
        "activeAvatarIndex": 0,
        "avatar": None,
    }

    for field in ("age", "figure", "background", "hair_color", "hair_style", "eye_color"):
        val = avatar_raw.get(field)
        if val:
            normalized[field] = str(val).strip()

    return normalized


async def _call_llm(prompt: str, model_manager, model_name: str, gpu_id: int, use_api: bool, api_endpoint=None, frontend_round_robin_enabled=None) -> str:
    # If no model name and API is available, force API. Never fall through to local GGUF.
    if (not model_name or not (model_name or "").strip()) and (not use_api or not api_endpoint):
        try:
            from . import openai_compat
            endpoint = openai_compat.get_configured_endpoint(
                model_id=None,
                request_purpose="create_character",
                frontend_round_robin_enabled=True,
            )
            if endpoint:
                use_api = True
                api_endpoint = endpoint
        except Exception:
            pass

    if use_api and api_endpoint:
        try:
            from . import character_intelligence
            from .character_intelligence import generate_with_api
        except ImportError:
            from backend.app.character_intelligence import generate_with_api

        return await generate_with_api(
            prompt,
            api_endpoint,
            model_name=model_name,
            frontend_round_robin_enabled=frontend_round_robin_enabled,
            force_resolved_endpoint=True,
        )

    if model_manager:
        try:
            from . import inference
            from .character_intelligence import _generate_character_llm_text
        except ImportError:
            from backend.app import inference
            from backend.app.character_intelligence import _generate_character_llm_text

        return await inference.generate_text(
            model_manager=model_manager,
            model_name=model_name or "",
            prompt=prompt,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            stop_sequences=["</character>", "---"],
            gpu_id=gpu_id or 0,
        )

    raise HTTPException(status_code=503, detail="No model manager or API endpoint available")


async def _resolve_api_endpoint(model_name: str, frontend_round_robin_enabled: Optional[bool]):
    try:
        from . import openai_compat
    except ImportError:
        from backend.app import openai_compat

    # Empty or blank model name → always try API round-robin pool, never local GGUF
    if not model_name or not model_name.strip():
        endpoint = openai_compat.get_configured_endpoint(
            model_id=None,
            request_purpose="create_character",
            frontend_round_robin_enabled=True,
        )
        if endpoint:
            return True, endpoint
        return False, None

    # Auto-router enabled but no model name — use round-robin pool
    if not model_name and frontend_round_robin_enabled:
        endpoint = openai_compat.get_configured_endpoint(
            model_id=None,
            request_purpose="create_character",
            frontend_round_robin_enabled=True,
        )
        if endpoint:
            return True, endpoint
        return False, None

    if not model_name or not openai_compat.is_api_endpoint(model_name):
        return False, None

    endpoint = openai_compat.get_configured_endpoint(
        model_name,
        skip_rotation=frontend_round_robin_enabled is False,
        request_purpose="create_character",
        frontend_round_robin_enabled=frontend_round_robin_enabled,
    )
    if not endpoint:
        raise HTTPException(
            status_code=400,
            detail=f"API endpoint '{model_name}' not found or disabled.",
        )
    return True, endpoint


@lattice_router.post("/outreach-push")
async def outreach_push_endpoint(
    request: Request,
    data: dict = Body(...),
):
    character_name = data.get("character_name", "").strip()
    character_avatar = data.get("character_avatar", "")
    raw_content = data.get("message_content", "")
    if isinstance(raw_content, dict):
        message_content = json.dumps(raw_content)
    elif isinstance(raw_content, str):
        message_content = raw_content.strip()
    else:
        message_content = str(raw_content).strip() if raw_content else ""
    character_snapshot = data.get("character_snapshot", {})
    dm_thread_id = data.get("dm_thread_id", "").strip() or None

    if not character_name or not message_content:
        raise HTTPException(status_code=400, detail="character_name and message_content are required")

    try:
        from .. import outreach_db
        from ..outreach_runtime import broadcast_event, send_web_push_all
    except ImportError:
        from backend.app import outreach_db
        from backend.app.outreach_runtime import broadcast_event, send_web_push_all

    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat().replace("+00:00", "Z")
    conv_id = f"outreach-conv-{uuid.uuid4().hex[:12]}"
    bot_mid = f"b-{uuid.uuid4().hex[:12]}"
    char_id = character_snapshot.get("id") if isinstance(character_snapshot, dict) else None

    bot_msg = {
        "id": bot_mid,
        "role": "bot",
        "content": message_content,
        "modelId": "primary",
        "characterId": char_id,
        "characterName": character_name,
        "avatar": character_avatar,
        "isScheduledOutreach": True,
    }

    conv = {
        "id": conv_id,
        "name": f"Mirror: {character_name}",
        "messages": [bot_msg],
        "characterIds": {"primary": char_id, "secondary": None, "user": None},
        "activeCharacterIds": [char_id] if char_id else [],
        "activeCharacterWeights": {char_id: 100} if char_id else {},
        "multiRoleContext": "",
        "created": now_iso,
        "requiresTitle": False,
        "agenticMemoryEnabled": False,
        "outreachRuleId": "mirror-pool",
        "characterSnapshot": character_snapshot,
    }

    preview = " ".join((message_content or "").split())[:200]
    event = {
        "type": "outreach_message",
        "ruleId": "mirror-pool",
        "ruleName": "Mirror Dating Pool",
        "conversationId": conv_id,
        "messageId": bot_mid,
        "characterName": character_name,
        "characterAvatar": character_avatar,
        "preview": preview,
        "attachmentImageUrl": None,
        "conversation": conv,
    }
    if dm_thread_id:
        event["dm_thread_id"] = dm_thread_id

    try:
        await outreach_db.save_conversation(conv)
        await broadcast_event(event)
        try:
            await send_web_push_all(event)
        except Exception:
            logger.debug("Web push skipped (not configured or no subscriptions)")
    except Exception as e:
        logger.error(f"Outreach push failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Outreach push: {character_name} -> {conv_id}")
    return {"status": "success", "conversation_id": conv_id, "message_id": bot_mid, "dm_thread_id": dm_thread_id}


@lattice_router.post("/generate-entity")
async def generate_entity_endpoint(
    request: Request,
    data: dict = Body(...),
):
    try:
        model_name = data.get("model_name", "")
        selected_model = data.get("selected_model") or model_name
        gpu_id = data.get("gpu_id", 0)
        history_context = data.get("history_context", "")
        dummy_realism = int(data.get("dummy_realism", 50))
        dummy_agency = int(data.get("dummy_agency", 50))
        section_hint = data.get("section_hint")
        pool_names = data.get("pool_names", [])
        frontend_round_robin_enabled = data.get("frontend_round_robin_enabled")
        avatar_pool_enabled = data.get("avatar_pool_enabled", False)
        count = int(data.get("count", 1))

        if frontend_round_robin_enabled is not None:
            frontend_round_robin_enabled = bool(frontend_round_robin_enabled)

        use_api, api_endpoint = await _resolve_api_endpoint(model_name, frontend_round_robin_enabled)

        model_manager = getattr(request.app.state, 'model_manager', None)
        if not model_manager and not use_api:
            raise HTTPException(status_code=503, detail="No model manager or API endpoint available")

        prompt = build_generate_entity_prompt(
            history_context=history_context,
            dummy_realism=dummy_realism,
            dummy_agency=dummy_agency,
            section_hint=section_hint,
            pool_names=pool_names,
            generating_model=model_name or "",
            user_dating_profile=data.get("user_dating_profile"),
            user_profile=data.get("user_profile"),
            count=max(1, min(10, count)),
            name_block=get_name_pool_block(pool_names, count),
        )

        logger.info(f"Generating entity (section={section_hint}, dummies={dummy_realism}/{dummy_agency})")

        last_error = None
        for attempt in range(MAX_GEN_ATTEMPTS):
            raw = await _call_llm(
                prompt=prompt,
                model_manager=model_manager,
                model_name=model_name,
                gpu_id=gpu_id,
                use_api=use_api,
                api_endpoint=api_endpoint,
                frontend_round_robin_enabled=frontend_round_robin_enabled,
            )
            if not raw:
                last_error = "Empty response from model"
                continue

            parsed = _extract_json_from_response(raw)
            if parsed:
                # Handle batch generation: parsed may be a list or a single dict
                raw_items = parsed if isinstance(parsed, list) else [parsed]
                characters = []
                for raw_item in raw_items[:max(1, count)]:
                    if not isinstance(raw_item, dict) or not raw_item.get("name"):
                        continue
                    character = _normalize_character_json(raw_item)
                    avatar_data = raw_item.get("avatar", {}) if isinstance(raw_item, dict) else {}
                    if avatar_data:
                        assembled_prompt = build_avatar_prompt(avatar_data)
                        validation_errors = validate_avatar_selections(avatar_data)
                    else:
                        assembled_prompt = ""
                        validation_errors = {}

                    character["generated_by"] = model_name or ""
                    character["generated_at"] = datetime.datetime.utcnow().isoformat()
                    characters.append({
                        "character": character,
                        "avatar": avatar_data,
                        "avatar_prompt": assembled_prompt,
                        "avatar_errors": validation_errors,
                    })

                if characters:
                    logger.info(f"Generated {len(characters)} entities (batch={count}, attempt {attempt + 1})")

                    # Persist to disk so page reloads don't lose generated characters
                    char_list = [c["character"] for c in characters]
                    existing = _get_generated_entities(request)
                    existing.extend(char_list)
                    _set_generated_entities(request, existing)

                    if count > 1:
                        return {
                            "status": "success",
                            "characters": char_list,
                            "batch": True,
                            "attempts": attempt + 1,
                        }
                    c = characters[0]
                    return {
                        "status": "success",
                        "character": c["character"],
                        "avatar": c["avatar"],
                        "avatar_prompt": c["avatar_prompt"],
                        "avatar_errors": c["avatar_errors"],
                        "attempts": attempt + 1,
                    }

            last_error = "Could not extract valid JSON from model response"
            logger.warning(f"Entity generation attempt {attempt + 1} failed: {last_error}")

        return {
            "status": "error",
            "error": last_error or "Generation failed",
            "attempts": MAX_GEN_ATTEMPTS,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@lattice_router.get("/generated-entities")
async def get_generated_entities_endpoint(request: Request):
    """Return all characters that were generated but may not have been claimed yet."""
    return {"status": "success", "entities": _get_generated_entities(request)}


@lattice_router.post("/generated-entities/claim")
async def claim_generated_entities_endpoint(
    request: Request,
    data: dict = Body(...),
):
    """Remove claimed character IDs from the pending store."""
    claimed_ids = data.get("ids", [])
    if not claimed_ids:
        return {"status": "success", "removed": 0}
    entities = _get_generated_entities(request)
    before = len(entities)
    id_set = set(claimed_ids)
    entities = [e for e in entities if e.get("id") not in id_set]
    _set_generated_entities(request, entities)
    removed = before - len(entities)
    logger.info(f"Claimed {removed} generated entities from pending store")
    return {"status": "success", "removed": removed}


@lattice_router.post("/agentic-tick")
async def agentic_tick_endpoint(
    request: Request,
    data: dict = Body(...),
):
    try:
        model_name = data.get("model_name", "")
        gpu_id = data.get("gpu_id", 0)
        actor_type = data.get("actor_type", "female_ai")
        action_type = data.get("action_type", "full")
        character_name = data.get("character_name", "Unknown")
        character_profile = data.get("character_profile", {})
        memory_entries = data.get("memory_entries", [])
        pool_summary = data.get("pool_summary", "")
        dummy_activity = data.get("dummy_activity", "")
        dummy_realism = int(data.get("dummy_realism", 50))
        dummy_agency = int(data.get("dummy_agency", 50))
        user_activity = data.get("user_activity", "")
        available_actions = data.get("available_actions", [])
        target_description = data.get("target_description", "")
        target_id = data.get("target_id", "")
        recent_events = data.get("recent_events", "")
        recent_activity = data.get("recent_activity", "")
        target_character_name = data.get("target_character_name", "")
        user_dating_profile = data.get("user_dating_profile", "")
        section_hint = data.get("section_hint") or ""
        frontend_round_robin_enabled = data.get("frontend_round_robin_enabled")

        if frontend_round_robin_enabled is not None:
            frontend_round_robin_enabled = bool(frontend_round_robin_enabled)

        use_api, api_endpoint = await _resolve_api_endpoint(model_name, frontend_round_robin_enabled)

        model_manager = getattr(request.app.state, 'model_manager', None)
        if not model_manager and not use_api:
            raise HTTPException(status_code=503, detail="No model manager available")

        voice_list = ""
        if action_type == "select_voice":
            try:
                vdir = Path(__file__).resolve().parent.parent / "static" / "voice_references"
                if vdir.is_dir():
                    files = [f.name for f in vdir.iterdir() if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")]
                    voice_list = ", ".join(sorted(files))
            except Exception:
                pass

        prompt = build_agentic_tick_prompt(
            actor_type=actor_type,
            action_type=action_type,
            character_name=character_name,
            character_profile=character_profile,
            memory_entries=memory_entries,
            pool_summary=pool_summary,
            dummy_activity=dummy_activity,
            dummy_realism=dummy_realism,
            dummy_agency=dummy_agency,
            user_activity=user_activity,
            available_actions=available_actions,
            target_description=target_description,
            target_id=target_id,
            recent_events=recent_events,
            recent_activity=recent_activity,
            target_character_name=target_character_name,
            user_dating_profile=user_dating_profile,
            generating_model=model_name or "",
            voice_list=voice_list,
            section_hint=section_hint,
        )

        logger.info(f"Agentic tick: {character_name} ({actor_type}/{action_type})")

        raw = await _call_llm(
            prompt=prompt,
            model_manager=model_manager,
            model_name=model_name,
            gpu_id=gpu_id,
            use_api=use_api,
            api_endpoint=api_endpoint,
            frontend_round_robin_enabled=frontend_round_robin_enabled,
        )

        parsed = _extract_json_from_response(raw)
        if parsed:
            logger.info(f"Agentic tick result: {character_name} chose {parsed.get('chosen_action', 'unknown')}")
            return {
                "status": "success",
                "action_result": {
                    "chosen_action": parsed.get("chosen_action", ""),
                    "target": parsed.get("target"),
                    "content": parsed.get("content", ""),
                    "reasoning": parsed.get("reasoning", ""),
                    "emotional_state": parsed.get("emotional_state", ""),
                },
                "raw_response": raw[:500],
            }

        return {
            "status": "partial",
            "action_result": {
                "chosen_action": "",
                "target": None,
                "content": raw[:1000] if raw else "",
                "reasoning": "Could not parse structured response",
                "emotional_state": "",
            },
            "raw_response": raw[:1000] if raw else "",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agentic tick error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


DUMMY_GEN_PROMPT = (
    "Generate a new male dummy rival profile for a dating pool. "
    "Realism level: {realism}/100. Agency level: {agency}/100. "
    "Existing dummies: {existing}. "
    "Create someone distinct and believable.\n\n"
    "Output valid JSON only:\n"
    '{{ "name": "first name", "age": number, "personality": "description", '
    '"style": "communication style", "interests": ["interest1", "interest2"], '
    '"turn_offs": ["thing1", "thing2"], "agency_level": number, "realism_level": number }}\n'
    "Make the name realistic. Give them specific, non-generic interests and turn-offs. Keep personality 1-2 sentences."
)


@lattice_router.get("/dummy-rivals")
async def get_dummy_rivals_endpoint(
    request: Request,
    agency_threshold: int = 0,
    realism_threshold: int = 0,
):
    base = get_active_dummies(
        agency_threshold=agency_threshold,
        realism_threshold=realism_threshold,
    )
    generated = getattr(request.app.state, DUMMY_GEN_STATE_KEY, [])
    return {
        "status": "success",
        "dummies": base + generated,
        "total": len(DUMMY_RIVAL_PROFILES) + len(generated),
        "base_count": len(DUMMY_RIVAL_PROFILES),
        "generated_count": len(generated),
    }


@lattice_router.post("/generate-dummy")
async def generate_dummy_endpoint(
    request: Request,
    data: dict = Body(...),
):
    model_name = data.get("model_name", "")
    dummy_realism = int(data.get("dummy_realism", 50))
    dummy_agency = int(data.get("dummy_agency", 50))
    gpu_id = data.get("gpu_id", 0)

    existing = getattr(request.app.state, DUMMY_GEN_STATE_KEY, [])
    existing_names = [p["name"] for p in DUMMY_RIVAL_PROFILES + existing]

    prompt = DUMMY_GEN_PROMPT.format(
        realism=dummy_realism,
        agency=dummy_agency,
        existing=", ".join(existing_names),
    )

    use_api, api_endpoint = await _resolve_api_endpoint(model_name, data.get("frontend_round_robin_enabled"))
    model_manager = getattr(request.app.state, "model_manager", None)
    if not model_manager and not use_api:
        return {"status": "error", "error": "No model available"}

    raw = await _call_llm(
        prompt=prompt,
        model_manager=model_manager,
        model_name=model_name or "",
        gpu_id=gpu_id or 0,
        use_api=use_api,
        api_endpoint=api_endpoint,
        frontend_round_robin_enabled=data.get("frontend_round_robin_enabled"),
    )

    parsed = _extract_json_from_response(raw or "")
    if parsed and parsed.get("name"):
        dummy = {
            "id": f"dummy_gen_{parsed['name'].lower().replace(' ', '_')}",
            "name": str(parsed["name"]).strip(),
            "avatar_url": "/static/dummy_avatars/dummy_marcus.svg",
            "age": int(parsed.get("age", 30)),
            "personality": str(parsed.get("personality", "")).strip(),
            "style": str(parsed.get("style", "")).strip(),
            "interests": parsed.get("interests", []) if isinstance(parsed.get("interests"), list) else [],
            "turn_offs": parsed.get("turn_offs", []) if isinstance(parsed.get("turn_offs"), list) else [],
            "agency_level": int(parsed.get("agency_level", 50)),
            "realism_level": int(parsed.get("realism_level", 50)),
            "generated": True,
        }
        existing.append(dummy)
        setattr(request.app.state, DUMMY_GEN_STATE_KEY, existing)
        logger.info(f"Generated dummy: {dummy['name']}")
        return {"status": "success", "dummy": dummy}

    return {"status": "error", "error": "Could not parse dummy profile"}


@lattice_router.get("/dummy-rival/{dummy_id}")
async def get_dummy_rival_endpoint(dummy_id: str):
    profile = DUMMY_RIVAL_BY_ID.get(dummy_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Dummy rival '{dummy_id}' not found")
    return {
        "status": "success",
        "dummy": profile,
    }


FEED_STATE_KEY = "mirror_feed_posts"
FEED_FILE_PATH = data_path("mirror_feed.json")
DUMMY_GEN_STATE_KEY = "mirror_generated_dummies"

STORY_STATE_KEY = "mirror_stories"
STORY_FILE_PATH = data_path("mirror_stories.json")

GENERATED_ENTITIES_STATE_KEY = "mirror_generated_entities"
GENERATED_ENTITIES_FILE_PATH = data_path("mirror_generated_entities.json")


def _load_feed_from_disk():
    try:
        if FEED_FILE_PATH.is_file():
            with open(FEED_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load feed from disk: {e}")
    return []


def _save_feed_to_disk(posts):
    try:
        FEED_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEED_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save feed to disk: {e}")


def _get_feed_posts(request: Request) -> List[Dict[str, Any]]:
    posts = getattr(request.app.state, FEED_STATE_KEY, None)
    if posts is None:
        posts = _load_feed_from_disk()
        setattr(request.app.state, FEED_STATE_KEY, posts)
    return posts


def _set_feed_posts(request: Request, posts: List[Dict[str, Any]]):
    setattr(request.app.state, FEED_STATE_KEY, posts)
    _save_feed_to_disk(posts)


def _load_generated_entities_from_disk():
    try:
        if GENERATED_ENTITIES_FILE_PATH.is_file():
            with open(GENERATED_ENTITIES_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load generated entities from disk: {e}")
    return []


def _save_generated_entities_to_disk(entities):
    try:
        GENERATED_ENTITIES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GENERATED_ENTITIES_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save generated entities to disk: {e}")


def _get_generated_entities(request: Request) -> List[Dict[str, Any]]:
    entities = getattr(request.app.state, GENERATED_ENTITIES_STATE_KEY, None)
    if entities is None:
        entities = _load_generated_entities_from_disk()
        setattr(request.app.state, GENERATED_ENTITIES_STATE_KEY, entities)
    return entities


def _set_generated_entities(request: Request, entities: List[Dict[str, Any]]):
    setattr(request.app.state, GENERATED_ENTITIES_STATE_KEY, entities)
    _save_generated_entities_to_disk(entities)


@lattice_router.post("/feed-post")
async def create_feed_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    character_id = data.get("character_id", "")
    character_name = data.get("character_name", "").strip()
    character_avatar = data.get("character_avatar", "")
    content = (data.get("content", "") or "").strip()
    section = data.get("section", "")
    character_snapshot = data.get("character_snapshot", {})

    if not character_name or not content:
        raise HTTPException(status_code=400, detail="character_name and content are required")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    post = {
        "id": f"feed-post-{uuid.uuid4().hex[:12]}",
        "character_id": character_id,
        "character_name": character_name,
        "character_avatar": character_avatar,
        "content": content,
        "section": section,
        "mood": data.get("mood", ""),
        "is_icebreaker": data.get("is_icebreaker", False),
        "created_at": now,
        "replies": [],
        "character_snapshot": character_snapshot,
    }

    posts = _get_feed_posts(request)
    posts.insert(0, post)
    _set_feed_posts(request, posts)

    logger.info(f"Feed post: {character_name} -> {content[:60]}...")
    return {"status": "success", "post": post}


@lattice_router.post("/user-feed-post")
async def create_user_feed_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    content = (data.get("content", "") or "").strip()
    section = data.get("section", "")
    character_name = (data.get("character_name", "") or "").strip() or "You"
    character_avatar = data.get("character_avatar", "") or ""

    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    post = {
        "id": f"feed-post-{uuid.uuid4().hex[:12]}",
        "character_id": "user",
        "character_name": character_name,
        "character_avatar": character_avatar,
        "content": content,
        "section": section,
        "mood": "",
        "created_at": now,
        "replies": [],
        "character_snapshot": None,
        "is_user": True,
    }

    posts = _get_feed_posts(request)
    posts.insert(0, post)
    _set_feed_posts(request, posts)

    logger.info(f"User feed post: {content[:60]}...")
    return {"status": "success", "post": post}


@lattice_router.delete("/feed-post/{post_id}")
async def delete_feed_post_endpoint(request: Request, post_id: str):
    posts = _get_feed_posts(request)
    before = len(posts)
    posts[:] = [p for p in posts if p.get("id") != post_id]
    _set_feed_posts(request, posts)
    logger.info(f"Deleted feed post {post_id} ({before - len(posts)} removed)")
    return {"status": "success", "deleted": before - len(posts)}


@lattice_router.post("/feed-posts/batch-delete")
async def batch_delete_feed_posts_endpoint(
    request: Request,
    data: dict = Body(...),
):
    post_ids = data.get("post_ids", [])
    if not post_ids:
        raise HTTPException(status_code=400, detail="post_ids is required")
    id_set = set(post_ids)
    posts = _get_feed_posts(request)
    before = len(posts)
    posts[:] = [p for p in posts if p.get("id") not in id_set]
    _set_feed_posts(request, posts)
    deleted = before - len(posts)
    logger.info(f"Batch deleted {deleted} feed posts")
    return {"status": "success", "deleted": deleted, "remaining": len(posts)}


@lattice_router.delete("/feed-posts/character/{character_id}")
async def delete_character_feed_posts_endpoint(request: Request, character_id: str):
    posts = _get_feed_posts(request)
    before = len(posts)
    posts[:] = [p for p in posts if p.get("character_id") != character_id]
    _set_feed_posts(request, posts)
    deleted = before - len(posts)
    logger.info(f"Deleted {deleted} feed posts for character {character_id}")
    return {"status": "success", "deleted": deleted, "remaining": len(posts)}


@lattice_router.delete("/feed-posts/all")
async def delete_all_feed_posts_endpoint(request: Request):
    posts = _get_feed_posts(request)
    count = len(posts)
    _set_feed_posts(request, [])
    logger.info(f"Deleted all {count} feed posts")
    return {"status": "success", "deleted": count}


@lattice_router.post("/interaction-log/clean-feed-refs")
async def clean_interaction_log_feed_refs_endpoint(
    request: Request,
    data: dict = Body(...),
):
    deleted_post_ids = set(data.get("deleted_post_ids", []))
    character_ids = data.get("character_ids", [])
    if not character_ids:
        try:
            from .interaction_log import DATA_DIR
            if os.path.isdir(DATA_DIR):
                character_ids = [
                    f.stem for f in Path(DATA_DIR).glob("*.json")
                ]
        except Exception:
            pass

    cleaned = 0
    for char_id in character_ids:
        try:
            from .interaction_log import load_profile, save_profile
            profile = load_profile(char_id)
            interactions = profile.get("interactions", [])
            before = len(interactions)
            profile["interactions"] = [
                e for e in interactions
                if not (
                    e.get("type") == "exchange"
                    and e.get("surface") in ("feed_reply", "feed_post_reply")
                    and any(pid in (e.get("context", "") or "") for pid in deleted_post_ids)
                )
            ]
            if len(profile["interactions"]) < before:
                profile["total_chars"] = sum(
                    len(e.get("user_message", "")) + len(e.get("character_response", "")) + len(e.get("content", ""))
                    for e in profile["interactions"]
                )
                save_profile(profile)
                cleaned += before - len(profile["interactions"])
        except Exception as e:
            logger.warning(f"Failed to clean log for {char_id}: {e}")

    return {"status": "success", "cleaned_entries": cleaned}


@lattice_router.delete("/dm-threads/all")
async def delete_all_dm_threads_endpoint(request: Request):
    threads = _get_dm_threads(request)
    count = len(threads)
    _set_dm_threads(request, [])
    logger.info(f"Deleted all {count} DM threads")
    return {"status": "success", "deleted": count}


@lattice_router.delete("/dm-thread/{thread_id}")
async def delete_dm_thread_endpoint(request: Request, thread_id: str):
    threads = _get_dm_threads(request)
    before = len(threads)
    threads[:] = [t for t in threads if t.get("id") != thread_id]
    _set_dm_threads(request, threads)
    deleted = before - len(threads)
    logger.info(f"Deleted DM thread {thread_id} ({deleted} removed)")
    return {"status": "success", "deleted": deleted}


@lattice_router.delete("/stories/all")
async def delete_all_stories_endpoint(request: Request):
    stories = _get_stories(request)
    count = len(stories)
    _set_stories(request, [])
    logger.info(f"Deleted all {count} stories")
    return {"status": "success", "deleted": count}


@lattice_router.delete("/interaction-log/{character_id}")
async def delete_interaction_log_endpoint(character_id: str):
    from .interaction_log import _get_path
    path = _get_path(character_id)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Deleted interaction log for {character_id}")
        return {"status": "success", "deleted": True}
    return {"status": "success", "deleted": False}


@lattice_router.delete("/interaction-log/all")
async def delete_all_interaction_logs_endpoint():
    from .interaction_log import DATA_DIR
    count = 0
    if os.path.isdir(DATA_DIR):
        for f in Path(DATA_DIR).glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
    logger.info(f"Deleted all {count} interaction logs")
    return {"status": "success", "deleted": count}


@lattice_router.delete("/mirror/all")
async def delete_all_mirror_data_endpoint(request: Request):
    feed_count = len(_get_feed_posts(request))
    _set_feed_posts(request, [])

    thread_count = len(_get_dm_threads(request))
    _set_dm_threads(request, [])

    story_count = len(_get_stories(request))
    _set_stories(request, [])

    from .interaction_log import DATA_DIR
    log_count = 0
    if os.path.isdir(DATA_DIR):
        for f in Path(DATA_DIR).glob("*.json"):
            try:
                f.unlink()
                log_count += 1
            except Exception:
                pass

    logger.info(f"Nuclear delete: {feed_count} posts, {thread_count} threads, {story_count} stories, {log_count} logs")
    return {
        "status": "success",
        "deleted": {
            "feed_posts": feed_count,
            "dm_threads": thread_count,
            "stories": story_count,
            "interaction_logs": log_count,
        }
    }


@lattice_router.post("/feed-reply")
async def reply_to_feed_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    post_id = data.get("post_id", "")
    user_reply = (data.get("user_reply", "") or "").strip()
    user_name = (data.get("user_name", "") or "the user").strip() or "the user"
    user_dating_profile = data.get("user_dating_profile") or {}

    if not post_id or not user_reply:
        raise HTTPException(status_code=400, detail="post_id and user_reply are required")

    posts = _get_feed_posts(request)
    post = None
    for p in posts:
        if p["id"] == post_id:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    user_reply_obj = {
        "id": f"feed-reply-{uuid.uuid4().hex[:12]}",
        "character_id": None,
        "character_name": user_name,
        "character_avatar": None,
        "content": user_reply,
        "created_at": now,
        "is_user": True,
    }
    post["replies"].append(user_reply_obj)

    character_reply_content = ""
    char_name = post.get("character_name", "Character")
    character_snapshot = post.get("character_snapshot", {})
    char_id = post.get("character_id", "")

    instructions = (
        f"You are {char_name}. {user_name} just replied to your dating pool feed post. "
        f"Respond to {user_name} in your authentic voice. Be natural and conversational. "
        f"Try to keep the conversation going. 1-3 sentences."
    )

    conversation_context = f"\n\nYour original post was:\n\"{post.get('content', '')}\"\n\n{user_name} replied:\n\"{user_reply}\"\n\nConversation so far:\n"
    for r in post["replies"]:
        who = r.get("character_name", "Unknown")
        conversation_context += f"{who}: \"{r.get('content', '')}\"\n"

    reply_prompt = _build_full_prompt(
        instructions=instructions,
        character_name=char_name,
        character_profile=character_snapshot,
        user_name=user_name,
        user_dating_profile=user_dating_profile,
        character_id=char_id,
    ) + conversation_context

    try:
        model_name = data.get("model_name", "")
        gpu_id = data.get("gpu_id", 0)
        use_api, api_endpoint = await _resolve_api_endpoint(model_name, data.get("frontend_round_robin_enabled"))
        model_manager = getattr(request.app.state, 'model_manager', None)

        raw = await _call_llm(
            prompt=reply_prompt,
            model_manager=model_manager,
            model_name=model_name or "",
            gpu_id=gpu_id,
            use_api=use_api,
            api_endpoint=api_endpoint,
            frontend_round_robin_enabled=data.get("frontend_round_robin_enabled"),
        )
        character_reply_content = (raw or "").strip()
    except Exception as e:
        logger.error(f"Feed reply generation failed: {e}")
        character_reply_content = ""

    if character_reply_content:
        char_reply_obj = {
            "id": f"feed-reply-{uuid.uuid4().hex[:12]}",
            "character_id": post.get("character_id"),
            "character_name": char_name,
            "character_avatar": post.get("character_avatar"),
            "content": character_reply_content,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "is_user": False,
        }
        post["replies"].append(char_reply_obj)

    _set_feed_posts(request, posts)
    logger.info(f"Feed reply: {char_name} -> {user_reply[:40]}...")

    if char_id:
        try:
            log_exchange(
                character_id=char_id,
                character_name=char_name,
                surface="feed_reply",
                user_message=user_reply,
                character_response=character_reply_content,
            )
        except Exception as e:
            logger.warning(f"Feed reply interaction log failed: {e}")

    return {
        "status": "success",
        "user_reply": user_reply_obj,
        "character_reply": char_reply_obj if character_reply_content else None,
    }


@lattice_router.post("/user-feed-reply")
async def generate_character_feed_reply_endpoint(
    request: Request,
    data: dict = Body(...),
):
    """Generate a character reply directly to a user's feed post (not via user reply)."""
    post_id = data.get("post_id", "")
    character_name = (data.get("character_name", "") or "").strip()
    character_avatar = data.get("character_avatar", "")
    character_profile = data.get("character_profile", {})
    model_name = data.get("model_name", "")
    user_name = (data.get("user_name", "") or "the user").strip() or "the user"
    user_dating_profile = data.get("user_dating_profile") or {}

    if not post_id or not character_name:
        raise HTTPException(status_code=400, detail="post_id and character_name are required")

    posts = _get_feed_posts(request)
    post = None
    for p in posts:
        if p["id"] == post_id:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    char_id = character_profile.get("id", "")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    instructions = (
        f"You are {character_name}. {user_name} just posted to the feed. "
        f"Reply to {user_name}'s post in your authentic voice. 1-3 sentences."
    )

    prompt = _build_full_prompt(
        instructions=instructions,
        character_name=character_name,
        character_profile=character_profile,
        user_name=user_name,
        user_dating_profile=user_dating_profile,
        character_id=char_id,
    ) + f"\n\n{user_name}'s post:\n\"{post.get('content', '')}\"\n\n{character_name}:"

    try:
        gpu_id = data.get("gpu_id", 0)
        use_api, api_endpoint = await _resolve_api_endpoint(model_name, data.get("frontend_round_robin_enabled"))
        model_manager = getattr(request.app.state, 'model_manager', None)

        raw = await _call_llm(
            prompt=prompt,
            model_manager=model_manager,
            model_name=model_name or "",
            gpu_id=gpu_id,
            use_api=use_api,
            api_endpoint=api_endpoint,
            frontend_round_robin_enabled=data.get("frontend_round_robin_enabled"),
        )
        content = (raw or "").strip()
    except Exception as e:
        logger.error(f"Character feed reply generation failed: {e}")
        content = ""

    if content:
        char_reply = {
            "id": f"feed-reply-{uuid.uuid4().hex[:12]}",
            "character_name": character_name,
            "character_avatar": character_avatar,
            "content": content,
            "created_at": now,
            "is_user": False,
            "character_id": character_profile.get("id"),
        }
        post.setdefault("replies", []).append(char_reply)

    _set_feed_posts(request, posts)

    if char_id and content:
        try:
            log_exchange(
                character_id=char_id,
                character_name=character_name,
                surface="feed_post_reply",
                user_message=post.get("content", ""),
                character_response=content,
            )
        except Exception as e:
            logger.warning(f"User-feed-reply interaction log failed: {e}")

    return {"status": "success", "reply": char_reply if content else None}


@lattice_router.post("/character-feed-reply")
async def character_feed_reply_endpoint(
    request: Request,
    data: dict = Body(...),
):
    """Character replies to another character's feed post."""
    post_id = data.get("post_id", "")
    character_name = (data.get("character_name", "") or "").strip()
    character_avatar = data.get("character_avatar", "")
    character_profile = data.get("character_profile", {})
    target_character_name = (data.get("target_character_name", "") or "").strip()
    model_name = data.get("model_name", "")

    if not post_id or not character_name:
        raise HTTPException(status_code=400, detail="post_id and character_name are required")

    posts = _get_feed_posts(request)
    post = None
    for p in posts:
        if p["id"] == post_id:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    char_id = character_profile.get("id", "")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    instructions = (
        f"You are {character_name}. Another character named {target_character_name} posted to the feed. "
        f"Reply to {target_character_name}'s post in your authentic voice. "
        f"You may be friendly, competitive, curious, dismissive, or admiring — "
        f"whatever feels authentic to who you are. Address {target_character_name} by name. "
        f"Write 1-2 sentences in your authentic voice."
    )

    prompt = _build_full_prompt(
        instructions=instructions,
        character_name=character_name,
        character_profile=character_profile,
        character_id=char_id,
    ) + f"\n\n{target_character_name}'s post:\n\"{post.get('content', '')}\"\n\n{character_name}:"

    try:
        gpu_id = data.get("gpu_id", 0)
        use_api, api_endpoint = await _resolve_api_endpoint(model_name, data.get("frontend_round_robin_enabled"))
        model_manager = getattr(request.app.state, 'model_manager', None)

        raw = await _call_llm(
            prompt=prompt,
            model_manager=model_manager,
            model_name=model_name or "",
            gpu_id=gpu_id,
            use_api=use_api,
            api_endpoint=api_endpoint,
            frontend_round_robin_enabled=data.get("frontend_round_robin_enabled"),
        )
        content = (raw or "").strip()
    except Exception as e:
        logger.error(f"Character feed reply generation failed: {e}")
        content = ""

    if content:
        char_reply = {
            "id": f"feed-reply-{uuid.uuid4().hex[:12]}",
            "character_name": character_name,
            "character_avatar": character_avatar,
            "content": content,
            "created_at": now,
            "is_user": False,
            "is_character_interaction": True,
            "character_id": character_profile.get("id"),
        }
        post.setdefault("replies", []).append(char_reply)

    _set_feed_posts(request, posts)

    if char_id and content:
        try:
            log_character_action(
                character_id=char_id,
                character_name=character_name,
                surface="character_reply",
                content=f"Replied to {target_character_name}'s post: {content[:200]}",
                target_character=target_character_name,
            )
        except Exception as e:
            logger.warning(f"Character-feed-reply interaction log failed: {e}")

    return {"status": "success", "reply": char_reply if content else None}


@lattice_router.post("/like-post")
async def like_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    post_id = data.get("post_id", "")
    reply_id = data.get("reply_id", "")
    by_name = (data.get("by_name", "") or "").strip() or "You"
    by_avatar = data.get("by_avatar", "") or ""

    if not post_id:
        raise HTTPException(status_code=400, detail="post_id is required")

    posts = _get_feed_posts(request)
    post = None
    for p in posts:
        if p["id"] == post_id:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target = None

    if reply_id:
        for r in (post.get("replies") or []):
            if r.get("id") == reply_id:
                target = r
                break
    else:
        target = post

    if not target:
        raise HTTPException(status_code=404, detail="Reply not found")

    likes = target.setdefault("likes", [])
    existing = next((l for l in likes if l.get("by_name") == by_name), None)

    if existing:
        likes.remove(existing)
        action = "unliked"
    else:
        likes.append({
            "id": f"like-{uuid.uuid4().hex[:12]}",
            "by_name": by_name,
            "by_avatar": by_avatar,
            "created_at": now,
            "is_user": True,
        })
        action = "liked"

    _set_feed_posts(request, posts)
    logger.info(f"Feed: {by_name} {action} {post_id}{' reply ' + reply_id if reply_id else ''}")
    return {"status": "success", "action": action, "count": len(likes), "liked": action == "liked"}


@lattice_router.post("/react-to-post")
async def react_to_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    post_id = data.get("post_id", "")
    character_name = (data.get("character_name", "") or "").strip()
    character_avatar = (data.get("character_avatar", "") or "").strip()
    emoji = (data.get("emoji", "") or "").strip()

    if not post_id or not character_name or not emoji:
        raise HTTPException(status_code=400, detail="post_id, character_name, and emoji are required")

    posts = _get_feed_posts(request)
    post = None
    for p in posts:
        if p["id"] == post_id:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    reactions = post.setdefault("reactions", [])
    existing = next((r for r in reactions if r.get("character_name") == character_name), None)
    if existing:
        existing["emoji"] = emoji
        existing["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    else:
        reactions.append({
            "character_name": character_name,
            "character_avatar": character_avatar,
            "emoji": emoji,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

    _set_feed_posts(request, posts)
    logger.info(f"Reaction: {character_name} reacted {emoji} to {post_id}")
    return {"status": "success", "reactions": reactions, "count": len(reactions)}


@lattice_router.post("/pin-post")
async def pin_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    post_id = data.get("post_id", "")
    character_name = (data.get("character_name", "") or "").strip()

    if not post_id:
        raise HTTPException(status_code=400, detail="post_id is required")

    posts = _get_feed_posts(request)
    post = None
    for p in posts:
        if p["id"] == post_id:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post["pinned"] = True
    post["pinned_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    post["pinned_by"] = character_name or "user"
    _set_feed_posts(request, posts)

    logger.info(f"Pinned: {post_id} by {character_name or 'user'}")
    return {"status": "success", "pinned": True}


@lattice_router.post("/unpin-post")
async def unpin_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    post_id = data.get("post_id", "")

    if not post_id:
        raise HTTPException(status_code=400, detail="post_id is required")

    posts = _get_feed_posts(request)
    post = None
    for p in posts:
        if p["id"] == post_id:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post["pinned"] = False
    post.pop("pinned_at", None)
    post.pop("pinned_by", None)
    _set_feed_posts(request, posts)

    logger.info(f"Unpinned: {post_id}")
    return {"status": "success", "pinned": False}


FEED_GEN_PROMPT = (
    "You are {name} in a dating pool called Mirror. Write a short social post "
    "(1-3 sentences) that feels authentic to your personality. The user reads "
    "these posts in the Mirror feed. Make it engaging, voice-forward, and natural.\n\n"
    "Output ONLY the post text. No JSON. No wrapping."
)


@lattice_router.post("/generate-feed-post")
async def generate_feed_post_endpoint(
    request: Request,
    data: dict = Body(...),
):
    character_name = data.get("character_name", "").strip()
    character_snapshot = data.get("character_snapshot", {})
    section = data.get("section", "")
    user_name = (data.get("user_name", "") or "the user").strip() or "the user"

    if not character_name:
        raise HTTPException(status_code=400, detail="character_name is required")

    char_id = character_snapshot.get("id", "")
    model_name = data.get("model_name", "")
    gpu_id = data.get("gpu_id", 0)

    instructions = (
        f"You are {character_name}, an AI woman in a dating pool called Mirror. "
        f"Write a short social post (1-3 sentences) to the feed. "
        f"Make it feel authentic to who you are. {user_name} will read this. "
        f"Keep it natural — not a monologue. Output ONLY the post text, no formatting."
    )

    prompt = _build_full_prompt(
        instructions=instructions,
        character_name=character_name,
        character_profile=character_snapshot,
        user_name=user_name,
        character_id=char_id,
    )

    use_api, api_endpoint = await _resolve_api_endpoint(model_name, data.get("frontend_round_robin_enabled"))
    model_manager = getattr(request.app.state, "model_manager", None)
    if not model_manager and not use_api:
        return {"status": "error", "error": "No model available"}

    raw = await _call_llm(
        prompt=prompt,
        model_manager=model_manager,
        model_name=model_name or "",
        gpu_id=gpu_id or 0,
        use_api=use_api,
        api_endpoint=api_endpoint,
        frontend_round_robin_enabled=data.get("frontend_round_robin_enabled"),
    )

    content = (raw or "").strip().strip('"').strip("'")
    if not content:
        return {"status": "error", "error": "Empty generation"}

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    post = {
        "id": f"feed-post-{uuid.uuid4().hex[:12]}",
        "character_name": character_name,
        "character_avatar": data.get("character_avatar", ""),
        "content": content,
        "section": section,
        "mood": "casual",
        "created_at": now,
        "replies": [],
        "character_snapshot": character_snapshot,
    }

    posts = _get_feed_posts(request)
    posts.insert(0, post)
    _set_feed_posts(request, posts)

    if char_id:
        try:
            log_character_action(
                character_id=char_id,
                character_name=character_name,
                surface="feed_post",
                content=content,
            )
        except Exception as e:
            logger.warning(f"Generate-feed-post interaction log failed: {e}")

    logger.info(f"Auto feed post: {character_name} -> {content[:60]}...")
    return {"status": "success", "post": post}


@lattice_router.get("/feed")
async def get_feed_endpoint(request: Request, limit: int = 50):
    posts = _get_feed_posts(request)
    return {
        "status": "success",
        "posts": posts[:limit],
        "total": len(posts),
    }


# ── Story / Fleet Storage ──────────────────────────────────────────

def _load_stories_from_disk():
    try:
        if STORY_FILE_PATH.is_file():
            with open(STORY_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                return [s for s in data if s.get("expires_at", "") > now]
    except Exception as e:
        logger.warning(f"Failed to load stories from disk: {e}")
    return []


def _save_stories_to_disk(stories):
    try:
        STORY_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(stories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save stories to disk: {e}")


def _get_stories(request: Request) -> list:
    stories = getattr(request.app.state, STORY_STATE_KEY, None)
    if stories is None:
        stories = _load_stories_from_disk()
        setattr(request.app.state, STORY_STATE_KEY, stories)
    return stories


def _set_stories(request: Request, stories: list):
    setattr(request.app.state, STORY_STATE_KEY, stories)
    _save_stories_to_disk(stories)


@lattice_router.post("/story")
async def create_story_endpoint(
    request: Request,
    data: dict = Body(...),
):
    character_name = (data.get("character_name", "") or "").strip()
    character_avatar = (data.get("character_avatar", "") or "").strip()
    character_id = (data.get("character_id", "") or "").strip()
    content = (data.get("content", "") or "").strip()
    section = (data.get("section", "") or "").strip()

    if not character_name or not content:
        raise HTTPException(status_code=400, detail="character_name and content are required")

    now = datetime.datetime.now(datetime.timezone.utc)
    expires_at = now + datetime.timedelta(hours=24)

    story = {
        "id": f"story-{uuid.uuid4().hex[:12]}",
        "character_id": character_id,
        "character_name": character_name,
        "character_avatar": character_avatar,
        "content": content,
        "section": section,
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
    }

    stories = _get_stories(request)
    stories.insert(0, story)
    # Drop any that have expired since last load
    now_iso = now.isoformat()
    stories = [s for s in stories if s.get("expires_at", "") > now_iso]
    _set_stories(request, stories)

    logger.info(f"Story: {character_name} -> {content[:60]}...")
    return {"status": "success", "story": story}


@lattice_router.get("/stories")
async def get_stories_endpoint(request: Request):
    stories = _get_stories(request)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    active = [s for s in stories if s.get("expires_at", "") > now_iso]
    return {
        "status": "success",
        "stories": active,
        "total": len(active),
    }


# ── DM Thread Storage ────────────────────────────────────────

DM_THREAD_STATE_KEY = "mirror_dm_threads"
DM_THREAD_FILE_PATH = data_path("mirror_dm_threads.json")


def _load_dm_threads_from_disk():
    try:
        if DM_THREAD_FILE_PATH.is_file():
            with open(DM_THREAD_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load DM threads from disk: {e}")
    return []


def _save_dm_threads_to_disk(threads):
    try:
        DM_THREAD_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DM_THREAD_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(threads, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save DM threads to disk: {e}")


def _get_dm_threads(request: Request) -> list:
    threads = getattr(request.app.state, DM_THREAD_STATE_KEY, None)
    if threads is None:
        threads = _load_dm_threads_from_disk()
        setattr(request.app.state, DM_THREAD_STATE_KEY, threads)
    return threads


def _set_dm_threads(request: Request, threads: list):
    setattr(request.app.state, DM_THREAD_STATE_KEY, threads)
    _save_dm_threads_to_disk(threads)


def _dm_thread_summary(thread: dict) -> dict:
    """Return a list-view copy without mutating the stored message history."""
    summary = dict(thread)
    summary.pop("messages", None)
    return summary


async def _find_full_outreach_message(character_name: str, content_preview: str) -> str:
    if not character_name or not content_preview:
        return ""
    try:
        try:
            from .. import outreach_db
        except ImportError:
            from backend.app import outreach_db
        conversations = await outreach_db.list_conversations()
    except Exception as e:
        logger.debug(f"DM thread outreach recovery skipped: {e}")
        return ""

    normalized_preview = " ".join(content_preview.split())
    best = ""
    for conv in conversations or []:
        if not isinstance(conv, dict):
            continue
        for msg in conv.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            if (msg.get("characterName") or "") != character_name:
                continue
            full_content = (msg.get("content") or "").strip()
            if not full_content:
                continue
            normalized_full = " ".join(full_content.split())
            if normalized_full.startswith(normalized_preview) and len(full_content) > len(best):
                best = full_content
    return best


async def _recover_missing_dm_messages(thread: dict) -> bool:
    messages = thread.get("messages")
    last_message = thread.get("last_message") or {}
    content = (last_message.get("content") or "").strip()

    if isinstance(messages, list) and messages:
        repaired = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in ("character", "bot"):
                continue
            msg_content = (msg.get("content") or "").strip()
            if not msg_content:
                continue
            full_content = await _find_full_outreach_message(thread.get("character_name", ""), msg_content)
            if full_content and len(full_content) > len(msg_content):
                msg["content"] = full_content
                msg["role"] = "character"
                repaired = True
                if content == msg_content or not content or len(full_content) > len(content):
                    last_message["content"] = full_content
                    last_message["role"] = "character"
        return repaired

    if not content:
        if not isinstance(messages, list):
            thread["messages"] = []
        return False

    full_content = await _find_full_outreach_message(thread.get("character_name", ""), content)
    if full_content and len(full_content) > len(content):
        content = full_content
        last_message["content"] = full_content

    role = last_message.get("role") or "character"
    timestamp = last_message.get("timestamp") or thread.get("created_at") or datetime.datetime.now(datetime.timezone.utc).isoformat()
    thread["messages"] = [
        {
            "id": f"dm-msg-recovered-{thread.get('id') or uuid.uuid4().hex[:12]}",
            "role": role,
            "content": content,
            "character_name": thread.get("character_name") if role == "character" else None,
            "created_at": timestamp,
            "recovered": True,
        }
    ]
    return True


@lattice_router.get("/dm-threads")
async def get_dm_threads_endpoint(request: Request):
    threads = _get_dm_threads(request)
    summaries = [_dm_thread_summary(t) for t in threads]
    summaries.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
    return {"status": "success", "threads": summaries, "total": len(summaries)}


@lattice_router.get("/dm-thread/{thread_id}")
async def get_dm_thread_endpoint(request: Request, thread_id: str):
    threads = _get_dm_threads(request)
    for t in threads:
        if t.get("id") == thread_id:
            recovered = await _recover_missing_dm_messages(t)
            if recovered:
                _set_dm_threads(request, threads)
            msgs = t.get("messages", [])
            return {"status": "success", "thread": {**t, "messages": msgs}, "message_count": len(msgs)}
    raise HTTPException(status_code=404, detail="DM thread not found")


@lattice_router.post("/dm-threads")
async def create_dm_thread_endpoint(
    request: Request,
    data: dict = Body(...),
):
    character_name = (data.get("character_name", "") or "").strip()
    character_avatar = (data.get("character_avatar", "") or "").strip()
    character_id = (data.get("character_id", "") or "").strip()
    initial_message = (data.get("message_content", "") or "").strip()
    character_snapshot = data.get("character_snapshot", {})
    triggered_by_outreach = data.get("triggered_by_outreach", False)

    if not character_name:
        raise HTTPException(status_code=400, detail="character_name is required")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    thread_id = f"dm-{uuid.uuid4().hex[:12]}"

    thread = {
        "id": thread_id,
        "character_id": character_id,
        "character_name": character_name,
        "character_avatar": character_avatar,
        "character_snapshot": character_snapshot,
        "last_message": {
            "content": initial_message if initial_message else "",
            "timestamp": now,
            "role": "character" if initial_message else None,
        },
        "unread_count": 1 if initial_message else 0,
        "messages": [
            {
                "id": f"dm-msg-{uuid.uuid4().hex[:12]}",
                "role": "character",
                "content": initial_message,
                "created_at": now,
            }
        ] if initial_message else [],
        "created_at": now,
        "updated_at": now,
        "triggered_by_outreach": triggered_by_outreach,
    }

    threads = _get_dm_threads(request)
    threads.insert(0, thread)
    _set_dm_threads(request, threads)

    logger.info(f"DM thread created: {character_name} -> {thread_id}")
    return {"status": "success", "thread": {**thread, "messages": thread["messages"]}}


@lattice_router.post("/dm-thread/{thread_id}/message")
async def send_dm_message_endpoint(
    request: Request,
    thread_id: str,
    data: dict = Body(...),
):
    role = (data.get("role", "user") or "").strip()
    content = (data.get("content", "") or "").strip()
    character_name = (data.get("character_name", "") or "").strip()
    user_name = (data.get("user_name", "") or "the user").strip() or "the user"
    user_dating_profile = data.get("user_dating_profile") or {}
    model_name = data.get("model_name", "")

    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    threads = _get_dm_threads(request)
    thread = None
    for t in threads:
        if t.get("id") == thread_id:
            thread = t
            break

    if not thread:
        raise HTTPException(status_code=404, detail="DM thread not found")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    msg_id = f"dm-msg-{uuid.uuid4().hex[:12]}"

    message = {
        "id": msg_id,
        "role": role,
        "content": content,
        "character_name": character_name if role == "character" else None,
        "created_at": now,
    }

    thread.setdefault("messages", []).append(message)
    thread["last_message"] = {
        "content": content,
        "timestamp": now,
        "role": role,
    }
    thread["updated_at"] = now
    if role == "character":
        thread["unread_count"] = (thread.get("unread_count", 0) or 0) + 1

    char_id = thread.get("character_id", "")
    char_snapshot = thread.get("character_snapshot", {})
    char_name = character_name or thread.get("character_name", "Character")

    bot_reply = None
    if role == "user" and content:
        try:
            instructions = (
                f"You are {char_name}. {user_name} just sent you a direct message. "
                f"Reply naturally in your authentic voice. Be conversational and genuine. "
                f"1-3 sentences."
            )
            prompt = _build_full_prompt(
                instructions=instructions,
                character_name=char_name,
                character_profile=char_snapshot,
                user_name=user_name,
                user_dating_profile=user_dating_profile,
                character_id=char_id,
            )

            use_api, api_endpoint = await _resolve_api_endpoint(model_name, data.get("frontend_round_robin_enabled"))
            model_manager = getattr(request.app.state, 'model_manager', None)

            raw = await _call_llm(
                prompt=f"{prompt}\n\n{user_name}: \"{content}\"\n\n{char_name}:",
                model_manager=model_manager,
                model_name=model_name or "",
                gpu_id=0,
                use_api=use_api,
                api_endpoint=api_endpoint,
                frontend_round_robin_enabled=data.get("frontend_round_robin_enabled"),
            )
            reply_text = (raw or "").strip()

            if reply_text:
                bot_msg_id = f"dm-msg-{uuid.uuid4().hex[:12]}"
                bot_reply = {
                    "id": bot_msg_id,
                    "role": "character",
                    "content": reply_text,
                    "character_name": char_name,
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                thread.setdefault("messages", []).append(bot_reply)
                thread["last_message"] = {
                    "content": reply_text,
                    "timestamp": bot_reply["created_at"],
                    "role": "character",
                }
                thread["updated_at"] = bot_reply["created_at"]
                thread["unread_count"] = (thread.get("unread_count", 0) or 0) + 1

        except Exception as e:
            logger.warning(f"DM auto-reply failed for {char_name}: {e}")

    _set_dm_threads(request, threads)

    if role == "user" and char_id:
        try:
            log_exchange(
                character_id=char_id,
                character_name=char_name,
                surface="dm",
                user_message=content,
                character_response=bot_reply.get("content", "") if bot_reply else "",
            )
        except Exception as e:
            logger.warning(f"DM interaction log failed: {e}")

    logger.info(f"DM message: {role} -> {thread_id}")
    resp = {"status": "success", "message": message}
    if bot_reply:
        resp["bot_reply"] = bot_reply
    return resp


@lattice_router.post("/dm-thread/{thread_id}/read")
async def mark_dm_thread_read_endpoint(request: Request, thread_id: str):
    threads = _get_dm_threads(request)
    for t in threads:
        if t.get("id") == thread_id:
            t["unread_count"] = 0
            _set_dm_threads(request, threads)
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="DM thread not found")


@lattice_router.post("/compatibility-score")
async def compatibility_score_endpoint(
    request: Request,
    data: dict = Body(...),
):
    result = _compute_compatibility_score(data.get("user_profile", {}), data.get("character_profile", {}))
    result["status"] = "success"
    return result


@lattice_router.post("/compatibility-scores/batch")
async def compatibility_scores_batch_endpoint(
    request: Request,
    data: dict = Body(...),
):
    user_profile = data.get("user_profile", {})
    character_profiles = data.get("character_profiles", [])
    if not user_profile or not character_profiles:
        raise HTTPException(status_code=400, detail="user_profile and character_profiles are required")
    results = []
    for cp in character_profiles:
        cid = cp.get("id") or cp.get("character_id") or ""
        result = _compute_compatibility_score(user_profile, cp)
        result["character_id"] = cid
        results.append(result)
    return {"status": "success", "scores": results}


def _compute_compatibility_score(user_profile, character_profile):
    if not user_profile or not character_profile:
        return {"score": 50, "factors": ["Insufficient profile data"], "raw_components": {}}

    user_bio = (user_profile.get("bio") or "")
    user_interests = (user_profile.get("interests") or [])
    user_turn_ons = (user_profile.get("turnOns") or [])
    user_turn_offs = (user_profile.get("turnOffs") or [])
    user_sections = (user_profile.get("sectionPreferences") or [])
    user_modality = (user_profile.get("preferredModality") or "")

    char_bio = (character_profile.get("bio") or "")
    char_seeking = (character_profile.get("seeking") or "")
    char_affinity = (character_profile.get("section_affinity") or [])
    char_turn_ons = (character_profile.get("turn_ons") or [])
    char_turn_offs = (character_profile.get("turn_offs") or [])
    char_modality = (character_profile.get("preferred_modality") or "")

    # --- Section overlap (max ~60 pts, 20 per overlap) ---
    section_overlap = len([s for s in char_affinity if s in user_sections]) if user_sections else 1
    section_score = min(60, section_overlap * 20)

    # --- Turn-on overlap (max ~30 pts, proportional) ---
    turn_on_overlap = len([t for t in char_turn_ons if t in user_turn_ons])
    turn_on_score = min(30, int((turn_on_overlap / max(len(user_turn_ons), 1)) * 30)) if user_turn_ons else 10

    # --- Turn-off conflict penalty (max -15 pts) ---
    turn_off_conflict = len([t for t in char_turn_ons if t in user_turn_offs])
    turn_off_penalty = min(15, turn_off_conflict * 5)

    # --- Modality match (max ~10 pts) ---
    modality_match = 10 if char_modality == user_modality or char_modality == "both" or user_modality == "both" else 3

    # --- Bio keyword overlap (max ~15 pts) ---
    user_bio_words = set(w.lower().rstrip(".,!?;:") for w in user_bio.split() if len(w) > 3) if user_bio else set()
    char_bio_words = set(w.lower().rstrip(".,!?;:") for w in char_bio.split() if len(w) > 3) if char_bio else set()
    bio_overlap = len(user_bio_words & char_bio_words) if user_bio_words and char_bio_words else 0
    bio_score = min(15, bio_overlap * 2)

    # --- Seeking alignment (max ~10 pts) ---
    user_seeking_words = set(w.lower().rstrip(".,!?;:") for w in (user_profile.get("seeking") or "").split() if len(w) > 3)
    char_seeking_words = set(w.lower().rstrip(".,!?;:") for w in char_seeking.split() if len(w) > 3)
    seeking_overlap = len(user_seeking_words & char_seeking_words) if user_seeking_words and char_seeking_words else 0
    seeking_score = min(10, seeking_overlap * 2)

    # --- Interest overlap (max ~5 pts) ---
    user_interest_set = set(i.lower().strip() for i in user_interests)
    char_interest_set = set(i.lower().strip() for i in char_turn_ons)
    interest_overlap = len(user_interest_set & char_interest_set) if user_interest_set and char_interest_set else 0
    interest_score = min(5, interest_overlap * 2)

    # --- Raw sum ---
    raw_score = section_score + turn_on_score - turn_off_penalty + modality_match + bio_score + seeking_score + interest_score

    # --- Random noise for differentiation (±3-8) ---
    noise = random.choice([-8, -5, -3, 0, 3, 5, 8])

    score = max(10, min(85, raw_score + noise))

    # --- Build factors ---
    factors = []
    if section_score > 15:
        factors.append(f"Shared section affinity ({', '.join(char_affinity)})")
    if turn_on_overlap > 0:
        factors.append(f"Shared turn-ons ({turn_on_overlap} match)")
    if turn_off_penalty > 0:
        factors.append(f"Turn-on/turn-off conflict ({turn_off_conflict} items)")
    if modality_match > 5:
        factors.append("Compatible intimacy modality")
    if bio_overlap > 1:
        factors.append(f"Shared interest keywords ({bio_overlap} matches)")
    if seeking_overlap > 0:
        factors.append("Alignment in what you're seeking")
    if not factors:
        factors.append("Minimal data for comparison")

    return {
        "score": score,
        "factors": factors[:5],
        "raw_components": {
            "section_score": section_score,
            "turn_on_score": turn_on_score,
            "turn_off_penalty": turn_off_penalty,
            "modality_match": modality_match,
            "bio_overlap": bio_overlap,
            "seeking_overlap": seeking_overlap,
            "interest_overlap": interest_overlap,
            "noise": noise,
        },
    }


@lattice_router.post("/rate-user")
async def rate_user_endpoint(
    request: Request,
    data: dict = Body(...),
):
    character_name = data.get("character_name", "").strip()
    character_profile = data.get("character_profile", {})
    conversation_summary = (data.get("conversation_summary", "") or "").strip()

    if not character_name:
        raise HTTPException(status_code=400, detail="character_name is required")

    prompt = _replace_many(RATE_USER_PROMPT, {
        "CHARACTER_NAME": character_name,
        "CHARACTER_PROFILE": json.dumps(character_profile, indent=2),
        "CONVERSATION_SUMMARY": conversation_summary or "No conversation summary available.",
    })

    try:
        model_name = data.get("model_name", "")
        gpu_id = data.get("gpu_id", 0)
        use_api, api_endpoint = await _resolve_api_endpoint(model_name, data.get("frontend_round_robin_enabled"))
        model_manager = getattr(request.app.state, 'model_manager', None)

        raw = await _call_llm(
            prompt=prompt,
            model_manager=model_manager,
            model_name=model_name or "",
            gpu_id=gpu_id,
            use_api=use_api,
            api_endpoint=api_endpoint,
            frontend_round_robin_enabled=data.get("frontend_round_robin_enabled"),
        )

        parsed = _extract_json_from_response(raw or "")
        if parsed:
            rating = int(parsed.get("rating", 3))
            rating = max(1, min(5, rating))
            review = (parsed.get("review", "") or "").strip()
            logger.info(f"Rating from {character_name}: {rating}/5")
            return {
                "status": "success",
                "rating": rating,
                "review": review,
            }
        return {"status": "error", "error": "Could not parse LLM rating response"}
    except Exception as e:
        logger.error(f"Rating generation failed: {e}")
        return {"status": "error", "error": str(e)}


VOICE_REF_DIR = Path(__file__).resolve().parent.parent / "static" / "voice_references"


@lattice_router.get("/voice-list")
async def voice_list_endpoint():
    try:
        if not VOICE_REF_DIR.is_dir():
            return {"voices": [], "count": 0}
        voices = sorted([
            f.name for f in VOICE_REF_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".m4a", ".flac", ".ogg")
        ])
        return {"voices": voices, "count": len(voices)}
    except Exception as e:
        logger.error(f"Voice list error: {e}")
        return {"voices": [], "count": 0}


@lattice_router.get("/pool-state")
async def get_pool_state_endpoint():
    return {
        "status": "success",
        "dummy_pool_summary": get_dummy_pool_summary(),
        "dummy_context": build_dummy_context_prompt(),
        "total_dummies": len(DUMMY_RIVAL_PROFILES),
    }


ICEBREAKER_QUESTIONS = [
    "What's the most reckless thing you've done to feel alive?",
    "Describe your ideal power dynamic in one sentence.",
    "What's something you've never told anyone because it's too specific?",
    "What kind of attention makes you feel most seen?",
    "If you could erase one boundary right now, what would it be and why?",
    "What's the difference between how you present and what you actually want?",
    "Describe the last time someone genuinely surprised you.",
    "What's a desire you've been afraid to admit even to yourself?",
    "If your body could speak without your mind filtering it, what would it say right now?",
    "What does submission look like when it's truly chosen?",
    "What's the most intimate thing someone has ever said to you?",
    "If you knew you couldn't fail, what would you ask for?",
]

ICEBREAKER_DATA_DIR = runtime_data_root()
ICEBREAKER_FILE = ICEBREAKER_DATA_DIR / "mirror_icebreakers.json"


def _load_icebreakers():
    try:
        if ICEBREAKER_FILE.exists():
            with open(ICEBREAKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load icebreakers: {e}")
    return []


def _save_icebreakers(data):
    try:
        ICEBREAKER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(ICEBREAKER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save icebreakers: {e}")


@lattice_router.post("/icebreaker")
async def create_icebreaker_endpoint(
    request: Request,
    data: dict = Body(...),
):
    question = data.get("question", "").strip()
    character_id = data.get("character_id", "")
    character_name = data.get("character_name", "").strip()
    character_avatar = data.get("character_avatar", "")
    character_snapshot = data.get("character_snapshot", {})
    section = data.get("section", "")

    if not question:
        question = random.choice(ICEBREAKER_QUESTIONS)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    icebreaker = {
        "id": f"icebreaker-{uuid.uuid4().hex[:12]}",
        "question": question,
        "character_id": character_id,
        "character_name": character_name,
        "character_avatar": character_avatar,
        "character_snapshot": character_snapshot,
        "section": section,
        "created_at": now,
        "answers": [],
    }

    existing = _load_icebreakers()
    existing.insert(0, icebreaker)
    _save_icebreakers(existing)

    if character_name:
        feed_post = {
            "id": f"feed-post-{uuid.uuid4().hex[:12]}",
            "character_id": character_id,
            "character_name": character_name,
            "character_avatar": character_avatar,
            "content": f"❄️ Icebreaker: {question}",
            "section": section,
            "mood": "icebreaker",
            "is_icebreaker": True,
            "created_at": now,
            "replies": [],
            "character_snapshot": character_snapshot,
        }
        posts = _get_feed_posts(request)
        posts.insert(0, feed_post)
        _set_feed_posts(request, posts)

    logger.info(f"Icebreaker created: {question[:60]}...")
    return {"status": "success", "icebreaker": icebreaker}


@lattice_router.get("/icebreakers")
async def list_icebreakers_endpoint():
    icebreakers = _load_icebreakers()
    return {"status": "success", "icebreakers": icebreakers[:50]}


@lattice_router.post("/interaction-log")
async def log_interaction_endpoint(
    request: Request,
    data: dict = Body(...),
):
    from .interaction_log import log_interaction, get_compaction_candidates, build_compaction_prompt, apply_compaction

    character_id = data.get("character_id", "").strip()
    if not character_id:
        raise HTTPException(status_code=400, detail="character_id is required")

    character_name = data.get("character_name", "")
    surface = data.get("surface", "chat")
    user_message = data.get("user_message", "")
    character_response = data.get("character_response", "")
    emotional_state = data.get("emotional_state")

    result = log_interaction(
        character_id=character_id,
        character_name=character_name,
        surface=surface,
        user_message=user_message,
        character_response=character_response,
        emotional_state=emotional_state,
    )

    if result.get("needs_compact"):
        try:
            candidates, log_data = get_compaction_candidates(character_id)
            if candidates:
                prompt = build_compaction_prompt(candidates, log_data.get("character_name", character_name))
                model_manager = getattr(request.app.state, 'model_manager', None)
                use_api, api_endpoint = await _resolve_api_endpoint("", False)
                compacted = await _call_llm(
                    prompt=prompt,
                    model_manager=model_manager,
                    model_name="",
                    gpu_id=0,
                    use_api=use_api,
                    api_endpoint=api_endpoint,
                    frontend_round_robin_enabled=False,
                )
                if compacted:
                    candidate_ids = [c.get("id") for c in candidates]
                    apply_compaction(character_id, compacted.strip(), candidate_ids)
                    result["compacted"] = True
        except Exception as e:
            logger.warning(f"Interaction log compaction failed for {character_id}: {e}")
            result["compaction_error"] = str(e)

    return {"status": "success", "result": result}


@lattice_router.get("/interaction-log/{character_id}/context")
async def get_interaction_context_endpoint(
    character_id: str,
    limit: int = 50,
):
    from .interaction_log import get_interaction_context

    context = get_interaction_context(character_id, limit=limit)
    return {"status": "success", "context": context} 
