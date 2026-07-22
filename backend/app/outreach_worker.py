"""
Background polling: run due outreach rules on the primary backend (port 8000).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid
from typing import Any, Dict, List, Set

import httpx

from . import outreach_assets, outreach_db
from .outreach_runtime import broadcast_event, send_web_push_all

logger = logging.getLogger(__name__)

OUTREACH_MIN_INTERVAL_MINUTES = 1
_processing: Set[str] = set()

_IM_END = "<|{}|>\n".format("im_end")


def _build_system_prompt(character: Dict[str, Any], user_name: str = "User") -> str:
    if not character:
        return "You are a helpful assistant."
    char_name = character.get("name") or "Character"

    def rep(t: Any) -> str:
        if not t:
            return ""
        s = str(t)
        return (
            s.replace("{{char}}", char_name)
            .replace("{{user}}", user_name)
            .replace("{{Char}}", char_name)
            .replace("{{User}}", user_name)
        )

    personality = rep(character.get("personality"))
    description = rep(character.get("description"))
    scenario = rep(character.get("scenario")) if character.get("scenario") else ""
    speech = rep(character.get("speech_style")) if character.get("speech_style") else ""
    background = rep(character.get("background")) if character.get("background") else ""
    parts: List[str] = [
        f"You are {char_name}, {description}.",
        f"PERSONALITY: {personality}",
        f"BACKGROUND: {background}",
    ]
    if scenario:
        parts.append(f"SCENARIO: {scenario}")
    if speech:
        parts.append(f"SPEAKING STYLE: {speech}")
    parts.append(
        f"IMPORTANT: Stay in character at all times. Respond as {char_name} would, "
        "maintaining the defined personality and speech patterns."
    )
    ed = character.get("example_dialogue")
    if isinstance(ed, list) and ed:
        ex_lines = []
        for msg in ed:
            role = msg.get("role")
            content = rep(msg.get("content"))
            who = char_name if role == "character" else user_name if role == "user" else "User"
            ex_lines.append(f"{who}: {content}")
        parts.append("EXAMPLE DIALOGUE:\n" + "\n".join(ex_lines))
    return "\n\n".join(parts)


def _format_chatml(messages: List[Dict[str, Any]], system_msg: str) -> str:
    chunks: List[str] = [f"<|im_start|>system\n{system_msg}{_IM_END}"]
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        tag = "user" if role == "user" else "assistant"
        chunks.append(f"<|im_start|>{tag}\n{content}{_IM_END}")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def _clean_output(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    for pref in ("<|im_start|>assistant", "[/INST]", "</s>"):
        if t.lower().startswith(pref.lower()):
            t = t[len(pref) :].strip()
    return t


def _normalize_user_profile(gen_settings: Dict[str, Any]) -> Dict[str, Any]:
    raw = gen_settings.get("user_profile")
    if not isinstance(raw, dict):
        raw = {}
    uid = raw.get("id") or gen_settings.get("user_profile_id")
    if not uid or str(uid).lower() == "anonymous":
        return {}
    out = {**raw, "id": str(uid)}
    return out


async def _enrich_system_for_outreach(
    system_base: str,
    character: Dict[str, Any],
    gen_settings: Dict[str, Any],
    user_query: str,
) -> tuple[str, str]:
    """
    Match chat UI: optional USER MEMORY PROFILE (get_all) + agentic character memory.
    Returns (system_text, user_profile_reinforcement_snippet).
    """
    uprof = _normalize_user_profile(gen_settings)
    uid = uprof.get("id")
    memory_base = (gen_settings.get("memory_api_base") or "").rstrip("/")
    if not uid or not memory_base:
        return system_base, ""

    char_name = character.get("name") or "Character"
    user_name = uprof.get("name") or uprof.get("username") or "User"

    def _tag(txt: str) -> str:
        if not txt:
            return ""
        return (
            str(txt)
            .replace("{{char}}", char_name)
            .replace("{{user}}", user_name)
            .replace("{{Char}}", char_name)
            .replace("{{User}}", user_name)
        )

    chunks: List[str] = [system_base]
    reinforcement = ""

    direct_inj = gen_settings.get("direct_profile_injection")
    if direct_inj is None:
        direct_inj = True

    if direct_inj:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.get(f"{memory_base}/memory/get_all", params={"user_id": uid})
                if r.status_code == 200:
                    data = r.json()
                    mems = data.get("memories") or []
                    if mems:
                        bullets = []
                        for mem in mems:
                            cat = str((mem.get("category") or "memory")).replace("_", " ")
                            imp = mem.get("importance")
                            try:
                                imp_s = f"{float(imp):.1f}" if imp is not None else "N/A"
                            except (TypeError, ValueError):
                                imp_s = "N/A"
                            content = _tag(str(mem.get("content") or ""))
                            bullets.append(f"• {content} (Category: {cat}, Importance: {imp_s})")
                        chunks.append("USER MEMORY PROFILE:\n" + "\n".join(bullets))
                        reinforcement = "\n".join(bullets[:5])
        except Exception as e:
            logger.warning("outreach memory get_all failed: %s", e)

    if character.get("id"):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                params: Dict[str, Any] = {
                    "user_id": uid,
                    "character_id": character["id"],
                    "use_rag": "true",
                }
                q = (user_query or "").strip()
                if q:
                    params["query"] = q[:800]
                un = uprof.get("name") or uprof.get("username")
                if un:
                    params["user_name"] = str(un)
                r = await client.get(f"{memory_base}/memory/agentic", params=params)
                if r.status_code == 200:
                    data = r.json()
                    ctx = (data.get("formatted_context") or "").strip()
                    if ctx:
                        chunks.append(ctx)
        except Exception as e:
            logger.warning("outreach memory agentic failed: %s", e)

    return "\n\n".join(chunks), reinforcement


async def _generate_local(app, prompt: str, model_name: str, character: Dict[str, Any], gen_settings: Dict[str, Any]) -> str:
    port = getattr(app.state, "port", 8000)
    base = f"http://127.0.0.1:{port}"
    uprof = _normalize_user_profile(gen_settings)
    if not uprof.get("id"):
        uprof = {"id": str(gen_settings.get("user_profile_id") or "anonymous")}

    dip = gen_settings.get("direct_profile_injection")
    if dip is None:
        dip = True

    reinforcement = (gen_settings.get("_outreach_reinforcement") or "").strip()

    body: Dict[str, Any] = {
        "directProfileInjection": bool(dip),
        "prompt": prompt,
        "model_name": model_name,
        "max_tokens": int(gen_settings.get("max_tokens") or 4096),
        "temperature": float(gen_settings.get("temperature", 0.7)),
        "top_p": float(gen_settings.get("top_p", 0.9)),
        "top_k": int(gen_settings.get("top_k", 50)),
        "repetition_penalty": float(gen_settings.get("repetition_penalty", 1.1)),
        "frequency_penalty": float(gen_settings.get("frequency_penalty", 0)),
        "presence_penalty": float(gen_settings.get("presence_penalty", 0)),
        "gpu_id": 0,
        "stream": False,
        "active_character": character,
        "userProfile": uprof,
        "injectTimestamp": bool(gen_settings.get("injectTimestamp", False)),
    }
    if reinforcement:
        body["userProfileReinforcement"] = reinforcement
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{base}/generate", json=body)
            if r.status_code != 200:
                logger.error("outreach generate HTTP %s: %s", r.status_code, (r.text or "")[:500])
                return f"[Outreach error: HTTP {r.status_code}]"
            data = r.json()
            return _clean_output(data.get("text") or "") or "[No response]"
    except Exception as e:
        logger.exception("outreach generate failed")
        return f"[Outreach error: {e}]"


async def process_rule(app, rule: Dict[str, Any], gen_defaults: Dict[str, Any]) -> None:
    rid = rule.get("id")
    if not rid or rid in _processing:
        return
    # Hard guard: only execute rules that still exist and are enabled in DB.
    live_rule = await outreach_db.get_rule_payload(rid)
    if not live_rule:
        return
    rule = live_rule
    character = rule.get("characterSnapshot")
    if not character:
        logger.warning("outreach rule %s missing characterSnapshot — open Settings and Save rules once from the app", rid)
        return
    prompt_text = (rule.get("prompt") or "").strip()
    if not prompt_text:
        return
    model_name = rule.get("modelName") or gen_defaults.get("primary_model")
    if not model_name:
        logger.warning("outreach rule %s: no modelName and no primary_model default", rid)
        return
    _processing.add(rid)
    try:
        now = dt.datetime.now(dt.timezone.utc)
        now_iso = now.isoformat().replace("+00:00", "Z")
        interval = int(rule.get("intervalMinutes") or 45)
        interval = max(OUTREACH_MIN_INTERVAL_MINUTES, interval)

        # Each scheduled run is its own two-message thread (no stacking the same prompt).
        conv_id = f"outreach-conv-{uuid.uuid4().hex[:12]}"
        char_id = character.get("id")
        conv = {
            "id": conv_id,
            "name": f"Outreach: {character.get('name') or 'Character'}",
            "messages": [],
            "characterIds": {"primary": char_id, "secondary": None, "user": None},
            "activeCharacterIds": [char_id] if char_id else [],
            "activeCharacterWeights": {char_id: 100} if char_id else {},
            "multiRoleContext": "",
            "created": now_iso,
            "requiresTitle": False,
            "agenticMemoryEnabled": False,
            "outreachRuleId": rid,
            # So the client can hydrate persona/avatar if the character was edited or not in local roster yet
            "characterSnapshot": character,
        }

        user_mid = f"u-{uuid.uuid4().hex[:12]}"
        user_msg = {"id": user_mid, "role": "user", "content": prompt_text, "isScheduledOutreach": True}
        history = [user_msg]
        merge_gen = {**gen_defaults, **(rule.get("generationSettings") or {})}
        uprof = _normalize_user_profile(merge_gen)
        user_name = (uprof.get("name") or uprof.get("username") or "User") if uprof.get("id") else "User"
        system_msg = _build_system_prompt(character, user_name=user_name)
        system_msg, reinforcement = await _enrich_system_for_outreach(system_msg, character, merge_gen, prompt_text)
        merge_gen = {**merge_gen, "_outreach_reinforcement": reinforcement}
        formatted = _format_chatml(history[-12:], system_msg)
        bot_content = await _generate_local(app, formatted, model_name, character, merge_gen)
        bot_mid = f"b-{uuid.uuid4().hex[:12]}"
        bot_msg = {
            "id": bot_mid,
            "role": "bot",
            "content": bot_content,
            "modelId": "primary",
            "characterId": character.get("id"),
            "characterName": character.get("name"),
            "avatar": character.get("avatar"),
            "isScheduledOutreach": True,
        }
        messages_out = history + [bot_msg]
        port = getattr(app.state, "port", 8000)
        base_url = f"http://127.0.0.1:{port}"
        img_msg = outreach_assets.build_image_message(rid, character, base_url=base_url)
        attachment_url = None
        if img_msg:
            messages_out.append(img_msg)
            attachment_url = img_msg.get("imagePath")
        conv["messages"] = messages_out
        await outreach_db.save_conversation(conv)

        uid = uprof.get("id")
        if uid and char_id:
            try:
                mem_base = merge_gen.get("memory_api_base", "").rstrip("/") or f"http://127.0.0.1:{getattr(app.state, 'port', 8000)}"
                async with httpx.AsyncClient(timeout=30.0) as mc:
                    await mc.post(f"{mem_base}/memory/agentic/process", json={
                        "user_id": uid,
                        "character_id": char_id,
                        "character_name": character.get("name", ""),
                        "user_message": f"[Scheduled Outreach] Character outreach message was sent: {bot_content[:500]}",
                        "ai_response": "[stored] Scheduled outreach message recorded.",
                    })
            except Exception:
                logger.debug("Scheduled outreach memory write skipped (non-critical)")

        next_run = now + dt.timedelta(minutes=interval)
        next_iso = next_run.isoformat().replace("+00:00", "Z")
        await outreach_db.update_rule_schedule(rid, next_iso, now_iso, None)

        preview = " ".join((bot_content or "").split())[:200]
        event = {
            "type": "outreach_message",
            "ruleId": rid,
            "ruleName": (rule.get("name") or "").strip() or "Scheduled Outreach",
            "conversationId": conv_id,
            "messageId": bot_mid,
            "characterName": character.get("name") or "Character",
            "characterAvatar": character.get("avatar"),
            "preview": preview,
            "attachmentImageUrl": attachment_url,
            "conversation": conv,
        }
        await broadcast_event(event)
        await send_web_push_all(event)
    finally:
        _processing.discard(rid)


async def outreach_loop(app) -> None:
    await asyncio.sleep(4)
    while True:
        try:
            if getattr(app.state, "port", 8000) != 8000:
                await asyncio.sleep(30)
                continue
            if getattr(app.state, "outreach_enabled", True) is not True:
                await asyncio.sleep(10)
                continue
            last = float(getattr(app.state, "outreach_conv_cleanup_ts", 0) or 0)
            now_ts = time.time()
            if now_ts - last >= 3600:
                app.state.outreach_conv_cleanup_ts = now_ts
                removed = await outreach_db.delete_outreach_conversations_older_than_hours(72)
                if removed:
                    logger.info("outreach: removed %s conversation(s) older than 72h", removed)
            now_ms = dt.datetime.now(dt.timezone.utc).timestamp() * 1000
            due = await outreach_db.list_rules_due(now_ms)
            gen_defaults = getattr(app.state, "outreach_generation_defaults", {}) or {}
            for rule in due[:2]:
                await process_rule(app, rule, gen_defaults)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("outreach_loop tick error")
        await asyncio.sleep(18)
