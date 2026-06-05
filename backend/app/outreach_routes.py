"""
HTTP API: sync outreach rules from clients, SSE stream, Web Push subscription, snapshots.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import outreach_assets, outreach_db
from .outreach_runtime import get_vapid_public_b64, register_sse_listener, unregister_sse_listener

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/outreach/v1", tags=["outreach"])


class OutreachSyncBody(BaseModel):
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    generationDefaults: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class PushSubscribeBody(BaseModel):
    subscription: Dict[str, Any]
    publicOrigin: str = ""


@router.post("/sync")
async def outreach_sync(request: Request, body: OutreachSyncBody):
    await outreach_db.replace_all_rules_from_sync(body.rules)
    if body.generationDefaults:
        request.app.state.outreach_generation_defaults = body.generationDefaults
    if body.enabled is not None:
        request.app.state.outreach_enabled = bool(body.enabled)
    return {"ok": True, "count": len(body.rules)}


@router.get("/rules")
async def outreach_list_rules(request: Request):
    convs = await outreach_db.list_conversations()
    rules_out = await outreach_db.list_rule_payloads()
    for r in rules_out:
        rid = r.get("id")
        if rid:
            r["imageCount"] = outreach_assets.rule_image_count(rid)
    return {"rules": rules_out, "conversations": convs, "enabled": bool(getattr(request.app.state, "outreach_enabled", True))}


@router.get("/vapid-public-key")
async def outreach_vapid():
    try:
        key = get_vapid_public_b64()
        return {"publicKey": key}
    except Exception as e:
        logger.warning("VAPID: %s", e)
        return {"publicKey": None, "error": str(e)}


@router.post("/push/subscribe")
async def outreach_push_subscribe(body: PushSubscribeBody):
    sub = body.subscription or {}
    endpoint = sub.get("endpoint")
    keys = sub.get("keys") or {}
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")
    if not endpoint or not p256dh or not auth:
        return {"ok": False, "error": "invalid subscription"}
    await outreach_db.add_push_subscription(endpoint, p256dh, auth, body.publicOrigin or "")
    return {"ok": True}


@router.get("/events/stream")
async def outreach_events_stream(request: Request):
    q = register_sse_listener()

    async def gen():
        try:
            yield "event: ping\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {line.strip()}\n\n"
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            unregister_sse_listener(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/conversation/{conv_id}")
async def outreach_get_conversation(conv_id: str):
    c = await outreach_db.get_conversation(conv_id)
    if not c:
        return {"conversation": None}
    return {"conversation": c}


@router.delete("/conversation/{conv_id}")
async def outreach_delete_conversation(conv_id: str):
    await outreach_db.delete_conversation(conv_id)
    return {"ok": True}


@router.post("/run/{rule_id}")
async def outreach_run_now(request: Request, rule_id: str):
    from .outreach_worker import process_rule

    payloads = await outreach_db.list_rule_payloads()
    rule = next((r for r in payloads if r.get("id") == rule_id), None)
    if not rule:
        raise HTTPException(status_code=404, detail="rule not found on server — save outreach rules in Settings (sync)")
    gen = getattr(request.app.state, "outreach_generation_defaults", {}) or {}
    await process_rule(request.app, rule, gen)
    return {"ok": True}


@router.get("/rules/{rule_id}/images")
async def outreach_rule_images_info(rule_id: str):
    count = outreach_assets.rule_image_count(rule_id)
    return {"ok": True, "ruleId": rule_id, "imageCount": count}


@router.post("/rules/{rule_id}/images")
async def outreach_rule_images_upload(
    rule_id: str,
    files: List[UploadFile] = File(...),
    replace: bool = Query(True),
):
    if not rule_id.strip():
        raise HTTPException(status_code=400, detail="rule id required")
    pairs = []
    for f in files:
        if not f.filename:
            continue
        data = await f.read()
        pairs.append((f.filename, data))
    if not pairs:
        raise HTTPException(status_code=400, detail="no image files provided")
    saved = outreach_assets.save_uploaded_images(rule_id, pairs, replace=replace)
    if saved == 0:
        raise HTTPException(status_code=400, detail="no supported images (png, jpg, gif, webp)")
    return {"ok": True, "ruleId": rule_id, "imageCount": saved}


@router.delete("/rules/{rule_id}/images")
async def outreach_rule_images_clear(rule_id: str):
    outreach_assets.clear_rule_images(rule_id)
    return {"ok": True, "ruleId": rule_id, "imageCount": 0}
