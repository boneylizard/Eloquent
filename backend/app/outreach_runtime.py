"""
SSE fan-out and optional Web Push for server-side outreach events.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Set

from .runtime_paths import runtime_data_root

logger = logging.getLogger(__name__)

_sse_queues: Set[asyncio.Queue] = set()

_DATA_DIR = runtime_data_root()
# pywebpush uses py_vapid: a *file path* uses Vapid.from_file() (reliable).
# A PEM *string* uses Vapid.from_string() and often fails (ASN.1 / deserialize on Windows).
VAPID_PEM_PATH = _DATA_DIR / "outreach_vapid_private.pem"
VAPID_LEGACY_JSON = _DATA_DIR / "outreach_vapid.json"


def register_sse_listener() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _sse_queues.add(q)
    return q


def unregister_sse_listener(q: asyncio.Queue) -> None:
    _sse_queues.discard(q)


async def broadcast_event(event: Dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    stale: List[asyncio.Queue] = []
    for q in list(_sse_queues):
        try:
            q.put_nowait(line)
        except Exception:
            stale.append(q)
    for q in stale:
        _sse_queues.discard(q)


def _load_pem_private_key(pem_bytes: bytes):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    return serialization.load_pem_private_key(pem_bytes, password=None, backend=default_backend())


def _public_b64url_from_private_key(private_key) -> str:
    from cryptography.hazmat.primitives import serialization

    pub = private_key.public_key()
    pub_raw = pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return base64.urlsafe_b64encode(pub_raw).decode("utf-8").rstrip("=")


def _write_new_vapid_pem() -> None:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    VAPID_PEM_PATH.write_bytes(priv_pem)
    logger.info("Generated VAPID EC private key at %s", VAPID_PEM_PATH)


def _migrate_legacy_json_pem() -> bool:
    """Return True if a usable PEM was written from outreach_vapid.json."""
    if not VAPID_LEGACY_JSON.exists():
        return False
    try:
        legacy = json.loads(VAPID_LEGACY_JSON.read_text(encoding="utf-8"))
    except Exception:
        return False
    pem_str = (legacy.get("privateKeyPem") or "").strip()
    if not pem_str:
        return False
    pem_bytes = pem_str.encode("utf-8").replace(b"\r\n", b"\n")
    try:
        _load_pem_private_key(pem_bytes)
    except Exception:
        logger.info("Legacy VAPID JSON contains unusable key material; will regenerate PEM")
        return False
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    VAPID_PEM_PATH.write_bytes(pem_bytes)
    logger.info("Migrated VAPID private key from %s to %s", VAPID_LEGACY_JSON, VAPID_PEM_PATH)
    try:
        pub = _public_b64url_from_private_key(_load_pem_private_key(pem_bytes))
        VAPID_LEGACY_JSON.write_text(json.dumps({"publicKey": pub}), encoding="utf-8")
    except Exception:
        pass
    return True


def _ensure_vapid_pem_path() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if VAPID_PEM_PATH.exists():
        try:
            _load_pem_private_key(VAPID_PEM_PATH.read_bytes())
            return VAPID_PEM_PATH
        except Exception as e:
            logger.warning("Existing VAPID PEM invalid (%s); regenerating", e)
            VAPID_PEM_PATH.unlink(missing_ok=True)

    if _migrate_legacy_json_pem() and VAPID_PEM_PATH.exists():
        return VAPID_PEM_PATH

    _write_new_vapid_pem()
    return VAPID_PEM_PATH


def _vapid_aud_for_endpoint(endpoint: str) -> str:
    """FCM/Mozilla/etc. require JWT aud = origin of the push service URL."""
    try:
        u = urlparse((endpoint or "").strip())
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}"
    except Exception:
        pass
    return "https://fcm.googleapis.com"


def get_vapid_public_b64() -> str:
    pem_path = _ensure_vapid_pem_path()
    sk = _load_pem_private_key(pem_path.read_bytes())
    return _public_b64url_from_private_key(sk)


async def send_web_push_all(event: Dict[str, Any]) -> None:
    try:
        from pywebpush import WebPushException, webpush
    except ImportError:
        logger.debug("pywebpush not installed; skipping push")
        return
    from . import outreach_db

    subs = await outreach_db.list_push_subscriptions()
    if not subs:
        return
    try:
        pem_path = _ensure_vapid_pem_path()
    except Exception as e:
        logger.warning("VAPID unavailable: %s", e)
        return

    # File path => pywebpush uses Vapid.from_file (reliable). String PEM uses from_string (fragile).
    vapid_key_arg: str = str(pem_path.resolve())
    if not os.path.isfile(vapid_key_arg):
        logger.error("VAPID PEM path is not a file: %s", vapid_key_arg)
        return

    vapid_sub = "mailto:noreply@example.com"

    for row in subs:
        origin = (row.get("public_origin") or "").strip().rstrip("/")
        cid = event.get("conversationId") or ""
        mid = event.get("messageId") or ""
        open_url = f"{origin}/?outreach=1&cid={cid}&mid={mid}" if origin else "/"
        attach = event.get("attachmentImageUrl")
        icon = event.get("characterAvatar") if _http_url(event.get("characterAvatar")) else None
        if _http_url(attach):
            icon = attach
        raw_payload = {
            "title": event.get("characterName") or "Eloquent",
            "body": f"Eloquent\nsent you a message:\n{(event.get('preview') or '')[:120]}",
            "url": open_url,
            "conversationId": cid,
            "messageId": mid,
            "icon": icon,
            "image": attach if _http_url(attach) else None,
        }
        payload = {k: v for k, v in raw_payload.items() if v is not None}
        endpoint_raw = (row.get("endpoint") or "").strip()
        p256dh = (row.get("p256dh") or "").strip()
        auth = (row.get("auth") or "").strip()
        subscription_info = {
            "endpoint": endpoint_raw,
            "keys": {"p256dh": p256dh, "auth": auth},
        }
        endpoint_url = endpoint_raw
        vapid_claims = {"sub": vapid_sub, "aud": _vapid_aud_for_endpoint(endpoint_raw)}
        try:
            webpush(
                subscription_info=subscription_info,
                data=json.dumps(payload, ensure_ascii=False),
                vapid_private_key=vapid_key_arg,
                vapid_claims=vapid_claims,
                ttl=86400,
            )
        except WebPushException as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None) if resp is not None else None
            detail = ""
            if resp is not None:
                try:
                    detail = (resp.text or "").strip()[:300]
                except Exception:
                    detail = ""
            msg_l = str(e).lower()
            vapid_mismatch = "vapid credentials" in msg_l and "subscriptions" in msg_l
            gone = status in (404, 410) or "gone" in msg_l
            invalid = status == 400
            if endpoint_url and (gone or vapid_mismatch or invalid):
                await outreach_db.delete_push_subscription(endpoint_url)
                logger.info(
                    "Removed push subscription (HTTP %s). Re-enable Browser push for outreach. detail=%r",
                    status,
                    detail or msg_l[:200],
                )
            else:
                ep_short = (endpoint_url[:96] + "...") if len(endpoint_url) > 96 else endpoint_url
                logger.warning("WebPush failed status=%s detail=%r endpoint=%r err=%s", status, detail, ep_short, e)
        except Exception as e:
            logger.warning("WebPush error: %s", e)


def _http_url(s: Any) -> bool:
    if not s or not isinstance(s, str):
        return False
    return s.startswith("http://") or s.startswith("https://")
