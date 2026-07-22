"""
SQLite persistence for server-side scheduled outreach (rules, conversations, push subscriptions).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from .runtime_paths import data_path

logger = logging.getLogger(__name__)

DB_PATH = data_path("outreach.db")

_schema_initialized = False


async def initialize() -> None:
    global _schema_initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS outreach_rules (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                next_run_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS outreach_conversations (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS outreach_push_subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL UNIQUE,
                p256dh TEXT NOT NULL,
                auth TEXT NOT NULL,
                public_origin TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        await db.commit()
    _schema_initialized = True
    logger.info("outreach DB initialized at %s", DB_PATH)


async def upsert_rule(rule_id: str, payload: Dict[str, Any], enabled: bool, next_run_at: Optional[str]) -> None:
    import datetime as dt

    now = dt.datetime.utcnow().isoformat() + "Z"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO outreach_rules (id, payload, enabled, next_run_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload = excluded.payload,
              enabled = excluded.enabled,
              next_run_at = excluded.next_run_at,
              updated_at = excluded.updated_at
            """,
            (rule_id, json.dumps(payload), 1 if enabled else 0, next_run_at, now),
        )
        await db.commit()


async def delete_rule(rule_id: str) -> None:
    from . import outreach_assets

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM outreach_rules WHERE id = ?", (rule_id,))
        await db.commit()
    outreach_assets.clear_rule_images(rule_id)


async def replace_all_rules_from_sync(rules: List[Dict[str, Any]]) -> None:
    """Replace rule set with payload from frontend (full sync)."""
    import datetime as dt

    from . import outreach_assets

    now = dt.datetime.utcnow().isoformat() + "Z"
    incoming_ids: List[str] = [r.get("id") for r in rules if r.get("id")]

    async with aiosqlite.connect(DB_PATH) as db:
        # Safety guard: check how many rules currently exist on the server.
        cur = await db.execute("SELECT id FROM outreach_rules")
        existing_rows = await cur.fetchall()
        existing_ids = {row[0] for row in existing_rows}

        # If the frontend is sending FEWER rules than we have, only delete rules
        # that are explicitly absent from the sync — never wipe rules the frontend
        # didn't even know about (e.g. due to a partial-state render during a crash).
        # Exception: if the incoming list is empty AND we have rules, that likely
        # means a bug — refuse to delete anything in that case.
        if len(incoming_ids) == 0 and len(existing_ids) > 0:
            logger.warning(
                "[outreach] sync received 0 rules but %d exist on server — refusing to wipe",
                len(existing_ids),
            )
            return

        # Delete only rules whose IDs are not in the incoming set.
        ids_to_delete = existing_ids - set(incoming_ids)
        for dead_id in ids_to_delete:
            await db.execute("DELETE FROM outreach_rules WHERE id = ?", (dead_id,))

        # Upsert each incoming rule.
        active_ids: List[str] = []
        for r in rules:
            rid = r.get("id")
            if not rid:
                continue
            active_ids.append(rid)
            enabled = bool(r.get("enabled"))
            next_run = r.get("nextRunAt")
            payload = dict(r)
            # One-shot outreach chats per run; never persist a sticky conversation id from clients.
            payload.pop("conversationId", None)
            await db.execute(
                """
                INSERT INTO outreach_rules (id, payload, enabled, next_run_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload = excluded.payload,
                  enabled = excluded.enabled,
                  next_run_at = excluded.next_run_at,
                  updated_at = excluded.updated_at
                """,
                (rid, json.dumps(payload), 1 if enabled else 0, next_run, now),
            )
        await db.commit()
    outreach_assets.prune_orphan_asset_dirs(active_ids)


async def list_rules_due(now_ms: float) -> List[Dict[str, Any]]:
    """Return enabled rules whose next_run_at is due (ISO string or missing = due)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT id, payload, enabled, next_run_at FROM outreach_rules WHERE enabled = 1"
        )
        rows = await cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["payload"])
        next_run = row["next_run_at"] or payload.get("nextRunAt")
        if not next_run:
            continue
        try:
            t = __import__("datetime").datetime.fromisoformat(next_run.replace("Z", "+00:00"))
            ts = t.timestamp() * 1000
        except Exception:
            continue
        if ts <= now_ms:
            out.append(payload)
    return out


async def update_rule_schedule(rule_id: str, next_run_at: str, last_run_at: str, conversation_id: Optional[str]) -> None:
    import datetime as dt

    now = dt.datetime.utcnow().isoformat() + "Z"
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT payload FROM outreach_rules WHERE id = ?", (rule_id,))
        row = await cur.fetchone()
        if not row:
            return
        payload = json.loads(row[0])
        payload["nextRunAt"] = next_run_at
        payload["lastRunAt"] = last_run_at
        if conversation_id:
            payload["conversationId"] = conversation_id
        else:
            payload.pop("conversationId", None)
        await db.execute(
            "UPDATE outreach_rules SET payload = ?, next_run_at = ?, updated_at = ? WHERE id = ?",
            (json.dumps(payload), next_run_at, now, rule_id),
        )
        await db.commit()


async def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT payload FROM outreach_conversations WHERE id = ?", (conv_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])


async def save_conversation(conv: Dict[str, Any]) -> None:
    import datetime as dt

    cid = conv.get("id")
    if not cid:
        return
    now = dt.datetime.utcnow().isoformat() + "Z"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO outreach_conversations (id, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
            """,
            (cid, json.dumps(conv), now),
        )
        await db.commit()


async def list_conversations() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT payload FROM outreach_conversations")
        rows = await cur.fetchall()
    return [json.loads(r[0]) for r in rows]


async def list_rule_payloads() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT payload FROM outreach_rules")
        rows = await cur.fetchall()
    return [json.loads(r[0]) for r in rows]


async def get_rule_payload(rule_id: str) -> Optional[Dict[str, Any]]:
    if not rule_id:
        return None
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT payload, enabled FROM outreach_rules WHERE id = ?",
            (rule_id,),
        )
        row = await cur.fetchone()
    if not row:
        return None
    enabled = int(row[1] or 0) == 1
    if not enabled:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


async def add_push_subscription(endpoint: str, p256dh: str, auth: str, public_origin: str) -> None:
    import datetime as dt

    now = dt.datetime.utcnow().isoformat() + "Z"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO outreach_push_subscriptions (endpoint, p256dh, auth, public_origin, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
              p256dh = excluded.p256dh,
              auth = excluded.auth,
              public_origin = excluded.public_origin,
              created_at = excluded.created_at
            """,
            (endpoint, p256dh, auth, public_origin or "", now),
        )
        await db.commit()


async def list_push_subscriptions() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT endpoint, p256dh, auth, public_origin FROM outreach_push_subscriptions"
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def delete_push_subscription(endpoint: str) -> None:
    if not endpoint:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM outreach_push_subscriptions WHERE endpoint = ?", (endpoint,))
        await db.commit()


async def delete_conversation(conv_id: str) -> None:
    if not conv_id:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM outreach_conversations WHERE id = ?", (conv_id,))
        await db.commit()


async def delete_outreach_conversations_older_than_hours(hours: int) -> int:
    """Remove server-side outreach transcripts that were never claimed (or old copies)."""
    import datetime as dt

    if hours < 1:
        hours = 1
    cutoff = (dt.datetime.utcnow() - dt.timedelta(hours=hours)).isoformat() + "Z"
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM outreach_conversations WHERE updated_at < ?", (cutoff,)
        )
        await db.commit()
        try:
            return int(cur.rowcount or 0)
        except Exception:
            return 0
