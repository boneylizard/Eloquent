"""
Chess OAuth SQLite layer: Eloquent users, linked Lichess/Chess.com accounts, imported PGNs.
"""
import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chess_auth.db"


def _token_encrypt(plain: str, secret: bytes) -> str:
    """Simple XOR obfuscation with key derived from secret. Not for high-security; avoids plaintext in DB."""
    key = hashlib.sha256(secret).digest()
    plain_b = plain.encode("utf-8")
    out = bytearray(len(plain_b))
    for i, c in enumerate(plain_b):
        out[i] = c ^ key[i % 32]
    return base64.b64encode(bytes(out)).decode("ascii")


def _token_decrypt(cipher: str, secret: bytes) -> str:
    key = hashlib.sha256(secret).digest()
    raw = base64.b64decode(cipher.encode("ascii"))
    out = bytearray(len(raw))
    for i, c in enumerate(raw):
        out[i] = c ^ key[i % 32]
    return out.decode("utf-8")


def _get_secret() -> bytes:
    s = os.environ.get("CHESS_OAUTH_SECRET", "eloquent-chess-auth-dev-secret")
    return hashlib.sha256(s.encode()).digest()


class ChessAuthDb:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or DB_PATH

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS eloquent_user (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chess_account (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    eloquent_user_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    platform_user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    access_token_enc TEXT NOT NULL,
                    token_expires_at TEXT,
                    scopes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(eloquent_user_id, platform),
                    FOREIGN KEY (eloquent_user_id) REFERENCES eloquent_user(id)
                );
                CREATE INDEX IF NOT EXISTS idx_chess_account_user ON chess_account(eloquent_user_id);

                CREATE TABLE IF NOT EXISTS imported_game (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chess_account_id INTEGER NOT NULL,
                    platform_game_id TEXT NOT NULL,
                    pgn_text TEXT NOT NULL,
                    played_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(chess_account_id, platform_game_id),
                    FOREIGN KEY (chess_account_id) REFERENCES chess_account(id)
                );
                CREATE INDEX IF NOT EXISTS idx_imported_game_account ON imported_game(chess_account_id);

                CREATE TABLE IF NOT EXISTS oauth_state (
                    state TEXT PRIMARY KEY,
                    code_verifier TEXT NOT NULL,
                    eloquent_user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            await db.commit()
        logger.info("Chess auth DB initialized at %s", self.path)

    async def ensure_user(self, user_id: str) -> str:
        """Create eloquent_user if not exists; return user_id."""
        if not user_id or not user_id.strip():
            user_id = secrets.token_hex(16)
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO eloquent_user (id, created_at) VALUES (?, ?)",
                (user_id, now),
            )
            await db.commit()
        return user_id

    async def save_oauth_state(self, state: str, code_verifier: str, eloquent_user_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO oauth_state (state, code_verifier, eloquent_user_id, created_at) VALUES (?, ?, ?, ?)",
                (state, code_verifier, eloquent_user_id, now),
            )
            await db.commit()

    async def consume_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Return { code_verifier, eloquent_user_id } and delete state. Returns None if invalid/expired."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            row = await db.execute(
                "SELECT code_verifier, eloquent_user_id FROM oauth_state WHERE state = ?",
                (state,),
            )
            r = await row.fetchone()
            if not r:
                return None
            await db.execute("DELETE FROM oauth_state WHERE state = ?", (state,))
            await db.commit()
            return {"code_verifier": r["code_verifier"], "eloquent_user_id": r["eloquent_user_id"]}

    async def upsert_chess_account(
        self,
        eloquent_user_id: str,
        platform: str,
        platform_user_id: str,
        username: str,
        access_token: str,
        token_expires_at: Optional[str] = None,
        scopes: Optional[str] = None,
    ) -> int:
        """Insert or replace account for this user+platform. Returns chess_account id."""
        secret = _get_secret()
        enc = _token_encrypt(access_token, secret)
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO chess_account (
                    eloquent_user_id, platform, platform_user_id, username,
                    access_token_enc, token_expires_at, scopes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(eloquent_user_id, platform) DO UPDATE SET
                    platform_user_id = excluded.platform_user_id,
                    username = excluded.username,
                    access_token_enc = excluded.access_token_enc,
                    token_expires_at = excluded.token_expires_at,
                    scopes = excluded.scopes,
                    updated_at = excluded.updated_at
                """,
                (eloquent_user_id, platform, platform_user_id, username, enc, token_expires_at, scopes, now, now),
            )
            cur = await db.execute(
                "SELECT id FROM chess_account WHERE eloquent_user_id = ? AND platform = ?",
                (eloquent_user_id, platform),
            )
            row = await cur.fetchone()
            await db.commit()
            return row[0] if row else 0

    async def get_account_token(self, chess_account_id: int) -> Optional[str]:
        """Return decrypted access_token for account, or None if not found."""
        async with aiosqlite.connect(self.path) as db:
            row = await db.execute(
                "SELECT access_token_enc FROM chess_account WHERE id = ?",
                (chess_account_id,),
            )
            r = await row.fetchone()
        if not r:
            return None
        return _token_decrypt(r[0], _get_secret())

    async def get_linked_accounts(self, eloquent_user_id: str) -> List[Dict[str, Any]]:
        """Return list of { id, platform, username, token_expires_at } (no token)."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT id, platform, username, token_expires_at, updated_at
                   FROM chess_account WHERE eloquent_user_id = ? ORDER BY platform""",
                (eloquent_user_id,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "platform": r["platform"],
                "username": r["username"],
                "token_expires_at": r["token_expires_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    async def unlink_account(self, eloquent_user_id: str, platform: str) -> bool:
        """Remove chess account for user+platform. Returns True if deleted."""
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "DELETE FROM chess_account WHERE eloquent_user_id = ? AND platform = ?",
                (eloquent_user_id, platform),
            )
            await db.commit()
            return cur.rowcount > 0

    async def add_imported_games(
        self,
        chess_account_id: int,
        games: List[Dict[str, Any]],
    ) -> int:
        """games: list of { platform_game_id, pgn_text, played_at? }. Returns count inserted (duplicates skipped)."""
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        async with aiosqlite.connect(self.path) as db:
            for g in games:
                pid = g.get("platform_game_id") or ""
                pgn = g.get("pgn_text") or ""
                played = g.get("played_at")
                try:
                    await db.execute(
                        """INSERT OR IGNORE INTO imported_game
                           (chess_account_id, platform_game_id, pgn_text, played_at, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (chess_account_id, pid, pgn, played, now),
                    )
                    if db.total_changes:
                        count += 1
                except Exception:
                    pass
            await db.commit()
        return count

    async def get_imported_games(
        self,
        eloquent_user_id: str,
        platform: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return imported games for user, optionally by platform. Newest first."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            if platform:
                cursor = await db.execute(
                    """SELECT g.id, g.platform_game_id, g.pgn_text, g.played_at, g.created_at, a.platform, a.username
                       FROM imported_game g
                       JOIN chess_account a ON g.chess_account_id = a.id
                       WHERE a.eloquent_user_id = ? AND a.platform = ?
                       ORDER BY g.created_at DESC LIMIT ?""",
                    (eloquent_user_id, platform, limit),
                )
            else:
                cursor = await db.execute(
                    """SELECT g.id, g.platform_game_id, g.pgn_text, g.played_at, g.created_at, a.platform, a.username
                       FROM imported_game g
                       JOIN chess_account a ON g.chess_account_id = a.id
                       WHERE a.eloquent_user_id = ?
                       ORDER BY g.created_at DESC LIMIT ?""",
                    (eloquent_user_id, limit),
                )
            rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "platform_game_id": r["platform_game_id"],
                "pgn_text": r["pgn_text"],
                "played_at": r["played_at"],
                "created_at": r["created_at"],
                "platform": r["platform"],
                "username": r["username"],
            }
            for r in rows
        ]


chess_auth_db = ChessAuthDb()
