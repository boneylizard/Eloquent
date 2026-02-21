"""
OAuth and account routes for Lichess/Chess.com linking and game import.
"""
import logging
import os
import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from . import chess_oauth
from .chess_auth_db import chess_auth_db

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/auth", tags=["chess-auth"])

# Redirect URI for Lichess must match exactly. Backend receives callback.
def _backend_base_url(request: Request) -> str:
    base = os.environ.get("CHESS_OAUTH_BACKEND_URL")
    if base:
        return base.rstrip("/")
    # Infer from request
    url = str(request.base_url).rstrip("/")
    return url


def _frontend_redirect_base(request: Request) -> str:
    base = os.environ.get("CHESS_OAUTH_FRONTEND_URL", "http://localhost:5173")
    return base.rstrip("/")


def _get_eloquent_user_id(x_user_id: Optional[str] = Header(None, alias="X-Eloquent-User-Id")) -> str:
    """Return user id from header or generate one (caller should persist in localStorage)."""
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()
    return secrets.token_hex(16)


@auth_router.get("/lichess/authorize")
async def lichess_authorize(
    request: Request,
    x_eloquent_user_id: Optional[str] = Header(None, alias="X-Eloquent-User-Id"),
):
    """Start Lichess OAuth: return { url, state }. Frontend redirects user to url."""
    user_id = _get_eloquent_user_id(x_eloquent_user_id)
    await chess_auth_db.ensure_user(user_id)

    state = secrets.token_urlsafe(24)
    code_verifier, code_challenge = chess_oauth.generate_lichess_pkce()
    await chess_auth_db.save_oauth_state(state, code_verifier, user_id)

    redirect_uri = f"{_backend_base_url(request)}/auth/callback/lichess"
    url = chess_oauth.build_lichess_authorize_url(
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )
    return {"url": url, "state": state}


@auth_router.get("/callback/lichess")
async def lichess_callback(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
):
    """Lichess OAuth callback. Exchanges code for token, stores account, redirects to frontend."""
    frontend_base = _frontend_redirect_base(request)
    if error:
        return RedirectResponse(url=f"{frontend_base}/?chess_error=access_denied", status_code=302)
    if not code or not state:
        return RedirectResponse(url=f"{frontend_base}/?chess_error=missing_code", status_code=302)

    data = await chess_auth_db.consume_oauth_state(state)
    if not data:
        return RedirectResponse(url=f"{frontend_base}/?chess_error=invalid_state", status_code=302)

    redirect_uri = f"{_backend_base_url(request)}/auth/callback/lichess"
    try:
        token_resp = await chess_oauth.exchange_lichess_code(
            code=code,
            code_verifier=data["code_verifier"],
            redirect_uri=redirect_uri,
        )
    except Exception as e:
        logger.exception("Lichess token exchange failed")
        return RedirectResponse(url=f"{frontend_base}/?chess_error=token_exchange", status_code=302)

    access_token = token_resp.get("access_token")
    if not access_token:
        return RedirectResponse(url=f"{frontend_base}/?chess_error=no_token", status_code=302)

    try:
        account = await chess_oauth.get_lichess_account(access_token)
    except Exception as e:
        logger.exception("Lichess account fetch failed")
        return RedirectResponse(url=f"{frontend_base}/?chess_error=account_fetch", status_code=302)

    username = account.get("username") or account.get("id") or "?"
    platform_user_id = account.get("id") or username

    await chess_auth_db.upsert_chess_account(
        eloquent_user_id=data["eloquent_user_id"],
        platform="lichess",
        platform_user_id=platform_user_id,
        username=username,
        access_token=access_token,
        token_expires_at=None,  # Lichess long-lived
        scopes=token_resp.get("scope"),
    )
    return RedirectResponse(url=f"{frontend_base}/?chess_linked=lichess", status_code=302)


@auth_router.get("/me")
async def auth_me(
    x_eloquent_user_id: Optional[str] = Header(None, alias="X-Eloquent-User-Id"),
) -> Dict[str, Any]:
    """Return linked accounts and optional user id for this client."""
    user_id = _get_eloquent_user_id(x_eloquent_user_id)
    await chess_auth_db.ensure_user(user_id)
    accounts = await chess_auth_db.get_linked_accounts(user_id)
    return {"eloquent_user_id": user_id, "accounts": accounts}


@auth_router.post("/unlink/{platform}")
async def unlink_platform(
    platform: str,
    x_eloquent_user_id: Optional[str] = Header(None, alias="X-Eloquent-User-Id"),
):
    """Unlink Lichess or Chess.com account."""
    if platform not in ("lichess", "chesscom"):
        raise HTTPException(status_code=400, detail="platform must be lichess or chesscom")
    user_id = _get_eloquent_user_id(x_eloquent_user_id)
    deleted = await chess_auth_db.unlink_account(user_id, platform)
    return {"unlinked": deleted}


@auth_router.post("/import-games")
async def import_games(
    x_eloquent_user_id: Optional[str] = Header(None, alias="X-Eloquent-User-Id"),
    platform: Optional[str] = Query(None),
    max_games: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
    """Import last N games from linked account(s). If platform omitted, import from all linked."""
    user_id = _get_eloquent_user_id(x_eloquent_user_id)
    accounts = await chess_auth_db.get_linked_accounts(user_id)
    if not accounts:
        raise HTTPException(status_code=400, detail="No linked accounts. Link Lichess or Chess.com first.")
    if platform:
        accounts = [a for a in accounts if a["platform"] == platform]
        if not accounts:
            raise HTTPException(status_code=400, detail=f"No linked account for platform: {platform}")

    imported_total = 0
    errors: List[str] = []
    for acc in accounts:
        if acc["platform"] != "lichess":
            errors.append(f"Import not implemented for {acc['platform']} yet")
            continue
        token = await chess_auth_db.get_account_token(acc["id"])
        if not token:
            errors.append(f"Could not get token for {acc['username']}")
            continue
        try:
            n = await _import_lichess_games(acc["id"], acc["username"], token, max_games)
            imported_total += n
        except Exception as e:
            logger.exception("Lichess import failed for %s", acc["username"])
            errors.append(str(e))
    return {"imported": imported_total, "errors": errors if errors else None}


async def _import_lichess_games(chess_account_id: int, username: str, access_token: str, max_games: int) -> int:
    """Fetch last max_games from Lichess API and store PGN in imported_game."""
    import asyncio
    import json as _json

    import httpx

    base = "https://lichess.org"
    headers = {"Authorization": f"Bearer {access_token}"}
    games_list: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.get(
                f"{base}/api/games/user/{username}",
                headers={**headers, "Accept": "application/x-ndjson"},
                params={"max": max_games},
            )
            r.raise_for_status()
            text = (r.text or "").strip()
            game_ids: List[tuple] = []  # (id, created_at optional)
            for line in text.split("\n"):
                if not line.strip():
                    continue
                try:
                    game = _json.loads(line)
                except Exception:
                    continue
                gid = game.get("id") or game.get("gameId")
                if gid:
                    game_ids.append((gid, game.get("createdAt")))
            if not game_ids:
                try:
                    arr = _json.loads(text)
                    if isinstance(arr, list):
                        for game in arr[:max_games]:
                            gid = game.get("id") or game.get("gameId")
                            if gid:
                                game_ids.append((gid, game.get("createdAt")))
                except Exception:
                    pass
            logger.info("Lichess import: got %d game ids for %s", len(game_ids), username)
            for i, (game_id, created_at) in enumerate(game_ids[:max_games]):
                await asyncio.sleep(0.3)
                pgn_r = await client.get(
                    f"{base}/game/export/{game_id}",
                    headers={**headers, "Accept": "application/x-chess-pgn, text/plain"},
                    params={"literate": "true"},
                )
                if not pgn_r.is_success:
                    logger.warning("Lichess PGN export failed for %s: %s", game_id, pgn_r.status_code)
                    continue
                pgn = (pgn_r.text or "").strip()
                if not pgn:
                    continue
                games_list.append({
                    "platform_game_id": game_id,
                    "pgn_text": pgn,
                    "played_at": str(created_at) if created_at else None,
                })
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise RuntimeError("Lichess token expired or revoked. Please link your account again.") from e
            if e.response.status_code == 429:
                raise RuntimeError("Lichess rate limit. Please try again in a minute.") from e
            raise RuntimeError(f"Lichess API error: {e}") from e
        except Exception as e:
            logger.exception("Lichess games fetch failed")
            raise RuntimeError(f"Lichess API error: {e}") from e

    if not games_list:
        return 0
    return await chess_auth_db.add_imported_games(chess_account_id, games_list)


@auth_router.get("/games")
async def list_imported_games(
    x_eloquent_user_id: Optional[str] = Header(None, alias="X-Eloquent-User-Id"),
    platform: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    """List imported PGN games for the current user."""
    user_id = _get_eloquent_user_id(x_eloquent_user_id)
    games = await chess_auth_db.get_imported_games(user_id, platform=platform, limit=limit)
    if games:
        first_pgn = games[0].get("pgn_text") or ""
        logger.debug(
            "Chess PGN debug: first game id=%s pgn_len=%d first_400=%r",
            games[0].get("id"), len(first_pgn), first_pgn[:400],
        )
    return {"games": games}


# Chess.com: stub for when developer app is approved
@auth_router.get("/chesscom/authorize")
async def chesscom_authorize(request: Request):
    """Chess.com OAuth not yet configured. Apply for developer access and set CHESSCOM_CLIENT_ID etc."""
    raise HTTPException(
        status_code=501,
        detail="Chess.com OAuth requires developer application approval. See docs/OAUTH_ARCHITECTURE.md",
    )
