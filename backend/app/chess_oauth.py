"""
Lichess OAuth 2.0 PKCE and Chess.com stub.
"""
import base64
import hashlib
import secrets
import urllib.parse
from typing import Any, Dict

import httpx

LICHESS_HOST = "https://lichess.org"
LICHESS_OAUTH_AUTHORIZE = f"{LICHESS_HOST}/oauth"
LICHESS_OAUTH_TOKEN = f"{LICHESS_HOST}/api/token"
LICHESS_API_ACCOUNT = f"{LICHESS_HOST}/api/account"

# Minimal scopes: read account (username) and read games. Lichess uses scope strings like "email:read" for account.
# For exporting our own games we need token with account access; games export is allowed with same token.
LICHESS_SCOPES = "email:read"  # grants account + games for the authenticated user

# Client ID for public PKCE client (no secret). Use a unique identifier for the app.
LICHESS_CLIENT_ID = "eloquent-chess-local"


def pkce_code_verifier() -> str:
    """Generate a PKCE code_verifier (43–128 chars, unreserved)."""
    return secrets.token_urlsafe(32)


def pkce_code_challenge_s256(verifier: str) -> str:
    """S256 code challenge: BASE64URL(SHA256(verifier))."""
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_lichess_authorize_url(
    redirect_uri: str,
    state: str,
    code_challenge: str,
    scopes: str = LICHESS_SCOPES,
    client_id: str = LICHESS_CLIENT_ID,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return LICHESS_OAUTH_AUTHORIZE + "?" + urllib.parse.urlencode(params)


async def exchange_lichess_code(
    code: str,
    code_verifier: str,
    redirect_uri: str,
    client_id: str = LICHESS_CLIENT_ID,
) -> Dict[str, Any]:
    """Exchange authorization code for access token. Raises on error."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            LICHESS_OAUTH_TOKEN,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    r.raise_for_status()
    return r.json()


async def get_lichess_account(access_token: str) -> Dict[str, Any]:
    """Fetch current account info (id, username, etc.)."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            LICHESS_API_ACCOUNT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    r.raise_for_status()
    return r.json()


def generate_lichess_pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge)."""
    verifier = pkce_code_verifier()
    challenge = pkce_code_challenge_s256(verifier)
    return verifier, challenge
