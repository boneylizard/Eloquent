# OAuth & Chess Integration Architecture (Eloquent)

## 1. Research Summary

### Lichess
- **Flow**: OAuth 2.0 **Authorization Code + PKCE** (no client secret).
- **Endpoints**: `GET https://lichess.org/oauth` (authorize), `POST https://lichess.org/api/token` (exchange code for token).
- **PKCE**: Required; only `S256` code challenge method. Generate `code_verifier` (random), `code_challenge = BASE64URL(SHA256(code_verifier))`.
- **Tokens**: Long-lived (~1 year). **No refresh tokens**; re-auth when expired or revoked.
- **Scopes**: Request minimal scope for our use case: read account (username) and export games. Lichess scope for games/account is typically **`email:read`** for account; games export is available with the same token (no separate scope documented; token grants access to account and games for that user).
- **Rate limit**: One request at a time; on 429 wait 1 minute.
- **References**: [Lichess API](https://lichess.org/api#tag/OAuth), [PKCE RFC 7636](https://datatracker.ietf.org/doc/html/rfc7636), [berserk](https://github.com/lichess-org/berserk) (Python client).

### Chess.com
- **Flow**: OAuth 2.0 **Authorization Code** (client_id + client_secret after app approval).
- **Status**: Developer application required ([Chess.com Developer Program](https://docs.google.com/forms/d/e/1FAIpQLScQsRMPRYej06DeswRPn770qJGWxZ4suD5WYlsFitHOC8zWoA/viewform)). Redirect URI and scopes must be agreed with Chess.com.
- **Public API**: Unauthenticated `https://api.chess.com/pub/player/{username}/games` exists for game archives; "Login with Chess.com" and linked identity require OAuth once approved.
- **Implementation**: Stub endpoints and documentation only until approval; game import can optionally use PubAPI by username (no OAuth) as a fallback.

---

## 2. Architectural Decisions

### 2.1 User model
- **Eloquent user**: One local identity per app instance, identified by a persistent **device/session id** (UUID in localStorage). No mandatory sign-up; linking chess accounts is optional.
- **Linking**: One Eloquent user can link **multiple chess accounts** (e.g. one Lichess + one Chess.com). Stored in `chess_account` table with `platform` and `platform_user_id`.

### 2.2 Database schema (SQLite)
- **eloquent_user**: `id` (UUID), `created_at`. One row per device/session.
- **chess_account**: `id`, `eloquent_user_id`, `platform` ('lichess' | 'chesscom'), `platform_user_id`, `username`, `access_token` (encrypted at rest), `token_expires_at` (nullable; Lichess long-lived), `scopes`, `created_at`, `updated_at`. Unique on `(eloquent_user_id, platform)` so at most one account per platform per user.
- **imported_game**: `id`, `chess_account_id`, `platform_game_id`, `pgn_text`, `played_at` (optional), `created_at`. Unique on `(chess_account_id, platform_game_id)` for deduplication.

### 2.3 PGN storage
- **Store as text**: One row per game; `pgn_text` holds full PGN. Keeps import simple and allows reuse by existing analysis/import tools. Optional parsed fields (e.g. FEN list, result) can be added later if needed.

### 2.4 OAuth token storage
- **Where**: Backend only; tokens never sent to frontend except in memory for API calls if we proxy. Frontend only gets "linked" state and username.
- **How**: Tokens stored in SQLite; **encrypted at rest** using a key from env `CHESS_OAUTH_SECRET` (or a default for dev). Simple symmetric encryption (e.g. Fernet) to avoid plaintext in DB.
- **Refresh**: Lichess has no refresh; we re-prompt login when token is invalid. Chess.com (when implemented) may provide refresh tokens—store and use when available.

### 2.5 Game import
- **When**: (1) **Manual**: "Import my last 10 games" (or N) from UI. (2) **Optional auto**: After successful link, one-time import of last 10 games as proof of concept.
- **Deduplication**: By `(chess_account_id, platform_game_id)`; same game re-imported is no-op.

### 2.6 Error handling
- **OAuth denied**: Redirect to frontend with `?error=access_denied`; show message and option to retry.
- **Token expired / revoked**: On next API call return 401; frontend shows "Reconnect Lichess/Chess.com" and clears stored link for that platform.
- **Rate limit (429)**: Backend retries after 1 minute or returns a clear message; frontend shows "Try again later".

### 2.7 Privacy (least privilege)
- **Lichess**: Request only scopes needed: account identity (username) and read games (no email if not required; we use `email:read` only if we need it for display—alternatively minimal scope; Lichess docs suggest scopes like `email:read` for account; for games the token from OAuth allows export). After verification we use the minimal set that works (e.g. no email if we only need username and games).
- **Chess.com**: Request only profile and game read when available.

---

## 3. Flow Summary

1. **Frontend**: User clicks "Login with Lichess". Frontend calls `GET /auth/lichess/authorize` with `X-Eloquent-User-Id: <device_uuid>` (or backend creates/returns user id in cookie).
2. **Backend**: Generates PKCE `code_verifier` + `code_challenge`, stores `state -> (code_verifier, eloquent_user_id)` in short-lived cache/DB, returns `{ url: "https://lichess.org/oauth?..." }`.
3. **Frontend**: Redirects user to Lichess. User authorizes; Lichess redirects to backend `GET /auth/callback/lichess?code=...&state=...`.
4. **Backend**: Validates `state`, exchanges `code` + `code_verifier` for access token, fetches account (username), encrypts token, upserts `chess_account`, redirects to frontend e.g. `/?chess_linked=lichess`.
5. **Import**: User clicks "Import last 10 games" or auto-import runs; backend uses stored token (berserk) to fetch games, saves PGNs to `imported_game`.
6. **Chess.com**: Same flow once approved; until then, stub and docs only.

---

## 4. Testing (requirements)

- Successful Lichess OAuth: authorize → callback → token stored → account visible.
- Token refresh: N/A for Lichess; for Chess.com when implemented, refresh and retry.
- Game import: Last 10 games imported and listed; no duplicate on re-import.
- Error states: Denied auth (redirect with error); expired token (401 + re-link); network failure (clear message, retry).

---

## 5. Chess.com developer application (for implementation)

When applying for Chess.com OAuth:
- **Application type**: Web app (Eloquent local AI chess).
- **Redirect URI**: `http://localhost:8000/auth/callback/chesscom` (and production URL if any).
- **Scopes**: Request read profile and read game history (exact scope names per their docs).
- **Use**: Store client_id and client_secret in env; use authorization code flow with secret only on backend.
