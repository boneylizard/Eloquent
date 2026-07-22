# Call Mode Demo Showcase

Fabricated test data for demonstrating call mode, character intelligence, profile memories, and agentic memory — without using real chats.

## One-click install

1. Start the backend and frontend as usual.
2. Open **Settings → General → Call Mode demo showcase**.
3. Click **Install demo showcase** (merges data; does not wipe your existing profiles or chats).
4. Open the chat **Late night — Chapter 7** with **Mira Vale** and enter call mode.

## Stable IDs

| Resource | ID |
|----------|-----|
| User profile | `profile_demo` |
| Character | `char_mira_vale` |
| Conversation | `conv_demo_mira` |

## What's included

- **Alex Chen** — rich user profile (UX designer / novelist)
- **20** profile memories (`profile_demo_memory_store.json` on disk)
- **12** agentic insights for Mira (`agentic/profile_demo_char_mira_vale.json`)
- **Mira Vale** — full character card with lore and speech style
- **26** chat messages (lighthouse scene rewrite)
- Story tracker entries for the About panel

## Editing the fiction

Edit `pack.json` in this folder. Re-run **Install demo showcase** to overwrite the demo files on disk and refresh browser data.

Backend API (optional):

- `GET /demo/showcase/status`
- `GET /demo/showcase/pack`
- `POST /demo/showcase/install` — body `{ "set_active": true }`

## Scenario

Late-night call: Alex is stuck on Chapter 7 of *Glass Harbor*. Mira pushes line-level craft feedback — fog openings, moral ambiguity, sister dynamics — so call mode About cards and memory injection have something real to work with.
