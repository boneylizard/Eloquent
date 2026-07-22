# Mirid v1.0 UI Control Audit

## Automated now

Run from `frontend`:

```powershell
npm run test:ui-smoke
```

The Playwright smoke suite verifies:

- Mirid loads without an uncaught JavaScript error while the backend is offline.
- Code Editor, Forensic Linguistics, Market Simulator, and Watch have no primary navigation controls.
- The compact sidebar contains exactly one Settings button.
- Documents, Characters, Pool, Chess, Memory tools, and Transcript search open from the sidebar.
- Pool onboarding is recognised and dismissed before navigation continues.
- Settings opens in-app when a native second window cannot be created.
- The Settings pop-out control is exercised and safely falls back in a browser run.
- All 13 Settings tabs render and become selected.
- Mirid-owned React warnings fail the suite; the legacy-context warning inside `chessboardjsx` is quarantined.
- Mermaid waits for text generation to finish before parsing and rendering a diagram; a browser smoke test requires a real SVG result.
- A per-panel visible-button inventory is attached to the Playwright result.

The unit suite also verifies the shared desktop-window helper's browser open and close fallbacks. Settings, Call Mode, and Cognitive Glass now route secondary windows through Tauri-aware helpers rather than calling browser pop-up APIs directly.

## Deliberately not clicked by the smoke suite

These controls need isolated state, mocked APIs, or a disposable sidecar profile before automation can exercise them safely:

- **Destructive data:** erase conversations, delete profiles, delete caches, delete logs, repair or recover chat stores.
- **Filesystem:** browse for directories, import backups, export backups, upload documents, upload images, and choose attachments.
- **Desktop lifecycle:** check for updates, install updates, verify native secondary-window creation, restart TTS, and shut down TTS.
- **Model operations:** load models, unload models, change GPU allocation, generate images, run TTS/STT, and submit chat generations.
- **Persistent writes:** save characters, edit memories, apply persona realignment, index transcript folders, and change endpoint credentials.

## Next automation layer

The next suite should launch the packaged Mirid shell against a disposable runtime profile, seed temporary user data, and verify each risky control against its backend endpoint and expected UI result. It must never use the developer's real conversations, profiles, models, or settings directories.

## Release gate

Mirid pins Node `24.14.0` in the repository root. Before producing a public installer, activate that version, install exactly what the tracked lockfile specifies, and run:

```powershell
nvm use
cd frontend
npm ci
npm test
npm run test:ui-smoke
npm run release:build
npm audit --audit-level=high
cd ..\src-tauri
cargo check
```

Tauri runs `npm run release:build` automatically before packaging. The preflight rejects unsupported Node versions, version drift between frontend/Tauri/Cargo, missing icons, an invalid runtime download contract, or an out-of-sync lockfile.

Passing this gate proves navigation and rendering coverage. It does not yet prove backend-dependent or destructive controls.

## Release module profiles

The normal frontend build does not bundle Elections or any retired panel. `npm run build` emits a Rollup source-module inventory and fails if optional or retired source crosses that boundary.

For the private Elections build:

```powershell
$env:VITE_MIRID_INCLUDED_MODULES = 'elections'
$env:VITE_MIRID_ENABLED_MODULES = 'elections'
npm run build
```

`VITE_MIRID_INCLUDED_MODULES` controls what is physically compiled into the application. `VITE_MIRID_ENABLED_MODULES` controls which compiled modules are enabled on first launch. Enabling a module that was not included cannot expose it.
