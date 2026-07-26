#!/usr/bin/env bash
# macOS/Linux counterpart of run_mirid_from_venv.ps1: launch Mirid in
# development mode with the backend served from the repo-root venv.
set -euo pipefail

FORCE_CPU=0
for arg in "$@"; do
  case "$arg" in
    --force-cpu) FORCE_CPU=1 ;;
    *) echo "Unknown option: $arg (supported: --force-cpu)" >&2; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
ENTRY_POINT="$ROOT/sidecar_entry.py"
TAURI="$ROOT/frontend/node_modules/.bin/tauri"

if [[ ! -x "$PYTHON" ]]; then
  echo "Mirid's development venv is missing: $PYTHON" >&2
  echo "Create it with: python3.12 -m venv venv && venv/bin/pip install -r requirements-macos.txt" >&2
  exit 1
fi
if [[ ! -f "$ENTRY_POINT" ]]; then
  echo "Mirid's service entry point is missing: $ENTRY_POINT" >&2
  exit 1
fi
if [[ ! -x "$TAURI" ]]; then
  echo "The Tauri development CLI is missing. Run npm install in frontend first." >&2
  exit 1
fi

for port in 8000 8002; do
  if nc -z 127.0.0.1 "$port" >/dev/null 2>&1; then
    echo "Port $port is already in use. Close installed or development copies of Mirid, then try again." >&2
    exit 1
  fi
done

if ! "$PYTHON" -c "import fastapi, uvicorn; import backend.app.compute_capabilities; import sys; print('Mirid venv ready:', sys.executable)"; then
  echo "Mirid's development venv could not import the desktop service dependencies." >&2
  exit 1
fi

export MIRID_DEV_USE_VENV=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
if [[ "$FORCE_CPU" == "1" ]]; then
  export MIRID_FORCE_CPU=1
else
  unset MIRID_FORCE_CPU
fi

cd "$ROOT"
exec "$TAURI" dev
