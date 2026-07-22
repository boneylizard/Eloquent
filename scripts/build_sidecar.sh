#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${MIRID_BUILD_PYTHON:-$ROOT/venv/bin/python}"
DIST="$ROOT/build/sidecar-dist"
BUILT="$DIST/mirid-sidecar"
RUNNERS="$ROOT/build/model-runners"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "The Apple sidecar must be built on Apple Silicon." >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Create Mirid's Python 3.12 build environment first, or set MIRID_BUILD_PYTHON." >&2
  exit 1
fi
if [[ ! -f "$RUNNERS/manifest.json" ]]; then
  "$ROOT/scripts/stage_model_runners.sh"
fi

MIRID_SIDECAR_PROFILE="${MIRID_SIDECAR_PROFILE:-default}" "$PYTHON" -m PyInstaller \
  --noconfirm \
  --distpath "$DIST" \
  --workpath "$ROOT/build/pyinstaller" \
  "$ROOT/mirid-sidecar-platform.spec"

"$PYTHON" "$ROOT/scripts/assert_runtime_stage_safe.py" "$BUILT/_internal"

mkdir -p "$BUILT/_internal/runners"
cp -R "$RUNNERS/"* "$BUILT/_internal/runners/"
echo "Apple Silicon sidecar built at $BUILT"
