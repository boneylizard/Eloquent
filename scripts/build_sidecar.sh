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

# PyInstaller resolves the bundle's interpreter library by basename, and
# torchcodec ships its own libpython3.12.dylib. When the vendored copy wins that
# slot the stage root ends up holding a symlink into torchcodec, and the frozen
# service dies at startup on "No module named '_struct'" because that copy
# carries none of the interpreter's builtin modules. Put the real one back.
INTERPRETER_LIBRARY="$("$PYTHON" -c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR") or "", sysconfig.get_config_var("LDLIBRARY") or ""))')"
STAGED_LIBRARY="$BUILT/_internal/$(basename "$INTERPRETER_LIBRARY")"
if [[ -L "$STAGED_LIBRARY" && -f "$INTERPRETER_LIBRARY" ]]; then
  echo "Replacing vendored interpreter library at $STAGED_LIBRARY"
  rm "$STAGED_LIBRARY"
  cp "$INTERPRETER_LIBRARY" "$STAGED_LIBRARY"
  chmod 755 "$STAGED_LIBRARY"
fi

"$PYTHON" "$ROOT/scripts/assert_runtime_stage_safe.py" "$BUILT/_internal"

mkdir -p "$BUILT/_internal/runners"
cp -R "$RUNNERS/"* "$BUILT/_internal/runners/"
echo "Apple Silicon sidecar built at $BUILT"
