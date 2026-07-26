#!/usr/bin/env bash
# Produce a runnable Mirid.app (and .dmg) on Apple Silicon, end to end.
#
#   scripts/install_macos.sh     once, to create the build environment
#   scripts/build_macos_app.sh   to build the artefact
#
# Stages the model runners, freezes the backend, installs that runtime where the
# desktop app looks for it, then bundles the app. The install step is what lets a
# locally built app run before a signed runtime has been published: without it
# the app tries to download the runtime described by src-tauri/src/runtime_macos.rs,
# which is deliberately checked in as "unpublished".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${MIRID_BUILD_PYTHON:-$ROOT/venv/bin/python}"
ASSET_LOCK="$ROOT/src-tauri/src/runtime_macos.rs"
BUILT="$ROOT/build/sidecar-dist/mirid-sidecar"
RUNTIME_HOME="$HOME/Library/Application Support/ai.mirid.desktop/runtime"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Mirid's macOS application must be built on Apple Silicon." >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "Build environment missing. Run scripts/install_macos.sh first." >&2
  exit 1
fi

read_const() {
  # read_const <CONST_NAME> -> the string literal assigned to it
  sed -n "s/^const $1: &str =[[:space:]]*\"\([^\"]*\)\";/\1/p" "$ASSET_LOCK" | head -1
}

echo "==> Staging model runners (llama.cpp Metal + CPU, MLX, Apple Foundation)"
bash "$ROOT/scripts/stage_model_runners.sh"

echo "==> Freezing the backend service"
bash "$ROOT/scripts/build_sidecar.sh"

RUNTIME_VERSION="$(read_const RUNTIME_VERSION)"
ARCHIVE_SHA="$(sed -n 's/^[[:space:]]*"\([a-f0-9]\{64\}\)";/\1/p' "$ASSET_LOCK" | sed -n 1p)"
SIDECAR_SHA="$(sed -n 's/^[[:space:]]*"\([a-f0-9]\{64\}\)";/\1/p' "$ASSET_LOCK" | sed -n 2p)"
SIDECAR_EXE="$(read_const SIDECAR_EXE)"
if [[ -z "$RUNTIME_VERSION" || -z "$ARCHIVE_SHA" || -z "$SIDECAR_SHA" || -z "$SIDECAR_EXE" ]]; then
  echo "Could not read the Apple asset lock from $ASSET_LOCK." >&2
  exit 1
fi

# The desktop app addresses each installed runtime by version and asset hashes.
RELEASE_DIR="$RUNTIME_HOME/releases/${RUNTIME_VERSION}-${ARCHIVE_SHA:0:12}-${SIDECAR_SHA:0:12}"
echo "==> Installing the runtime for local use at $RELEASE_DIR"
mkdir -p "$RELEASE_DIR"
rsync -a --delete "$BUILT/_internal/" "$RELEASE_DIR/_internal/"
cp "$BUILT/$SIDECAR_EXE" "$RELEASE_DIR/$SIDECAR_EXE"
chmod 755 "$RELEASE_DIR/$SIDECAR_EXE"
printf '%s' "$RUNTIME_VERSION" > "$RUNTIME_HOME/runtime.ready"

echo "==> Bundling the application"
cd "$ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run tauri build

BUNDLE="$ROOT/src-tauri/target/release/bundle"
echo
echo "Built:"
echo "  $BUNDLE/macos/Mirid.app"
ls "$BUNDLE/dmg/"*.dmg 2>/dev/null | sed 's/^/  /' || true
echo
echo "The application is ad-hoc signed and uses the runtime installed above."
echo "Publishing to other Macs additionally needs a signed, hashed runtime"
echo "(scripts/package_runtime_release.sh) and Apple notarisation."
