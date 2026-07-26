#!/usr/bin/env bash
# Build environment for Mirid on Apple Silicon: the macOS counterpart of install.bat.
#
# Creates <repo>/venv from Python 3.12, installs the pinned dependency set, and
# compiles the two inference bindings against Metal. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/venv"
PYTHON_BIN="${MIRID_PYTHON:-python3.12}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Mirid's macOS build environment is for Apple Silicon." >&2
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Install the Xcode command line tools first: xcode-select --install" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to compile the Metal inference bindings: brew install cmake" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 is required and '$PYTHON_BIN' was not found." >&2
  echo "Install it (brew install python@3.12, or uv python install 3.12) or set MIRID_PYTHON." >&2
  exit 1
fi

version="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$version" != "3.12" ]]; then
  echo "Python 3.12 is required; '$PYTHON_BIN' is $version." >&2
  echo "The MLX runner wheels and the frozen sidecar are both built for 3.12." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> Creating the build environment at $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"

echo "==> Installing pinned dependencies"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -r "$ROOT/requirements-macos.txt"

# llama.cpp bindings. The published Metal wheel for 0.3.34 is corrupt (bad CRC on
# lib/libggml-base.0.16.0.dylib under both pip and uv), so compile from source.
if "$PY" -c 'import llama_cpp' >/dev/null 2>&1; then
  echo "==> llama-cpp-python already present"
else
  echo "==> Compiling llama-cpp-python with Metal (several minutes)"
  CMAKE_ARGS="-DGGML_METAL=on" "$PY" -m pip install --no-binary llama-cpp-python llama-cpp-python==0.3.34
fi

# stable-diffusion.cpp bindings, likewise built against Metal.
if "$PY" -c 'import stable_diffusion_cpp' >/dev/null 2>&1; then
  echo "==> stable-diffusion-cpp-python already present"
else
  echo "==> Compiling stable-diffusion-cpp-python with Metal (several minutes)"
  CMAKE_ARGS="-DSD_METAL=ON -DGGML_NATIVE=OFF -DSD_BUILD_EXAMPLES=OFF" \
    "$PY" -m pip install --no-binary stable-diffusion-cpp-python stable-diffusion-cpp-python==0.4.7
fi

echo "==> Verifying the service imports"
cd "$ROOT"
"$PY" - <<'PY'
import llama_cpp
import stable_diffusion_cpp
import backend.app.compute_capabilities  # noqa: F401
import backend.app.main  # noqa: F401
import backend.app.tts_backend  # noqa: F401
print("Mirid backend imports cleanly:", llama_cpp.__version__, stable_diffusion_cpp.__version__)
PY

echo
echo "Build environment ready at $VENV"
echo "Next: scripts/build_macos_app.sh   (or scripts/run_mirid_from_venv.sh to develop)"
