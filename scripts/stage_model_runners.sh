#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-b10068}"
MLX_VERSION="${MLX_VERSION:-0.32.0}"
MLX_METAL_VERSION="${MLX_METAL_VERSION:-0.32.0}"
MLX_LM_VERSION="${MLX_LM_VERSION:-0.31.3}"
MLX_SHA256="${MLX_SHA256:-ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467}"
MLX_METAL_SHA256="${MLX_METAL_SHA256:-5b64b20ac24b0c401f489de01e8209edc4d372125201f19314e6f39e385322aa}"
MLX_LM_SHA256="${MLX_LM_SHA256:-758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${MIRID_RUNNER_OUTPUT:-$ROOT/build/model-runners}"
CACHE="${MIRID_RUNNER_CACHE:-$ROOT/build/downloads/model-runners}"
PLATFORM_DIR="$OUTPUT/macos-aarch64"
ARCHIVE="llama-${LLAMA_CPP_VERSION}-bin-macos-arm64.tar.gz"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${ARCHIVE}"
RUNTIME_PACKS_URL="${MIRID_RUNTIME_PACKS_URL:-https://huggingface.co/datasets/boneylizardwizard/mirid-runtime-packs/resolve/main}"

download_verified() {
  local url="$1"
  local destination="$2"
  local expected="$3"
  local actual
  if [[ -f "$destination" ]]; then
    actual="$(shasum -a 256 "$destination" | awk '{print $1}')"
    if [[ "$actual" == "$expected" ]]; then
      return
    fi
  fi
  local partial="${destination}.part"
  rm -f "$partial"
  curl --fail --location --retry 5 --output "$partial" "$url"
  mv "$partial" "$destination"
  actual="$(shasum -a 256 "$destination" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $destination" >&2
    exit 1
  fi
}

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script builds the Apple Silicon runners on an Apple Silicon Mac." >&2
  exit 1
fi
if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "Python 3.12 is required for the pinned MLX runner wheels." >&2
  exit 1
fi

mkdir -p "$CACHE" "$PLATFORM_DIR/metal" "$PLATFORM_DIR/cpu" "$PLATFORM_DIR/mlx" "$PLATFORM_DIR/apple"
curl --fail --location --retry 5 --continue-at - --output "$CACHE/$ARCHIVE" "$URL"
rm -rf "$CACHE/llama-$LLAMA_CPP_VERSION"
mkdir -p "$CACHE/llama-$LLAMA_CPP_VERSION"
tar -xzf "$CACHE/$ARCHIVE" -C "$CACHE/llama-$LLAMA_CPP_VERSION"
SERVER="$(find "$CACHE/llama-$LLAMA_CPP_VERSION" -type f -name llama-server -print -quit)"
if [[ -z "$SERVER" ]]; then
  echo "The llama.cpp archive did not contain llama-server." >&2
  exit 1
fi
cp -R "$(dirname "$SERVER")/"* "$PLATFORM_DIR/metal/"
cp -R "$(dirname "$SERVER")/"* "$PLATFORM_DIR/cpu/"
"$PLATFORM_DIR/metal/llama-server" --version

MLX_VENV="$ROOT/build/mlx-runner-venv"
MLX_WHEEL="mlx-${MLX_VERSION}-cp312-cp312-macosx_14_0_arm64.whl"
MLX_METAL_WHEEL="mlx_metal-${MLX_METAL_VERSION}-py3-none-macosx_14_0_arm64.whl"
MLX_LM_WHEEL="mlx_lm-${MLX_LM_VERSION}-py3-none-any.whl"
download_verified \
  "$RUNTIME_PACKS_URL/bindings/mlx/macos-aarch64/metal/$MLX_WHEEL" \
  "$CACHE/$MLX_WHEEL" \
  "$MLX_SHA256"
download_verified \
  "$RUNTIME_PACKS_URL/bindings/mlx-metal/macos-aarch64/metal/$MLX_METAL_WHEEL" \
  "$CACHE/$MLX_METAL_WHEEL" \
  "$MLX_METAL_SHA256"
download_verified \
  "$RUNTIME_PACKS_URL/bindings/mlx-lm/macos-aarch64/metal/$MLX_LM_WHEEL" \
  "$CACHE/$MLX_LM_WHEEL" \
  "$MLX_LM_SHA256"
python3 -m venv "$MLX_VENV"
"$MLX_VENV/bin/python" -m pip install --upgrade pip
"$MLX_VENV/bin/python" -m pip install \
  "$CACHE/$MLX_WHEEL" \
  "$CACHE/$MLX_METAL_WHEEL" \
  "$CACHE/$MLX_LM_WHEEL" \
  pyinstaller
"$MLX_VENV/bin/pyinstaller" --noconfirm --clean --onefile \
  --name mirid-mlx-runner \
  --distpath "$PLATFORM_DIR/mlx" \
  --workpath "$ROOT/build/pyinstaller-mlx" \
  --specpath "$ROOT/build" \
  --collect-all mlx_lm \
  --collect-all mlx \
  "$ROOT/sidecars/mlx_runner.py"
"$PLATFORM_DIR/mlx/mirid-mlx-runner" --probe

swift build --package-path "$ROOT/sidecars/apple_foundation" -c release
cp "$ROOT/sidecars/apple_foundation/.build/release/mirid-apple-runner" "$PLATFORM_DIR/apple/mirid-apple-runner"
"$PLATFORM_DIR/apple/mirid-apple-runner" --probe || true

cp "$ROOT/runtime/model-runners.json" "$OUTPUT/manifest.json"
cat > "$OUTPUT/assets.lock.json" <<EOF
{
  "schemaVersion": 1,
  "llamaCppVersion": "$LLAMA_CPP_VERSION",
  "mlxVersion": "$MLX_VERSION",
  "mlxMetalVersion": "$MLX_METAL_VERSION",
  "mlxLmVersion": "$MLX_LM_VERSION",
  "platform": "macos-aarch64"
}
EOF
echo "Model runners staged at $OUTPUT"
