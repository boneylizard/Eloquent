#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^v[0-9]+(\.[0-9]+){0,2}$ ]]; then
  echo "Usage: $0 vN[.N[.N]] [base-url]" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "The Apple runtime must be packaged on Apple Silicon." >&2
  exit 1
fi

VERSION="$1"
BASE_URL="${2:-https://huggingface.co/boneylizardwizard/mirid-runtime/resolve/main}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${MIRID_BUILD_PYTHON:-$ROOT/venv/bin/python}"
BUILT="$ROOT/build/sidecar-dist/mirid-sidecar"
INTERNAL="$BUILT/_internal"
SIDECAR="$BUILT/mirid-sidecar-aarch64-apple-darwin"
OUTPUT="$ROOT/build/runtime-release/$VERSION/macos-aarch64"
ARCHIVE="$OUTPUT/mirid-runtime-aarch64-apple-darwin.7z"
RELEASE_SIDECAR="$OUTPUT/mirid-sidecar-aarch64-apple-darwin"

"$PYTHON" "$ROOT/scripts/assert_runtime_stage_safe.py" "$INTERNAL"
RUST_ASSETS="$ROOT/src-tauri/src/runtime_macos.rs"

for path in "$INTERNAL" "$SIDECAR" "$INTERNAL/runners/manifest.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing Apple runtime artefact: $path" >&2
    exit 1
  fi
done
if ! command -v 7z >/dev/null 2>&1; then
  echo "Install 7-Zip before packaging the Apple runtime." >&2
  exit 1
fi

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"
(cd "$INTERNAL" && 7z a -t7z -mx=9 -m0=lzma2 -ms=on "$ARCHIVE" ./\*)
cp "$SIDECAR" "$RELEASE_SIDECAR"
chmod 755 "$RELEASE_SIDECAR"

archive_size="$(stat -f%z "$ARCHIVE")"
sidecar_size="$(stat -f%z "$RELEASE_SIDECAR")"
archive_hash="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
sidecar_hash="$(shasum -a 256 "$RELEASE_SIDECAR" | awk '{print $1}')"

python3 - "$RUST_ASSETS" "$VERSION" "$BASE_URL" "$archive_size" "$sidecar_size" "$archive_hash" "$sidecar_hash" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
version, base_url, archive_size, sidecar_size, archive_hash, sidecar_hash = sys.argv[2:]
source = path.read_text(encoding="utf-8")
replacements = {
    r'const RUNTIME_VERSION: &str = "[^"]+";': f'const RUNTIME_VERSION: &str = "{version}";',
    r'const HF_BASE: &str = "[^"]+";': f'const HF_BASE: &str = "{base_url.rstrip("/")}";',
    r'const RUNTIME_ARCHIVE_SIZE: u64 = [\d_]+;': f'const RUNTIME_ARCHIVE_SIZE: u64 = {archive_size};',
    r'const SIDECAR_EXE_SIZE: u64 = [\d_]+;': f'const SIDECAR_EXE_SIZE: u64 = {sidecar_size};',
    r'const RUNTIME_ARCHIVE_SHA256: &str =\s*"[a-f0-9]+";': f'const RUNTIME_ARCHIVE_SHA256: &str =\n    "{archive_hash}";',
    r'const SIDECAR_EXE_SHA256: &str =\s*"[a-f0-9]+";': f'const SIDECAR_EXE_SHA256: &str =\n    "{sidecar_hash}";',
}
for pattern, replacement in replacements.items():
    source, count = re.subn(pattern, replacement, source)
    if count != 1:
        raise SystemExit(f"Expected one match for {pattern}, found {count}")
path.write_text(source, encoding="utf-8")
PY

cat > "$OUTPUT/runtime-release.json" <<EOF
{
  "schemaVersion": 1,
  "modelRunnerContractVersion": 1,
  "channel": "stable",
  "runtimeVersion": "$VERSION",
  "platform": "macos-aarch64",
  "baseUrl": "${BASE_URL%/}",
  "assets": {
    "runtimeArchive": {"filename": "$(basename "$ARCHIVE")", "size": $archive_size, "sha256": "$archive_hash"},
    "sidecarExecutable": {"filename": "$(basename "$RELEASE_SIDECAR")", "size": $sidecar_size, "sha256": "$sidecar_hash"}
  }
}
EOF
echo "Apple runtime release staged at $OUTPUT"
