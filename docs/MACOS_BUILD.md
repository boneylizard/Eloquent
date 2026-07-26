# Building Mirid on macOS

Apple Silicon only. An Intel Mac cannot build or run this runtime.

## Requirements

- Apple Silicon Mac, macOS 14 or later
- Xcode command line tools (`xcode-select --install`)
- `cmake` (`brew install cmake`) — the inference bindings are compiled against Metal
- Python 3.12 exactly (`brew install python@3.12`, or `uv python install 3.12`)
- Node 24 (`.nvmrc` pins the version the frontend expects)
- Rust stable, for the Tauri shell

Python 3.12 is not a preference: the MLX runner wheels and the frozen sidecar are
both built for it.

## Build

```bash
./scripts/install_macos.sh      # once: creates ./venv and compiles the Metal bindings
./scripts/build_macos_app.sh    # produces Mirid.app and a .dmg
```

The artefacts land in `src-tauri/target/release/bundle/`.

`install_macos.sh` installs `requirements-macos.txt`, then compiles
`llama-cpp-python` and `stable-diffusion-cpp-python` from source against Metal.
Compiling these is deliberate. The published Metal wheel for `llama-cpp-python`
0.3.34 — the one named in `runtime/runtime-packages.lock.json` — is corrupt: it
fails a CRC check on `lib/libggml-base.0.16.0.dylib` under both `pip` and `uv`.

`build_macos_app.sh` stages the model runners, freezes the backend, installs that
runtime under `~/Library/Application Support/ai.mirid.desktop/runtime/`, and then
bundles the app.

That runtime installation step is what makes a locally built app actually run. The
desktop shell locates its runtime from the compile-time asset lock in
`src-tauri/src/runtime_macos.rs`, which is checked in as `unpublished` with zero
hashes; without a local install the app would try to download a runtime that has
not been published yet. The script reads the same constants the Rust code does, so
it keeps working once a real runtime is published.

## Development

```bash
./scripts/run_mirid_from_venv.sh            # tauri dev against ./venv
./scripts/run_mirid_from_venv.sh --force-cpu
```

This is the counterpart of `scripts/run_mirid_from_venv.ps1`. It requires a debug
build: `MIRID_DEV_USE_VENV` is ignored in release builds.

Useful overrides while developing: `MIRID_FORCE_CPU=1`,
`MIRID_LOCAL_BACKEND=cpu|metal|mlx|apple`, `MIRID_NATIVE_RUNNERS=0`.

## What a local build does not give you

The application is **ad-hoc signed and not notarised**. It runs on the machine
that built it. Distributing it to other Macs additionally requires:

1. A signed, hashed runtime published to the Hugging Face runtime repository,
   staged with `./scripts/package_runtime_release.sh vN`, which rewrites the
   version, sizes and SHA-256 values in `src-tauri/src/runtime_macos.rs`.
2. An Apple Developer ID signature and notarisation for the `.app` and `.dmg`.

Until both are done, keep the checked-in Apple asset lock reading `unpublished`.

Updater artefacts are disabled for macOS in `src-tauri/tauri.macos.conf.json`,
because a macOS update feed cannot be honoured before a signed runtime exists.

## Verifying a build

```bash
# the frozen service answers on its own
./build/sidecar-dist/mirid-sidecar/mirid-sidecar-aarch64-apple-darwin backend --port 8010
curl -s http://127.0.0.1:8010/health

# the staged runners each report themselves
./build/model-runners/macos-aarch64/metal/llama-server --list-devices
./build/model-runners/macos-aarch64/mlx/mirid-mlx-runner --probe
./build/model-runners/macos-aarch64/apple/mirid-apple-runner --probe
```

`scripts/assert_runtime_stage_safe.py` runs as part of the sidecar build and will
fail it if the staged interpreter library has been displaced by a dependency that
vendors its own — a real failure mode that produces a service which cannot start.

Confirm GGUF work is reaching the GPU rather than silently falling back: the
Metal runner logs `using device MTL0` and assigns layers to it.
