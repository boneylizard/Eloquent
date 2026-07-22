# Mirid local model runtime

Research and implementation baseline: 19 July 2026.

## Decision

Mirid keeps one stable application backend and starts a private model runner only when a local model is loaded. Every runner binds to `127.0.0.1`, exposes the small OpenAI-compatible surface Mirid already understands, and is selected by proof rather than hardware-name guessing.

The application probes each compatible runner in priority order. If a runner is absent, cannot see its device, or fails while loading the model, Mirid tries the next one. CPU is the final GGUF fallback. A computer without a GPU must never prevent Mirid from opening.

## July 2026 baseline

- **llama.cpp `b10068`** is pinned from the 18 July 2026 release. Its official release supplies CPU, CUDA, HIP Radeon, Vulkan, Metal, ROCm, OpenCL Adreno, SYCL and OpenVINO builds. `llama-server` supplies the health check, device probe and OpenAI-compatible generation contract Mirid uses. See the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- **Windows NVIDIA:** use the official CUDA 12.4 build first. CUDA 13.3 is available upstream but 12.4 remains the broader packaged default. Vulkan and CPU remain fallbacks.
- **Windows AMD:** try the official HIP Radeon build first, then Vulkan, then CPU. AMD's Windows HIP SDK supports only a defined hardware subset, so a successful self-probe is required. See [AMD HIP SDK for Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html).
- **DirectML:** do not make it the GGUF default. DirectML is in maintenance mode and Microsoft directs new Windows work towards Windows ML. llama.cpp's maintained HIP and Vulkan backends are the better fit for Mirid's GGUF contract. See the [DirectML repository notice](https://github.com/microsoft/DirectML).
- **Apple Silicon GGUF:** use the official arm64 llama.cpp build with Metal, then explicitly disable device offload for the CPU fallback.
- **MLX:** pin `mlx-lm 0.31.3`. Its local server supports health checks, model listing, chat, completion and streaming. Mirid uses it for MLX model directories and Hugging Face MLX repositories. See the [MLX LM server guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) and [MLX LM releases](https://github.com/ml-explore/mlx-lm/releases).
- **Apple Intelligence:** expose the on-device system model as `mirid/apple-intelligence`. Availability is checked at runtime because the person may have disabled Apple Intelligence, the Mac may be ineligible, or the model may still be downloading. It is an additional system model, not a replacement for arbitrary GGUF or MLX roleplay models. See [Foundation Models](https://developer.apple.com/documentation/FoundationModels), [SystemLanguageModel](https://developer.apple.com/documentation/FoundationModels/SystemLanguageModel), and [Foundation Models updates](https://developer.apple.com/documentation/Updates/FoundationModels).
- **Tauri:** packaged sidecars remain target-specific and the managed backend receives only the installed runner directory and manifest path. See [Tauri sidecars](https://v2.tauri.app/develop/sidecar/).

## Public package registry

Mirid's portable dependency supply is separate from any one installer build. `runtime/runtime-packages.sources.json` declares the supported platform matrix. `scripts/refresh_runtime_package_lock.py` resolves current upstream channels to immutable assets, including source URL, size and SHA-256. `scripts/stage_runtime_packages.py` downloads and re-verifies them, while `scripts/publish_runtime_packages.py` publishes the complete repository and its static browser to Hugging Face.

The July 2026 registry contains native text runners for Windows x64, Windows on ARM, Apple Silicon, Linux x64 and Linux ARM; native image runners for Windows x64, Apple Silicon and Linux x64; official `llama-cpp-python` wheels where upstream publishes them; and the complete MLX core, Metal-kernel and MLX-LM wheel set for Mirid's Python 3.12 Apple sidecar. Each platform receives CPU or Metal baseline support plus the accelerators upstream actually supplies.

For image inference, prefer the native `stable-diffusion.cpp` release. The latest `stable-diffusion-cpp-python 0.4.7` source distribution vendors an older engine snapshot, so Mirid publishes only its reproducibly built and import-tested Windows CPU wheel as a compatibility artefact. The current native CPU, Vulkan, ROCm and Metal archives carry the newer engine. Do not label an older binding wheel as state of the art merely because it imports from Python.

Hugging Face stores these large binary files in its repository storage; the Dataset remains the source of bytes and provenance, while the static Space renders the manifest for people. Mirid installers should request only the packages selected by platform and successful local probing rather than download the entire registry.

## Selection order

| Computer | GGUF order | Additional local models |
| --- | --- | --- |
| Windows x64, NVIDIA | CUDA 12.4 → Vulkan → CPU | — |
| Windows x64, AMD | HIP → Vulkan → CPU | — |
| Windows x64, other graphics | Vulkan → CPU | — |
| Windows on ARM | OpenCL Adreno → CPU | — |
| Apple Silicon | Metal → CPU | MLX; Apple Intelligence when available |
| Linux x64, AMD | ROCm → Vulkan → CPU | — |
| Linux x64, other | Vulkan → CPU | — |

The manifest contains a single ordered candidate list. A candidate is selected only after its executable exists and its probe succeeds. A model-load failure continues the same fallback chain.

## Contract and versioning

`runtime/model-runners.json` has two independent version layers:

1. `schemaVersion` describes the JSON shape.
2. `contractVersion` describes the launch and HTTP behaviour the Python backend expects.

A runtime release records its own version and the model-runner contract version. Upstream engine versions may be updated without changing the contract. Change `contractVersion` only when arguments, health semantics or response shapes become incompatible; ship the matching backend and runners in the same runtime release.

Each staged release also writes `assets.lock.json` with upstream filenames, sizes, hashes and source URLs. Release builds copy the complete staged directory to `_internal/runners` and set:

- `MIRID_RUNNER_ROOT`
- `MIRID_RUNNER_MANIFEST`

Useful developer overrides are `MIRID_FORCE_CPU=1`, `MIRID_LOCAL_BACKEND=cpu|vulkan|amd|nvidia|metal|apple`, and `MIRID_NATIVE_RUNNERS=0`.

## Building

### Windows x64

```powershell
.\scripts\stage_model_runners.ps1
```

This downloads and verifies the official CPU, Vulkan, HIP Radeon and CUDA 12.4 archives, stages their complete dependency directories, probes each executable, and records SHA-256 hashes. Use `-Backends cpu` for a quick CPU-only validation.

The ordinary frozen Windows build includes CUDA, Vulkan and CPU. Vulkan gives AMD and other Windows graphics hardware an accelerated first run without adding the much larger, hardware-specific HIP payload to every installation. HIP remains a separately staged runner and takes priority when included in an AMD-targeted runtime.

### Apple Silicon

```bash
./scripts/stage_model_runners.sh
```

Run this on Apple Silicon with current Xcode command-line tools and Python 3.12. It stages llama.cpp Metal and CPU, freezes the pinned MLX wrapper, builds the Swift Apple Intelligence runner, and writes the same manifest contract.

Build the managed Mirid backend from its prepared Python 3.12 environment with:

```bash
./scripts/build_sidecar.sh
```

The platform-neutral PyInstaller specification treats GPU and image bindings as optional. Apple and Linux release archives still require their own signed, hashed runtime publication; a Windows archive cannot be reused across operating systems.

After signing the Apple sidecar, stage its hashed runtime and update Tauri's Apple-only asset lock with:

```bash
./scripts/package_runtime_release.sh vN
```

Tauri selects a target-specific, compile-time asset lock. It restores executable permissions after download on macOS and refuses to compile unsupported desktop targets. The checked-in Apple lock deliberately reads `unpublished`; the macOS app should not be distributed until the native build script replaces it with the real version, sizes and SHA-256 hashes.

## Release checks

1. Probe every staged executable on its intended operating system.
2. Force CPU and load a small GGUF with no supported graphics device present.
3. On AMD Windows, prove HIP selection and then prove Vulkan fallback with HIP removed.
4. On Apple Silicon, load one GGUF and one MLX model; separately test Apple Intelligence in available, disabled and model-not-ready states.
5. Run onboarding's **Check this computer** action and confirm its copy remains platform-neutral.
6. Exercise non-streaming and streaming completions through Mirid's public model endpoint.

Apple artefacts cannot be compiled or hardware-tested on a Windows release workstation. They must be built and signed on Apple Silicon before a macOS runtime can be published.

The 19 July 2026 Windows publication was byte-verified in full. On an NVIDIA Windows x64 workstation, all official `llama-server` CPU, Vulkan and HIP Radeon executables passed `--version`; the current `stable-diffusion.cpp` CPU and Vulkan executables passed `--version`; and the Mirid-built CPU Python wheel passed a clean import. The ROCm image executables were mirrored and hash-verified but could not launch without AMD's runtime. That is a test boundary, not a product claim.
