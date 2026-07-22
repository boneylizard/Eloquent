---
pretty_name: Mirid Runtime Packs
license: other
task_categories:
  - text-generation
  - text-to-image
tags:
  - mirid
  - llama.cpp
  - stable-diffusion.cpp
  - local-inference
  - vulkan
  - rocm
  - metal
---

# Mirid Runtime Packs

Verified, independently updateable local-inference packages used by [Mirid](https://mirid.ai).

Each entry in `runtime-packages.manifest.json` records the upstream project and release, original download URL, byte size and SHA-256 digest. Files mirrored from upstream are stored without modification. Mirid-built Python wheels additionally record their source archive, build arguments and validation result.

## Layout

- `runners/text/`: private GGUF inference runners from `llama.cpp`.
- `runners/image/`: private image inference runners from `stable-diffusion.cpp`.
- `bindings/`: Python wheels retained for Mirid's embedded compatibility path.

CPU packages are the universal baseline. Vulkan, HIP, ROCm and Metal packages are optional acceleration paths selected only after a local device probe succeeds.

The Apple Silicon binding set includes `mlx`, `mlx-metal` and `mlx-lm` for Python 3.12 on macOS 14 or newer. Their ordinary pure-Python dependencies should still be resolved by the prepared Mirid environment.

## Trust and licences

These packages are redistributions or builds of the named upstream projects. Their original licences remain authoritative:

- [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
- [`abetlen/llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)
- [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
- [`william-murray1204/stable-diffusion-cpp-python`](https://github.com/william-murray1204/stable-diffusion-cpp-python)

`source-sha256-verified` means Mirid matched the published size and SHA-256 after downloading the file. `import-passed` means a Mirid-built wheel was also installed into a clean environment and imported successfully. Neither label claims that every accelerator has been exercised on physical hardware; Mirid's release notes record those tests separately.
