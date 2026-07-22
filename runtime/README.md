# Mirid inference runtime releases

Mirid freezes one application backend and ships hardware-specific model runners behind it. NVIDIA is an acceleration path, not a launch requirement. The runner contract and platform order live in `model-runners.json`; architecture notes and validation requirements live in `docs/CROSS_PLATFORM_RUNTIME.md`.

Public runtime packages are published independently of the desktop installer:

- [Mirid Runtime Packs](https://huggingface.co/datasets/boneylizardwizard/mirid-runtime-packs) stores immutable upstream archives, official binding wheels, Mirid-built compatibility wheels and the generated manifest.
- [Mirid Runtime Registry](https://huggingface.co/spaces/boneylizardwizard/mirid-runtime-registry) gives people a readable view of the same manifest.

## Refresh the package registry

```powershell
python .\scripts\refresh_runtime_package_lock.py
.\scripts\build_portable_inference_wheels.ps1 -Backends cpu
python .\scripts\stage_runtime_packages.py
python .\scripts\publish_runtime_packages.py
```

The resolver turns moving upstream release channels into exact filenames, byte sizes and SHA-256 hashes in `runtime-packages.lock.json`. The stager verifies every byte before writing `runtime-packages.manifest.json`. The publisher uses the authenticated Hugging Face CLI and creates the Dataset and static Space when they do not yet exist.

`stable-diffusion.cpp` native runners are the current image-inference path. The `stable-diffusion-cpp-python` wheel remains a compatibility package because the Python project's latest source distribution trails the native engine. Its Windows CPU wheel is built from the pinned source archive and import-tested in a clean Python 3.12 environment. Accelerator-specific image runners come from the newer official native release instead of pretending an older Python binding is current.

## Update and publish

1. Review and update the runner pins in `model-runners.json`.
2. Stage and probe the platform runners with `scripts\stage_model_runners.ps1` on Windows or `scripts/stage_model_runners.sh` on Apple Silicon.
3. Refresh the portable package lock and stage the public runtime registry with the commands above.
4. Refresh optional embedded CUDA wheel pins with `scripts\refresh_inference_wheel_lock.ps1`.
5. Review upstream API changes used by `backend\app\model_manager.py` and `backend\app\sd_manager.py`.
6. Build and validate release wheels with `scripts\build_inference_wheels.ps1` when retaining the embedded CUDA fallback.
7. Upload the wheels and generated manifest to the public `boneylizardwizard/mirid-cuda-wheels` Dataset, then run `npm run release:check:inference-wheels` from `frontend`.
8. Copy the reviewed generated manifest to `runtime\inference-wheels.release.json`.
9. Build the frozen backend with `scripts\build_sidecar.ps1`. The default release carries CUDA, Vulkan and CPU; pass `-ModelRunnerBackends hip,vulkan,cpu` for an AMD-targeted runtime. The staged runners are copied into `_internal\runners`.
10. Stage a hashed runtime release with `scripts\package_runtime_release.ps1 -RuntimeVersion vN`.
11. Upload the staged runtime assets, then run `npm run release:check:hosted-runtime` from `frontend`.

`scripts\fetch_inference_wheels.ps1` lets a clean release workstation download the exact reviewed wheels with resume, size validation, and SHA-256 verification.
