from __future__ import annotations

import argparse
from pathlib import Path

import pefile


CUDA_RUNTIME_DLLS = {
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublaslt64_12.dll",
}


def imported_dlls(path: Path) -> set[str]:
    image = pefile.PE(str(path), fast_load=True)
    image.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
    )
    return {
        entry.dll.decode("ascii", errors="replace").lower()
        for entry in getattr(image, "DIRECTORY_ENTRY_IMPORT", [])
    }


def require_imports(path: Path, required: set[str]) -> None:
    imports = imported_dlls(path)
    missing = sorted(required - imports)
    if missing:
        raise SystemExit(
            f"{path.name} is missing expected imports: {', '.join(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the frozen Windows image runtime without loading a GPU driver."
    )
    parser.add_argument("internal", type=Path)
    args = parser.parse_args()

    internal = args.internal.resolve()
    stable_library = (
        internal / "stable_diffusion_cpp" / "lib" / "stable-diffusion.dll"
    )
    torch_library = internal / "torch" / "lib"

    if not stable_library.is_file():
        raise SystemExit(f"Stable Diffusion library is missing: {stable_library}")
    if not torch_library.is_dir():
        raise SystemExit(f"Bundled CUDA directory is missing: {torch_library}")

    bundled = {path.name.lower() for path in torch_library.glob("*.dll")}
    missing = sorted(CUDA_RUNTIME_DLLS - bundled)
    if missing:
        raise SystemExit(
            "The bundled image runtime is incomplete: " + ", ".join(missing)
        )

    require_imports(
        stable_library,
        {"cudart64_12.dll", "cublas64_12.dll", "nvcuda.dll"},
    )
    require_imports(
        torch_library / "cublas64_12.dll",
        {"cublaslt64_12.dll"},
    )
    print(
        "Frozen image runtime dependency closure verified:",
        stable_library,
        torch_library,
    )


if __name__ == "__main__":
    main()
