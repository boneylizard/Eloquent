from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


_DLL_DIRECTORY_HANDLES: list[object] = []
_PRELOADED_DLL_HANDLES: dict[str, object] = {}
_CUDA_RUNTIME_DLLS = (
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
)


def _deduplicate_existing_directories(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _installed_package_directory(package: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, AttributeError, ValueError):
        return None
    origin = getattr(spec, "origin", None)
    if not origin:
        return None
    return Path(origin).resolve().parent


def stable_diffusion_dependency_directories() -> list[Path]:
    """Return bundled CUDA dependency directories in preferred load order."""
    candidates: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        internal = Path(frozen_root)
        candidates.extend(
            [
                internal / "torch" / "lib",
                internal / "stable_diffusion_cpp" / "lib",
            ]
        )
    else:
        torch_directory = _installed_package_directory("torch")
        if torch_directory is not None:
            candidates.append(torch_directory / "lib")

    return _deduplicate_existing_directories(candidates)


def _preload_bundled_cuda_runtime(directories: Iterable[Path]) -> list[Path]:
    """
    Load one complete CUDA runtime set by absolute path.

    Mirid also ships CUDA libraries for independent model-runner executables.
    Those may target a different CUDA minor release and must never be mixed
    into the stable-diffusion.cpp process.
    """
    runtime_directory = next(
        (
            directory
            for directory in directories
            if all((directory / filename).is_file() for filename in _CUDA_RUNTIME_DLLS)
        ),
        None,
    )
    if runtime_directory is None:
        return []

    loaded: list[Path] = []
    for filename in _CUDA_RUNTIME_DLLS:
        dll_path = (runtime_directory / filename).resolve()
        key = os.path.normcase(str(dll_path))
        if key not in _PRELOADED_DLL_HANDLES:
            try:
                handle = ctypes.WinDLL(str(dll_path), winmode=0x00000008)
            except OSError as error:
                raise RuntimeError(
                    f"Mirid could not load its bundled image dependency '{dll_path}': {error}"
                ) from error
            _PRELOADED_DLL_HANDLES[key] = handle
        loaded.append(dll_path)
    return loaded


def prepare_stable_diffusion_dll_search() -> list[Path]:
    """
    Make the CUDA DLLs shipped with Mirid visible to stable-diffusion.cpp.

    The upstream binding passes ``winmode=0`` to ``ctypes.CDLL``. On Windows,
    that legacy loader path does not reliably honour ``os.add_dll_directory``;
    it does honour the process PATH. Keep both mechanisms active for the
    lifetime of the process so a normal user does not need a CUDA toolkit or
    CUDA_PATH installed. Preload the three CUDA dependencies from one directory
    so an installed toolkit cannot supply a conflicting DLL with the same name.
    """
    if sys.platform != "win32":
        return []

    directories = stable_diffusion_dependency_directories()
    if not directories:
        return []

    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    existing = {os.path.normcase(os.path.abspath(entry)) for entry in current_entries}
    prepend = [str(path) for path in directories if os.path.normcase(str(path)) not in existing]
    if prepend:
        os.environ["PATH"] = os.pathsep.join([*prepend, *current_entries])

    add_directory = getattr(os, "add_dll_directory", None)
    if add_directory is not None:
        held = {
            os.path.normcase(str(getattr(handle, "path", "")))
            for handle in _DLL_DIRECTORY_HANDLES
        }
        for directory in directories:
            key = os.path.normcase(str(directory))
            if key in held:
                continue
            try:
                handle = add_directory(str(directory))
            except OSError:
                continue
            _DLL_DIRECTORY_HANDLES.append(handle)
            held.add(key)

    _preload_bundled_cuda_runtime(directories)
    return directories


@contextmanager
def stable_diffusion_import_environment() -> Iterator[list[Path]]:
    """Prepare Mirid's runtime and hide an unrelated CUDA toolkit during import."""
    directories = prepare_stable_diffusion_dll_search()
    saved_cuda_path = os.environ.pop("CUDA_PATH", None) if directories else None
    try:
        yield directories
    finally:
        if saved_cuda_path is not None:
            os.environ["CUDA_PATH"] = saved_cuda_path
