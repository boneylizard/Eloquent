from pathlib import Path

from . import windows_dll_paths


def test_frozen_dependency_directories_exclude_model_runner_cuda(monkeypatch, tmp_path):
    torch_lib = tmp_path / "torch" / "lib"
    stable_diffusion_lib = tmp_path / "stable_diffusion_cpp" / "lib"
    runner_lib = tmp_path / "runners" / "windows-x86_64" / "cuda12"
    torch_lib.mkdir(parents=True)
    stable_diffusion_lib.mkdir(parents=True)
    runner_lib.mkdir(parents=True)
    monkeypatch.setattr(windows_dll_paths.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert windows_dll_paths.stable_diffusion_dependency_directories() == [
        torch_lib.resolve(),
        stable_diffusion_lib.resolve(),
    ]


def test_prepare_stable_diffusion_search_prepends_path_and_holds_handles(
    monkeypatch,
    tmp_path,
):
    dependency_dir = tmp_path / "torch" / "lib"
    dependency_dir.mkdir(parents=True)
    created: list[str] = []

    class Handle:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(windows_dll_paths.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_dll_paths,
        "stable_diffusion_dependency_directories",
        lambda: [dependency_dir.resolve()],
    )
    monkeypatch.setattr(
        windows_dll_paths.os,
        "add_dll_directory",
        lambda path: created.append(path) or Handle(path),
        raising=False,
    )
    monkeypatch.setenv("PATH", str(tmp_path / "existing"))
    monkeypatch.setattr(windows_dll_paths, "_DLL_DIRECTORY_HANDLES", [])

    first = windows_dll_paths.prepare_stable_diffusion_dll_search()
    second = windows_dll_paths.prepare_stable_diffusion_dll_search()

    assert first == second == [dependency_dir.resolve()]
    assert windows_dll_paths.os.environ["PATH"].split(windows_dll_paths.os.pathsep)[0] == str(
        dependency_dir.resolve()
    )
    assert created == [str(dependency_dir.resolve())]
    assert len(windows_dll_paths._DLL_DIRECTORY_HANDLES) == 1


def test_prepare_preloads_one_complete_cuda_runtime_in_dependency_order(
    monkeypatch,
    tmp_path,
):
    torch_lib = tmp_path / "torch" / "lib"
    runner_lib = tmp_path / "runners" / "windows-x86_64" / "cuda12"
    torch_lib.mkdir(parents=True)
    runner_lib.mkdir(parents=True)
    for filename in windows_dll_paths._CUDA_RUNTIME_DLLS:
        (torch_lib / filename).write_bytes(b"torch")
        (runner_lib / filename).write_bytes(b"runner")

    loaded: list[str] = []
    monkeypatch.setattr(windows_dll_paths.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_dll_paths,
        "stable_diffusion_dependency_directories",
        lambda: [torch_lib.resolve(), runner_lib.resolve()],
    )
    monkeypatch.setattr(
        windows_dll_paths.os,
        "add_dll_directory",
        lambda path: type("Handle", (), {"path": path})(),
        raising=False,
    )
    monkeypatch.setattr(
        windows_dll_paths.ctypes,
        "WinDLL",
        lambda path, winmode: loaded.append(path) or object(),
    )
    monkeypatch.setattr(windows_dll_paths, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(windows_dll_paths, "_PRELOADED_DLL_HANDLES", {})

    windows_dll_paths.prepare_stable_diffusion_dll_search()

    assert loaded == [
        str((torch_lib / filename).resolve())
        for filename in windows_dll_paths._CUDA_RUNTIME_DLLS
    ]


def test_import_environment_temporarily_hides_cuda_path(monkeypatch, tmp_path):
    dependency_dir = tmp_path / "torch" / "lib"
    dependency_dir.mkdir(parents=True)
    monkeypatch.setattr(
        windows_dll_paths,
        "prepare_stable_diffusion_dll_search",
        lambda: [dependency_dir],
    )
    monkeypatch.setenv("CUDA_PATH", "C:\\system-cuda")

    with windows_dll_paths.stable_diffusion_import_environment() as directories:
        assert directories == [dependency_dir]
        assert "CUDA_PATH" not in windows_dll_paths.os.environ

    assert windows_dll_paths.os.environ["CUDA_PATH"] == "C:\\system-cuda"


def test_prepare_stable_diffusion_search_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(windows_dll_paths.sys, "platform", "linux")
    monkeypatch.setattr(
        windows_dll_paths,
        "stable_diffusion_dependency_directories",
        lambda: [Path("/should/not/be/used")],
    )

    assert windows_dll_paths.prepare_stable_diffusion_dll_search() == []
