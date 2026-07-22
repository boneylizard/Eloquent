import json
import sys
from pathlib import Path

from .local_runtime import (
    LocalRuntimeBroker,
    LocalRuntimeUnavailable,
    OpenAICompatibleModel,
    RunnerProbe,
    RuntimeRegistry,
    current_platform_key,
    format_for_model,
)


def _write_manifest(path: Path, runners):
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "contractVersion": 1,
                "versions": {"llamaCpp": "test"},
                "runners": runners,
            }
        ),
        encoding="utf-8",
    )


def test_platform_keys_cover_supported_desktops():
    assert current_platform_key("Windows", "AMD64") == "windows-x86_64"
    assert current_platform_key("Windows", "ARM64") == "windows-aarch64"
    assert current_platform_key("Darwin", "arm64") == "macos-aarch64"
    assert current_platform_key("Linux", "x86_64") == "linux-x86_64"


def test_windows_runner_order_prefers_cuda_then_hip_then_vulkan_then_cpu(tmp_path):
    manifest = Path(__file__).resolve().parents[2] / "runtime" / "model-runners.json"
    registry = RuntimeRegistry(
        manifest_path=manifest,
        runner_root=tmp_path,
        platform_key="windows-x86_64",
    )

    assert [candidate["accelerator"] for candidate in registry.candidates_for("gguf")] == [
        "nvidia",
        "amd",
        "vulkan",
        "cpu",
    ]


def test_force_cpu_removes_accelerated_candidates(tmp_path, monkeypatch):
    manifest = Path(__file__).resolve().parents[2] / "runtime" / "model-runners.json"
    monkeypatch.setenv("MIRID_FORCE_CPU", "1")
    registry = RuntimeRegistry(
        manifest_path=manifest,
        runner_root=tmp_path,
        platform_key="windows-x86_64",
    )

    candidates = registry.candidates_for("gguf")

    assert [candidate["id"] for candidate in candidates] == ["windows-cpu"]


def test_probe_and_capabilities_select_first_working_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("MIRID_FORCE_CPU", raising=False)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "id": "test-missing-gpu",
                "platform": "windows-x86_64",
                "engine": "llama.cpp",
                "accelerator": "nvidia",
                "priority": 500,
                "modelFormats": ["gguf"],
                "executable": "missing.exe",
                "probeArgs": ["--version"],
                "probeMarkers": ["llama"],
            },
            {
                "id": "test-cpu",
                "platform": "windows-x86_64",
                "engine": "llama.cpp",
                "accelerator": "cpu",
                "priority": 100,
                "modelFormats": ["gguf"],
                "executable": sys.executable,
                "probeArgs": ["-c", "print('llama version test')"],
                "probeMarkers": ["llama"],
            },
        ],
    )
    registry = RuntimeRegistry(
        manifest_path=manifest,
        runner_root=tmp_path,
        platform_key="windows-x86_64",
    )

    capabilities = registry.capabilities(diagnose_all=True)

    assert capabilities["contract_version"] == 1
    assert capabilities["formats"]["gguf"]["selected"]["id"] == "test-cpu"
    assert {runner["id"]: runner["available"] for runner in capabilities["runners"]} == {
        "test-missing-gpu": False,
        "test-cpu": True,
    }


def test_cpu_runner_command_disables_device_offload(tmp_path):
    executable = tmp_path / "llama-server.exe"
    executable.touch()
    runner = OpenAICompatibleModel(
        candidate={
            "id": "test-cpu",
            "accelerator": "cpu",
            "launchKind": "llama-server",
        },
        executable=executable,
        model_name="test.gguf",
        model_path=str(tmp_path / "test.gguf"),
        context_length=8192,
    )

    command = runner._command()

    assert command[command.index("--device") + 1] == "none"
    assert command[command.index("--gpu-layers") + 1] == "0"
    assert command[command.index("--ctx-size") + 1] == "8192"
    runner.shutdown()


def test_model_format_routes_system_and_mlx_models():
    assert format_for_model("mirid/apple-intelligence") == "system"
    assert format_for_model("mlx:mlx-community/Qwen3") == "mlx"
    assert format_for_model("roleplay.Q4_K_M.gguf") == "gguf"


def test_mlx_runner_uses_its_loaded_model_alias(tmp_path):
    executable = tmp_path / "mirid-mlx-runner"
    executable.touch()
    runner = OpenAICompatibleModel(
        candidate={
            "id": "test-mlx",
            "accelerator": "apple",
            "launchKind": "mlx-server",
        },
        executable=executable,
        model_name="mlx:mlx-community/Qwen3",
        model_path="mlx-community/Qwen3",
        context_length=8192,
    )

    assert runner.request_model_name == "default_model"
    runner.shutdown()


def test_broker_falls_through_when_accelerated_model_start_fails(tmp_path, monkeypatch):
    candidates = [
        {"id": "test-cuda", "accelerator": "nvidia", "launchKind": "llama-server"},
        {"id": "test-cpu", "accelerator": "cpu", "launchKind": "llama-server"},
    ]

    class Registry:
        def candidates_for(self, _model_format):
            return candidates

        def probe(self, candidate):
            return RunnerProbe(
                runner_id=candidate["id"],
                available=True,
                accelerator=candidate["accelerator"],
                engine="llama.cpp",
                executable=str(tmp_path / candidate["id"]),
            )

        def executable_for(self, candidate):
            return tmp_path / candidate["id"]

    def start(runner):
        if runner.runtime_id == "test-cuda":
            raise LocalRuntimeUnavailable("device unavailable")
        return runner

    monkeypatch.setattr(OpenAICompatibleModel, "start", start)
    broker = LocalRuntimeBroker(registry=Registry())

    runner = broker.start_model("test.gguf", str(tmp_path / "test.gguf"), 4096)

    assert runner.runtime_id == "test-cpu"
    runner.shutdown()
