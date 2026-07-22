import sys

from .compute_capabilities import disable_incompatible_torchao
from .model_manager import ModelManager
from .sd_worker import SDWorkerClient


def test_model_manager_uses_cpu_parameters_without_a_gpu():
    manager = ModelManager.__new__(ModelManager)
    manager.has_gpu = False
    manager.gpu_info = {"count": 0, "names": [], "memory": [], "cuda_version": None}
    manager.gpu_usage_mode = "unified_model"

    params = manager._get_gpu_params(0, context_length=8192)

    assert params["n_ctx"] == 8192
    assert params["n_gpu_layers"] == 0
    assert params["offload_kqv"] is False
    assert params["flash_attn"] is False
    assert "main_gpu" not in params


def test_sd_worker_does_not_spawn_during_backend_startup():
    client = SDWorkerClient()

    assert client._process is None
    assert client._address is None


def test_force_cpu_mode_makes_model_manager_ignore_an_installed_gpu(monkeypatch):
    monkeypatch.setenv("MIRID_FORCE_CPU", "1")

    manager = ModelManager(gpu_usage_mode="split_services")

    assert manager.has_gpu is False
    assert manager.gpu_info["count"] == 0


def test_optional_torchao_is_hidden_from_transformers(monkeypatch):
    monkeypatch.delitem(sys.modules, "torchao", raising=False)

    disable_incompatible_torchao()

    assert sys.modules["torchao"] is None
