"""MLX models are directories, so a *.gguf scan never finds them."""

import json
from pathlib import Path

from .model_manager import ModelManager


class _Broker:
    def __init__(self, mlx_available=True):
        self._mlx_available = mlx_available

    def capabilities(self, *args, **kwargs):
        return {
            "formats": {
                "mlx": {"available": self._mlx_available},
                "system": {"available": False},
            }
        }


def _manager(models_dir: Path, mlx_available=True) -> ModelManager:
    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = models_dir
    manager.runtime_broker = _Broker(mlx_available)
    return manager


def _write_model(directory: Path, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"weights")
    return directory


def test_discovers_quantised_mlx_directory(tmp_path):
    _write_model(tmp_path / "Qwen-4bit", {"model_type": "qwen2", "quantization": {"bits": 4}})

    assert _manager(tmp_path).discover_mlx_models() == ["mlx:Qwen-4bit"]


def test_discovers_directory_named_for_mlx_without_quantisation(tmp_path):
    _write_model(tmp_path / "Qwen3-MLX-bf16", {"model_type": "qwen3"})

    assert _manager(tmp_path).discover_mlx_models() == ["mlx:Qwen3-MLX-bf16"]


def test_ignores_plain_transformers_checkpoint(tmp_path):
    _write_model(tmp_path / "roberta-base", {"model_type": "roberta"})

    assert _manager(tmp_path).discover_mlx_models() == []


def test_ignores_directory_without_weights(tmp_path):
    directory = tmp_path / "Qwen-4bit"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}), encoding="utf-8")

    assert _manager(tmp_path).discover_mlx_models() == []


def test_reports_nothing_when_the_mlx_runner_is_unavailable(tmp_path):
    _write_model(tmp_path / "Qwen-4bit", {"quantization": {"bits": 4}})

    assert _manager(tmp_path, mlx_available=False).discover_mlx_models() == []


def test_survives_a_missing_models_directory(tmp_path):
    assert _manager(tmp_path / "absent").discover_mlx_models() == []


def test_ignores_unreadable_configuration(tmp_path):
    directory = tmp_path / "Broken-4bit"
    directory.mkdir()
    (directory / "config.json").write_text("{not json", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"weights")

    assert _manager(tmp_path).discover_mlx_models() == []
