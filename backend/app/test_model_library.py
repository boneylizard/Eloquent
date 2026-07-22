import pytest
import huggingface_hub

from . import model_library
from .model_library import (
    _clean_model_card,
    _civitai_error,
    _recommended_file,
    _run_download,
    _suggest_destination,
    companion_shards,
    normalise_civitai_model,
    normalize_nanogpt_subscription_models,
    parse_huggingface_reference,
)


def test_normalizes_nanogpt_subscription_model_payloads():
    payload = {"data": [{"id": "glm-5", "name": "GLM 5"}, {"name": "missing id"}]}
    assert normalize_nanogpt_subscription_models(payload) == [{"id": "glm-5", "name": "GLM 5"}]


def test_parses_repository_and_quantisation_reference():
    assert parse_huggingface_reference("bartowski/Qwen-GGUF:Q4_K_M") == (
        "bartowski/Qwen-GGUF",
        "main",
        "Q4_K_M",
    )


def test_parses_huggingface_tree_url():
    assert parse_huggingface_reference(
        "https://huggingface.co/author/model/tree/release"
    ) == ("author/model", "release", None)


def test_rejects_non_huggingface_urls():
    with pytest.raises(ValueError):
        parse_huggingface_reference("https://example.com/author/model")


def test_suggests_model_destinations_from_filename():
    assert _suggest_destination("model-Q4_K_M.gguf") == "text"
    assert _suggest_destination("sdxl-checkpoint.safetensors") == "image"
    assert _suggest_destination("face_yolov8n.pt") == "adetailer"
    assert _suggest_destination("4x-UltraSharp.pth") == "upscaler"


def test_collects_every_required_gguf_shard():
    files = [
        "model-Q8_0-00002-of-00002.gguf",
        "model-Q4_K_M.gguf",
        "model-Q8_0-00001-of-00002.gguf",
    ]
    assert companion_shards("model-Q8_0-00001-of-00002.gguf", files) == [
        "model-Q8_0-00001-of-00002.gguf",
        "model-Q8_0-00002-of-00002.gguf",
    ]


def test_recommended_file_totals_split_shards():
    class Sibling:
        def __init__(self, name, size):
            self.rfilename = name
            self.size = size
            self.lfs = None

    class Info:
        siblings = [
            Sibling("model-Q4_K_M-00001-of-00002.gguf", 10),
            Sibling("model-Q4_K_M-00002-of-00002.gguf", 20),
            Sibling("model-Q8_0.gguf", 40),
        ]

    result = _recommended_file(Info(), "Q4_K_M")
    assert result["size"] == 30
    assert len(result["filenames"]) == 2


def test_clean_model_card_keeps_current_prose_not_badges():
    card = """---
license: apache-2.0
---
![badge](badge.svg)

# Model

This is a sufficiently long paragraph describing the model and its intended use for conversational work.

| A | B | C | D |
"""
    cleaned = _clean_model_card(card)
    assert "intended use" in cleaned
    assert "license:" not in cleaned
    assert "badge.svg" not in cleaned


def test_downloads_directly_into_library_and_flattens_subfolders(tmp_path, monkeypatch):
    destination = tmp_path / "models"

    def fake_download(*, filename, local_dir, **_kwargs):
        downloaded = destination / filename
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"model")
        assert Path(local_dir) == destination
        return str(downloaded)

    from pathlib import Path

    monkeypatch.setattr(
        model_library,
        "model_destinations",
        lambda: {"text": {"path": str(destination)}},
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    _run_download("job", "owner/repo", ["quant/model.gguf"], "main", "text")

    assert (destination / "model.gguf").read_bytes() == b"model"
    assert not (destination / "quant" / "model.gguf").exists()


def test_civitai_keeps_only_scan_passed_primary_image_checkpoints():
    model = normalise_civitai_model({
        "id": 42,
        "name": "Example checkpoint",
        "type": "Checkpoint",
        "creator": {"username": "maker"},
        "stats": {"downloadCount": 12},
        "modelVersions": [{
            "id": 7,
            "name": "v1",
            "baseModel": "SDXL 1.0",
            "files": [
                {
                    "id": 8,
                    "name": "example.safetensors",
                    "sizeKB": 1024,
                    "primary": True,
                    "virusScanResult": "Success",
                    "pickleScanResult": "Success",
                    "metadata": {"format": "SafeTensor", "fp": "fp16"},
                    "hashes": {"SHA256": "ABC"},
                },
                {
                    "id": 9,
                    "name": "unsafe.ckpt",
                    "primary": False,
                    "virusScanResult": "Success",
                    "pickleScanResult": "Danger",
                },
            ],
        }],
    })

    assert model["creator"] == "maker"
    assert model["versions"][0]["file"] == {
        "id": 8,
        "filename": "example.safetensors",
        "size": 1024 * 1024,
        "format": "SafeTensor",
        "precision": "fp16",
        "primary": True,
        "virus_scan": "Success",
        "pickle_scan": "Success",
        "sha256": "ABC",
    }


def test_civitai_regional_failure_points_to_hugging_face():
    assert "Hugging Face" in _civitai_error(451)
    assert "regional" in _civitai_error(403)
