from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from .sillytavern_bridge import (
    a1111_generation_arguments,
    active_image_model,
    list_image_models,
    normalise_base_url,
    resolve_image_model_path,
    router,
)


class FakeSdManager:
    def __init__(self):
        self.loaded = r"C:\models\portrait.gguf"

    def get_status(self):
        return {"loaded_models": {0: self.loaded}}

    def load_model(self, model_path, gpu_id=0):
        self.loaded = model_path
        return True

    def generate_image(self, **kwargs):
        assert kwargs["prompt"] == "a lighthouse"
        return b"png-bytes"


def test_normalise_base_url_removes_only_trailing_slashes():
    assert normalise_base_url(" http://127.0.0.1:8000/// ") == "http://127.0.0.1:8000"


def test_list_image_models_filters_and_sorts(tmp_path: Path):
    (tmp_path / "zeta.gguf").write_bytes(b"z")
    (tmp_path / "Alpha.safetensors").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("not a model", encoding="utf-8")

    models = list_image_models(str(tmp_path))

    assert [model["title"] for model in models] == ["Alpha.safetensors", "zeta.gguf"]
    assert models[0]["model_name"] == "Alpha"


def test_active_image_model_returns_loaded_filename():
    assert active_image_model(FakeSdManager()) == "portrait.gguf"


def test_resolve_image_model_path_stays_inside_directory(tmp_path: Path):
    model = tmp_path / "portrait.gguf"
    model.write_bytes(b"model")
    assert resolve_image_model_path(str(tmp_path), "portrait.gguf") == model.resolve()

    with pytest.raises(HTTPException) as error:
        resolve_image_model_path(str(tmp_path), "missing.gguf")
    assert error.value.status_code == 404


def test_generation_arguments_translate_a1111_names():
    arguments = a1111_generation_arguments(
        {
            "prompt": "  a moonlit harbour  ",
            "negative_prompt": "fog",
            "width": 1024,
            "height": 768,
            "steps": 28,
            "cfg_scale": 5.5,
            "seed": 42,
            "sampler_name": "dpm++2m",
        }
    )

    assert arguments == {
        "prompt": "a moonlit harbour",
        "negative_prompt": "fog",
        "width": 1024,
        "height": 768,
        "steps": 28,
        "cfg_scale": 5.5,
        "seed": 42,
        "sample_method": "dpm++2m",
        "gpu_id": 0,
        "task_id": arguments["task_id"],
    }
    assert arguments["task_id"].startswith("st-")


def test_capabilities_and_a1111_generation(tmp_path: Path):
    model = tmp_path / "portrait.gguf"
    model.write_bytes(b"model")
    app = FastAPI()
    app.include_router(router)
    app.state.sd_manager = FakeSdManager()
    app.state.sd_model_directory = str(tmp_path)
    client = TestClient(app)

    capabilities = client.get("/integrations/sillytavern/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["images"]["available"] is True

    models = client.get("/sdapi/v1/sd-models")
    assert models.json()[0]["title"] == "portrait.gguf"

    generated = client.post(
        "/sdapi/v1/txt2img",
        json={"prompt": "a lighthouse", "model": "portrait.gguf", "seed": 7},
    )
    assert generated.status_code == 200
    assert generated.json()["images"] == ["cG5nLWJ5dGVz"]
    assert generated.json()["parameters"]["seed"] == 7

    openai_image = client.post(
        "/v1/images/generations",
        json={"prompt": "a lighthouse", "model": "portrait.gguf", "size": "1024x768"},
    )
    assert openai_image.status_code == 200
    assert openai_image.json()["data"] == [{"b64_json": "cG5nLWJ5dGVz"}]
