from __future__ import annotations

import asyncio
import base64
import os
import random
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field


router = APIRouter(tags=["SillyTavern integration"])

IMAGE_EXTENSIONS = {".safetensors", ".ckpt", ".gguf"}
SD_SAMPLERS = [
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
]
SD_SCHEDULERS = ["discrete", "karras", "exponential", "ays", "gits"]


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str = Field(min_length=1)
    model: str = "tts-1"
    voice: str = "af_heart"
    response_format: str = "wav"
    speed: float = 1.0
    engine: Optional[str] = None


def normalise_base_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def list_image_models(model_directory: Optional[str]) -> List[Dict[str, Any]]:
    if not model_directory:
        return []
    directory = Path(model_directory)
    if not directory.is_dir():
        return []

    return [
        {
            "title": path.name,
            "model_name": path.stem,
            "hash": None,
            "sha256": None,
            "filename": str(path),
            "config": None,
        }
        for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def active_image_model(sd_manager: Any) -> str:
    if not sd_manager:
        return ""
    try:
        loaded = (sd_manager.get_status() or {}).get("loaded_models", {})
    except Exception:
        return ""
    if not isinstance(loaded, dict) or not loaded:
        return ""
    first_path = next(iter(loaded.values()), "")
    return Path(str(first_path)).name if first_path else ""


def resolve_image_model_path(model_directory: Optional[str], model_name: str) -> Path:
    if not model_directory:
        raise HTTPException(status_code=409, detail="Set an image model directory in Mirid first.")

    directory = Path(model_directory).resolve()
    candidate = (directory / Path(model_name).name).resolve()
    if candidate.parent != directory or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Image model not found: {model_name}")
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="That file is not a supported image model.")
    return candidate


def a1111_generation_arguments(body: Dict[str, Any]) -> Dict[str, Any]:
    seed = body.get("seed", -1)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    return {
        "prompt": str(body.get("prompt") or "").strip(),
        "negative_prompt": str(body.get("negative_prompt") or ""),
        "width": int(body.get("width") or 768),
        "height": int(body.get("height") or 512),
        "steps": int(body.get("steps") or 20),
        "cfg_scale": float(body.get("cfg_scale", body.get("guidance_scale", 7.0))),
        "seed": seed,
        "sample_method": str(body.get("sampler_name") or body.get("sampler") or "euler"),
        "gpu_id": int(body.get("gpu_id") or 0),
        "task_id": str(body.get("task_id") or f"st-{uuid.uuid4().hex}"),
    }


def capabilities_payload(request: Request) -> Dict[str, Any]:
    base_url = normalise_base_url(str(request.base_url))
    sd_manager = getattr(request.app.state, "sd_manager", None)
    model_directory = getattr(request.app.state, "sd_model_directory", None)
    image_models = list_image_models(model_directory)
    return {
        "service": "Mirid",
        "integration": "SillyTavern",
        "status": "ready",
        "text": {
            "available": True,
            "openai_base_url": f"{base_url}/v1",
            "models_url": f"{base_url}/v1/models",
            "streaming": True,
        },
        "speech": {
            "available": True,
            "openai_speech_url": f"{base_url}/v1/audio/speech",
            "voices_url": f"{base_url}/tts/voices",
        },
        "transcription": {
            "available": True,
            "openai_transcriptions_url": f"{base_url}/v1/audio/transcriptions",
            "engines_url": f"{base_url}/stt/available-engines",
        },
        "images": {
            "available": bool(sd_manager),
            "openai_images_url": f"{base_url}/v1/images/generations",
            "automatic1111_base_url": base_url,
            "stable_diffusion_cpp_base_url": base_url,
            "configured_models": len(image_models),
            "loaded_model": active_image_model(sd_manager),
        },
        "authentication": {
            "scheme": "bearer_or_basic",
            "note": "Use Mirid's remote-access password when one is configured.",
        },
    }


@router.get("/integrations/sillytavern/capabilities")
async def sillytavern_capabilities(request: Request):
    return capabilities_payload(request)


@router.post("/v1/audio/speech")
async def openai_audio_speech(body: SpeechRequest):
    if body.response_format.lower() not in {"wav", "wave"}:
        raise HTTPException(
            status_code=400,
            detail="Mirid currently returns WAV audio. Set response_format to wav.",
        )

    engine = body.engine or (body.model if body.model not in {"tts-1", "tts-1-hd"} else "kokoro")
    try:
        from .tts_service import synthesize_speech

        audio_bytes = await synthesize_speech(
            text=body.input,
            voice=body.voice,
            engine=engine,
            speed=body.speed,
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Mirid could not synthesise that speech: {error}") from error


@router.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(
    request: Request,
):
    form = await request.form()
    file = form.get("file")
    if not file or not hasattr(file, "read"):
        raise HTTPException(status_code=400, detail="Attach an audio file as multipart field 'file'.")
    model = str(form.get("model") or "whisper-1")
    engine_value = form.get("engine")
    engine = str(engine_value) if engine_value else None
    response_format = str(form.get("response_format") or "json")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The audio upload is empty.")

    selected_engine = engine or ("whisper" if model in {"whisper-1", "whisper"} else model)
    try:
        from .stt_service import transcribe_audio, transcribe_audio_bytes

        if content[:4] == b"RIFF":
            transcript = await transcribe_audio_bytes(content, selected_engine)
        else:
            suffix = Path(file.filename or "recording.webm").suffix or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                temporary_file.write(content)
                temporary_path = temporary_file.name
            try:
                transcript = await transcribe_audio(temporary_path, selected_engine)
            finally:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Mirid could not transcribe that recording: {error}") from error

    if response_format == "text":
        return Response(content=str(transcript), media_type="text/plain")
    return {"text": transcript, "transcript": transcript}


@router.options("/v1/images/generations")
async def stable_diffusion_cpp_probe():
    return Response(status_code=200)


@router.post("/v1/images/generations")
async def openai_image_generations(body: Dict[str, Any], request: Request):
    payload = dict(body)
    size = str(payload.get("size") or "")
    if "x" in size.lower():
        try:
            width, height = size.lower().split("x", 1)
            payload["width"] = int(width)
            payload["height"] = int(height)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Image size must look like 1024x1024.")
    result = await a1111_txt2img(payload, request)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": image} for image in result["images"]],
    }


@router.get("/sdapi/v1/sd-models")
async def a1111_models(request: Request):
    return list_image_models(getattr(request.app.state, "sd_model_directory", None))


@router.get("/sdapi/v1/options")
async def a1111_options(request: Request):
    return {
        "sd_model_checkpoint": active_image_model(getattr(request.app.state, "sd_manager", None)),
        "samples_format": "png",
    }


@router.post("/sdapi/v1/options")
async def a1111_set_options(body: Dict[str, Any], request: Request):
    model_name = str(body.get("sd_model_checkpoint") or "").strip()
    if not model_name:
        return await a1111_options(request)

    sd_manager = getattr(request.app.state, "sd_manager", None)
    if not sd_manager:
        raise HTTPException(status_code=503, detail="Mirid's local image engine is not available.")
    model_path = resolve_image_model_path(getattr(request.app.state, "sd_model_directory", None), model_name)
    loaded = await asyncio.to_thread(sd_manager.load_model, str(model_path), 0)
    if not loaded:
        raise HTTPException(status_code=500, detail=f"Mirid could not load {model_path.name}.")
    return {"sd_model_checkpoint": model_path.name, "samples_format": "png"}


@router.get("/sdapi/v1/samplers")
async def a1111_samplers():
    return [{"name": name, "aliases": [], "options": {}} for name in SD_SAMPLERS]


@router.get("/sdapi/v1/schedulers")
async def a1111_schedulers():
    return [{"name": name, "label": name} for name in SD_SCHEDULERS]


@router.get("/sdapi/v1/upscalers")
async def a1111_upscalers():
    return [{"name": "None", "model_name": None, "model_path": None, "model_url": None, "scale": 1}]


@router.get("/sdapi/v1/latent-upscale-modes")
async def a1111_latent_upscalers():
    return [{"name": "Latent", "mode": {"mode": "bilinear", "antialias": False}}]


@router.get("/sdapi/v1/sd-vae")
async def a1111_vaes():
    return []


@router.get("/sdapi/v1/progress")
async def a1111_progress(request: Request):
    state = getattr(request.app.state, "sillytavern_image_job", None) or {}
    progress = float(state.get("progress", 0.0))
    active = bool(state.get("active", False))
    return {
        "progress": progress,
        "eta_relative": float(state.get("eta_relative", 0.0)),
        "state": {"job_count": 1 if active else 0, "job": state.get("job", "")},
        "current_image": None,
        "textinfo": state.get("textinfo", ""),
    }


@router.post("/sdapi/v1/interrupt")
async def a1111_interrupt(request: Request):
    request.app.state.sillytavern_image_job = {
        "active": False,
        "progress": 0.0,
        "textinfo": "The client stopped waiting. Native generation may still be finishing.",
    }
    return {}


@router.post("/sdapi/v1/txt2img")
async def a1111_txt2img(body: Dict[str, Any], request: Request):
    sd_manager = getattr(request.app.state, "sd_manager", None)
    if not sd_manager:
        raise HTTPException(status_code=503, detail="Mirid's local image engine is not available.")

    arguments = a1111_generation_arguments(body)
    if not arguments["prompt"]:
        raise HTTPException(status_code=400, detail="An image prompt is required.")

    override_settings = body.get("override_settings")
    if not isinstance(override_settings, dict):
        override_settings = {}
    requested_model = str(body.get("model") or override_settings.get("sd_model_checkpoint") or "").strip()
    if requested_model and requested_model != active_image_model(sd_manager):
        model_path = resolve_image_model_path(
            getattr(request.app.state, "sd_model_directory", None),
            requested_model,
        )
        loaded = await asyncio.to_thread(sd_manager.load_model, str(model_path), arguments["gpu_id"])
        if not loaded:
            raise HTTPException(status_code=500, detail=f"Mirid could not load {model_path.name}.")

    request.app.state.sillytavern_image_job = {
        "active": True,
        "progress": 0.01,
        "job": arguments["task_id"],
        "textinfo": "Mirid is rendering the image.",
    }
    try:
        image_bytes = await asyncio.to_thread(sd_manager.generate_image, **arguments)
    except Exception as error:
        request.app.state.sillytavern_image_job = {
            "active": False,
            "progress": 0.0,
            "textinfo": str(error),
        }
        raise HTTPException(status_code=500, detail=f"Mirid could not render that image: {error}") from error

    request.app.state.sillytavern_image_job = {
        "active": False,
        "progress": 0.0,
        "job": arguments["task_id"],
        "textinfo": "Image complete.",
    }
    return {
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "parameters": {**body, "seed": arguments["seed"]},
        "info": "Generated by Mirid with stable-diffusion.cpp",
    }
