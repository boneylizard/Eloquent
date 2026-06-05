"""HTTP routes for the voice reference merge pipeline (optional clean → morph → optional RVC)."""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .automation_service import AutomationService, SculptRequest

logger = logging.getLogger("voice_sculpt_routes")

voice_sculpt_router = APIRouter(tags=["voice-sculpt"])


class SculptStreamBody(BaseModel):
    source: str = Field(default="", description="One path or newline/pipe-separated paths")
    sources: Optional[list[str]] = Field(default=None, description="Explicit list of audio paths")
    source_type: str = Field(default="local_path", description="local_path or youtube_url")
    output_name: Optional[str] = None
    accent_model: str = Field(default="default", description="RVC model name or default | target_accent")
    skip_rvc: bool = Field(default=True, description="Skip optional RVC polish (merge output is the product)")
    skip_uvr: bool = Field(default=True, description="Skip UVR — inputs are usually pre-clipped clean refs")
    combine_mode: str = Field(default="morph", description="morph (timbre blend) | mix | concat")
    morph_balance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Two-clip morph: 0=first voice, 1=second (full float, e.g. 0.953)",
    )
    pitch: int = Field(default=0, ge=-12, le=12)
    index_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    volume_envelope: float = Field(default=1.0, ge=0.0, le=1.0)
    voice_prompt: Optional[str] = Field(
        default=None,
        description="Optional text note saved beside output (.prompt.txt); TTS engines ignore for now",
    )


class AutoSetupBody(BaseModel):
    clone_applio: bool = False
    install_uvr: bool = True
    write_env_file: bool = True
    applio_dest: Optional[str] = None


class InstallHfModelBody(BaseModel):
    url: str = Field(..., description="Hugging Face repo or file URL")
    applio_dest: Optional[str] = None


def get_automation_service(request: Request) -> AutomationService:
    svc = getattr(request.app.state, "automation_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Voice sculpt automation service is not available.")
    return svc


@voice_sculpt_router.get("/discover")
async def voice_sculpt_discover(request: Request):
    """Auto-detect Applio install, models, and UVR/ffmpeg binaries."""
    svc = get_automation_service(request)
    return await svc.discover()


@voice_sculpt_router.post("/auto-setup")
async def voice_sculpt_auto_setup(request: Request, body: AutoSetupBody = Body(default_factory=AutoSetupBody)):
    """
    One-click bootstrap: pip install audio-separator, optional git clone Applio,
    write sculpt.env.bat, apply env to running backend (no restart required).
    """
    svc = get_automation_service(request)
    try:
        return await svc.auto_setup(
            clone_applio=body.clone_applio,
            install_uvr=body.install_uvr,
            write_env_file=body.write_env_file,
            applio_dest=body.applio_dest,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc


@voice_sculpt_router.post("/install-hf-model")
async def voice_sculpt_install_hf_model(request: Request, body: InstallHfModelBody):
    """
    Install an RVC voice model from Hugging Face using huggingface_hub (lists repo files,
    downloads .pth + .index). Applio's own Model Link box only scrapes HTML for .zip files.
    """
    svc = get_automation_service(request)
    try:
        return await svc.install_huggingface_model(body.url, applio_dest=body.applio_dest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc


@voice_sculpt_router.get("/preflight")
async def voice_sculpt_preflight(
    request: Request,
    for_youtube: bool = False,
    for_uvr: bool = False,
    for_rvc: bool = False,
    for_morph: bool = True,
    accent_model: str = "default",
):
    """Check whether required external tools are available."""
    svc = get_automation_service(request)
    return await svc.preflight(
        for_youtube=for_youtube,
        for_uvr=for_uvr,
        for_rvc=for_rvc,
        for_morph=for_morph,
        accent_model=accent_model,
    )


@voice_sculpt_router.post("/sculpt-stream")
async def voice_sculpt_stream(request: Request, body: SculptStreamBody):
    """
    SSE stream for the full sculpt pipeline.
    Events: progress (step 1–3), done (voice_id, path), or error (412/500).
    """
    if body.source_type not in ("local_path", "youtube_url"):
        raise HTTPException(status_code=400, detail="source_type must be local_path or youtube_url")

    svc = get_automation_service(request)
    sculpt_request = SculptRequest(
        source=body.source.strip(),
        sources=body.sources,
        source_type=body.source_type,
        output_name=(body.output_name or "").strip() or None,
        accent_model=body.accent_model or "default",
        skip_rvc=body.skip_rvc,
        skip_uvr=body.skip_uvr,
        combine_mode=body.combine_mode or "morph",
        morph_balance=body.morph_balance,
        pitch=body.pitch,
        index_rate=body.index_rate,
        protect=body.protect,
        volume_envelope=body.volume_envelope,
        voice_prompt=(body.voice_prompt or "").strip() or None,
    )

    async def event_stream():
        async for event in svc.sculpt_stream(sculpt_request):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
