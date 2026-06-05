"""HTTP routes for D-ID talking-head pipeline (optional; requires env credentials)."""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import d_id_assets, d_id_batch, d_id_service, d_id_vision_screen

logger = logging.getLogger("d_id_routes")

d_id_router = APIRouter(tags=["d-id"])


def _allowed_concat_roots():
    exp = d_id_service.get_d_id_export_dir().resolve()
    return [
        exp,
        (exp.parent / "tts_full_exports").resolve(),
        d_id_service.get_d_id_batch_runs_dir().resolve(),
    ]


def _is_under_allowed_root(p: Path) -> bool:
    sp = str(p)
    for root in _allowed_concat_roots():
        r = str(root)
        if sp == r or sp.startswith(r + os.sep):
            return True
    return False


@d_id_router.get("/d-id/status")
async def d_id_status():
    """Whether D-ID credentials are present (no secrets returned)."""
    configured = d_id_service.d_id_credentials_configured()
    vision_model = (os.environ.get("D_ID_VISION_SCREEN_MODEL") or "").strip()
    public_base = (os.environ.get("D_ID_PUBLIC_BASE_URL") or "").strip()
    return {
        "status": "ok",
        "d_id_configured": configured,
        "api_base": d_id_service.get_api_base(),
        "vision_screen_model_configured": bool(vision_model),
        "public_base_configured": bool(public_base),
    }


@d_id_router.post("/d-id/talk-from-wav")
async def d_id_talk_from_wav(
    audio: UploadFile = File(..., description="WAV (or other) audio for lip-sync"),
    source_url: Optional[str] = Form(
        None,
        description="Public https URL of the face image. If omitted, uses D_ID_DEFAULT_SOURCE_URL.",
    ),
    avatar_ref: Optional[str] = Form(
        None,
        description="Alias: same as source_url if it is an https URL; otherwise combined with default.",
    ),
    emotion: Optional[str] = Form(None, description="happy | neutral | surprised | serious"),
    movement: Optional[str] = Form(None, description="active | still"),
    background_url: Optional[str] = Form(None, description="Optional https background image URL."),
):
    """
    Upload audio → D-ID /audios → POST /talks → poll → save MP4 under backend/data/d_id_exports/.
    """
    try:
        ref = (source_url or avatar_ref or "").strip()
        suffix = Path(audio.filename or "speech.wav").suffix or ".wav"
        tmp = Path(tempfile.mkdtemp(prefix="did_wav_")) / f"upload{suffix}"
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio upload.")
        tmp.write_bytes(data)
        out, meta = await d_id_service.wav_to_talk_mp4(
            tmp,
            ref or "",
            emotion=emotion,
            movement=movement,
            background_url=background_url,
        )
        return {
            "status": "success",
            "mp4_path": str(out),
            "talk": meta,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("d-id talk-from-wav failed")
        raise HTTPException(status_code=500, detail=str(e))


class DIdConcatBody(BaseModel):
    """Absolute or repo-relative paths to MP4 segments (server-side only)."""

    input_paths: List[str] = Field(..., min_length=1)
    output_filename: str = Field(default="d_id_concat_output.mp4")


@d_id_router.post("/d-id/concat-mp4s")
async def d_id_concat_mp4s(body: DIdConcatBody):
    """
    ffmpeg concat demuxer → one MP4 (same approach as merge_videos.py).
    Paths must be under allowed export directories.
    """
    resolved_inputs: List[Path] = []
    for raw in body.input_paths:
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {raw}")
        if not _is_under_allowed_root(p):
            raise HTTPException(
                status_code=403,
                detail="Each path must be under d_id_exports, tts_full_exports, or d_id_batch_runs.",
            )
        resolved_inputs.append(p)

    out_dir = d_id_service.get_d_id_export_dir()
    out = out_dir / Path(body.output_filename).name
    try:
        d_id_service.concat_mp4_files_ffmpeg(resolved_inputs, out)
    except Exception as e:
        logger.exception("d-id concat failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "output_path": str(out)}


async def _batch_ndjson_stream(
    *,
    segment_files: List[UploadFile],
    avatar_source_url: str,
    emotion: Optional[str],
    movement: Optional[str],
    background_url: Optional[str],
    concurrency: int,
    require_vision: bool,
):
    tmpdir = None
    try:
        if require_vision:
            mid = (os.environ.get("D_ID_VISION_SCREEN_MODEL") or "").strip()
            if not mid:
                yield json.dumps(
                    {
                        "event": "error",
                        "message": "require_vision is true but D_ID_VISION_SCREEN_MODEL is not set.",
                    }
                ) + "\n"
                return
            pub = d_id_service.resolve_public_image_url(
                d_id_service.resolve_source_url(avatar_source_url.strip())
            )
            try:
                vres = await d_id_vision_screen.screen_image_from_url(pub, vision_endpoint_model_id=mid)
            except Exception as e:
                logger.exception("vision screen")
                yield json.dumps({"event": "error", "message": f"Vision screening failed: {e}"}) + "\n"
                return
            yield json.dumps({"event": "vision_screen", "result": vres}) + "\n"
            if not vres.get("pass"):
                yield json.dumps(
                    {
                        "event": "error",
                        "message": "Avatar failed vision screening; batch aborted.",
                        "vision": vres,
                    }
                ) + "\n"
                return

        pairs: List[tuple] = []
        for uf in segment_files:
            raw = await uf.read()
            if raw:
                pairs.append((uf.filename or "segment.wav", raw))
        tmpdir = d_id_batch.write_wav_uploads_to_temp_dir(pairs)
        wav_paths = list(tmpdir.iterdir())
        async for line in d_id_batch.run_batch_ndjson(
            wav_paths,
            avatar_source_url.strip(),
            concurrency=concurrency,
            emotion=emotion,
            movement=movement,
            background_url=background_url,
        ):
            yield line
    finally:
        if tmpdir and tmpdir.is_dir():
            shutil.rmtree(tmpdir, ignore_errors=True)


@d_id_router.post("/d-id/batch-run")
async def d_id_batch_run(
    segments: List[UploadFile] = File(..., description="Ordered WAV segments"),
    avatar_source_url: str = Form(..., description="Public https avatar URL (or /static/... with D_ID_PUBLIC_BASE_URL)"),
    emotion: str = Form("neutral"),
    movement: str = Form("active"),
    background_url: Optional[str] = Form(None),
    concurrency: int = Form(2),
    require_vision: bool = Form(True),
):
    """
    NDJSON stream: vision_screen (optional) → started → segment_done × N → concat_started → complete | error.
    """
    if not segments:
        raise HTTPException(status_code=400, detail="At least one WAV segment is required.")
    return StreamingResponse(
        _batch_ndjson_stream(
            segment_files=segments,
            avatar_source_url=avatar_source_url,
            emotion=emotion.strip() or None,
            movement=movement.strip() or None,
            background_url=(background_url or "").strip() or None,
            concurrency=max(1, min(int(concurrency), 4)),
            require_vision=bool(require_vision),
        ),
        media_type="application/x-ndjson",
    )


class VisionUrlBody(BaseModel):
    image_url: str
    vision_model: Optional[str] = None


@d_id_router.post("/d-id/vision-screen-url")
async def d_id_vision_screen_url(body: VisionUrlBody):
    """Vision screening when the image is already at a public URL (server-side fetch)."""
    mid = (body.vision_model or os.environ.get("D_ID_VISION_SCREEN_MODEL") or "").strip()
    if not mid:
        raise HTTPException(
            status_code=400,
            detail="Pass vision_model or set D_ID_VISION_SCREEN_MODEL.",
        )
    if not body.image_url.strip():
        raise HTTPException(status_code=400, detail="image_url is required")
    try:
        pub = d_id_service.resolve_public_image_url(body.image_url.strip())
        result = await d_id_vision_screen.screen_image_from_url(pub, vision_endpoint_model_id=mid)
        return {"status": "ok", "result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("vision-screen-url")
        raise HTTPException(status_code=500, detail=str(e))


@d_id_router.post("/d-id/vision-screen")
async def d_id_vision_screen_endpoint(
    image: UploadFile = File(...),
    vision_model: Optional[str] = Form(
        None,
        description="Override endpoint id; defaults to env D_ID_VISION_SCREEN_MODEL",
    ),
):
    """Manual vision check on an uploaded image."""
    mid = (vision_model or os.environ.get("D_ID_VISION_SCREEN_MODEL") or "").strip()
    if not mid:
        raise HTTPException(
            status_code=400,
            detail="Pass vision_model form field or set D_ID_VISION_SCREEN_MODEL.",
        )
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image.")
    ct = image.content_type or "image/png"
    try:
        result = await d_id_vision_screen.screen_image_bytes(data, vision_endpoint_model_id=mid, content_type=ct)
        return {"status": "ok", "result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("vision-screen")
        raise HTTPException(status_code=500, detail=str(e))


@d_id_router.get("/d-id/saved-assets")
async def d_id_list_saved(kind: Optional[str] = None):
    if kind not in (None, "", "avatar", "background"):
        raise HTTPException(status_code=400, detail="kind must be avatar or background")
    items = d_id_assets.list_assets(kind if kind else None)  # type: ignore[arg-type]
    return {"status": "ok", "items": items}


class SavedAssetBody(BaseModel):
    kind: str
    label: str = ""
    url: str


@d_id_router.post("/d-id/saved-assets")
async def d_id_add_saved(body: SavedAssetBody):
    if body.kind not in ("avatar", "background"):
        raise HTTPException(status_code=400, detail="kind must be avatar or background")
    if not body.url.strip():
        raise HTTPException(status_code=400, detail="url is required")
    entry = d_id_assets.add_asset(kind=body.kind, label=body.label, url=body.url)  # type: ignore[arg-type]
    return {"status": "ok", "item": entry}


@d_id_router.delete("/d-id/saved-assets/{asset_id}")
async def d_id_delete_saved(asset_id: str):
    if d_id_assets.delete_asset(asset_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Not found")
