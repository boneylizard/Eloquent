import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Body, HTTPException
import httpx


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model-library", tags=["Model Library"])

DOWNLOADABLE_EXTENSIONS = {
    ".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".bin", ".index", ".zip"
}
QUANT_PATTERN = re.compile(
    r"(?i)(?:^|[-_.])((?:iq\d|q\d|f16|f32|bf16)[a-z0-9_-]*)(?:[-_.]|$)"
)
SHARD_PATTERN = re.compile(r"^(?P<prefix>.+)-\d{5}-of-(?P<count>\d{5})(?P<suffix>\.[^.]+)$", re.IGNORECASE)
DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
DOWNLOAD_JOBS_LOCK = threading.Lock()
RECOMMENDATIONS_CACHE: dict[str, Any] = {"updated_at": 0.0, "models": []}
RECOMMENDATIONS_CACHE_LOCK = threading.Lock()
RECOMMENDATIONS_CACHE_SECONDS = 6 * 60 * 60
NANOGPT_SUBSCRIPTION_MODELS_URL = "https://nano-gpt.com/api/subscription/v1/models?detailed=true"
CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"
CIVITAI_IMAGE_EXTENSIONS = {".safetensors", ".gguf"}
MIRID_PICKS = (
    {
        "repo_id": "unsloth/Qwen3.5-4B-GGUF",
        "quantisation": "Q4_K_M",
        "minimum_vram_gb": 4,
        "comfortable_vram_gb": 6,
        "title": "Qwen 3.5 4B",
        "reason": "A compact starting point for everyday chat, roleplay and instruction following.",
    },
    {
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "quantisation": "Q4_K_M",
        "minimum_vram_gb": 8,
        "comfortable_vram_gb": 10,
        "title": "Qwen 3.5 9B",
        "reason": "A useful middle ground when you want stronger writing and reasoning without a very large download.",
    },
    {
        "repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "quantisation": "Q4_K_M",
        "minimum_vram_gb": 20,
        "comfortable_vram_gb": 24,
        "title": "Qwen 3.6 27B",
        "reason": "The higher-capacity pick for 24 GB-class GPUs, with partial offload available when context or other services claim VRAM.",
    },
)


def _settings() -> dict[str, Any]:
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    try:
        if settings_path.exists():
            return json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("Could not read model-library settings: %s", error)
    return {}


def normalize_nanogpt_subscription_models(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("data") or payload.get("models") or []
        if isinstance(rows, dict):
            rows = list(rows.values())
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict) and (row.get("id") or row.get("model"))]


@router.post("/nanogpt/subscription-models")
async def list_nanogpt_subscription_models(payload: dict = Body(...)):
    api_key = str(payload.get("api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Add your NanoGPT API key first.")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(NANOGPT_SUBSCRIPTION_MODELS_URL, headers=headers)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="NanoGPT's subscription catalogue could not be reached.") from error
    if response.status_code in {401, 403}:
        raise HTTPException(status_code=401, detail="NanoGPT rejected that API key or it has no subscription access.")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"NanoGPT returned HTTP {response.status_code}.")
    try:
        models = normalize_nanogpt_subscription_models(response.json())
    except ValueError as error:
        raise HTTPException(status_code=502, detail="NanoGPT returned an unreadable subscription catalogue.") from error
    return {"models": models, "count": len(models)}


def model_destinations(create: bool = True) -> dict[str, dict[str, Any]]:
    settings = _settings()
    defaults = {
        "text": Path.home() / "models" / "gguf",
        "image": Path.home() / "models" / "stable-diffusion",
        "adetailer": Path.home() / "models" / "adetailer",
        "upscaler": Path.home() / "models" / "upscalers",
        "speech": Path.home() / "models" / "speech",
    }
    setting_keys = {
        "text": "modelDirectory",
        "image": "sdModelDirectory",
        "adetailer": "adetailerModelDirectory",
        "upscaler": "upscalerModelDirectory",
        "speech": "speechModelDirectory",
    }
    labels = {
        "text": "Text / GGUF",
        "image": "Image generation",
        "adetailer": "ADetailer",
        "upscaler": "Upscalers",
        "speech": "Speech",
    }
    result = {}
    for destination_type, default_path in defaults.items():
        configured = str(settings.get(setting_keys[destination_type]) or "").strip()
        path = Path(configured).expanduser() if configured else default_path
        if create:
            path.mkdir(parents=True, exist_ok=True)
        result[destination_type] = {
            "type": destination_type,
            "label": labels[destination_type],
            "path": str(path),
            "setting_key": setting_keys[destination_type],
            "custom": bool(configured),
        }
    return result


def _civitai_params(**values) -> dict[str, Any]:
    params = {key: value for key, value in values.items() if value not in (None, "")}
    token = str(_settings().get("civitaiApiKey") or "").strip()
    if token:
        params["token"] = token
    return params


def _civitai_error(status_code: int) -> str:
    if status_code in {403, 451}:
        return (
            "Civitai is not available from this connection. Mirid cannot change regional "
            "availability; use Hugging Face instead."
        )
    if status_code == 401:
        return "Civitai rejected that API key. Check it in the Model Library."
    if status_code == 404:
        return "That Civitai model or version no longer exists."
    return f"Civitai returned HTTP {status_code}."


def _civitai_file(file_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    filename = Path(str(file_data.get("name") or "")).name
    suffix = Path(filename).suffix.lower()
    virus_scan = str(file_data.get("virusScanResult") or "").strip()
    pickle_scan = str(file_data.get("pickleScanResult") or "").strip()
    if suffix not in CIVITAI_IMAGE_EXTENSIONS:
        return None
    if virus_scan.lower() != "success" or pickle_scan.lower() in {"danger", "error", "pending"}:
        return None
    try:
        size = int(float(file_data.get("sizeKB") or 0) * 1024) or None
    except (TypeError, ValueError):
        size = None
    hashes = file_data.get("hashes") if isinstance(file_data.get("hashes"), dict) else {}
    metadata = file_data.get("metadata") if isinstance(file_data.get("metadata"), dict) else {}
    return {
        "id": int(file_data.get("id") or 0),
        "filename": filename,
        "size": size,
        "format": metadata.get("format") or suffix.lstrip(".").upper(),
        "precision": metadata.get("fp"),
        "primary": bool(file_data.get("primary")),
        "virus_scan": virus_scan,
        "pickle_scan": pickle_scan,
        "sha256": hashes.get("SHA256"),
    }


def normalise_civitai_model(model: dict[str, Any]) -> Optional[dict[str, Any]]:
    versions = []
    for version in model.get("modelVersions") or []:
        if not isinstance(version, dict):
            continue
        files = [
            serialised
            for file_data in version.get("files") or []
            if isinstance(file_data, dict)
            for serialised in [_civitai_file(file_data)]
            if serialised
        ]
        primary = next((file_data for file_data in files if file_data["primary"]), None)
        if not primary:
            continue
        versions.append({
            "id": int(version.get("id") or 0),
            "name": str(version.get("name") or "Unnamed version"),
            "base_model": version.get("baseModel"),
            "created_at": version.get("createdAt"),
            "trained_words": list(version.get("trainedWords") or [])[:12],
            "file": primary,
        })
    if not versions:
        return None
    stats = model.get("stats") if isinstance(model.get("stats"), dict) else {}
    creator = model.get("creator") if isinstance(model.get("creator"), dict) else {}
    return {
        "id": int(model.get("id") or 0),
        "name": str(model.get("name") or "Unnamed model"),
        "url": f"https://civitai.com/models/{int(model.get('id') or 0)}",
        "type": model.get("type"),
        "creator": creator.get("username"),
        "nsfw": model.get("nsfw", False),
        "downloads": int(stats.get("downloadCount") or 0),
        "rating": stats.get("rating"),
        "versions": versions,
    }


def parse_huggingface_reference(reference: str) -> tuple[str, str, Optional[str]]:
    raw = str(reference or "").strip()
    if not raw:
        raise ValueError("Enter a Hugging Face repository or URL.")

    revision = "main"
    quant = None
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("That URL is not on huggingface.co.")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("The Hugging Face URL must include an owner and repository.")
        repo_id = "/".join(parts[:2])
        if len(parts) >= 4 and parts[2] in {"tree", "resolve", "blob"}:
            revision = parts[3]
    else:
        if ":" in raw:
            raw, quant = raw.rsplit(":", 1)
            quant = quant.strip() or None
        repo_id = raw.strip().strip("/")

    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
        raise ValueError("Use owner/repository, optionally followed by :quantisation.")
    return repo_id, revision, quant


def _file_size(sibling) -> Optional[int]:
    size = getattr(sibling, "size", None)
    if isinstance(size, int):
        return size
    lfs = getattr(sibling, "lfs", None)
    lfs_size = getattr(lfs, "size", None) if lfs else None
    return lfs_size if isinstance(lfs_size, int) else None


def _suggest_destination(filename: str) -> str:
    lower = filename.lower()
    suffix = Path(lower).suffix
    if any(word in lower for word in ("adetailer", "yolo", "face_yolov", "hand_yolov")):
        return "adetailer"
    if any(word in lower for word in ("upscaler", "realesrgan", "ultrasharp", "remacri", "swinir")):
        return "upscaler"
    if any(word in lower for word in ("stable-diffusion", "sdxl", "sd3", "flux", "z-image", "wan2")):
        return "image"
    if suffix == ".gguf":
        return "text"
    if suffix in {".safetensors", ".ckpt"}:
        return "image"
    return "speech" if suffix in {".index", ".zip"} else "text"


def companion_shards(filename: str, available_filenames: list[str]) -> list[str]:
    match = SHARD_PATTERN.match(filename)
    if not match:
        return [filename]
    prefix = match.group("prefix")
    count = match.group("count")
    suffix = match.group("suffix")
    expected = re.compile(
        rf"^{re.escape(prefix)}-\d{{5}}-of-{re.escape(count)}{re.escape(suffix)}$",
        re.IGNORECASE,
    )
    companions = sorted(name for name in available_filenames if expected.match(name))
    return companions or [filename]


def _serialise_model(model) -> dict[str, Any]:
    last_modified = getattr(model, "last_modified", None)
    return {
        "id": model.id,
        "author": getattr(model, "author", None),
        "downloads": int(getattr(model, "downloads", 0) or 0),
        "likes": int(getattr(model, "likes", 0) or 0),
        "pipeline_tag": getattr(model, "pipeline_tag", None),
        "tags": list(getattr(model, "tags", []) or [])[:24],
        "gated": bool(getattr(model, "gated", False)),
        "private": bool(getattr(model, "private", False)),
        "last_modified": last_modified.isoformat() if last_modified else None,
    }


def _recommended_file(info, quantisation: str) -> Optional[dict[str, Any]]:
    siblings = list(info.siblings or [])
    available_names = [sibling.rfilename for sibling in siblings]
    candidates = [
        sibling for sibling in siblings
        if sibling.rfilename.lower().endswith(".gguf")
        and quantisation.lower() in sibling.rfilename.lower()
        and "mmproj" not in sibling.rfilename.lower()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda sibling: ("-00001-of-" not in sibling.rfilename.lower(), sibling.rfilename.lower()))
    selected = candidates[0]
    companions = companion_shards(selected.rfilename, available_names)
    sizes = {sibling.rfilename: _file_size(sibling) for sibling in siblings}
    total_size = sum(sizes.get(filename) or 0 for filename in companions) or None
    return {
        "filename": selected.rfilename,
        "filenames": companions,
        "size": total_size,
        "quantisation": quantisation,
    }


def _load_recommendations() -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi(token=_settings().get("huggingFaceToken") or None)
    models = []
    for pick in MIRID_PICKS:
        try:
            info = api.model_info(pick["repo_id"], files_metadata=True)
            selected_file = _recommended_file(info, pick["quantisation"])
            if not selected_file:
                logger.warning("Mirid pick has no matching GGUF: %s", pick["repo_id"])
                continue
            models.append({
                **pick,
                **selected_file,
                "downloads": int(getattr(info, "downloads", 0) or 0),
                "likes": int(getattr(info, "likes", 0) or 0),
                "last_modified": (
                    info.last_modified.isoformat() if getattr(info, "last_modified", None) else None
                ),
                "reference": f'{pick["repo_id"]}:{pick["quantisation"]}',
            })
        except Exception as error:
            logger.warning("Could not refresh Mirid pick %s: %s", pick["repo_id"], error)
    return models


@router.get("/recommendations")
async def get_recommendations():
    now = time.time()
    with RECOMMENDATIONS_CACHE_LOCK:
        cached_models = list(RECOMMENDATIONS_CACHE["models"])
        cache_age = now - float(RECOMMENDATIONS_CACHE["updated_at"] or 0)
    if cached_models and cache_age < RECOMMENDATIONS_CACHE_SECONDS:
        return {"models": cached_models, "cached": True}

    models = await asyncio.to_thread(_load_recommendations)
    if models:
        with RECOMMENDATIONS_CACHE_LOCK:
            RECOMMENDATIONS_CACHE.update({"updated_at": time.time(), "models": models})
        return {"models": models, "cached": False}
    if cached_models:
        return {"models": cached_models, "cached": True, "stale": True}
    raise HTTPException(status_code=502, detail="Mirid could not refresh its model picks from Hugging Face.")


def _clean_model_card(markdown: str) -> str:
    text = re.sub(r"\A---\s.*?\s---\s*", "", markdown, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*[#>*-]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) < 80 or paragraph.count("|") >= 3:
            continue
        paragraphs.append(paragraph)
        if sum(len(item) for item in paragraphs) >= 12000:
            break
    return "\n\n".join(paragraphs)[:12000]


@router.get("/huggingface/model-card")
async def get_model_card(repo_id: str, revision: str = "main"):
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
        raise HTTPException(status_code=422, detail="Invalid Hugging Face repository.")
    try:
        from huggingface_hub import hf_hub_download

        path = await asyncio.to_thread(
            hf_hub_download,
            repo_id=repo_id,
            filename="README.md",
            revision=revision,
            token=_settings().get("huggingFaceToken") or None,
        )
        excerpt = _clean_model_card(Path(path).read_text(encoding="utf-8", errors="replace"))
        if not excerpt:
            raise HTTPException(status_code=404, detail="This repository has no readable model card text.")
        return {"repo_id": repo_id, "excerpt": excerpt}
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Could not read Hugging Face model card")
        raise HTTPException(status_code=502, detail=f"Could not read that model card: {error}") from error


@router.get("/destinations")
async def get_destinations():
    return {"destinations": list(model_destinations().values())}


@router.get("/huggingface/search")
async def search_huggingface(q: str, limit: int = 20, kind: Optional[str] = None):
    query = q.strip()
    if len(query) < 2:
        raise HTTPException(status_code=422, detail="Enter at least two characters.")
    try:
        from huggingface_hub import HfApi

        filters = "text-to-image" if kind == "image" else None
        models = await asyncio.to_thread(
            lambda: list(HfApi().list_models(
                search=query,
                filter=filters,
                sort="downloads",
                direction=-1,
                limit=max(1, min(limit, 40)),
                token=_settings().get("huggingFaceToken") or None,
            ))
        )
        return {"models": [_serialise_model(model) for model in models]}
    except Exception as error:
        logger.exception("Hugging Face search failed")
        raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {error}") from error


@router.get("/civitai/search")
async def search_civitai(q: str, limit: int = 20):
    query = q.strip()
    if len(query) < 2:
        raise HTTPException(status_code=422, detail="Enter at least two characters.")
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                f"{CIVITAI_API_BASE}/models",
                params=_civitai_params(
                    query=query,
                    limit=max(1, min(limit, 40)),
                    types="Checkpoint",
                    sort="Most Downloaded",
                    period="AllTime",
                    primaryFileOnly="true",
                ),
                headers={"Accept": "application/json", "User-Agent": "Mirid/1.0"},
            )
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail="Civitai could not be reached from this connection. Use Hugging Face instead.",
        ) from error
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=_civitai_error(response.status_code))
    try:
        payload = response.json()
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Civitai returned an unreadable model catalogue.") from error
    models = [
        serialised
        for model in payload.get("items") or []
        if isinstance(model, dict)
        for serialised in [normalise_civitai_model(model)]
        if serialised
    ]
    return {"models": models, "count": len(models)}


@router.get("/civitai/models/{model_id}")
async def inspect_civitai_model(model_id: int):
    if model_id <= 0:
        raise HTTPException(status_code=422, detail="Choose a valid Civitai model.")
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                f"{CIVITAI_API_BASE}/models/{model_id}",
                params=_civitai_params(),
                headers={"Accept": "application/json", "User-Agent": "Mirid/1.0"},
            )
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail="Civitai could not be reached from this connection. Use Hugging Face instead.",
        ) from error
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=_civitai_error(response.status_code))
    try:
        model = normalise_civitai_model(response.json())
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Civitai returned unreadable model details.") from error
    if not model:
        raise HTTPException(
            status_code=422,
            detail="This model has no scan-passed Safetensors or GGUF checkpoint that Mirid can install safely.",
        )
    return {"model": model}


@router.post("/huggingface/inspect")
async def inspect_huggingface(payload: dict = Body(...)):
    try:
        from huggingface_hub import HfApi

        repo_id, revision, quant = parse_huggingface_reference(payload.get("reference", ""))
        info = await asyncio.to_thread(
            HfApi().model_info,
            repo_id,
            revision=revision,
            files_metadata=True,
            token=_settings().get("huggingFaceToken") or None,
        )
        downloadable_names = [
            sibling.rfilename
            for sibling in info.siblings or []
            if Path(sibling.rfilename).suffix.lower() in DOWNLOADABLE_EXTENSIONS
        ]
        files = []
        for sibling in info.siblings or []:
            filename = sibling.rfilename
            if Path(filename).suffix.lower() not in DOWNLOADABLE_EXTENSIONS:
                continue
            quant_match = not quant or quant.lower() in filename.lower()
            quant_match_name = QUANT_PATTERN.search(Path(filename).name)
            files.append({
                "filename": filename,
                "size": _file_size(sibling),
                "quantisation": quant_match_name.group(1) if quant_match_name else None,
                "quant_match": quant_match,
                "suggested_destination": _suggest_destination(filename),
                "companion_files": companion_shards(filename, downloadable_names),
                "role": "vision_companion" if "mmproj" in filename.lower() else "model",
            })
        files.sort(key=lambda item: (
            not item["quant_match"],
            item["role"] != "model",
            item["filename"].lower(),
        ))
        return {
            "repository": _serialise_model(info),
            "revision": revision,
            "requested_quantisation": quant,
            "files": files,
            "destinations": list(model_destinations().values()),
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Hugging Face repository inspection failed")
        raise HTTPException(status_code=502, detail=f"Could not inspect that repository: {error}") from error


def _set_job(job_id: str, **patch):
    with DOWNLOAD_JOBS_LOCK:
        current = DOWNLOAD_JOBS.get(job_id, {})
        DOWNLOAD_JOBS[job_id] = {**current, **patch, "updated_at": time.time()}


def _run_download(job_id: str, repo_id: str, filenames: list[str], revision: str, destination_type: str):
    try:
        from huggingface_hub import hf_hub_download

        destinations = model_destinations()
        destination = destinations[destination_type]
        installed_paths = []
        for index, filename in enumerate(filenames, start=1):
            _set_job(
                job_id,
                status="downloading",
                message=f"Downloading {Path(filename).name} ({index} of {len(filenames)})…",
                completed_files=index - 1,
                total_files=len(filenames),
            )
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=_settings().get("huggingFaceToken") or None,
                local_dir=destination["path"],
            )
            target = Path(destination["path"]) / Path(filename).name
            downloaded = Path(downloaded_path)
            if downloaded.resolve() != target.resolve():
                os.replace(downloaded, target)
                parent = downloaded.parent
                destination_root = Path(destination["path"]).resolve()
                while parent.resolve() != destination_root:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            installed_paths.append(str(target))
        _set_job(
            job_id,
            status="complete",
            message=f"Installed {len(installed_paths)} model file{'s' if len(installed_paths) != 1 else ''}.",
            path=installed_paths[0],
            paths=installed_paths,
            filename=Path(installed_paths[0]).name,
            completed_files=len(installed_paths),
            total_files=len(installed_paths),
        )
    except Exception as error:
        logger.exception("Hugging Face model download failed")
        _set_job(job_id, status="failed", message="Download failed.", error=str(error))


def _format_download_rate(bytes_per_second: float) -> str:
    if bytes_per_second >= 1024 ** 2:
        return f"{bytes_per_second / (1024 ** 2):.1f} MB/s"
    if bytes_per_second >= 1024:
        return f"{bytes_per_second / 1024:.0f} KB/s"
    return f"{bytes_per_second:.0f} B/s"


def _run_civitai_download(job_id: str, version_id: int, file_id: int):
    destination = model_destinations()["image"]
    partial_path: Optional[Path] = None
    try:
        token = str(_settings().get("civitaiApiKey") or "").strip()
        params = {"token": token} if token else {}
        headers = {"Accept": "application/json", "User-Agent": "Mirid/1.0"}
        with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
            details = client.get(f"{CIVITAI_API_BASE}/model-versions/{version_id}", params=params)
            if details.status_code >= 400:
                raise RuntimeError(_civitai_error(details.status_code))
            version = details.json()
            selected = next(
                (
                    serialised
                    for file_data in version.get("files") or []
                    if isinstance(file_data, dict) and int(file_data.get("id") or 0) == file_id and file_data.get("primary")
                    for serialised in [_civitai_file(file_data)]
                    if serialised
                ),
                None,
            )
            if not selected:
                raise RuntimeError(
                    "That file is no longer the scan-passed primary Safetensors or GGUF checkpoint for this version."
                )

            filename = Path(selected["filename"]).name
            target = Path(destination["path"]) / filename
            partial_path = target.with_suffix(f"{target.suffix}.part")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_bytes = 0
            started_at = time.monotonic()
            last_update = 0.0
            sha256 = hashlib.sha256()

            with client.stream(
                "GET",
                f"{CIVITAI_DOWNLOAD_BASE}/{version_id}",
                params=params,
                headers={"Accept": "application/octet-stream", "User-Agent": "Mirid/1.0"},
            ) as response:
                if response.status_code >= 400:
                    raise RuntimeError(_civitai_error(response.status_code))
                total_bytes = int(response.headers.get("content-length") or selected.get("size") or 0)
                with partial_path.open("wb") as output:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        sha256.update(chunk)
                        downloaded_bytes += len(chunk)
                        now = time.monotonic()
                        if now - last_update >= 0.5:
                            elapsed = max(now - started_at, 0.001)
                            rate = downloaded_bytes / elapsed
                            percent = round((downloaded_bytes / total_bytes) * 100, 1) if total_bytes else None
                            message = f"Downloading {filename} at {_format_download_rate(rate)}"
                            if percent is not None:
                                message = f"Downloading {filename}: {percent:.1f}% at {_format_download_rate(rate)}"
                            _set_job(
                                job_id,
                                status="downloading",
                                message=message,
                                downloaded_bytes=downloaded_bytes,
                                total_bytes=total_bytes or None,
                                progress=percent,
                                bytes_per_second=rate,
                            )
                            last_update = now

            expected_hash = str(selected.get("sha256") or "").strip().lower()
            actual_hash = sha256.hexdigest().lower()
            if expected_hash and actual_hash != expected_hash:
                raise RuntimeError("The downloaded file failed its SHA-256 check and was not installed.")
            os.replace(partial_path, target)
            partial_path = None
            _set_job(
                job_id,
                status="complete",
                message="Image model installed and verified.",
                path=str(target),
                paths=[str(target)],
                filename=filename,
                downloaded_bytes=downloaded_bytes,
                total_bytes=downloaded_bytes,
                progress=100,
                sha256=actual_hash,
            )
    except Exception as error:
        if partial_path and partial_path.exists():
            try:
                partial_path.unlink()
            except OSError:
                pass
        logger.exception("Civitai model download failed")
        _set_job(job_id, status="failed", message="Download failed.", error=str(error))


@router.post("/huggingface/download")
async def download_huggingface_file(payload: dict = Body(...)):
    repo_id = str(payload.get("repo_id") or "").strip()
    filenames = payload.get("filenames")
    if not isinstance(filenames, list):
        filenames = [payload.get("filename")]
    filenames = [str(filename or "").strip() for filename in filenames]
    filenames = list(dict.fromkeys(filename for filename in filenames if filename))
    revision = str(payload.get("revision") or "main").strip()
    destination_type = str(payload.get("destination_type") or "text").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
        raise HTTPException(status_code=422, detail="Invalid Hugging Face repository.")
    if not filenames or len(filenames) > 64 or any(
        Path(filename).suffix.lower() not in DOWNLOADABLE_EXTENSIONS for filename in filenames
    ):
        raise HTTPException(status_code=422, detail="Choose a supported model file.")
    if destination_type not in model_destinations():
        raise HTTPException(status_code=422, detail="Choose a valid model destination.")

    job_id = uuid.uuid4().hex
    _set_job(
        job_id,
        id=job_id,
        status="queued",
        message="Download queued.",
        repo_id=repo_id,
        filename=filenames[0],
        filenames=filenames,
        destination_type=destination_type,
        created_at=time.time(),
        error=None,
    )
    threading.Thread(
        target=_run_download,
        args=(job_id, repo_id, filenames, revision, destination_type),
        daemon=True,
        name=f"mirid-hf-{job_id[:8]}",
    ).start()
    return DOWNLOAD_JOBS[job_id]


@router.post("/civitai/download")
async def download_civitai_file(payload: dict = Body(...)):
    try:
        version_id = int(payload.get("version_id") or 0)
        file_id = int(payload.get("file_id") or 0)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Choose a valid Civitai model version.") from error
    if version_id <= 0 or file_id <= 0:
        raise HTTPException(status_code=422, detail="Choose a valid Civitai model version.")

    job_id = uuid.uuid4().hex
    _set_job(
        job_id,
        id=job_id,
        status="queued",
        message="Download queued.",
        provider="civitai",
        version_id=version_id,
        file_id=file_id,
        destination_type="image",
        created_at=time.time(),
        error=None,
    )
    threading.Thread(
        target=_run_civitai_download,
        args=(job_id, version_id, file_id),
        daemon=True,
        name=f"mirid-civitai-{job_id[:8]}",
    ).start()
    return DOWNLOAD_JOBS[job_id]


@router.get("/downloads/{job_id}")
async def get_download_job(job_id: str):
    with DOWNLOAD_JOBS_LOCK:
        job = DOWNLOAD_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Download job not found.")
    return job
