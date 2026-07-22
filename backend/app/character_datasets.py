import json
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Body, HTTPException


router = APIRouter(prefix="/character-datasets", tags=["Character Datasets"])
DATASETS_SERVER = "https://datasets-server.huggingface.co"


def _huggingface_token() -> str | None:
    settings_path = Path.home() / ".LiangLocal" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        return str(settings.get("huggingFaceToken") or "").strip() or None
    except (OSError, ValueError, TypeError):
        return None


def _validate_repo_id(repo_id: str) -> str:
    value = str(repo_id or "").strip().strip("/")
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", value):
        raise HTTPException(status_code=422, detail="Use a Hugging Face dataset name such as owner/dataset.")
    return value


@router.post("/huggingface/preview")
async def preview_huggingface_dataset(payload: dict[str, Any] = Body(...)):
    repo_id = _validate_repo_id(payload.get("repo_id", ""))
    requested_config = str(payload.get("config") or "").strip()
    requested_split = str(payload.get("split") or "").strip()
    headers = {}
    token = _huggingface_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
            splits_response = await client.get(f"{DATASETS_SERVER}/splits", params={"dataset": repo_id})
            splits_response.raise_for_status()
            available = splits_response.json().get("splits") or []
            if not available:
                raise HTTPException(status_code=422, detail="Hugging Face could not find a readable tabular split in that dataset.")

            selected = next((item for item in available if (
                (not requested_config or item.get("config") == requested_config)
                and (not requested_split or item.get("split") == requested_split)
            )), available[0])
            config = str(selected.get("config") or "default")
            split = str(selected.get("split") or "train")
            rows_response = await client.get(
                f"{DATASETS_SERVER}/first-rows",
                params={"dataset": repo_id, "config": config, "split": split},
            )
            rows_response.raise_for_status()
            data = rows_response.json()
    except HTTPException:
        raise
    except httpx.HTTPStatusError as error:
        detail = "Hugging Face could not preview that dataset. Check its name, access and Dataset Viewer status."
        raise HTTPException(status_code=502, detail=detail) from error
    except (httpx.RequestError, ValueError) as error:
        raise HTTPException(status_code=502, detail="Mirid could not reach the Hugging Face dataset service.") from error

    rows = [item.get("row") for item in data.get("rows") or [] if isinstance(item.get("row"), dict)]
    columns = sorted({str(key) for row in rows for key in row.keys()})
    return {
        "repo_id": repo_id,
        "config": config,
        "split": split,
        "available_splits": available,
        "columns": columns,
        "rows": rows[:100],
        "preview_only": True,
    }
