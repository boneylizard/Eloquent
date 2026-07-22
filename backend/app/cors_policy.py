import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


LOCAL_ORIGIN_REGEX = (
    r"^(?:https?://(?:localhost|127\.0\.0\.1|tauri\.localhost)(?::\d+)?|tauri://localhost)$"
)


def cors_options() -> dict:
    configured = [
        origin.strip()
        for origin in os.environ.get("MIRID_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if "*" in configured:
        return {
            "allow_origins": ["*"],
            "allow_origin_regex": None,
            "allow_credentials": False,
        }
    return {
        "allow_origins": configured,
        "allow_origin_regex": LOCAL_ORIGIN_REGEX,
        "allow_credentials": True,
    }


def configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        **cors_options(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
