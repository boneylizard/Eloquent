"""PyInstaller entry point for Mirid's managed desktop services."""

import argparse
import os
import multiprocessing
import traceback

from backend.app.compute_capabilities import disable_incompatible_torchao


disable_incompatible_torchao()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("backend", "tts", "probe-image-runtime"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    os.environ.setdefault("TORCH_DYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    if os.environ.get("MIRID_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("RAG_EMBEDDING_DEVICE", "cpu")
    os.environ.setdefault("GPU_ID", "0")

    if args.mode == "probe-image-runtime":
        from backend.app.windows_dll_paths import stable_diffusion_import_environment

        with stable_diffusion_import_environment() as dependency_directories:
            import stable_diffusion_cpp

        print(
            "Mirid image runtime ready:",
            stable_diffusion_cpp.__version__,
            ",".join(str(path) for path in dependency_directories),
        )
        return

    import uvicorn

    # Import directly so frozen-build dependency failures are not hidden by
    # Transformers' lazy module wrapper.
    import transformers.trainer  # noqa: F401

    if args.mode == "backend":
        port = args.port or 8000
        os.environ["PORT"] = str(port)
        os.environ.setdefault("TTS_PORT", "8002")
        application = "backend.app.main:app"
    else:
        port = args.port or 8002
        os.environ["TTS_PORT"] = str(port)
        application = "backend.app.tts_backend:app"

    uvicorn.run(
        application,
        host=args.host,
        port=port,
        log_level="info",
        access_log=False,
        ws_ping_interval=300,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
