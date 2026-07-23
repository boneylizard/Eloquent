import platform
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata


datas = []
binaries = []
hiddenimports = [
    "backend.app.main",
    "backend.app.tts_backend",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "transformers.trainer",
    "transformers.trainer_callback",
    "transformers.trainer_utils",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
]


def collect_optional(package, include_sources=False):
    global datas, binaries, hiddenimports
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
        if include_sources:
            datas += collect_data_files(package, include_py_files=True)
    except Exception:
        pass


def collect_required(package, include_sources=False):
    global datas, binaries, hiddenimports
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
    if include_sources:
        datas += collect_data_files(package, include_py_files=True)


def copy_optional_metadata(distribution):
    global datas
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass


for distribution in ("torchcodec", "pymatting"):
    copy_optional_metadata(distribution)

for package in ("chatterbox", "voxcpm", "kokoro"):
    collect_optional(package, include_sources=True)

collect_required("espeakng_loader", include_sources=True)
collect_required("en_core_web_sm", include_sources=True)
collect_required("misaki", include_sources=True)
datas += copy_metadata("en_core_web_sm")

for package in (
    "language_tags",
    "tiktoken_ext",
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "llama_cpp",
    "transformers",
    "sentence_transformers",
    "tiktoken",
    "triton",
):
    collect_optional(package)

if sys.platform == "win32":
    collect_required("stable_diffusion_cpp")
else:
    collect_optional("stable_diffusion_cpp")

machine = platform.machine().lower()
profile = os.environ.get("MIRID_SIDECAR_PROFILE", "default").strip().lower()
if profile not in {"default", "full"}:
    raise RuntimeError(f"Unsupported Mirid sidecar profile: {profile}")
runtime_hook = "backend/full_runtime_profile.py" if profile == "full" else "backend/default_runtime_profile.py"
election_excludes = [] if profile == "full" else [
    "backend.app.election_forecast",
    "backend.app.election_db",
    "backend.app.election_data_service",
    "backend.app.votehub_service",
    "backend.app.rcp_service",
    "backend.app.election_ai_service",
    "backend.app.election_simulation",
    "backend.app.ballotpedia_scraper",
]
if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
    target_triple = "aarch64-apple-darwin"
elif sys.platform == "linux" and machine in {"x86_64", "amd64"}:
    target_triple = "x86_64-unknown-linux-gnu"
elif sys.platform == "win32" and machine in {"amd64", "x86_64"}:
    target_triple = "x86_64-pc-windows-msvc"
else:
    raise RuntimeError(f"Unsupported Mirid sidecar target: {sys.platform} {machine}")

a = Analysis(
    ["sidecar_entry.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[runtime_hook],
    excludes=election_excludes + [
        "backend.app.model_server",
        "backend.app.model_subprocess",
        "backend.app.tests",
        "backend.app.forensic_linguistics_service",
        "backend.app.market_sim",
        "backend.app.chatlog_condenser",
        "backend.app.chatlog_condenser_orchestrator",
        "backend.app.chatlog_condenser_prompt",
        "backend.app.chatlog_condenser_rag",
        "backend.app.chatlog_condenser_routes",
        "backend.app.chatlog_condenser_session",
        "backend.app.auth_routes",
        "backend.app.chess_ai_service",
        "backend.app.chess_auth_db",
        "backend.app.chess_engine",
        "backend.app.chess_historian",
        "backend.app.chess_oauth",
        "backend.app.chess_research_agent",
        "backend.app.download_book",
        "chess",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    name=f"mirid-sidecar-{target_triple}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=sys.platform == "win32",
    console=False,
    exclude_binaries=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=sys.platform == "win32",
    name="mirid-sidecar",
)
