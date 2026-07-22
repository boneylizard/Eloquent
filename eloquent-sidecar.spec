# PyInstaller specification for the single Eloquent backend/TTS sidecar.
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
for distribution in ("torchcodec", "pymatting"):
    datas += copy_metadata(distribution)

# Chatterbox uses inspect.getsource() while defining its custom Llama model.
datas += collect_data_files("chatterbox", include_py_files=True)
datas += collect_data_files("voxcpm", include_py_files=True)
datas += collect_data_files("tiktoken_ext", include_py_files=True)

# Language metadata + Kokoro TTS model assets (staged manually in prior builds).
# Kokoro needs its .py sources on disk (collect_data_files without include_py_files
# collects 0 files for it).
datas += collect_data_files("language_tags")
datas += collect_data_files("kokoro", include_py_files=True)

for package in ("uvicorn", "fastapi", "starlette", "pydantic", "llama_cpp", "stable_diffusion_cpp", "transformers", "sentence_transformers", "tiktoken"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

triton_datas, triton_binaries, triton_hiddenimports = collect_all("triton")
datas += triton_datas
binaries += triton_binaries
hiddenimports += triton_hiddenimports

a = Analysis(
    ["sidecar_entry.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=["backend/default_runtime_profile.py"],
    excludes=[
        "backend.app.model_server",
        "backend.app.model_subprocess",
        "backend.app.tests",
        "backend.app.forensic_linguistics_service",
        "backend.app.election_forecast",
        "backend.app.election_db",
        "backend.app.election_data_service",
        "backend.app.votehub_service",
        "backend.app.rcp_service",
        "backend.app.election_ai_service",
        "backend.app.election_simulation",
        "backend.app.ballotpedia_scraper",
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
    name="eloquent-sidecar-x86_64-pc-windows-msvc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    exclude_binaries=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="eloquent-sidecar",
)
