@echo off
setlocal

echo 🧠 [Eloquent] Initializing install...

REM Activate or create virtual environment
if not exist venv (
    echo 🐍 Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM Detect Python version
for /f %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%i

echo 🔍 Detected Python version: %PYVER%

REM Install precompiled LLaMA and Stable Diffusion wheels for supported Python versions
if "%PYVER%"=="3.11" (
    echo 📦 Installing LLaMA + SD wheels for Python 3.11...
    pip install wheels\llama_cpp_python-0.3.9-cp311-cp311-win_amd64.whl
    pip install wheels\stable_diffusion_cpp_python-0.2.9-cp311-cp311-win_amd64.whl
) else if "%PYVER%"=="3.12" (
    echo 📦 Installing LLaMA + SD wheels for Python 3.12...
    pip install wheels\llama_cpp_python-0.3.12-cp312-cp312-win_amd64.whl
    pip install wheels\stable_diffusion_cpp_python-0.2.9-cp312-cp312-win_amd64.whl
) else (
    echo ❌ Unsupported Python version: %PYVER%
    exit /b 1
)

REM Install PyTorch with CUDA 12.1
echo ⚙️ Installing PyTorch with CUDA 12.1...
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

REM Preinstall core runtime dependencies (to prevent resolver conflict)
echo 📦 Installing critical FastAPI / audio / vision packages...
pip install --upgrade setuptools
pip install pynvml httpx fastapi uvicorn soundfile librosa python-multipart opencv-python beautifulsoup4 kokoro websockets nemo-toolkit sentence-transformers faiss-cpu protobuf openai opentelemetry-proto onnxruntime googleapis-common-protos


REM Install all remaining pinned dependencies
echo 📜 Installing full requirements.txt...
pip install -r requirements.txt

echo ✅ Eloquent install complete.
pause
