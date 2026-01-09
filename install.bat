@echo off
setlocal enabledelayedexpansion
echo [Eloquent] Initializing install...

REM --- 1. Activate or create virtual environment ---
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM --- 2. Detect Python version ---
for /f %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%i
echo Detected Python version: %PYVER%

REM --- 3. Install precompiled LLaMA and SD wheels ---
if "%PYVER%"=="3.11" (
    echo Installing LLaMA + SD wheels for Python 3.11...
    pip install wheels\llama_cpp_python-0.3.16-cp311-cp311-win_amd64.whl
    pip install wheels\stable_diffusion_cpp_python-0.2.9-cp311-cp311-win_amd64.whl
) else if "%PYVER%"=="3.12" (
    echo Installing LLaMA + SD wheels for Python 3.12...
    pip install wheels\llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl
    pip install wheels\stable_diffusion_cpp_python-0.2.9-cp312-cp312-win_amd64.whl
) else (
    echo ERROR: Unsupported Python version: %PYVER%
    exit /b 1
)

REM --- 4. Install PyTorch with CUDA 12.1 ---
echo Installing PyTorch with CUDA 12.1...
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

REM --- 5. Install core runtime packages ---
echo Installing critical FastAPI / audio / vision packages...
pip install --upgrade setuptools
pip install pynvml httpx fastapi uvicorn soundfile librosa python-multipart opencv-python beautifulsoup4 kokoro websockets nemo-toolkit sentence-transformers faiss-cpu protobuf openai opentelemetry-proto onnxruntime googleapis-common-protos

REM --- 6. Pre-install numpy and chatterbox (pkuseg build fix) ---
echo Installing numpy first (required for pkuseg build)...
pip install numpy

echo Installing chatterbox-faster with --no-build-isolation (fixes pkuseg build error)...
pip install --no-build-isolation git+https://github.com/iMash-io/chatterbox-faster.git

REM --- 7. Install all remaining requirements.txt packages ---
echo Installing full requirements.txt...
pip install -r requirements.txt

REM --- 8. Downgrade numpy to 1.x (PyTorch compatibility) ---
echo Downgrading numpy to 1.x for PyTorch compatibility...
pip install "numpy<2"

REM --- 9. Node.js version check ---
echo Checking Node.js version...
node -v > temp_node_version.txt 2>nul
if errorlevel 1 (
    echo ERROR: Node.js not found. Please install Node.js v21.7.3
    exit /b 1
)

set /p NODE_VER=<temp_node_version.txt
del temp_node_version.txt
echo Detected Node.js version: !NODE_VER!

if not "!NODE_VER!"=="v21.7.3" (
    echo.
    echo WARNING: This project has only been tested with Node.js v21.7.3
    echo You are using !NODE_VER!
    echo.
    echo Compatibility issues are likely. Things may break in weird ways.
    echo We strongly recommend switching to Node.js v21.7.3 before continuing.
    echo.
    echo Press ENTER to acknowledge and continue anyway at your own risk...
    pause >nul
)

REM --- 10. Install frontend dependencies ---
if exist frontend (
    echo Installing frontend dependencies via npm...
    cd frontend
    
    REM Clean any existing install artifacts that might cause issues
    if exist node_modules (
        echo Cleaning existing node_modules...
        rmdir /s /q node_modules
    )
    if exist package-lock.json (
        echo Cleaning existing package-lock.json...
        del package-lock.json
    )
    
    REM Force fresh install
    echo Running fresh npm install...
    call npm install
    
    if errorlevel 1 (
        echo ERROR: npm install failed
        cd ..
        exit /b 1
    )
    
    cd ..
) else (
    echo ERROR: 'frontend' folder not found. Skipping npm install.
)

echo Eloquent installation complete.
pause