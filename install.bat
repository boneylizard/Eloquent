@echo off
setlocal

echo 🧠 [Eloquent] Initializing install...

REM Activate or create venv
if not exist venv (
    echo 🐍 Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM Detect Python version
for /f %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%i

echo 🔍 Detected Python version: %PYVER%

REM Install matching wheels
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

REM Install all remaining dependencies
echo 📜 Installing remaining dependencies...
pip install -r requirements.txt

echo ✅ Eloquent install complete.
pause