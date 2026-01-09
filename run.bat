@echo off
cls
echo ==========================================================
echo ==         Starting Eloquent AI Application             ==
echo ==========================================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM --- 1. Port configuration ---
REM Ports can be customized in ~/.LiangLocal/settings.json:
REM   "backend_port": 8000, "secondary_port": 8001, "tts_port": 8002
echo Ports are configurable via settings.json (default: 8000, 8001, 8002)
echo.

REM --- 2. Activate Python Virtual Environment ---
echo Activating backend Python environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Backend venv not found at ".\venv". Please run setup.
    pause
    exit /b 1
)
call ".\venv\Scripts\activate.bat"
echo.

REM --- 3. Detect GPU mode for the Frontend ---
echo Detecting GPU configuration for frontend...
python -c "import torch; print('SINGLE' if torch.cuda.device_count() <= 1 else 'DUAL')" > gpu_mode.tmp
set /p GPU_MODE=<gpu_mode.tmp
del gpu_mode.tmp
if "%GPU_MODE%"=="SINGLE" (
    set VITE_SINGLE_GPU_MODE=true
    echo Single GPU mode detected.
) else (
    set VITE_SINGLE_GPU_MODE=false
    echo Dual GPU mode detected.
)
echo.

REM --- 4. Start All Services ---
echo Launching Backend Servers via launch.py (new window)...
START "Eloquent Backend" cmd /c "cd /d "%PROJECT_DIR%" && python launch.py"

echo Launching Frontend Development Server (new window)...
set "FRONTEND_DIR=%PROJECT_DIR%frontend"
START "Eloquent Frontend" cmd /c "cd /d "%FRONTEND_DIR%" && npm run dev"

echo.
echo ==========================================================
echo ==      Eloquent is starting up in new windows.       ==
echo == This window can now be closed.                   ==
echo ==========================================================
echo.
timeout /t 5 > nul
exit
