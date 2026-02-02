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
set "GPU_COUNT="
for /f %%A in ('powershell -NoProfile -Command "if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { (nvidia-smi --query-gpu=index --format=csv,noheader 2>$null | Measure-Object).Count } else { 0 }"') do set "GPU_COUNT=%%A"

if not defined GPU_COUNT (
    set "GPU_COUNT=0"
)

if %GPU_COUNT% LEQ 1 (
    set VITE_SINGLE_GPU_MODE=true
    if %GPU_COUNT%==0 (
        echo GPU detection failed. Defaulting to single GPU mode.
    ) else (
        echo Single GPU mode detected.
    )
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
if not exist "%FRONTEND_DIR%\package.json" (
    echo ERROR: Frontend folder or package.json missing at "%FRONTEND_DIR%".
    echo The frontend will not start.
    pause
    exit /b 1
)
if not exist "%FRONTEND_DIR%\node_modules" (
    echo Frontend dependencies missing. Running npm install...
    pushd "%FRONTEND_DIR%"
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed. Frontend will not start.
        popd
        pause
        exit /b 1
    )
    popd
) else if not exist "%FRONTEND_DIR%\node_modules\.bin\vite.cmd" (
    echo Frontend dependencies incomplete. Running npm install...
    pushd "%FRONTEND_DIR%"
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed. Frontend will not start.
        popd
        pause
        exit /b 1
    )
    popd
)
START "Eloquent Frontend" cmd /c "cd /d "%FRONTEND_DIR%" && npm run dev"

echo.
echo ==========================================================
echo ==      Eloquent is starting up in new windows.       ==
echo == This window can now be closed.                   ==
echo ==========================================================
echo.
timeout /t 5 > nul
exit
