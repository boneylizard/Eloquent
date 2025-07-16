@echo off
cls
echo ==========================================================
echo ==         Starting Eloquent AI Application             ==
echo ==========================================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM --- 1. Clean up old processes on ports ---
echo Killing any processes on ports 8000, 8001...
FOR /F "tokens=5" %%P IN ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') DO (taskkill /PID %%P /F /T > nul)
FOR /F "tokens=5" %%P IN ('netstat -aon ^| findstr ":8001" ^| findstr "LISTENING"') DO (taskkill /PID %%P /F /T > nul)
echo Cleanup complete.
echo.
timeout /t 2 /nobreak > nul

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
