@echo off
echo Starting backend on port 8000 with GPU 0...
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=0
set GPU_ID=0
set PORT=8000
set PYTHONPATH=%~dp0
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --log-level info --ws-ping-interval 300
if errorlevel 1 (
    echo.
    echo ERROR: Backend crashed with error code %errorlevel%
    echo.
    pause
) else (
    echo.
    echo SUCCESS: Backend stopped normally
    echo.
    pause
)
