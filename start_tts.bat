@echo off
echo Starting TTS Service on port 8002...
cd /d "%~dp0"
set TTS_PORT=8002
set TTS_HOST=127.0.0.1
set CUDA_VISIBLE_DEVICES=1
set CUDA_DEVICE=0
python launch_tts.py
if errorlevel 1 (
    echo.
    echo ERROR: TTS Service crashed with error code %errorlevel%
    echo.
    pause
) else (
    echo.
    echo SUCCESS: TTS Service stopped normally
    echo.
    pause
)
