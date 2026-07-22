@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found at "%~dp0venv"
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"

if exist "sculpt.env.bat" (
    echo Loading voice sculpt config from sculpt.env.bat...
    call "sculpt.env.bat"
)

echo Starting backend via launch.py...
python launch.py
if errorlevel 1 (
    echo.
    echo Backend exited with an error.
    pause
)
