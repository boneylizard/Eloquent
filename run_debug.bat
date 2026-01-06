@echo off
echo Starting LiangLocal in DEBUG mode...
echo This will keep windows open if there are crashes.
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting debug launcher...
python launch_debug.py

echo.
echo Debug launcher finished.
pause
