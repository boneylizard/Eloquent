@echo off
echo Installing Stockfish for Eloquent Chess tab...
cd /d "%~dp0"
python scripts\install_stockfish.py
if errorlevel 1 (
  echo Install failed.
  pause
  exit /b 1
)
echo.
echo Done. You can now use the Chess tab in Eloquent.
pause
