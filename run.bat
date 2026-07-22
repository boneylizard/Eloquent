@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo ==              Starting Mirid Desktop                  ==
echo ==========================================================
echo.

REM Use the already-built release exe directly (no reinstall needed).
REM The runtime is cached in %LOCALAPPDATA%\com.eloquent.app\runtime (legacy id kept on purpose).
REM Tauri names the release exe after productName ("Mirid"); older builds produced eloquent.exe.
if exist "src-tauri\target\release\mirid.exe" (
    copy /Y "src-tauri\target\release\mirid.exe" "Mirid.exe" >nul
    start "" "%~dp0Mirid.exe"
    goto :done
)
if exist "src-tauri\target\release\eloquent.exe" (
    copy /Y "src-tauri\target\release\eloquent.exe" "Mirid.exe" >nul
    start "" "%~dp0Mirid.exe"
    goto :done
)

echo Release build not found. Falling back to dev mode...
if not exist "frontend\node_modules\.bin\tauri.cmd" (
    echo Installing frontend dependencies...
    pushd "frontend"
    call npm install
    popd
)
call "frontend\node_modules\.bin\tauri.cmd" dev

:done
