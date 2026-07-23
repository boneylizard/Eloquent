param(
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$entryPoint = Join-Path $root "sidecar_entry.py"
$tauri = Join-Path $root "frontend\node_modules\.bin\tauri.cmd"

function Test-LocalPort {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.ConnectAsync("127.0.0.1", $Port)
        return $connection.Wait(300) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Mirid's development venv is missing: $python"
}
if (-not (Test-Path -LiteralPath $entryPoint -PathType Leaf)) {
    throw "Mirid's service entry point is missing: $entryPoint"
}
if (-not (Test-Path -LiteralPath $tauri -PathType Leaf)) {
    throw "The Tauri development CLI is missing. Run npm install in frontend first."
}

foreach ($port in 8000, 8002) {
    if (Test-LocalPort -Port $port) {
        throw "Port $port is already in use. Close installed or development copies of Mirid, then try again."
    }
}

& $python -c "import fastapi, uvicorn; import backend.app.compute_capabilities; print('Mirid venv ready:', __import__('sys').executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Mirid's development venv could not import the desktop service dependencies."
}

$previousDevVenv = $env:MIRID_DEV_USE_VENV
$previousForceCpu = $env:MIRID_FORCE_CPU
$previousPythonUtf8 = $env:PYTHONUTF8
$previousPythonEncoding = $env:PYTHONIOENCODING

try {
    $env:MIRID_DEV_USE_VENV = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    if ($ForceCpu) {
        $env:MIRID_FORCE_CPU = "1"
    } else {
        Remove-Item Env:MIRID_FORCE_CPU -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "Starting Mirid from $python"
    Write-Host "The Tauri window, backend and voice service will stop together."
    Write-Host ""

    Push-Location $root
    try {
        & $tauri dev
        if ($LASTEXITCODE -ne 0) {
            throw "Mirid development mode exited with code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:MIRID_DEV_USE_VENV = $previousDevVenv
    $env:MIRID_FORCE_CPU = $previousForceCpu
    $env:PYTHONUTF8 = $previousPythonUtf8
    $env:PYTHONIOENCODING = $previousPythonEncoding
}
