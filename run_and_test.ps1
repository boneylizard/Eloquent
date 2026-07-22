$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = $PSScriptRoot

# Start backend in background
$backend = Start-Process python -ArgumentList "-c", "`$env:CUDA_VISIBLE_DEVICES='0'; `$env:PYTHONPATH='$PSScriptRoot'; import uvicorn; uvicorn.run('backend.app.main:app', host='127.0.0.1', port=8000, log_level='info', lifespan='on')" -WindowStyle Hidden -PassThru

Write-Host "Backend started with PID: $($backend.Id)"
Start-Sleep 30

# Test health endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 10
    Write-Host "Health check: $($response.StatusCode)"
} catch {
    Write-Host "Health check failed: $($_.Exception.Message)"
}

Start-Sleep 10

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 10
    Write-Host "Health check 2: $($response.StatusCode)"
} catch {
    Write-Host "Health check 2 failed: $($_.Exception.Message)"
}

# Check if process is still alive
$proc = Get-Process -Id $backend.Id -ErrorAction SilentlyContinue
if ($proc) {
    Write-Host "Backend process still alive: $($proc.Id)"
} else {
    Write-Host "Backend process DIED!"
}

Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
