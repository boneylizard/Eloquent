[CmdletBinding()]
param(
    [string]$Executable = (Join-Path $PSScriptRoot "..\build\sidecar-dist\mirid-sidecar\mirid-sidecar-x86_64-pc-windows-msvc.exe"),
    [int]$BackendPort = 8765,
    [int]$TtsPort = 8766,
    [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$sidecar = (Resolve-Path -LiteralPath $Executable).Path
$backendProcess = $null
$ttsProcess = $null
$audioPath = Join-Path $env:TEMP "mirid-sidecar-smoke-$PID.wav"
$logDirectory = Join-Path $PSScriptRoot "..\build\release-smoke"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$backendStdout = Join-Path $logDirectory "frozen-backend.stdout.log"
$backendStderr = Join-Path $logDirectory "frozen-backend.stderr.log"
$ttsStdout = Join-Path $logDirectory "frozen-tts.stdout.log"
$ttsStderr = Join-Path $logDirectory "frozen-tts.stderr.log"
$previousLogDirectory = $env:MIRID_LOG_DIR
foreach ($logPath in @($backendStdout, $backendStderr, $ttsStdout, $ttsStderr)) {
    if (Test-Path -LiteralPath $logPath) {
        [System.IO.File]::Delete($logPath)
    }
}

function Wait-ForHealth {
    param(
        [System.Diagnostics.Process]$Process,
        [int]$Port,
        [string]$Name,
        [string]$ErrorLog
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 4
        if ($Process.HasExited) {
            $tail = if (Test-Path -LiteralPath $ErrorLog) {
                (Get-Content -LiteralPath $ErrorLog -Tail 30) -join [Environment]::NewLine
            } else {
                "No error log was written."
            }
            throw "$Name sidecar exited with code $($Process.ExitCode).`n$tail"
        }
        try {
            if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq "healthy") {
                return
            }
        } catch {
        }
    }

    $tail = if (Test-Path -LiteralPath $ErrorLog) {
        (Get-Content -LiteralPath $ErrorLog -Tail 30) -join [Environment]::NewLine
    } else {
        "No error log was written."
    }
    throw "$Name sidecar did not become healthy within $TimeoutSeconds seconds.`n$tail"
}

try {
    $env:MIRID_LOG_DIR = $logDirectory
    $backendProcess = Start-Process -FilePath $sidecar -ArgumentList @(
        "backend", "--host", "127.0.0.1", "--port", $BackendPort
    ) -WindowStyle Hidden -RedirectStandardOutput $backendStdout -RedirectStandardError $backendStderr -PassThru
    Wait-ForHealth -Process $backendProcess -Port $BackendPort -Name "Backend" -ErrorLog $backendStderr

    $ttsProcess = Start-Process -FilePath $sidecar -ArgumentList @(
        "tts", "--host", "127.0.0.1", "--port", $TtsPort
    ) -WindowStyle Hidden -RedirectStandardOutput $ttsStdout -RedirectStandardError $ttsStderr -PassThru
    Wait-ForHealth -Process $ttsProcess -Port $TtsPort -Name "TTS" -ErrorLog $ttsStderr

    $parakeetStatus = Invoke-RestMethod "http://127.0.0.1:$BackendPort/stt/parakeet-cpp/status" -TimeoutSec 10
    if (-not $parakeetStatus.available) {
        throw "Frozen backend cannot find parakeet-cli.exe."
    }

    $body = @{
        text = "Mirid is ready."
        engine = "kokoro"
        voice = "af_heart"
    } | ConvertTo-Json
    Invoke-WebRequest "http://127.0.0.1:$TtsPort/tts/synthesize" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -OutFile $audioPath `
        -TimeoutSec 300 | Out-Null
    $audio = [System.IO.File]::ReadAllBytes($audioPath)
    $header = [System.Text.Encoding]::ASCII.GetString($audio, 0, [Math]::Min(4, $audio.Length))
    if ($header -ne "RIFF" -or $audio.Length -lt 1000) {
        throw "Kokoro returned invalid WAV data."
    }

    [pscustomobject]@{
        Backend = "healthy"
        Tts = "healthy"
        ParakeetCpp = $parakeetStatus.available
        ParakeetBinary = $parakeetStatus.binary_path
        KokoroBytes = $audio.Length
    } | Format-List
} finally {
    $env:MIRID_LOG_DIR = $previousLogDirectory
    foreach ($process in @($backendProcess, $ttsProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
    }
    if (Test-Path -LiteralPath $audioPath) {
        [System.IO.File]::Delete($audioPath)
    }
}
