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
$knownDescendantIds = @()
$audioPath = Join-Path $env:TEMP "mirid-sidecar-smoke-$PID.wav"
$logDirectory = Join-Path $PSScriptRoot "..\build\release-smoke"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$backendStdout = Join-Path $logDirectory "frozen-backend.stdout.log"
$backendStderr = Join-Path $logDirectory "frozen-backend.stderr.log"
$ttsStdout = Join-Path $logDirectory "frozen-tts.stdout.log"
$ttsStderr = Join-Path $logDirectory "frozen-tts.stderr.log"
$previousLogDirectory = $env:MIRID_LOG_DIR
$previousCudaPath = $env:CUDA_PATH
$previousProcessPath = $env:PATH
$previousUserProfile = $env:USERPROFILE
$previousHome = $env:HOME
$previousDataDirectory = $env:MIRID_DATA_DIR
$previousLocalAppData = $env:LOCALAPPDATA
$previousAppData = $env:APPDATA
$previousNumbaCache = $env:NUMBA_CACHE_DIR
$isolatedUserProfile = Join-Path $env:TEMP "mirid-sidecar-smoke-user-$PID"
$isolatedLocalAppData = Join-Path $isolatedUserProfile "AppData\Local"
$isolatedAppData = Join-Path $isolatedUserProfile "AppData\Roaming"
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

function Get-DescendantProcessIds {
    param([int]$RootProcessId)

    $descendants = [System.Collections.Generic.List[int]]::new()
    $pending = [System.Collections.Generic.Queue[int]]::new()
    $pending.Enqueue($RootProcessId)
    while ($pending.Count -gt 0) {
        $parentId = $pending.Dequeue()
        foreach ($child in Get-CimInstance Win32_Process -Filter "ParentProcessId = $parentId") {
            $childId = [int]$child.ProcessId
            $descendants.Add($childId)
            $pending.Enqueue($childId)
        }
    }
    return $descendants.ToArray()
}

try {
    New-Item -ItemType Directory -Path $isolatedLocalAppData -Force | Out-Null
    New-Item -ItemType Directory -Path $isolatedAppData -Force | Out-Null
    $env:MIRID_LOG_DIR = $logDirectory
    $env:USERPROFILE = $isolatedUserProfile
    $env:HOME = $isolatedUserProfile
    $env:LOCALAPPDATA = $isolatedLocalAppData
    $env:APPDATA = $isolatedAppData
    $env:NUMBA_CACHE_DIR = Join-Path $isolatedLocalAppData "Numba\Cache"
    $env:MIRID_DATA_DIR = Join-Path $isolatedUserProfile "data"
    Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
    $env:PATH = (($previousProcessPath -split ';') | Where-Object {
        $_ -and $_ -notmatch '(?i)CUDA|NVIDIA GPU Computing Toolkit'
    }) -join ';'
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

    $imageStatus = Invoke-RestMethod "http://127.0.0.1:$BackendPort/sd-local/status" -TimeoutSec 120
    if (-not $imageStatus.available) {
        throw "Frozen backend could not start the spawned local image worker."
    }
    $knownDescendantIds = @(Get-DescendantProcessIds -RootProcessId $backendProcess.Id)

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
        ImageWorker = "spawned"
        ParakeetCpp = $parakeetStatus.available
        ParakeetBinary = $parakeetStatus.binary_path
        KokoroBytes = $audio.Length
    } | Format-List
} finally {
    $env:MIRID_LOG_DIR = $previousLogDirectory
    $env:CUDA_PATH = $previousCudaPath
    $env:PATH = $previousProcessPath
    $env:USERPROFILE = $previousUserProfile
    $env:HOME = $previousHome
    $env:MIRID_DATA_DIR = $previousDataDirectory
    $env:LOCALAPPDATA = $previousLocalAppData
    $env:APPDATA = $previousAppData
    $env:NUMBA_CACHE_DIR = $previousNumbaCache
    $descendantIds = @(
        $knownDescendantIds
        foreach ($process in @($backendProcess, $ttsProcess)) {
            if ($null -ne $process -and -not $process.HasExited) {
                Get-DescendantProcessIds -RootProcessId $process.Id
            }
        }
    ) | Select-Object -Unique
    foreach ($process in @($backendProcess, $ttsProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
    }
    foreach ($descendantId in $descendantIds) {
        $descendant = Get-Process -Id $descendantId -ErrorAction SilentlyContinue
        if ($null -ne $descendant -and -not $descendant.HasExited) {
            $descendant.Kill()
            $descendant.WaitForExit()
        }
    }
    if (Test-Path -LiteralPath $audioPath) {
        [System.IO.File]::Delete($audioPath)
    }
    if (Test-Path -LiteralPath $isolatedUserProfile) {
        $tempRoot = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        )
        $isolatedRoot = [System.IO.Path]::GetFullPath($isolatedUserProfile)
        $tempPrefix = $tempRoot + [System.IO.Path]::DirectorySeparatorChar
        if (-not $isolatedRoot.StartsWith($tempPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove smoke-test directory outside TEMP: $isolatedRoot"
        }
        Remove-Item -LiteralPath $isolatedRoot -Recurse -Force
    }
}
