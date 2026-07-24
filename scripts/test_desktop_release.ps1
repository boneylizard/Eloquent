[CmdletBinding()]
param(
    [string]$Executable = (Join-Path $PSScriptRoot "..\src-tauri\target\release\mirid.exe"),
    [string]$ExpectedRuntimeVersion,
    [int]$TimeoutSeconds = 1800,
    [int]$DevToolsPort = 9229,
    [switch]$ValidateKokoro,
    [string]$InspectionOutput,
    [string]$ScreenshotPath,
    [switch]$ExpectFirstRunSetup,
    [switch]$ExpectDarkTheme,
    [switch]$LeaveRunning
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$app = (Resolve-Path -LiteralPath $Executable).Path
$manifestPath = Join-Path $root "runtime\runtime-release.json"
if (-not $ExpectedRuntimeVersion) {
    $ExpectedRuntimeVersion = (Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json).runtimeVersion
}
$runtimeMarker = Join-Path $env:LOCALAPPDATA "ai.mirid.desktop\runtime\runtime.ready"
$logRoot = Join-Path $env:LOCALAPPDATA "ai.mirid.desktop\logs"
$previousBrowserArguments = $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS
$process = $null
$passed = $false

function Get-MiridServicePorts {
    if (-not (Test-Path -LiteralPath $logRoot)) {
        return $null
    }
    $latestLog = Get-ChildItem -LiteralPath $logRoot -Filter "*.log" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latestLog) {
        return $null
    }
    $text = Get-Content -LiteralPath $latestLog.FullName -Raw
    $backendMatches = [regex]::Matches($text, "Starting backend sidecar on \S+:(\d+)")
    $ttsMatches = [regex]::Matches($text, "Starting tts sidecar on \S+:(\d+)")
    if ($backendMatches.Count -eq 0 -or $ttsMatches.Count -eq 0) {
        return $null
    }
    return [pscustomobject]@{
        Backend = [int]$backendMatches[$backendMatches.Count - 1].Groups[1].Value
        Tts = [int]$ttsMatches[$ttsMatches.Count - 1].Groups[1].Value
    }
}

Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$occupiedPorts = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {
    $_.LocalPort -eq $DevToolsPort
})
if ($occupiedPorts.Count -gt 0) {
    $owners = $occupiedPorts | ForEach-Object {
        $owner = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        "$($_.LocalPort) ($($owner.ProcessName), PID $($_.OwningProcess))"
    }
    throw "Release smoke-test ports are already in use: $($owners -join ', ')."
}

try {
    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = "--remote-debugging-port=$DevToolsPort"
    $process = Start-Process -FilePath $app -PassThru
    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = $previousBrowserArguments

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $backendReady = $false
    $ttsReady = $false
    $frontendReady = $false
    $frontendTitle = ""
    $frontendUrl = ""
    $runtimeVersion = "missing"
    $backendPort = 0
    $ttsPort = 0

    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        if ($process.HasExited) {
            throw "Mirid exited during the desktop release smoke test with code $($process.ExitCode)."
        }

        if (Test-Path -LiteralPath $runtimeMarker) {
            $runtimeVersion = (Get-Content -LiteralPath $runtimeMarker -Raw).Trim()
        }
        $runtimeReady = $runtimeVersion -eq $ExpectedRuntimeVersion
        if (-not $runtimeReady) { continue }

        $servicePorts = Get-MiridServicePorts
        if (-not $servicePorts) { continue }
        $backendPort = $servicePorts.Backend
        $ttsPort = $servicePorts.Tts
        try {
            $backendHealth = Invoke-RestMethod "http://127.0.0.1:$backendPort/health" -TimeoutSec 3
            $backendReady = $backendHealth.status -eq "healthy"
        } catch {
            $backendReady = $false
        }
        try {
            $ttsHealth = Invoke-RestMethod "http://127.0.0.1:$ttsPort/health" -TimeoutSec 3
            $ttsReady = $ttsHealth.status -eq "healthy"
        } catch {
            $ttsReady = $false
        }
        try {
            $targets = @(Invoke-RestMethod "http://127.0.0.1:$DevToolsPort/json" -TimeoutSec 3)
            $frontend = $targets | Where-Object {
                $_.type -eq "page" -and $_.url -like "http://tauri.localhost/*"
            } | Select-Object -First 1
            if ($frontend) {
                $frontendTitle = [string]$frontend.title
                $frontendUrl = [string]$frontend.url
                $frontendReady = $frontendTitle -notmatch "can't reach|cannot be reached|not found"
            }
        } catch {
            $frontendReady = $false
        }

        if ($backendReady -and $ttsReady -and $frontendReady) { break }
    }

    if (-not ($backendReady -and $ttsReady -and $frontendReady)) {
        throw "Mirid did not complete the $ExpectedRuntimeVersion desktop release smoke test within $TimeoutSeconds seconds (runtime=$runtimeVersion, backend=$backendReady, tts=$ttsReady, frontend=$frontendReady)."
    }
    if ($backendPort -ne 8000) {
        throw "Mirid's main engine moved away from required port 8000."
    }

    if ($ValidateKokoro) {
        $audioPath = Join-Path $env:TEMP "mirid-desktop-kokoro-$PID.wav"
        $body = @{
            text = "Mirid is ready."
            engine = "kokoro"
            voice = "af_heart"
        } | ConvertTo-Json
        try {
            Invoke-WebRequest "http://127.0.0.1:$ttsPort/tts/synthesize" `
                -Method Post `
                -ContentType "application/json" `
                -Body $body `
                -OutFile $audioPath `
                -TimeoutSec 300 | Out-Null
            $audio = [System.IO.File]::ReadAllBytes($audioPath)
            $header = [System.Text.Encoding]::ASCII.GetString($audio, 0, [Math]::Min(4, $audio.Length))
            if ($header -ne "RIFF" -or $audio.Length -lt 1000) {
                throw "Installed Kokoro returned invalid WAV data ($($audio.Length) bytes, header '$header')."
            }
            Write-Host "Installed Kokoro synthesis returned $($audio.Length) bytes of WAV audio."
        } finally {
            if (Test-Path -LiteralPath $audioPath) {
                [System.IO.File]::Delete($audioPath)
            }
        }
    }

    if ($InspectionOutput -or $ScreenshotPath -or $ExpectFirstRunSetup -or $ExpectDarkTheme) {
        $inspectionScript = Join-Path $root "frontend\scripts\inspect-desktop.mjs"
        $inspectionArguments = @($inspectionScript, "--port", $DevToolsPort.ToString())
        $temporaryInspection = $false
        if (-not $InspectionOutput) {
            $InspectionOutput = Join-Path $env:TEMP "mirid-desktop-inspection-$PID.json"
            $temporaryInspection = $true
        }
        $inspectionArguments += @("--output", $InspectionOutput)
        if ($ScreenshotPath) {
            $inspectionArguments += @("--screenshot", $ScreenshotPath)
        }
        if ($ExpectFirstRunSetup) {
            $inspectionArguments += "--expect-first-run"
        }
        if ($ExpectDarkTheme) {
            $inspectionArguments += "--expect-dark"
        }

        Push-Location (Join-Path $root "frontend")
        try {
            & node @inspectionArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Desktop UI inspection failed with exit code $LASTEXITCODE."
            }
        } finally {
            Pop-Location
            if ($temporaryInspection -and (Test-Path -LiteralPath $InspectionOutput)) {
                Remove-Item -LiteralPath $InspectionOutput -Force
            }
        }
    }

    $passed = $true
    [pscustomobject]@{
        RuntimeVersion = $ExpectedRuntimeVersion
        Backend = "healthy"
        Tts = "healthy"
        BackendPort = $backendPort
        TtsPort = $ttsPort
        FrontendTitle = $frontendTitle
        FrontendUrl = $frontendUrl
        ProcessId = $process.Id
    } | Format-List
} finally {
    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = $previousBrowserArguments
    if (-not $LeaveRunning -or -not $passed) {
        Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
    }
}
