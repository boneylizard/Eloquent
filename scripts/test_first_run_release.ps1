[CmdletBinding()]
param(
    [string]$Executable = (Join-Path $PSScriptRoot "..\src-tauri\target\release\mirid.exe"),
    [ValidateSet("Full", "RuntimeOnly")]
    [string]$Mode = "Full",
    [int]$TimeoutSeconds = 7200,
    [int]$DevToolsPort = 9229,
    [switch]$ValidateKokoro,
    [switch]$Interactive,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$app = (Resolve-Path -LiteralPath $Executable).Path
$runId = (Get-Date).ToString("yyyyMMdd-HHmmss")
$reportDirectory = Join-Path $root "artifacts\first-run\$runId-$($Mode.ToLowerInvariant())"
$inspectionPath = Join-Path $reportDirectory "desktop-state.json"
$screenshotPath = Join-Path $reportDirectory "first-run.png"
$appRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "ai.mirid.desktop"))
$legacyRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "com.eloquent.app"))
$backendRoot = [System.IO.Path]::GetFullPath((Join-Path $HOME ".LiangLocal"))
$runtimeRoot = [System.IO.Path]::GetFullPath((Join-Path $appRoot "runtime"))
$localBackupRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "Mirid-QA-Backups\$runId"))
$homeBackupRoot = [System.IO.Path]::GetFullPath((Join-Path $HOME ".Mirid-QA-Backups\$runId"))
$startedAt = Get-Date
$outcome = "failed"
$failureMessage = $null
$previousQaAutoBeginSetup = $env:MIRID_QA_AUTO_BEGIN_SETUP
$moves = [System.Collections.Generic.List[object]]::new()
$cleanupTargets = [System.Collections.Generic.List[string]]::new()

if ($PlanOnly) {
    [pscustomobject]@{
        Mode = $Mode
        Executable = $app
        AppData = $appRoot
        LegacyAppData = if ($Mode -eq "Full") { $legacyRoot } else { "not moved" }
        BackendSettings = if ($Mode -eq "Full") { $backendRoot } else { "not moved" }
        Runtime = if ($Mode -eq "RuntimeOnly") { $runtimeRoot } else { "included with app data" }
        ReportDirectory = $reportDirectory
        Action = "Preview only; no process or file was changed."
    } | Format-List
    exit 0
}

New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null

function Assert-ExactPath {
    param([string]$Actual, [string]$Expected, [string]$Label)
    $actualFull = [System.IO.Path]::GetFullPath($Actual).TrimEnd('\')
    $expectedFull = [System.IO.Path]::GetFullPath($Expected).TrimEnd('\')
    if (-not $actualFull.Equals($expectedFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to alter unexpected $Label path: $actualFull"
    }
}

function Stop-MiridProcesses {
    Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc", "eloquent-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Move-StateToBackup {
    param([string]$Source, [string]$Destination, [string]$ExpectedSource, [string]$Label)
    Assert-ExactPath -Actual $Source -Expected $ExpectedSource -Label $Label
    $script:cleanupTargets.Add($Source)
    if (-not (Test-Path -LiteralPath $Source)) { return }
    if (Test-Path -LiteralPath $Destination) {
        throw "The QA backup destination already exists: $Destination"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Move-Item -LiteralPath $Source -Destination $Destination
    $script:moves.Add([pscustomobject]@{ Source = $Source; Backup = $Destination; Label = $Label })
}

function Remove-TestState {
    param([string]$Path)
    $knownTargets = @($appRoot, $legacyRoot, $backendRoot, $runtimeRoot)
    if (-not ($knownTargets | Where-Object {
        [System.IO.Path]::GetFullPath($_).TrimEnd('\').Equals(
            [System.IO.Path]::GetFullPath($Path).TrimEnd('\'),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    })) {
        throw "Refusing to remove an unrecognised QA path: $Path"
    }
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Copy-TestLogs {
    $logRoot = Join-Path $appRoot "logs"
    if (-not (Test-Path -LiteralPath $logRoot)) { return }
    $destination = Join-Path $reportDirectory "logs"
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Get-ChildItem -LiteralPath $logRoot -Force -ErrorAction SilentlyContinue |
        Copy-Item -Destination $destination -Recurse -Force -ErrorAction SilentlyContinue
}

Stop-MiridProcesses

try {
    if ($Mode -eq "Full") {
        Move-StateToBackup -Source $appRoot -Destination (Join-Path $localBackupRoot "ai.mirid.desktop") -ExpectedSource $appRoot -Label "Mirid app data"
        Move-StateToBackup -Source $legacyRoot -Destination (Join-Path $localBackupRoot "com.eloquent.app") -ExpectedSource $legacyRoot -Label "legacy app data"
        Move-StateToBackup -Source $backendRoot -Destination (Join-Path $homeBackupRoot ".LiangLocal") -ExpectedSource $backendRoot -Label "backend settings"
    } else {
        Move-StateToBackup -Source $runtimeRoot -Destination (Join-Path $localBackupRoot "runtime") -ExpectedSource $runtimeRoot -Label "Mirid runtime"
    }

    Write-Host "Mirid QA profile is isolated. Your normal data is in a temporary backup."
    Write-Host "Running $Mode first-launch test from $app"

    $testParameters = @{
        Executable = $app
        TimeoutSeconds = $TimeoutSeconds
        DevToolsPort = $DevToolsPort
        ValidateKokoro = $ValidateKokoro
        InspectionOutput = $inspectionPath
        ScreenshotPath = $screenshotPath
        ExpectFirstRunSetup = ($Mode -eq "Full")
        ExpectDarkTheme = ($Mode -eq "Full")
        LeaveRunning = $Interactive
    }
    if (-not $Interactive) {
        $env:MIRID_QA_AUTO_BEGIN_SETUP = "1"
    }
    & (Join-Path $PSScriptRoot "test_desktop_release.ps1") @testParameters | Out-Host
    $env:MIRID_QA_AUTO_BEGIN_SETUP = $previousQaAutoBeginSetup

    if ($Interactive) {
        Write-Host ""
        Write-Host "Mirid is running with a disposable first-user profile."
        Write-Host "Anything entered now, including API keys, will be discarded when this test ends."
        Read-Host "Finish your checks, then press Enter to restore your normal Mirid profile"
    }

    $outcome = "passed"
} catch {
    $failureMessage = $_.Exception.Message
    Write-Host "First-launch test failed: $failureMessage" -ForegroundColor Red
} finally {
    $env:MIRID_QA_AUTO_BEGIN_SETUP = $previousQaAutoBeginSetup
    Stop-MiridProcesses
    Copy-TestLogs

    $profileRestored = $false
    $restoreFailure = $null
    try {
        foreach ($target in @($cleanupTargets)) {
            Remove-TestState -Path $target
        }
        $restoreMoves = @($moves)
        [array]::Reverse($restoreMoves)
        foreach ($move in $restoreMoves) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $move.Source) | Out-Null
            Move-Item -LiteralPath $move.Backup -Destination $move.Source
        }
        $profileRestored = $true
    } catch {
        $restoreFailure = $_.Exception.Message
        $outcome = "failed"
        if (-not $failureMessage) { $failureMessage = $restoreFailure }
        Write-Host "Mirid could not restore the QA backup automatically: $restoreFailure" -ForegroundColor Red
        Write-Host "Backups remain under $localBackupRoot and $homeBackupRoot" -ForegroundColor Yellow
    }

    $finishedAt = Get-Date
    $report = [ordered]@{
        runId = $runId
        mode = $Mode
        outcome = $outcome
        failure = $failureMessage
        executable = $app
        executableSha256 = (Get-FileHash -LiteralPath $app -Algorithm SHA256).Hash.ToLowerInvariant()
        startedAt = $startedAt.ToString("o")
        finishedAt = $finishedAt.ToString("o")
        durationSeconds = [Math]::Round(($finishedAt - $startedAt).TotalSeconds, 1)
        reportDirectory = $reportDirectory
        profileRestored = $profileRestored
        restoreFailure = $restoreFailure
        localBackupRoot = $localBackupRoot
        homeBackupRoot = $homeBackupRoot
    }
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportDirectory "first-run-report.json") -Encoding utf8
    if ($profileRestored) {
        Write-Host "Your normal Mirid profile has been restored."
    }
    Write-Host "QA report: $reportDirectory"
}

if ($outcome -ne "passed") {
    exit 1
}
