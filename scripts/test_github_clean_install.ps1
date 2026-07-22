[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerUrl,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-fA-F0-9]{64}$")]
    [string]$InstallerSha256,
    [string]$ExpectedVersion = "1.0.2",
    [string]$ExpectedRuntimeVersion = "v3",
    [int]$TimeoutSeconds = 14400,
    [string]$EvidenceDirectory = (Join-Path $PSScriptRoot "..\artifacts\github-clean-install")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$evidence = [System.IO.Path]::GetFullPath($EvidenceDirectory)
$installer = Join-Path $env:RUNNER_TEMP "Mirid-Setup.exe"
$installedExecutable = Join-Path $env:LOCALAPPDATA "Mirid\mirid.exe"
$appDataRoot = Join-Path $env:LOCALAPPDATA "ai.mirid.desktop"
$runtimeRoot = Join-Path $appDataRoot "runtime"
$runtimeMarker = Join-Path $runtimeRoot "runtime.ready"
$logRoot = Join-Path $appDataRoot "logs"
$devToolsPort = 9229
$startedAt = Get-Date
$downloadSeconds = 0
$firstLaunchSeconds = 0
$failure = ""
$passed = $false
$process = $null
$previousQaAutoBeginSetup = $env:MIRID_QA_AUTO_BEGIN_SETUP

function Stop-MiridProcesses {
    Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc", "eloquent-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Get-FreeDiskGiB {
    $drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
    return [math]::Round($drive.FreeSpace / 1GB, 2)
}

function Wait-ForMirid {
    param([Diagnostics.Process]$DesktopProcess, [int]$DeadlineSeconds)
    $deadline = (Get-Date).AddSeconds($DeadlineSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        $DesktopProcess.Refresh()
        if ($DesktopProcess.HasExited) {
            throw "Mirid exited during first launch with code $($DesktopProcess.ExitCode)."
        }

        $runtimeReady = (Test-Path -LiteralPath $runtimeMarker) -and
            ((Get-Content -LiteralPath $runtimeMarker -Raw).Trim() -eq $ExpectedRuntimeVersion)
        if (-not $runtimeReady) { continue }

        try {
            $backend = Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 5
            $backendReady = $backend.status -eq "healthy"
        } catch {
            $backendReady = $false
        }
        try {
            $tts = Invoke-RestMethod "http://127.0.0.1:8002/health" -TimeoutSec 5
            $ttsReady = $tts.status -eq "healthy"
        } catch {
            $ttsReady = $false
        }
        try {
            $targets = @(Invoke-RestMethod "http://127.0.0.1:$devToolsPort/json" -TimeoutSec 5)
            $frontendReady = [bool]($targets | Where-Object {
                $_.type -eq "page" -and $_.url -like "http://tauri.localhost/*"
            } | Select-Object -First 1)
        } catch {
            $frontendReady = $false
        }

        if ($backendReady -and $ttsReady -and $frontendReady) {
            return
        }
    }
    throw "Mirid did not finish first launch within $DeadlineSeconds seconds."
}

New-Item -ItemType Directory -Path $evidence -Force | Out-Null
Stop-MiridProcesses

try {
    foreach ($path in @(
        (Join-Path $env:LOCALAPPDATA "Mirid"),
        $appDataRoot,
        (Join-Path $env:LOCALAPPDATA "com.eloquent.app"),
        (Join-Path $HOME ".LiangLocal")
    )) {
        if (Test-Path -LiteralPath $path) {
            throw "GitHub runner was not clean before installation: $path"
        }
    }

    $freeBefore = Get-FreeDiskGiB
    if ($freeBefore -lt 12) {
        throw "The GitHub runner has only $freeBefore GiB free; at least 12 GiB is required."
    }

    $downloadStarted = Get-Date
    Invoke-WebRequest -Uri $InstallerUrl -OutFile $installer -UseBasicParsing
    $downloadSeconds = [math]::Round(((Get-Date) - $downloadStarted).TotalSeconds, 1)
    $actualInstallerHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
    if (-not $actualInstallerHash.Equals($InstallerSha256, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Installer SHA-256 mismatch."
    }

    $install = Start-Process -FilePath $installer -ArgumentList "/S" -PassThru -Wait
    if ($install.ExitCode -ne 0) {
        throw "Installer exited with code $($install.ExitCode)."
    }
    $installedFile = Get-Item -LiteralPath $installedExecutable -ErrorAction Stop
    if ($installedFile.VersionInfo.ProductVersion -ne $ExpectedVersion) {
        throw "Installed version is $($installedFile.VersionInfo.ProductVersion); expected $ExpectedVersion."
    }
    if (Test-Path -LiteralPath $appDataRoot) {
        throw "Mirid created runtime state before its first launch."
    }

    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = "--remote-debugging-port=$devToolsPort"
    $env:MIRID_QA_AUTO_BEGIN_SETUP = "1"
    $launchStarted = Get-Date
    $process = Start-Process -FilePath $installedExecutable -PassThru
    $env:MIRID_QA_AUTO_BEGIN_SETUP = $previousQaAutoBeginSetup
    Wait-ForMirid -DesktopProcess $process -DeadlineSeconds $TimeoutSeconds
    $firstLaunchSeconds = [math]::Round(((Get-Date) - $launchStarted).TotalSeconds, 1)

    $releaseDirectories = @(Get-ChildItem -LiteralPath (Join-Path $runtimeRoot "releases") -Directory -ErrorAction SilentlyContinue)
    $validLayouts = @($releaseDirectories | Where-Object {
        (Test-Path -LiteralPath (Join-Path $_.FullName "mirid-sidecar-x86_64-pc-windows-msvc.exe")) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "_internal\python312.dll"))
    })
    if ($validLayouts.Count -ne 1) {
        throw "Expected exactly one complete versioned PyInstaller runtime; found $($validLayouts.Count)."
    }
    if (Get-ChildItem -LiteralPath $runtimeRoot -Directory -Filter "_internal-v3-*" -ErrorAction SilentlyContinue) {
        throw "The obsolete renamed _internal layout was created."
    }

    $logText = Get-ChildItem -LiteralPath $logRoot -Filter "*.log" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 |
        Get-Content -Raw
    if ($logText -match "failed to load Python DLL|cannot stage previous runtime|Access is denied \(os error 5\)|Runtime setup failed") {
        throw "The first-launch log contains a release-blocking runtime error."
    }
    if ($logText -notmatch "Local services are ready\.") {
        throw "The first-launch log does not confirm healthy local services."
    }

    $passed = $true
} catch {
    $failure = $_.Exception.Message
    throw
} finally {
    $env:MIRID_QA_AUTO_BEGIN_SETUP = $previousQaAutoBeginSetup
    Stop-MiridProcesses
    if (Test-Path -LiteralPath $logRoot) {
        Copy-Item -LiteralPath $logRoot -Destination (Join-Path $evidence "logs") -Recurse -Force -ErrorAction SilentlyContinue
    }
    $videoControllers = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name, AdapterRAM)
    [ordered]@{
        passed = $passed
        failure = $failure
        runner = $env:ImageOS
        expectedVersion = $ExpectedVersion
        installerUrl = $InstallerUrl
        installerSha256 = $InstallerSha256.ToLowerInvariant()
        downloadSeconds = $downloadSeconds
        firstLaunchSeconds = $firstLaunchSeconds
        freeDiskGiBBefore = if (Get-Variable freeBefore -ErrorAction SilentlyContinue) { $freeBefore } else { $null }
        freeDiskGiBAfter = Get-FreeDiskGiB
        runtimeVersion = if (Test-Path -LiteralPath $runtimeMarker) { (Get-Content -LiteralPath $runtimeMarker -Raw).Trim() } else { "" }
        videoControllers = $videoControllers
        durationSeconds = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidence "summary.json") -Encoding utf8
}
