[CmdletBinding()]
param(
    [string]$ExpectedVersion = "1.0.12",
    [string]$OutputDirectory = (Join-Path $env:PUBLIC "Documents\Mirid Release Test"),
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$installedExecutable = Join-Path $env:LOCALAPPDATA "Mirid\mirid.exe"
$appDataRoot = Join-Path $env:LOCALAPPDATA "ai.mirid.desktop"
$runtimeRoot = Join-Path $appDataRoot "runtime"
$runtimeMarker = Join-Path $runtimeRoot "runtime.ready"
$logRoot = Join-Path $appDataRoot "logs"
$resultPath = Join-Path $OutputDirectory "fresh-test-result.json"

if ($PlanOnly) {
    [pscustomobject]@{
        WindowsUser = [Environment]::UserName
        InstalledExecutable = $installedExecutable
        RuntimeMarker = $runtimeMarker
        LogDirectory = $logRoot
        Result = $resultPath
        MiridWillLaunch = $false
        Action = "Preview only; no file was changed."
    } | Format-List
    exit 0
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$errors = [System.Collections.Generic.List[string]]::new()
$installedFile = Get-Item -LiteralPath $installedExecutable -ErrorAction SilentlyContinue
$installedVersion = if ($installedFile) { [string]$installedFile.VersionInfo.ProductVersion } else { "" }
$installedHash = if ($installedFile) { (Get-FileHash -LiteralPath $installedExecutable -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
if (-not $installedFile) {
    $errors.Add("Mirid is not installed for this Windows account.")
} elseif ($installedVersion -ne $ExpectedVersion) {
    $errors.Add("Installed Mirid version is $installedVersion; expected $ExpectedVersion.")
}

$runtimeVersion = if (Test-Path -LiteralPath $runtimeMarker) {
    (Get-Content -LiteralPath $runtimeMarker -Raw).Trim()
} else {
    ""
}
if ($runtimeVersion -ne "v9") {
    $errors.Add("The v9 runtime-ready marker is missing.")
}

$runtimeDirectories = if (Test-Path -LiteralPath $runtimeRoot) {
    @(
        Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Directory -Filter "_internal*" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    )
} else {
    @()
}
if (-not $runtimeDirectories.Count) {
    $errors.Add("No installed runtime directory was found.")
}

$latestLog = if (Test-Path -LiteralPath $logRoot) {
    Get-ChildItem -LiteralPath $logRoot -Filter "*.log" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
} else {
    $null
}
$logText = if ($latestLog) { Get-Content -LiteralPath $latestLog.FullName -Raw } else { "" }
$servicesReady = $logText -match "Local services are ready\."
$runtimeFailure = $logText -match "Runtime setup failed|cannot stage previous runtime|Access is denied \(os error 5\)"
$serviceFailure = $logText -match "winerror 10048|process exited before its local endpoint became ready|Failed to start services"
$backendMatches = [regex]::Matches($logText, "Starting backend sidecar on \S+:(\d+)")
$ttsMatches = [regex]::Matches($logText, "Starting tts sidecar on \S+:(\d+)")
$backendPort = if ($backendMatches.Count) { [int]$backendMatches[$backendMatches.Count - 1].Groups[1].Value } else { $null }
$ttsPort = if ($ttsMatches.Count) { [int]$ttsMatches[$ttsMatches.Count - 1].Groups[1].Value } else { $null }
if (-not $latestLog) {
    $errors.Add("No Mirid first-launch log was found.")
} elseif (-not $servicesReady) {
    $errors.Add("The log does not confirm that local services became ready.")
}
if ($runtimeFailure) {
    $errors.Add("The log contains a runtime setup or access-denied failure.")
}
if ($serviceFailure) {
    $errors.Add("The log contains a local-service startup or port-binding failure.")
}

$runningProcesses = @(Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc", "eloquent-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue |
    Select-Object ProcessName, Id, StartTime)

$evidenceDirectory = Join-Path $OutputDirectory "evidence"
New-Item -ItemType Directory -Path $evidenceDirectory -Force | Out-Null
if ($latestLog) {
    Copy-Item -LiteralPath $latestLog.FullName -Destination (Join-Path $evidenceDirectory $latestLog.Name) -Force
}

$result = [ordered]@{
    passed = ($errors.Count -eq 0)
    collectedAt = (Get-Date).ToString("o")
    windowsUser = [Environment]::UserName
    expectedVersion = $ExpectedVersion
    installedVersion = $installedVersion
    installedExecutable = $installedExecutable
    installedSha256 = $installedHash
    runtimeVersion = $runtimeVersion
    runtimeDirectories = $runtimeDirectories
    latestLog = if ($latestLog) { $latestLog.FullName } else { "" }
    servicesReady = $servicesReady
    runtimeFailure = $runtimeFailure
    serviceFailure = $serviceFailure
    backendPort = $backendPort
    ttsPort = $ttsPort
    runningProcesses = $runningProcesses
    errors = @($errors)
}
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultPath -Encoding utf8

if ($result.passed) {
    Write-Host "Fresh-account Mirid test passed." -ForegroundColor Green
} else {
    Write-Host "Fresh-account Mirid test needs attention." -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "- $_" }
}
Write-Host "Result: $resultPath"
Write-Host "Mirid was not launched or stopped by this collector."

if (-not $result.passed) {
    exit 1
}
