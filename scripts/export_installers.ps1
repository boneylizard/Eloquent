[CmdletBinding()]
param(
    [string]$BundleRoot = (Join-Path $PSScriptRoot "..\src-tauri\target\release\bundle"),
    [string]$DestinationRoot = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Mirid Installers")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bundle = (Resolve-Path -LiteralPath $BundleRoot).Path
$tauriConfig = Get-Content -LiteralPath (Join-Path $root "src-tauri\tauri.conf.json") -Raw | ConvertFrom-Json
$version = [string]$tauriConfig.version
$timestamp = (Get-Date).ToString("yyyy-MM-dd_HHmmss")
$releaseName = "Mirid-$version-$timestamp"
$latestDirectory = Join-Path $DestinationRoot "Latest"
$releaseDirectory = Join-Path $DestinationRoot "Releases\$releaseName"

$nsisInstaller = Get-ChildItem -LiteralPath (Join-Path $bundle "nsis") -Filter "Mirid_${version}_*-setup.exe" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $nsisInstaller) { throw "No Mirid $version NSIS installer was found under $bundle\nsis." }

New-Item -ItemType Directory -Force -Path $latestDirectory, $releaseDirectory | Out-Null
Remove-Item -LiteralPath (Join-Path $latestDirectory "Mirid.msi") -Force -ErrorAction SilentlyContinue

$latestInstaller = Join-Path $latestDirectory "Mirid-Setup.exe"
$archivedInstaller = Join-Path $releaseDirectory $nsisInstaller.Name
Copy-Item -LiteralPath $nsisInstaller.FullName -Destination $latestInstaller -Force
Copy-Item -LiteralPath $nsisInstaller.FullName -Destination $archivedInstaller -Force

$sourceFile = Get-Item -LiteralPath $nsisInstaller.FullName
$hash = (Get-FileHash -LiteralPath $nsisInstaller.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestAssets = @(
    [ordered]@{
        kind = "Recommended Windows installer"
        filename = $nsisInstaller.Name
        size = $sourceFile.Length
        sha256 = $hash
    }
)

$manifest = [ordered]@{
    product = "Mirid"
    version = $version
    exportedAt = (Get-Date).ToString("o")
    recommendedInstaller = $latestInstaller
    latestDirectory = $latestDirectory
    releaseDirectory = $releaseDirectory
    assets = $manifestAssets
}
$manifestJson = $manifest | ConvertTo-Json -Depth 5
$manifestJson | Set-Content -LiteralPath (Join-Path $latestDirectory "release-info.json") -Encoding utf8
$manifestJson | Set-Content -LiteralPath (Join-Path $releaseDirectory "release-info.json") -Encoding utf8

"$hash  $($nsisInstaller.Name)" |
    Set-Content -LiteralPath (Join-Path $releaseDirectory "SHA256SUMS.txt") -Encoding ascii

$freshTestHelper = Join-Path $root "scripts\prepare_fresh_windows_test.ps1"
$freshTestLauncher = Join-Path $DestinationRoot "PREPARE FRESH WINDOWS TEST.cmd"
@"
@echo off
title Prepare fresh Mirid release test
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$freshTestHelper"
if errorlevel 1 echo.&echo The fresh-account setup did not complete.
echo.
pause
"@ | Set-Content -LiteralPath $freshTestLauncher -Encoding ascii

Write-Host "Mirid installer exported."
Write-Host "Latest:   $latestInstaller"
Write-Host "Archived: $releaseDirectory"
Write-Host "Fresh QA: $freshTestLauncher"
