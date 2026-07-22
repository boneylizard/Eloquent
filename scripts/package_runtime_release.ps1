[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern("^v[0-9]+(?:\.[0-9]+){0,2}$")]
    [string]$RuntimeVersion,
    [ValidateSet("default", "full")]
    [string]$Profile = "default",
    [string]$BaseUrl = "https://huggingface.co/boneylizardwizard/mirid-runtime/resolve/main",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build\runtime-release"),
    [string]$SevenZip,
    [switch]$SkipSidecarBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sidecarDirectory = Join-Path $root "build\sidecar-dist\mirid-sidecar"
$internalDirectory = Join-Path $sidecarDirectory "_internal"
$sidecarName = "mirid-sidecar-x86_64-pc-windows-msvc.exe"
$archiveName = "mirid-runtime-windows-x64.7z"
$sidecarPath = Join-Path $sidecarDirectory $sidecarName
$releaseDirectory = Join-Path ([System.IO.Path]::GetFullPath($OutputDirectory)) $RuntimeVersion

if (-not $SkipSidecarBuild) {
    & (Join-Path $PSScriptRoot "build_sidecar.ps1") -Profile $Profile
    if (-not $?) { throw "Sidecar build failed" }
}
if (-not (Test-Path -LiteralPath $internalDirectory -PathType Container)) {
    throw "Frozen runtime is missing: $internalDirectory"
}
if (-not (Test-Path -LiteralPath (Join-Path $internalDirectory "base_library.zip") -PathType Leaf)) {
    throw "Frozen runtime does not contain the expected Python standard library."
}
if (-not (Test-Path -LiteralPath (Join-Path $internalDirectory "runners\manifest.json") -PathType Leaf)) {
    throw "Frozen runtime does not contain the staged model runners."
}
if (-not (Test-Path -LiteralPath (Join-Path $internalDirectory "backend\parakeet_cpp\parakeet-cli.exe") -PathType Leaf)) {
    throw "Frozen runtime does not contain the Parakeet.cpp speech-to-text runner."
}
if (-not (Test-Path -LiteralPath $sidecarPath -PathType Leaf)) {
    throw "Frozen sidecar executable is missing: $sidecarPath"
}

& (Join-Path $root "venv\Scripts\python.exe") (Join-Path $PSScriptRoot "assert_runtime_stage_safe.py") $internalDirectory
if (-not $?) { throw "Runtime stage safety check failed" }

if (-not $SevenZip) {
    $sevenZipCommand = Get-Command "7z.exe" -ErrorAction SilentlyContinue
    if ($sevenZipCommand) {
        $SevenZip = $sevenZipCommand.Source
    } else {
        $SevenZip = Join-Path $env:ProgramFiles "7-Zip\7z.exe"
    }
}
if (-not (Test-Path -LiteralPath $SevenZip -PathType Leaf)) {
    throw "7-Zip is required on the release workstation to create the runtime archive."
}

if (Test-Path -LiteralPath $releaseDirectory) {
    Remove-Item -LiteralPath $releaseDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $releaseDirectory -Force | Out-Null
$releaseArchive = Join-Path $releaseDirectory $archiveName
$releaseSidecar = Join-Path $releaseDirectory $sidecarName

Write-Host "Compressing the Mirid $RuntimeVersion runtime..."
Push-Location $internalDirectory
try {
    & $SevenZip a -t7z -mx=9 -m0=lzma2 -ms=on $releaseArchive ".\*"
    if (-not $?) { throw "Runtime compression failed" }
} finally {
    Pop-Location
}
Copy-Item -LiteralPath $sidecarPath -Destination $releaseSidecar -Force

$internalSize = [int64]((Get-ChildItem -LiteralPath $internalDirectory -Recurse -File | Measure-Object -Property Length -Sum).Sum)
$installedSize = $internalSize + [int64](Get-Item -LiteralPath $sidecarPath).Length

function Get-AssetRecord {
    param([string]$Path)
    $file = Get-Item -LiteralPath $Path
    return [ordered]@{
        filename = $file.Name
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$release = [ordered]@{
    schemaVersion = 1
    modelRunnerContractVersion = 1
    channel = "stable"
    runtimeVersion = $RuntimeVersion
    installedSize = $installedSize
    baseUrl = $BaseUrl.TrimEnd("/")
    assets = [ordered]@{
        runtimeArchive = Get-AssetRecord $releaseArchive
        sidecarExecutable = Get-AssetRecord $releaseSidecar
    }
}
$releaseManifestPath = Join-Path $root "runtime\runtime-release.json"
$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
$releaseJson = $release | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($releaseManifestPath, $releaseJson, $utf8WithoutBom)
Copy-Item -LiteralPath $releaseManifestPath -Destination (Join-Path $releaseDirectory "runtime-release.json") -Force

$rustPath = Join-Path $root "src-tauri\src\runtime_windows.rs"
$rust = Get-Content -LiteralPath $rustPath -Raw
$rust = [regex]::Replace($rust, 'const RUNTIME_VERSION: &str = "[^"]+";', "const RUNTIME_VERSION: &str = `"$RuntimeVersion`";")
$rust = [regex]::Replace($rust, 'const HF_BASE: &str = "[^"]+";', "const HF_BASE: &str = `"$($release.baseUrl)`";")
$rust = [regex]::Replace($rust, 'const RUNTIME_ARCHIVE_SIZE: u64 = [\d_]+;', "const RUNTIME_ARCHIVE_SIZE: u64 = $($release.assets.runtimeArchive.size);")
$rust = [regex]::Replace($rust, 'const RUNTIME_INSTALLED_SIZE: u64 = [\d_]+;', "const RUNTIME_INSTALLED_SIZE: u64 = $($release.installedSize);")
$rust = [regex]::Replace($rust, 'const SIDECAR_EXE_SIZE: u64 = [\d_]+;', "const SIDECAR_EXE_SIZE: u64 = $($release.assets.sidecarExecutable.size);")
$rust = [regex]::Replace($rust, 'const RUNTIME_ARCHIVE_SHA256: &str =\s*"[a-f0-9]+";', "const RUNTIME_ARCHIVE_SHA256: &str =`n    `"$($release.assets.runtimeArchive.sha256)`";")
$rust = [regex]::Replace($rust, 'const SIDECAR_EXE_SHA256: &str =\s*"[a-f0-9]+";', "const SIDECAR_EXE_SHA256: &str =`n    `"$($release.assets.sidecarExecutable.sha256)`";")
[System.IO.File]::WriteAllText($rustPath, $rust, $utf8WithoutBom)

Write-Host "Runtime release staged at $releaseDirectory"
Write-Host "Upload both assets and runtime-release.json, then run npm run release:check:hosted-runtime."
