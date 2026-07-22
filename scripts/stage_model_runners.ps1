[CmdletBinding()]
param(
    [ValidatePattern("^b[0-9]+$")]
    [string]$LlamaCppVersion = "b10068",
    [ValidateSet("cpu", "vulkan", "hip", "cuda12")]
    [string[]]$Backends = @("cpu", "vulkan", "hip", "cuda12"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build\model-runners"),
    [string]$CacheDirectory = (Join-Path $PSScriptRoot "..\build\downloads\llama.cpp")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$cache = [System.IO.Path]::GetFullPath($CacheDirectory)
$platformDirectory = Join-Path $output "windows-x86_64"
$releaseUrl = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$LlamaCppVersion"

$assetsByBackend = [ordered]@{
    cpu = @("llama-$LlamaCppVersion-bin-win-cpu-x64.zip")
    vulkan = @("llama-$LlamaCppVersion-bin-win-vulkan-x64.zip")
    hip = @("llama-$LlamaCppVersion-bin-win-hip-radeon-x64.zip")
    cuda12 = @(
        "llama-$LlamaCppVersion-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip"
    )
}

New-Item -ItemType Directory -Path $output, $cache, $platformDirectory -Force | Out-Null
Write-Host "Reading llama.cpp $LlamaCppVersion release metadata..."
$release = Invoke-RestMethod -Uri $releaseUrl -Headers @{ "User-Agent" = "Mirid-Release-Builder" }
$releaseAssets = @{}
foreach ($asset in $release.assets) {
    $releaseAssets[$asset.name] = $asset
}

$assetRecords = @()
foreach ($backend in $Backends) {
    $destination = Join-Path $platformDirectory $backend
    $resolvedDestination = [System.IO.Path]::GetFullPath($destination)
    if (-not $resolvedDestination.StartsWith($platformDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to stage outside the model runner directory: $resolvedDestination"
    }
    if (Test-Path -LiteralPath $resolvedDestination) {
        Remove-Item -LiteralPath $resolvedDestination -Recurse -Force
    }
    New-Item -ItemType Directory -Path $resolvedDestination -Force | Out-Null

    foreach ($assetName in $assetsByBackend[$backend]) {
        $asset = $releaseAssets[$assetName]
        if (-not $asset) {
            throw "llama.cpp $LlamaCppVersion does not contain $assetName"
        }
        $archive = Join-Path $cache $assetName
        $partial = "$archive.part"
        if (-not (Test-Path -LiteralPath $archive) -or (Get-Item -LiteralPath $archive).Length -ne $asset.size) {
            Write-Host "Downloading $assetName..."
            & curl.exe --fail --location --retry 5 --retry-all-errors --continue-at - --output $partial $asset.browser_download_url
            if (-not $?) { throw "Download failed: $($asset.browser_download_url)" }
            if ((Get-Item -LiteralPath $partial).Length -ne $asset.size) {
                throw "$assetName did not match the published size."
            }
            Move-Item -LiteralPath $partial -Destination $archive -Force
        }

        $extractDirectory = Join-Path $cache ([System.IO.Path]::GetFileNameWithoutExtension($assetName))
        if (Test-Path -LiteralPath $extractDirectory) {
            Remove-Item -LiteralPath $extractDirectory -Recurse -Force
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $extractDirectory -Force
        $server = Get-ChildItem -LiteralPath $extractDirectory -Filter "llama-server.exe" -Recurse | Select-Object -First 1
        if ($server) {
            Copy-Item -Path (Join-Path $server.Directory.FullName "*") -Destination $resolvedDestination -Recurse -Force
        } else {
            Get-ChildItem -LiteralPath $extractDirectory -File -Recurse | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $resolvedDestination $_.Name) -Force
            }
        }

        $assetRecords += [ordered]@{
            filename = $assetName
            size = [long]$asset.size
            sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
            sourceUrl = $asset.browser_download_url
        }
    }

    $serverPath = Join-Path $resolvedDestination "llama-server.exe"
    if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
        throw "$backend did not stage llama-server.exe"
    }
    Write-Host "Validating $backend runner..."
    & $serverPath --version | Out-Host
    if (-not $?) { throw "$backend runner failed its version check" }
}

$runnerManifest = Join-Path $root "runtime\model-runners.json"
Copy-Item -LiteralPath $runnerManifest -Destination (Join-Path $output "manifest.json") -Force
$lock = [ordered]@{
    schemaVersion = 1
    generatedAt = [DateTimeOffset]::UtcNow.ToString("o")
    llamaCppVersion = $LlamaCppVersion
    platform = "windows-x86_64"
    backends = $Backends
    assets = $assetRecords
}
$lock | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output "assets.lock.json") -Encoding utf8
Write-Host "Model runners staged at $output"
