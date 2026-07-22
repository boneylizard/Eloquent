[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$DestinationDirectory,
    [string]$CacheDirectory = (Join-Path $PSScriptRoot "..\build\native-dependencies\parakeet-cpp")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$version = "v0.4.0"
$assetName = "parakeet-v0.4.0-bin-win-cpu-x64.zip"
$assetUrl = "https://github.com/mudler/parakeet.cpp/releases/download/$version/$assetName"
$assetSha256 = "2880150a1bad2944baed46f2e6bb9f1bc55263a9f2bb85573785a7ec4fa35f27"
$cacheRoot = [System.IO.Path]::GetFullPath($CacheDirectory)
$versionDirectory = Join-Path $cacheRoot $version
$archivePath = Join-Path $versionDirectory $assetName
$expandedDirectory = Join-Path $versionDirectory "expanded"
$bundleDirectory = Join-Path $expandedDirectory "parakeet-v0.4.0-bin-win-cpu-x64"
$sourceExecutable = Join-Path $bundleDirectory "parakeet-cli.exe"
$destination = [System.IO.Path]::GetFullPath($DestinationDirectory)

New-Item -ItemType Directory -Path $versionDirectory -Force | Out-Null

$downloadRequired = -not (Test-Path -LiteralPath $archivePath -PathType Leaf)
if (-not $downloadRequired) {
    $cachedHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $downloadRequired = $cachedHash -ne $assetSha256
}

if ($downloadRequired) {
    Write-Host "Downloading verified Parakeet.cpp $version Windows CPU runtime..."
    Invoke-WebRequest -Uri $assetUrl -OutFile $archivePath
}

$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $assetSha256) {
    throw "Parakeet.cpp archive failed SHA-256 verification."
}

if ($downloadRequired -or -not (Test-Path -LiteralPath $sourceExecutable -PathType Leaf)) {
    New-Item -ItemType Directory -Path $expandedDirectory -Force | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $expandedDirectory -Force
}
if (-not (Test-Path -LiteralPath $sourceExecutable -PathType Leaf)) {
    throw "Parakeet.cpp archive did not contain parakeet-cli.exe."
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null
Copy-Item -LiteralPath $sourceExecutable -Destination (Join-Path $destination "parakeet-cli.exe") -Force
Copy-Item -LiteralPath (Join-Path $bundleDirectory "LICENSE") -Destination (Join-Path $destination "LICENSE-parakeet.cpp.txt") -Force

$manifest = [ordered]@{
    name = "parakeet.cpp"
    version = $version
    source = "https://github.com/mudler/parakeet.cpp"
    asset = $assetName
    assetSha256 = $assetSha256
    executableSha256 = (Get-FileHash -LiteralPath $sourceExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
}
$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText(
    (Join-Path $destination "manifest.json"),
    ($manifest | ConvertTo-Json -Depth 4),
    $utf8WithoutBom
)

Write-Host "Parakeet.cpp $version staged at $destination"
