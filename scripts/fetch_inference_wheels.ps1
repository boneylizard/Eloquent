[CmdletBinding()]
param(
    [string]$ReleaseManifest = (Join-Path $PSScriptRoot "..\runtime\inference-wheels.release.json"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\wheelhouse")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$releasePath = (Resolve-Path -LiteralPath $ReleaseManifest).Path
$release = Get-Content -LiteralPath $releasePath -Raw | ConvertFrom-Json
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $output -Force | Out-Null

foreach ($package in $release.packages) {
    $destination = Join-Path $output $package.filename
    $valid = $false
    if (Test-Path -LiteralPath $destination) {
        $file = Get-Item -LiteralPath $destination
        if ($file.Length -eq $package.size) {
            $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
            $valid = $hash -eq $package.sha256
        }
    }
    if ($valid) {
        Write-Host "$($package.name) $($package.version) is already downloaded and verified."
        continue
    }

    $partial = "$destination.part"
    $url = "$($release.publishBaseUrl.TrimEnd('/'))/$($package.filename)"
    Write-Host "Downloading $($package.name) $($package.version)..."
    & curl.exe --fail --location --retry 5 --retry-all-errors --continue-at - --output $partial $url
    if (-not $?) { throw "Download failed: $url" }
    $partialFile = Get-Item -LiteralPath $partial
    if ($partialFile.Length -ne $package.size) {
        throw "$($package.filename) has size $($partialFile.Length); expected $($package.size)."
    }
    $hash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $package.sha256) {
        throw "$($package.filename) failed SHA-256 verification."
    }
    Move-Item -LiteralPath $partial -Destination $destination -Force
}

Copy-Item -LiteralPath $releasePath -Destination (Join-Path $output "inference-wheels.manifest.json") -Force
Write-Host "Verified Mirid inference wheels are ready in $output"
