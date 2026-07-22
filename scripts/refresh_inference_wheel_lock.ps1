[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$LockFile = (Join-Path $PSScriptRoot "..\runtime\inference-wheels.lock.json")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$lockPath = (Resolve-Path -LiteralPath $LockFile).Path
$lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
$updates = 0

foreach ($package in $lock.packages) {
    $metadata = Invoke-RestMethod "https://pypi.org/pypi/$($package.name)/json"
    $version = $metadata.info.version
    $source = $metadata.urls | Where-Object packagetype -eq "sdist" | Select-Object -First 1
    if (-not $source) {
        throw "PyPI did not publish a source archive for $($package.name) $version."
    }

    if ($package.version -ne $version) {
        Write-Host "$($package.name): $($package.version) -> $version"
        $updates++
    } else {
        Write-Host "$($package.name): $version is current"
    }
    $package.version = $version
    $package.sourceUrl = $source.url
    $package.sourceSha256 = $source.digests.sha256
}

if ($updates -eq 0) {
    Write-Host "Inference wheel lock is already current."
    return
}

if ($PSCmdlet.ShouldProcess($lockPath, "write $updates inference dependency update(s)")) {
    $lock | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $lockPath -Encoding utf8
    Write-Host "Updated $lockPath. Review upstream changes, then build and test the wheels."
}
