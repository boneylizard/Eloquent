[CmdletBinding()]
param(
    [string]$Installer = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Mirid Installers\Latest\Mirid-Setup.exe"),
    [string]$Repository = "boneylizardwizard/mirid-runtime",
    [switch]$Publish,
    [switch]$AllowUnsigned
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$installerPath = (Resolve-Path -LiteralPath $Installer).Path
$tauriConfig = Get-Content -LiteralPath (Join-Path $root "src-tauri\tauri.conf.json") -Raw | ConvertFrom-Json
$runtimeManifest = Get-Content -LiteralPath (Join-Path $root "runtime\runtime-release.json") -Raw | ConvertFrom-Json
$version = [string]$tauriConfig.version
$versionedFilename = "Mirid_${version}_x64-setup.exe"
$remoteDirectory = "releases/v$version"
$remoteUrl = "https://huggingface.co/$Repository/resolve/main/$remoteDirectory/${versionedFilename}?download=true"
$installerFile = Get-Item -LiteralPath $installerPath
$installerHash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
$signature = Get-AuthenticodeSignature -LiteralPath $installerPath

if ($installerFile.VersionInfo.ProductVersion -ne $version) {
    throw "Installer version $($installerFile.VersionInfo.ProductVersion) does not match Tauri version $version."
}

$desktopReleaseInfoPath = Join-Path (Split-Path -Parent $installerPath) "release-info.json"
if (-not (Test-Path -LiteralPath $desktopReleaseInfoPath)) {
    throw "Installer release metadata is missing: $desktopReleaseInfoPath"
}

$desktopReleaseInfo = Get-Content -LiteralPath $desktopReleaseInfoPath -Raw | ConvertFrom-Json
$desktopAsset = $desktopReleaseInfo.assets | Select-Object -First 1
if ($desktopReleaseInfo.version -ne $version -or $desktopAsset.sha256 -ne $installerHash -or [long]$desktopAsset.size -ne $installerFile.Length) {
    throw "Desktop release metadata does not match the installer. Run scripts\export_installers.ps1 again."
}

$stagingDirectory = Join-Path $root "build\installer-release\v$version"
New-Item -ItemType Directory -Path $stagingDirectory -Force | Out-Null
$stagedInstaller = Join-Path $stagingDirectory $versionedFilename
Copy-Item -LiteralPath $installerPath -Destination $stagedInstaller -Force

$releaseMetadata = [ordered]@{
    product = "Mirid"
    version = $version
    platform = "windows-x86_64"
    published = (Get-Date).ToString("yyyy-MM-dd")
    runtime = [ordered]@{
        version = [string]$runtimeManifest.runtimeVersion
        downloadBytes = [long]$runtimeManifest.assets.runtimeArchive.size
        sha256 = [string]$runtimeManifest.assets.runtimeArchive.sha256
    }
    installers = @(
        [ordered]@{
            format = "exe"
            filename = $versionedFilename
            bytes = $installerFile.Length
            sha256 = $installerHash
            url = $remoteUrl
        }
    )
}

$releaseJson = $releaseMetadata | ConvertTo-Json -Depth 6
$stagedReleaseJson = Join-Path $stagingDirectory "release.json"
$stagedChecksums = Join-Path $stagingDirectory "SHA256SUMS.txt"
$releaseJson | Set-Content -LiteralPath $stagedReleaseJson -Encoding utf8
"$installerHash  $versionedFilename" | Set-Content -LiteralPath $stagedChecksums -Encoding ascii

$plan = [pscustomobject]@{
    Version = $version
    Installer = $installerPath
    Size = $installerFile.Length
    SHA256 = $installerHash
    Signature = [string]$signature.Status
    Repository = $Repository
    RemoteDirectory = $remoteDirectory
    RemoteUrl = $remoteUrl
    StagingDirectory = $stagingDirectory
    Action = if ($Publish) { "Publish, download, and verify the installer" } else { "Staged only; nothing was uploaded or advertised" }
}
$plan | Format-List

if (-not $Publish) {
    exit 0
}

if ($signature.Status -ne "Valid" -and -not $AllowUnsigned) {
    throw "The installer is $($signature.Status). Pass -AllowUnsigned only after consciously accepting the Windows unknown-publisher warning."
}

& hf auth whoami | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Hugging Face authentication failed."
}

& hf upload $Repository $stagingDirectory $remoteDirectory --repo-type model --commit-message "Publish Mirid $version Windows installer" | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Hugging Face upload failed."
}

$verificationDirectory = Join-Path $root "build\installer-release\verification"
New-Item -ItemType Directory -Path $verificationDirectory -Force | Out-Null
$downloadedInstaller = Join-Path $verificationDirectory $versionedFilename
Invoke-WebRequest -Uri $remoteUrl -OutFile $downloadedInstaller -UseBasicParsing
$downloadedFile = Get-Item -LiteralPath $downloadedInstaller
$downloadedHash = (Get-FileHash -LiteralPath $downloadedInstaller -Algorithm SHA256).Hash.ToLowerInvariant()
if ($downloadedFile.Length -ne $installerFile.Length -or $downloadedHash -ne $installerHash) {
    throw "The published installer failed remote verification."
}

Write-Host ""
Write-Host "Mirid $version is published and remotely verified." -ForegroundColor Green
Write-Host "Download: $remoteUrl"
