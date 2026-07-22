[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExpectedUser,
    [switch]$ConfirmReset
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$currentUser = [Environment]::UserName
$targets = @(
    (Join-Path $env:LOCALAPPDATA "Mirid"),
    (Join-Path $env:LOCALAPPDATA "ai.mirid.desktop"),
    (Join-Path $env:LOCALAPPDATA "com.eloquent.app"),
    (Join-Path $HOME ".LiangLocal")
)

if (-not $currentUser.Equals($ExpectedUser, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to reset '$currentUser'; this command was authorised only for '$ExpectedUser'."
}

if (-not $ConfirmReset) {
    Write-Host "Clean-install reset preview for Windows user '$currentUser':"
    $targets | ForEach-Object { Write-Host "- $_" }
    Write-Host "Run again with -ConfirmReset to remove these paths and force a complete first-user download."
    exit 0
}

Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc", "eloquent-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$uninstaller = Join-Path $env:LOCALAPPDATA "Mirid\uninstall.exe"
if (Test-Path -LiteralPath $uninstaller) {
    $process = Start-Process -FilePath $uninstaller -ArgumentList "/S" -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "Mirid uninstaller exited with code $($process.ExitCode)."
    }
}

foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

Write-Host "Mirid state removed for '$currentUser'. The next installation will download and extract everything from zero." -ForegroundColor Green
