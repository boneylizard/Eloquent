[CmdletBinding()]
param(
    [ValidatePattern("^[A-Za-z0-9_.-]{1,20}$")]
    [string]$AccountName = "MiridReleaseTest",
    [string]$Password = "MiridFresh!2026",
    [string]$Installer = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Mirid Installers\Latest\Mirid-Setup.exe"),
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

$installerPath = (Resolve-Path -LiteralPath $Installer).Path

if ($PlanOnly) {
    [pscustomobject]@{
        Account = $AccountName
        AccountType = "Standard local Windows user"
        Installer = $installerPath
        SharedInstaller = (Join-Path $env:PUBLIC "Documents\Mirid Release Test\Mirid-Setup.exe")
        MiridWillLaunch = $false
        Action = "Preview only; no account or file was changed."
    } | Format-List
    exit 0
}

if (-not (Test-IsAdministrator)) {
    $scriptPath = $PSCommandPath.Replace("'", "''")
    $accountArgument = $AccountName.Replace("'", "''")
    $passwordArgument = $Password.Replace("'", "''")
    $installerArgument = $installerPath.Replace("'", "''")
    $command = "& '$scriptPath' -AccountName '$accountArgument' -Password '$passwordArgument' -Installer '$installerArgument'"
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $process = Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile", "-EncodedCommand", $encodedCommand -PassThru -Wait
    exit $process.ExitCode
}

if (Get-LocalUser -Name $AccountName -ErrorAction SilentlyContinue) {
    throw "The Windows account '$AccountName' already exists. Run this script again with a new -AccountName so the test remains clean."
}

$securePassword = ConvertTo-SecureString $Password -AsPlainText -Force
$testUser = New-LocalUser `
    -Name $AccountName `
    -Password $securePassword `
    -AccountNeverExpires `
    -PasswordNeverExpires `
    -UserMayNotChangePassword `
    -Description "Disposable standard account for Mirid release testing"

$usersGroup = Get-LocalGroup -SID "S-1-5-32-545"
$isUsersMember = Get-LocalGroupMember -Group $usersGroup -ErrorAction SilentlyContinue |
    Where-Object { $_.SID -eq $testUser.SID }
if (-not $isUsersMember) {
    Add-LocalGroupMember -Group $usersGroup -Member $testUser
}

$sharedDirectory = Join-Path $env:PUBLIC "Documents\Mirid Release Test"
New-Item -ItemType Directory -Path $sharedDirectory -Force | Out-Null
$sharedInstaller = Join-Path $sharedDirectory "Mirid-Setup.exe"
Copy-Item -LiteralPath $installerPath -Destination $sharedInstaller -Force

$collectorSource = Join-Path $PSScriptRoot "collect_fresh_windows_test.ps1"
$collectorDestination = Join-Path $sharedDirectory "collect_fresh_windows_test.ps1"
Copy-Item -LiteralPath $collectorSource -Destination $collectorDestination -Force
$collectorLauncher = Join-Path $sharedDirectory "COLLECT FRESH TEST RESULT.cmd"
@"
@echo off
title Collect Mirid fresh-account test result
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0collect_fresh_windows_test.ps1"
echo.
pause
"@ | Set-Content -LiteralPath $collectorLauncher -Encoding ascii

$testSteps = Join-Path $sharedDirectory "TEST STEPS.txt"
@"
Mirid portable first release test

1. Run Mirid-Setup.exe.
2. Complete the first download and extraction.
3. Wait until Mirid opens and the local services are ready.
4. Test Shutdown, then open Mirid again and test Restart.
5. Close Mirid.
6. Run COLLECT FRESH TEST RESULT.cmd.

The collector does not launch or stop Mirid. It writes fresh-test-result.json in this folder.
"@ | Set-Content -LiteralPath $testSteps -Encoding utf8

$sourceHash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash
$sharedHash = (Get-FileHash -LiteralPath $sharedInstaller -Algorithm SHA256).Hash
if ($sourceHash -ne $sharedHash) {
    throw "The shared installer copy failed verification."
}

$releaseDirectory = Split-Path -Parent (Split-Path -Parent $installerPath)
$loginDetails = Join-Path $releaseDirectory "FRESH ACCOUNT LOGIN.txt"
@"
Mirid fresh-account release test

Windows account: $AccountName
Password: $Password

1. Sign out of Windows. Do not uninstall or launch Mirid in the current account.
2. Sign in as $AccountName.
3. Open C:\Users\Public\Documents\Mirid Release Test.
4. Run Mirid-Setup.exe and complete the first download and extraction.
5. Follow TEST STEPS.txt and run COLLECT FRESH TEST RESULT.cmd when finished.

Installer SHA-256: $sharedHash
"@ | Set-Content -LiteralPath $loginDetails -Encoding utf8

Write-Host ""
Write-Host "Fresh Mirid test account created." -ForegroundColor Green
Write-Host "Account:   $AccountName"
Write-Host "Password:  $Password"
Write-Host "Installer: $sharedInstaller"
Write-Host "Handoff:   $loginDetails"
Write-Host ""
Write-Host "Sign out of Windows, choose $AccountName, then run Mirid-Setup.exe from the shared folder above."
Write-Host "This script has not installed or launched Mirid."
