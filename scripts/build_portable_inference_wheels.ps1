[CmdletBinding()]
param(
    [ValidateSet("cpu", "vulkan")]
    [string[]]$Backends = @("cpu"),
    [string]$Python = "python",
    [string]$LockFile = (Join-Path $PSScriptRoot "..\runtime\runtime-packages.lock.json"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\build\runtime-packages\repository"),
    [ValidateRange(1, 64)]
    [int]$ParallelJobs = 8,
    [switch]$KeepBuildDirectory,
    [switch]$SkipImportChecks
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

function Assert-FileHash {
    param([string]$Path, [string]$ExpectedHash)
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedHash.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $ExpectedHash, received $actual."
    }
}

function Enter-VisualStudioEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "Visual Studio 2022 Build Tools with Desktop development with C++ are required."
    }
    $visualStudio = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
    if (-not $visualStudio) {
        throw "Visual Studio 2022 Build Tools with the MSVC x64 toolchain are required."
    }
    $vsDevCmd = Join-Path $visualStudio "Common7\Tools\VsDevCmd.bat"
    $environmentScript = [System.IO.Path]::GetTempFileName() + ".cmd"
    try {
        @(
            "@call `"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64"
            "@if errorlevel 1 exit /b %errorlevel%"
            "@set"
        ) | Set-Content -LiteralPath $environmentScript -Encoding ascii
        $developerEnvironment = & $env:ComSpec /d /c $environmentScript
        if ($LASTEXITCODE -ne 0) {
            throw "Visual Studio developer environment initialisation failed."
        }
        foreach ($line in $developerEnvironment) {
            $separator = $line.IndexOf("=")
            if ($separator -gt 0) {
                Set-Item -Path "Env:$($line.Substring(0, $separator))" -Value $line.Substring($separator + 1)
            }
        }
    } finally {
        Remove-Item -LiteralPath $environmentScript -Force -ErrorAction SilentlyContinue
    }
}

if ($env:OS -ne "Windows_NT") {
    throw "This builder produces Windows x64 wheels. Use the platform workflow for Linux and Apple Silicon."
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$lock = Get-Content -LiteralPath (Resolve-Path $LockFile) -Raw | ConvertFrom-Json
$pythonVersion = (& $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne "3.12") {
    throw "Python 3.12 is required; found $pythonVersion."
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake is required."
}
Enter-VisualStudioEnvironment

$vulkanRoot = $env:VULKAN_SDK
if (-not $vulkanRoot -and (Test-Path -LiteralPath "C:\VulkanSDK")) {
    $vulkanRoot = Get-ChildItem -LiteralPath "C:\VulkanSDK" -Directory |
        Sort-Object { [version]$_.Name } -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if ($Backends -contains "vulkan") {
    if (-not $vulkanRoot -or -not (Test-Path -LiteralPath (Join-Path $vulkanRoot "Bin\glslc.exe"))) {
        throw "The Vulkan SDK is required for the Vulkan binding wheel."
    }
    $env:VULKAN_SDK = $vulkanRoot
    $env:Path = "$(Join-Path $vulkanRoot 'Bin');$env:Path"
}

$buildRoot = Join-Path $root ".portable-wheel-build"
$sourceRoot = Join-Path $buildRoot "sources"
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
$receiptPath = Join-Path $resolvedOutput "bindings\stable-diffusion-cpp-python\windows-x86_64\build-receipts.json"
if (Test-Path -LiteralPath $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $sourceRoot, $resolvedOutput -Force | Out-Null
$receipts = if (Test-Path -LiteralPath $receiptPath) {
    @(Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json)
} else {
    @()
}

foreach ($backend in $Backends) {
    $build = $lock.bindingBuilds | Where-Object {
        $_.platform -eq "windows-x86_64" -and $_.accelerator -eq $backend
    } | Select-Object -First 1
    if (-not $build) {
        throw "No Windows $backend binding recipe exists in the runtime package lock."
    }

    $sourcePath = Join-Path $sourceRoot $build.source.filename
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        Invoke-WebRequest -Uri $build.source.url -OutFile $sourcePath
    }
    Assert-FileHash $sourcePath $build.source.sha256

    $venv = Join-Path $buildRoot "venv-$backend"
    $venvPython = Join-Path $venv "Scripts\python.exe"
    $wheelDirectory = Join-Path $resolvedOutput "bindings\stable-diffusion-cpp-python\windows-x86_64\$backend"
    New-Item -ItemType Directory -Path $wheelDirectory -Force | Out-Null
    Get-ChildItem -LiteralPath $wheelDirectory -Filter "*.whl" -File -ErrorAction SilentlyContinue | Remove-Item -Force

    Invoke-Checked $Python @("-m", "venv", $venv)
    Invoke-Checked $venvPython @("-m", "pip", "install", "--upgrade", "pip", "wheel", "ninja")
    $env:CMAKE_GENERATOR = "Ninja"
    $backendJobs = if ($backend -eq "vulkan") { 1 } else { $ParallelJobs }
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $backendJobs.ToString()
    $env:FORCE_CMAKE = "1"
    $env:CMAKE_ARGS = @($build.cmakeArgs) -join " "
    Write-Host "Building stable-diffusion-cpp-python $($build.version) for $backend..."
    Write-Host "CMAKE_ARGS=$env:CMAKE_ARGS"
    Invoke-Checked $venvPython @(
        "-m", "pip", "wheel", $sourcePath,
        "--wheel-dir", $wheelDirectory,
        "--no-deps", "--no-cache-dir", "--verbose"
    )
    $wheel = Get-ChildItem -LiteralPath $wheelDirectory -Filter "*.whl" -File | Select-Object -First 1
    if (-not $wheel) {
        throw "The $backend build completed without producing a wheel."
    }
    if (-not $SkipImportChecks) {
        Invoke-Checked $venvPython @("-m", "pip", "install", "--force-reinstall", $wheel.FullName)
        Invoke-Checked $venvPython @("-c", "import stable_diffusion_cpp; print('stable-diffusion-cpp-python import passed')")
    }
    $receipt = [ordered]@{
        id = $build.id
        package = $build.package
        version = $build.version
        platform = $build.platform
        accelerator = $build.accelerator
        path = $wheel.FullName.Substring($resolvedOutput.TrimEnd("\").Length).TrimStart("\").Replace("\", "/")
        filename = $wheel.Name
        size = $wheel.Length
        sha256 = (Get-FileHash -LiteralPath $wheel.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        source = $build.source
        cmakeArgs = @($build.cmakeArgs)
        validation = if ($SkipImportChecks) { "build-only" } else { "import-passed" }
    }
    $receipts = @($receipts | Where-Object { $_.id -ne $receipt.id }) + $receipt
    $receiptJson = ConvertTo-Json -InputObject @($receipts) -Depth 8
    [System.IO.File]::WriteAllText(
        $receiptPath,
        $receiptJson + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

Remove-Item Env:CMAKE_ARGS, Env:CMAKE_GENERATOR, Env:CMAKE_BUILD_PARALLEL_LEVEL, Env:FORCE_CMAKE -ErrorAction SilentlyContinue
if (-not $KeepBuildDirectory) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
Write-Host "Portable Windows binding wheels are ready in $resolvedOutput"
