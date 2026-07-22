[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$LockFile = (Join-Path $PSScriptRoot "..\runtime\inference-wheels.lock.json"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\wheelhouse"),
    [string[]]$CudaArchitectures = @(),
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

function Assert-Command {
    param([string]$Name, [string]$Message)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw $Message
    }
}

function Assert-FileHash {
    param([string]$Path, [string]$ExpectedHash)
    $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $ExpectedHash.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $ExpectedHash, received $actualHash."
    }
}

if ($env:OS -ne "Windows_NT") {
    throw "Mirid CUDA wheels must be built on Windows."
}

$lockPath = (Resolve-Path -LiteralPath $LockFile).Path
$lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
if ($lock.schemaVersion -ne 1) {
    throw "Unsupported inference wheel lock schema: $($lock.schemaVersion)"
}

Assert-Command $Python "Python $($lock.python) is required."
Assert-Command "cmake" "CMake is required. Install it and ensure cmake.exe is on PATH."
Assert-Command "nvcc" "CUDA Toolkit $($lock.cuda) is required. Install it and ensure nvcc.exe is on PATH."
Assert-Command "git" "Git is required by native package build tooling."

$pythonVersion = (& $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne $lock.python) {
    throw "Python $($lock.python) is required; found $pythonVersion."
}

$nvccOutput = (& nvcc --version | Out-String)
if ($LASTEXITCODE -ne 0 -or $nvccOutput -notmatch "release\s+$([regex]::Escape($lock.cuda))([,\s])") {
    throw "CUDA Toolkit $($lock.cuda) is required. nvcc reported:`n$nvccOutput"
}

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
            $name = $line.Substring(0, $separator)
            $value = $line.Substring($separator + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }
} finally {
    Remove-Item -LiteralPath $environmentScript -Force -ErrorAction SilentlyContinue
}
if (-not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    throw "Visual Studio developer environment loaded without cl.exe on PATH."
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = Join-Path $repositoryRoot ".wheel-build"
$sourceDirectory = Join-Path $buildRoot "sources"
$venvDirectory = Join-Path $buildRoot "venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)

if (Test-Path -LiteralPath $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $sourceDirectory, $resolvedOutput -Force | Out-Null
Get-ChildItem -LiteralPath $resolvedOutput -Filter "*.whl" -File -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "Creating clean Python $pythonVersion build environment..."
Invoke-Checked $Python @("-m", "venv", $venvDirectory)
Invoke-Checked $venvPython @("-m", "pip", "install", "--upgrade", "pip", "build", "wheel", "ninja")

$selectedArchitectures = if ($CudaArchitectures.Count -gt 0) { $CudaArchitectures } else { @($lock.cudaArchitectures) }
$architectureList = $selectedArchitectures -join ";"
$builtPackages = @()

foreach ($package in $lock.packages) {
    $sourceName = [System.IO.Path]::GetFileName(([uri]$package.sourceUrl).AbsolutePath)
    $sourcePath = Join-Path $sourceDirectory $sourceName
    Write-Host "Downloading $($package.name) $($package.version)..."
    Invoke-WebRequest -Uri $package.sourceUrl -OutFile $sourcePath
    Assert-FileHash $sourcePath $package.sourceSha256

    $env:CMAKE_GENERATOR = "Ninja"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = [Environment]::ProcessorCount.ToString()
    $env:FORCE_CMAKE = "1"
    $env:CMAKE_ARGS = (@($package.cmakeArgs) + "-DCMAKE_CUDA_ARCHITECTURES=$architectureList") -join " "

    Write-Host "Building $($package.name) $($package.version) with CUDA $($lock.cuda)..."
    Write-Host "CMAKE_ARGS=$env:CMAKE_ARGS"
    Invoke-Checked $venvPython @(
        "-m", "pip", "wheel", $sourcePath,
        "--wheel-dir", $resolvedOutput,
        "--no-deps", "--no-cache-dir", "--verbose"
    )

    $wheelPrefix = $package.name.Replace("-", "_")
    $wheel = Get-ChildItem -LiteralPath $resolvedOutput -Filter "$wheelPrefix-$($package.version)-*.whl" -File |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $wheel) {
        throw "Build completed without producing a wheel for $($package.name)."
    }

    $builtPackages += [ordered]@{
        name = $package.name
        version = $package.version
        filename = $wheel.Name
        size = $wheel.Length
        sha256 = (Get-FileHash -LiteralPath $wheel.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        sourceUrl = $package.sourceUrl
        sourceSha256 = $package.sourceSha256
    }
}

Remove-Item Env:CMAKE_ARGS, Env:CMAKE_GENERATOR, Env:CMAKE_BUILD_PARALLEL_LEVEL, Env:FORCE_CMAKE -ErrorAction SilentlyContinue

if (-not $SkipImportChecks) {
    Write-Host "Installing the new wheels into the clean validation environment..."
    foreach ($package in $builtPackages) {
        Invoke-Checked $venvPython @("-m", "pip", "install", "--force-reinstall", (Join-Path $resolvedOutput $package.filename))
    }
    Invoke-Checked $venvPython @(
        "-c",
        "import llama_cpp, stable_diffusion_cpp; from llama_cpp import llama_cpp as api; assert api.llama_supports_gpu_offload(), 'llama.cpp reports no GPU offload support'; print('llama-cpp-python', llama_cpp.__version__); print('stable-diffusion-cpp-python import passed'); print('CUDA GPU offload compiled: yes')"
    )
}

$releaseManifest = [ordered]@{
    schemaVersion = 1
    createdAt = [DateTime]::UtcNow.ToString("o")
    python = $lock.python
    platform = $lock.platform
    cuda = $lock.cuda
    cudaArchitectures = @($selectedArchitectures)
    publishBaseUrl = $lock.publishBaseUrl
    packages = $builtPackages
}
$manifestPath = Join-Path $resolvedOutput "inference-wheels.manifest.json"
$releaseManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

if (-not $KeepBuildDirectory) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}

Write-Host "Mirid inference wheels are ready in $resolvedOutput"
Write-Host "Release manifest: $manifestPath"
