param(
    [ValidateSet("default", "full")]
    [string]$Profile = "default",
    [string]$Wheelhouse = (Join-Path $PSScriptRoot "..\wheelhouse"),
    [ValidateSet("cpu", "vulkan", "hip", "cuda12")]
    [string[]]$ModelRunnerBackends = @("cpu", "vulkan", "cuda12"),
    [switch]$SkipInferenceWheelInstall,
    [switch]$SkipModelRunnerStage,
    [switch]$SkipParakeetCppStage
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$distDir = Join-Path $root "build\sidecar-dist"
$builtDir = Join-Path $distDir "mirid-sidecar"
$specName = "mirid-sidecar-platform.spec"
$specPath = Join-Path $root $specName
$modelRunnerDirectory = Join-Path $root "build\model-runners"

if (-not $SkipModelRunnerStage) {
    & (Join-Path $PSScriptRoot "stage_model_runners.ps1") -Backends $ModelRunnerBackends
    if (-not $?) { throw "Model runner staging failed" }
}

if (-not $SkipInferenceWheelInstall) {
    $wheelManifestPath = Join-Path $Wheelhouse "inference-wheels.manifest.json"
    if (-not (Test-Path -LiteralPath $wheelManifestPath)) {
        Write-Host "Inference wheels are not staged; fetching the trusted Mirid release..."
        & (Join-Path $PSScriptRoot "fetch_inference_wheels.ps1") -OutputDirectory $Wheelhouse
        if (-not $?) { throw "Inference wheel download failed" }
    }
    $wheelManifest = Get-Content -LiteralPath $wheelManifestPath -Raw | ConvertFrom-Json
    $pythonVersion = (& $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($pythonVersion -ne $wheelManifest.python) {
        throw "Sidecar Python $pythonVersion does not match wheel ABI $($wheelManifest.python)."
    }
    foreach ($package in $wheelManifest.packages) {
        $wheelPath = Join-Path $Wheelhouse $package.filename
        if (-not (Test-Path -LiteralPath $wheelPath)) {
            throw "Locked inference wheel is missing: $wheelPath"
        }
        $actualHash = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $package.sha256) {
            throw "Inference wheel failed SHA-256 verification: $($package.filename)"
        }
        Write-Host "Installing verified $($package.name) $($package.version) into the sidecar environment..."
        & $python -m pip install --no-deps --force-reinstall $wheelPath
        if (-not $?) { throw "Failed to install $($package.name)" }
    }
    & $python -c "import llama_cpp, stable_diffusion_cpp; from llama_cpp import llama_cpp as api; assert api.llama_supports_gpu_offload(); print(f'CUDA inference wheels ready: llama-cpp-python {llama_cpp.__version__}')"
    if (-not $?) { throw "Inference wheel validation failed in the sidecar environment" }
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
Write-Host "Building Mirid sidecar profile: $Profile ($specName)"
$previousProfile = $env:MIRID_SIDECAR_PROFILE
$env:MIRID_SIDECAR_PROFILE = $Profile
try {
    & $python -m PyInstaller --noconfirm --distpath $distDir --workpath (Join-Path $root "build\pyinstaller") $specPath
} finally {
    $env:MIRID_SIDECAR_PROFILE = $previousProfile
}
if (-not $?) { throw "PyInstaller failed" }

if (-not $SkipParakeetCppStage) {
    $parakeetDirectory = Join-Path $builtDir "_internal\backend\parakeet_cpp"
    & (Join-Path $PSScriptRoot "stage_parakeet_cpp.ps1") -DestinationDirectory $parakeetDirectory
    if (-not $?) { throw "Parakeet.cpp staging failed" }
}

if (Test-Path -LiteralPath (Join-Path $modelRunnerDirectory "manifest.json") -PathType Leaf) {
    $frozenRunnerDirectory = Join-Path $builtDir "_internal\runners"
    New-Item -ItemType Directory -Force -Path $frozenRunnerDirectory | Out-Null
    Copy-Item -Path (Join-Path $modelRunnerDirectory "*") -Destination $frozenRunnerDirectory -Recurse -Force
} elseif (-not $SkipModelRunnerStage) {
    throw "Model runner manifest was not staged."
}

$internalDirectory = Join-Path $builtDir "_internal"
& $python (Join-Path $PSScriptRoot "assert_runtime_stage_safe.py") $internalDirectory
if (-not $?) { throw "Frozen runtime safety check failed" }

& $python (Join-Path $PSScriptRoot "assert_image_runtime_bundle.py") $internalDirectory
if (-not $?) { throw "Frozen image runtime dependency check failed" }

$savedCudaPath = $env:CUDA_PATH
$savedProcessPath = $env:PATH
try {
    Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
    $env:PATH = (($savedProcessPath -split ';') | Where-Object {
        $_ -and $_ -notmatch '(?i)CUDA|NVIDIA GPU Computing Toolkit'
    }) -join ';'
    & (Join-Path $builtDir "mirid-sidecar-x86_64-pc-windows-msvc.exe") "probe-image-runtime"
    if (-not $?) {
        throw "Frozen NVIDIA image runtime could not load without a system CUDA toolkit"
    }
} finally {
    $env:CUDA_PATH = $savedCudaPath
    $env:PATH = $savedProcessPath
}

# Tauri externalBin expects the binary in src-tauri/binaries with the target triple suffix.
$binDir = Join-Path $root "src-tauri\binaries"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$sidecarExe = Join-Path $builtDir "mirid-sidecar-x86_64-pc-windows-msvc.exe"
Copy-Item $sidecarExe (Join-Path $binDir "mirid-sidecar-x86_64-pc-windows-msvc.exe") -Force

# Stage the frozen runtime next to the Rust dev build so `tauri dev` can find _internal.
$debugInternal = Join-Path $root "src-tauri\target\debug\_internal"
if (Test-Path -LiteralPath $debugInternal) {
    Remove-Item -LiteralPath $debugInternal -Recurse -Force
}
Copy-Item (Join-Path $builtDir "_internal") $debugInternal -Recurse -Force

Write-Host "Sidecar built and staged. Exe -> src-tauri/binaries, runtime -> target/debug/_internal"
