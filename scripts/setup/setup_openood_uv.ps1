param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..\..'),
    [string]$VenvPath = '.venv',
    [string]$PythonVersion = '',
    [string]$TorchBackend = 'auto',
    [string[]]$Datasets = @('imagenet-200'),
    [string[]]$Checkpoints = @('imagenet200_res18_v1.5'),
    [string]$DatasetMode = 'benchmark',
    [switch]$ForceRecreateVenv,
    [switch]$ForceReinstall,
    [switch]$SkipInstall,
    [switch]$SkipDownload,
    [switch]$SkipLibMR
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    Write-Host "$FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Resolve-AbsolutePath {
    param(
        [string]$BasePath,
        [string]$ChildPath
    )

    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $ChildPath))
}

function Normalize-ListArgument {
    param([string[]]$Values)

    $normalized = [System.Collections.Generic.List[string]]::new()
    foreach ($value in $Values) {
        foreach ($item in ($value -split ',')) {
            $trimmed = $item.Trim()
            if ($trimmed) {
                $normalized.Add($trimmed)
            }
        }
    }

    return $normalized.ToArray()
}

function Get-NvidiaCudaVersion {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return $null
    }

    try {
        $output = & $nvidiaSmi.Source 2>$null | Out-String
    }
    catch {
        return $null
    }

    $match = [regex]::Match($output, 'CUDA Version:\s*(?<version>\d+\.\d+)')
    if (-not $match.Success) {
        return $null
    }

    try {
        return [version]$match.Groups['version'].Value
    }
    catch {
        return $null
    }
}

function Resolve-TorchBackendProfile {
    param([string]$RequestedBackend = 'auto')

    $normalized = $RequestedBackend.Trim().ToLowerInvariant()
    if (-not $normalized) {
        $normalized = 'auto'
    }

    $reason = ''
    if ($normalized -eq 'auto') {
        $detectedCudaVersion = Get-NvidiaCudaVersion
        if ($null -eq $detectedCudaVersion) {
            $normalized = 'cpu'
            $reason = 'no NVIDIA CUDA runtime detected'
        }
        elseif ($detectedCudaVersion -ge [version]'12.8') {
            $normalized = 'cu128'
            $reason = "detected NVIDIA CUDA $detectedCudaVersion"
        }
        elseif ($detectedCudaVersion -ge [version]'12.6') {
            $normalized = 'cu126'
            $reason = "detected NVIDIA CUDA $detectedCudaVersion"
        }
        elseif ($detectedCudaVersion -ge [version]'11.8') {
            $normalized = 'cu118'
            $reason = "detected NVIDIA CUDA $detectedCudaVersion"
        }
        else {
            Write-Host "Detected NVIDIA CUDA $detectedCudaVersion but the scripted wheel mapping only covers cu118, cu126, and cu128. Falling back to CPU PyTorch." -ForegroundColor Yellow
            $normalized = 'cpu'
            $reason = "detected NVIDIA CUDA $detectedCudaVersion (unsupported for auto mapping)"
        }
    }
    else {
        $reason = 'user override'
    }

    $expectedCudaVersion = ''
    $expectCuda = $false
    switch ($normalized) {
        'cpu' {
        }
        'cu118' {
            $expectedCudaVersion = '11.8'
            $expectCuda = $true
        }
        'cu126' {
            $expectedCudaVersion = '12.6'
            $expectCuda = $true
        }
        'cu128' {
            $expectedCudaVersion = '12.8'
            $expectCuda = $true
        }
        default {
            throw "Unsupported TorchBackend '$RequestedBackend'. Supported values: auto, cpu, cu118, cu126, cu128."
        }
    }

    return @{
        Backend = $normalized
        Description = $reason
        ExpectCuda = $expectCuda
        ExpectedCudaVersion = $expectedCudaVersion
    }
}

function Invoke-ImportVerification {
    param(
        [string]$PythonExe,
        [string]$RepoPath,
        [hashtable]$TorchProfile
    )

    Write-Step 'Verifying key imports'
    $expectedCudaLiteral = if ($TorchProfile.ExpectCuda) { 'True' } else { 'False' }
    $verifyCode = @'
import importlib
modules = [
    "openood",
    "torch",
    "timm",
    "pyarrow",
    "statsmodels",
    "gdown",
    "cv2",
]
for name in modules:
    importlib.import_module(name)
import torch
import numpy
import setuptools
expected_backend = "__EXPECTED_BACKEND__"
expected_cuda = __EXPECTED_CUDA__
expected_cuda_version = "__EXPECTED_CUDA_VERSION__"
if expected_cuda:
    assert torch.version.cuda is not None, f"expected CUDA-backed torch for {expected_backend}, got {torch.__version__}"
    assert torch.version.cuda.startswith(expected_cuda_version), (torch.version.cuda, expected_cuda_version)
    assert torch.cuda.is_available(), "CUDA-backed torch is installed but torch.cuda.is_available() is False"
else:
    assert torch.version.cuda is None, f"expected CPU torch, got CUDA build {torch.version.cuda}"
assert int(numpy.__version__.split(".", 1)[0]) < 2, numpy.__version__
assert int(setuptools.__version__.split(".", 1)[0]) < 82, setuptools.__version__
print("import verification ok", numpy.__version__, setuptools.__version__, torch.__version__, torch.version.cuda or "cpu")
'@
    $verifyCode = $verifyCode.Replace('__EXPECTED_BACKEND__', $TorchProfile.Backend)
    $verifyCode = $verifyCode.Replace('__EXPECTED_CUDA__', $expectedCudaLiteral)
    $verifyCode = $verifyCode.Replace('__EXPECTED_CUDA_VERSION__', $TorchProfile.ExpectedCudaVersion)
    $tempVerify = Join-Path $RepoPath 'results\_setup_verify_imports.py'
    New-Item -ItemType Directory -Force -Path (Split-Path $tempVerify -Parent) | Out-Null
    Set-Content -LiteralPath $tempVerify -Value $verifyCode -Encoding UTF8
    try {
        Invoke-External -FilePath $PythonExe `
            -Arguments @($tempVerify) `
            -WorkingDirectory $RepoPath
    }
    finally {
        Remove-Item -LiteralPath $tempVerify -Force -ErrorAction SilentlyContinue
    }
}

function Resolve-UvPythonVersion {
    param([string]$RequestedVersion)

    if ($RequestedVersion) {
        return $RequestedVersion
    }

    return '3.12'
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    throw 'uv is required but was not found on PATH.'
}

$repoPath = [System.IO.Path]::GetFullPath($RepoRoot)
$venvAbsolutePath = Resolve-AbsolutePath -BasePath $repoPath -ChildPath $VenvPath
$reusedExistingVenv = (Test-Path $venvAbsolutePath) -and (-not $ForceRecreateVenv)
$pythonExe = Join-Path $venvAbsolutePath 'Scripts\python.exe'
$dataPath = Join-Path $repoPath 'data'
$resultsPath = Join-Path $repoPath 'results'
$downloadScript = Join-Path $repoPath 'scripts\download\download.py'

if (-not (Test-Path $downloadScript)) {
    throw "Download script not found: $downloadScript"
}

$Datasets = Normalize-ListArgument -Values $Datasets
$Checkpoints = Normalize-ListArgument -Values $Checkpoints
$torchProfile = Resolve-TorchBackendProfile -RequestedBackend $TorchBackend

if ($ForceRecreateVenv -and (Test-Path $venvAbsolutePath)) {
    Write-Step "Removing existing virtual environment at $venvAbsolutePath"
    Remove-Item -LiteralPath $venvAbsolutePath -Recurse -Force
}

if (-not (Test-Path $venvAbsolutePath)) {
    Write-Step 'Creating uv virtual environment'
    $resolvedPythonVersion = Resolve-UvPythonVersion -RequestedVersion $PythonVersion
    $venvArgs = @('venv')
    $venvArgs += @('--python', $resolvedPythonVersion)
    $venvArgs += $venvAbsolutePath
    Invoke-External -FilePath $uv.Source -Arguments $venvArgs -WorkingDirectory $repoPath
}
else {
    Write-Step "Using existing virtual environment at $venvAbsolutePath"
}

if (-not (Test-Path $pythonExe)) {
    throw "Expected virtual environment Python not found: $pythonExe"
}

if (-not $SkipInstall) {
    $shouldInstall = $true
    if ($reusedExistingVenv -and -not $ForceReinstall) {
        Write-Step 'Checking existing environment before reinstall'
        Write-Host "Expecting PyTorch backend $($torchProfile.Backend) ($($torchProfile.Description))." -ForegroundColor DarkGray
        try {
            Invoke-ImportVerification -PythonExe $pythonExe -RepoPath $repoPath -TorchProfile $torchProfile
            Write-Step 'Existing environment is valid; skipping dependency reinstall'
            $shouldInstall = $false
        }
        catch {
            Write-Host 'Existing environment verification failed; reinstalling dependencies.' -ForegroundColor Yellow
            $shouldInstall = $true
        }
    }

    if ($shouldInstall) {
    Write-Step 'Installing repo and required analysis dependencies with uv'
    Write-Host "Using PyTorch backend $($torchProfile.Backend) ($($torchProfile.Description))." -ForegroundColor DarkGray

    Invoke-External -FilePath $uv.Source `
        -Arguments @('pip', 'install', '--python', $pythonExe, 'numpy<2') `
        -WorkingDirectory $repoPath

    Invoke-External -FilePath $uv.Source `
        -Arguments @('pip', 'install', '--python', $pythonExe, '--torch-backend', $torchProfile.Backend, '--upgrade', '--reinstall-package', 'torch', '--reinstall-package', 'torchvision', 'torch', 'torchvision') `
        -WorkingDirectory $repoPath

    Invoke-External -FilePath $uv.Source `
        -Arguments @('pip', 'install', '--python', $pythonExe, '-e', $repoPath) `
        -WorkingDirectory $repoPath

    Invoke-External -FilePath $uv.Source `
        -Arguments @('pip', 'install', '--python', $pythonExe, 'numpy<2') `
        -WorkingDirectory $repoPath

    Invoke-External -FilePath $uv.Source `
        -Arguments @('pip', 'install', '--python', $pythonExe, 'setuptools<82') `
        -WorkingDirectory $repoPath

    Invoke-External -FilePath $uv.Source `
        -Arguments @('pip', 'install', '--python', $pythonExe, 'statsmodels', 'timm', 'pyarrow') `
        -WorkingDirectory $repoPath

    if (-not $SkipLibMR) {
        Invoke-External -FilePath $uv.Source `
            -Arguments @('pip', 'install', '--python', $pythonExe, '--no-build-isolation', 'libmr') `
            -WorkingDirectory $repoPath
    }

        Invoke-ImportVerification -PythonExe $pythonExe -RepoPath $repoPath -TorchProfile $torchProfile
    }
}
else {
    Write-Step 'Skipping dependency installation'
}

if (-not $SkipDownload) {
    $datasetList = $Datasets -join ', '
    $checkpointList = $Checkpoints -join ', '
    Write-Step "Downloading dataset bundles [$datasetList] and checkpoint bundles [$checkpointList]"
    New-Item -ItemType Directory -Force -Path $dataPath | Out-Null
    New-Item -ItemType Directory -Force -Path $resultsPath | Out-Null

    $downloadArgs = @(
        $downloadScript,
        '--contents', 'datasets', 'checkpoints',
        '--datasets'
    ) + $Datasets + @(
        '--checkpoints'
    ) + $Checkpoints + @(
        '--save_dir', '.\data', '.\results',
        '--dataset_mode', $DatasetMode
    )

    Invoke-External -FilePath $pythonExe `
        -Arguments $downloadArgs `
        -WorkingDirectory $repoPath
}
else {
    Write-Step 'Skipping dataset/checkpoint download'
}

Write-Step 'Setup complete'
Write-Host "Repository root: $repoPath"
Write-Host "Virtual env:      $venvAbsolutePath"
Write-Host "Data root:        $dataPath"
Write-Host "Results root:     $resultsPath"
