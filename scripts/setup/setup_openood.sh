#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_path=".venv"
python_version=""
torch_backend="auto"
datasets=("imagenet-200")
checkpoints=("imagenet200_res18_v1.5")
dataset_mode="benchmark"
installer="pip"
force_recreate_venv=0
force_reinstall=0
skip_install=0
skip_download=0
skip_libmr=0

write_step() {
  echo
  echo "==> $1"
}

run_cmd() {
  echo "$*" >&2
  "$@"
}

verify_imports() {
  write_step "Verifying key imports"
  mkdir -p "$results_path"
  local verify_script="$results_path/_setup_verify_imports.py"
  cat >"$verify_script" <<PY
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
expected_backend = "${torch_backend_resolved}"
expected_cuda = ${torch_expect_cuda}
expected_cuda_version = "${torch_expected_cuda_version}"
if expected_cuda:
    assert torch.version.cuda is not None, f"expected CUDA-backed torch for {expected_backend}, got {torch.__version__}"
    assert torch.version.cuda.startswith(expected_cuda_version), (torch.version.cuda, expected_cuda_version)
    assert torch.cuda.is_available(), "CUDA-backed torch is installed but torch.cuda.is_available() is False"
else:
    assert torch.version.cuda is None, f"expected CPU torch, got CUDA build {torch.version.cuda}"
assert int(numpy.__version__.split(".", 1)[0]) < 2, numpy.__version__
assert int(setuptools.__version__.split(".", 1)[0]) < 82, setuptools.__version__
print("import verification ok", numpy.__version__, setuptools.__version__, torch.__version__, torch.version.cuda or "cpu")
PY
  if ! run_cmd "$python_exe" "$verify_script"; then
    local status=$?
    rm -f "$verify_script"
    return "$status"
  fi
  rm -f "$verify_script"
}

version_ge() {
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" == "$1" ]]
}

get_nvidia_cuda_version() {
  local output
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  if ! output="$(nvidia-smi 2>/dev/null)"; then
    return 1
  fi
  if [[ "$output" =~ CUDA\ Version:\ ([0-9]+\.[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

resolve_torch_backend() {
  local requested="${1,,}"
  local detected_cuda=""
  local reason=""

  if [[ -z "$requested" ]]; then
    requested="auto"
  fi

  if [[ "$requested" == "auto" ]]; then
    if detected_cuda="$(get_nvidia_cuda_version)"; then
      if version_ge "$detected_cuda" "12.8"; then
        requested="cu128"
        reason="detected NVIDIA CUDA $detected_cuda"
      elif version_ge "$detected_cuda" "12.6"; then
        requested="cu126"
        reason="detected NVIDIA CUDA $detected_cuda"
      elif version_ge "$detected_cuda" "11.8"; then
        requested="cu118"
        reason="detected NVIDIA CUDA $detected_cuda"
      else
        echo "Detected NVIDIA CUDA $detected_cuda but the scripted wheel mapping only covers cu118, cu126, and cu128. Falling back to CPU PyTorch." >&2
        requested="cpu"
        reason="detected NVIDIA CUDA $detected_cuda (unsupported for auto mapping)"
      fi
    else
      requested="cpu"
      reason="no NVIDIA CUDA runtime detected"
    fi
  else
    reason="user override"
  fi

  case "$requested" in
    cpu)
      torch_backend_resolved="cpu"
      torch_index_url="https://download.pytorch.org/whl/cpu"
      torch_expect_cuda=0
      torch_expected_cuda_version=""
      ;;
    cu118)
      torch_backend_resolved="cu118"
      torch_index_url="https://download.pytorch.org/whl/cu118"
      torch_expect_cuda=1
      torch_expected_cuda_version="11.8"
      ;;
    cu126)
      torch_backend_resolved="cu126"
      torch_index_url="https://download.pytorch.org/whl/cu126"
      torch_expect_cuda=1
      torch_expected_cuda_version="12.6"
      ;;
    cu128)
      torch_backend_resolved="cu128"
      torch_index_url="https://download.pytorch.org/whl/cu128"
      torch_expect_cuda=1
      torch_expected_cuda_version="12.8"
      ;;
    *)
      echo "Unsupported --torch-backend '$1'. Use auto, cpu, cu118, cu126, or cu128." >&2
      exit 1
      ;;
  esac

  torch_backend_reason="$reason"
}

resolve_default_python() {
  local candidate
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

usage() {
  cat <<'EOF'
Usage: ./scripts/setup/setup_openood.sh [options]

Options:
  --repo-root PATH              Repository root. Defaults to the script's grandparent.
  --venv-path PATH              Virtual environment path relative to repo root. Default: .venv
  --python VERSION_OR_BIN       Python version for uv (e.g. 3.11) or python executable for pip.
  --torch-backend BACKEND       PyTorch backend: auto, cpu, cu118, cu126, cu128. Default: auto
  --datasets LIST               Comma-separated dataset bundles. Default: imagenet-200
  --checkpoints LIST            Comma-separated checkpoint bundles. Default: imagenet200_res18_v1.5
  --dataset-mode MODE           Download mode. Default: benchmark
  --installer uv|pip            Environment/install backend. Default: pip
  --force-recreate-venv         Delete and recreate the virtual environment.
  --force-reinstall             Reinstall repo dependencies even if the venv verifies cleanly.
  --skip-install                Skip dependency installation.
  --skip-download               Skip dataset/checkpoint download.
  --skip-libmr                  Skip libmr installation.
  -h, --help                    Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      repo_root="$(cd "$2" && pwd)"
      shift 2
      ;;
    --venv-path)
      venv_path="$2"
      shift 2
      ;;
    --python)
      python_version="$2"
      shift 2
      ;;
    --torch-backend)
      torch_backend="$2"
      shift 2
      ;;
    --datasets)
      IFS=',' read -r -a datasets <<< "$2"
      shift 2
      ;;
    --checkpoints)
      IFS=',' read -r -a checkpoints <<< "$2"
      shift 2
      ;;
    --dataset-mode)
      dataset_mode="$2"
      shift 2
      ;;
    --installer)
      installer="$2"
      shift 2
      ;;
    --force-recreate-venv)
      force_recreate_venv=1
      shift
      ;;
    --force-reinstall)
      force_reinstall=1
      shift
      ;;
    --skip-install)
      skip_install=1
      shift
      ;;
    --skip-download)
      skip_download=1
      shift
      ;;
    --skip-libmr)
      skip_libmr=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$installer" in
  uv|pip)
    ;;
  *)
    echo "Unsupported installer '$installer'. Use 'uv' or 'pip'." >&2
    exit 1
    ;;
esac

cd "$repo_root"

venv_abs="$repo_root/$venv_path"
python_exe="$venv_abs/bin/python"
pip_exe="$venv_abs/bin/pip"
data_path="$repo_root/data"
results_path="$repo_root/results"
download_script="$repo_root/scripts/download/download.py"

if [[ ! -f "$download_script" ]]; then
  echo "Download script not found: $download_script" >&2
  exit 1
fi

torch_backend_resolved=""
torch_index_url=""
torch_expect_cuda=0
torch_expected_cuda_version=""
torch_backend_reason=""
resolve_torch_backend "$torch_backend"

if [[ "$force_recreate_venv" -eq 1 && -d "$venv_abs" ]]; then
  write_step "Removing existing virtual environment at $venv_abs"
  rm -rf "$venv_abs"
fi

if [[ ! -d "$venv_abs" ]]; then
  if [[ "$installer" == "uv" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "uv is required for --installer uv but was not found on PATH." >&2
      exit 1
    fi

    write_step "Creating uv virtual environment"
    resolved_python="${python_version:-3.12}"
    run_cmd uv venv --python "$resolved_python" "$venv_abs"
  else
    bootstrap_python="${python_version:-$(resolve_default_python)}"
    if ! command -v "$bootstrap_python" >/dev/null 2>&1; then
      echo "Python executable '$bootstrap_python' was not found on PATH." >&2
      exit 1
    fi

    write_step "Creating virtual environment with stdlib venv"
    run_cmd "$bootstrap_python" -m venv "$venv_abs"
  fi
else
  write_step "Using existing virtual environment at $venv_abs"
fi

if [[ ! -x "$python_exe" ]]; then
  echo "Expected virtual environment Python not found: $python_exe" >&2
  exit 1
fi

if [[ ! -x "$pip_exe" ]]; then
  echo "Expected virtual environment pip not found: $pip_exe" >&2
  exit 1
fi

if [[ "$skip_install" -eq 0 ]]; then
  should_install=1
  if [[ -d "$venv_abs" && "$force_recreate_venv" -eq 0 && "$force_reinstall" -eq 0 ]]; then
    write_step "Checking existing environment before reinstall"
    echo "Expecting PyTorch backend $torch_backend_resolved ($torch_backend_reason)." >&2
    if verify_imports; then
      write_step "Existing environment is valid; skipping dependency reinstall"
      should_install=0
    else
      echo "Existing environment verification failed; reinstalling dependencies." >&2
      should_install=1
    fi
  fi

  if [[ "$should_install" -eq 1 ]]; then
    if [[ "$installer" == "uv" ]]; then
      write_step "Installing repo and required analysis dependencies with uv"
      echo "Using PyTorch backend $torch_backend_resolved ($torch_backend_reason)." >&2
      run_cmd uv pip install --python "$python_exe" "numpy<2"
      run_cmd uv pip install --python "$python_exe" --torch-backend "$torch_backend_resolved" --upgrade --reinstall-package torch --reinstall-package torchvision torch torchvision
      run_cmd uv pip install --python "$python_exe" -e "$repo_root"
      run_cmd uv pip install --python "$python_exe" "numpy<2"
      run_cmd uv pip install --python "$python_exe" "setuptools<82"
      run_cmd uv pip install --python "$python_exe" statsmodels timm pyarrow
      if [[ "$skip_libmr" -eq 0 ]]; then
        run_cmd uv pip install --python "$python_exe" --no-build-isolation libmr
      fi
    else
      write_step "Installing repo and required analysis dependencies with pip"
      echo "Using PyTorch backend $torch_backend_resolved ($torch_backend_reason)." >&2
      run_cmd "$python_exe" -m pip install --upgrade pip wheel "setuptools<82"
      run_cmd "$pip_exe" install "numpy<2"
      run_cmd "$pip_exe" install --upgrade --force-reinstall --index-url "$torch_index_url" torch torchvision
      run_cmd "$pip_exe" install -e "$repo_root"
      run_cmd "$pip_exe" install "numpy<2"
      run_cmd "$pip_exe" install statsmodels timm pyarrow
      if [[ "$skip_libmr" -eq 0 ]]; then
        run_cmd "$pip_exe" install --no-build-isolation libmr
      fi
    fi

    verify_imports
  fi
else
  write_step "Skipping dependency installation"
fi

if [[ "$skip_download" -eq 0 ]]; then
  dataset_list="$(IFS=', '; echo "${datasets[*]}")"
  checkpoint_list="$(IFS=', '; echo "${checkpoints[*]}")"
  write_step "Downloading dataset bundles [$dataset_list] and checkpoint bundles [$checkpoint_list]"
  mkdir -p "$data_path" "$results_path"
  run_cmd "$python_exe" "$download_script" \
    --contents datasets checkpoints \
    --datasets "${datasets[@]}" \
    --checkpoints "${checkpoints[@]}" \
    --save_dir ./data ./results \
    --dataset_mode "$dataset_mode"
else
  write_step "Skipping dataset/checkpoint download"
fi

write_step "Setup complete"
echo "Repository root: $repo_root"
echo "Virtual env:      $venv_abs"
echo "Data root:        $data_path"
echo "Results root:     $results_path"
