#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="li-model"
KERNEL="$(uname -s)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

echo "Kernel detected: ${KERNEL}"
echo "Workspace root: ${ROOT_DIR}"

have_pip_pkg() {
  local pkg="$1"
  conda run -n "${ENV_NAME}" python -m pip show "${pkg}" >/dev/null 2>&1
}

install_if_missing() {
  local pkg="$1"
  if have_pip_pkg "${pkg}"; then
    echo "Package already present: ${pkg}"
  else
    echo "Installing missing package: ${pkg}"
    conda run -n "${ENV_NAME}" pip install "${pkg}"
  fi
}

if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Reusing it."
else
  echo "Creating conda env '${ENV_NAME}' with Python 3.10..."
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

if conda run -n "${ENV_NAME}" python -m pip show irff >/dev/null 2>&1; then
  EDITABLE_LOC="$(conda run -n "${ENV_NAME}" python -m pip show irff | awk -F': ' '/Editable project location/ {print $2}')"
  if [[ "${EDITABLE_LOC}" == "${ROOT_DIR}/I-ReaxFF" ]]; then
    echo "Editable IRFF already points to ${ROOT_DIR}/I-ReaxFF"
  else
    echo "Reinstalling editable IRFF from ${ROOT_DIR}/I-ReaxFF"
    conda run -n "${ENV_NAME}" pip install -e "${ROOT_DIR}/I-ReaxFF"
  fi
else
  echo "Installing editable IRFF..."
  conda run -n "${ENV_NAME}" pip install -e "${ROOT_DIR}/I-ReaxFF"
fi

echo "Checking core Python dependencies..."
install_if_missing "numpy"
install_if_missing "ase"
install_if_missing "pandas"
install_if_missing "argh"
install_if_missing "matplotlib"
install_if_missing "cython"

if [[ "${KERNEL}" == "Darwin" ]]; then
  echo "Checking macOS TensorFlow stack..."
  install_if_missing "tensorflow"
  install_if_missing "tensorflow-metal"
elif [[ "${KERNEL}" == "Linux" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if have_pip_pkg "tensorflow" && have_pip_pkg "nvidia-cublas-cu12"; then
      echo "Linux NVIDIA TensorFlow stack already present."
    else
      echo "NVIDIA GPU detected. Installing TensorFlow with CUDA extras..."
      conda run -n "${ENV_NAME}" pip install --extra-index-url https://pypi.nvidia.com "tensorflow[and-cuda]"
    fi
  else
    echo "Checking Linux CPU TensorFlow..."
    install_if_missing "tensorflow"
  fi
else
  echo "Unsupported kernel '${KERNEL}'. Install TensorFlow manually for this platform."
  exit 1
fi

echo
echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Optional (for matplotlib cache permissions):"
echo "  export MPLCONFIGDIR=\"${ROOT_DIR}/.mplconfig\""
