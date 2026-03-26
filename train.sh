#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-li-model}"
unset CONDA_VERBOSITY CONDA_LOG_LEVEL

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

if ! conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Error: conda env '${ENV_NAME}' was not found. Run bash setup.sh first."
  exit 1
fi

cd "${ROOT_DIR}"
echo "Working directory: ${ROOT_DIR}"
echo "Activating ${ENV_NAME}..."

CONDA_BASE="$(conda info --base)"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Error: could not find conda.sh under ${CONDA_BASE}."
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"
set +u
conda activate "${ENV_NAME}"
set -u

export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

TRAIN_DIR="${TRAIN_DIR:-data/traj/train}"
TRAIN_FFIELD="${TRAIN_FFIELD:-ffield.json}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
TRAIN_LR="${TRAIN_LR:-0.0001}"
TRAIN_PR="${TRAIN_PR:-200}"
TRAIN_WRITELIB="${TRAIN_WRITELIB:-1000}"
TRAIN_BATCH="${TRAIN_BATCH:-50}"
TRAIN_BO="${TRAIN_BO:-0}"
TRAIN_H="${TRAIN_H:-0}"
TRAIN_A="${TRAIN_A:-0}"
TRAIN_T="${TRAIN_T:-0}"
TRAIN_F="${TRAIN_F:-0}"
TRAIN_VDW="${TRAIN_VDW:-1}"

python -u "${ROOT_DIR}/run_scripts/train_without_force_Li_NN.py" \
  --train_dir "${ROOT_DIR}/${TRAIN_DIR}" \
  --ffield "${ROOT_DIR}/${TRAIN_FFIELD}" \
  --s "${TRAIN_STEPS}" \
  --lr "${TRAIN_LR}" \
  --pr "${TRAIN_PR}" \
  --writelib "${TRAIN_WRITELIB}" \
  --batch "${TRAIN_BATCH}" \
  --bo "${TRAIN_BO}" \
  --h "${TRAIN_H}" \
  --a "${TRAIN_A}" \
  --t "${TRAIN_T}" \
  --f "${TRAIN_F}" \
  --vdw "${TRAIN_VDW}"
