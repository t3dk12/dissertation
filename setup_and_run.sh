#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${ROOT_DIR}/setup.sh"

if [[ ! -x "${SETUP_SCRIPT}" ]]; then
  echo "Error: setup script not found or not executable: ${SETUP_SCRIPT}"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

echo "[1/4] Running setup..."
"${SETUP_SCRIPT}"

echo "[2/4] Activating li-model..."
eval "$(conda shell.bash hook)"
conda activate li-model

BUILD_INPUT_DIR="${BUILD_INPUT_DIR:-data/Li-metal_OUTCARs}"
BUILD_OUTPUT_DIR="${BUILD_OUTPUT_DIR:-data/traj}"
BUILD_DUPLICATE="${BUILD_DUPLICATE:-1}"
BUILD_FFIELD="${BUILD_FFIELD:-ffield.json}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"

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

is_dataset_ready() {
  local train_dir="${ROOT_DIR}/${BUILD_OUTPUT_DIR}/train"
  local test_dir="${ROOT_DIR}/${BUILD_OUTPUT_DIR}/test"
  local csv_file="${ROOT_DIR}/${BUILD_OUTPUT_DIR}/dft_summary.csv"

  if [[ ! -d "${train_dir}" || ! -d "${test_dir}" ]]; then
    return 1
  fi

  local train_count
  local test_count
  train_count="$(find "${train_dir}" -type f -name '*.traj' | wc -l | tr -d ' ')"
  test_count="$(find "${test_dir}" -type f -name '*.traj' | wc -l | tr -d ' ')"

  if [[ "${train_count}" -gt 0 && "${test_count}" -gt 0 && -f "${csv_file}" ]]; then
    return 0
  fi
  return 1
}

is_training_complete() {
  local training_log="${ROOT_DIR}/training.log"

  if [[ -f "${training_log}" ]] && grep -q "Convergence Occurred, job compeleted." "${training_log}"; then
    return 0
  fi
  return 1
}

if is_dataset_ready; then
  echo "[3/4] Dataset already built. Skipping dataset build."
else
  echo "[3/4] Running dataset build..."
  python "${ROOT_DIR}/run_scripts/build_dataset_duplicate.py" \
    --input_dir "${ROOT_DIR}/${BUILD_INPUT_DIR}" \
    --output_dir "${ROOT_DIR}/${BUILD_OUTPUT_DIR}" \
    --duplicate "${BUILD_DUPLICATE}" \
    --ffield "${ROOT_DIR}/${BUILD_FFIELD}" \
    --workers "${BUILD_WORKERS}"
fi

if is_training_complete; then
  echo "[4/4] Training already appears complete. Skipping training."
else
  echo "[4/4] Running training..."
  python "${ROOT_DIR}/run_scripts/train_without_force_Li_NN.py" \
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
fi

echo "Done."
