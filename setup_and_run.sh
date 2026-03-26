#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${ROOT_DIR}/setup.sh"
ENV_NAME="${ENV_NAME:-li-model}"

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

run_in_env() {
  env -u CONDA_VERBOSITY -u CONDA_LOG_LEVEL \
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

echo "[2/4] Using ${ENV_NAME} via conda run..."

BUILD_INPUT_DIR="${BUILD_INPUT_DIR:-data/Li-metal_OUTCARs}"
BUILD_OUTPUT_DIR="${BUILD_OUTPUT_DIR:-data/traj}"
BUILD_DUPLICATE="${BUILD_DUPLICATE:-1}"
BUILD_FFIELD="${BUILD_FFIELD:-ffield.json}"
BUILD_CUTOFF="${BUILD_CUTOFF:-5.0}"
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
TRAIN_DATASET_GROUP="${TRAIN_DATASET_GROUP:-all}"

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

dataset_has_small_cells() {
  local train_dir="${ROOT_DIR}/${BUILD_OUTPUT_DIR}/train"
  local test_dir="${ROOT_DIR}/${BUILD_OUTPUT_DIR}/test"

  run_in_env python - <<PY
import json
import numpy as np
from pathlib import Path
from ase.io import read

root = Path(r"${ROOT_DIR}")
train_dir = Path(r"${train_dir}")
test_dir = Path(r"${test_dir}")
ffield = root / "${BUILD_FFIELD}"
cutoff = float("${BUILD_CUTOFF}")

if ffield.exists():
    try:
        data = json.loads(ffield.read_text())
        rcut = data.get("rcut", cutoff)
        if isinstance(rcut, dict):
            cutoff = max(float(v) for v in rcut.values())
        else:
            cutoff = float(rcut)
    except Exception:
        pass

minimum_box_width = 2.0 * cutoff

for split_dir in (train_dir, test_dir):
    if not split_dir.exists():
        continue
    for traj_path in split_dir.rglob("*.traj"):
        frames = read(str(traj_path), index=":")
        if not isinstance(frames, list):
            frames = [frames]

        for frame in frames:
            cell = frame.get_cell()
            volume = cell.volume
            widths = [
                volume / np.linalg.norm(np.cross(cell[1], cell[2])),
                volume / np.linalg.norm(np.cross(cell[2], cell[0])),
                volume / np.linalg.norm(np.cross(cell[0], cell[1])),
            ]
            if any(width < minimum_box_width for width in widths):
                print(f"Found undersized cell in: {traj_path}")
                raise SystemExit(0)

raise SystemExit(1)
PY
}

is_training_complete() {
  local training_log="${ROOT_DIR}/training.log"

  if [[ -f "${training_log}" ]] && grep -q "Convergence Occurred, job compeleted." "${training_log}"; then
    return 0
  fi
  return 1
}

if is_dataset_ready; then
  if dataset_has_small_cells; then
    echo "[3/4] Existing dataset has cells below 2*rcut. Rebuilding dataset..."
    rm -rf "${ROOT_DIR}/${BUILD_OUTPUT_DIR}"
    run_in_env python "${ROOT_DIR}/run_scripts/build_dataset_duplicate.py" \
      --input_dir "${ROOT_DIR}/${BUILD_INPUT_DIR}" \
      --output_dir "${ROOT_DIR}/${BUILD_OUTPUT_DIR}" \
      --duplicate "${BUILD_DUPLICATE}" \
      --ffield "${ROOT_DIR}/${BUILD_FFIELD}" \
      --cutoff "${BUILD_CUTOFF}" \
      --workers "${BUILD_WORKERS}"
  else
    echo "[3/4] Dataset already built. Skipping dataset build."
  fi
else
  echo "[3/4] Running dataset build..."
  run_in_env python "${ROOT_DIR}/run_scripts/build_dataset_duplicate.py" \
    --input_dir "${ROOT_DIR}/${BUILD_INPUT_DIR}" \
    --output_dir "${ROOT_DIR}/${BUILD_OUTPUT_DIR}" \
    --duplicate "${BUILD_DUPLICATE}" \
    --ffield "${ROOT_DIR}/${BUILD_FFIELD}" \
    --cutoff "${BUILD_CUTOFF}" \
    --workers "${BUILD_WORKERS}"
fi

if is_training_complete; then
  echo "[4/4] Training already appears complete. Skipping training."
else
  echo "[4/4] Running training..."
  run_in_env python "${ROOT_DIR}/run_scripts/train_without_force_Li_NN.py" \
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
    --vdw "${TRAIN_VDW}" \
    --dataset_group "${TRAIN_DATASET_GROUP}"
fi

echo "Done."
