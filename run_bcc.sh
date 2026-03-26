#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TRAIN_DATASET_GROUP="${TRAIN_DATASET_GROUP:-bcc}"
exec bash "${ROOT_DIR}/train.sh"
