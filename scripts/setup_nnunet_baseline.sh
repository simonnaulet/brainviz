#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PLAN_AND_PREPROCESS="${PROJECT_ROOT}/.venv/bin/nnUNetv2_plan_and_preprocess"

if [[ ! -x "${PYTHON}" || ! -x "${PLAN_AND_PREPROCESS}" ]]; then
  echo "Environnement incomplet. Exécuter d'abord : uv sync" >&2
  exit 1
fi

source "${PROJECT_ROOT}/scripts/nnunet_env.sh"
cd "${PROJECT_ROOT}"

if [[ ! -d "${PROJECT_ROOT}/dataset/train" ]]; then
  "${PYTHON}" scripts/prepare_dataset.py
fi

RAW_DATASET="${nnUNet_raw}/Dataset501_iSeg2017"
if [[ ! -f "${RAW_DATASET}/dataset.json" ]]; then
  "${PYTHON}" scripts/prepare_nnunet_dataset.py
else
  echo "Dataset nnU-Net déjà converti : ${RAW_DATASET}"
fi

# La RAM WSL disponible est limitée : deux workers évitent le swap pendant le preprocessing.
"${PLAN_AND_PREPROCESS}" \
  -d 501 \
  -c 2d \
  -npfp 2 \
  -np 2 \
  --verify_dataset_integrity

"${PYTHON}" scripts/create_nnunet_splits.py

echo "Setup terminé. Aucun entraînement n'a été lancé."
echo "Pour entraîner le fold 0 : scripts/train_nnunet_baseline.sh 0"
