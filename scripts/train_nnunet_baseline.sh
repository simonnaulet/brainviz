#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN="${PROJECT_ROOT}/.venv/bin/nnUNetv2_train"
FOLD="${1:-0}"

if [[ ! "${FOLD}" =~ ^[0-4]$ ]]; then
  echo "Usage : $0 [fold 0..4]" >&2
  exit 2
fi
if [[ ! -x "${TRAIN}" ]]; then
  echo "nnU-Net absent. Exécuter d'abord : uv sync" >&2
  exit 1
fi

source "${PROJECT_ROOT}/scripts/nnunet_env.sh"
cd "${PROJECT_ROOT}"

# RTX 5070 Ti (GPU 0). Surcharge possible : CUDA_VISIBLE_DEVICES=1 ... pour la RTX 3080.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnUNet_compile="${nnUNet_compile:-true}"
# La machine fournit 12 CPU logiques mais seulement 7,7 Gio de RAM à WSL.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"

exec "${TRAIN}" 501 2d "${FOLD}"
