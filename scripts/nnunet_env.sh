#!/usr/bin/env bash

# Ce fichier doit être sourcé afin que les variables restent dans le shell courant :
#   source scripts/nnunet_env.sh

NNUNET_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export nnUNet_raw="${NNUNET_PROJECT_ROOT}/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_PROJECT_ROOT}/nnunet/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_PROJECT_ROOT}/nnunet/nnUNet_results"

unset NNUNET_PROJECT_ROOT
