#!/usr/bin/env bash
# Source this file to export MLIR/LLVM paths from a conda env.

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source scripts/apxm-activate.sh" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is not set. Activate your env first (e.g., conda activate apxm)." >&2
  return 1
fi

export MLIR_DIR="${CONDA_PREFIX}/lib/cmake/mlir"
export LLVM_DIR="${CONDA_PREFIX}/lib/cmake/llvm"
export MLIR_PREFIX="${CONDA_PREFIX}"
export LLVM_PREFIX="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "APXM MLIR env activated from ${CONDA_PREFIX}"
