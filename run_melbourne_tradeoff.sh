#!/bin/bash
#SBATCH --job-name=mel_trade
#SBATCH --partition=long
#SBATCH --output=logs/mel_trade_%j.out
#SBATCH --error=logs/mel_trade_%j.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-user=yannis.montreuil@u.nus.edu
#SBATCH --mail-type=START,END,FAIL

mkdir -p logs

echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================="

cd "${SLURM_SUBMIT_DIR}"

PYTHON="${PYTHON:-}"
CONDA_BASE="${HOME}/miniconda3"
ENV_NAME="Routing_LLM"
if command -v uname >/dev/null 2>&1; then
  echo "Arch: $(uname -m)"
fi
if [[ -z "${PYTHON}" ]]; then
  candidate="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
  if [[ -x "${candidate}" ]] && "${candidate}" -V >/dev/null 2>&1; then
    PYTHON="${candidate}"
  fi
fi
if [[ -z "${PYTHON}" ]]; then
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    base_py="${CONDA_BASE}/bin/python"
    if [[ -x "${base_py}" ]] && "${base_py}" -V >/dev/null 2>&1; then
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${ENV_NAME}"
      PYTHON="$(command -v python)"
    else
      echo "ERROR: ${base_py} is not runnable on this node (likely arch mismatch)."
      echo "Set PYTHON=/path/to/python or rebuild conda on this node type."
      exit 1
    fi
  fi
fi
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Could not resolve a usable Python interpreter."
  exit 1
fi

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
set -euo pipefail

echo "Checking Python env..."
"${PYTHON}" - <<'PY'
import sys
import yaml, numpy, pandas, sklearn, matplotlib
print("Python:", sys.executable)
print("imports_ok=1")
PY

HORIZON_ARG=""
if [[ -n "${MEL_TRADEOFF_HORIZON:-}" ]]; then
  HORIZON_ARG="--horizon ${MEL_TRADEOFF_HORIZON}"
fi

NAMES_ARG=""
if [[ -n "${MEL_TRADEOFF_NAMES:-}" ]]; then
  NAMES_ARG="--names ${MEL_TRADEOFF_NAMES}"
fi

REPEATS="${MEL_TRADEOFF_REPEATS:-2}"
OUT_CSV="${MEL_TRADEOFF_OUT:-out/melbourne_tradeoff_sweep_${SLURM_JOB_ID}.csv}"

echo "Running Melbourne tradeoff sweep..."
echo "Horizon override: ${MEL_TRADEOFF_HORIZON:-full}"
echo "Repeats: ${REPEATS}"
echo "Candidates: ${MEL_TRADEOFF_NAMES:-all}"
echo "Output CSV: ${OUT_CSV}"

FACTOR_DISABLE_PLOT_SHOW=1 MPLBACKEND=Agg \
  "${PYTHON}" -u scripts/melbourne_tradeoff_sweep.py \
  --repeats "${REPEATS}" \
  --output "${OUT_CSV}" \
  ${HORIZON_ARG} \
  ${NAMES_ARG}

echo "================================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "================================================="
