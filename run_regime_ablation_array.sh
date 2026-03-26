#!/bin/bash
#SBATCH --job-name=regime_ablt
#SBATCH --partition=long
#SBATCH --output=logs/regime_ablt_%A_%a.out
#SBATCH --error=logs/regime_ablt_%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-5

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE="${HOME}/miniconda3"
ENV_NAME="Routing_LLM"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  PYTHON="$(command -v python)"
fi

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
set -euo pipefail

CONFIGS=(
  "config/ablation_jena_M1_g1_g.yaml"
  "config/ablation_jena_M2_g1_g.yaml"
  "config/ablation_jena_M4_g1_g.yaml"
  "config/ablation_melbourne_M1_g1_gz12.yaml"
  "config/ablation_melbourne_M2_g1_gz12.yaml"
  "config/ablation_melbourne_M4_g1_gz12.yaml"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
STEM="$(basename "${CFG}" .yaml)"
OUT_DIR="out/regime_count_ablation"
OUT_CSV="${OUT_DIR}/${STEM}.csv"

echo "Python: ${PYTHON}"
echo "Config: ${CFG}"
echo "Output: ${OUT_CSV}"

FACTOR_DISABLE_PLOT_SHOW=1 MPLBACKEND=Agg "${PYTHON}" -u scripts/regime_count_ablation.py \
  --config "${CFG}" \
  --seeds 11 12 13 14 15 \
  --out-csv "${OUT_CSV}"

echo
cat "${OUT_CSV}"
