#!/bin/bash
#SBATCH --job-name=synth_em_final_arr
#SBATCH --partition=long
#SBATCH --output=logs/synth_em_final_arr_%A_%a.out
#SBATCH --error=logs/synth_em_final_arr_%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-4

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

SEEDS=(11 12 13 14 15)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="out/synth_em_best_mild_tk2000_n8_s10_b20_seed${SEED}"

echo "Python: ${PYTHON}"
echo "Seed: ${SEED}"
FACTOR_DISABLE_PLOT_SHOW=1 MPLBACKEND=Agg "${PYTHON}" -u scripts/synthetic_em_ablation.py \
  --seeds "${SEED}" \
  --misspec-profile mild \
  --em-tk 2000 \
  --em-iters 8 \
  --em-samples 10 \
  --em-burn-in 20 \
  --em-use-validation false \
  --out-dir "${OUT_DIR}"

echo
cat "${OUT_DIR}/synthetic_em_ablation_summary.csv"
