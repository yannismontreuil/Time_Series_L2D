#!/bin/bash
#SBATCH --job-name=synth_em_final
#SBATCH --partition=long
#SBATCH --output=logs/synth_em_final_%j.out
#SBATCH --error=logs/synth_em_final_%j.err
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00

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

echo "Python: ${PYTHON}"
FACTOR_DISABLE_PLOT_SHOW=1 MPLBACKEND=Agg "${PYTHON}" -u scripts/synthetic_em_ablation.py \
  --seeds 11 12 13 14 15 \
  --misspec-profile mild \
  --em-tk 2000 \
  --em-iters 8 \
  --em-samples 10 \
  --em-burn-in 20 \
  --em-use-validation false \
  --out-dir out/synth_em_best_mild_tk2000_n8_s10_b20_seeds1115

echo
cat out/synth_em_best_mild_tk2000_n8_s10_b20_seeds1115/synthetic_em_ablation_summary.csv
