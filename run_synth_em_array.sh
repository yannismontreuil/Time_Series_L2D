#!/bin/bash
#SBATCH --job-name=synth_em_arr
#SBATCH --partition=long
#SBATCH --output=logs/synth_em_arr_%A_%a.out
#SBATCH --error=logs/synth_em_arr_%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-7

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

runs=(
  "mild 1000 5 8 10"
  "mild 1500 5 8 10"
  "mild 2000 5 8 10"
  "mild 1000 8 10 20"
  "mild 1500 8 10 20"
  "mild 2000 8 10 20"
  "strong 1000 5 8 10"
  "strong 1500 5 8 10"
)

spec="${runs[$SLURM_ARRAY_TASK_ID]}"
read -r profile emtk niter nsamp burn <<<"${spec}"
out_dir="out/synth_em_search_${profile}_tk${emtk}_n${niter}_s${nsamp}_b${burn}"

echo "Python: ${PYTHON}"
echo "RUN profile=${profile} em_tk=${emtk} n=${niter} samples=${nsamp} burn=${burn}"
echo "OUT ${out_dir}"
FACTOR_DISABLE_PLOT_SHOW=1 MPLBACKEND=Agg "${PYTHON}" -u scripts/synthetic_em_ablation.py \
  --seeds 11 \
  --misspec-profile "${profile}" \
  --em-tk "${emtk}" \
  --em-iters "${niter}" \
  --em-samples "${nsamp}" \
  --em-burn-in "${burn}" \
  --em-use-validation false \
  --out-dir "${out_dir}"

echo
echo "SUMMARY ${out_dir}"
cat "${out_dir}/synthetic_em_ablation_summary.csv"
