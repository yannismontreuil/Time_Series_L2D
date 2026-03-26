#!/bin/bash
#SBATCH --job-name=synth_trans_arr
#SBATCH --partition=long
#SBATCH --output=logs/synth_trans_arr_%A_%a.out
#SBATCH --error=logs/synth_trans_arr_%A_%a.err
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
  "1500 6 8 10 0.05 60"
  "2000 6 8 10 0.05 60"
  "1500 8 10 20 0.05 100"
  "2000 8 10 20 0.05 100"
  "1500 8 12 20 0.05 120"
  "2000 8 12 20 0.05 120"
  "1500 10 12 20 0.03 120"
  "2000 10 12 20 0.03 120"
)

spec="${runs[$SLURM_ARRAY_TASK_ID]}"
read -r emtk niter nsamp burn tlr tsteps <<<"${spec}"
out_dir="out/synth_transition_search_tk${emtk}_n${niter}_s${nsamp}_b${burn}_lr${tlr}_ts${tsteps}"

echo "Python: ${PYTHON}"
echo "RUN em_tk=${emtk} n=${niter} samples=${nsamp} burn=${burn} theta_lr=${tlr} theta_steps=${tsteps}"
echo "OUT ${out_dir}"
FACTOR_DISABLE_PLOT_SHOW=1 MPLBACKEND=Agg "${PYTHON}" -u scripts/synthetic_transition_ablation.py \
  --seeds 11 \
  --T 2500 \
  --em-tk "${emtk}" \
  --em-iters "${niter}" \
  --em-samples "${nsamp}" \
  --em-burn-in "${burn}" \
  --em-theta-lr "${tlr}" \
  --em-theta-steps "${tsteps}" \
  --transition-init random \
  --out-dir "${out_dir}"

echo
echo "SUMMARY ${out_dir}"
cat "${out_dir}/synthetic_transition_ablation_summary.csv"
