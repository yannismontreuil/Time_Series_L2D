#!/bin/bash
#SBATCH --job-name=fred_gap5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-11

set -euo pipefail

cd ~/scratch/Time_Series_L2D
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate Routing_LLM

export FACTOR_DISABLE_PLOT_SHOW=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUT_DIR="${FRED_GAP5_OUT_DIR:-out/fred_gap_fiveseed}"
python scripts/fred_gap_fiveseed.py --index "${SLURM_ARRAY_TASK_ID}" --out-dir "${OUT_DIR}"
