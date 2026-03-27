#!/bin/bash
#SBATCH --job-name=fred_gap
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-11

set -euo pipefail

cd ~/scratch/Time_Series_L2D
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate Routing_LLM

export FACTOR_DISABLE_PLOT_SHOW=1
export MPLBACKEND=Agg

python scripts/tune_fred_gap.py --index "${SLURM_ARRAY_TASK_ID}" --out-dir out/fred_gap_sweep
