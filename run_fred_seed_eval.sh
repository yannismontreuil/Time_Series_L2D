#!/bin/bash
#SBATCH --job-name=fred_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

cd ~/scratch/Time_Series_L2D
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate Routing_LLM

export FACTOR_DISABLE_PLOT_SHOW=1
export MPLBACKEND=Agg

python scripts/fred_seed_eval.py --config config/exp_FRED_rebuttal.yaml --seeds 11 12 13 14 15
