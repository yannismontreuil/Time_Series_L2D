#!/bin/bash
#SBATCH --job-name=registry_all
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail

cd ~/scratch/Time_Series_L2D

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate Routing_LLM

python scripts/registry_complexity_experiment.py \
  --include-neural \
  --timing-repeats 1 \
  --out-dir out/complexity_registry_allbaselines_full
