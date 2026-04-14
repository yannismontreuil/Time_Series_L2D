#!/bin/bash
# Slurm array launcher for sliding-window NeuralUCB on Jena/Melbourne.

#SBATCH --job-name=sw_neural
#SBATCH --output=sw_neural_%A_%a.out
#SBATCH --error=sw_neural_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cases=(
  "jena 30 config/config_jena_tuned.yaml out/paper_testing_sw_neural/jena_w030.csv"
  "jena 90 config/config_jena_tuned.yaml out/paper_testing_sw_neural/jena_w090.csv"
  "jena 180 config/config_jena_tuned.yaml out/paper_testing_sw_neural/jena_w180.csv"
  "melbourne 30 config/config_melbourne_review_tuned.yaml out/paper_testing_sw_neural/melbourne_w030.csv"
  "melbourne 90 config/config_melbourne_review_tuned.yaml out/paper_testing_sw_neural/melbourne_w090.csv"
  "melbourne 180 config/config_melbourne_review_tuned.yaml out/paper_testing_sw_neural/melbourne_w180.csv"
)

entry="${cases[$SLURM_ARRAY_TASK_ID]}"
read -r dataset window config_path out_path <<<"$entry"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TORCHDYNAMO_DISABLE=1

cd "$REPO_DIR"
mkdir -p "$(dirname "$out_path")"

echo "[sw-neural] dataset=$dataset window=$window config=$config_path out=$out_path"
"$PYTHON_BIN" scripts/eval_paper_testing_baselines.py \
  --config "$config_path" \
  --method sw-neuralucb \
  --sw-neural-window "$window" \
  --out "$out_path"
