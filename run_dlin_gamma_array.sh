#!/bin/bash
# Slurm array launcher for D-LinUCB gamma sweep on Jena/Melbourne.

#SBATCH --job-name=dlin_gamma
#SBATCH --output=dlin_gamma_%A_%a.out
#SBATCH --error=dlin_gamma_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cases=(
  "jena 0.90 config/config_jena_tuned.yaml out/paper_testing_dlin_gamma/jena_g090.csv"
  "jena 0.95 config/config_jena_tuned.yaml out/paper_testing_dlin_gamma/jena_g095.csv"
  "jena 0.98 config/config_jena_tuned.yaml out/paper_testing_dlin_gamma/jena_g098.csv"
  "jena 0.99 config/config_jena_tuned.yaml out/paper_testing_dlin_gamma/jena_g099.csv"
  "jena 0.995 config/config_jena_tuned.yaml out/paper_testing_dlin_gamma/jena_g0995.csv"
  "jena 0.998 config/config_jena_tuned.yaml out/paper_testing_dlin_gamma/jena_g0998.csv"
  "melbourne 0.90 config/config_melbourne_review_tuned.yaml out/paper_testing_dlin_gamma/melbourne_g090.csv"
  "melbourne 0.95 config/config_melbourne_review_tuned.yaml out/paper_testing_dlin_gamma/melbourne_g095.csv"
  "melbourne 0.98 config/config_melbourne_review_tuned.yaml out/paper_testing_dlin_gamma/melbourne_g098.csv"
  "melbourne 0.99 config/config_melbourne_review_tuned.yaml out/paper_testing_dlin_gamma/melbourne_g099.csv"
  "melbourne 0.995 config/config_melbourne_review_tuned.yaml out/paper_testing_dlin_gamma/melbourne_g0995.csv"
  "melbourne 0.998 config/config_melbourne_review_tuned.yaml out/paper_testing_dlin_gamma/melbourne_g0998.csv"
)

entry="${cases[$SLURM_ARRAY_TASK_ID]}"
read -r dataset gamma config_path out_path <<<"$entry"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TORCHDYNAMO_DISABLE=1

cd "$REPO_DIR"
mkdir -p "$(dirname "$out_path")"

echo "[dlin-gamma] dataset=$dataset gamma=$gamma config=$config_path out=$out_path"
"$PYTHON_BIN" scripts/eval_paper_testing_baselines.py \
  --config "$config_path" \
  --method d-linucb \
  --dlin-gamma "$gamma" \
  --out "$out_path"
