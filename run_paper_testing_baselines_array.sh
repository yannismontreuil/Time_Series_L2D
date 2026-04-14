#!/bin/bash
# Slurm array launcher for the new paper-testing baselines on Jena/Melbourne.

#SBATCH --job-name=paper_cmp
#SBATCH --output=paper_cmp_%A_%a.out
#SBATCH --error=paper_cmp_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cases=(
  "jena d-linucb config/config_jena_tuned.yaml out/paper_testing/jena_dlinucb.csv"
  "jena cusum-linucb config/config_jena_tuned.yaml out/paper_testing/jena_cusum_linucb.csv"
  "jena glr-linucb config/config_jena_tuned.yaml out/paper_testing/jena_glr_linucb.csv"
  "melbourne d-linucb config/config_melbourne_review_tuned.yaml out/paper_testing/melbourne_dlinucb.csv"
  "melbourne cusum-linucb config/config_melbourne_review_tuned.yaml out/paper_testing/melbourne_cusum_linucb.csv"
  "melbourne glr-linucb config/config_melbourne_review_tuned.yaml out/paper_testing/melbourne_glr_linucb.csv"
)

entry="${cases[$SLURM_ARRAY_TASK_ID]}"
read -r dataset method config_path out_path <<<"$entry"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TORCHDYNAMO_DISABLE=1

cd "$REPO_DIR"
mkdir -p "$(dirname "$out_path")"

echo "[paper-cmp] dataset=$dataset method=$method config=$config_path out=$out_path"
"$PYTHON_BIN" scripts/eval_paper_testing_baselines.py \
  --config "$config_path" \
  --method "$method" \
  --out "$out_path"
