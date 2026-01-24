#!/bin/bash
#SBATCH --job-name=l2d_fred
#SBATCH --partition=long
#SBATCH --output=logs/fred_%A_%a.out
#SBATCH --error=logs/fred_%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --array=0-2
#SBATCH --mail-user=yannis.montreuil@u.nus.edu
#SBATCH --mail-type=START,END,FAIL

set -euo pipefail

mkdir -p logs

echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================="

cd "${SLURM_SUBMIT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Routing_LLM

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "Running FRED hyperparameter sweep under Slurm (array)..."

BASE_CONFIG="${SLURM_SUBMIT_DIR}/config/exp_FRED.yaml"
RUN_DIR="${SLURM_SUBMIT_DIR}/out/fred_sweep_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

RUNS=(
  "em_online_enabled=true em_online_window=400 em_online_period=400 em_online_theta_lr=0.001 em_online_theta_steps=1 em_online_n=1 em_online_samples=15 em_online_burn_in=3"
  "em_online_enabled=true em_online_window=800 em_online_period=600 em_online_theta_lr=0.002 em_online_theta_steps=1 em_online_n=1 em_online_samples=25 em_online_burn_in=4"
  "em_online_enabled=true em_online_window=1200 em_online_period=800 em_online_theta_lr=0.002 em_online_theta_steps=2 em_online_n=1 em_online_samples=30 em_online_burn_in=5"
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array."
  exit 1
fi

task_id=${SLURM_ARRAY_TASK_ID}
if [[ task_id =~ ^[0-9]+$ ]]; then
  array_idx=$task_id
else
  array_idx=0
fi

run_id=$((array_idx + 1))
run_cfg="${RUNS[array_idx]:-}"
if [[ -z "${run_cfg}" ]]; then
  echo "ERROR: No run config for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

cfg_out="${RUN_DIR}/config_fred_sweep_${run_id}.yaml"
echo "-------------------------------------------------"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Run ${run_id}: ${run_cfg}"
echo "Config: ${cfg_out}"

python - <<'PY' "${BASE_CONFIG}" "${cfg_out}" "${run_cfg}"
import sys
import yaml

base_path, out_path, updates_raw = sys.argv[1:]

updates = {}
for token in updates_raw.split():
    if "=" not in token:
        continue
    key, val = token.split("=", 1)
    if val.lower() in ("true", "false"):
        cast = val.lower() == "true"
    else:
        try:
            cast = int(val)
        except ValueError:
            try:
                cast = float(val)
            except ValueError:
                cast = val
    updates[key] = cast

with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

routers = cfg.setdefault("routers", {})
fact = routers.setdefault("factorized_slds", {})
fact.update(updates)

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "Starting Python script..."
python -u slds_imm_router.py --config "${cfg_out}"

echo "================================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "================================================="
