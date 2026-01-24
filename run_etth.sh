#!/bin/bash
#SBATCH --job-name=l2d_etth2
#SBATCH --partition=long
#SBATCH --output=logs/etth2_%j.out
#SBATCH --error=logs/etth2_%j.err
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=xcnf[6-25]
#SBATCH --mail-user=yannis.montreuil@u.nus.edu
#SBATCH --mail-type=START,END,FAIL

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================="

cd "${SLURM_SUBMIT_DIR}"

# Optionally activate your conda/virtualenv here
# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate Routing_LLM

# Ensure local package imports (ablation, utils, model, etc.) work
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "Running ETTh2 hyperparameter sweep under Slurm..."

BASE_CONFIG="config/config_etth2.yaml"
RUN_DIR="${SLURM_SUBMIT_DIR}/out/etth2_sweep_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

RUNS=(
  "em_online_enabled=true em_online_window=200 em_online_period=500 em_online_theta_lr=0.001 em_online_theta_steps=1
  em_online_n=1 em_online_samples=10 em_online_burn_in=3"
  "em_online_enabled=true em_online_window=400 em_online_period=500 em_online_theta_lr=0.001 em_online_theta_steps=1
  em_online_n=1 em_online_samples=20 em_online_burn_in=3"
  "em_online_enabled=true em_online_window=800 em_online_period=500 em_online_theta_lr=0.002 em_online_theta_steps=1
  em_online_n=1 em_online_samples=20 em_online_burn_in=5"
)

run_id=0
for run_cfg in "${RUNS[@]}"; do
  run_id=$((run_id + 1))
  cfg_out="${RUN_DIR}/config_etth2_online_${run_id}.yaml"
  echo "-------------------------------------------------"
  echo "Run ${run_id}: ${run_cfg}"
  echo "Config: ${cfg_out}"

  python - <<'PY' "${BASE_CONFIG}" "${cfg_out}" "${run_cfg}"
import sys
try:
    import yaml  # type: ignore
except Exception as exc:
    raise SystemExit(f"PyYAML is required to build configs: {exc}")

base_path = sys.argv[1]
out_path = sys.argv[2]
updates_raw = sys.argv[3]

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
for k, v in updates.items():
    fact[k] = v

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "Starting Python script..."
  python -u slds_imm_router.py --config "${cfg_out}"
done

echo "================================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "================================================="
