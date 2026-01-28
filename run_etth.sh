#!/bin/bash
#SBATCH --job-name=l2d_etth1
#SBATCH --partition=long
#SBATCH --output=logs/etth1_%A_%a.out
#SBATCH --error=logs/etth1_%A_%a.err
#SBATCH --nodes=1
#SBATCH --nodelist=xcnz2,xcnz4,xcnz5
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --mail-user=yuletian@u.nus.edu
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
# Allow override via env var; otherwise try conda env, then conda activation.
PYTHON="${PYTHON:-}"
CONDA_BASE="${HOME}/miniconda3"
ENV_NAME="Time_Series_L2D"
if command -v uname >/dev/null 2>&1; then
  echo "Arch: $(uname -m)"
fi
if [[ -z "${PYTHON}" ]]; then
  candidate="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
  if [[ -x "${candidate}" ]] && "${candidate}" -V >/dev/null 2>&1; then
    PYTHON="${candidate}"
  fi
fi
if [[ -z "${PYTHON}" ]]; then
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    base_py="${CONDA_BASE}/bin/python"
    if [[ -x "${base_py}" ]] && "${base_py}" -V >/dev/null 2>&1; then
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${ENV_NAME}"
      PYTHON="$(command -v python)"
    else
      echo "ERROR: ${base_py} is not runnable on this node (likely arch mismatch)."
      echo "Set PYTHON=/path/to/python or rebuild conda on this node type."
      exit 1
    fi
  fi
fi
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Could not resolve a usable Python interpreter."
  echo "Set PYTHON=/path/to/python or ensure conda env ${ENV_NAME} exists on this node."
  exit 1
fi

# Ensure local package imports (ablation, utils, model, etc.) work
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

set -euo pipefail

echo "Checking PyTorch availability..."
"${PYTHON}" - <<'PY'
import sys
print("Python:", sys.executable)
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

echo "Running ETTh1 hyperparameter sweep under Slurm (array)..."

BASE_CONFIG="${SLURM_SUBMIT_DIR}/config/config_etth1.yaml"
RUN_DIR="${SLURM_SUBMIT_DIR}/out/etth1_sweep_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

RUNS=(
  "seed=40 em_online_enabled=true em_online_window=1000 em_online_period=500 em_online_theta_lr=0.001 em_online_theta_steps=1 em_online_n=1 em_online_samples=20 em_online_burn_in=5"
  "seed=41 em_online_enabled=true em_online_window=1000 em_online_period=500 em_online_theta_lr=0.001 em_online_theta_steps=1 em_online_n=1 em_online_samples=20 em_online_burn_in=5"
  "seed=42 em_online_enabled=true em_online_window=1000 em_online_period=500 em_online_theta_lr=0.001 em_online_theta_steps=1 em_online_n=1 em_online_samples=20 em_online_burn_in=5"
  "seed=43 em_online_enabled=true em_online_window=1000 em_online_period=500 em_online_theta_lr=0.001 em_online_theta_steps=1 em_online_n=1 em_online_samples=20 em_online_burn_in=5"
  "seed=44 em_online_enabled=true em_online_window=1000 em_online_period=500 em_online_theta_lr=0.001 em_online_theta_steps=1 em_online_n=1 em_online_samples=20 em_online_burn_in=5"
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array."
  exit 1
fi

run_id=$((SLURM_ARRAY_TASK_ID + 1))
run_cfg="${RUNS[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "${run_cfg}" ]]; then
  echo "ERROR: No run config for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

cfg_out="${RUN_DIR}/config_etth1_online_${run_id}.yaml"
echo "-------------------------------------------------"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Run ${run_id}: ${run_cfg}"
echo "Config: ${cfg_out}"

  "${PYTHON}" - "${BASE_CONFIG}" "${cfg_out}" "${run_cfg}" <<'PY'
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
"${PYTHON}" -u slds_imm_router.py --config "${cfg_out}"

echo "================================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "================================================="
