#!/usr/bin/env bash
set -euo pipefail

# Run the exp_synthetic_1 config across multiple seeds, writing logs per seed.
# Usage:
#   scripts/run_exp_synth_seeds.sh 0 1 2 3
# Environment overrides:
#   CONFIG=path/to/config.yaml
#   OUT_DIR=out/seed_runs/...

CONFIG="${CONFIG:-config/exp_synthetic_1.yaml}"
OUT_DIR="${OUT_DIR:-out/seed_runs/$(date +%Y%m%d_%H%M%S)}"

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(0 1 2 3 4)
fi

mkdir -p "$OUT_DIR"

for seed in "${SEEDS[@]}"; do
  tmp_cfg="${OUT_DIR}/exp_synthetic_1_seed_${seed}.yaml"
  python - "$CONFIG" "$tmp_cfg" "$seed" <<'PY'
import sys
import yaml

config_path, out_path, seed_str = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(seed_str)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

env_cfg = cfg.get("environment", {})
env_cfg["seed"] = seed
cfg["environment"] = env_cfg

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "[seed ${seed}] running with ${tmp_cfg}"
  FACTOR_DISABLE_PLOT_SHOW=1 \
    python slds_imm_router.py --config "$tmp_cfg" \
    | tee "${OUT_DIR}/seed_${seed}.log"
done
