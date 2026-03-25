#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NEW_ENV_PY="${NEW_ENV_PY:-/home/yannis/anaconda3/envs/new_env/bin/python}"

echo "[1/3] tail-validation regression"
python3 scripts/check_tail_validation_prefix.py

echo "[2/3] multivariate R-update regression"
python3 scripts/check_r_update_dy.py

echo "[3/3] covariance PSD sanity check"
python3 - <<'PY'
import numpy as np
from models.factorized_slds import FactorizedSLDS

router = FactorizedSLDS(
    M=2,
    d_g=2,
    d_phi=2,
    feature_fn=lambda x: np.eye(2),
    num_experts=1,
    R=0.1,
    A_g=np.stack([np.eye(2), np.eye(2)]),
    A_u=np.stack([np.eye(2), np.eye(2)]),
    Q_g=np.stack([0.01 * np.eye(2), 0.02 * np.eye(2)]),
    Q_u=np.stack([0.03 * np.eye(2), 0.04 * np.eye(2)]),
    seed=0,
)
router.reset_beliefs()
router.manage_registry([0], t_now=1)
phi = router._compute_phi(np.array([0.0, 0.0]))
w_pred, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred = router._interaction_and_time_update(
    np.array([0.0, 0.0]), router.w, router.mu_g, router.Sigma_g, router.mu_u, router.Sigma_u
)
_, _, Sigma_g_post, _, Sigma_u_post = router._update_from_predicted(
    0, np.array([1.0, -0.5]), phi, w_pred, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred
)
for m in range(router.M):
    eig_g = np.linalg.eigvalsh(Sigma_g_post[m])
    eig_u = np.linalg.eigvalsh(Sigma_u_post[0][m])
    if eig_g.min() <= 0.0 or eig_u.min() <= 0.0:
        raise SystemExit(
            f"FAIL: non-PSD posterior covariance in mode {m}: "
            f"min_eig_g={eig_g.min()} min_eig_u={eig_u.min()}"
        )
    print(
        f"mode={m} min_eig_g={float(eig_g.min()):.6e} "
        f"min_eig_u={float(eig_u.min()):.6e}"
    )
print("PASS")
PY

if [[ -x "$NEW_ENV_PY" ]]; then
  echo "[4/6] transition branch regression"
  "$NEW_ENV_PY" scripts/check_transition_branches.py

  echo "[5/6] EM branch smoke check"
  "$NEW_ENV_PY" scripts/check_em_branches.py

  echo "[6/6] rolling-validation regression"
  "$NEW_ENV_PY" scripts/check_rolling_validation.py
else
  echo "[4/6] skipped transition/EM branch checks because NEW_ENV_PY is not executable: $NEW_ENV_PY"
fi

echo "All algorithm checks passed."
