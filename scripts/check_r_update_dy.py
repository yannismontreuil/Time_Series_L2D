import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.factorized_slds import FactorizedSLDS


def _make_feature_fn(d_y: int):
    def feature_fn(_x: np.ndarray) -> np.ndarray:
        return np.zeros((0, d_y), dtype=float)

    return feature_fn


def main() -> None:
    rng = np.random.default_rng(0)
    d_y = 3
    T = 5000
    sigma = 2.5

    residuals = rng.normal(scale=sigma, size=(T, d_y))
    contexts = [np.zeros(1, dtype=float) for _ in range(T)]
    available_sets = [[0] for _ in range(T)]
    actions = [0 for _ in range(T)]
    residuals_list = [residuals[t].astype(float) for t in range(T)]
    residuals_full = [residuals[t].reshape(1, d_y) for t in range(T)]

    router = FactorizedSLDS(
        M=1,
        d_g=0,
        d_phi=0,
        feature_fn=_make_feature_fn(d_y),
        num_experts=1,
        R=1.0,
        feedback_mode="full",
        seed=0,
    )
    router.fit_em(
        contexts=contexts,
        available_sets=available_sets,
        actions=actions,
        residuals=residuals_list,
        residuals_full=residuals_full,
        n_em=1,
        n_samples=1,
        burn_in=0,
        use_validation=False,
        set_em_tk=False,
        theta_steps=0,
        priors={"a_R": 0.0, "b_R": 0.0},
    )

    if np.ndim(router.R) == 0:
        r_hat = float(router.R)
    else:
        r_hat = float(np.asarray(router.R)[0, 0])

    empirical = float(np.mean(residuals * residuals))
    diff = abs(r_hat - empirical)

    print(f"d_y={d_y} T={T} sigma={sigma}")
    print(f"R_hat={r_hat:.6f} empirical_var={empirical:.6f} diff={diff:.6f}")

    tol = 0.05 * sigma * sigma
    if diff > tol:
        raise SystemExit(f"FAIL: diff {diff:.6f} > tol {tol:.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
