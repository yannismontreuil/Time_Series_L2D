import itertools
from typing import Dict, Tuple, List

import numpy as np

from router_model import SLDSIMMRouter, feature_phi
from synthetic_env import SyntheticTimeSeriesEnv
from router_eval import run_router_on_env


def build_environment(seed: int) -> SyntheticTimeSeriesEnv:
    """
    Environment configuration matching the main experiment in slds_imm_router.py.

    We keep the same number of experts/regimes, horizon length, and
    availability pattern so that hyperparameter search is tailored to
    the actual experiment setup.
    """
    M = 2
    N = 3
    T = 300
    return SyntheticTimeSeriesEnv(
        num_experts=N,
        num_regimes=M,
        T=T,
        seed=seed,
        unavailable_expert_idx=1,
        unavailable_intervals=[[10, 20], [150, 200]],
    )


def build_routers(
    lambda_risk: float,
    q_scale: float,
    r_scale: float,
) -> Tuple[SLDSIMMRouter, SLDSIMMRouter]:
    """
    Construct partial- and full-feedback routers with given hyperparameters.

    We start from the base SLDS+IMM configuration used in slds_imm_router.py
    and scale the process and observation noise by q_scale and r_scale.
    """
    M = 2  # regimes
    N = 3  # experts
    d = 2  # state dimension (= dim φ(x))

    A = np.stack([np.eye(d), np.eye(d)], axis=0)
    Q_base = np.stack([0.01 * np.eye(d), 0.1 * np.eye(d)], axis=0)
    R_base = np.ones((M, N), dtype=float) * 0.5
    Pi = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    beta = np.array([0.0, 0.0, 0.0], dtype=float)

    Q = Q_base * float(q_scale)
    R = R_base * float(r_scale)

    router_partial = SLDSIMMRouter(
        num_experts=N,
        num_regimes=M,
        state_dim=d,
        feature_fn=feature_phi,
        A=A,
        Q=Q,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=float(lambda_risk),
        feedback_mode="partial",
    )

    router_full = SLDSIMMRouter(
        num_experts=N,
        num_regimes=M,
        state_dim=d,
        feature_fn=feature_phi,
        A=A,
        Q=Q,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=float(lambda_risk),
        feedback_mode="full",
    )

    return router_partial, router_full


def evaluate_config(
    lambda_risk: float,
    q_scale: float,
    r_scale: float,
    seeds: List[int],
) -> Tuple[float, float]:
    """
    Average router costs over a list of environment seeds for one config.

    Returns (avg_cost_partial, avg_cost_full).
    """
    costs_partial = []
    costs_full = []

    for seed in seeds:
        env = build_environment(seed)
        router_partial, router_full = build_routers(
            lambda_risk=lambda_risk,
            q_scale=q_scale,
            r_scale=r_scale,
        )

        c_partial, _ = run_router_on_env(router_partial, env)
        c_full, _ = run_router_on_env(router_full, env)

        costs_partial.append(float(c_partial.mean()))
        costs_full.append(float(c_full.mean()))

    return float(np.mean(costs_partial)), float(np.mean(costs_full))


def run_hyperparam_search() -> None:
    """
    Grid search over a small, tailored set of SLDS+IMM router hyperparameters.

    Hyperparameters:
      - lambda_risk: risk sensitivity in the router score
      - q_scale: scaling of process noise Q
      - r_scale: scaling of observation noise R

    The grid below is designed around the values used in slds_imm_router.py.
    """
    lambda_risk_grid = [-0.5, -0.2, 0.0, 0.2]
    q_scale_grid = [0.5, 1.0, 2.0]
    r_scale_grid = [0.5, 1.0, 2.0]

    seeds = [42, 43, 44, 45, 46]

    results: List[Dict[str, float]] = []

    for lambda_risk, q_scale, r_scale in itertools.product(
        lambda_risk_grid,
        q_scale_grid,
        r_scale_grid,
    ):
        avg_partial, avg_full = evaluate_config(
            lambda_risk=lambda_risk,
            q_scale=q_scale,
            r_scale=r_scale,
            seeds=seeds,
        )

        result = {
            "lambda_risk": float(lambda_risk),
            "q_scale": float(q_scale),
            "r_scale": float(r_scale),
            "avg_cost_partial": float(avg_partial),
            "avg_cost_full": float(avg_full),
        }
        results.append(result)

        print(
            "Config:",
            f"lambda_risk={lambda_risk:+.2f},",
            f"q_scale={q_scale:.2f},",
            f"r_scale={r_scale:.2f}",
            "|",
            f"partial={avg_partial:.4f}, full={avg_full:.4f}",
        )

    # Sort results by partial-feedback router performance (ascending cost)
    results_sorted_partial = sorted(
        results,
        key=lambda r: r["avg_cost_partial"],
    )
    best_partial = results_sorted_partial[0]

    # Sort results by full-feedback router performance (ascending cost)
    results_sorted_full = sorted(
        results,
        key=lambda r: r["avg_cost_full"],
    )
    best_full = results_sorted_full[0]

    print("\n=== Best config (partial-feedback router) ===")
    print(
        f"lambda_risk={best_partial['lambda_risk']:+.2f}, "
        f"q_scale={best_partial['q_scale']:.2f}, "
        f"r_scale={best_partial['r_scale']:.2f} | "
        f"avg_cost_partial={best_partial['avg_cost_partial']:.4f}"
    )

    print("\n=== Best config (full-feedback router) ===")
    print(
        f"lambda_risk={best_full['lambda_risk']:+.2f}, "
        f"q_scale={best_full['q_scale']:.2f}, "
        f"r_scale={best_full['r_scale']:.2f} | "
        f"avg_cost_full={best_full['avg_cost_full']:.4f}"
    )


if __name__ == "__main__":
    run_hyperparam_search()

