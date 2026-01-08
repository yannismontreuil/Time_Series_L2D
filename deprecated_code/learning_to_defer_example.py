import numpy as np

from models.l2d_baseline import LearningToDeferBaseline
from slds_imm_router import (
    SLDSIMMRouter,
    SyntheticTimeSeriesEnv,
    feature_phi,
    evaluate_routers_and_baselines,
)


def run_learning_to_defer_example(T: int = 300, seed: int = 42) -> None:
    """
    Standalone script to compare the usual learning-to-defer baseline
    against the SLDS+IMM routers (partial and full feedback) on the
    synthetic time-series environment.
    """
    # Model dimensions
    M = 2          # regimes
    N = 3          # experts
    d = 2          # state dimension (= dim φ(x))

    # SLDS parameters (simple example, same as in slds_imm_router.__main__)
    A = np.stack([np.eye(d), np.eye(d)], axis=0)         # identity dynamics
    Q = np.stack([0.01 * np.eye(d), 0.1 * np.eye(d)], axis=0)  # different drift scales
    R = np.ones((M, N), dtype=float) * 0.5              # observation noise
    Pi = np.array([[0.9, 0.1],
                   [0.2, 0.8]], dtype=float)            # regime transitions
    beta = np.array([0.0, 0.0, 0.0], dtype=float)       # consultation costs

    # Routers
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
        lambda_risk=0.0,
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
        lambda_risk=0.0,
        feedback_mode="full",
    )

    # L2D baseline
    l2d_baseline = LearningToDeferBaseline(
        num_experts=N,
        feature_fn=feature_phi,
        alpha=np.ones(N, dtype=float),
        beta=beta,
        learning_rate=1e-2,
    )

    # Environment (expert 1 unavailable on [T/4, 2T/4) by default)
    env = SyntheticTimeSeriesEnv(
        num_experts=N,
        num_regimes=M,
        T=T,
        seed=seed,
        unavailable_expert_idx=1,
        unavailable_start_t=None,
    )

    # Evaluate and plot
    evaluate_routers_and_baselines(env, router_partial, router_full, l2d_baseline)


if __name__ == "__main__":
    run_learning_to_defer_example()
