import numpy as np

from router_model import SLDSIMMRouter, feature_phi
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import LearningToDeferBaseline
from plot_utils import plot_time_series, evaluate_routers_and_baselines
from horizon_planning import evaluate_horizon_planning


if __name__ == "__main__":
    # Model dimensions
    M = 2          # regimes
    N = 3          # experts
    d = 2          # state dimension (= dim φ(x))
    lambda_risk = -0.2

    # SLDS parameters (simple example)
    A = np.stack([np.eye(d), np.eye(d)], axis=0)         # identity dynamics
    Q = np.stack([0.01 * np.eye(d), 0.1 * np.eye(d)], axis=0)  # different drift scales
    R = np.ones((M, N), dtype=float) * 0.5              # observation noise
    Pi = np.array([[0.9, 0.1],
                   [0.2, 0.8]], dtype=float)            # regime transitions
    beta = np.array([0.0, 0.0, 0.0], dtype=float)       # consultation costs

    # Routers for partial and full feedback
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
        lambda_risk=lambda_risk,
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
        lambda_risk=lambda_risk,
        feedback_mode="full",
    )

    # Environment (expert 1 unavailable on specified intervals)
    env = SyntheticTimeSeriesEnv(
        num_experts=N,
        num_regimes=M,
        T=300,
        seed=42,
        unavailable_expert_idx=1,
        # expert 1 unavailable on [10, 20] and [150, 200] (inclusive)
        unavailable_intervals=[[10, 20], [150, 200]],
    )

    # Plot the true series and expert predictions
    plot_time_series(env)

    # L2D baseline (usual learning-to-defer) for full-horizon evaluation
    l2d_baseline = LearningToDeferBaseline(
        num_experts=N,
        feature_fn=feature_phi,
        alpha=np.ones(N, dtype=float),
        beta=beta,
        learning_rate=1e-2,
    )

    # Evaluate routers, L2D baseline, and constant-expert baselines,
    # and plot their induced prediction time series.
    evaluate_routers_and_baselines(env, router_partial, router_full, l2d_baseline)

    # --------------------------------------------------------
    # Example: horizon-H planning from a given time t
    # --------------------------------------------------------

    # Build expert prediction functions for planning
    def experts_predict_factory(env_: SyntheticTimeSeriesEnv):
        def f(j: int):
            return lambda x: env_.expert_predict(j, x)
        return [f(j) for j in range(env_.num_experts)]

    experts_predict = experts_predict_factory(env)

    # Simple context update: x_{t+1} := y_hat (recursive forecasting)
    def context_update(x: np.ndarray, y_hat: float) -> np.ndarray:
        return np.array([y_hat], dtype=float)

    # Take current context at t0 and plan H steps ahead, and evaluate.
    t0 = 100
    H = 5
    # Separate L2D baseline instance for horizon-only evaluation (trained up to t0)
    l2d_baseline_horizon = LearningToDeferBaseline(
        num_experts=N,
        feature_fn=feature_phi,
        alpha=np.ones(N, dtype=float),
        beta=beta,
        learning_rate=1e-2,
    )

    evaluate_horizon_planning(
        env=env,
        router_partial=router_partial,
        router_full=router_full,
        beta=beta,
        t0=t0,
        H=H,
        experts_predict=experts_predict,
        context_update=context_update,
        l2d_baseline=l2d_baseline_horizon,
    )
