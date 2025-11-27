import numpy as np

from router_model import SLDSIMMRouter, feature_phi
from router_model_corr import SLDSIMMRouter_Corr
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import LearningToDeferBaseline, L2D_RNN
from plot_utils import plot_time_series, evaluate_routers_and_baselines
from horizon_planning import evaluate_horizon_planning


if __name__ == "__main__":
    # Model dimensions
    # We consider a universe of 5 experts indexed j=0,...,4.
    # Expert 4 (index 4) will be dynamically added after t=100
    # and removed again at t=150 via the environment availability.
    N = 5          # experts
    d = 2          # state dimension (= dim φ(x))
    lambda_risk = - 0.2

    M = 2          # regimes
    # SLDS parameters (simple example, independent experts)
    A = np.stack([np.eye(d), np.eye(d)], axis=0)         # identity dynamics
    Q = np.stack([0.01 * np.eye(d), 0.1 * np.eye(d)], axis=0)  # different drift scales
    R = np.ones((M, N), dtype=float) * 0.5              # observation noise
    Pi = np.array([[0.9, 0.1],
                   [0.2, 0.8]], dtype=float)            # regime transitions
    beta = np.zeros(N, dtype=float)                     # consultation costs

    # Routers for partial and full feedback (independent experts)
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

    # --------------------------------------------------------
    # Correlated-expert SLDS-IMM routers (shared factor model)
    # --------------------------------------------------------

    d_g = 1          # shared-factor dimension
    d_u = d          # idiosyncratic dimension, tied to φ(x)

    staleness_threshold = None

    A_g = np.stack([np.eye(d_g), np.eye(d_g)], axis=0)
    Q_g = np.stack([0.01 * np.eye(d_g), 0.05 * np.eye(d_g)], axis=0)
    A_u = np.stack([np.eye(d_u), np.eye(d_u)], axis=0)
    Q_u = np.stack([0.01 * np.eye(d_u), 0.1 * np.eye(d_u)], axis=0)

    # Shared-factor loadings: first feature (intercept) loads on g_t.
    B = np.zeros((N, d_u, d_g), dtype=float)
    for j in range(N):
        B[j, 0, 0] = 1.0

    router_partial_corr = SLDSIMMRouter_Corr(
        num_experts=N,
        num_regimes=M,
        shared_dim=d_g,
        idiosyncratic_dim=d_u,
        feature_fn=feature_phi,
        A_g=A_g,
        Q_g=Q_g,
        A_u=A_u,
        Q_u=Q_u,
        B=B,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=lambda_risk,
        staleness_threshold=staleness_threshold,
        feedback_mode="partial",
    )

    router_full_corr = SLDSIMMRouter_Corr(
        num_experts=N,
        num_regimes=M,
        shared_dim=d_g,
        idiosyncratic_dim=d_u,
        feature_fn=feature_phi,
        A_g=A_g,
        Q_g=Q_g,
        A_u=A_u,
        Q_u=Q_u,
        B=B,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=lambda_risk,
        staleness_threshold=staleness_threshold,
        feedback_mode="full",
    )

    # Environment with dynamic expert availability.
    # - Expert 1: unavailable on [10, 50] and [200, 250] (inclusive).
    # - Expert 4: arrives after t=100 and leaves at t=150, i.e.
    #   available on [101, 150] and unavailable outside that window.
    env = SyntheticTimeSeriesEnv(
        num_experts=N,
        num_regimes=M,
        T=300,
        seed=42,
        unavailable_expert_idx=1,
        unavailable_intervals=[[10, 50], [200, 250]],
        arrival_expert_idx=4,
        arrival_intervals=[[120, 200]],
    )

    # Plot the true series and expert predictions
    plot_time_series(env)

    # L2D baselines (usual linear policy and RNN policy) for full-horizon evaluation
    l2d_baseline = LearningToDeferBaseline(
        num_experts=N,
        feature_fn=feature_phi,
        alpha=np.ones(N, dtype=float),
        beta=beta,
        learning_rate=1e-2,
    )
    l2d_rnn_baseline = L2D_RNN(
        num_experts=N,
        feature_fn=feature_phi,
        alpha=np.ones(N, dtype=float),
        beta=beta,
    )

    # Evaluate routers, L2D baseline, and constant-expert baselines,
    # and plot their induced prediction time series.
    evaluate_routers_and_baselines(
        env,
        router_partial,
        router_full,
        l2d_baseline,
        router_partial_corr=router_partial_corr,
        router_full_corr=router_full_corr,
        l2d_rnn_baseline=l2d_rnn_baseline,
    )

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
    t0 = 175
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
        router_partial_corr=router_partial_corr,
        router_full_corr=router_full_corr,
    )
