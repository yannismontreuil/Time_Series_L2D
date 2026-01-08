import numpy as np

from example_config import (
    A,
    M,
    N,
    Pi,
    Q,
    R,
    beta,
    d,
)
from train import SLDSIMMRouter


def make_linear_expert(w):
    """Create a simple linear expert y = w^T x."""
    w = np.asarray(w, float)

    def f(x):
        x_arr = np.asarray(x, float)
        return float(w @ x_arr)

    return f


experts_predict = [
    make_linear_expert(np.array([1.0, 0.0])),
    make_linear_expert(np.array([0.0, 1.0])),
    make_linear_expert(np.array([1.0, 1.0])),
]


def context_update(x, y_hat):
    """
    Example context update: append forecast as a new feature.

    Here we keep phi(x) = projection to first d coordinates,
    but you can design phi and Psi jointly.
    """
    x_arr = np.asarray(x, float)
    return np.concatenate([x_arr, np.array([y_hat])])


def phi_ts(x):
    """Feature map for time-series example: first d coordinates."""
    x_arr = np.asarray(x, float)
    return x_arr[:d]


def run_horizon_example(H: int = 5):
    """Run horizon-H planning example."""
    router_ts = SLDSIMMRouter(
        num_experts=N,
        num_regimes=M,
        state_dim=d,
        feature_fn=phi_ts,
        A=A,
        Q=Q,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=0.0,
        feedback_mode="partial",
    )

    x_t = np.zeros(d)  # initial context (adapt to your setting)

    schedule, contexts, scores = router_ts.plan_horizon_schedule(
        x_t=x_t,
        H=H,
        experts_predict=experts_predict,
        context_update=context_update,
        # available_experts_per_h=None => all experts at all horizons
    )

    print("Planned experts:", schedule)
    print("Planned contexts:", contexts)
    print("Planned scores:", scores)


if __name__ == "__main__":
    run_horizon_example()
