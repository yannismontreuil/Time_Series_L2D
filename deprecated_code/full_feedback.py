
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
    phi,
    rng,
)
from train import SLDSIMMRouter


def run_full_feedback_example(T: int = 10):
    """Run a simple full-feedback simulation."""
    router_full = SLDSIMMRouter(
        num_experts=N,
        num_regimes=M,
        state_dim=d,
        feature_fn=phi,
        A=A,
        Q=Q,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=0.0,
        feedback_mode="full",
    )

    for _ in range(T):
        x_t = rng.normal(size=d)
        K_t = [0, 1, 2]

        r_t, cache = router_full.select_expert(x_t, K_t)

        # Environment returns losses for all experts in full-feedback mode
        losses_all = rng.normal(
            loc=np.array([0.2, 0.4, 0.6]),
            scale=0.1,
            size=N,
        )

        router_full.update_beliefs(
            r_t=r_t,
            loss_obs=None,
            losses_full=losses_all,
            available_experts=K_t,
            cache=cache,
        )


if __name__ == "__main__":
    run_full_feedback_example()
