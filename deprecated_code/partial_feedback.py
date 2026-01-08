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


def run_partial_feedback_example(T: int = 10):
    """Run a simple partial-feedback simulation."""
    router = SLDSIMMRouter(
        num_experts=N,
        num_regimes=M,
        state_dim=d,
        feature_fn=phi,
        A=A,
        Q=Q,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=0.0,  # risk-neutral; set >0 for risk-averse
        feedback_mode="partial",
    )

    for _ in range(T):
        x_t = rng.normal(size=d)  # context
        K_t = [0, 1, 2]  # available experts

        # 1) Selection (before observing losses at time t)
        r_t, cache = router.select_expert(x_t, K_t)

        # 2) Environment returns loss only for selected expert (partial feedback)
        #    Here fake squared-loss, but any scalar loss is fine.
        true_loss_r_t = float(rng.normal(loc=0.5 + 0.1 * r_t, scale=0.2))

        # 3) Update beliefs
        router.update_beliefs(
            r_t=r_t,
            loss_obs=true_loss_r_t,
            losses_full=None,
            available_experts=K_t,
            cache=cache,
        )


if __name__ == "__main__":
    run_partial_feedback_example()
