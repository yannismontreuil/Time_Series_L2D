import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.factorized_slds import FactorizedSLDS


def main() -> None:
    # One-expert, one-regime toy problem where the validation tail is only
    # predictable if evaluation is conditioned on the training prefix.
    router = FactorizedSLDS(
        M=1,
        d_g=0,
        d_phi=1,
        feature_fn=lambda _x: np.array([1.0]),
        num_experts=1,
        R=0.1,
        A_u=np.array([[[1.0]]]),
        Q_u=np.array([[[1.0e-6]]]),
        feedback_mode="partial",
        seed=0,
    )
    router.u_mean0 = np.array([[0.0]])
    router.u_cov0 = np.array([[[1.0]]])

    # Prefix observations near 10 should induce a sharp posterior close to 10.
    prefix_len = 20
    val_len = 5
    contexts = [np.array([0.0]) for _ in range(prefix_len + val_len)]
    available_sets = [[0] for _ in range(prefix_len + val_len)]
    actions = [0 for _ in range(prefix_len + val_len)]
    residuals = [np.array([10.0]) for _ in range(prefix_len + val_len)]

    # Manually build the "correct" validation score: filter the prefix, then
    # score the tail from the resulting state.
    router.reset_beliefs()
    router._filter_block_nll(
        contexts[:prefix_len],
        available_sets[:prefix_len],
        actions[:prefix_len],
        residuals[:prefix_len],
        accumulate=False,
    )
    expected_pre_em = router._filter_block_nll(
        contexts[prefix_len:],
        available_sets[prefix_len:],
        actions[prefix_len:],
        residuals[prefix_len:],
        accumulate=True,
    ) / float(val_len)

    # Now run fit_em's tail-validation path and ensure it matches the
    # prefix-conditioned score rather than a reset-from-prior score.
    router = FactorizedSLDS(
        M=1,
        d_g=0,
        d_phi=1,
        feature_fn=lambda _x: np.array([1.0]),
        num_experts=1,
        R=0.1,
        A_u=np.array([[[1.0]]]),
        Q_u=np.array([[[1.0e-6]]]),
        feedback_mode="partial",
        seed=0,
    )
    router.u_mean0 = np.array([[0.0]])
    router.u_cov0 = np.array([[[1.0]]])

    call_counts = {"with_prefix": 0, "plain": 0}
    observed_scores = []

    orig_with_prefix = router._evaluate_nll_with_prefix
    orig_plain = router._evaluate_nll

    def wrapped_with_prefix(*args, **kwargs):
        call_counts["with_prefix"] += 1
        val = orig_with_prefix(*args, **kwargs)
        observed_scores.append(float(val))
        return val

    def wrapped_plain(*args, **kwargs):
        call_counts["plain"] += 1
        return orig_plain(*args, **kwargs)

    router._evaluate_nll_with_prefix = wrapped_with_prefix
    router._evaluate_nll = wrapped_plain

    router.fit_em(
        contexts=contexts,
        available_sets=available_sets,
        actions=actions,
        residuals=residuals,
        n_em=1,
        n_samples=3,
        burn_in=0,
        val_fraction=val_len / float(prefix_len + val_len),
        val_strategy="tail",
        use_validation=True,
        set_em_tk=False,
        theta_steps=0,
        print_val_loss=False,
    )

    got = orig_with_prefix(
        contexts[:prefix_len],
        available_sets[:prefix_len],
        actions[:prefix_len],
        residuals[:prefix_len],
        None,
        contexts[prefix_len:],
        available_sets[prefix_len:],
        actions[prefix_len:],
        residuals[prefix_len:],
        residuals_full=None,
    )

    diff = abs(got - observed_scores[0])
    print(f"expected_pre_em_tail_nll={expected_pre_em:.6f}")
    print(f"fit_em_recorded_tail_nll={observed_scores[0]:.6f}")
    print(f"got_tail_nll={got:.6f}")
    print(f"diff={diff:.6e}")
    print(
        f"fit_em_calls with_prefix={call_counts['with_prefix']} plain={call_counts['plain']}"
    )
    if call_counts["with_prefix"] <= 0:
        raise SystemExit("FAIL: fit_em tail validation did not call _evaluate_nll_with_prefix")
    if call_counts["plain"] != 0:
        raise SystemExit("FAIL: fit_em tail validation incorrectly called _evaluate_nll")
    if diff > 1e-8:
        raise SystemExit(
            f"FAIL: recomputed post-EM tail validation mismatch diff={diff:.6e}"
        )
    print("PASS")


if __name__ == "__main__":
    main()
