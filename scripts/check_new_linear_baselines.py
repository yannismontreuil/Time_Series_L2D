import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.router_model import feature_phi
from models.shared_linear_bandits import (
    LinearEnsembleSampling,
    LinearThompsonSampling,
    SharedLinUCB,
)


def _run_baseline(
    baseline,
    theta_true: np.ndarray,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_experts = int(baseline.N)
    oracle_costs = np.zeros(horizon, dtype=float)
    incurred = np.zeros(horizon, dtype=float)

    for t in range(horizon):
        x_t = rng.normal(size=3)
        available = np.arange(num_experts, dtype=int)
        phi_x = baseline._context_features(x_t)
        losses = np.zeros(num_experts, dtype=float)
        for j in range(num_experts):
            feat = baseline._joint_features(phi_x, j)
            losses[j] = 2.0 + float(theta_true @ feat) + 0.05 * rng.normal()
        losses = np.maximum(losses, 0.0)

        choice = int(baseline.select_expert(x_t, available))
        incurred[t] = float(losses[choice])
        oracle_costs[t] = float(np.min(losses))
        baseline.update(x_t, losses, available, selected_expert=choice)

    return incurred, oracle_costs


def _run_random(
    theta_true: np.ndarray,
    baseline_template,
    horizon: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_experts = int(baseline_template.N)
    costs = np.zeros(horizon, dtype=float)
    for t in range(horizon):
        x_t = rng.normal(size=3)
        phi_x = baseline_template._context_features(x_t)
        losses = np.zeros(num_experts, dtype=float)
        for j in range(num_experts):
            feat = baseline_template._joint_features(phi_x, j)
            losses[j] = 2.0 + float(theta_true @ feat) + 0.05 * rng.normal()
        losses = np.maximum(losses, 0.0)
        choice = int(rng.integers(num_experts))
        costs[t] = float(losses[choice])
    return costs


def main() -> None:
    seed = 7
    horizon = 300
    num_experts = 4

    template = SharedLinUCB(
        num_experts=num_experts,
        feature_fn=feature_phi,
        alpha_ucb=1.5,
        lambda_reg=1.0,
        feedback_mode="partial",
        context_dim=3,
        seed=seed,
    )
    theta_true = np.linspace(-0.5, 0.5, template.d)
    random_costs = _run_random(theta_true, template, horizon=horizon, seed=seed + 1)
    random_last = float(np.mean(random_costs[-50:]))

    checks = [
        (
            "SharedLinUCB-partial",
            SharedLinUCB(
                num_experts=num_experts,
                feature_fn=feature_phi,
                alpha_ucb=1.5,
                lambda_reg=1.0,
                feedback_mode="partial",
                context_dim=3,
                seed=seed,
            ),
        ),
        (
            "SharedLinUCB-full",
            SharedLinUCB(
                num_experts=num_experts,
                feature_fn=feature_phi,
                alpha_ucb=1.5,
                lambda_reg=1.0,
                feedback_mode="full",
                context_dim=3,
                seed=seed,
            ),
        ),
        (
            "LinTS-partial",
            LinearThompsonSampling(
                num_experts=num_experts,
                feature_fn=feature_phi,
                lambda_reg=1.0,
                posterior_scale=0.5,
                feedback_mode="partial",
                context_dim=3,
                seed=seed,
            ),
        ),
        (
            "LinTS-full",
            LinearThompsonSampling(
                num_experts=num_experts,
                feature_fn=feature_phi,
                lambda_reg=1.0,
                posterior_scale=0.5,
                feedback_mode="full",
                context_dim=3,
                seed=seed,
            ),
        ),
        (
            "Ensemble-partial",
            LinearEnsembleSampling(
                num_experts=num_experts,
                feature_fn=feature_phi,
                ensemble_size=12,
                lambda_reg=1.0,
                obs_noise_std=1.0,
                feedback_mode="partial",
                context_dim=3,
                seed=seed,
            ),
        ),
        (
            "Ensemble-full",
            LinearEnsembleSampling(
                num_experts=num_experts,
                feature_fn=feature_phi,
                ensemble_size=12,
                lambda_reg=1.0,
                obs_noise_std=1.0,
                feedback_mode="full",
                context_dim=3,
                seed=seed,
            ),
        ),
    ]

    failures = []
    for label, baseline in checks:
        costs, oracle_costs = _run_baseline(
            baseline,
            theta_true=theta_true,
            horizon=horizon,
            seed=seed + 10,
        )
        if not np.all(np.isfinite(costs)):
            failures.append(f"{label}: non-finite costs")
            continue
        last_cost = float(np.mean(costs[-50:]))
        oracle_last = float(np.mean(oracle_costs[-50:]))
        print(
            f"{label}: last50={last_cost:.4f}, oracle_last50={oracle_last:.4f}, "
            f"random_last50={random_last:.4f}"
        )
        if last_cost >= random_last:
            failures.append(
                f"{label}: last-window mean cost {last_cost:.4f} >= random {random_last:.4f}"
            )

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        raise SystemExit(1)

    print("PASS: new shared linear baselines improved over random in the toy check.")


if __name__ == "__main__":
    main()
