import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from router_model import SLDSIMMRouter
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import LearningToDeferBaseline
from router_eval import (
    run_router_on_env,
    run_l2d_on_env,
    compute_predictions_from_choices,
)


def plot_time_series(
    env: SyntheticTimeSeriesEnv,
    num_points: Optional[int] = None,
) -> None:
    """
    Plot the true time series y_t together with each expert's prediction,
    and the underlying regime z_t and expert availability.
    """
    T = env.T if num_points is None else min(num_points, env.T)
    t_grid = np.arange(T)

    # True target and regimes
    y = env.y[:T]
    z = env.z[:T]

    # Expert predictions for each time step
    preds = np.zeros((T, env.num_experts), dtype=float)
    for t in range(T):
        x_t = env.get_context(t)
        preds[t, :] = env.all_expert_predictions(x_t)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Top: true series and expert predictions
    ax1.plot(t_grid, y, label="True $y_t$", color="black", linewidth=2)
    for j in range(env.num_experts):
        ax1.plot(
            t_grid,
            preds[:, j],
            label=f"Expert {j} prediction",
            alpha=0.7,
        )
    ax1.set_ylabel("Value")
    ax1.set_title("Synthetic time series and expert predictions")
    ax1.legend(loc="upper left")

    # Bottom: regime sequence
    ax2.step(t_grid, z, where="post", label="Regime $z_t$")
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("Regime")
    unique_z = np.unique(z)
    ax2.set_yticks(unique_z)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    # Plot expert availability over time (1 = available, 0 = not available)
    avail = getattr(env, "availability", None)
    if avail is not None:
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        # Availability is defined for t=0,...,T-1; align with t=1,...,T-1.
        t_grid_avail = np.arange(1, T)
        avail_sub = avail[1:T, :]
        for j in range(env.num_experts):
            ax3.step(
                t_grid_avail,
                avail_sub[:, j],
                where="post",
                label=f"Expert {j}",
            )
        ax3.set_xlabel("Time $t$")
        ax3.set_ylabel("Availability")
        ax3.set_yticks([0, 1])
        ax3.set_title("Expert availability over time (1 = available, 0 = not)")
        ax3.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


def evaluate_routers_and_baselines(
    env: SyntheticTimeSeriesEnv,
    router_partial: SLDSIMMRouter,
    router_full: SLDSIMMRouter,
    l2d_baseline: Optional[LearningToDeferBaseline] = None,
) -> None:
    """
    Evaluate how the partial- and full-feedback routers behave on the
    environment, compare their average cost to constant-expert baselines,
    and plot the induced prediction time series and selections.
    """
    # Run both routers to obtain costs and choices
    costs_partial, choices_partial = run_router_on_env(router_partial, env)
    costs_full, choices_full = run_router_on_env(router_full, env)

    # Run learning-to-defer baseline if provided
    if l2d_baseline is not None:
        costs_l2d, choices_l2d = run_l2d_on_env(l2d_baseline, env)
    else:
        costs_l2d, choices_l2d = None, None

    # Prediction series induced by router and L2D choices
    preds_partial = compute_predictions_from_choices(env, choices_partial)
    preds_full = compute_predictions_from_choices(env, choices_full)
    preds_l2d = (
        compute_predictions_from_choices(env, choices_l2d)
        if choices_l2d is not None
        else None
    )

    T = env.T
    t_grid = np.arange(1, T)
    y_true = env.y[1:T]

    # Constant-expert baselines (always pick the same expert)
    beta = router_partial.beta[: env.num_experts]
    cum_costs = np.zeros(env.num_experts, dtype=float)
    for t in range(1, T):
        loss_all = env.losses(t)
        cum_costs += loss_all + beta
    avg_cost_experts = cum_costs / (T - 1)

    avg_cost_partial = costs_partial.mean()
    avg_cost_full = costs_full.mean()
    avg_cost_l2d = costs_l2d.mean() if costs_l2d is not None else None

    print("=== Average costs ===")
    print(f"Router (partial feedback): {avg_cost_partial:.4f}")
    print(f"Router (full feedback):    {avg_cost_full:.4f}")
    if avg_cost_l2d is not None:
        print(f"L2D baseline:              {avg_cost_l2d:.4f}")
    for j in range(env.num_experts):
        print(f"Always using expert {j}:   {avg_cost_experts[j]:.4f}")

    # Selection distribution (how often each expert is chosen)
    entries = [("partial", choices_partial), ("full", choices_full)]
    if choices_l2d is not None:
        entries.append(("l2d", choices_l2d))
    for name, choices in entries:
        values, counts = np.unique(choices, return_counts=True)
        freqs = counts / choices.shape[0]
        print(f"Selection distribution ({name} router):")
        for v, c, f in zip(values, counts, freqs):
            print(f"  expert {int(v)}: count={int(c)}, freq={f:.3f}")

    # Plot true series vs router-based prediction series
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_grid, y_true, label="True $y_t$", color="black", linewidth=2)
    ax.plot(t_grid, preds_partial, label="Router (partial)", alpha=0.8)
    ax.plot(t_grid, preds_full, label="Router (full)", alpha=0.8)
    if preds_l2d is not None:
        ax.plot(t_grid, preds_l2d, label="L2D baseline", alpha=0.8)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Value")
    ax.set_title("True series vs router-induced predictions")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Plot the expert index chosen over time for each router and baseline,
    # together with expert availability (0 = not available, 1 = available).
    avail = getattr(env, "availability", None)
    has_l2d = choices_l2d is not None
    has_avail = avail is not None

    n_rows = 2 + (1 if has_l2d else 0) + (1 if has_avail else 0)
    fig2, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10, 2 * n_rows))

    idx = 0
    ax_p = axes[idx]
    ax_p.step(t_grid, choices_partial, where="post")
    ax_p.set_ylabel("Expert\n(partial)")
    ax_p.set_yticks(np.arange(env.num_experts))
    ax_p.set_title("Selections and availability over time")
    idx += 1

    ax_f = axes[idx]
    ax_f.step(t_grid, choices_full, where="post", color="tab:orange")
    ax_f.set_ylabel("Expert\n(full)")
    ax_f.set_yticks(np.arange(env.num_experts))
    idx += 1

    if has_l2d:
        ax_l2d = axes[idx]
        ax_l2d.step(t_grid, choices_l2d, where="post", color="tab:green")
        ax_l2d.set_ylabel("Expert\n(L2D)")
        ax_l2d.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_avail:
        ax_avail = axes[idx]
        t_grid_avail = np.arange(1, T)
        avail_sub = avail[1:T, :]
        for j in range(env.num_experts):
            ax_avail.step(
                t_grid_avail,
                avail_sub[:, j],
                where="post",
                label=f"Expert {j}",
            )
        ax_avail.set_ylabel("Avail.")
        ax_avail.set_yticks([0, 1])
        ax_avail.set_xlabel("Time $t$")
        ax_avail.legend(loc="upper right")
    else:
        axes[idx - 1].set_xlabel("Time $t$")

    plt.tight_layout()
    plt.show()

