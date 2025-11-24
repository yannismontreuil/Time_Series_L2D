import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from router_model import SLDSIMMRouter
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import LearningToDeferBaseline
from router_eval import (
    run_router_on_env,
    run_l2d_on_env,
    run_random_on_env,
    run_oracle_on_env,
    compute_predictions_from_choices,
)

# ---------------------------------------------------------------------
# Global plotting style (scienceplots + IEEE, Times New Roman, etc.)
# ---------------------------------------------------------------------
try:
    import scienceplots  # type: ignore  # noqa: F401

    plt.style.use(["science", "ieee"])
except Exception:
    # If scienceplots is not installed, fall back silently; font settings
    # below still apply.
    pass

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 13,
        "figure.dpi": 100,
    }
)


def get_expert_color(j: int) -> str:
    """
    Deterministic color for expert index j, shared across plots.

    Uses matplotlib's default color cycle (C0, C1, ...) so that
    expert j always has the same color in every figure.
    """
    return f"C{int(j) % 10}"


def get_model_color(name: str) -> str:
    """
    Consistent colors for different models across all plots.
    """
    name = str(name).lower()
    mapping = {
        "true": "black",
        "oracle": "tab:gray",
        "partial": "tab:blue",
        "full": "tab:orange",
        "l2d": "tab:red",
        "random": "tab:green",
    }
    return mapping.get(name, "tab:purple")


def add_unavailability_regions(ax: plt.Axes, env: SyntheticTimeSeriesEnv) -> None:
    """
    Shade time intervals where experts are unavailable in a given axis.

    For each expert j, we draw light-colored vertical bands over times
    t where env.availability[t, j] == 0, and annotate them with
    \"j unavailable\" near the bottom of the plot.
    """
    avail = getattr(env, "availability", None)
    if avail is None:
        return
    T = env.T
    if T <= 1 or env.num_experts <= 0:
        return

    # Work on t = 1,...,T-1 to match the prediction plots.
    avail_sub = avail[1:T, :]
    y_min, y_max = ax.get_ylim()
    height = y_max - y_min if y_max > y_min else 1.0

    for j in range(env.num_experts):
        series = avail_sub[:, j]
        in_block = False
        start_t = None
        for idx, val in enumerate(series):
            t = idx + 1  # actual time index
            if not in_block and val == 0:
                in_block = True
                start_t = t
            elif in_block and val == 1:
                end_t = t
                color = get_expert_color(j)
                ax.axvspan(
                    start_t,
                    end_t,
                    facecolor=color,
                    alpha=0.08,
                    zorder=0,
                )
                mid = 0.5 * (start_t + end_t)
                ax.text(
                    mid,
                    y_min + 0.05 * height,
                    f"{j} unavailable",
                    color=color,
                    alpha=0.8,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    zorder=1,
                )
                in_block = False
        if in_block and start_t is not None:
            end_t = T - 1
            color = get_expert_color(j)
            ax.axvspan(
                start_t,
                end_t,
                facecolor=color,
                alpha=0.08,
                zorder=0,
            )
            mid = 0.5 * (start_t + end_t)
            ax.text(
                mid,
                y_min + 0.05 * height,
                f"{j} unavailable",
                color=color,
                alpha=0.8,
                ha="center",
                va="bottom",
                fontsize=9,
                zorder=1,
            )

    # Keep original y-limits
    ax.set_ylim(y_min, y_max)


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
    ax1.plot(
        t_grid,
        y,
        label="True $y_t$",
        color=get_model_color("true"),
        linewidth=2,
    )
    for j in range(env.num_experts):
        ax1.plot(
            t_grid,
            preds[:, j],
            label=f"Expert {j} prediction",
            color=get_expert_color(j),
            linestyle="--",
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
        # Availability is defined for t=0,...,T-1; align with t=1,...,T-1.
        t_grid_avail = np.arange(1, T)
        avail_sub = avail[1:T, :]

        # One subplot per expert for clarity.
        n_experts = env.num_experts
        fig3, axes = plt.subplots(
            n_experts, 1, sharex=True, figsize=(10, 1.5 * max(n_experts, 1))
        )
        if n_experts == 1:
            axes = [axes]

        for j in range(n_experts):
            ax_j = axes[j]
            ax_j.step(
                t_grid_avail,
                avail_sub[:, j],
                where="post",
                color=get_expert_color(j),
                linestyle="--",
            )
            ax_j.set_ylabel(f"Exp {j}")
            ax_j.set_yticks([0, 1])
            if j == 0:
                ax_j.set_title(
                    "Expert availability over time (1 = available, 0 = not)"
                )

        axes[-1].set_xlabel("Time $t$")
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

    # Common consultation costs (assumed shared across methods)
    beta = router_partial.beta[: env.num_experts]

    # Random and oracle baselines
    costs_random, choices_random = run_random_on_env(env, beta, seed=0)
    costs_oracle, choices_oracle = run_oracle_on_env(env, beta)

    # Prediction series induced by router and L2D choices
    preds_partial = compute_predictions_from_choices(env, choices_partial)
    preds_full = compute_predictions_from_choices(env, choices_full)
    preds_l2d = (
        compute_predictions_from_choices(env, choices_l2d)
        if choices_l2d is not None
        else None
    )
    preds_random = compute_predictions_from_choices(env, choices_random)
    preds_oracle = compute_predictions_from_choices(env, choices_oracle)

    T = env.T
    t_grid = np.arange(1, T)
    y_true = env.y[1:T]

    # Constant-expert baselines (always pick the same expert)
    cum_costs = np.zeros(env.num_experts, dtype=float)
    for t in range(1, T):
        loss_all = env.losses(t)
        cum_costs += loss_all + beta
    avg_cost_experts = cum_costs / (T - 1)

    avg_cost_partial = costs_partial.mean()
    avg_cost_full = costs_full.mean()
    avg_cost_l2d = costs_l2d.mean() if costs_l2d is not None else None
    avg_cost_random = costs_random.mean()
    avg_cost_oracle = costs_oracle.mean()

    print("=== Average costs ===")
    print(f"Router (partial feedback): {avg_cost_partial:.4f}")
    print(f"Router (full feedback):    {avg_cost_full:.4f}")
    if avg_cost_l2d is not None:
        print(f"L2D baseline:              {avg_cost_l2d:.4f}")
    print(f"Random baseline:           {avg_cost_random:.4f}")
    print(f"Oracle baseline:           {avg_cost_oracle:.4f}")
    for j in range(env.num_experts):
        print(f"Always using expert {j}:   {avg_cost_experts[j]:.4f}")

    # Selection distribution (how often each expert is chosen)
    entries = [
        ("partial", choices_partial),
        ("full", choices_full),
        ("random", choices_random),
        ("oracle", choices_oracle),
    ]
    if choices_l2d is not None:
        entries.append(("l2d", choices_l2d))
    for name, choices in entries:
        values, counts = np.unique(choices, return_counts=True)
        freqs = counts / choices.shape[0]
        print(f"Selection distribution ({name}):")
        for v, c, f in zip(values, counts, freqs):
            print(f"  expert {int(v)}: count={int(c)}, freq={f:.3f}")

    # Plot true series vs router-based prediction series (top)
    # and cumulative costs over time for each baseline (bottom).
    fig, (ax_pred, ax_cost) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    # Top subplot: predictions
    ax_pred.plot(
        t_grid,
        y_true,
        label="True $y_t$",
        color=get_model_color("true"),
        linewidth=2,
        linestyle="-",
    )
    ax_pred.plot(
        t_grid,
        preds_partial,
        label="Router (partial)",
        color=get_model_color("partial"),
        linestyle="-",
        alpha=0.8,
    )
    ax_pred.plot(
        t_grid,
        preds_full,
        label="Router (full)",
        color=get_model_color("full"),
        linestyle="-",
        alpha=0.8,
    )
    if preds_l2d is not None:
        ax_pred.plot(
            t_grid,
            preds_l2d,
            label="L2D baseline",
            color=get_model_color("l2d"),
            linestyle="-",
            alpha=0.8,
        )
    ax_pred.plot(
        t_grid,
        preds_random,
        label="Random baseline",
        color=get_model_color("random"),
        linestyle="-",
        alpha=0.7,
    )
    ax_pred.plot(
        t_grid,
        preds_oracle,
        label="Oracle baseline",
        color=get_model_color("oracle"),
        linestyle="-",
        alpha=0.9,
    )

    # Shade intervals where experts are unavailable
    add_unavailability_regions(ax_pred, env)
    ax_pred.set_ylabel("Value")
    ax_pred.set_title("True series vs router-induced predictions")
    ax_pred.legend(loc="upper left")

    # Bottom subplot: cumulative costs for all baselines
    cum_partial = np.cumsum(costs_partial)
    cum_full = np.cumsum(costs_full)
    cum_random = np.cumsum(costs_random)
    cum_oracle = np.cumsum(costs_oracle)
    cum_l2d = np.cumsum(costs_l2d) if costs_l2d is not None else None

    ax_cost.plot(
        t_grid,
        cum_partial,
        label="Partial (cumulative cost)",
        color=get_model_color("partial"),
        linestyle="-",
    )
    ax_cost.plot(
        t_grid,
        cum_full,
        label="Full (cumulative cost)",
        color=get_model_color("full"),
        linestyle="-",
    )
    if cum_l2d is not None:
        ax_cost.plot(
            t_grid,
            cum_l2d,
            label="L2D (cumulative cost)",
            color=get_model_color("l2d"),
            linestyle="-",
        )
    ax_cost.plot(
        t_grid,
        cum_random,
        label="Random (cumulative cost)",
        color=get_model_color("random"),
        linestyle="-",
    )
    ax_cost.plot(
        t_grid,
        cum_oracle,
        label="Oracle (cumulative cost)",
        color=get_model_color("oracle"),
        linestyle="-",
    )

    ax_cost.set_xlabel("Time $t$")
    ax_cost.set_ylabel("Cumulative cost")
    ax_cost.set_title("Cumulative cost over time")
    ax_cost.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    # Plot the expert index chosen over time for each router and baseline,
    # together with expert availability (0 = not available, 1 = available).
    avail = getattr(env, "availability", None)
    has_l2d = choices_l2d is not None
    has_avail = avail is not None

    # Rows: partial router, full router, optional L2D baseline,
    # random baseline, oracle baseline, and optional availability.
    n_rows = 4 + (1 if has_l2d else 0) + (1 if has_avail else 0)
    fig2, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10, 2 * n_rows))

    idx = 0
    ax_p = axes[idx]
    ax_p.step(
        t_grid,
        choices_partial,
        where="post",
        color=get_model_color("partial"),
    )
    ax_p.set_ylabel("Expert\n(partial)")
    ax_p.set_yticks(np.arange(env.num_experts))
    ax_p.set_title("Selections and availability over time")
    idx += 1

    ax_f = axes[idx]
    ax_f.step(
        t_grid,
        choices_full,
        where="post",
        color=get_model_color("full"),
    )
    ax_f.set_ylabel("Expert\n(full)")
    ax_f.set_yticks(np.arange(env.num_experts))
    idx += 1

    if has_l2d:
        ax_l2d = axes[idx]
        ax_l2d.step(
            t_grid,
            choices_l2d,
            where="post",
            color=get_model_color("l2d"),
        )
        ax_l2d.set_ylabel("Expert\n(L2D)")
        ax_l2d.set_yticks(np.arange(env.num_experts))
        idx += 1

    ax_rand = axes[idx]
    ax_rand.step(
        t_grid,
        choices_random,
        where="post",
        color=get_model_color("random"),
    )
    ax_rand.set_ylabel("Expert\n(random)")
    ax_rand.set_yticks(np.arange(env.num_experts))
    idx += 1

    ax_oracle = axes[idx]
    ax_oracle.step(
        t_grid,
        choices_oracle,
        where="post",
        color=get_model_color("oracle"),
    )
    ax_oracle.set_ylabel("Expert\n(oracle)")
    ax_oracle.set_yticks(np.arange(env.num_experts))
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
                color=get_expert_color(j),
            )
        ax_avail.set_ylabel("Avail.")
        ax_avail.set_yticks([0, 1])
        ax_avail.set_xlabel("Time $t$")
        ax_avail.legend(loc="upper right")
    else:
        axes[idx - 1].set_xlabel("Time $t$")

    plt.tight_layout()
    plt.show()
