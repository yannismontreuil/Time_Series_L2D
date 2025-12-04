import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from router_model import SLDSIMMRouter
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import L2D
from router_eval import (
    run_router_on_env,
    run_l2d_on_env,
    run_random_on_env,
    run_oracle_on_env,
    compute_predictions_from_choices,
    run_linucb_on_env,
    run_neuralucb_on_env,
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

    Uses a fixed qualitative palette (tab colors) so that
    experts up to at least 10 have clearly distinguishable
    colors. This avoids accidental reuse of visually similar
    colors (e.g., for expert 0 and expert 4).
    """
    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    return palette[int(j) % len(palette)]


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
        "partial_corr": "tab:cyan",
        "full_corr": "tab:olive",
        "partial_corr_em": "tab:blue",
        "full_corr_em": "tab:red",
        "neural_partial": "tab:brown",
        "neural_full": "tab:purple",
        "l2d": "tab:red",
        "l2d_sw": "tab:pink",
        "linucb_partial": "tab:olive",
        "linucb_full": "tab:cyan",
        "neuralucb": "tab:brown",
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
    l2d_baseline: Optional[L2D] = None,
    router_partial_corr=None,
    router_full_corr=None,
    router_partial_corr_em=None,
    router_full_corr_em=None,
    l2d_sw_baseline: Optional[L2D] = None,
    linucb_partial=None,
    linucb_full=None,
    neuralucb_partial=None,
    neuralucb_full=None,
    router_partial_neural=None,
    router_full_neural=None,
) -> None:
    """
    Evaluate how the partial- and full-feedback routers behave on the
    environment, compare their average cost to constant-expert baselines,
    and plot the induced prediction time series and selections.
    """
    # Run both base routers to obtain costs and choices
    costs_partial, choices_partial = run_router_on_env(router_partial, env)
    costs_full, choices_full = run_router_on_env(router_full, env)

    # Neural routers (if provided)
    if router_partial_neural is not None:
        costs_partial_neural, choices_partial_neural = run_router_on_env(
            router_partial_neural, env
        )
    else:
        costs_partial_neural, choices_partial_neural = None, None

    if router_full_neural is not None:
        costs_full_neural, choices_full_neural = run_router_on_env(
            router_full_neural, env
        )
    else:
        costs_full_neural, choices_full_neural = None, None

    # Correlated routers (if provided)
    if router_partial_corr is not None:
        costs_partial_corr, choices_partial_corr = run_router_on_env(
            router_partial_corr, env
        )
    else:
        costs_partial_corr, choices_partial_corr = None, None

    if router_full_corr is not None:
        costs_full_corr, choices_full_corr = run_router_on_env(
            router_full_corr, env
        )
    else:
        costs_full_corr, choices_full_corr = None, None

    # EM-capable correlated routers (if provided)
    if router_partial_corr_em is not None:
        costs_partial_corr_em, choices_partial_corr_em = run_router_on_env(
            router_partial_corr_em, env
        )
    else:
        costs_partial_corr_em, choices_partial_corr_em = None, None

    if router_full_corr_em is not None:
        costs_full_corr_em, choices_full_corr_em = run_router_on_env(
            router_full_corr_em, env
        )
    else:
        costs_full_corr_em, choices_full_corr_em = None, None

    # Run learning-to-defer baselines if provided
    if l2d_baseline is not None:
        costs_l2d, choices_l2d = run_l2d_on_env(l2d_baseline, env)
    else:
        costs_l2d, choices_l2d = None, None

    if l2d_sw_baseline is not None:
        costs_l2d_sw, choices_l2d_sw = run_l2d_on_env(l2d_sw_baseline, env)
    else:
        costs_l2d_sw, choices_l2d_sw = None, None

    # Run LinUCB baselines if provided
    if linucb_partial is not None:
        costs_linucb_partial, choices_linucb_partial = run_linucb_on_env(
            linucb_partial, env
        )
    else:
        costs_linucb_partial, choices_linucb_partial = None, None

    if linucb_full is not None:
        costs_linucb_full, choices_linucb_full = run_linucb_on_env(
            linucb_full, env
        )
    else:
        costs_linucb_full, choices_linucb_full = None, None

    # Run NeuralUCB baselines if provided
    if neuralucb_partial is not None:
        costs_neuralucb_partial, choices_neuralucb_partial = run_neuralucb_on_env(
            neuralucb_partial, env
        )
    else:
        costs_neuralucb_partial, choices_neuralucb_partial = None, None

    if neuralucb_full is not None:
        costs_neuralucb_full, choices_neuralucb_full = run_neuralucb_on_env(
            neuralucb_full, env
        )
    else:
        costs_neuralucb_full, choices_neuralucb_full = None, None

    # Common consultation costs (assumed shared across methods)
    beta = router_partial.beta[: env.num_experts]

    # Random and oracle baselines
    costs_random, choices_random = run_random_on_env(env, beta, seed=0)
    costs_oracle, choices_oracle = run_oracle_on_env(env, beta)

    # Prediction series induced by router and L2D choices
    preds_partial = compute_predictions_from_choices(env, choices_partial)
    preds_full = compute_predictions_from_choices(env, choices_full)
    preds_partial_corr = (
        compute_predictions_from_choices(env, choices_partial_corr)
        if choices_partial_corr is not None
        else None
    )
    preds_full_corr = (
        compute_predictions_from_choices(env, choices_full_corr)
        if choices_full_corr is not None
        else None
    )
    preds_partial_corr_em = (
        compute_predictions_from_choices(env, choices_partial_corr_em)
        if choices_partial_corr_em is not None
        else None
    )
    preds_full_corr_em = (
        compute_predictions_from_choices(env, choices_full_corr_em)
        if choices_full_corr_em is not None
        else None
    )
    preds_partial_neural = (
        compute_predictions_from_choices(env, choices_partial_neural)
        if choices_partial_neural is not None
        else None
    )
    preds_full_neural = (
        compute_predictions_from_choices(env, choices_full_neural)
        if choices_full_neural is not None
        else None
    )
    preds_l2d = (
        compute_predictions_from_choices(env, choices_l2d)
        if choices_l2d is not None
        else None
    )
    preds_l2d_sw = (
        compute_predictions_from_choices(env, choices_l2d_sw)
        if choices_l2d_sw is not None
        else None
    )
    preds_linucb_partial = (
        compute_predictions_from_choices(env, choices_linucb_partial)
        if choices_linucb_partial is not None
        else None
    )
    preds_linucb_full = (
        compute_predictions_from_choices(env, choices_linucb_full)
        if choices_linucb_full is not None
        else None
    )
    preds_neuralucb_partial = (
        compute_predictions_from_choices(env, choices_neuralucb_partial)
        if choices_neuralucb_partial is not None
        else None
    )
    preds_neuralucb_full = (
        compute_predictions_from_choices(env, choices_neuralucb_full)
        if choices_neuralucb_full is not None
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
    avg_cost_partial_corr = (
        costs_partial_corr.mean() if costs_partial_corr is not None else None
    )
    avg_cost_full_corr = (
        costs_full_corr.mean() if costs_full_corr is not None else None
    )
    avg_cost_partial_corr_em = (
        costs_partial_corr_em.mean() if costs_partial_corr_em is not None else None
    )
    avg_cost_full_corr_em = (
        costs_full_corr_em.mean() if costs_full_corr_em is not None else None
    )
    avg_cost_neural_partial = (
        costs_partial_neural.mean() if costs_partial_neural is not None else None
    )
    avg_cost_neural_full = (
        costs_full_neural.mean() if costs_full_neural is not None else None
    )
    avg_cost_l2d = costs_l2d.mean() if costs_l2d is not None else None
    avg_cost_l2d_sw = costs_l2d_sw.mean() if costs_l2d_sw is not None else None
    avg_cost_linucb_partial = (
        costs_linucb_partial.mean() if costs_linucb_partial is not None else None
    )
    avg_cost_linucb_full = (
        costs_linucb_full.mean() if costs_linucb_full is not None else None
    )
    avg_cost_neuralucb_partial = (
        costs_neuralucb_partial.mean()
        if costs_neuralucb_partial is not None
        else None
    )
    avg_cost_neuralucb_full = (
        costs_neuralucb_full.mean()
        if costs_neuralucb_full is not None
        else None
    )
    avg_cost_random = costs_random.mean()
    avg_cost_oracle = costs_oracle.mean()

    print("=== Average costs ===")
    print(f"Router (partial feedback):      {avg_cost_partial:.4f}")
    print(f"Router (full feedback):         {avg_cost_full:.4f}")
    if avg_cost_partial_corr is not None:
        print(
            f"Router Corr (partial feedback): {avg_cost_partial_corr:.4f}"
        )
    if avg_cost_full_corr is not None:
        print(f"Router Corr (full feedback):    {avg_cost_full_corr:.4f}")
    if avg_cost_partial_corr_em is not None:
        print(
            f"Router Corr EM (partial fb):   {avg_cost_partial_corr_em:.4f}"
        )
    if avg_cost_full_corr_em is not None:
        print(
            f"Router Corr EM (full fb):      {avg_cost_full_corr_em:.4f}"
        )
    if avg_cost_neural_partial is not None:
        print(
            f"Neural router (partial fb):     {avg_cost_neural_partial:.4f}"
        )
    if avg_cost_neural_full is not None:
        print(
            f"Neural router (full fb):        {avg_cost_neural_full:.4f}"
        )
    if avg_cost_l2d is not None:
        print(f"L2D baseline:                  {avg_cost_l2d:.4f}")
    if avg_cost_l2d_sw is not None:
        print(f"L2D_SW baseline:               {avg_cost_l2d_sw:.4f}")
    if avg_cost_linucb_partial is not None:
        print(f"LinUCB (partial feedback):     {avg_cost_linucb_partial:.4f}")
    if avg_cost_linucb_full is not None:
        print(f"LinUCB (full feedback):        {avg_cost_linucb_full:.4f}")
    if avg_cost_neuralucb_partial is not None:
        print(f"NeuralUCB (partial feedback):  {avg_cost_neuralucb_partial:.4f}")
    if avg_cost_neuralucb_full is not None:
        print(f"NeuralUCB (full feedback):     {avg_cost_neuralucb_full:.4f}")
    print(f"Random baseline:               {avg_cost_random:.4f}")
    print(f"Oracle baseline:               {avg_cost_oracle:.4f}")
    for j in range(env.num_experts):
        print(f"Always using expert {j}:       {avg_cost_experts[j]:.4f}")

    # Selection distribution (how often each expert is chosen)
    entries = [
        ("partial", choices_partial),
        ("full", choices_full),
    ]
    if choices_partial_corr is not None:
        entries.append(("partial_corr", choices_partial_corr))
    if choices_full_corr is not None:
        entries.append(("full_corr", choices_full_corr))
    if choices_partial_corr_em is not None:
        entries.append(("partial_corr_em", choices_partial_corr_em))
    if choices_full_corr_em is not None:
        entries.append(("full_corr_em", choices_full_corr_em))
    entries.extend(
        [
            ("random", choices_random),
            ("oracle", choices_oracle),
        ]
    )
    if choices_partial_neural is not None:
        entries.append(("neural_partial", choices_partial_neural))
    if choices_full_neural is not None:
        entries.append(("neural_full", choices_full_neural))
    if choices_l2d is not None:
        entries.append(("l2d", choices_l2d))
    if choices_l2d_sw is not None:
        entries.append(("l2d_sw", choices_l2d_sw))
    if choices_linucb_partial is not None:
        entries.append(("linucb_partial", choices_linucb_partial))
    if choices_linucb_full is not None:
        entries.append(("linucb_full", choices_linucb_full))
    if choices_neuralucb_partial is not None:
        entries.append(("neuralucb_partial", choices_neuralucb_partial))
    if choices_neuralucb_full is not None:
        entries.append(("neuralucb_full", choices_neuralucb_full))
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
    if preds_partial_neural is not None:
        ax_pred.plot(
            t_grid,
            preds_partial_neural,
            label="Neural router (partial)",
            color=get_model_color("neural_partial"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_full_neural is not None:
        ax_pred.plot(
            t_grid,
            preds_full_neural,
            label="Neural router (full)",
            color=get_model_color("neural_full"),
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
    if preds_l2d_sw is not None:
        ax_pred.plot(
            t_grid,
            preds_l2d_sw,
            label="L2D_SW baseline",
            color=get_model_color("l2d_sw"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_linucb_partial is not None:
        ax_pred.plot(
            t_grid,
            preds_linucb_partial,
            label="LinUCB (partial)",
            color=get_model_color("linucb_partial"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_linucb_full is not None:
        ax_pred.plot(
            t_grid,
            preds_linucb_full,
            label="LinUCB (full)",
            color=get_model_color("linucb_full"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_neuralucb_partial is not None:
        ax_pred.plot(
            t_grid,
            preds_neuralucb_partial,
            label="NeuralUCB (partial)",
            color=get_model_color("neuralucb"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_neuralucb_full is not None:
        ax_pred.plot(
            t_grid,
            preds_neuralucb_full,
            label="NeuralUCB (full)",
            color=get_model_color("neuralucb"),
            linestyle="--",
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

    # Bottom subplot: running average costs for all baselines
    denom = np.arange(1, T, dtype=float)
    avg_partial_t = np.cumsum(costs_partial) / denom
    avg_full_t = np.cumsum(costs_full) / denom
    avg_partial_corr_t = (
        np.cumsum(costs_partial_corr) / denom
        if costs_partial_corr is not None
        else None
    )
    avg_full_corr_t = (
        np.cumsum(costs_full_corr) / denom if costs_full_corr is not None else None
    )
    avg_partial_corr_em_t = (
        np.cumsum(costs_partial_corr_em) / denom
        if costs_partial_corr_em is not None
        else None
    )
    avg_full_corr_em_t = (
        np.cumsum(costs_full_corr_em) / denom
        if costs_full_corr_em is not None
        else None
    )
    avg_neural_partial_t = (
        np.cumsum(costs_partial_neural) / denom
        if costs_partial_neural is not None
        else None
    )
    avg_neural_full_t = (
        np.cumsum(costs_full_neural) / denom
        if costs_full_neural is not None
        else None
    )
    avg_random_t = np.cumsum(costs_random) / denom
    avg_oracle_t = np.cumsum(costs_oracle) / denom
    avg_l2d_t = (
        np.cumsum(costs_l2d) / denom if costs_l2d is not None else None
    )
    avg_l2d_sw_t = (
        np.cumsum(costs_l2d_sw) / denom if costs_l2d_sw is not None else None
    )
    avg_linucb_partial_t = (
        np.cumsum(costs_linucb_partial) / denom
        if costs_linucb_partial is not None
        else None
    )
    avg_linucb_full_t = (
        np.cumsum(costs_linucb_full) / denom
        if costs_linucb_full is not None
        else None
    )
    avg_neuralucb_partial_t = (
        np.cumsum(costs_neuralucb_partial) / denom
        if costs_neuralucb_partial is not None
        else None
    )
    avg_neuralucb_full_t = (
        np.cumsum(costs_neuralucb_full) / denom
        if costs_neuralucb_full is not None
        else None
    )

    ax_cost.plot(
        t_grid,
        avg_partial_t,
        label="Partial (avg cost)",
        color=get_model_color("partial"),
        linestyle="-",
    )
    ax_cost.plot(
        t_grid,
        avg_full_t,
        label="Full (avg cost)",
        color=get_model_color("full"),
        linestyle="-",
    )
    if avg_neural_partial_t is not None:
        ax_cost.plot(
            t_grid,
            avg_neural_partial_t,
            label="Neural partial (avg cost)",
            color=get_model_color("neural_partial"),
            linestyle="-",
        )
    if avg_neural_full_t is not None:
        ax_cost.plot(
            t_grid,
            avg_neural_full_t,
            label="Neural full (avg cost)",
            color=get_model_color("neural_full"),
            linestyle="-",
        )
    if avg_partial_corr_t is not None:
        ax_cost.plot(
            t_grid,
            avg_partial_corr_t,
            label="Partial Corr (avg cost)",
            color=get_model_color("partial_corr"),
            linestyle="-",
        )
    if avg_full_corr_t is not None:
        ax_cost.plot(
            t_grid,
            avg_full_corr_t,
            label="Full Corr (avg cost)",
            color=get_model_color("full_corr"),
            linestyle="-",
        )
    if avg_partial_corr_em_t is not None:
        ax_cost.plot(
            t_grid,
            avg_partial_corr_em_t,
            label="Partial Corr EM (avg cost)",
            color=get_model_color("partial_corr_em"),
            linestyle="-",
        )
    if avg_full_corr_em_t is not None:
        ax_cost.plot(
            t_grid,
            avg_full_corr_em_t,
            label="Full Corr EM (avg cost)",
            color=get_model_color("full_corr_em"),
            linestyle="-",
        )
    if avg_l2d_t is not None:
        ax_cost.plot(
            t_grid,
            avg_l2d_t,
            label="L2D (avg cost)",
            color=get_model_color("l2d"),
            linestyle="-",
        )
    if avg_l2d_sw_t is not None:
        ax_cost.plot(
            t_grid,
            avg_l2d_sw_t,
            label="L2D_SW (avg cost)",
            color=get_model_color("l2d_sw"),
            linestyle="-",
        )
    if avg_linucb_partial_t is not None:
        ax_cost.plot(
            t_grid,
            avg_linucb_partial_t,
            label="LinUCB (partial, avg cost)",
            color=get_model_color("linucb_partial"),
            linestyle="-",
        )
    if avg_linucb_full_t is not None:
        ax_cost.plot(
            t_grid,
            avg_linucb_full_t,
            label="LinUCB (full, avg cost)",
            color=get_model_color("linucb_full"),
            linestyle="-",
        )
    if avg_neuralucb_partial_t is not None:
        ax_cost.plot(
            t_grid,
            avg_neuralucb_partial_t,
            label="NeuralUCB (partial, avg cost)",
            color=get_model_color("neuralucb"),
            linestyle="-",
        )
    if avg_neuralucb_full_t is not None:
        ax_cost.plot(
            t_grid,
            avg_neuralucb_full_t,
            label="NeuralUCB (full, avg cost)",
            color=get_model_color("neuralucb"),
            linestyle="--",
        )
    ax_cost.plot(
        t_grid,
        avg_random_t,
        label="Random (avg cost)",
        color=get_model_color("random"),
        linestyle="-",
    )
    ax_cost.plot(
        t_grid,
        avg_oracle_t,
        label="Oracle (avg cost)",
        color=get_model_color("oracle"),
        linestyle="-",
    )

    ax_cost.set_xlabel("Time $t$")
    ax_cost.set_ylabel("Average cost")
    ax_cost.set_title("Average cost over time")
    ax_cost.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Correlated Sidekick Trap: cumulative regret from t = 2000
    # --------------------------------------------------------------
    # For the "sidekick_trap" setting, highlight the theoretical gap
    # between correlated router and baselines by plotting cumulative
    # regret from t = 2000 onward with respect to the oracle.
    if getattr(env, "setting", None) == "sidekick_trap":
        t0_regret = 2000
        idx0 = max(0, min(t0_regret - 1, T - 2))

        def cum_reg(costs: np.ndarray | None) -> np.ndarray | None:
            if costs is None:
                return None
            delta = costs[idx0:] - costs_oracle[idx0:]
            return np.cumsum(delta)

        reg_partial = cum_reg(costs_partial)
        reg_full = cum_reg(costs_full)
        reg_partial_corr = cum_reg(costs_partial_corr)
        reg_full_corr = cum_reg(costs_full_corr)
        reg_l2d = cum_reg(costs_l2d)
        reg_l2d_sw = cum_reg(costs_l2d_sw)
        reg_linucb_partial = cum_reg(costs_linucb_partial)
        reg_linucb_full = cum_reg(costs_linucb_full)

        # Numeric summary of mean costs / regret from t >= t0_regret
        def mean_cost(costs: np.ndarray | None) -> float | None:
            if costs is None:
                return None
            return float(costs[idx0:].mean())

        mean_oracle = float(costs_oracle[idx0:].mean())
        mean_partial = mean_cost(costs_partial)
        mean_full = mean_cost(costs_full)
        mean_partial_corr = mean_cost(costs_partial_corr)
        mean_full_corr = mean_cost(costs_full_corr)
        mean_l2d = mean_cost(costs_l2d)
        mean_l2d_sw = mean_cost(costs_l2d_sw)
        mean_linucb_partial = mean_cost(costs_linucb_partial)
        mean_linucb_full = mean_cost(costs_linucb_full)

        print(f"\n=== Sidekick Trap: mean costs from t={t0_regret} ===")
        print(f"Oracle (truth):                {mean_oracle:.4f}")
        print(f"Router (partial):              {mean_partial:.4f}")
        print(f"Router (full):                 {mean_full:.4f}")
        if mean_partial_corr is not None:
            print(f"Router Corr (partial):         {mean_partial_corr:.4f}")
        if mean_full_corr is not None:
            print(f"Router Corr (full):            {mean_full_corr:.4f}")
        if mean_l2d is not None:
            print(f"L2D baseline:                  {mean_l2d:.4f}")
        if mean_l2d_sw is not None:
            print(f"L2D_SW baseline:               {mean_l2d_sw:.4f}")
        if mean_linucb_partial is not None:
            print(f"LinUCB (partial):              {mean_linucb_partial:.4f}")
        if mean_linucb_full is not None:
            print(f"LinUCB (full):                 {mean_linucb_full:.4f}")

        t_reg = t_grid[idx0:]
        fig_reg, ax_reg = plt.subplots(1, 1, figsize=(10, 4))
        ax_reg.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

        ax_reg.plot(
            t_reg,
            reg_partial,
            label="Router (partial)",
            color=get_model_color("partial"),
        )
        ax_reg.plot(
            t_reg,
            reg_full,
            label="Router (full)",
            color=get_model_color("full"),
        )
        if reg_partial_corr is not None:
            ax_reg.plot(
                t_reg,
                reg_partial_corr,
                label="Router Corr (partial)",
                color=get_model_color("partial_corr"),
            )
        if reg_full_corr is not None:
            ax_reg.plot(
                t_reg,
                reg_full_corr,
                label="Router Corr (full)",
                color=get_model_color("full_corr"),
            )
        if reg_l2d is not None:
            ax_reg.plot(
                t_reg,
                reg_l2d,
                label="L2D",
                color=get_model_color("l2d"),
            )
        if reg_l2d_sw is not None:
            ax_reg.plot(
                t_reg,
                reg_l2d_sw,
                label="L2D_SW",
                color=get_model_color("l2d_sw"),
            )
        if reg_linucb_partial is not None:
            ax_reg.plot(
                t_reg,
                reg_linucb_partial,
                label="LinUCB (partial)",
                color=get_model_color("linucb_partial"),
            )
        if reg_linucb_full is not None:
            ax_reg.plot(
                t_reg,
                reg_linucb_full,
                label="LinUCB (full)",
                color=get_model_color("linucb_full"),
            )

        ax_reg.set_xlabel("Time $t$ (from 2000)")
        ax_reg.set_ylabel("Cumulative regret vs oracle")
        ax_reg.set_title("Correlated Sidekick Trap: Regret from $t=2000$")
        ax_reg.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    # Plot the expert index chosen over time for each router and baseline,
    # together with expert availability (0 = not available, 1 = available).
    avail = getattr(env, "availability", None)
    has_l2d = choices_l2d is not None
    has_l2d_sw = choices_l2d_sw is not None
    has_linucb_partial = choices_linucb_partial is not None
    has_linucb_full = choices_linucb_full is not None
    has_neuralucb_partial = choices_neuralucb_partial is not None
    has_neuralucb_full = choices_neuralucb_full is not None
    has_avail = avail is not None
    has_partial_corr = choices_partial_corr is not None
    has_full_corr = choices_full_corr is not None
    has_partial_corr_em = choices_partial_corr_em is not None
    has_full_corr_em = choices_full_corr_em is not None
    has_neural_partial = choices_partial_neural is not None
    has_neural_full = choices_full_neural is not None

    # Rows: base routers, optional correlated routers, optional L2D / L2D_SW /
    # LinUCB partial/full / NeuralUCB partial/full, random baseline,
    # oracle baseline, and optional availability.
    n_rows = 4 + (1 if has_l2d else 0) + (1 if has_l2d_sw else 0)
    n_rows += (1 if has_linucb_partial else 0) + (1 if has_linucb_full else 0)
    n_rows += (1 if has_neuralucb_partial else 0) + (1 if has_neuralucb_full else 0)
    n_rows += (1 if has_avail else 0)
    n_rows += (1 if has_partial_corr else 0) + (1 if has_full_corr else 0)
    n_rows += (1 if has_partial_corr_em else 0) + (1 if has_full_corr_em else 0)
    n_rows += (1 if has_neural_partial else 0) + (1 if has_neural_full else 0)
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

    if has_partial_corr:
        ax_pc = axes[idx]
        ax_pc.step(
            t_grid,
            choices_partial_corr,
            where="post",
            color=get_model_color("partial_corr"),
        )
        ax_pc.set_ylabel("Expert\n(partial corr)")
        ax_pc.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_full_corr:
        ax_fc = axes[idx]
        ax_fc.step(
            t_grid,
            choices_full_corr,
            where="post",
            color=get_model_color("full_corr"),
        )
        ax_fc.set_ylabel("Expert\n(full corr)")
        ax_fc.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_partial_corr_em:
        ax_pc_em = axes[idx]
        ax_pc_em.step(
            t_grid,
            choices_partial_corr_em,
            where="post",
            color=get_model_color("partial_corr_em"),
        )
        ax_pc_em.set_ylabel("Expert\n(partial corr EM)")
        ax_pc_em.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_full_corr_em:
        ax_fc_em = axes[idx]
        ax_fc_em.step(
            t_grid,
            choices_full_corr_em,
            where="post",
            color=get_model_color("full_corr_em"),
        )
        ax_fc_em.set_ylabel("Expert\n(full corr EM)")
        ax_fc_em.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_neural_partial:
        ax_np = axes[idx]
        ax_np.step(
            t_grid,
            choices_partial_neural,
            where="post",
            color=get_model_color("neural_partial"),
        )
        ax_np.set_ylabel("Expert\n(neural partial)")
        ax_np.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_neural_full:
        ax_nf = axes[idx]
        ax_nf.step(
            t_grid,
            choices_full_neural,
            where="post",
            color=get_model_color("neural_full"),
        )
        ax_nf.set_ylabel("Expert\n(neural full)")
        ax_nf.set_yticks(np.arange(env.num_experts))
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
    if has_l2d_sw:
        ax_l2d_sw = axes[idx]
        ax_l2d_sw.step(
            t_grid,
            choices_l2d_sw,
            where="post",
            color=get_model_color("l2d_sw"),
        )
        ax_l2d_sw.set_ylabel("Expert\n(L2D_SW)")
        ax_l2d_sw.set_yticks(np.arange(env.num_experts))
        idx += 1
    if has_linucb_partial:
        ax_lin_p = axes[idx]
        ax_lin_p.step(
            t_grid,
            choices_linucb_partial,
            where="post",
            color=get_model_color("linucb_partial"),
        )
        ax_lin_p.set_ylabel("Expert\n(LinUCB P)")
        ax_lin_p.set_yticks(np.arange(env.num_experts))
        idx += 1
    if has_linucb_full:
        ax_lin_f = axes[idx]
        ax_lin_f.step(
            t_grid,
            choices_linucb_full,
            where="post",
            color=get_model_color("linucb_full"),
        )
        ax_lin_f.set_ylabel("Expert\n(LinUCB F)")
        ax_lin_f.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_neuralucb_partial:
        ax_nucb_p = axes[idx]
        ax_nucb_p.step(
            t_grid,
            choices_neuralucb_partial,
            where="post",
            color=get_model_color("neuralucb"),
        )
        ax_nucb_p.set_ylabel("Expert\n(NeuralUCB P)")
        ax_nucb_p.set_yticks(np.arange(env.num_experts))
        idx += 1
    if has_neuralucb_full:
        ax_nucb_f = axes[idx]
        ax_nucb_f.step(
            t_grid,
            choices_neuralucb_full,
            where="post",
            color=get_model_color("neuralucb"),
        )
        ax_nucb_f.set_ylabel("Expert\n(NeuralUCB F)")
        ax_nucb_f.set_yticks(np.arange(env.num_experts))
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


def analysis_late_arrival(
    env: SyntheticTimeSeriesEnv,
    router_partial: SLDSIMMRouter,
    router_full: SLDSIMMRouter,
    l2d_baseline: Optional[L2D] = None,
    router_partial_corr=None,
    router_full_corr=None,
    router_partial_corr_em=None,
    router_full_corr_em=None,
    l2d_sw_baseline: Optional[L2D] = None,
    linucb_partial=None,
    linucb_full=None,
    neuralucb_partial=None,
    neuralucb_full=None,
    new_expert_idx: int | None = None,
    window: int = 500,
    adoption_threshold: float = 0.5,
) -> None:
    """
    Analyze how different policies react to the late arrival of a new
    expert that was previously unavailable.

    For a given expert index j_new and its first availability interval
    [t_start, t_end], this function:
      - runs all routers / baselines on the environment,
      - extracts costs and chosen experts over a window starting at
        t_start (clipped by `window` and the end of the first arrival
        interval),
      - computes, for each method:
            * selection frequency of j_new in the window,
            * mean cost in the window,
            * mean regret vs the oracle in the window,
            * "time to adoption" = first time in the window where the
              running selection frequency of j_new exceeds
              `adoption_threshold`.

    Results are printed as a compact textual summary.
    """
    T = env.T
    N = env.num_experts

    if new_expert_idx is None:
        raise ValueError(
            "analysis_late_arrival requires `new_expert_idx` "
            "(e.g., the arrival_expert_idx configured in the environment)."
        )
    j_new = int(new_expert_idx)
    if not (0 <= j_new < N):
        raise ValueError(
            f"new_expert_idx={j_new} is out of range for env.num_experts={N}."
        )

    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if not (0.0 < adoption_threshold <= 1.0):
        raise ValueError("adoption_threshold must be in (0, 1].")

    avail = getattr(env, "availability", None)
    if avail is None:
        raise ValueError(
            "Environment has no `availability` matrix; cannot analyze late arrival."
        )
    avail_arr = np.asarray(avail, dtype=int)
    if avail_arr.shape != (T, N):
        raise ValueError(
            f"env.availability has shape {avail_arr.shape}, expected ({T}, {N})."
        )

    # Determine the first arrival interval for the new expert:
    # contiguous block of availability beginning at the first time t
    # where availability switches from 0 to 1 (or is 1 for the first
    # time).
    avail_j = avail_arr[:, j_new].astype(bool)
    if not avail_j.any():
        print(
            f"[analysis_late_arrival] Expert {j_new} is never available; "
            "nothing to analyze."
        )
        return

    first_one_indices = np.where(avail_j)[0]
    t_start = int(first_one_indices[0])
    t_end = T - 1
    for t in range(t_start + 1, T):
        if not avail_j[t]:
            t_end = t - 1
            break

    # Warn if the expert was already available from t=0 (no true "late" arrival).
    if t_start == 0 and not np.any(~avail_j[:t_start]):
        print(
            f"[analysis_late_arrival] Expert {j_new} is available from t=0; "
            "t_start=0 will be treated as the analysis start, but this is "
            "not a strict late-arrival scenario."
        )

    # Analysis window: from t_start to min(t_start + window, t_end, T-1).
    t_window_end = min(t_start + int(window), t_end, T - 1)
    if t_window_end <= t_start:
        print(
            f"[analysis_late_arrival] Arrival window [{t_start}, {t_window_end}] "
            "is too short; skipping analysis."
        )
        return

    # Map time indices t=1,...,T-1 to array indices 0,...,T-2.
    start_idx = max(t_start, 1) - 1
    end_idx_excl = t_window_end
    if end_idx_excl <= start_idx:
        print(
            f"[analysis_late_arrival] Effective analysis range is empty after "
            f"aligning to t>=1 (start_idx={start_idx}, end_idx_excl={end_idx_excl})."
        )
        return

    idx_range = np.arange(start_idx, end_idx_excl, dtype=int)
    num_steps = int(idx_range.shape[0])

    # Reset router beliefs and L2D internal state so that this analysis
    # run starts from a fresh prior, independent of any previous calls.
    router_partial.reset_beliefs()
    router_full.reset_beliefs()
    if router_partial_corr is not None:
        router_partial_corr.reset_beliefs()
    if router_full_corr is not None:
        router_full_corr.reset_beliefs()
    if router_partial_corr_em is not None:
        router_partial_corr_em.reset_beliefs()
    if router_full_corr_em is not None:
        router_full_corr_em.reset_beliefs()
    if l2d_baseline is not None:
        l2d_baseline.reset_state()
    if l2d_sw_baseline is not None:
        l2d_sw_baseline.reset_state()

    # Common consultation costs (shared across methods)
    beta = router_partial.beta[: N]

    # Run routers and baselines to obtain costs and choices.
    costs_partial, choices_partial = run_router_on_env(router_partial, env)
    costs_full, choices_full = run_router_on_env(router_full, env)

    costs_partial_corr, choices_partial_corr = None, None
    if router_partial_corr is not None:
        costs_partial_corr, choices_partial_corr = run_router_on_env(
            router_partial_corr, env
        )

    costs_full_corr, choices_full_corr = None, None
    if router_full_corr is not None:
        costs_full_corr, choices_full_corr = run_router_on_env(
            router_full_corr, env
        )

    costs_partial_corr_em, choices_partial_corr_em = None, None
    if router_partial_corr_em is not None:
        costs_partial_corr_em, choices_partial_corr_em = run_router_on_env(
            router_partial_corr_em, env
        )

    costs_full_corr_em, choices_full_corr_em = None, None
    if router_full_corr_em is not None:
        costs_full_corr_em, choices_full_corr_em = run_router_on_env(
            router_full_corr_em, env
        )

    costs_l2d, choices_l2d = None, None
    if l2d_baseline is not None:
        costs_l2d, choices_l2d = run_l2d_on_env(l2d_baseline, env)

    costs_l2d_sw, choices_l2d_sw = None, None
    if l2d_sw_baseline is not None:
        costs_l2d_sw, choices_l2d_sw = run_l2d_on_env(l2d_sw_baseline, env)

    costs_linucb_partial, choices_linucb_partial = None, None
    if linucb_partial is not None:
        costs_linucb_partial, choices_linucb_partial = run_linucb_on_env(
            linucb_partial, env
        )

    costs_linucb_full, choices_linucb_full = None, None
    if linucb_full is not None:
        costs_linucb_full, choices_linucb_full = run_linucb_on_env(
            linucb_full, env
        )

    costs_neuralucb_partial, choices_neuralucb_partial = None, None
    if neuralucb_partial is not None:
        costs_neuralucb_partial, choices_neuralucb_partial = run_neuralucb_on_env(
            neuralucb_partial, env
        )

    costs_neuralucb_full, choices_neuralucb_full = None, None
    if neuralucb_full is not None:
        costs_neuralucb_full, choices_neuralucb_full = run_neuralucb_on_env(
            neuralucb_full, env
        )

    # Random and oracle baselines
    costs_random, choices_random = run_random_on_env(env, beta, seed=0)
    costs_oracle, choices_oracle = run_oracle_on_env(env, beta)

    # Collect methods with valid outputs.
    methods: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("partial", costs_partial, choices_partial),
        ("full", costs_full, choices_full),
        ("random", costs_random, choices_random),
        ("oracle", costs_oracle, choices_oracle),
    ]
    if costs_partial_corr is not None and choices_partial_corr is not None:
        methods.append(("partial_corr", costs_partial_corr, choices_partial_corr))
    if costs_full_corr is not None and choices_full_corr is not None:
        methods.append(("full_corr", costs_full_corr, choices_full_corr))
    if costs_partial_corr_em is not None and choices_partial_corr_em is not None:
        methods.append(("partial_corr_em", costs_partial_corr_em, choices_partial_corr_em))
    if costs_full_corr_em is not None and choices_full_corr_em is not None:
        methods.append(("full_corr_em", costs_full_corr_em, choices_full_corr_em))
    if costs_l2d is not None and choices_l2d is not None:
        methods.append(("l2d", costs_l2d, choices_l2d))
    if costs_l2d_sw is not None and choices_l2d_sw is not None:
        methods.append(("l2d_sw", costs_l2d_sw, choices_l2d_sw))
    if costs_linucb_partial is not None and choices_linucb_partial is not None:
        methods.append(("linucb_partial", costs_linucb_partial, choices_linucb_partial))
    if costs_linucb_full is not None and choices_linucb_full is not None:
        methods.append(("linucb_full", costs_linucb_full, choices_linucb_full))
    if costs_neuralucb_partial is not None and choices_neuralucb_partial is not None:
        methods.append(
            ("neuralucb_partial", costs_neuralucb_partial, choices_neuralucb_partial)
        )
    if costs_neuralucb_full is not None and choices_neuralucb_full is not None:
        methods.append(
            ("neuralucb_full", costs_neuralucb_full, choices_neuralucb_full)
        )

    # Sanity check on index range vs arrays.
    for _, costs_m, choices_m in methods:
        if costs_m.shape[0] < end_idx_excl or choices_m.shape[0] < end_idx_excl:
            raise ValueError(
                "Inconsistent time horizon between methods and env; "
                "costs/choices shorter than expected."
            )

    costs_oracle_window = costs_oracle[start_idx:end_idx_excl]

    print(
        f"\n=== Late-arrival analysis for expert {j_new} ===\n"
        f"First availability interval: t in [{t_start}, {t_end}] (inclusive)\n"
        f"Analysis window: t in [{max(t_start,1)}, {t_window_end}] "
        f"({num_steps} steps), adoption_threshold={adoption_threshold:.2f}\n"
    )
    header = (
        f"{'Method':<20} {'freq_new':>9} {'mean_cost':>11} "
        f"{'mean_regret':>12} {'t_adopt':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, costs_m, choices_m in methods:
        costs_win = costs_m[start_idx:end_idx_excl]
        choices_win = choices_m[start_idx:end_idx_excl]

        sel_new = (choices_win == j_new)
        freq_new = float(sel_new.mean()) if num_steps > 0 else 0.0
        regret_win = float((costs_win - costs_oracle_window).mean())
        mean_cost_win = float(costs_win.mean())

        t_adopt_str = "never"
        if num_steps > 0 and sel_new.any():
            running_freq = np.cumsum(sel_new.astype(float)) / np.arange(
                1, num_steps + 1, dtype=float
            )
            idx_adopt = np.where(running_freq >= adoption_threshold)[0]
            if idx_adopt.size > 0:
                t_adopt = int((start_idx + idx_adopt[0]) + 1)  # map index to time t
                t_adopt_str = str(t_adopt)

        print(
            f"{name:<20} {freq_new:9.3f} {mean_cost_win:11.4f} "
            f"{regret_win:12.4f} {t_adopt_str:>9}"
        )

    # ------------------------------------------------------------------
    # Visualization: reaction to late-arriving expert
    # ------------------------------------------------------------------
    if num_steps > 0:
        t_grid_win = np.arange(start_idx, end_idx_excl, dtype=int) + 1
        t_start_eff = max(t_start, 1)

        fig, (ax_freq, ax_reg) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6)
        )

        # Running selection frequency of the new expert in the window.
        for name, _, choices_m in methods:
            choices_win = choices_m[start_idx:end_idx_excl]
            sel_new = (choices_win == j_new).astype(float)
            running_freq = np.cumsum(sel_new) / np.arange(
                1, num_steps + 1, dtype=float
            )
            color = get_model_color(name)
            ax_freq.plot(
                t_grid_win,
                running_freq,
                label=name,
                color=color,
            )

        ax_freq.axhline(
            adoption_threshold,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"adoption threshold={adoption_threshold:.2f}",
        )
        ax_freq.axvline(
            t_start_eff,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        ax_freq.set_ylabel(f"Freq(exp {j_new})")
        ax_freq.set_title(
            f"Running selection frequency for late expert {j_new}"
        )
        ax_freq.legend(loc="upper left", fontsize=9)

        # Cumulative regret vs oracle in the same window.
        for name, costs_m, _ in methods:
            costs_win = costs_m[start_idx:end_idx_excl]
            delta = costs_win - costs_oracle_window
            cum_regret = np.cumsum(delta)
            color = get_model_color(name)
            ax_reg.plot(
                t_grid_win,
                cum_regret,
                label=name,
                color=color,
            )

        ax_reg.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax_reg.set_xlabel("Time $t$")
        ax_reg.set_ylabel("Cumulative regret vs oracle")
        ax_reg.set_title(
            f"Late-arrival window cumulative regret (expert {j_new})"
        )
        ax_reg.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        plt.show()
