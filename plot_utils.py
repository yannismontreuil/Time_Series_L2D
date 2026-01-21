import copy
import itertools
import json
import os
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from environment.etth1_env import ETTh1TimeSeriesEnv
from models.router_model import SLDSIMMRouter
from models.factorized_slds import FactorizedSLDS
from environment.synthetic_env import SyntheticTimeSeriesEnv
from models.l2d_baseline import L2D
from router_eval import (
    run_router_on_env,
    run_l2d_on_env,
    run_random_on_env,
    run_oracle_on_env,
    compute_predictions_from_choices,
    run_linucb_on_env,
    run_neuralucb_on_env, run_factored_router_on_env,
    run_router_on_env_em_split,
    get_transition_log_store,
    get_transition_log_config,
    set_transition_log_config,
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
        "partial_corr_rec": "tab:red",
        "full_corr_rec": "tab:olive",
        "partial_corr_em": "tab:blue",
        "full_corr_em": "tab:red",
        "partial_rec": "tab:purple",
        "full_rec": "tab:brown",
        "neural_partial": "tab:brown",
        "neural_full": "tab:purple",
        "l2d": "tab:red",
        "l2d_sw": "tab:pink",
        "linucb_partial": "tab:olive",
        "linucb_full": "tab:cyan",
        "neuralucb": "tab:brown",
        "factorized_partial": "tab:purple",
        "factorized_full": "tab:red",
        "factorized_linear_partial": "tab:gray",
        "factorized_linear_full": "tab:olive",
        "random": "tab:green",
    }
    return mapping.get(name, "tab:purple")


def _sanitize_transition_label(label: str) -> str:
    safe = []
    for ch in str(label):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "transition"


def plot_transition_matrices(
    log_store: dict[str, list[tuple[int, Optional[np.ndarray]]]],
    cfg: dict,
) -> None:
    if not log_store:
        return
    show_plots = bool(cfg.get("plot_show", True))
    save_plots = bool(cfg.get("plot_save", False))
    plot_na = bool(cfg.get("plot_na", False))
    if not show_plots and not save_plots:
        return
    out_dir = cfg.get("plot_dir", "out/transition_matrices")
    if save_plots:
        os.makedirs(out_dir, exist_ok=True)

    def _is_ours(label: str) -> bool:
        label_l = str(label).lower()
        return (
            label_l.startswith("slds-imm")
            or label_l.startswith("corr slds-imm")
            or label_l.startswith("factorized slds")
        )

    plot_only_ours = bool(cfg.get("plot_only_ours", False))

    plot_entropy = bool(cfg.get("plot_entropy", True))

    for label in sorted(log_store.keys()):
        if plot_only_ours and not _is_ours(label):
            continue
        series = log_store[label]
        if not series:
            continue
        series_sorted = sorted(series, key=lambda x: x[0])
        valid = [(t, mat) for t, mat in series_sorted if mat is not None]
        if not valid:
            if not plot_na:
                continue
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{label}\nPi=N/A",
                ha="center",
                va="center",
            )
            fig.tight_layout()
            if save_plots:
                fname = f"transition_{_sanitize_transition_label(label)}.png"
                fig.savefig(os.path.join(out_dir, fname))
            if show_plots:
                plt.show()
            plt.close(fig)
            continue

        times = [t for t, _ in valid]
        mats = np.stack([np.asarray(m, dtype=float) for _, m in valid], axis=0)
        if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
            print(f"[transition-plot] Skipping {label}: invalid Pi shape {mats.shape}")
            continue
        M = mats.shape[1]
        fig, axes = plt.subplots(
            M,
            1,
            sharex=True,
            figsize=(9, max(2.2 * M, 2.2)),
        )
        if M == 1:
            axes = [axes]
        for i in range(M):
            for j in range(M):
                axes[i].plot(
                    times,
                    mats[:, i, j],
                    label=f"{i}->{j}",
                )
            axes[i].set_ylabel(f"Row {i}")
            axes[i].set_ylim(-0.05, 1.05)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper right", fontsize=8, ncol=min(M, 3))
        axes[-1].set_xlabel("Time $t$")
        fig.suptitle(f"Transition matrix over time: {label}")
        fig.tight_layout()
        if save_plots:
            fname = f"transition_{_sanitize_transition_label(label)}.png"
            fig.savefig(os.path.join(out_dir, fname))
        if show_plots:
            plt.show()
        plt.close(fig)

        if plot_entropy:
            entropy = -np.sum(
                mats * np.log(np.maximum(mats, 1e-12)),
                axis=2,
            )
            fig_ent, ax_ent = plt.subplots(figsize=(9, 3))
            for i in range(M):
                ax_ent.plot(times, entropy[:, i], label=f"Row {i}")
            ax_ent.set_xlabel("Time $t$")
            ax_ent.set_ylabel("Row entropy")
            ax_ent.set_ylim(-0.05, np.log(float(M)) + 0.05)
            ax_ent.grid(True, alpha=0.3)
            ax_ent.legend(loc="upper right", fontsize=8, ncol=min(M, 3))
            fig_ent.suptitle(f"Transition row entropy over time: {label}")
            fig_ent.tight_layout()
            if save_plots:
                fname = f"transition_entropy_{_sanitize_transition_label(label)}.png"
                fig_ent.savefig(os.path.join(out_dir, fname))
            if show_plots:
                plt.show()
            plt.close(fig_ent)


def add_unavailability_regions(
    ax: Axes, env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv
) -> None:
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
                if start_t is None:
                    # Should not happen, but guard for type safety.
                    in_block = False
                    continue
                end_t = t
                color = get_expert_color(j)
                # here f means float
                start_f = float(start_t)
                end_f = float(end_t)
                ax.axvspan(
                    start_f,
                    end_f,
                    facecolor=color,
                    alpha=0.08,
                    zorder=0,
                )
                mid = 0.5 * (start_f + end_f)
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
            start_f = float(start_t)
            end_f = float(end_t)
            ax.axvspan(
                start_f,
                end_f,
                facecolor=color,
                alpha=0.08,
                zorder=0,
            )
            mid = 0.5 * (start_f + end_f)
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
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    num_points: Optional[int] = None,
) -> None:
    """
    Plot the true time series y_t together with each expert's prediction,
    and the underlying regime z_t and expert availability.
    """
    T = env.T if num_points is None else min(num_points, env.T)
    t_grid = np.arange(T)

    plot_target = str(getattr(env, "plot_target", "y")).lower()
    if plot_target == "x":
        y = env.x[:T]
        true_label = "Context $x_t$ (lagged)"
    else:
        y = env.y[:T]
        true_label = "True $y_t$"
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
        label=true_label,
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
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    router_partial: SLDSIMMRouter,
    router_full: SLDSIMMRouter,
    router_factorial_partial: Optional[FactorizedSLDS],
    router_factorial_full: Optional[FactorizedSLDS],
    factorized_label: str = "L2D SLDS w/ $g_t$",
    router_factorial_partial_linear: Optional[FactorizedSLDS] = None,
    router_factorial_full_linear: Optional[FactorizedSLDS] = None,
    factorized_linear_label: str = "L2D SLDS",
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
    seed: int = 0,
    analysis_cfg: Optional[dict] = None,
    planning_snapshot_t: Optional[int] = None,
    planning_snapshots: Optional[dict] = None,
) -> None:
    """
    Evaluate how the partial- and full-feedback routers behave on the
    environment, compare their average cost to constant-expert baselines,
    and plot the induced prediction time series and selections.
    """
    def _get_em_tk_anchor(*routers) -> Optional[int]:
        candidates = []
        for router in routers:
            if router is None:
                continue
            em_tk_val = getattr(router, "em_tk", None)
            if em_tk_val is None:
                continue
            try:
                em_tk_int = int(em_tk_val)
            except (TypeError, ValueError):
                continue
            if em_tk_int > 0:
                candidates.append(em_tk_int)
        return max(candidates) if candidates else None

    em_tk_anchor = _get_em_tk_anchor(
        router_partial,
        router_full,
        router_factorial_partial,
        router_factorial_full,
        router_factorial_partial_linear,
        router_factorial_full_linear,
        router_partial_corr_em,
        router_full_corr_em,
    )
    baselines_train_from_start = False
    if analysis_cfg is not None:
        baselines_train_from_start = bool(
            analysis_cfg.get("baselines_train_from_start", False)
        )
    if baselines_train_from_start or em_tk_anchor is None:
        baseline_start_t = 1
    else:
        baseline_start_t = int(em_tk_anchor) + 1
    if analysis_cfg is not None and bool(analysis_cfg.get("debug_env_fingerprint", False)):
        if hasattr(env, "fingerprint"):
            data_seed = getattr(env, "data_seed", None)
            print(f"[env] data_seed={data_seed} fingerprint={env.fingerprint()}")
    if baselines_train_from_start:
        em_anchor_str = "none" if em_tk_anchor is None else str(int(em_tk_anchor))
        print(
            f"[baselines] training from t=1 (full history); "
            f"evaluation masked after em_tk={em_anchor_str}."
        )
    elif em_tk_anchor is not None:
        print(
            f"[baselines] training starts at t={baseline_start_t} "
            f"(em_tk_anchor={int(em_tk_anchor)})."
        )
    transition_cfg = get_transition_log_config()
    if transition_cfg is not None and transition_cfg.get("online_only", False):
        if transition_cfg.get("start_t") is None:
            transition_cfg = dict(transition_cfg)
            transition_cfg["start_t"] = baseline_start_t
            set_transition_log_config(transition_cfg)

    # Run both base routers to obtain costs and choices
    def _run_base_router(router: SLDSIMMRouter, snapshot_key: Optional[str] = None):
        em_tk = getattr(router, "em_tk", None)
        if em_tk is not None:
            return run_router_on_env_em_split(
                router,
                env,
                int(em_tk),
                snapshot_at_t=planning_snapshot_t,
                snapshot_dict=planning_snapshots,
                snapshot_key=snapshot_key,
            )
        return run_router_on_env(
            router,
            env,
            snapshot_at_t=planning_snapshot_t,
            snapshot_dict=planning_snapshots,
            snapshot_key=snapshot_key,
        )

    costs_partial, choices_partial = _run_base_router(router_partial, "router_partial")
    costs_full, choices_full = _run_base_router(router_full, "router_full")

    costs_factorial_partial = None
    choices_factorial_partial = None
    if router_factorial_partial is not None:
        em_tk_fact = getattr(router_factorial_partial, "em_tk", None)
        if em_tk_fact is not None:
            costs_factorial_partial, choices_factorial_partial = run_router_on_env_em_split(
                router_factorial_partial,
                env,
                int(em_tk_fact),
                snapshot_at_t=planning_snapshot_t,
                snapshot_dict=planning_snapshots,
                snapshot_key="fact_router_partial",
            )
        else:
            costs_factorial_partial, choices_factorial_partial = run_factored_router_on_env(
                router_factorial_partial,
                env,
                snapshot_at_t=planning_snapshot_t,
                snapshot_dict=planning_snapshots,
                snapshot_key="fact_router_partial",
            )

    costs_factorial_full = None
    choices_factorial_full = None
    if router_factorial_full is not None:
        em_tk_fact = getattr(router_factorial_full, "em_tk", None)
        if em_tk_fact is not None:
            costs_factorial_full, choices_factorial_full = run_router_on_env_em_split(
                router_factorial_full,
                env,
                int(em_tk_fact),
                snapshot_at_t=planning_snapshot_t,
                snapshot_dict=planning_snapshots,
                snapshot_key="fact_router_full",
            )
        else:
            costs_factorial_full, choices_factorial_full = run_factored_router_on_env(
                router_factorial_full,
                env,
                snapshot_at_t=planning_snapshot_t,
                snapshot_dict=planning_snapshots,
                snapshot_key="fact_router_full",
            )

    costs_factorial_linear_partial = None
    choices_factorial_linear_partial = None
    if router_factorial_partial_linear is not None:
        em_tk_fact = getattr(router_factorial_partial_linear, "em_tk", None)
        if em_tk_fact is not None:
            costs_factorial_linear_partial, choices_factorial_linear_partial = (
                run_router_on_env_em_split(
                    router_factorial_partial_linear,
                    env,
                    int(em_tk_fact),
                    snapshot_at_t=planning_snapshot_t,
                    snapshot_dict=planning_snapshots,
                    snapshot_key="fact_router_partial_linear",
                )
            )
        else:
            costs_factorial_linear_partial, choices_factorial_linear_partial = (
                run_factored_router_on_env(
                    router_factorial_partial_linear,
                    env,
                    snapshot_at_t=planning_snapshot_t,
                    snapshot_dict=planning_snapshots,
                    snapshot_key="fact_router_partial_linear",
                )
            )

    costs_factorial_linear_full = None
    choices_factorial_linear_full = None
    if router_factorial_full_linear is not None:
        em_tk_fact = getattr(router_factorial_full_linear, "em_tk", None)
        if em_tk_fact is not None:
            costs_factorial_linear_full, choices_factorial_linear_full = (
                run_router_on_env_em_split(
                    router_factorial_full_linear,
                    env,
                    int(em_tk_fact),
                    snapshot_at_t=planning_snapshot_t,
                    snapshot_dict=planning_snapshots,
                    snapshot_key="fact_router_full_linear",
                )
            )
        else:
            costs_factorial_linear_full, choices_factorial_linear_full = (
                run_factored_router_on_env(
                    router_factorial_full_linear,
                    env,
                    snapshot_at_t=planning_snapshot_t,
                    snapshot_dict=planning_snapshots,
                    snapshot_key="fact_router_full_linear",
                )
            )

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
        em_tk = getattr(router_partial_corr_em, "em_tk", None)
        if em_tk is not None:
            costs_partial_corr_em, choices_partial_corr_em = run_router_on_env_em_split(
                router_partial_corr_em, env, int(em_tk)
            )
        else:
            costs_partial_corr_em, choices_partial_corr_em = run_router_on_env(
                router_partial_corr_em, env
            )
    else:
        costs_partial_corr_em, choices_partial_corr_em = None, None

    if router_full_corr_em is not None:
        em_tk = getattr(router_full_corr_em, "em_tk", None)
        if em_tk is not None:
            costs_full_corr_em, choices_full_corr_em = run_router_on_env_em_split(
                router_full_corr_em, env, int(em_tk)
            )
        else:
            costs_full_corr_em, choices_full_corr_em = run_router_on_env(
                router_full_corr_em, env
            )
    else:
        costs_full_corr_em, choices_full_corr_em = None, None

    # Run learning-to-defer baselines if provided
    if l2d_baseline is not None:
        costs_l2d, choices_l2d = run_l2d_on_env(
            l2d_baseline, env, t_start=baseline_start_t
        )
    else:
        costs_l2d, choices_l2d = None, None

    if l2d_sw_baseline is not None:
        costs_l2d_sw, choices_l2d_sw = run_l2d_on_env(
            l2d_sw_baseline, env, t_start=baseline_start_t
        )
    else:
        costs_l2d_sw, choices_l2d_sw = None, None

    # Run LinUCB baselines if provided
    if linucb_partial is not None:
        costs_linucb_partial, choices_linucb_partial = run_linucb_on_env(
            linucb_partial, env, t_start=baseline_start_t
        )
    else:
        costs_linucb_partial, choices_linucb_partial = None, None

    if linucb_full is not None:
        costs_linucb_full, choices_linucb_full = run_linucb_on_env(
            linucb_full, env, t_start=baseline_start_t
        )
    else:
        costs_linucb_full, choices_linucb_full = None, None

    # Run NeuralUCB baselines if provided
    if neuralucb_partial is not None:
        costs_neuralucb_partial, choices_neuralucb_partial = run_neuralucb_on_env(
            neuralucb_partial, env, t_start=baseline_start_t
        )
    else:
        costs_neuralucb_partial, choices_neuralucb_partial = None, None

    if neuralucb_full is not None:
        costs_neuralucb_full, choices_neuralucb_full = run_neuralucb_on_env(
            neuralucb_full, env, t_start=baseline_start_t
        )
    else:
        costs_neuralucb_full, choices_neuralucb_full = None, None

    # Common consultation costs (assumed shared across methods)
    beta = router_partial.beta[: env.num_experts]

    # Random and oracle baselines
    costs_random, choices_random = run_random_on_env(env, beta, seed=int(seed))
    costs_oracle, choices_oracle = run_oracle_on_env(env, beta)

    # Prediction series induced by router and L2D choices
    preds_partial = compute_predictions_from_choices(env, choices_partial)
    preds_full = compute_predictions_from_choices(env, choices_full)
    preds_factorized_partial = (
        compute_predictions_from_choices(env, choices_factorial_partial)
        if choices_factorial_partial is not None
        else None
    )
    preds_factorized_full = (
        compute_predictions_from_choices(env, choices_factorial_full)
        if choices_factorial_full is not None
        else None
    )
    preds_factorized_linear_partial = (
        compute_predictions_from_choices(env, choices_factorial_linear_partial)
        if choices_factorial_linear_partial is not None
        else None
    )
    preds_factorized_linear_full = (
        compute_predictions_from_choices(env, choices_factorial_linear_full)
        if choices_factorial_linear_full is not None
        else None
    )
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

    def _mask_em_costs(
        costs: Optional[np.ndarray], em_tk: Optional[int]
    ) -> Optional[np.ndarray]:
        if costs is None or em_tk is None:
            return costs
        em_tk = int(em_tk)
        if em_tk <= 0:
            return costs
        costs = np.asarray(costs, dtype=float).copy()
        cut = min(em_tk, costs.shape[0])
        if cut > 0:
            costs[:cut] = np.nan
        return costs

    def _mask_em_choices(
        choices: Optional[np.ndarray], em_tk: Optional[int]
    ) -> Optional[np.ndarray]:
        if choices is None or em_tk is None:
            return choices
        em_tk = int(em_tk)
        if em_tk <= 0:
            return choices
        arr = np.asarray(choices, dtype=float).copy()
        cut = min(em_tk, arr.shape[0])
        if cut > 0:
            arr[:cut] = np.nan
        return arr

    costs_partial = _mask_em_costs(
        costs_partial, getattr(router_partial, "em_tk", None)
    )
    costs_full = _mask_em_costs(costs_full, getattr(router_full, "em_tk", None))
    if router_partial_corr_em is not None:
        costs_partial_corr_em = _mask_em_costs(
            costs_partial_corr_em, getattr(router_partial_corr_em, "em_tk", None)
        )
    if router_full_corr_em is not None:
        costs_full_corr_em = _mask_em_costs(
            costs_full_corr_em, getattr(router_full_corr_em, "em_tk", None)
        )
    if router_factorial_partial is not None:
        costs_factorial_partial = _mask_em_costs(
            costs_factorial_partial, getattr(router_factorial_partial, "em_tk", None)
        )
    if router_factorial_full is not None:
        costs_factorial_full = _mask_em_costs(
            costs_factorial_full, getattr(router_factorial_full, "em_tk", None)
        )
    if router_factorial_partial_linear is not None:
        costs_factorial_linear_partial = _mask_em_costs(
            costs_factorial_linear_partial,
            getattr(router_factorial_partial_linear, "em_tk", None),
        )
    if router_factorial_full_linear is not None:
        costs_factorial_linear_full = _mask_em_costs(
            costs_factorial_linear_full,
            getattr(router_factorial_full_linear, "em_tk", None),
        )
    if em_tk_anchor is not None:
        costs_l2d = _mask_em_costs(costs_l2d, em_tk_anchor)
        costs_l2d_sw = _mask_em_costs(costs_l2d_sw, em_tk_anchor)
        costs_linucb_partial = _mask_em_costs(costs_linucb_partial, em_tk_anchor)
        costs_linucb_full = _mask_em_costs(costs_linucb_full, em_tk_anchor)
        costs_neuralucb_partial = _mask_em_costs(costs_neuralucb_partial, em_tk_anchor)
        costs_neuralucb_full = _mask_em_costs(costs_neuralucb_full, em_tk_anchor)
        costs_random = _mask_em_costs(costs_random, em_tk_anchor)
        costs_oracle = _mask_em_costs(costs_oracle, em_tk_anchor)

    choices_partial_plot = _mask_em_choices(
        choices_partial, getattr(router_partial, "em_tk", None)
    )
    choices_full_plot = _mask_em_choices(
        choices_full, getattr(router_full, "em_tk", None)
    )
    choices_partial_corr_em_plot = _mask_em_choices(
        choices_partial_corr_em,
        getattr(router_partial_corr_em, "em_tk", None)
        if router_partial_corr_em is not None
        else None,
    )
    choices_full_corr_em_plot = _mask_em_choices(
        choices_full_corr_em,
        getattr(router_full_corr_em, "em_tk", None)
        if router_full_corr_em is not None
        else None,
    )
    choices_factorial_partial_plot = _mask_em_choices(
        choices_factorial_partial,
        getattr(router_factorial_partial, "em_tk", None)
        if router_factorial_partial is not None
        else None,
    )
    choices_factorial_full_plot = _mask_em_choices(
        choices_factorial_full,
        getattr(router_factorial_full, "em_tk", None)
        if router_factorial_full is not None
        else None,
    )
    choices_factorial_linear_partial_plot = _mask_em_choices(
        choices_factorial_linear_partial,
        getattr(router_factorial_partial_linear, "em_tk", None)
        if router_factorial_partial_linear is not None
        else None,
    )
    choices_factorial_linear_full_plot = _mask_em_choices(
        choices_factorial_linear_full,
        getattr(router_factorial_full_linear, "em_tk", None)
        if router_factorial_full_linear is not None
        else None,
    )
    choices_partial_corr_plot = choices_partial_corr
    choices_full_corr_plot = choices_full_corr
    choices_partial_neural_plot = choices_partial_neural
    choices_full_neural_plot = choices_full_neural
    choices_l2d_plot = choices_l2d
    choices_l2d_sw_plot = choices_l2d_sw
    choices_linucb_partial_plot = choices_linucb_partial
    choices_linucb_full_plot = choices_linucb_full
    choices_neuralucb_partial_plot = choices_neuralucb_partial
    choices_neuralucb_full_plot = choices_neuralucb_full
    choices_random_plot = choices_random
    choices_oracle_plot = choices_oracle
    if em_tk_anchor is not None:
        choices_partial_corr_plot = _mask_em_choices(choices_partial_corr, em_tk_anchor)
        choices_full_corr_plot = _mask_em_choices(choices_full_corr, em_tk_anchor)
        choices_partial_neural_plot = _mask_em_choices(
            choices_partial_neural, em_tk_anchor
        )
        choices_full_neural_plot = _mask_em_choices(choices_full_neural, em_tk_anchor)
        choices_l2d_plot = _mask_em_choices(choices_l2d, em_tk_anchor)
        choices_l2d_sw_plot = _mask_em_choices(choices_l2d_sw, em_tk_anchor)
        choices_linucb_partial_plot = _mask_em_choices(
            choices_linucb_partial, em_tk_anchor
        )
        choices_linucb_full_plot = _mask_em_choices(
            choices_linucb_full, em_tk_anchor
        )
        choices_neuralucb_partial_plot = _mask_em_choices(
            choices_neuralucb_partial, em_tk_anchor
        )
        choices_neuralucb_full_plot = _mask_em_choices(
            choices_neuralucb_full, em_tk_anchor
        )
        choices_random_plot = _mask_em_choices(choices_random, em_tk_anchor)
        choices_oracle_plot = _mask_em_choices(choices_oracle, em_tk_anchor)

    T = env.T
    t_grid = np.arange(1, T)
    plot_target = str(getattr(env, "plot_target", "y")).lower()
    if plot_target == "x":
        y_true = env.x[1:T]
        true_label = "Context $x_t$ (lagged)"
    else:
        y_true = env.y[1:T]
        true_label = "True $y_t$"

    # Constant-expert baselines (always pick the same expert)
    t_start = 1
    if em_tk_anchor is not None and em_tk_anchor > 0:
        t_start = min(int(em_tk_anchor) + 1, T)
    if t_start >= T:
        avg_cost_experts = np.full(env.num_experts, np.nan, dtype=float)
    else:
        cum_costs = np.zeros(env.num_experts, dtype=float)
        for t in range(t_start, T):
            loss_all = env.losses(t)
            cum_costs += loss_all + beta
        avg_cost_experts = cum_costs / float(T - t_start)

    avg_cost_partial = float(np.nanmean(costs_partial))
    avg_cost_full = float(np.nanmean(costs_full))
    avg_cost_partial_corr = (
        float(np.nanmean(costs_partial_corr)) if costs_partial_corr is not None else None
    )
    avg_cost_full_corr = (
        float(np.nanmean(costs_full_corr)) if costs_full_corr is not None else None
    )
    avg_cost_factorized_partial = (
        float(np.nanmean(costs_factorial_partial))
        if costs_factorial_partial is not None
        else None
    )
    avg_cost_factorized_full = (
        float(np.nanmean(costs_factorial_full)) if costs_factorial_full is not None else None
    )
    avg_cost_factorized_linear_partial = (
        float(np.nanmean(costs_factorial_linear_partial))
        if costs_factorial_linear_partial is not None
        else None
    )
    avg_cost_factorized_linear_full = (
        float(np.nanmean(costs_factorial_linear_full))
        if costs_factorial_linear_full is not None
        else None
    )
    avg_cost_partial_corr_em = (
        float(np.nanmean(costs_partial_corr_em)) if costs_partial_corr_em is not None else None
    )
    avg_cost_full_corr_em = (
        float(np.nanmean(costs_full_corr_em)) if costs_full_corr_em is not None else None
    )
    avg_cost_neural_partial = (
        float(np.nanmean(costs_partial_neural)) if costs_partial_neural is not None else None
    )
    avg_cost_neural_full = (
        float(np.nanmean(costs_full_neural)) if costs_full_neural is not None else None
    )
    avg_cost_l2d = float(np.nanmean(costs_l2d)) if costs_l2d is not None else None
    avg_cost_l2d_sw = (
        float(np.nanmean(costs_l2d_sw)) if costs_l2d_sw is not None else None
    )
    avg_cost_linucb_partial = (
        float(np.nanmean(costs_linucb_partial)) if costs_linucb_partial is not None else None
    )
    avg_cost_linucb_full = (
        float(np.nanmean(costs_linucb_full)) if costs_linucb_full is not None else None
    )
    avg_cost_neuralucb_partial = (
        float(np.nanmean(costs_neuralucb_partial))
        if costs_neuralucb_partial is not None
        else None
    )
    avg_cost_neuralucb_full = (
        float(np.nanmean(costs_neuralucb_full))
        if costs_neuralucb_full is not None
        else None
    )
    avg_cost_random = float(np.nanmean(costs_random))
    avg_cost_oracle = float(np.nanmean(costs_oracle))

    cost_len = max(T - 1, 1)
    last_frac = 0.2
    last_start_idx = int(np.floor((1.0 - last_frac) * cost_len))
    last_start_idx = max(0, min(last_start_idx, cost_len - 1))
    last_t_start = last_start_idx + 1
    if em_tk_anchor is not None and em_tk_anchor > 0:
        last_t_start = max(last_t_start, int(em_tk_anchor) + 1)

    def _mean_last(costs: Optional[np.ndarray]) -> Optional[float]:
        if costs is None:
            return None
        arr = np.asarray(costs, dtype=float)
        if arr.size == 0:
            return None
        start = min(last_start_idx, arr.size - 1)
        return float(np.nanmean(arr[start:]))

    last_cost_partial = _mean_last(costs_partial)
    last_cost_full = _mean_last(costs_full)
    last_cost_partial_corr = _mean_last(costs_partial_corr)
    last_cost_full_corr = _mean_last(costs_full_corr)
    last_cost_factorized_partial = _mean_last(costs_factorial_partial)
    last_cost_factorized_full = _mean_last(costs_factorial_full)
    last_cost_factorized_linear_partial = _mean_last(costs_factorial_linear_partial)
    last_cost_factorized_linear_full = _mean_last(costs_factorial_linear_full)
    last_cost_partial_corr_em = _mean_last(costs_partial_corr_em)
    last_cost_full_corr_em = _mean_last(costs_full_corr_em)
    last_cost_neural_partial = _mean_last(costs_partial_neural)
    last_cost_neural_full = _mean_last(costs_full_neural)
    last_cost_l2d = _mean_last(costs_l2d)
    last_cost_l2d_sw = _mean_last(costs_l2d_sw)
    last_cost_linucb_partial = _mean_last(costs_linucb_partial)
    last_cost_linucb_full = _mean_last(costs_linucb_full)
    last_cost_neuralucb_partial = _mean_last(costs_neuralucb_partial)
    last_cost_neuralucb_full = _mean_last(costs_neuralucb_full)
    last_cost_random = _mean_last(costs_random)
    last_cost_oracle = _mean_last(costs_oracle)

    if last_t_start >= T:
        avg_cost_experts_last = np.full(env.num_experts, np.nan, dtype=float)
    else:
        cum_costs_last = np.zeros(env.num_experts, dtype=float)
        for t in range(last_t_start, T):
            loss_all = env.losses(t)
            cum_costs_last += loss_all + beta
        avg_cost_experts_last = cum_costs_last / float(T - last_t_start)

    print("=== Average costs ===")
    print(f"L2D SLDS w/t $g_t$ (partial fb): {avg_cost_partial:.4f}")
    print(f"L2D SLDS w/t $g_t$ (full fb):    {avg_cost_full:.4f}")
    if avg_cost_partial_corr is not None:
        print(
            f"Router Corr (partial feedback): {avg_cost_partial_corr:.4f}"
        )
    if avg_cost_full_corr is not None:
        print(f"Router Corr (full feedback):    {avg_cost_full_corr:.4f}")
    if avg_cost_factorized_partial is not None:
        print(f"{factorized_label} (partial fb):   {avg_cost_factorized_partial:.4f}")
    if avg_cost_factorized_full is not None:
        print(f"{factorized_label} (full fb):      {avg_cost_factorized_full:.4f}")
    if avg_cost_factorized_linear_partial is not None:
        print(
            f"{factorized_linear_label} (partial fb): {avg_cost_factorized_linear_partial:.4f}"
        )
    if avg_cost_factorized_linear_full is not None:
        print(
            f"{factorized_linear_label} (full fb):    {avg_cost_factorized_linear_full:.4f}"
        )
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
        print(f"L2D (full feedback):           {avg_cost_l2d:.4f}")
    if avg_cost_l2d_sw is not None:
        print(f"L2D_SW (full feedback):        {avg_cost_l2d_sw:.4f}")
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

    print(
        f"\n=== Mean costs (last 20% of horizon, t >= {last_t_start}) ==="
    )
    if last_cost_partial is not None:
        print(f"L2D SLDS w/t $g_t$ (partial fb): {last_cost_partial:.4f}")
    if last_cost_full is not None:
        print(f"L2D SLDS w/t $g_t$ (full fb):    {last_cost_full:.4f}")
    if last_cost_partial_corr is not None:
        print(
            f"Router Corr (partial feedback): {last_cost_partial_corr:.4f}"
        )
    if last_cost_full_corr is not None:
        print(f"Router Corr (full feedback):    {last_cost_full_corr:.4f}")
    if last_cost_factorized_partial is not None:
        print(f"{factorized_label} (partial fb):   {last_cost_factorized_partial:.4f}")
    if last_cost_factorized_full is not None:
        print(f"{factorized_label} (full fb):      {last_cost_factorized_full:.4f}")
    if last_cost_factorized_linear_partial is not None:
        print(
            f"{factorized_linear_label} (partial fb): {last_cost_factorized_linear_partial:.4f}"
        )
    if last_cost_factorized_linear_full is not None:
        print(
            f"{factorized_linear_label} (full fb):    {last_cost_factorized_linear_full:.4f}"
        )
    if last_cost_partial_corr_em is not None:
        print(
            f"Router Corr EM (partial fb):   {last_cost_partial_corr_em:.4f}"
        )
    if last_cost_full_corr_em is not None:
        print(
            f"Router Corr EM (full fb):      {last_cost_full_corr_em:.4f}"
        )
    if last_cost_neural_partial is not None:
        print(
            f"Neural router (partial fb):     {last_cost_neural_partial:.4f}"
        )
    if last_cost_neural_full is not None:
        print(
            f"Neural router (full fb):        {last_cost_neural_full:.4f}"
        )
    if last_cost_l2d is not None:
        print(f"L2D (full feedback):           {last_cost_l2d:.4f}")
    if last_cost_l2d_sw is not None:
        print(f"L2D_SW (full feedback):        {last_cost_l2d_sw:.4f}")
    if last_cost_linucb_partial is not None:
        print(f"LinUCB (partial feedback):     {last_cost_linucb_partial:.4f}")
    if last_cost_linucb_full is not None:
        print(f"LinUCB (full feedback):        {last_cost_linucb_full:.4f}")
    if last_cost_neuralucb_partial is not None:
        print(f"NeuralUCB (partial feedback):  {last_cost_neuralucb_partial:.4f}")
    if last_cost_neuralucb_full is not None:
        print(f"NeuralUCB (full feedback):     {last_cost_neuralucb_full:.4f}")
    if last_cost_random is not None:
        print(f"Random baseline:               {last_cost_random:.4f}")
    if last_cost_oracle is not None:
        print(f"Oracle baseline:               {last_cost_oracle:.4f}")
    for j in range(env.num_experts):
        print(
            f"Always using expert {j}:       {avg_cost_experts_last[j]:.4f}"
        )

    # Selection distribution (how often each expert is chosen)
    entries = [
        ("partial", choices_partial_plot),
        ("full", choices_full_plot),
    ]
    if choices_partial_corr_plot is not None:
        entries.append(("partial_corr", choices_partial_corr_plot))
    if choices_full_corr_plot is not None:
        entries.append(("full_corr", choices_full_corr_plot))
    if choices_partial_corr_em_plot is not None:
        entries.append(("partial_corr_em", choices_partial_corr_em_plot))
    if choices_full_corr_em_plot is not None:
        entries.append(("full_corr_em", choices_full_corr_em_plot))
    if choices_factorial_partial_plot is not None:
        entries.append(("factorized_partial", choices_factorial_partial_plot))
    if choices_factorial_full_plot is not None:
        entries.append(("factorized_full", choices_factorial_full_plot))
    if choices_factorial_linear_partial_plot is not None:
        entries.append(
            ("factorized_linear_partial", choices_factorial_linear_partial_plot)
        )
    if choices_factorial_linear_full_plot is not None:
        entries.append(("factorized_linear_full", choices_factorial_linear_full_plot))
    entries.extend(
        [
            ("random", choices_random_plot),
            ("oracle", choices_oracle_plot),
        ]
    )
    if choices_partial_neural_plot is not None:
        entries.append(("neural_partial", choices_partial_neural_plot))
    if choices_full_neural_plot is not None:
        entries.append(("neural_full", choices_full_neural_plot))
    if choices_l2d_plot is not None:
        entries.append(("l2d", choices_l2d_plot))
    if choices_l2d_sw_plot is not None:
        entries.append(("l2d_sw", choices_l2d_sw_plot))
    if choices_linucb_partial_plot is not None:
        entries.append(("linucb_partial", choices_linucb_partial_plot))
    if choices_linucb_full_plot is not None:
        entries.append(("linucb_full", choices_linucb_full_plot))
    if choices_neuralucb_partial_plot is not None:
        entries.append(("neuralucb_partial", choices_neuralucb_partial_plot))
    if choices_neuralucb_full_plot is not None:
        entries.append(("neuralucb_full", choices_neuralucb_full_plot))

    # ---------------------------------------------
    # Uncomment the following lines to print selection distributions
    # ---------------------------------------------
    # for name, choices in entries:
    #     values, counts = np.unique(choices, return_counts=True)
    #     freqs = counts / choices.shape[0]
    #     print(f"Selection distribution ({name}):")
    #     for v, c, f in zip(values, counts, freqs):
    #         print(f"  expert {int(v)}: count={int(c)}, freq={f:.3f}")

    # Plot true series vs router-based prediction series (top)
    # and cumulative costs over time for each baseline (bottom).
    fig, (ax_pred, ax_cost) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    # Visualization-only shift; positive values advance predictions (shift left).
    vis_shift = int(getattr(env, "plot_shift", 1))

    def _shift_preds_for_plot(preds: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if preds is None:
            return None
        preds = np.asarray(preds, dtype=float)
        if vis_shift <= 0:
            if vis_shift == 0:
                return preds
            if preds.size <= abs(vis_shift):
                return np.zeros(0, dtype=float)
            return preds[:vis_shift]
        if preds.size <= vis_shift:
            return np.zeros(0, dtype=float)
        return preds[vis_shift:]

    if vis_shift > 0:
        t_grid_plot = t_grid[:-vis_shift]
        y_true_plot = y_true[:-vis_shift]
    elif vis_shift < 0:
        t_grid_plot = t_grid[-vis_shift:]
        y_true_plot = y_true[-vis_shift:]
    else:
        t_grid_plot = t_grid
        y_true_plot = y_true

    preds_partial_plot = _shift_preds_for_plot(preds_partial)
    preds_full_plot = _shift_preds_for_plot(preds_full)
    preds_partial_corr_plot = _shift_preds_for_plot(preds_partial_corr)
    preds_full_corr_plot = _shift_preds_for_plot(preds_full_corr)
    preds_partial_corr_em_plot = _shift_preds_for_plot(preds_partial_corr_em)
    preds_full_corr_em_plot = _shift_preds_for_plot(preds_full_corr_em)
    preds_partial_neural_plot = _shift_preds_for_plot(preds_partial_neural)
    preds_full_neural_plot = _shift_preds_for_plot(preds_full_neural)
    preds_l2d_plot = _shift_preds_for_plot(preds_l2d)
    preds_l2d_sw_plot = _shift_preds_for_plot(preds_l2d_sw)
    preds_factorized_partial_plot = _shift_preds_for_plot(preds_factorized_partial)
    preds_factorized_full_plot = _shift_preds_for_plot(preds_factorized_full)
    preds_factorized_linear_partial_plot = _shift_preds_for_plot(
        preds_factorized_linear_partial
    )
    preds_factorized_linear_full_plot = _shift_preds_for_plot(
        preds_factorized_linear_full
    )
    preds_linucb_partial_plot = _shift_preds_for_plot(preds_linucb_partial)
    preds_linucb_full_plot = _shift_preds_for_plot(preds_linucb_full)
    preds_neuralucb_partial_plot = _shift_preds_for_plot(preds_neuralucb_partial)
    preds_neuralucb_full_plot = _shift_preds_for_plot(preds_neuralucb_full)
    preds_random_plot = _shift_preds_for_plot(preds_random)
    preds_oracle_plot = _shift_preds_for_plot(preds_oracle)

    # Top subplot: predictions
    ax_pred.plot(
        t_grid_plot,
        y_true_plot,
        label=true_label,
        color=get_model_color("true"),
        linewidth=2,
        linestyle="-",
    )
    ax_pred.plot(
        t_grid_plot,
        preds_partial_plot,
        label="L2D SLDS w/t $g_t$ (partial)",
        color=get_model_color("partial"),
        linestyle="-",
        alpha=0.8,
    )
    ax_pred.plot(
        t_grid_plot,
        preds_full_plot,
        label="L2D SLDS w/t $g_t$ (full)",
        color=get_model_color("full"),
        linestyle="-",
        alpha=0.8,
    )
    if preds_partial_corr_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_partial_corr_plot,
            label="Router Corr (partial)",
            color=get_model_color("partial_corr"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_full_corr_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_full_corr_plot,
            label="Router Corr (full)",
            color=get_model_color("full_corr"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_partial_corr_em_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_partial_corr_em_plot,
            label="Router Corr EM (partial)",
            color=get_model_color("partial_corr_em"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_full_corr_em_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_full_corr_em_plot,
            label="Router Corr EM (full)",
            color=get_model_color("full_corr_em"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_partial_neural_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_partial_neural_plot,
            label="Neural router (partial)",
            color=get_model_color("neural_partial"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_full_neural_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_full_neural_plot,
            label="Neural router (full)",
            color=get_model_color("neural_full"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_l2d_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_l2d_plot,
            label="L2D (full feedback)",
            color=get_model_color("l2d"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_l2d_sw_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_l2d_sw_plot,
            label="L2D_SW (full feedback)",
            color=get_model_color("l2d_sw"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_factorized_partial_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_factorized_partial_plot,
            label=f"{factorized_label} (partial)",
            color=get_model_color("factorized_partial"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_factorized_full_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_factorized_full_plot,
            label=f"{factorized_label} (full)",
            color=get_model_color("factorized_full"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_factorized_linear_partial_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_factorized_linear_partial_plot,
            label=f"{factorized_linear_label} (partial)",
            color=get_model_color("factorized_linear_partial"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_factorized_linear_full_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_factorized_linear_full_plot,
            label=f"{factorized_linear_label} (full)",
            color=get_model_color("factorized_linear_full"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_linucb_partial_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_linucb_partial_plot,
            label="LinUCB (partial)",
            color=get_model_color("linucb_partial"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_linucb_full_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_linucb_full_plot,
            label="LinUCB (full)",
            color=get_model_color("linucb_full"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_neuralucb_partial_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_neuralucb_partial_plot,
            label="NeuralUCB (partial)",
            color=get_model_color("neuralucb"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_neuralucb_full_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_neuralucb_full_plot,
            label="NeuralUCB (full)",
            color=get_model_color("neuralucb"),
            linestyle="--",
            alpha=0.8,
        )
    ax_pred.plot(
        t_grid_plot,
        preds_oracle_plot,
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
    def _running_avg(costs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if costs is None:
            return None
        costs = np.asarray(costs, dtype=float)
        mask = ~np.isnan(costs)
        if mask.size == 0:
            return costs
        cum_sum = np.cumsum(np.where(mask, costs, 0.0))
        cum_count = np.cumsum(mask.astype(float))
        avg = cum_sum / np.maximum(cum_count, 1.0)
        avg[cum_count == 0] = np.nan
        return avg

    avg_partial_t = _running_avg(costs_partial)
    avg_full_t = _running_avg(costs_full)
    avg_partial_corr_t = (
        _running_avg(costs_partial_corr) if costs_partial_corr is not None else None
    )
    avg_full_corr_t = (
        _running_avg(costs_full_corr) if costs_full_corr is not None else None
    )
    avg_partial_corr_em_t = (
        _running_avg(costs_partial_corr_em)
        if costs_partial_corr_em is not None
        else None
    )
    avg_full_corr_em_t = (
        _running_avg(costs_full_corr_em) if costs_full_corr_em is not None else None
    )
    avg_neural_partial_t = (
        _running_avg(costs_partial_neural)
        if costs_partial_neural is not None
        else None
    )
    avg_neural_full_t = (
        _running_avg(costs_full_neural) if costs_full_neural is not None else None
    )
    avg_random_t = _running_avg(costs_random)
    avg_oracle_t = _running_avg(costs_oracle)
    avg_l2d_t = (
        _running_avg(costs_l2d) if costs_l2d is not None else None
    )
    avg_l2d_sw_t = (
        _running_avg(costs_l2d_sw) if costs_l2d_sw is not None else None
    )
    avg_linucb_partial_t = (
        _running_avg(costs_linucb_partial)
        if costs_linucb_partial is not None
        else None
    )
    avg_linucb_full_t = (
        _running_avg(costs_linucb_full) if costs_linucb_full is not None else None
    )
    avg_neuralucb_partial_t = (
        _running_avg(costs_neuralucb_partial)
        if costs_neuralucb_partial is not None
        else None
    )
    avg_neuralucb_full_t = (
        _running_avg(costs_neuralucb_full)
        if costs_neuralucb_full is not None
        else None
    )
    avg_factorized_partial_t = (
        _running_avg(costs_factorial_partial)
        if costs_factorial_partial is not None
        else None
    )
    avg_factorized_full_t = (
        _running_avg(costs_factorial_full) if costs_factorial_full is not None else None
    )
    avg_factorized_linear_partial_t = (
        _running_avg(costs_factorial_linear_partial)
        if costs_factorial_linear_partial is not None
        else None
    )
    avg_factorized_linear_full_t = (
        _running_avg(costs_factorial_linear_full)
        if costs_factorial_linear_full is not None
        else None
    )

    ax_cost.plot(
        t_grid,
        avg_partial_t,
        label="L2D SLDS w/t $g_t$ (partial, avg cost)",
        color=get_model_color("partial"),
        linestyle="-",
    )
    ax_cost.plot(
        t_grid,
        avg_full_t,
        label="L2D SLDS w/t $g_t$ (full, avg cost)",
        color=get_model_color("full"),
        linestyle="-",
    )
    if avg_partial_corr_t is not None:
        ax_cost.plot(
            t_grid,
            avg_partial_corr_t,
            label="Corr partial (avg cost)",
            color=get_model_color("partial_corr"),
            linestyle="-",
        )
    if avg_full_corr_t is not None:
        ax_cost.plot(
            t_grid,
            avg_full_corr_t,
            label="Corr full (avg cost)",
            color=get_model_color("full_corr"),
            linestyle="-",
        )
    if avg_partial_corr_em_t is not None:
        ax_cost.plot(
            t_grid,
            avg_partial_corr_em_t,
            label="Corr EM partial (avg cost)",
            color=get_model_color("partial_corr_em"),
            linestyle="-",
        )
    if avg_full_corr_em_t is not None:
        ax_cost.plot(
            t_grid,
            avg_full_corr_em_t,
            label="Corr EM full (avg cost)",
            color=get_model_color("full_corr_em"),
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
            label="L2D (full feedback, avg cost)",
            color=get_model_color("l2d"),
            linestyle="-",
        )
    if avg_l2d_sw_t is not None:
        ax_cost.plot(
            t_grid,
            avg_l2d_sw_t,
            label="L2D_SW (full feedback, avg cost)",
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
    if avg_factorized_partial_t is not None:
        ax_cost.plot(
            t_grid,
            avg_factorized_partial_t,
            label=f"{factorized_label} (partial, avg cost)",
            color=get_model_color("factorized_partial"),
            linestyle="-",
        )
    if avg_factorized_full_t is not None:
        ax_cost.plot(
            t_grid,
            avg_factorized_full_t,
            label=f"{factorized_label} (full, avg cost)",
            color=get_model_color("factorized_full"),
            linestyle="-",
        )
    if avg_factorized_linear_partial_t is not None:
        ax_cost.plot(
            t_grid,
            avg_factorized_linear_partial_t,
            label=f"{factorized_linear_label} (partial, avg cost)",
            color=get_model_color("factorized_linear_partial"),
            linestyle="-",
        )
    if avg_factorized_linear_full_t is not None:
        ax_cost.plot(
            t_grid,
            avg_factorized_linear_full_t,
            label=f"{factorized_linear_label} (full, avg cost)",
            color=get_model_color("factorized_linear_full"),
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

    transition_cfg = get_transition_log_config()
    if transition_cfg is not None and transition_cfg.get("plot", False):
        log_store = get_transition_log_store(reset=True)
        plot_transition_matrices(log_store, transition_cfg)

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
            return float(np.nanmean(costs[idx0:]))

        mean_oracle = float(np.nanmean(costs_oracle[idx0:]))
        mean_factorized_partial = mean_cost(costs_factorial_partial)
        mean_factorized_full = mean_cost(costs_factorial_full)
        mean_factorized_linear_partial = mean_cost(costs_factorial_linear_partial)
        mean_factorized_linear_full = mean_cost(costs_factorial_linear_full)
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
        print(f"L2D SLDS w/t $g_t$ (partial): {mean_partial:.4f}")
        print(f"L2D SLDS w/t $g_t$ (full):    {mean_full:.4f}")
        if mean_factorized_partial is not None:
            print(f"{factorized_label} (partial):        {mean_factorized_partial:.4f}")
        if mean_factorized_full is not None:
            print(f"{factorized_label} (full):           {mean_factorized_full:.4f}")
        if mean_factorized_linear_partial is not None:
            print(
                f"{factorized_linear_label} (partial): {mean_factorized_linear_partial:.4f}"
            )
        if mean_factorized_linear_full is not None:
            print(
                f"{factorized_linear_label} (full):    {mean_factorized_linear_full:.4f}"
            )
        if mean_partial_corr is not None:
            print(f"Router Corr (partial):         {mean_partial_corr:.4f}")
        if mean_full_corr is not None:
            print(f"Router Corr (full):            {mean_full_corr:.4f}")
        if mean_l2d is not None:
            print(f"L2D (full feedback):           {mean_l2d:.4f}")
        if mean_l2d_sw is not None:
            print(f"L2D_SW (full feedback):        {mean_l2d_sw:.4f}")
        if mean_linucb_partial is not None:
            print(f"LinUCB (partial):              {mean_linucb_partial:.4f}")
        if mean_linucb_full is not None:
            print(f"LinUCB (full):                 {mean_linucb_full:.4f}")

        t_reg = t_grid[idx0:]
        fig_reg, ax_reg = plt.subplots(1, 1, figsize=(10, 4))
        ax_reg.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

        if reg_partial is not None:
            ax_reg.plot(
                t_reg,
                reg_partial,
                label="L2D SLDS w/t $g_t$ (partial)",
                color=get_model_color("partial"),
            )
        if reg_full is not None:
            ax_reg.plot(
                t_reg,
                reg_full,
                label="L2D SLDS w/t $g_t$ (full)",
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
                label="L2D (full feedback)",
                color=get_model_color("l2d"),
            )
        if reg_l2d_sw is not None:
            ax_reg.plot(
                t_reg,
                reg_l2d_sw,
                label="L2D_SW (full feedback)",
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
    has_factorized_partial = choices_factorial_partial is not None
    has_factorized_full = choices_factorial_full is not None
    has_factorized_linear_partial = choices_factorial_linear_partial is not None
    has_factorized_linear_full = choices_factorial_linear_full is not None

    # Rows: base routers, optional correlated routers, optional L2D / L2D_SW /
    # LinUCB partial/full / NeuralUCB partial/full, oracle baseline,
    # and optional availability.
    n_rows = 3 + (1 if has_l2d else 0) + (1 if has_l2d_sw else 0)
    n_rows += (1 if has_linucb_partial else 0) + (1 if has_linucb_full else 0)
    n_rows += (1 if has_neuralucb_partial else 0) + (1 if has_neuralucb_full else 0)
    n_rows += (1 if has_avail else 0)
    n_rows += (1 if has_partial_corr else 0) + (1 if has_full_corr else 0)
    n_rows += (1 if has_partial_corr_em else 0) + (1 if has_full_corr_em else 0)
    n_rows += (1 if has_neural_partial else 0) + (1 if has_neural_full else 0)
    n_rows += (1 if has_factorized_partial else 0) + (1 if has_factorized_full else 0)
    n_rows += (1 if has_factorized_linear_partial else 0) + (
        1 if has_factorized_linear_full else 0
    )
    fig2, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10, 2 * n_rows))

    idx = 0
    ax_p = axes[idx]
    ax_p.step(
        t_grid,
        choices_partial_plot,
        where="post",
        color=get_model_color("partial"),
    )
    ax_p.set_ylabel("Expert\n(L2D w/t $g_t$ P)")
    ax_p.set_yticks(np.arange(env.num_experts))
    ax_p.set_title("Selections and availability over time")
    idx += 1

    ax_f = axes[idx]
    ax_f.step(
        t_grid,
        choices_full_plot,
        where="post",
        color=get_model_color("full"),
    )
    ax_f.set_ylabel("Expert\n(L2D w/t $g_t$ F)")
    ax_f.set_yticks(np.arange(env.num_experts))
    idx += 1

    if has_partial_corr:
        ax_pc = axes[idx]
        ax_pc.step(
            t_grid,
            choices_partial_corr_plot,
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
            choices_full_corr_plot,
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
            choices_partial_corr_em_plot,
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
            choices_full_corr_em_plot,
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
            choices_partial_neural_plot,
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
            choices_full_neural_plot,
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
            choices_l2d_plot,
            where="post",
            color=get_model_color("l2d"),
        )
        ax_l2d.set_ylabel("Expert\n(L2D full)")
        ax_l2d.set_yticks(np.arange(env.num_experts))
        idx += 1
    if has_l2d_sw:
        ax_l2d_sw = axes[idx]
        ax_l2d_sw.step(
            t_grid,
            choices_l2d_sw_plot,
            where="post",
            color=get_model_color("l2d_sw"),
        )
        ax_l2d_sw.set_ylabel("Expert\n(L2D_SW full)")
        ax_l2d_sw.set_yticks(np.arange(env.num_experts))
        idx += 1
    if has_linucb_partial:
        ax_lin_p = axes[idx]
        ax_lin_p.step(
            t_grid,
            choices_linucb_partial_plot,
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
            choices_linucb_full_plot,
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
            choices_neuralucb_partial_plot,
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
            choices_neuralucb_full_plot,
            where="post",
            color=get_model_color("neuralucb"),
        )
        ax_nucb_f.set_ylabel("Expert\n(NeuralUCB F)")
        ax_nucb_f.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_factorized_partial:
        ax_fact_p = axes[idx]
        ax_fact_p.step(
            t_grid,
            choices_factorial_partial_plot,
            where="post",
            color=get_model_color("factorized_partial"),
        )
        ax_fact_p.set_ylabel("Expert\n(Fact P)")
        ax_fact_p.set_yticks(np.arange(env.num_experts))
        idx += 1
    if has_factorized_full:
        ax_fact_f = axes[idx]
        ax_fact_f.step(
            t_grid,
            choices_factorial_full_plot,
            where="post",
            color=get_model_color("factorized_full"),
        )
        ax_fact_f.set_ylabel("Expert\n(Fact F)")
        ax_fact_f.set_yticks(np.arange(env.num_experts))
        idx += 1

    if has_factorized_linear_partial:
        ax_fact_lp = axes[idx]
        ax_fact_lp.step(
            t_grid,
            choices_factorial_linear_partial_plot,
            where="post",
            color=get_model_color("factorized_linear_partial"),
        )
        ax_fact_lp.set_ylabel("Expert\n(L2D P)")
        ax_fact_lp.set_yticks(np.arange(env.num_experts))
        idx += 1
    if has_factorized_linear_full:
        ax_fact_lf = axes[idx]
        ax_fact_lf.step(
            t_grid,
            choices_factorial_linear_full_plot,
            where="post",
            color=get_model_color("factorized_linear_full"),
        )
        ax_fact_lf.set_ylabel("Expert\n(L2D F)")
        ax_fact_lf.set_yticks(np.arange(env.num_experts))
        idx += 1

    ax_oracle = axes[idx]
    ax_oracle.step(
        t_grid,
        choices_oracle_plot,
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

    tri_cfg = analysis_cfg.get("tri_cycle_corr", {}) if analysis_cfg else {}
    if (
        tri_cfg.get("expert_structure_baselines", False)
        and getattr(env, "setting", None) == "tri_cycle_corr"
    ):
        out_dir = str(tri_cfg.get("out_dir", "out/tri_cycle_corr"))
        os.makedirs(out_dir, exist_ok=True)
        choices_map = {
            "L2D SLDS w/t $g_t$ (partial)": (choices_partial, costs_partial),
            "L2D SLDS w/t $g_t$ (full)": (choices_full, costs_full),
            "Corr partial": (choices_partial_corr, costs_partial_corr),
            "Corr full": (choices_full_corr, costs_full_corr),
            "Corr EM partial": (choices_partial_corr_em, costs_partial_corr_em),
            "Corr EM full": (choices_full_corr_em, costs_full_corr_em),
            "L2D SLDS (partial)": (choices_factorial_partial, costs_factorial_partial),
            "L2D SLDS (full)": (choices_factorial_full, costs_factorial_full),
            "L2D": (choices_l2d, costs_l2d),
            "L2D_SW": (choices_l2d_sw, costs_l2d_sw),
            "LinUCB partial": (choices_linucb_partial, costs_linucb_partial),
            "LinUCB full": (choices_linucb_full, costs_linucb_full),
            "NeuralUCB partial": (choices_neuralucb_partial, costs_neuralucb_partial),
            "NeuralUCB full": (choices_neuralucb_full, costs_neuralucb_full),
            "Oracle": (choices_oracle, costs_oracle),
        }
        rows = [
            [("Oracle", (choices_oracle, costs_oracle))],
            [
                ("L2D SLDS (partial)", (choices_factorial_partial, costs_factorial_partial)),
                ("L2D SLDS w/t $g_t$ (partial)", (choices_partial, costs_partial)),
                ("Corr partial", (choices_partial_corr, costs_partial_corr)),
                ("Corr EM partial", (choices_partial_corr_em, costs_partial_corr_em)),
                ("LinUCB partial", (choices_linucb_partial, costs_linucb_partial)),
                ("NeuralUCB partial", (choices_neuralucb_partial, costs_neuralucb_partial)),
            ],
            [
                ("L2D SLDS (full)", (choices_factorial_full, costs_factorial_full)),
                ("L2D SLDS w/t $g_t$ (full)", (choices_full, costs_full)),
                ("Corr full", (choices_full_corr, costs_full_corr)),
                ("Corr EM full", (choices_full_corr_em, costs_full_corr_em)),
                ("L2D", (choices_l2d, costs_l2d)),
                ("L2D_SW", (choices_l2d_sw, costs_l2d_sw)),
                ("LinUCB full", (choices_linucb_full, costs_linucb_full)),
                ("NeuralUCB full", (choices_neuralucb_full, costs_neuralucb_full)),
            ],
        ]
        plot_selection_freq_by_regime(
            env=env,
            choices_map=choices_map,
            out_dir=out_dir,
            show_plots=bool(tri_cfg.get("show_plots", False)),
            save_plots=bool(tri_cfg.get("save_plots", True)),
            save_png=bool(tri_cfg.get("save_png", True)),
            save_pdf=bool(tri_cfg.get("save_pdf", True)),
            rows=rows,
        )


def analysis_late_arrival(
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    router_partial: SLDSIMMRouter,
    router_full: SLDSIMMRouter,
    l2d_baseline: Optional[L2D] = None,
    router_partial_rec=None,
    router_full_rec=None,
    router_partial_corr=None,
    router_full_corr=None,
    router_partial_corr_rec=None,
    router_full_corr_rec=None,
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
    seed: int = 0,
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


def plot_pruning_dynamics(
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    router_full: FactorizedSLDS,
    router_no_g: FactorizedSLDS,
    expert_idx: int,
    rolling_window: int = 100,
    out_dir: str = "out/pruning",
    show_plots: bool = False,
    save_plots: bool = True,
    save_png: bool = True,
    save_pdf: bool = True,
    label_full: str = "L2D SLDS w/ $g_t$",
    label_no_g: str = "L2D SLDS w/t $g_t$",
) -> None:
    if not isinstance(router_full, FactorizedSLDS) or not isinstance(router_no_g, FactorizedSLDS):
        raise ValueError("plot_pruning_dynamics expects FactorizedSLDS routers.")
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive.")

    env_avail = getattr(env, "availability", None)
    if env_avail is None:
        raise ValueError("Environment has no availability matrix for pruning analysis.")
    avail_arr = np.asarray(env_avail, dtype=int)
    T, N = avail_arr.shape
    j = int(expert_idx)
    if not (0 <= j < N):
        raise ValueError(f"expert_idx={j} out of range for env.num_experts={N}.")

    times = np.arange(1, T, dtype=int)
    avail_j = avail_arr[1:T, j].astype(bool)

    def _run_router(router: FactorizedSLDS):
        router = copy.deepcopy(router)
        router.reset_beliefs()
        in_registry = np.zeros(times.shape[0], dtype=bool)
        selected = np.full(times.shape[0], -1, dtype=int)
        pred_var = np.full(times.shape[0], np.nan, dtype=float)
        prune_times = []
        prev_in_registry = None

        for idx, t in enumerate(times):
            x_t = env.get_context(int(t))
            available = np.asarray(env.get_available_experts(int(t)), dtype=int)
            r_t, cache = router.select_expert(x_t, available)
            selected[idx] = int(r_t)

            curr_in_registry = int(j) in set(getattr(router, "registry", []))
            in_registry[idx] = curr_in_registry
            if prev_in_registry is not None and prev_in_registry and not curr_in_registry:
                prune_times.append(int(t))
            prev_in_registry = curr_in_registry

            w_pred = cache.get("w_pred", None)
            stats = cache.get("stats", {})
            if w_pred is not None and j in stats:
                var_modes = np.asarray(stats[j]["var"], dtype=float)
                pred_var[idx] = float(np.dot(np.asarray(w_pred, dtype=float), var_modes))
            elif curr_in_registry and w_pred is not None:
                phi = cache["phi"]
                mu_g_pred = cache["mu_g_pred"]
                Sigma_g_pred = cache["Sigma_g_pred"]
                mu_u_pred = cache["mu_u_pred"]
                Sigma_u_pred = cache["Sigma_u_pred"]
                _, _, stats_tmp = router._compute_predictive_stats(
                    phi,
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                    [j],
                )
                if j in stats_tmp:
                    var_modes = np.asarray(stats_tmp[j]["var"], dtype=float)
                    pred_var[idx] = float(np.dot(np.asarray(w_pred, dtype=float), var_modes))

            if getattr(router, "observation_mode", "loss") == "residual":
                y_t = float(env.y[int(t)])
                preds = env.all_expert_predictions(x_t)
                residuals = preds - y_t
                residual_r = float(residuals[int(r_t)])
                loss_obs = residual_r
                losses_full = None
                if getattr(router, "feedback_mode", "partial") == "full":
                    losses_full = np.full(residuals.shape, np.nan, dtype=float)
                    losses_full[available] = residuals[available]
            else:
                loss_all = env.losses(int(t))
                loss_obs = float(loss_all[int(r_t)])
                losses_full = None
                if getattr(router, "feedback_mode", "partial") == "full":
                    losses_full = np.full(loss_all.shape, np.nan, dtype=float)
                    losses_full[available] = loss_all[available]

            router.update_beliefs(
                r_t=int(r_t),
                loss_obs=float(loss_obs),
                losses_full=losses_full,
                available_experts=available,
                cache=cache,
            )

        return in_registry, selected, pred_var, prune_times

    in_reg_full, sel_full, var_full, prune_full = _run_router(router_full)
    in_reg_no_g, sel_no_g, var_no_g, prune_no_g = _run_router(router_no_g)

    def _rolling_mean(mask: np.ndarray, window: int) -> np.ndarray:
        if mask.size == 0:
            return mask.astype(float)
        win = min(int(window), int(mask.size))
        kernel = np.ones(win, dtype=float) / float(win)
        return np.convolve(mask.astype(float), kernel, mode="same")

    sel_rate_full = _rolling_mean(sel_full == j, rolling_window)
    sel_rate_no_g = _rolling_mean(sel_no_g == j, rolling_window)

    unavail_spans = []
    start_idx = None
    for idx, is_avail in enumerate(avail_j):
        if not is_avail and start_idx is None:
            start_idx = idx
        if is_avail and start_idx is not None:
            unavail_spans.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        unavail_spans.append((start_idx, len(avail_j) - 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for ax in axes:
        for seg_start, seg_end in unavail_spans:
            t_start = int(times[seg_start])
            t_end = int(times[seg_end]) + 1
            ax.axvspan(t_start, t_end, color="gray", alpha=0.15, lw=0)

    axes[0].step(times, avail_j.astype(float), where="post", color="gray", alpha=0.6, label="available")
    axes[0].step(times, in_reg_full.astype(float), where="post", label=f"registry ({label_full})")
    axes[0].step(times, in_reg_no_g.astype(float), where="post", label=f"registry ({label_no_g})", linestyle="--")
    for t_prune in prune_full:
        axes[0].axvline(t_prune, color="tab:blue", alpha=0.3, linewidth=1.0)
    for t_prune in prune_no_g:
        axes[0].axvline(t_prune, color="tab:orange", alpha=0.3, linewidth=1.0)
    axes[0].set_ylabel("Availability / Registry")
    axes[0].set_yticks([0, 1])
    axes[0].set_title(f"Expert {j}: availability and registry membership")
    axes[0].legend(loc="upper right", fontsize=10)

    axes[1].plot(times, var_full, label=f"pred var ({label_full})")
    axes[1].plot(times, var_no_g, label=f"pred var ({label_no_g})", linestyle="--")
    axes[1].set_ylabel("Pred. variance")
    axes[1].set_title(f"Expert {j}: predictive uncertainty")
    axes[1].legend(loc="upper right", fontsize=10)

    axes[2].plot(times, sel_rate_full, label=f"select freq ({label_full})")
    axes[2].plot(times, sel_rate_no_g, label=f"select freq ({label_no_g})", linestyle="--")
    axes[2].set_ylabel(f"Selection freq (win={rolling_window})")
    axes[2].set_xlabel("Time $t$")
    axes[2].set_title(f"Expert {j}: rolling selection frequency")
    axes[2].legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    if save_plots:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"pruning_dynamics_expert_{j}")
        if save_pdf:
            fig.savefig(f"{base}.pdf", bbox_inches="tight")
        if save_png:
            fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def _mask_feedback_vector_local(
    values: np.ndarray,
    available: np.ndarray,
    selected: int,
    full_feedback: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    masked = np.full(values.shape, np.nan, dtype=float)
    if full_feedback:
        masked[available] = values[available]
    else:
        masked[int(selected)] = values[int(selected)]
    return masked


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("corr_matrix expects 2D arrays.")
    d1 = int(a.shape[1])
    d2 = int(b.shape[1])
    out = np.zeros((d1, d2), dtype=float)
    for i in range(d1):
        for j in range(d2):
            out[i, j] = _safe_corr(a[:, i], b[:, j])
    return out


def _align_latent_dims(
    true_g: np.ndarray, est_g: np.ndarray
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    true_g = np.asarray(true_g, dtype=float)
    est_g = np.asarray(est_g, dtype=float)
    if true_g.ndim != 2 or est_g.ndim != 2:
        raise ValueError("align_latent_dims expects 2D arrays.")
    d = min(true_g.shape[1], est_g.shape[1])
    if d == 0:
        return est_g, [], np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    true_g = true_g[:, :d]
    est_g = est_g[:, :d]
    corr = _corr_matrix(true_g, est_g)
    best_perm = list(range(d))
    best_score = -np.inf
    if d <= 6:
        for perm in itertools.permutations(range(d)):
            score = sum(abs(corr[i, perm[i]]) for i in range(d))
            if score > best_score:
                best_score = score
                best_perm = list(perm)
    else:
        remaining = set(range(d))
        best_perm = []
        for i in range(d):
            if not remaining:
                best_perm.append(i)
                continue
            best_j = max(remaining, key=lambda j: abs(corr[i, j]))
            remaining.remove(best_j)
            best_perm.append(best_j)
    aligned = est_g[:, best_perm]
    signs = np.array([np.sign(corr[i, best_perm[i]]) for i in range(d)], dtype=float)
    signs[signs == 0.0] = 1.0
    aligned = aligned * signs
    return aligned, best_perm, signs, corr


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_to_corr expects a square matrix.")
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / denom
    corr = np.clip(corr, -1.0, 1.0)
    for i in range(corr.shape[0]):
        if var[i] > 0.0:
            corr[i, i] = 1.0
    return corr


def _compute_residual_matrix(
    env: SyntheticTimeSeriesEnv, t_start: int, t_end: int
) -> np.ndarray:
    t_start = max(int(t_start), 1)
    t_end = min(int(t_end), env.T - 1)
    if t_start > t_end:
        return np.zeros((0, env.num_experts), dtype=float)
    residuals = np.zeros((t_end - t_start + 1, env.num_experts), dtype=float)
    for idx, t in enumerate(range(t_start, t_end + 1)):
        x_t = env.get_context(t)
        preds = env.all_expert_predictions(x_t)
        residuals[idx] = preds - float(env.y[t])
    return residuals


def _corr_by_regime(
    residuals: np.ndarray, z: np.ndarray, num_regimes: int
) -> list[np.ndarray]:
    residuals = np.asarray(residuals, dtype=float)
    z = np.asarray(z, dtype=int)
    n_experts = int(residuals.shape[1]) if residuals.ndim == 2 else 0
    corr_list: list[np.ndarray] = []
    for m in range(num_regimes):
        mask = z == m
        if residuals.ndim != 2 or mask.sum() < 2:
            corr = np.full((n_experts, n_experts), np.nan, dtype=float)
        else:
            corr = np.corrcoef(residuals[mask].T)
        corr_list.append(corr)
    return corr_list


def _avg_corr_by_regime(
    corr_series: np.ndarray, z: np.ndarray, num_regimes: int
) -> list[np.ndarray]:
    corr_series = np.asarray(corr_series, dtype=float)
    z = np.asarray(z, dtype=int)
    if corr_series.ndim != 3:
        return [np.zeros((0, 0), dtype=float) for _ in range(num_regimes)]
    n_experts = int(corr_series.shape[1])
    avg_list: list[np.ndarray] = []
    for m in range(num_regimes):
        mask = z == m
        if mask.sum() < 1:
            avg = np.full((n_experts, n_experts), np.nan, dtype=float)
        else:
            avg = np.nanmean(corr_series[mask], axis=0)
        avg_list.append(avg)
    return avg_list


def _predictive_corr_from_cache(
    cache: dict,
    available: Sequence[int],
    n_experts: int,
) -> np.ndarray:
    stats = cache.get("stats", {}) or {}
    w_pred = np.asarray(cache.get("w_pred", []), dtype=float).reshape(-1)
    Sigma_g_pred = np.asarray(cache.get("Sigma_g_pred", []), dtype=float)
    avail = [int(k) for k in available if int(k) in stats]
    full = np.full((n_experts, n_experts), np.nan, dtype=float)
    if not avail or w_pred.size == 0:
        return full
    M = int(w_pred.shape[0])
    means_m = np.zeros((M, len(avail)), dtype=float)
    s_m = np.zeros((M, len(avail)), dtype=float)
    H = None
    if Sigma_g_pred.ndim == 3 and Sigma_g_pred.shape[0] == M:
        d_g = int(Sigma_g_pred.shape[1])
    else:
        d_g = 0
    if d_g > 0:
        H = np.vstack([stats[k]["h"] for k in avail]).astype(float)
    for m in range(M):
        for idx, k in enumerate(avail):
            means_m[m, idx] = float(stats[k]["mean"][m])
            s_m[m, idx] = float(stats[k]["s"][m])
    mu = w_pred @ means_m
    cov = np.zeros((len(avail), len(avail)), dtype=float)
    for m in range(M):
        if d_g > 0:
            cov_shared = H @ Sigma_g_pred[m] @ H.T
        else:
            cov_shared = np.zeros((len(avail), len(avail)), dtype=float)
        cov_m = cov_shared + np.diag(s_m[m])
        diff = means_m[m] - mu
        cov += w_pred[m] * (cov_m + np.outer(diff, diff))
    cov = 0.5 * (cov + cov.T)
    corr_small = _cov_to_corr(cov)
    for i, ki in enumerate(avail):
        for j, kj in enumerate(avail):
            full[ki, kj] = corr_small[i, j]
    return full


def _collect_factorized_diagnostics(
    router: FactorizedSLDS,
    env: SyntheticTimeSeriesEnv,
    t_start: int = 1,
    t_end: Optional[int] = None,
) -> dict:
    if t_end is None:
        t_end = env.T - 1
    t_start = max(int(t_start), 1)
    t_end = min(int(t_end), env.T - 1)
    router_local = copy.deepcopy(router)
    router_local.reset_beliefs()

    times = []
    choices = []
    costs = []
    w_pred_list = []
    w_post_list = []
    g_post_list = []
    corr_list = []

    for t in range(t_start, t_end + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)
        r_t, cache = router_local.select_expert(x_t, available)
        pred_corr = _predictive_corr_from_cache(cache, available, env.num_experts)

        preds = env.all_expert_predictions(x_t)
        residuals = preds - float(env.y[t])
        residual_r = float(residuals[int(r_t)])
        loss_r = residual_r * residual_r
        cost_t = loss_r + router_local._get_beta(int(r_t))
        if router_local.feedback_mode == "full":
            residuals_full = _mask_feedback_vector_local(
                residuals, np.asarray(available, dtype=int), int(r_t), True
            )
        else:
            residuals_full = None

        router_local.update_beliefs(
            r_t=int(r_t),
            loss_obs=residual_r,
            losses_full=residuals_full,
            available_experts=available,
            cache=cache,
        )

        times.append(int(t))
        choices.append(int(r_t))
        costs.append(float(cost_t))
        w_pred_list.append(np.asarray(cache.get("w_pred"), dtype=float))
        w_post_list.append(np.asarray(router_local.w, dtype=float))
        corr_list.append(pred_corr)
        if router_local.d_g > 0:
            g_post_list.append(router_local.w @ router_local.mu_g)

    return {
        "times": np.asarray(times, dtype=int),
        "choices": np.asarray(choices, dtype=int),
        "costs": np.asarray(costs, dtype=float),
        "w_pred": np.asarray(w_pred_list, dtype=float),
        "w_post": np.asarray(w_post_list, dtype=float),
        "g_post": np.asarray(g_post_list, dtype=float) if g_post_list else None,
        "pred_corr": np.asarray(corr_list, dtype=float),
    }


def _compute_probe_stats(
    router: FactorizedSLDS,
    phi: np.ndarray,
    w: np.ndarray,
    mu_g: np.ndarray,
    Sigma_g: np.ndarray,
    mu_u: dict,
    Sigma_u: dict,
    expert: int,
) -> tuple[float, float, float]:
    costs, _, stats = router._compute_predictive_stats(
        phi,
        w,
        mu_g,
        Sigma_g,
        mu_u,
        Sigma_u,
        [int(expert)],
    )
    if int(expert) not in stats:
        return float("nan"), float("nan"), float("nan")
    mean_modes = np.asarray(stats[int(expert)]["mean"], dtype=float)
    var_modes = np.asarray(stats[int(expert)]["var"], dtype=float)
    w = np.asarray(w, dtype=float)
    mean = float(w @ mean_modes)
    second = float(w @ (var_modes + mean_modes * mean_modes))
    var = max(second - mean * mean, 0.0)
    loss = float(costs[int(expert)] - router._get_beta(int(expert)))
    return mean, var, loss


def _collect_transfer_probe(
    router: FactorizedSLDS,
    env: SyntheticTimeSeriesEnv,
    target_expert: int,
    t_start: int,
    t_end: int,
) -> dict:
    router_local = copy.deepcopy(router)
    router_local.reset_beliefs()

    times = []
    selected = []
    avail_target = []
    pre_mean = []
    pre_var = []
    pre_loss = []
    post_mean = []
    post_var = []
    post_loss = []
    true_loss = []

    for t in range(t_start, t_end + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)
        r_t, cache = router_local.select_expert(x_t, available)
        phi = cache["phi"]

        mean, var, loss = _compute_probe_stats(
            router_local,
            phi,
            cache["w_pred"],
            cache["mu_g_pred"],
            cache["Sigma_g_pred"],
            cache["mu_u_pred"],
            cache["Sigma_u_pred"],
            target_expert,
        )

        preds = env.all_expert_predictions(x_t)
        residuals = preds - float(env.y[t])
        residual_r = float(residuals[int(r_t)])
        if router_local.feedback_mode == "full":
            residuals_full = _mask_feedback_vector_local(
                residuals, np.asarray(available, dtype=int), int(r_t), True
            )
        else:
            residuals_full = None

        router_local.update_beliefs(
            r_t=int(r_t),
            loss_obs=residual_r,
            losses_full=residuals_full,
            available_experts=available,
            cache=cache,
        )

        mean_post, var_post, loss_post = _compute_probe_stats(
            router_local,
            phi,
            router_local.w,
            router_local.mu_g,
            router_local.Sigma_g,
            router_local.mu_u,
            router_local.Sigma_u,
            target_expert,
        )

        times.append(int(t))
        selected.append(int(r_t))
        avail_target.append(int(env.availability[t, int(target_expert)]))
        pre_mean.append(mean)
        pre_var.append(var)
        pre_loss.append(loss)
        post_mean.append(mean_post)
        post_var.append(var_post)
        post_loss.append(loss_post)
        true_loss.append(float(residuals[int(target_expert)] ** 2))

    return {
        "times": np.asarray(times, dtype=int),
        "selected": np.asarray(selected, dtype=int),
        "avail_target": np.asarray(avail_target, dtype=int),
        "pre_mean": np.asarray(pre_mean, dtype=float),
        "pre_var": np.asarray(pre_var, dtype=float),
        "pre_loss": np.asarray(pre_loss, dtype=float),
        "post_mean": np.asarray(post_mean, dtype=float),
        "post_var": np.asarray(post_var, dtype=float),
        "post_loss": np.asarray(post_loss, dtype=float),
        "true_loss": np.asarray(true_loss, dtype=float),
    }


def _save_fig(
    fig: plt.Figure,
    out_dir: str,
    name: str,
    save_png: bool,
    save_pdf: bool,
    show: bool,
) -> None:
    if save_png:
        fig.savefig(
            os.path.join(out_dir, f"{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
    if save_pdf:
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _shade_unavailability(
    ax: Axes,
    times: np.ndarray,
    avail_mask: np.ndarray,
    color: str = "gray",
    alpha: float = 0.15,
) -> None:
    if times.size == 0:
        return
    in_block = False
    start_t = None
    for t, avail in zip(times, avail_mask):
        if not in_block and avail == 0:
            in_block = True
            start_t = int(t)
        elif in_block and avail != 0:
            in_block = False
            end_t = int(t)
            ax.axvspan(start_t, end_t, color=color, alpha=alpha, zorder=0)
    if in_block and start_t is not None:
        ax.axvspan(start_t, int(times[-1]), color=color, alpha=alpha, zorder=0)


def plot_selection_freq_by_regime(
    env: SyntheticTimeSeriesEnv,
    choices_map: dict,
    out_dir: str,
    show_plots: bool,
    save_plots: bool,
    save_png: bool,
    save_pdf: bool,
    rows: Optional[list[list[tuple[str, Optional[np.ndarray]]]]] = None,
) -> None:
    if not choices_map and not rows:
        return
    z = np.asarray(env.z[1: env.T], dtype=int)
    num_regimes = int(env.num_regimes)
    n_experts = int(env.num_experts)
    def _has_valid_choices(choice_entry: Optional[np.ndarray]) -> bool:
        if choice_entry is None:
            return False
        if isinstance(choice_entry, tuple) and len(choice_entry) == 2:
            choices_arr = np.asarray(choice_entry[0], dtype=float).reshape(-1)
        else:
            choices_arr = np.asarray(choice_entry, dtype=float).reshape(-1)
        if choices_arr.size == 0:
            return False
        return bool(np.any(np.isfinite(choices_arr)))

    use_rows = rows is not None
    if not use_rows:
        entries = [
            (name, arr)
            for name, arr in choices_map.items()
            if _has_valid_choices(arr)
        ]
        if not entries:
            return
        ncols = 3
        nrows = int(np.ceil(len(entries) / ncols))
        rows = [entries[r * ncols : (r + 1) * ncols] for r in range(nrows)]
    else:
        rows = [
            [(name, arr) for name, arr in row if _has_valid_choices(arr)]
            for row in rows
        ]
        rows = [row for row in rows if row]
        if not rows:
            return
        ncols = max(len(row) for row in rows)
        nrows = len(rows)

    vmax = 1.0
    last_im = None

    debug_lines = []
    if use_rows:
        fig = plt.figure(figsize=(3.3 * ncols, 2.8 * nrows))
        outer = GridSpec(nrows, 1, figure=fig, hspace=0.45)
        axes_rows = []
        for r, row in enumerate(rows):
            sub = GridSpecFromSubplotSpec(
                1, len(row), subplot_spec=outer[r], wspace=0.3
            )
            axes_row = [fig.add_subplot(sub[0, c]) for c in range(len(row))]
            axes_rows.append(axes_row)
    else:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.3 * ncols, 2.6 * nrows),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        axes_rows = [list(axes[r]) for r in range(nrows)]

    for r, row in enumerate(rows):
        for c, (name, choices) in enumerate(row):
            ax = axes_rows[r][c]
            if isinstance(choices, tuple) and len(choices) == 2:
                choices_arr = np.asarray(choices[0], dtype=float).reshape(-1)
            else:
                choices_arr = np.asarray(choices, dtype=float).reshape(-1)
            if choices_arr.size:
                choices_arr = np.where(choices_arr < 0, np.nan, choices_arr)
            if choices_arr.size != z.size:
                if choices_arr.size == 0:
                    ax.axis("off")
                    continue
                min_len = min(choices_arr.size, z.size)
                choices_arr = choices_arr[:min_len]
                if min_len < z.size:
                    pad = np.full(z.size - min_len, np.nan, dtype=float)
                    choices_arr = np.concatenate([choices_arr, pad], axis=0)
            debug_lines.append(
                f"{name}: choices_len={choices_arr.size}, valid={int(np.isfinite(choices_arr).sum())}"
            )
            freq = np.full((num_regimes, n_experts), np.nan, dtype=float)
            for m in range(num_regimes):
                mask = z == m
                if not np.any(mask):
                    continue
                sub = choices_arr[mask]
                sub = sub[np.isfinite(sub)]
                for k in range(n_experts):
                    freq[m, k] = float(np.mean(sub == k)) if sub.size else np.nan
            freq = np.nan_to_num(freq, nan=0.0)
            im = ax.imshow(freq, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
            last_im = im
            ax.set_title(name, fontsize=8, pad=6)
            ax.set_xticks(np.arange(n_experts))
            ax.set_yticks(np.arange(num_regimes))
            if r == nrows - 1:
                ax.set_xlabel("Expert")
            if c == 0:
                ax.set_ylabel("Regime")

    if last_im is None:
        return
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(last_im, cax=cax, label="Selection freq")
    fig.suptitle("Selection frequency by regime (all baselines)", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.9)
    if debug_lines:
        with open(os.path.join(out_dir, "expert_structure_all_debug.txt"), "w") as f:
            f.write("Selection freq debug\n")
            f.write("\n".join(debug_lines))
            f.write("\n")
    _save_fig(fig, out_dir, "expert_structure_all", save_png, save_pdf, show_plots)


def run_tri_cycle_corr_diagnostics(
    env: SyntheticTimeSeriesEnv,
    router: FactorizedSLDS,
    router_no_g: Optional[FactorizedSLDS] = None,
    label: str = "L2D SLDS w/ $g_t$",
    out_dir: str = "out/tri_cycle_corr",
    show_plots: bool = False,
    save_plots: bool = True,
    save_png: bool = True,
    save_pdf: bool = True,
    pairs: Optional[Sequence[Sequence[int]]] = None,
    t_start: int = 1,
    t_end: Optional[int] = None,
    corr_smooth_window: int = 1,
    transfer_probe: Optional[dict] = None,
) -> None:
    if getattr(env, "setting", None) != "tri_cycle_corr":
        print("[tri-cycle] Diagnostics skipped: env.setting != tri_cycle_corr.")
        return
    corr_smooth_window = max(int(corr_smooth_window), 1)
    if not save_plots:
        save_png = False
        save_pdf = False
    os.makedirs(out_dir, exist_ok=True)
    if pairs is None:
        pairs = [(0, 1), (2, 3)]
    t_end = env.T - 1 if t_end is None else min(int(t_end), env.T - 1)
    t_start = max(int(t_start), 1)
    if t_start > t_end:
        print("[tri-cycle] Diagnostics skipped: empty time window.")
        return

    residuals = _compute_residual_matrix(env, t_start, t_end)
    z = np.asarray(env.z[t_start : t_end + 1], dtype=int)
    num_regimes = int(env.num_regimes)
    true_corr = _corr_by_regime(residuals, z, num_regimes)

    diag = _collect_factorized_diagnostics(router, env, t_start, t_end)
    pred_corr = np.asarray(diag["pred_corr"], dtype=float)
    est_corr = _avg_corr_by_regime(pred_corr, z, num_regimes)
    diag_no_g = None
    est_corr_no_g = None
    if router_no_g is not None:
        diag_no_g = _collect_factorized_diagnostics(router_no_g, env, t_start, t_end)
        pred_corr_no_g = np.asarray(diag_no_g["pred_corr"], dtype=float)
        est_corr_no_g = _avg_corr_by_regime(pred_corr_no_g, z, num_regimes)
    t_grid = diag["times"]

    pred_labels = np.argmax(diag["w_post"], axis=1)
    regime_accuracy = float(np.mean(pred_labels == z)) if z.size else float("nan")

    switch_delays = []
    for idx in range(1, z.size):
        if z[idx] == z[idx - 1]:
            continue
        new_reg = int(z[idx])
        match_idx = np.where(pred_labels[idx:] == new_reg)[0]
        if match_idx.size:
            switch_delays.append(int(match_idx[0]))
    delay_mean = float(np.mean(switch_delays)) if switch_delays else float("nan")
    delay_median = float(np.median(switch_delays)) if switch_delays else float("nan")

    corr_mae = []
    for m in range(num_regimes):
        diff = est_corr[m] - true_corr[m]
        if diff.size == 0:
            corr_mae.append(float("nan"))
            continue
        mask = ~np.eye(diff.shape[0], dtype=bool)
        corr_mae.append(float(np.nanmean(np.abs(diff[mask]))))

    g_metrics = {}
    g_true = getattr(env, "_tri_cycle_g", None)
    if g_true is not None and diag["g_post"] is not None and diag["g_post"].size:
        g_true_slice = np.asarray(g_true[t_start : t_end + 1], dtype=float)
        g_pred = np.asarray(diag["g_post"], dtype=float)
        g_aligned, perm, signs, corr = _align_latent_dims(g_true_slice, g_pred)
        g_corrs = []
        for i in range(min(g_true_slice.shape[1], g_aligned.shape[1])):
            g_corrs.append(_safe_corr(g_true_slice[:, i], g_aligned[:, i]))
        g_metrics = {
            "perm": perm,
            "signs": signs.tolist(),
            "corr_matrix": corr.tolist(),
            "mean_abs_corr": float(np.nanmean(np.abs(g_corrs))),
            "per_dim_corr": g_corrs,
        }
    else:
        g_aligned = None

    n_experts = env.num_experts
    pairs = [
        (int(p[0]), int(p[1]))
        for p in pairs
        if int(p[0]) < n_experts and int(p[1]) < n_experts
    ]

    summary = {
        "label": label,
        "t_start": int(t_start),
        "t_end": int(t_end),
        "regime_accuracy": regime_accuracy,
        "switch_delay_mean": delay_mean,
        "switch_delay_median": delay_median,
        "corr_mae_by_regime": corr_mae,
        "g_recovery": g_metrics,
        "avg_cost": float(np.nanmean(diag["costs"])) if diag["costs"].size else float("nan"),
    }
    if diag_no_g is not None:
        summary["avg_cost_no_g"] = float(np.nanmean(diag_no_g["costs"]))
        summary["avg_cost_gain"] = float(
            summary["avg_cost_no_g"] - summary["avg_cost"]
        )

    with open(os.path.join(out_dir, "tri_cycle_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "tri_cycle_metrics.txt"), "w") as f:
        f.write("Tri-cycle diagnostics summary\n")
        f.write(f"Label: {label}\n")
        f.write(f"Window: t={t_start}..{t_end}\n")
        f.write(f"Regime accuracy: {regime_accuracy:.4f}\n")
        f.write(f"Switch delay (mean): {delay_mean:.3f}\n")
        f.write(f"Switch delay (median): {delay_median:.3f}\n")
        for m, err in enumerate(corr_mae):
            f.write(f"Corr MAE regime {m}: {err:.4f}\n")
        for pair in pairs:
            i, j = int(pair[0]), int(pair[1])
            if i >= n_experts or j >= n_experts:
                continue
            f.write(f"Pair {i}-{j} corr:\n")
            for m in range(num_regimes):
                f.write(
                    f"  Regime {m}: true={true_corr[m][i, j]:.4f}, "
                    f"est={est_corr[m][i, j]:.4f}\n"
                )
        if g_metrics:
            f.write(f"G recovery mean |corr|: {g_metrics['mean_abs_corr']:.4f}\n")

    # Correlation heatmaps (true vs estimated)
    n_cols = 3 if est_corr_no_g is not None else 2
    fig, axes = plt.subplots(
        num_regimes,
        n_cols,
        figsize=(3.6 * n_cols, 2.9 * max(num_regimes, 1)),
        sharex=True,
        sharey=True,
    )
    if num_regimes == 1:
        axes = np.array([axes])
    for m in range(num_regimes):
        ax_true = axes[m, 0]
        ax_est = axes[m, 1]
        im_true = ax_true.imshow(true_corr[m], vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax_est.imshow(est_corr[m], vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax_true.set_title(f"Regime {m}: true")
        ax_est.set_title(f"Regime {m}: estimated")
        if est_corr_no_g is not None:
            ax_ng = axes[m, 2]
            ax_ng.imshow(est_corr_no_g[m], vmin=-1.0, vmax=1.0, cmap="coolwarm")
            ax_ng.set_title(f"Regime {m}: no-g")
            ax_ng.set_yticks(np.arange(n_experts))
            ax_ng.set_xticks(np.arange(n_experts))
        ax_true.set_yticks(np.arange(n_experts))
        ax_est.set_yticks(np.arange(n_experts))
        ax_true.set_xticks(np.arange(n_experts))
        ax_est.set_xticks(np.arange(n_experts))
        ax_true.set_ylabel("Expert")
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im_true, cax=cax, label="Corr")
    fig.suptitle("Residual correlation by regime", y=0.98)
    fig.subplots_adjust(
        top=0.86, bottom=0.08, left=0.08, right=0.88, wspace=0.25, hspace=0.3
    )
    _save_fig(fig, out_dir, "corr_heatmaps", save_png, save_pdf, show_plots)

    # Pairwise correlation tracking over time
    if pairs:
        t_grid = diag["times"]
        change_points = np.where(z[1:] != z[:-1])[0] + 1
        n_pairs = len(pairs)
        fig, axes = plt.subplots(
            n_pairs,
            1,
            figsize=(9.0, 2.6 * max(n_pairs, 1)),
            sharex=True,
        )
        if n_pairs == 1:
            axes = [axes]
        for ax, pair in zip(axes, pairs):
            i, j = int(pair[0]), int(pair[1])
            pred_series = pred_corr[:, i, j]
            if corr_smooth_window > 1:
                kernel = np.ones(int(corr_smooth_window), dtype=float)
                kernel /= float(kernel.size)
                pred_series = np.convolve(pred_series, kernel, mode="same")
            true_series = np.array([true_corr[z_t][i, j] for z_t in z], dtype=float)
            ax.plot(t_grid, pred_series, label=f"Pred {i}-{j}")
            ax.step(
                t_grid, true_series, label="True", linestyle="--", color="black", alpha=0.6
            )
            for cp in change_points:
                if cp < t_grid.size:
                    ax.axvline(t_grid[cp], color="gray", alpha=0.2, linewidth=0.8)
            ax.set_ylabel(f"corr({i},{j})")
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Time $t$")
        axes[0].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=9,
        )
        fig.suptitle("Correlation tracking for key expert pairs", y=0.98)
        fig.subplots_adjust(top=0.86, bottom=0.1, left=0.08, right=0.82, hspace=0.35)
        _save_fig(fig, out_dir, "corr_pairs", save_png, save_pdf, show_plots)

    # Regime posterior tracking
    fig, (ax_w, ax_z) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 5.4))
    for m in range(diag["w_post"].shape[1]):
        ax_w.plot(t_grid, diag["w_post"][:, m], label=f"Regime {m}")
    ax_w.set_ylabel("Posterior weight")
    ax_w.set_title("Regime posterior over time", pad=10)
    ax_w.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        fontsize=9,
    )
    ax_w.grid(True, alpha=0.2)
    ax_z.step(t_grid, z, where="post", color="black", label="True regime")
    ax_z.set_xlabel("Time $t$")
    ax_z.set_ylabel("Regime")
    ax_z.set_yticks(np.arange(num_regimes))
    ax_z.grid(True, alpha=0.2)
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.82, hspace=0.3)
    _save_fig(fig, out_dir, "regime_posterior", save_png, save_pdf, show_plots)

    # Shared factor recovery
    if g_true is not None and g_aligned is not None:
        d_g = min(g_true.shape[1], g_aligned.shape[1])
        fig, axes = plt.subplots(
            d_g,
            1,
            sharex=True,
            figsize=(9.0, 2.6 * d_g),
        )
        if d_g == 1:
            axes = [axes]
        for i in range(d_g):
            axes[i].plot(t_grid, g_true[t_start : t_end + 1, i], label="True")
            axes[i].plot(t_grid, g_aligned[:, i], label="Inferred", alpha=0.8)
            axes[i].set_ylabel(f"g[{i}]")
            axes[i].grid(True, alpha=0.2)
            if i == 0:
                axes[i].legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    fontsize=9,
                )
        axes[-1].set_xlabel("Time $t$")
        fig.suptitle("Shared-factor recovery (aligned)", y=0.98)
        fig.subplots_adjust(
            top=0.86, bottom=0.1, left=0.08, right=0.82, hspace=0.35
        )
        _save_fig(fig, out_dir, "g_recovery", save_png, save_pdf, show_plots)

    # Expert structure: relative losses and selection frequency by regime
    losses = residuals * residuals
    avg_losses = np.full((num_regimes, n_experts), np.nan, dtype=float)
    sel_freq = np.full((num_regimes, n_experts), np.nan, dtype=float)
    for m in range(num_regimes):
        mask = z == m
        if mask.sum() == 0:
            continue
        avg_losses[m] = np.mean(losses[mask], axis=0)
        choices_m = diag["choices"][mask]
        for k in range(n_experts):
            sel_freq[m, k] = float(np.mean(choices_m == k))

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.0))
    im1 = ax.imshow(sel_freq, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Selection freq by regime", pad=8)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Regime")
    ax.set_xticks(np.arange(n_experts))
    ax.set_yticks(np.arange(num_regimes))
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.1, right=0.92)
    _save_fig(fig, out_dir, "expert_structure", save_png, save_pdf, show_plots)

    probe_cfg = transfer_probe or {}
    if probe_cfg.get("enabled", False):
        target = int(probe_cfg.get("target_expert", 0))
        source = probe_cfg.get("source_expert", None)
        source = None if source is None else int(source)
        compare_no_g = bool(probe_cfg.get("compare_no_g", True))
        show_truth = bool(probe_cfg.get("show_truth", True))
        if 0 <= target < n_experts:
            series = _collect_transfer_probe(router, env, target, t_start, t_end)
            series_no_g = None
            if compare_no_g and router_no_g is not None:
                series_no_g = _collect_transfer_probe(
                    router_no_g, env, target, t_start, t_end
                )

            fig, (ax_loss, ax_delta) = plt.subplots(
                2, 1, sharex=True, figsize=(9.0, 5.6)
            )
            ax_loss.plot(
                series["times"],
                series["post_loss"],
                label=f"Expert {target} belief (L2D SLDS)",
                color="tab:blue",
            )
            if show_truth:
                ax_loss.plot(
                    series["times"],
                    series["true_loss"],
                    label=f"Expert {target} true loss",
                    color="black",
                    alpha=0.4,
                )
            if series_no_g is not None:
                ax_loss.plot(
                    series_no_g["times"],
                    series_no_g["post_loss"],
                    label=f"Expert {target} belief (L2D SLDS w/t $g_t$)",
                    color="tab:orange",
                    linestyle="--",
                )
            _shade_unavailability(ax_loss, series["times"], series["avail_target"])
            ax_loss.set_ylabel("Predicted loss")
            ax_loss.set_title(f"Expert {target} predicted loss")
            ax_loss.grid(True, alpha=0.2)

            delta = np.abs(series["post_loss"] - series["pre_loss"])
            ax_delta.plot(
                series["times"],
                delta,
                label="L2D SLDS",
                color="tab:green",
            )
            if series_no_g is not None:
                delta_no_g = np.abs(
                    series_no_g["post_loss"] - series_no_g["pre_loss"]
                )
                ax_delta.plot(
                    series_no_g["times"],
                    delta_no_g,
                    label="L2D SLDS w/t $g_t$",
                    color="tab:red",
                    linestyle="--",
                )
            _shade_unavailability(ax_delta, series["times"], series["avail_target"])
            ax_delta.set_xlabel("Time $t$ (shaded = unavailable)")
            ax_delta.set_ylabel("Update magnitude")
            ax_delta.grid(True, alpha=0.2)

            if source is not None:
                sel_mask = series["selected"] == source
                if np.any(sel_mask):
                    y_min, y_max = ax_loss.get_ylim()
                    y_tick = y_min + 0.03 * (y_max - y_min)
                    ax_loss.vlines(
                        series["times"][sel_mask],
                        y_min,
                        y_tick,
                        color="tab:purple",
                        alpha=0.4,
                        linewidth=0.8,
                        label=f"Selected {source}",
                    )
            obs_mask = series["selected"] == target
            if np.any(obs_mask):
                y_min, y_max = ax_loss.get_ylim()
                y_tick = y_min + 0.06 * (y_max - y_min)
                ax_loss.vlines(
                    series["times"][obs_mask],
                    y_min,
                    y_tick,
                    color="tab:green",
                    alpha=0.4,
                    linewidth=0.8,
                    label=f"Selected {target}",
                )

            ax_loss.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )
            ax_delta.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )
            fig.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.8, hspace=0.35)
            _save_fig(fig, out_dir, "transfer_probe", save_png, save_pdf, show_plots)
