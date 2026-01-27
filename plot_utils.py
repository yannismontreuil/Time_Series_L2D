import copy
import itertools
import json
import os
from typing import Optional, Sequence

import numpy as np
try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    _HAS_MPL = True
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None  # type: ignore[assignment]
    Axes = object  # type: ignore[assignment]
    GridSpec = object  # type: ignore[assignment]
    GridSpecFromSubplotSpec = object  # type: ignore[assignment]
    _HAS_MPL = False

from environment.etth1_env import ETTh1TimeSeriesEnv
from models.router_model import SLDSIMMRouter
from models.factorized_slds import FactorizedSLDS
from environment.synthetic_env import SyntheticTimeSeriesEnv
from models.l2d_baseline import L2D
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB
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
if _HAS_MPL:
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


def _plots_available() -> bool:
    if not _HAS_MPL:
        return False
    return not bool(os.environ.get("FACTOR_DISABLE_PLOT_SHOW"))


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
    if not _plots_available():
        print("[plot_utils] plotting disabled or matplotlib missing; skip transition matrices.")
        return
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
    if not _plots_available():
        print("[plot_utils] plotting disabled or matplotlib missing; skip time series plot.")
        return
    T = env.T if num_points is None else min(num_points, env.T)
    t_grid = np.arange(T)

    plot_target = str(getattr(env, "plot_target", "y")).lower()
    if plot_target == "x":
        y = env.x[:T]
        if y.ndim > 1:
            feat_names = getattr(env, "context_feature_names", None)
            feat_label = (
                str(feat_names[0]) if feat_names and len(feat_names) > 0 else "x_t[0]"
            )
            y = y[:, 0]
            true_label = f"Context $x_t$ ({feat_label})"
        else:
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


def plot_time_series_experts_only(
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    num_points: Optional[int] = None,
    out_dir: str = "out/plots",
    name: str = "time_series_experts",
    save_pdf: bool = True,
    save_png: bool = False,
    show: bool = False,
) -> None:
    """
    Plot the true time series (top) and per-expert absolute errors below.
    Intended for clean ICML-style figures (no titles).
    """
    if not _plots_available():
        print("[plot_utils] plotting disabled or matplotlib missing; skip time series plot.")
        return
    T = env.T if num_points is None else min(num_points, env.T)
    t_grid = np.arange(T)

    plot_target = str(getattr(env, "plot_target", "y")).lower()
    if plot_target == "x":
        y = env.x[:T]
        if y.ndim > 1:
            feat_names = getattr(env, "context_feature_names", None)
            feat_label = (
                str(feat_names[0]) if feat_names and len(feat_names) > 0 else "x_t[0]"
            )
            y = y[:, 0]
            true_label = f"Context $x_t$ ({feat_label})"
        else:
            true_label = "Context $x_t$ (lagged)"
    else:
        y = env.y[:T]
        true_label = "True $y_t$"

    preds = np.zeros((T, env.num_experts), dtype=float)
    for t in range(T):
        x_t = env.get_context(t)
        preds[t, :] = env.all_expert_predictions(x_t)

    abs_err = np.abs(preds - y.reshape(-1, 1))
    err_mask = t_grid > 10

    n_rows = 1 + int(env.num_experts)
    fig_h = max(3.5, 1.4 * n_rows)
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10, fig_h))
    if n_rows == 1:
        axes = [axes]
    ax_top = axes[0]

    ax_top.plot(
        t_grid,
        y,
        color=get_model_color("true"),
        linewidth=2,
    )
    ax_top.set_ylabel("Value")

    for j in range(env.num_experts):
        ax = axes[j + 1]
        ax.plot(
            t_grid[err_mask],
            abs_err[err_mask, j],
            color=get_expert_color(j),
            linewidth=1.2,
            alpha=0.85,
        )
        ax.set_ylabel(f"Expert {j}\nabs. error")
        ax.set_ylim(bottom=0.0)
    axes[-1].set_xlabel("Time $t$")
    for ax in axes:
        ax.set_xlim(0, max(0, T - 1))
    plt.tight_layout()

    if save_pdf or save_png:
        os.makedirs(out_dir, exist_ok=True)
        _save_fig(fig, out_dir, name, save_png, save_pdf, show)
    else:
        if show:
            plt.show()
        plt.close(fig)

    for i in range(env.num_experts):
        rmse = np.sqrt(np.mean(abs_err[err_mask, i] ** 2))
        print(f"Expert {i}'s RMSE: {rmse:.4f}")


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
    def _run_base_router(
        router: Optional[SLDSIMMRouter], snapshot_key: Optional[str] = None
    ):
        if router is None:
            return None, None
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

    costs_partial, choices_partial = _run_base_router(
        router_partial, "router_partial"
    )
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
    beta_source = router_partial if router_partial is not None else router_full
    if beta_source is None:
        raise ValueError("At least one base router must be provided for beta.")
    beta = beta_source.beta[: env.num_experts]

    # Random and oracle baselines
    costs_random, choices_random = run_random_on_env(env, beta, seed=int(seed))
    costs_oracle, choices_oracle = run_oracle_on_env(env, beta)

    # Prediction series induced by router and L2D choices
    preds_partial = (
        compute_predictions_from_choices(env, choices_partial)
        if choices_partial is not None
        else None
    )
    preds_full = (
        compute_predictions_from_choices(env, choices_full)
        if choices_full is not None
        else None
    )
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
        costs_partial,
        getattr(router_partial, "em_tk", None) if router_partial is not None else None,
    )
    costs_full = _mask_em_costs(
        costs_full,
        getattr(router_full, "em_tk", None) if router_full is not None else None,
    )
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
        choices_partial,
        getattr(router_partial, "em_tk", None) if router_partial is not None else None,
    )
    choices_full_plot = _mask_em_choices(
        choices_full,
        getattr(router_full, "em_tk", None) if router_full is not None else None,
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
        if y_true.ndim > 1:
            feat_names = getattr(env, "context_feature_names", None)
            feat_label = (
                str(feat_names[0]) if feat_names and len(feat_names) > 0 else "x_t[0]"
            )
            y_true = y_true[:, 0]
            true_label = f"Context $x_t$ ({feat_label})"
        else:
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

    def _safe_nanmean(costs: Optional[np.ndarray]) -> Optional[float]:
        if costs is None:
            return None
        arr = np.asarray(costs, dtype=float)
        if arr.size == 0:
            return None
        return float(np.nanmean(arr))

    avg_cost_partial = _safe_nanmean(costs_partial)
    avg_cost_full = _safe_nanmean(costs_full)
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

    cost_precision = 4
    if analysis_cfg is not None:
        try:
            cost_precision = int(analysis_cfg.get("cost_precision", cost_precision))
        except (TypeError, ValueError):
            cost_precision = cost_precision
    cost_precision = max(2, min(cost_precision, 10))

    def _fmt(val: Optional[float]) -> str:
        if val is None:
            return "nan"
        try:
            return f"{float(val):.{cost_precision}f}"
        except (TypeError, ValueError):
            return "nan"

    print("=== Average costs ===")
    print(f"L2D SLDS w/t $g_t$ (partial fb): {_fmt(avg_cost_partial)}")
    print(f"L2D SLDS w/t $g_t$ (full fb):    {_fmt(avg_cost_full)}")
    if avg_cost_partial_corr is not None:
        print(
            f"Router Corr (partial feedback): {_fmt(avg_cost_partial_corr)}"
        )
    if avg_cost_full_corr is not None:
        print(f"Router Corr (full feedback):    {_fmt(avg_cost_full_corr)}")
    if avg_cost_factorized_partial is not None:
        print(f"{factorized_label} (partial fb):   {_fmt(avg_cost_factorized_partial)}")
    if avg_cost_factorized_full is not None:
        print(f"{factorized_label} (full fb):      {_fmt(avg_cost_factorized_full)}")
    if avg_cost_factorized_linear_partial is not None:
        print(
            f"{factorized_linear_label} (partial fb): {_fmt(avg_cost_factorized_linear_partial)}"
        )
    if avg_cost_factorized_linear_full is not None:
        print(
            f"{factorized_linear_label} (full fb):    {_fmt(avg_cost_factorized_linear_full)}"
        )
    if avg_cost_partial_corr_em is not None:
        print(
            f"Router Corr EM (partial fb):   {_fmt(avg_cost_partial_corr_em)}"
        )
    if avg_cost_full_corr_em is not None:
        print(
            f"Router Corr EM (full fb):      {_fmt(avg_cost_full_corr_em)}"
        )
    if avg_cost_neural_partial is not None:
        print(
            f"Neural router (partial fb):     {_fmt(avg_cost_neural_partial)}"
        )
    if avg_cost_neural_full is not None:
        print(
            f"Neural router (full fb):        {_fmt(avg_cost_neural_full)}"
        )
    if avg_cost_l2d is not None:
        print(f"L2D (full feedback):           {_fmt(avg_cost_l2d)}")
    if avg_cost_l2d_sw is not None:
        print(f"L2D_SW (full feedback):        {_fmt(avg_cost_l2d_sw)}")
    if avg_cost_linucb_partial is not None:
        print(f"LinUCB (partial feedback):     {_fmt(avg_cost_linucb_partial)}")
    if avg_cost_linucb_full is not None:
        print(f"LinUCB (full feedback):        {_fmt(avg_cost_linucb_full)}")
    if avg_cost_neuralucb_partial is not None:
        print(f"NeuralUCB (partial feedback):  {_fmt(avg_cost_neuralucb_partial)}")
    if avg_cost_neuralucb_full is not None:
        print(f"NeuralUCB (full feedback):     {_fmt(avg_cost_neuralucb_full)}")
    print(f"Random baseline:               {_fmt(avg_cost_random)}")
    print(f"Oracle baseline:               {_fmt(avg_cost_oracle)}")
    for j in range(env.num_experts):
        print(f"Always using expert {j}:       {_fmt(avg_cost_experts[j])}")

    print(
        f"\n=== Mean costs (last 20% of horizon, t >= {last_t_start}) ==="
    )
    if last_cost_partial is not None:
        print(f"L2D SLDS w/t $g_t$ (partial fb): {_fmt(last_cost_partial)}")
    if last_cost_full is not None:
        print(f"L2D SLDS w/t $g_t$ (full fb):    {_fmt(last_cost_full)}")
    if last_cost_partial_corr is not None:
        print(
            f"Router Corr (partial feedback): {_fmt(last_cost_partial_corr)}"
        )
    if last_cost_full_corr is not None:
        print(f"Router Corr (full feedback):    {_fmt(last_cost_full_corr)}")
    if last_cost_factorized_partial is not None:
        print(f"{factorized_label} (partial fb):   {_fmt(last_cost_factorized_partial)}")
    if last_cost_factorized_full is not None:
        print(f"{factorized_label} (full fb):      {_fmt(last_cost_factorized_full)}")
    if last_cost_factorized_linear_partial is not None:
        print(
            f"{factorized_linear_label} (partial fb): {_fmt(last_cost_factorized_linear_partial)}"
        )
    if last_cost_factorized_linear_full is not None:
        print(
            f"{factorized_linear_label} (full fb):    {_fmt(last_cost_factorized_linear_full)}"
        )
    if last_cost_partial_corr_em is not None:
        print(
            f"Router Corr EM (partial fb):   {_fmt(last_cost_partial_corr_em)}"
        )
    if last_cost_full_corr_em is not None:
        print(
            f"Router Corr EM (full fb):      {_fmt(last_cost_full_corr_em)}"
        )
    if last_cost_neural_partial is not None:
        print(
            f"Neural router (partial fb):     {_fmt(last_cost_neural_partial)}"
        )
    if last_cost_neural_full is not None:
        print(
            f"Neural router (full fb):        {_fmt(last_cost_neural_full)}"
        )
    if last_cost_l2d is not None:
        print(f"L2D (full feedback):           {_fmt(last_cost_l2d)}")
    if last_cost_l2d_sw is not None:
        print(f"L2D_SW (full feedback):        {_fmt(last_cost_l2d_sw)}")
    if last_cost_linucb_partial is not None:
        print(f"LinUCB (partial feedback):     {_fmt(last_cost_linucb_partial)}")
    if last_cost_linucb_full is not None:
        print(f"LinUCB (full feedback):        {_fmt(last_cost_linucb_full)}")
    if last_cost_neuralucb_partial is not None:
        print(f"NeuralUCB (partial feedback):  {_fmt(last_cost_neuralucb_partial)}")
    if last_cost_neuralucb_full is not None:
        print(f"NeuralUCB (full feedback):     {_fmt(last_cost_neuralucb_full)}")
    if last_cost_random is not None:
        print(f"Random baseline:               {_fmt(last_cost_random)}")
    if last_cost_oracle is not None:
        print(f"Oracle baseline:               {_fmt(last_cost_oracle)}")
    for j in range(env.num_experts):
        print(
            f"Always using expert {j}:       {_fmt(avg_cost_experts_last[j])}"
        )

    corr_cfg = analysis_cfg.get("pred_target_corr", {}) if analysis_cfg else {}
    corr_enabled = bool(corr_cfg.get("enabled", False))
    if corr_enabled:
        corr_window = corr_cfg.get("window", None)
        if corr_window is None:
            corr_window = getattr(env, "analysis_window", None)
        min_points = int(corr_cfg.get("min_points", 10))
        use_fisher = bool(corr_cfg.get("use_fisher", True))
        corr_out_dir = corr_cfg.get("out_dir", None)
        if corr_out_dir is None and analysis_cfg is not None:
            tri_cfg = analysis_cfg.get("tri_cycle_corr", {}) or {}
            corr_out_dir = tri_cfg.get("out_dir", None)

        target_series = np.asarray(env.y[1:T], dtype=float)

        def _mask_preds(
            preds: Optional[np.ndarray], em_tk: Optional[int]
        ) -> Optional[np.ndarray]:
            if preds is None or em_tk is None:
                return preds
            arr = np.asarray(preds, dtype=float).copy()
            try:
                cut = min(int(em_tk), arr.shape[0])
            except (TypeError, ValueError):
                return arr
            if cut > 0:
                arr[:cut] = np.nan
            return arr

        corr_entries = []

        def _add_corr(
            label: str, preds: Optional[np.ndarray], em_tk: Optional[int]
        ) -> None:
            if preds is None:
                return
            preds_masked = _mask_preds(preds, em_tk)
            if preds_masked is None:
                return
            preds_arr = np.asarray(preds_masked, dtype=float).reshape(-1)
            n = min(preds_arr.size, target_series.size)
            if n == 0:
                corr_val = float("nan")
                global_corr = float("nan")
                n_windows = 0
                n_points = 0
                corr_list = []
                window_sizes = []
                corr_window_mean = float("nan")
                corr_window_std = float("nan")
            else:
                preds_arr = preds_arr[:n]
                target_arr = target_series[:n]
                (
                    corr_val,
                    n_windows,
                    n_points,
                    corr_list,
                    window_sizes,
                ) = _windowed_corr(
                    preds_arr, target_arr, corr_window, min_points, use_fisher
                )
                global_corr = _safe_corr_masked(preds_arr, target_arr)
                if corr_list:
                    corr_window_mean = float(np.mean(corr_list))
                    if len(corr_list) > 1:
                        corr_window_std = float(np.std(corr_list, ddof=1))
                    else:
                        corr_window_std = 0.0
                else:
                    corr_window_mean = float("nan")
                    corr_window_std = float("nan")
            corr_entries.append(
                {
                    "label": label,
                    "corr": corr_val,
                    "corr_global": global_corr,
                    "n_windows": n_windows,
                    "n_points": n_points,
                    "corr_window_mean": corr_window_mean,
                    "corr_window_std": corr_window_std,
                    "window_corrs": corr_list if n > 0 else [],
                    "window_sizes": window_sizes if n > 0 else [],
                }
            )

        _add_corr(
            "L2D SLDS w/t $g_t$ (partial)",
            preds_partial,
            getattr(router_partial, "em_tk", None),
        )
        _add_corr(
            "L2D SLDS w/t $g_t$ (full)",
            preds_full,
            getattr(router_full, "em_tk", None),
        )
        _add_corr(
            "Router Corr (partial)",
            preds_partial_corr,
            getattr(router_partial_corr, "em_tk", None)
            if router_partial_corr is not None
            else None,
        )
        _add_corr(
            "Router Corr (full)",
            preds_full_corr,
            getattr(router_full_corr, "em_tk", None)
            if router_full_corr is not None
            else None,
        )
        _add_corr(
            "Router Corr EM (partial)",
            preds_partial_corr_em,
            getattr(router_partial_corr_em, "em_tk", None)
            if router_partial_corr_em is not None
            else None,
        )
        _add_corr(
            "Router Corr EM (full)",
            preds_full_corr_em,
            getattr(router_full_corr_em, "em_tk", None)
            if router_full_corr_em is not None
            else None,
        )
        _add_corr(
            f"{factorized_label} (partial)",
            preds_factorized_partial,
            getattr(router_factorial_partial, "em_tk", None)
            if router_factorial_partial is not None
            else None,
        )
        _add_corr(
            f"{factorized_label} (full)",
            preds_factorized_full,
            getattr(router_factorial_full, "em_tk", None)
            if router_factorial_full is not None
            else None,
        )
        _add_corr(
            f"{factorized_linear_label} (partial)",
            preds_factorized_linear_partial,
            getattr(router_factorial_partial_linear, "em_tk", None)
            if router_factorial_partial_linear is not None
            else None,
        )
        _add_corr(
            f"{factorized_linear_label} (full)",
            preds_factorized_linear_full,
            getattr(router_factorial_full_linear, "em_tk", None)
            if router_factorial_full_linear is not None
            else None,
        )
        _add_corr(
            "Neural router (partial)",
            preds_partial_neural,
            getattr(router_partial_neural, "em_tk", None)
            if router_partial_neural is not None
            else None,
        )
        _add_corr(
            "Neural router (full)",
            preds_full_neural,
            getattr(router_full_neural, "em_tk", None)
            if router_full_neural is not None
            else None,
        )
        _add_corr("L2D (full)", preds_l2d, em_tk_anchor)
        _add_corr("L2D_SW (full)", preds_l2d_sw, em_tk_anchor)
        _add_corr("LinUCB (partial)", preds_linucb_partial, em_tk_anchor)
        _add_corr("LinUCB (full)", preds_linucb_full, em_tk_anchor)
        _add_corr("NeuralUCB (partial)", preds_neuralucb_partial, em_tk_anchor)
        _add_corr("NeuralUCB (full)", preds_neuralucb_full, em_tk_anchor)
        _add_corr("Random baseline", preds_random, em_tk_anchor)
        _add_corr("Oracle baseline", preds_oracle, em_tk_anchor)

        expert_preds = np.zeros((T - 1, env.num_experts), dtype=float)
        for t in range(1, T):
            expert_preds[t - 1] = env.all_expert_predictions(env.get_context(t))
        for j in range(env.num_experts):
            _add_corr(f"Expert {j}", expert_preds[:, j], em_tk_anchor)

        if corr_entries:
            method_label = "Fisher-avg" if use_fisher else "mean"
            try:
                window_int = int(corr_window) if corr_window is not None else None
            except (TypeError, ValueError):
                window_int = None
            window_label = (
                "full horizon"
                if window_int is None or window_int <= 1
                else f"window={window_int}"
            )
            print(f"\n=== Prediction-target correlation ({method_label}, {window_label}) ===")
            for entry in corr_entries:
                print(
                    f"{entry['label']}: {_fmt(entry['corr'])} "
                    f"(global={_fmt(entry['corr_global'])}, "
                    f"windows={entry['n_windows']}, n={entry['n_points']}, "
                    f"win_mean={_fmt(entry['corr_window_mean'])}, "
                    f"win_std={_fmt(entry['corr_window_std'])})"
                )

            if corr_out_dir:
                os.makedirs(corr_out_dir, exist_ok=True)
                payload = {
                    "window": corr_window,
                    "min_points": min_points,
                    "use_fisher": use_fisher,
                    "metrics": corr_entries,
                }
                with open(
                    os.path.join(corr_out_dir, "pred_target_corr.json"), "w"
                ) as f:
                    json.dump(payload, f, indent=2)
                with open(
                    os.path.join(corr_out_dir, "pred_target_corr.txt"), "w"
                ) as f:
                    f.write(
                        f"Prediction-target correlation ({method_label}, {window_label})\n"
                    )
                    for entry in corr_entries:
                        f.write(
                        f"{entry['label']}: {_fmt(entry['corr'])} "
                        f"(global={_fmt(entry['corr_global'])}, "
                        f"windows={entry['n_windows']}, n={entry['n_points']}, "
                        f"win_mean={_fmt(entry['corr_window_mean'])}, "
                        f"win_std={_fmt(entry['corr_window_std'])})\n"
                    )

    # Selection distribution (how often each expert is chosen)
    entries = []
    if choices_partial_plot is not None:
        entries.append(("partial", choices_partial_plot))
    if choices_full_plot is not None:
        entries.append(("full", choices_full_plot))
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
    if not _plots_available():
        print("[plot_utils] plotting disabled or matplotlib missing; skipping plots.")
        return
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
    if preds_partial_plot is not None:
        ax_pred.plot(
            t_grid_plot,
            preds_partial_plot,
            label="L2D SLDS w/t $g_t$ (partial)",
            color=get_model_color("partial"),
            linestyle="-",
            alpha=0.8,
        )
    if preds_full_plot is not None:
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

    if avg_partial_t is not None:
        ax_cost.plot(
            t_grid,
            avg_partial_t,
            label="L2D SLDS w/t $g_t$ (partial, avg cost)",
            color=get_model_color("partial"),
            linestyle="-",
        )
    if avg_full_t is not None:
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
        print(f"Oracle (truth):                {_fmt(mean_oracle)}")
        if mean_partial is not None:
            print(f"L2D SLDS w/t $g_t$ (partial): {_fmt(mean_partial)}")
        if mean_full is not None:
            print(f"L2D SLDS w/t $g_t$ (full):    {_fmt(mean_full)}")
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
            print(f"Router Corr (partial):         {_fmt(mean_partial_corr)}")
        if mean_full_corr is not None:
            print(f"Router Corr (full):            {_fmt(mean_full_corr)}")
        if mean_l2d is not None:
            print(f"L2D (full feedback):           {_fmt(mean_l2d)}")
        if mean_l2d_sw is not None:
            print(f"L2D_SW (full feedback):        {_fmt(mean_l2d_sw)}")
        if mean_linucb_partial is not None:
            print(f"LinUCB (partial):              {_fmt(mean_linucb_partial)}")
        if mean_linucb_full is not None:
            print(f"LinUCB (full):                 {_fmt(mean_linucb_full)}")

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

    # Plot the expert index chosen over time for partial-feedback methods
    # (ICML style: no titles; partial baselines only).
    if _plots_available():
        sel_cfg = analysis_cfg or {}
        plot_enabled = bool(sel_cfg.get("selection_plot", True))
        if plot_enabled:
            avail = getattr(env, "availability", None)
            has_avail = avail is not None

            # Labels consistent with Paper/Section/AppendixParts/Experiments.tex.
            label_main = "L2D-SLDS"
            label_ablation = r"L2D-SLDS w/o $\mathbf{g}_t$"

            series = []
            if choices_factorial_partial_plot is not None:
                series.append(
                    (label_main, choices_factorial_partial_plot, "factorized_partial")
                )
            if choices_partial_plot is not None:
                base_label = label_ablation if choices_factorial_partial_plot is not None else label_main
                series.append((base_label, choices_partial_plot, "partial"))
            if choices_linucb_partial_plot is not None:
                series.append(("LinUCB", choices_linucb_partial_plot, "linucb_partial"))
            if choices_neuralucb_partial_plot is not None:
                series.append(("NeuralUCB", choices_neuralucb_partial_plot, "neuralucb"))
            if choices_oracle_plot is not None:
                series.append(("Oracle", choices_oracle_plot, "oracle"))

            if series or has_avail:
                n_rows = len(series) + (1 if has_avail else 0)
                fig2, axes = plt.subplots(
                    n_rows, 1, sharex=True, figsize=(10, 2 * n_rows)
                )
                if n_rows == 1:
                    axes = [axes]

                idx = 0
                for label, choices_plot, color_key in series:
                    ax = axes[idx]
                    ax.step(
                        t_grid,
                        choices_plot,
                        where="post",
                        color=get_model_color(color_key),
                    )
                    ax.set_ylabel(f"{label}")
                    ax.set_yticks(np.arange(env.num_experts))
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
                    ax_avail.set_ylabel("Availability")
                    ax_avail.set_yticks([0, 1])
                    ax_avail.set_xlabel("Time $t$")
                    ax_avail.legend(
                        loc="upper left",
                        bbox_to_anchor=(1.01, 1.0),
                        borderaxespad=0.0,
                        ncol=1,
                        fontsize=9,
                        frameon=False,
                    )
                else:
                    axes[idx - 1].set_xlabel("Time $t$")

                plt.tight_layout()

                out_dir = str(sel_cfg.get("selection_plot_out_dir", "out/plots"))
                name = str(sel_cfg.get("selection_plot_name", "selections_availability"))
                save_pdf = bool(sel_cfg.get("selection_plot_save_pdf", True))
                save_png = bool(sel_cfg.get("selection_plot_save_png", False))
                show_plots = bool(sel_cfg.get("selection_plot_show", False))
                if save_pdf or save_png:
                    os.makedirs(out_dir, exist_ok=True)
                    _save_fig(fig2, out_dir, name, save_png, save_pdf, show_plots)
                else:
                    if show_plots:
                        plt.show()
                    plt.close(fig2)

    tri_cfg = analysis_cfg.get("tri_cycle_corr", {}) if analysis_cfg else {}
    if tri_cfg.get("expert_structure_baselines", False):
        out_dir = str(tri_cfg.get("out_dir", "out/tri_cycle_corr"))
        os.makedirs(out_dir, exist_ok=True)
        def _label_for_factorized(
            router_obj: Optional[object],
            with_g_label: str = "L2D-SLDS",
            no_g_label: str = "L2D-SLDS w/o $g_t$",
            fallback: str = "SLDS-IMM",
        ) -> str:
            if isinstance(router_obj, FactorizedSLDS):
                d_g_val = getattr(router_obj, "d_g", None)
                if d_g_val == 0:
                    return no_g_label
                return with_g_label
            return fallback

        base_partial_label = _label_for_factorized(router_partial)
        base_full_label = _label_for_factorized(router_full)
        factorized_partial_label = _label_for_factorized(
            router_factorial_partial, with_g_label="L2D-SLDS", fallback="L2D-SLDS"
        )
        factorized_full_label = _label_for_factorized(
            router_factorial_full, with_g_label="L2D-SLDS", fallback="L2D-SLDS"
        )
        choices_map = {
            f"{base_partial_label}": (choices_partial, costs_partial),
            f"{base_full_label} (full)": (choices_full, costs_full),
            "Corr SLDS-IMM": (choices_partial_corr, costs_partial_corr),
            "Corr SLDS-IMM (full)": (choices_full_corr, costs_full_corr),
            "Corr SLDS-IMM EM": (choices_partial_corr_em, costs_partial_corr_em),
            "Corr SLDS-IMM EM (full)": (choices_full_corr_em, costs_full_corr_em),
            f"{factorized_partial_label}": (choices_factorial_partial, costs_factorial_partial),
            f"{factorized_full_label} (full)": (choices_factorial_full, costs_factorial_full),
            "L2D": (choices_l2d, costs_l2d),
            "L2D_SW": (choices_l2d_sw, costs_l2d_sw),
            "LinUCB": (choices_linucb_partial, costs_linucb_partial),
            "LinUCB full": (choices_linucb_full, costs_linucb_full),
            "NeuralUCB": (choices_neuralucb_partial, costs_neuralucb_partial),
            "NeuralUCB full": (choices_neuralucb_full, costs_neuralucb_full),
            "Oracle": (choices_oracle, costs_oracle),
        }
        rows = [
            [("Oracle", (choices_oracle, costs_oracle))],
            [
                (f"{factorized_partial_label}", (choices_factorial_partial, costs_factorial_partial)),
                (f"{base_partial_label}", (choices_partial, costs_partial)),
                ("Corr SLDS-IMM", (choices_partial_corr, costs_partial_corr)),
                ("Corr SLDS-IMM EM", (choices_partial_corr_em, costs_partial_corr_em)),
                ("LinUCB", (choices_linucb_partial, costs_linucb_partial)),
                ("NeuralUCB", (choices_neuralucb_partial, costs_neuralucb_partial)),
            ],
            [
                (f"{factorized_full_label} (full)", (choices_factorial_full, costs_factorial_full)),
                (f"{base_full_label} (full)", (choices_full, costs_full)),
                ("Corr SLDS-IMM (full)", (choices_full_corr, costs_full_corr)),
                ("Corr SLDS-IMM EM (full)", (choices_full_corr_em, costs_full_corr_em)),
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
            show_figure_title=False,
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
    event_window_pre: int = 100,
    event_window_post: int = 200,
    adoption_threshold: float = 0.5,
    out_dir: str = "out/pruning",
    show_plots: bool = False,
    save_plots: bool = True,
    save_png: bool = True,
    save_pdf: bool = True,
    label_full: str = "L2D-SLDS",
    label_no_g: str = "L2D-SLDS w/o $g_t$",
) -> None:
    if not _plots_available():
        print("[plot_utils] plotting disabled or matplotlib missing; skip pruning dynamics.")
        return
    if not isinstance(router_full, FactorizedSLDS) or not isinstance(router_no_g, FactorizedSLDS):
        raise ValueError("plot_pruning_dynamics expects FactorizedSLDS routers.")
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive.")
    if event_window_pre < 0 or event_window_post < 0:
        raise ValueError("event_window_pre/post must be non-negative.")
    if not (0.0 < adoption_threshold <= 1.0):
        raise ValueError("adoption_threshold must be in (0, 1].")

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
        rebirth_times = []
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
            if prev_in_registry is not None and not prev_in_registry and curr_in_registry:
                rebirth_times.append(int(t))
            prev_in_registry = curr_in_registry

            w_pred = cache.get("w_pred", None)
            stats = cache.get("stats", {})
            if w_pred is not None and j in stats:
                cov_modes = np.asarray(stats[j]["cov"], dtype=float)
                w_arr = np.asarray(w_pred, dtype=float)
                if cov_modes.ndim == 3 and cov_modes.shape[1:] == (1, 1):
                    var_modes = cov_modes.reshape(-1)
                    pred_var[idx] = float(np.dot(w_arr, var_modes))
                else:
                    pred_var[idx] = float(
                        np.sum(
                            [
                                w_arr[m] * float(np.trace(cov_modes[m]))
                                for m in range(w_arr.shape[0])
                            ]
                        )
                    )
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
                    cov_modes = np.asarray(stats_tmp[j]["cov"], dtype=float)
                    w_arr = np.asarray(w_pred, dtype=float)
                    if cov_modes.ndim == 3 and cov_modes.shape[1:] == (1, 1):
                        var_modes = cov_modes.reshape(-1)
                        pred_var[idx] = float(np.dot(w_arr, var_modes))
                    else:
                        pred_var[idx] = float(
                            np.sum(
                                [
                                    w_arr[m] * float(np.trace(cov_modes[m]))
                                    for m in range(w_arr.shape[0])
                                ]
                            )
                        )

            if getattr(router, "observation_mode", "loss") == "residual":
                y_t = np.asarray(env.y[int(t)], dtype=float)
                preds = np.asarray(env.all_expert_predictions(x_t), dtype=float)
                residuals = preds - y_t
                residual_r = np.asarray(residuals[int(r_t)], dtype=float)
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

        return in_registry, selected, pred_var, prune_times, rebirth_times

    in_reg_full, sel_full, var_full, prune_full, rebirth_full = _run_router(
        router_full
    )
    in_reg_no_g, sel_no_g, var_no_g, prune_no_g, rebirth_no_g = _run_router(
        router_no_g
    )

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
    do_save_png = bool(save_plots and save_png)
    do_save_pdf = bool(save_plots and save_pdf)
    _save_fig(
        fig,
        out_dir,
        f"pruning_dynamics_expert_{j}",
        do_save_png,
        do_save_pdf,
        show_plots,
    )

    def _align_series(
        series: np.ndarray, event_times: Sequence[int]
    ) -> Optional[np.ndarray]:
        if not event_times:
            return None
        series = np.asarray(series, dtype=float).reshape(-1)
        aligned = []
        for t_ev in event_times:
            idx_ev = int(t_ev) - int(times[0])
            start = idx_ev - int(event_window_pre)
            end = idx_ev + int(event_window_post)
            window = np.full(event_window_pre + event_window_post + 1, np.nan, dtype=float)
            src_start = max(start, 0)
            src_end = min(end, series.size - 1)
            dst_start = src_start - start
            dst_end = dst_start + (src_end - src_start) + 1
            if src_end >= src_start:
                window[dst_start:dst_end] = series[src_start : src_end + 1]
            aligned.append(window)
        return np.vstack(aligned) if aligned else None

    def _event_mean_ci(arr: Optional[np.ndarray]) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if arr is None or arr.size == 0:
            return None
        n = np.sum(np.isfinite(arr), axis=0)
        if not np.any(n > 0):
            return None
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
        sem = np.where(n > 1, std / np.sqrt(n), np.nan)
        return mean, mean - 1.96 * sem, mean + 1.96 * sem

    sel_ind_full = (sel_full == j).astype(float)
    sel_ind_no_g = (sel_no_g == j).astype(float)
    sel_smooth_full = _rolling_mean(sel_ind_full, rolling_window)
    sel_smooth_no_g = _rolling_mean(sel_ind_no_g, rolling_window)

    aligned_sel_full = _align_series(sel_smooth_full, rebirth_full)
    aligned_sel_no_g = _align_series(sel_smooth_no_g, rebirth_no_g)
    aligned_var_full = _align_series(var_full, rebirth_full)
    aligned_var_no_g = _align_series(var_no_g, rebirth_no_g)

    stats_sel_full = _event_mean_ci(aligned_sel_full)
    stats_sel_no_g = _event_mean_ci(aligned_sel_no_g)
    stats_var_full = _event_mean_ci(aligned_var_full)
    stats_var_no_g = _event_mean_ci(aligned_var_no_g)

    if stats_sel_full or stats_sel_no_g:
        x_rel = np.arange(-event_window_pre, event_window_post + 1, dtype=int)
        fig2, axes2 = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True)
        axes2[0].axvline(0, color="black", linewidth=1.0, alpha=0.6)
        axes2[1].axvline(0, color="black", linewidth=1.0, alpha=0.6)

        if stats_sel_full:
            mean, lo, hi = stats_sel_full
            axes2[0].plot(x_rel, mean, color="tab:blue", label=label_full)
            axes2[0].fill_between(x_rel, lo, hi, color="tab:blue", alpha=0.2)
        if stats_sel_no_g:
            mean, lo, hi = stats_sel_no_g
            axes2[0].plot(x_rel, mean, color="tab:orange", linestyle="--", label=label_no_g)
            axes2[0].fill_between(x_rel, lo, hi, color="tab:orange", alpha=0.2)
        axes2[0].set_ylabel(f"Selection freq (win={rolling_window})")
        axes2[0].set_title(f"Expert {j}: rebirth-aligned selection frequency")
        axes2[0].legend(loc="upper right", fontsize=9)

        if stats_var_full:
            mean, lo, hi = stats_var_full
            axes2[1].plot(x_rel, mean, color="tab:blue", label=label_full)
            axes2[1].fill_between(x_rel, lo, hi, color="tab:blue", alpha=0.2)
        if stats_var_no_g:
            mean, lo, hi = stats_var_no_g
            axes2[1].plot(x_rel, mean, color="tab:orange", linestyle="--", label=label_no_g)
            axes2[1].fill_between(x_rel, lo, hi, color="tab:orange", alpha=0.2)
        axes2[1].set_ylabel("Predictive variance")
        axes2[1].set_xlabel("Time since rebirth")
        axes2[1].set_title(f"Expert {j}: rebirth-aligned uncertainty")

        plt.tight_layout()
        _save_fig(
            fig2,
            out_dir,
            f"pruning_rebirth_event_study_expert_{j}",
            do_save_png,
            do_save_pdf,
            show_plots,
            keep_axis_titles_pdf=True,
        )

        def _recovery_times(
            aligned_sel: Optional[np.ndarray],
            threshold: float,
        ) -> list[int]:
            if aligned_sel is None:
                return []
            times_list = []
            for row in aligned_sel:
                post = row[event_window_pre:]
                above = np.where(post >= threshold)[0]
                if above.size == 0:
                    continue
                times_list.append(int(above[0]))
            return times_list

        rec_full = _recovery_times(aligned_sel_full, adoption_threshold)
        rec_no_g = _recovery_times(aligned_sel_no_g, adoption_threshold)
        summary = {
            "expert_idx": int(j),
            "event_window_pre": int(event_window_pre),
            "event_window_post": int(event_window_post),
            "rolling_window": int(rolling_window),
            "adoption_threshold": float(adoption_threshold),
            "n_rebirth_full": int(len(rebirth_full)),
            "n_rebirth_no_g": int(len(rebirth_no_g)),
            "recovery_times_full": rec_full,
            "recovery_times_no_g": rec_no_g,
        }
        summary["recovery_mean_full"] = (
            float(np.mean(rec_full)) if rec_full else float("nan")
        )
        summary["recovery_mean_no_g"] = (
            float(np.mean(rec_no_g)) if rec_no_g else float("nan")
        )
        summary["recovery_median_full"] = (
            float(np.median(rec_full)) if rec_full else float("nan")
        )
        summary["recovery_median_no_g"] = (
            float(np.median(rec_no_g)) if rec_no_g else float("nan")
        )
        summary_path = os.path.join(
            out_dir, f"pruning_rebirth_summary_expert_{j}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

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


def _safe_corr_masked(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _windowed_corr(
    preds: np.ndarray,
    target: np.ndarray,
    window: Optional[int],
    min_points: int,
    use_fisher: bool,
) -> tuple[float, int, int, list[float], list[int]]:
    preds = np.asarray(preds, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    n = min(preds.size, target.size)
    if n == 0:
        return float("nan"), 0, 0, [], []
    preds = preds[:n]
    target = target[:n]
    if window is None:
        mask = np.isfinite(preds) & np.isfinite(target)
        n_points = int(mask.sum())
        if n_points < int(min_points):
            return float("nan"), 0, n_points, [], []
        corr_val = _safe_corr_masked(preds, target)
        if not np.isfinite(corr_val):
            return float("nan"), 0, n_points, [], []
        return corr_val, 1, n_points, [corr_val], [n_points]
    try:
        window = int(window)
    except (TypeError, ValueError):
        window = n
    if window <= 1 or window >= n:
        mask = np.isfinite(preds) & np.isfinite(target)
        n_points = int(mask.sum())
        if n_points < int(min_points):
            return float("nan"), 0, n_points, [], []
        corr_val = _safe_corr_masked(preds, target)
        if not np.isfinite(corr_val):
            return float("nan"), 0, n_points, [], []
        return corr_val, 1, n_points, [corr_val], [n_points]

    min_points = max(int(min_points), 2)
    corr_list: list[float] = []
    weights: list[int] = []
    total_points = 0
    for start in range(0, n, window):
        p_seg = preds[start : start + window]
        t_seg = target[start : start + window]
        mask = np.isfinite(p_seg) & np.isfinite(t_seg)
        n_seg = int(mask.sum())
        if n_seg < min_points:
            continue
        corr = _safe_corr_masked(p_seg, t_seg)
        if not np.isfinite(corr):
            continue
        corr_list.append(corr)
        weights.append(n_seg)
        total_points += n_seg
    if not corr_list:
        return float("nan"), 0, total_points, [], []

    corr_arr = np.asarray(corr_list, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    if use_fisher:
        z = np.arctanh(np.clip(corr_arr, -0.999999, 0.999999))
        z_mean = float(np.average(z, weights=weights_arr))
        corr_val = float(np.tanh(z_mean))
    else:
        corr_val = float(np.average(corr_arr, weights=weights_arr))
    return corr_val, len(corr_list), total_points, corr_list, weights


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
        preds = np.asarray(env.all_expert_predictions(x_t), dtype=float)
        y_t = np.asarray(env.y[t], dtype=float)
        res_all = preds - y_t
        if res_all.ndim == 2:
            residuals[idx] = np.linalg.norm(res_all, axis=1)
        else:
            residuals[idx] = res_all
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


def _corr_by_regime_masked(
    series: np.ndarray, z: np.ndarray, num_regimes: int
) -> list[np.ndarray]:
    series = np.asarray(series, dtype=float)
    z = np.asarray(z, dtype=int)
    n_experts = int(series.shape[1]) if series.ndim == 2 else 0
    corr_list: list[np.ndarray] = []
    for m in range(num_regimes):
        mask = z == m
        if series.ndim != 2 or int(mask.sum()) < 2:
            corr = np.full((n_experts, n_experts), np.nan, dtype=float)
        else:
            corr = np.full((n_experts, n_experts), np.nan, dtype=float)
            for i in range(n_experts):
                for j in range(n_experts):
                    corr[i, j] = _safe_corr_masked(series[mask, i], series[mask, j])
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


def _avg_cond_corr_by_regime(
    corr_series: np.ndarray, z: np.ndarray, num_regimes: int
) -> list[np.ndarray]:
    corr_series = np.asarray(corr_series, dtype=float)
    z = np.asarray(z, dtype=int)
    if corr_series.ndim != 4:
        return [np.zeros((0, 0), dtype=float) for _ in range(num_regimes)]
    n_experts = int(corr_series.shape[2])
    avg_list: list[np.ndarray] = []
    for m in range(num_regimes):
        mask = z == m
        if mask.sum() < 1:
            avg = np.full((n_experts, n_experts), np.nan, dtype=float)
        else:
            avg = np.nanmean(corr_series[mask, m], axis=0)
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
        H_rows = []
        for k in avail:
            H_g = np.asarray(stats[k]["H_g"], dtype=float)
            if H_g.ndim == 2:
                H_rows.append(H_g[0])
            else:
                H_rows.append(np.asarray(H_g, dtype=float).reshape(-1))
        H = np.vstack(H_rows).astype(float)
    for m in range(M):
        for idx, k in enumerate(avail):
            mean_modes = np.asarray(stats[k]["mean"], dtype=float)
            noise_modes = np.asarray(stats[k]["noise"], dtype=float)
            if mean_modes.ndim == 2 and mean_modes.shape[1] > 1:
                means_m[m, idx] = float(mean_modes[m, 0])
                s_m[m, idx] = float(noise_modes[m, 0, 0])
            else:
                means_m[m, idx] = float(mean_modes.reshape(-1)[m])
                s_m[m, idx] = float(noise_modes.reshape(-1)[m])
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


def _predictive_corrs_by_regime_from_cache(
    cache: dict,
    available: Sequence[int],
    n_experts: int,
    num_regimes: int,
) -> np.ndarray:
    stats = cache.get("stats", {}) or {}
    Sigma_g_pred = np.asarray(cache.get("Sigma_g_pred", []), dtype=float)
    avail = [int(k) for k in available if int(k) in stats]
    full = np.full((num_regimes, n_experts, n_experts), np.nan, dtype=float)
    if not avail:
        return full
    if Sigma_g_pred.ndim == 3 and Sigma_g_pred.shape[0] >= num_regimes:
        d_g = int(Sigma_g_pred.shape[1])
    else:
        d_g = 0
    H = None
    if d_g > 0:
        H_rows = []
        for k in avail:
            H_g = np.asarray(stats[k]["H_g"], dtype=float)
            if H_g.ndim == 2:
                H_rows.append(H_g[0])
            else:
                H_rows.append(np.asarray(H_g, dtype=float).reshape(-1))
        H = np.vstack(H_rows).astype(float)
    for m in range(num_regimes):
        if d_g > 0:
            cov_shared = H @ Sigma_g_pred[m] @ H.T
        else:
            cov_shared = np.zeros((len(avail), len(avail)), dtype=float)
        s_m = np.zeros(len(avail), dtype=float)
        for idx, k in enumerate(avail):
            noise_modes = np.asarray(stats[k].get("noise", []), dtype=float)
            if noise_modes.ndim == 3 and noise_modes.shape[1] > 1:
                s_m[idx] = float(noise_modes[m, 0, 0])
            else:
                s_arr = noise_modes.reshape(-1)
                s_m[idx] = float(s_arr[m]) if m < s_arr.size else float("nan")
        cov = cov_shared + np.diag(s_m)
        corr_small = _cov_to_corr(cov)
        for i, ki in enumerate(avail):
            for j, kj in enumerate(avail):
                full[m, ki, kj] = corr_small[i, j]
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
    corr_regime_list = []
    pred_loss_list = []

    for t in range(t_start, t_end + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)
        r_t, cache = router_local.select_expert(x_t, available)
        pred_corr = _predictive_corr_from_cache(cache, available, env.num_experts)
        pred_corr_regime = _predictive_corrs_by_regime_from_cache(
            cache, available, env.num_experts, env.num_regimes
        )
        pred_loss = np.full(env.num_experts, np.nan, dtype=float)
        w_pred = np.asarray(cache.get("w_pred", []), dtype=float).reshape(-1)
        stats = cache.get("stats", {}) or {}
        if w_pred.size > 0 and stats:
            for k, stat in stats.items():
                mean_modes = np.asarray(stat.get("mean", []), dtype=float)
                cov_modes = np.asarray(stat.get("cov", []), dtype=float)
                if mean_modes.shape[0] != w_pred.size or cov_modes.shape[0] != w_pred.size:
                    continue
                loss_modes = np.zeros(w_pred.size, dtype=float)
                for m in range(w_pred.size):
                    loss_modes[m] = router_local._expected_loss_gaussian(
                        mean_modes[m], cov_modes[m]
                    )
                pred_loss[int(k)] = float(w_pred @ loss_modes)

        preds = np.asarray(env.all_expert_predictions(x_t), dtype=float)
        y_t = np.asarray(env.y[t], dtype=float)
        residuals = preds - y_t
        residual_r = np.asarray(residuals[int(r_t)], dtype=float)
        if hasattr(router_local, "_loss_from_residual"):
            loss_r = float(router_local._loss_from_residual(residual_r))
        else:
            loss_r = float(np.sum(residual_r * residual_r))
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
        corr_regime_list.append(pred_corr_regime)
        pred_loss_list.append(pred_loss)
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
        "pred_corr_regime": np.asarray(corr_regime_list, dtype=float),
        "pred_loss": np.asarray(pred_loss_list, dtype=float),
    }


def _collect_ucb_predicted_losses(
    baseline: LinUCB | NeuralUCB,
    env: SyntheticTimeSeriesEnv,
    t_start: int = 1,
    t_end: Optional[int] = None,
) -> dict:
    if t_end is None:
        t_end = env.T - 1
    t_start = max(int(t_start), 1)
    t_end = min(int(t_end), env.T - 1)
    baseline_local = copy.deepcopy(baseline)

    times = []
    choices = []
    pred_loss_list = []

    for t in range(t_start, t_end + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)

        pred_loss = np.full(env.num_experts, np.nan, dtype=float)
        if isinstance(baseline_local, LinUCB):
            phi = baseline_local._get_phi(x_t)
            for k in range(env.num_experts):
                mu_k, _ = baseline_local._theta_and_sigma(int(k), phi)
                pred_loss[int(k)] = float(mu_k)
        elif isinstance(baseline_local, NeuralUCB):
            phi = baseline_local._phi(x_t)
            h, _ = baseline_local._embed(phi)
            for k in range(env.num_experts):
                mu_k, _ = baseline_local._theta_and_sigma(int(k), h)
                pred_loss[int(k)] = float(mu_k)
        else:
            raise ValueError("Unsupported baseline type for UCB diagnostics.")

        pred_loss_list.append(pred_loss)

        if len(available) == 0:
            times.append(int(t))
            choices.append(-1)
            continue

        r_t = baseline_local.select_expert(x_t, available)
        loss_all = env.losses(t)
        baseline_local.update(
            x_t, loss_all, available, selected_expert=int(r_t)
        )

        times.append(int(t))
        choices.append(int(r_t))

    return {
        "times": np.asarray(times, dtype=int),
        "choices": np.asarray(choices, dtype=int),
        "pred_loss": np.asarray(pred_loss_list, dtype=float),
    }


def _collect_l2d_scores(
    baseline: L2D,
    env: SyntheticTimeSeriesEnv,
    t_start: int = 1,
    t_end: Optional[int] = None,
) -> dict:
    if t_end is None:
        t_end = env.T - 1
    t_start = max(int(t_start), 1)
    t_end = min(int(t_end), env.T - 1)
    baseline_local = copy.deepcopy(baseline)
    baseline_local.reset_state()

    times = []
    choices = []
    scores_list = []

    for t in range(t_start, t_end + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)

        phi_x = baseline_local._advance_and_get_phi(x_t)
        scores = baseline_local._scores(phi_x)
        scores_list.append(np.asarray(scores, dtype=float))

        if len(available) == 0:
            times.append(int(t))
            choices.append(-1)
            continue

        available = np.asarray(list(available), dtype=int)
        r_t = int(available[int(np.argmax(scores[available]))])
        loss_all = env.losses(t)
        loss_masked = _mask_feedback_vector_local(
            loss_all, available, r_t, True
        )
        baseline_local.update(x_t, loss_masked, available, selected_expert=r_t)

        times.append(int(t))
        choices.append(int(r_t))

    return {
        "times": np.asarray(times, dtype=int),
        "choices": np.asarray(choices, dtype=int),
        "scores": np.asarray(scores_list, dtype=float),
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
    cov_modes = np.asarray(stats[int(expert)]["cov"], dtype=float)
    w = np.asarray(w, dtype=float)
    if mean_modes.ndim == 1 or (mean_modes.ndim == 2 and mean_modes.shape[1] == 1):
        mean_vec = mean_modes.reshape(-1)
        var_vec = cov_modes.reshape(-1)
        mean = float(w @ mean_vec)
        second = float(w @ (var_vec + mean_vec * mean_vec))
        var = max(second - mean * mean, 0.0)
    else:
        mean_vec = w @ mean_modes
        second = 0.0
        for m in range(w.shape[0]):
            second += float(w[m]) * (
                float(np.trace(cov_modes[m])) + float(mean_modes[m] @ mean_modes[m])
            )
        mean = float(np.linalg.norm(mean_vec))
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
    prune_times = []
    rebirth_times = []

    for t in range(t_start, t_end + 1):
        registry_before = None
        if hasattr(router_local, "registry"):
            registry_before = set(getattr(router_local, "registry", []))
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

        preds = np.asarray(env.all_expert_predictions(x_t), dtype=float)
        y_t = np.asarray(env.y[t], dtype=float)
        residuals = preds - y_t
        residual_r = np.asarray(residuals[int(r_t)], dtype=float)
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
        if registry_before is not None:
            registry_after = set(getattr(router_local, "registry", []))
            if int(target_expert) in registry_before and int(target_expert) not in registry_after:
                prune_times.append(int(t))
            if int(target_expert) not in registry_before and int(target_expert) in registry_after:
                rebirth_times.append(int(t))

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
        true_resid = np.asarray(residuals[int(target_expert)], dtype=float)
        true_loss.append(float(np.sum(true_resid * true_resid)))

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
        "prune_times": np.asarray(prune_times, dtype=int),
        "rebirth_times": np.asarray(rebirth_times, dtype=int),
    }


def _save_fig(
    fig: object,
    out_dir: str,
    name: str,
    save_png: bool,
    save_pdf: bool,
    show: bool,
    keep_axis_titles_pdf: bool = False,
    keep_suptitle_pdf: bool = False,
) -> None:
    def _strip_titles() -> None:
        if not keep_axis_titles_pdf:
            for ax in getattr(fig, "axes", []):
                try:
                    ax.set_title("")
                except Exception:
                    continue
        if not keep_suptitle_pdf:
            suptitle = getattr(fig, "_suptitle", None)
            if suptitle is not None:
                try:
                    suptitle.set_text("")
                    suptitle.set_visible(False)
                except Exception:
                    pass

    if save_png:
        fig.savefig(
            os.path.join(out_dir, f"{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
    if save_pdf:
        _strip_titles()
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
    show_figure_title: bool = True,
) -> None:
    if not choices_map and not rows:
        return
    z = np.asarray(env.z[1: env.T], dtype=int)
    if z.size == 0:
        return
    unique_z = np.unique(z)
    if unique_z.size == 0:
        return
    num_regimes = max(int(unique_z.max()) + 1, 1)
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
    if show_figure_title:
        fig.suptitle("Selection frequency by regime (all baselines)", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.9)
    if debug_lines:
        with open(os.path.join(out_dir, "expert_structure_all_debug.txt"), "w") as f:
            f.write("Selection freq debug\n")
            f.write("\n".join(debug_lines))
            f.write("\n")
    _save_fig(
        fig,
        out_dir,
        "expert_structure_all",
        save_png,
        save_pdf,
        show_plots,
        keep_axis_titles_pdf=True,
    )


def run_tri_cycle_corr_diagnostics(
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    router: FactorizedSLDS,
    router_no_g: Optional[FactorizedSLDS] = None,
    router_no_g_partial: Optional[FactorizedSLDS] = None,
    router_no_g_full: Optional[FactorizedSLDS] = None,
    router_partial: Optional[FactorizedSLDS] = None,
    router_full: Optional[FactorizedSLDS] = None,
    linucb_partial: Optional[LinUCB] = None,
    linucb_full: Optional[LinUCB] = None,
    neuralucb_partial: Optional[NeuralUCB] = None,
    neuralucb_full: Optional[NeuralUCB] = None,
    l2d_baseline: Optional[L2D] = None,
    l2d_sw_baseline: Optional[L2D] = None,
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
    if not _plots_available():
        print("[plot_utils] plotting disabled or matplotlib missing; skip tri-cycle diagnostics.")
        return
    # Allow diagnostics on real datasets by treating all samples as a single
    # regime when no explicit regime sequence is provided.
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
    z_full = getattr(env, "z", None)
    if z_full is None or np.asarray(z_full).size < env.T:
        z_full = np.zeros(env.T, dtype=int)
    z = np.asarray(z_full[t_start : t_end + 1], dtype=int)
    if z.size == 0:
        print("[tri-cycle] Diagnostics skipped: empty regime sequence.")
        return
    unique_z = np.unique(z)
    if unique_z.size == 0:
        print("[tri-cycle] Diagnostics skipped: invalid regime sequence.")
        return
    num_regimes = max(int(unique_z.max()) + 1, 1)
    true_corr = _corr_by_regime(residuals, z, num_regimes)
    losses = residuals * residuals
    true_loss_corr = _corr_by_regime_masked(losses, z, num_regimes)

    diag = _collect_factorized_diagnostics(router, env, t_start, t_end)
    pred_corr = np.asarray(diag["pred_corr"], dtype=float)
    est_corr = _avg_corr_by_regime(pred_corr, z, num_regimes)
    pred_corr_regime = np.asarray(diag.get("pred_corr_regime", []), dtype=float)
    est_corr_cond = (
        _avg_cond_corr_by_regime(pred_corr_regime, z, num_regimes)
        if pred_corr_regime.size
        else [np.zeros((0, 0), dtype=float) for _ in range(num_regimes)]
    )
    diag_no_g = None
    est_corr_no_g = None
    est_corr_no_g_cond = None
    pred_loss_corr = _corr_by_regime_masked(
        np.asarray(diag.get("pred_loss", []), dtype=float), z, num_regimes
    )
    pred_loss_corr_no_g = None
    router_no_g_full_local = router_no_g_full if router_no_g_full is not None else router_no_g
    if router_no_g_full_local is not None:
        diag_no_g = _collect_factorized_diagnostics(
            router_no_g_full_local, env, t_start, t_end
        )
        pred_corr_no_g = np.asarray(diag_no_g["pred_corr"], dtype=float)
        est_corr_no_g = _avg_corr_by_regime(pred_corr_no_g, z, num_regimes)
        pred_corr_no_g_regime = np.asarray(
            diag_no_g.get("pred_corr_regime", []), dtype=float
        )
        est_corr_no_g_cond = (
            _avg_cond_corr_by_regime(pred_corr_no_g_regime, z, num_regimes)
            if pred_corr_no_g_regime.size
            else [np.zeros((0, 0), dtype=float) for _ in range(num_regimes)]
        )
        pred_loss_corr_no_g = _corr_by_regime_masked(
            np.asarray(diag_no_g.get("pred_loss", []), dtype=float), z, num_regimes
        )
    t_grid = diag["times"]

    router_mode = getattr(router, "feedback_mode", None)
    router_full_local = router_full
    router_partial_local = router_partial

    diag_full_v2 = None
    pred_loss_corr_full_v2 = None
    if router_full_local is not None:
        diag_full_v2 = _collect_factorized_diagnostics(
            router_full_local, env, t_start, t_end
        )
    elif router_mode == "full":
        diag_full_v2 = diag
    if diag_full_v2 is not None:
        pred_loss_corr_full_v2 = _corr_by_regime_masked(
            np.asarray(diag_full_v2.get("pred_loss", []), dtype=float),
            z,
            num_regimes,
        )
    pred_loss_corr_partial_v2 = None
    diag_partial_v2 = None
    if router_partial_local is not None:
        diag_partial_v2 = _collect_factorized_diagnostics(
            router_partial_local, env, t_start, t_end
        )
    elif router_mode == "partial":
        diag_partial_v2 = diag
    if diag_partial_v2 is not None:
        pred_loss_corr_partial_v2 = _corr_by_regime_masked(
            np.asarray(diag_partial_v2.get("pred_loss", []), dtype=float),
            z,
            num_regimes,
        )
    pred_loss_corr_no_g_full_v2 = None
    if router_no_g_full_local is not None:
        diag_no_g_full_v2 = (
            diag_no_g if diag_no_g is not None else _collect_factorized_diagnostics(
                router_no_g_full_local, env, t_start, t_end
            )
        )
        pred_loss_corr_no_g_full_v2 = _corr_by_regime_masked(
            np.asarray(diag_no_g_full_v2.get("pred_loss", []), dtype=float),
            z,
            num_regimes,
        )
    pred_loss_corr_no_g_partial_v2 = None
    if router_no_g_partial is not None:
        diag_no_g_partial_v2 = _collect_factorized_diagnostics(
            router_no_g_partial, env, t_start, t_end
        )
        pred_loss_corr_no_g_partial_v2 = _corr_by_regime_masked(
            np.asarray(diag_no_g_partial_v2.get("pred_loss", []), dtype=float),
            z,
            num_regimes,
        )

    linucb_partial_corr = None
    linucb_full_corr = None
    neuralucb_partial_corr = None
    neuralucb_full_corr = None
    l2d_score_corr = None
    l2d_sw_score_corr = None
    if linucb_partial is not None:
        diag_lin_p = _collect_ucb_predicted_losses(
            linucb_partial, env, t_start, t_end
        )
        linucb_partial_corr = _corr_by_regime_masked(
            np.asarray(diag_lin_p["pred_loss"], dtype=float), z, num_regimes
        )
    if linucb_full is not None:
        diag_lin_f = _collect_ucb_predicted_losses(
            linucb_full, env, t_start, t_end
        )
        linucb_full_corr = _corr_by_regime_masked(
            np.asarray(diag_lin_f["pred_loss"], dtype=float), z, num_regimes
        )
    if neuralucb_partial is not None:
        diag_nu_p = _collect_ucb_predicted_losses(
            neuralucb_partial, env, t_start, t_end
        )
        neuralucb_partial_corr = _corr_by_regime_masked(
            np.asarray(diag_nu_p["pred_loss"], dtype=float), z, num_regimes
        )
    if neuralucb_full is not None:
        diag_nu_f = _collect_ucb_predicted_losses(
            neuralucb_full, env, t_start, t_end
        )
        neuralucb_full_corr = _corr_by_regime_masked(
            np.asarray(diag_nu_f["pred_loss"], dtype=float), z, num_regimes
        )
    if l2d_baseline is not None:
        diag_l2d = _collect_l2d_scores(l2d_baseline, env, t_start, t_end)
        l2d_score_corr = _corr_by_regime_masked(
            np.asarray(diag_l2d["scores"], dtype=float), z, num_regimes
        )
    if l2d_sw_baseline is not None:
        diag_l2d_sw = _collect_l2d_scores(l2d_sw_baseline, env, t_start, t_end)
        l2d_sw_score_corr = _corr_by_regime_masked(
            np.asarray(diag_l2d_sw["scores"], dtype=float), z, num_regimes
        )

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

    corr_mae_cond = []
    for m in range(num_regimes):
        diff = est_corr_cond[m] - true_corr[m]
        if diff.size == 0:
            corr_mae_cond.append(float("nan"))
            continue
        mask = ~np.eye(diff.shape[0], dtype=bool)
        corr_mae_cond.append(float(np.nanmean(np.abs(diff[mask]))))

    def _corr_mae_list(est_list: list[np.ndarray]) -> list[float]:
        out = []
        for m in range(num_regimes):
            diff = est_list[m] - true_loss_corr[m]
            if diff.size == 0:
                out.append(float("nan"))
                continue
            mask = ~np.eye(diff.shape[0], dtype=bool)
            out.append(float(np.nanmean(np.abs(diff[mask]))))
        return out

    pred_loss_corr_mae = _corr_mae_list(pred_loss_corr)
    pred_loss_corr_mae_no_g = (
        _corr_mae_list(pred_loss_corr_no_g)
        if pred_loss_corr_no_g is not None
        else None
    )
    linucb_partial_corr_mae = (
        _corr_mae_list(linucb_partial_corr)
        if linucb_partial_corr is not None
        else None
    )
    linucb_full_corr_mae = (
        _corr_mae_list(linucb_full_corr) if linucb_full_corr is not None else None
    )
    neuralucb_partial_corr_mae = (
        _corr_mae_list(neuralucb_partial_corr)
        if neuralucb_partial_corr is not None
        else None
    )
    neuralucb_full_corr_mae = (
        _corr_mae_list(neuralucb_full_corr)
        if neuralucb_full_corr is not None
        else None
    )

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
    tick_step = max(1, int(np.ceil(n_experts / 10)))
    tick_locs = np.arange(0, n_experts, tick_step)

    def _normalize_corr_matrix(corr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if corr is None:
            return None
        corr = np.asarray(corr, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            return None
        if corr.shape[0] == n_experts:
            return corr
        if corr.shape[0] > n_experts:
            return corr[:n_experts, :n_experts]
        out = np.full((n_experts, n_experts), np.nan, dtype=float)
        out[: corr.shape[0], : corr.shape[1]] = corr
        return out

    summary = {
        "label": label,
        "t_start": int(t_start),
        "t_end": int(t_end),
        "regime_accuracy": regime_accuracy,
        "switch_delay_mean": delay_mean,
        "switch_delay_median": delay_median,
        "corr_mae_by_regime": corr_mae,
        "corr_mae_cond_by_regime": corr_mae_cond,
        "pred_loss_corr_mae_by_regime": pred_loss_corr_mae,
        "g_recovery": g_metrics,
        "avg_cost": float(np.nanmean(diag["costs"])) if diag["costs"].size else float("nan"),
    }
    if diag_no_g is not None:
        summary["avg_cost_no_g"] = float(np.nanmean(diag_no_g["costs"]))
        summary["avg_cost_gain"] = float(
            summary["avg_cost_no_g"] - summary["avg_cost"]
        )
    if pred_loss_corr_mae_no_g is not None:
        summary["pred_loss_corr_mae_no_g_by_regime"] = pred_loss_corr_mae_no_g
    if linucb_partial_corr_mae is not None:
        summary["pred_loss_corr_mae_linucb_partial_by_regime"] = (
            linucb_partial_corr_mae
        )
    if linucb_full_corr_mae is not None:
        summary["pred_loss_corr_mae_linucb_full_by_regime"] = linucb_full_corr_mae
    if neuralucb_partial_corr_mae is not None:
        summary["pred_loss_corr_mae_neuralucb_partial_by_regime"] = (
            neuralucb_partial_corr_mae
        )
    if neuralucb_full_corr_mae is not None:
        summary["pred_loss_corr_mae_neuralucb_full_by_regime"] = (
            neuralucb_full_corr_mae
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
        for m, err in enumerate(corr_mae_cond):
            f.write(f"Corr MAE (cond) regime {m}: {err:.4f}\n")
        for m, err in enumerate(pred_loss_corr_mae):
            f.write(f"Pred-loss corr MAE regime {m}: {err:.4f}\n")
        if pred_loss_corr_mae_no_g is not None:
            for m, err in enumerate(pred_loss_corr_mae_no_g):
                f.write(f"Pred-loss corr MAE no-g regime {m}: {err:.4f}\n")
        if linucb_partial_corr_mae is not None:
            for m, err in enumerate(linucb_partial_corr_mae):
                f.write(f"Pred-loss corr MAE LinUCB P regime {m}: {err:.4f}\n")
        if linucb_full_corr_mae is not None:
            for m, err in enumerate(linucb_full_corr_mae):
                f.write(f"Pred-loss corr MAE LinUCB F regime {m}: {err:.4f}\n")
        if neuralucb_partial_corr_mae is not None:
            for m, err in enumerate(neuralucb_partial_corr_mae):
                f.write(f"Pred-loss corr MAE NeuralUCB P regime {m}: {err:.4f}\n")
        if neuralucb_full_corr_mae is not None:
            for m, err in enumerate(neuralucb_full_corr_mae):
                f.write(f"Pred-loss corr MAE NeuralUCB F regime {m}: {err:.4f}\n")
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

    # Regime-conditional correlation heatmaps (model-only, no regime mixing)
    n_cols_cond = 3 if est_corr_no_g_cond is not None else 2
    fig, axes = plt.subplots(
        num_regimes,
        n_cols_cond,
        figsize=(3.6 * n_cols_cond, 2.9 * max(num_regimes, 1)),
        sharex=True,
        sharey=True,
    )
    if num_regimes == 1:
        axes = np.array([axes])
    for m in range(num_regimes):
        ax_true = axes[m, 0]
        ax_est = axes[m, 1]
        im_true = ax_true.imshow(true_corr[m], vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax_est.imshow(est_corr_cond[m], vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax_true.set_title(f"Regime {m}: true")
        ax_est.set_title(f"Regime {m}: estimated (cond)")
        if est_corr_no_g_cond is not None:
            ax_ng = axes[m, 2]
            ax_ng.imshow(est_corr_no_g_cond[m], vmin=-1.0, vmax=1.0, cmap="coolwarm")
            ax_ng.set_title(f"Regime {m}: no-g (cond)")
            ax_ng.set_yticks(np.arange(n_experts))
            ax_ng.set_xticks(np.arange(n_experts))
        ax_true.set_yticks(np.arange(n_experts))
        ax_est.set_yticks(np.arange(n_experts))
        ax_true.set_xticks(np.arange(n_experts))
        ax_est.set_xticks(np.arange(n_experts))
        ax_true.set_ylabel("Expert")
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im_true, cax=cax, label="Corr")
    fig.suptitle("Regime-conditional residual correlation", y=0.98)
    fig.subplots_adjust(
        top=0.86, bottom=0.08, left=0.08, right=0.88, wspace=0.25, hspace=0.3
    )
    _save_fig(fig, out_dir, "corr_cond", save_png, save_pdf, show_plots)

    # Method-agnostic correlation heatmaps from predicted losses
    corr_entries: list[tuple[str, list[np.ndarray]]] = [
        ("True (loss corr)", true_loss_corr),
        (f"{label} (pred loss)", pred_loss_corr),
    ]
    if pred_loss_corr_no_g is not None:
        corr_entries.append(("No-g (pred loss)", pred_loss_corr_no_g))
    if linucb_partial_corr is not None:
        corr_entries.append(("LinUCB P", linucb_partial_corr))
    if linucb_full_corr is not None:
        corr_entries.append(("LinUCB F", linucb_full_corr))
    if neuralucb_partial_corr is not None:
        corr_entries.append(("NeuralUCB P", neuralucb_partial_corr))
    if neuralucb_full_corr is not None:
        corr_entries.append(("NeuralUCB F", neuralucb_full_corr))

    n_cols_new = len(corr_entries)
    fig, axes = plt.subplots(
        num_regimes,
        n_cols_new,
        figsize=(3.4 * n_cols_new, 2.9 * max(num_regimes, 1)),
        sharex=True,
        sharey=True,
    )
    if num_regimes == 1:
        axes = np.array([axes])
    last_im = None
    for m in range(num_regimes):
        for c_idx, (title, corr_list) in enumerate(corr_entries):
            ax = axes[m, c_idx]
            corr_mat = corr_list[m]
            im = ax.imshow(corr_mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            last_im = im
            ax.set_title(f"Regime {m}: {title}", fontsize=9, pad=6)
            ax.set_yticks(np.arange(n_experts))
            ax.set_xticks(np.arange(n_experts))
            if c_idx == 0:
                ax.set_ylabel("Expert")
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(last_im, cax=cax, label="Corr")
    fig.suptitle("Predicted-loss correlation by regime", y=0.98)
    fig.subplots_adjust(
        top=0.86, bottom=0.08, left=0.08, right=0.9, wspace=0.25, hspace=0.3
    )
    _save_fig(fig, out_dir, "corr_new", save_png, save_pdf, show_plots)

    # Regime-0 summary: partial vs full baselines
    reg_idx = 0
    if num_regimes > 0 and reg_idx < num_regimes:
        if num_regimes > 1:
            reg_counts = np.bincount(z, minlength=num_regimes)
            reg_idx = int(np.argmax(reg_counts))
        has_partial_v2 = (
            pred_loss_corr_partial_v2 is not None
            or pred_loss_corr_no_g_partial_v2 is not None
            or linucb_partial_corr is not None
            or neuralucb_partial_corr is not None
        )
        has_full_v2 = (
            pred_loss_corr_full_v2 is not None
            or pred_loss_corr_no_g_full_v2 is not None
            or linucb_full_corr is not None
            or neuralucb_full_corr is not None
            or l2d_score_corr is not None
            or l2d_sw_score_corr is not None
        )

        rows_v2: list[list[tuple[str, np.ndarray]]] = []
        row_labels_v2: list[str] = []
        def _append_v2_entry(
            entries: list[tuple[str, np.ndarray]],
            title: str,
            corr_list: Optional[list[np.ndarray]],
        ) -> None:
            if corr_list is None or reg_idx >= len(corr_list):
                return
            corr_mat = _normalize_corr_matrix(corr_list[reg_idx])
            if corr_mat is None or corr_mat.size == 0:
                return
            entries.append((title, corr_mat))

        if has_partial_v2:
            partial_entries_v2: list[tuple[str, np.ndarray]] = []
            _append_v2_entry(partial_entries_v2, "Ground truth", true_loss_corr)
            _append_v2_entry(partial_entries_v2, "L2D-SLDS", pred_loss_corr_partial_v2)
            _append_v2_entry(
                partial_entries_v2,
                "L2D-SLDS w/o $g_t$",
                pred_loss_corr_no_g_partial_v2,
            )
            _append_v2_entry(partial_entries_v2, "LinUCB", linucb_partial_corr)
            _append_v2_entry(partial_entries_v2, "NeuralUCB", neuralucb_partial_corr)
            if partial_entries_v2:
                rows_v2.append(partial_entries_v2)
                row_labels_v2.append("Partial baselines")

        if has_full_v2:
            full_entries_v2: list[tuple[str, np.ndarray]] = []
            _append_v2_entry(full_entries_v2, "Ground truth", true_loss_corr)
            _append_v2_entry(full_entries_v2, "L2D-SLDS", pred_loss_corr_full_v2)
            _append_v2_entry(
                full_entries_v2,
                "L2D-SLDS w/o $g_t$",
                pred_loss_corr_no_g_full_v2,
            )
            _append_v2_entry(full_entries_v2, "LinUCB", linucb_full_corr)
            _append_v2_entry(full_entries_v2, "NeuralUCB", neuralucb_full_corr)
            _append_v2_entry(full_entries_v2, "L2D", l2d_score_corr)
            _append_v2_entry(
                full_entries_v2,
                "L2D_SW (W=500)",
                l2d_sw_score_corr,
            )
            if full_entries_v2:
                rows_v2.append(full_entries_v2)
                row_labels_v2.append("Full baselines")

        if rows_v2:
            n_rows_v2 = len(rows_v2)
            n_cols_v2 = max(len(entries) for entries in rows_v2)
            fig, axes = plt.subplots(
                n_rows_v2,
                n_cols_v2,
                figsize=(3.4 * n_cols_v2, 2.7 * n_rows_v2),
                sharex=True,
                sharey=True,
            )
            if n_rows_v2 == 1:
                axes = np.array([axes])
            if n_cols_v2 == 1:
                axes = axes.reshape(n_rows_v2, 1)
            last_im = None
            for r_idx, entries in enumerate(rows_v2):
                for c_idx in range(n_cols_v2):
                    ax = axes[r_idx, c_idx]
                    if c_idx >= len(entries):
                        ax.axis("off")
                        continue
                    title, corr_mat = entries[c_idx]
                    im = ax.imshow(corr_mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
                    last_im = im
                    ax.set_title(title, fontsize=9, pad=6)
                    ax.set_xticks(tick_locs)
                    ax.set_yticks(tick_locs)
                    if c_idx == 0:
                        row_label = row_labels_v2[r_idx]
                        if str(row_label).strip().lower() == "partial baselines":
                            row_label = ""
                        ax.set_ylabel(row_label)

            if last_im is not None:
                cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
                fig.colorbar(last_im, cax=cax, label="Corr")
            fig.suptitle(f"Regime {reg_idx}: correlation between experts", y=0.98)
            try:
                fig.supxlabel("Experts")
                fig.supylabel("Experts")
            except Exception:
                pass
            fig.subplots_adjust(
                top=0.86, bottom=0.08, left=0.08, right=0.9, wspace=0.25, hspace=0.3
            )
            _save_fig(
                fig,
                out_dir,
                "corr_new_v2",
                save_png,
                save_pdf,
                show_plots,
                keep_axis_titles_pdf=True,
            )

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
            diff = np.abs(series["post_loss"] - series["true_loss"])
            ax_loss.plot(
                series["times"],
                diff,
                label=f"Expert {target} (post - true) L2D SLDS",
                color="tab:blue",
            )
            if series_no_g is not None:
                diff_no_g = np.abs(
                    series_no_g["post_loss"] - series_no_g["true_loss"]
                )
                ax_loss.plot(
                    series_no_g["times"],
                    diff_no_g,
                    label=f"Expert {target} (post - true) L2D SLDS w/t $g_t$",
                    color="tab:orange",
                    linestyle="--",
                )
            if show_truth:
                ax_loss.axhline(
                    0.0,
                    color="black",
                    alpha=0.3,
                    linewidth=1.0,
                    linestyle=":",
                    label="True (0)",
                )
            _shade_unavailability(ax_loss, series["times"], series["avail_target"])
            ax_loss.set_ylabel("Absolute loss error |post - true|")
            ax_loss.set_title(f"Expert {target} absolute loss error |post - true|")
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

            def _add_prune_lines(
                ax: Axes,
                times_arr: np.ndarray,
                color: str,
                label: Optional[str],
            ) -> None:
                if times_arr is None or len(times_arr) == 0:
                    return
                for idx, t_prune in enumerate(times_arr):
                    ax.axvline(
                        int(t_prune),
                        color=color,
                        alpha=0.25,
                        linewidth=1.0,
                        label=label if idx == 0 and label else "_nolegend_",
                    )

            _add_prune_lines(
                ax_loss,
                series.get("prune_times", np.asarray([], dtype=int)),
                "tab:blue",
                "Prune (L2D SLDS)",
            )
            if series_no_g is not None:
                _add_prune_lines(
                    ax_loss,
                    series_no_g.get("prune_times", np.asarray([], dtype=int)),
                    "tab:orange",
                    "Prune (L2D SLDS w/t $g_t$)",
                )
            _add_prune_lines(
                ax_delta,
                series.get("prune_times", np.asarray([], dtype=int)),
                "tab:blue",
                None,
            )
            if series_no_g is not None:
                _add_prune_lines(
                    ax_delta,
                    series_no_g.get("prune_times", np.asarray([], dtype=int)),
                    "tab:orange",
                    None,
                )

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

            if router_partial is not None:
                series_p = _collect_transfer_probe(
                    router_partial, env, target, t_start, t_end
                )
                series_p_no_g = None
                if compare_no_g and router_no_g_partial is not None:
                    series_p_no_g = _collect_transfer_probe(
                        router_no_g_partial, env, target, t_start, t_end
                    )

                fig, (ax_loss, ax_delta) = plt.subplots(
                    2, 1, sharex=True, figsize=(9.0, 5.6)
                )
                diff = np.abs(series_p["post_loss"] - series_p["true_loss"])
                ax_loss.plot(
                    series_p["times"],
                    diff,
                    label=f"Expert {target} (post - true) L2D SLDS (partial)",
                    color="tab:blue",
                )
                if series_p_no_g is not None:
                    diff_no_g = np.abs(
                        series_p_no_g["post_loss"] - series_p_no_g["true_loss"]
                    )
                    ax_loss.plot(
                        series_p_no_g["times"],
                        diff_no_g,
                        label=f"Expert {target} (post - true) L2D SLDS w/t $g_t$ (partial)",
                        color="tab:orange",
                        linestyle="--",
                    )
                if show_truth:
                    ax_loss.axhline(
                        0.0,
                        color="black",
                        alpha=0.3,
                        linewidth=1.0,
                        linestyle=":",
                        label="True (0)",
                    )
                _shade_unavailability(ax_loss, series_p["times"], series_p["avail_target"])
                ax_loss.set_ylabel("Absolute loss error |post - true|")
                ax_loss.set_title(f"Expert {target} absolute loss error |post - true|")
                ax_loss.grid(True, alpha=0.2)

                delta = np.abs(series_p["post_loss"] - series_p["pre_loss"])
                ax_delta.plot(
                    series_p["times"],
                    delta,
                    label="L2D SLDS (partial)",
                    color="tab:green",
                )
                if series_p_no_g is not None:
                    delta_no_g = np.abs(
                        series_p_no_g["post_loss"] - series_p_no_g["pre_loss"]
                    )
                    ax_delta.plot(
                        series_p_no_g["times"],
                        delta_no_g,
                        label="L2D SLDS w/t $g_t$ (partial)",
                        color="tab:red",
                        linestyle="--",
                    )
                _shade_unavailability(ax_delta, series_p["times"], series_p["avail_target"])
                ax_delta.set_xlabel("Time $t$ (shaded = unavailable)")
                ax_delta.set_ylabel("Update magnitude")
                ax_delta.grid(True, alpha=0.2)

                _add_prune_lines(
                    ax_loss,
                    series_p.get("prune_times", np.asarray([], dtype=int)),
                    "tab:blue",
                    "Prune (L2D SLDS partial)",
                )
                if series_p_no_g is not None:
                    _add_prune_lines(
                        ax_loss,
                        series_p_no_g.get("prune_times", np.asarray([], dtype=int)),
                        "tab:orange",
                        "Prune (L2D SLDS w/t $g_t$ partial)",
                    )
                _add_prune_lines(
                    ax_delta,
                    series_p.get("prune_times", np.asarray([], dtype=int)),
                    "tab:blue",
                    None,
                )
                if series_p_no_g is not None:
                    _add_prune_lines(
                        ax_delta,
                        series_p_no_g.get("prune_times", np.asarray([], dtype=int)),
                        "tab:orange",
                        None,
                    )

                if source is not None:
                    sel_mask = series_p["selected"] == source
                    if np.any(sel_mask):
                        y_min, y_max = ax_loss.get_ylim()
                        y_tick = y_min + 0.03 * (y_max - y_min)
                        ax_loss.vlines(
                            series_p["times"][sel_mask],
                            y_min,
                            y_tick,
                            color="tab:purple",
                            alpha=0.4,
                            linewidth=0.8,
                            label=f"Selected {source}",
                        )
                obs_mask = series_p["selected"] == target
                if np.any(obs_mask):
                    y_min, y_max = ax_loss.get_ylim()
                    y_tick = y_min + 0.06 * (y_max - y_min)
                    ax_loss.vlines(
                        series_p["times"][obs_mask],
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
                fig.subplots_adjust(
                    top=0.9, bottom=0.1, left=0.08, right=0.8, hspace=0.35
                )
                _save_fig(
                    fig, out_dir, "transfer_probe_partial", save_png, save_pdf, show_plots
                )

                def _slice_series(series_in: dict, t_lo: int, t_hi: int) -> dict:
                    times_arr = np.asarray(series_in["times"], dtype=int)
                    mask = (times_arr >= t_lo) & (times_arr <= t_hi)
                    series_out = {}
                    for key, val in series_in.items():
                        if isinstance(val, np.ndarray) and val.shape == times_arr.shape:
                            series_out[key] = val[mask]
                        elif isinstance(val, np.ndarray) and key in ("prune_times", "rebirth_times"):
                            series_out[key] = val[(val >= t_lo) & (val <= t_hi)]
                        else:
                            series_out[key] = val
                    return series_out

                t_win_start, t_win_end = 1900, 2600
                series_p_win = _slice_series(series_p, t_win_start, t_win_end)
                series_p_no_g_win = (
                    _slice_series(series_p_no_g, t_win_start, t_win_end)
                    if series_p_no_g is not None
                    else None
                )
                if series_p_win["times"].size > 0:
                    fig, (ax_loss, ax_delta) = plt.subplots(
                        2, 1, sharex=True, figsize=(9.0, 5.6)
                    )
                    diff = np.abs(series_p_win["post_loss"] - series_p_win["true_loss"])
                    ax_loss.plot(
                        series_p_win["times"],
                        diff,
                        label=f"Expert {target} (post - true) L2D SLDS (partial)",
                        color="tab:blue",
                    )
                    if series_p_no_g_win is not None:
                        diff_no_g = np.abs(
                            series_p_no_g_win["post_loss"] - series_p_no_g_win["true_loss"]
                        )
                        ax_loss.plot(
                            series_p_no_g_win["times"],
                            diff_no_g,
                            label=f"Expert {target} (post - true) L2D SLDS w/t $g_t$ (partial)",
                            color="tab:orange",
                            linestyle="--",
                        )
                    if show_truth:
                        ax_loss.axhline(
                            0.0,
                            color="black",
                            alpha=0.3,
                            linewidth=1.0,
                            linestyle=":",
                            label="True (0)",
                        )
                    _shade_unavailability(
                        ax_loss, series_p_win["times"], series_p_win["avail_target"]
                    )
                    ax_loss.set_ylabel("Absolute loss error |post - true|")
                    ax_loss.set_title(
                        f"Expert {target} absolute loss error |post - true|"
                    )
                    ax_loss.grid(True, alpha=0.2)

                    delta = np.abs(series_p_win["post_loss"] - series_p_win["pre_loss"])
                    ax_delta.plot(
                        series_p_win["times"],
                        delta,
                        label="L2D SLDS (partial)",
                        color="tab:green",
                    )
                    if series_p_no_g_win is not None:
                        delta_no_g = np.abs(
                            series_p_no_g_win["post_loss"]
                            - series_p_no_g_win["pre_loss"]
                        )
                        ax_delta.plot(
                            series_p_no_g_win["times"],
                            delta_no_g,
                            label="L2D SLDS w/t $g_t$ (partial)",
                            color="tab:red",
                            linestyle="--",
                        )
                    _shade_unavailability(
                        ax_delta, series_p_win["times"], series_p_win["avail_target"]
                    )
                    ax_delta.set_xlabel("Time $t$ (shaded = unavailable)")
                    ax_delta.set_ylabel("Update magnitude")
                    ax_delta.grid(True, alpha=0.2)

                    _add_prune_lines(
                        ax_loss,
                        series_p_win.get("prune_times", np.asarray([], dtype=int)),
                        "tab:blue",
                        "Prune (L2D SLDS partial)",
                    )
                    if series_p_no_g_win is not None:
                        _add_prune_lines(
                            ax_loss,
                            series_p_no_g_win.get("prune_times", np.asarray([], dtype=int)),
                            "tab:orange",
                            "Prune (L2D SLDS w/t $g_t$ partial)",
                        )
                    _add_prune_lines(
                        ax_delta,
                        series_p_win.get("prune_times", np.asarray([], dtype=int)),
                        "tab:blue",
                        None,
                    )
                    if series_p_no_g_win is not None:
                        _add_prune_lines(
                            ax_delta,
                            series_p_no_g_win.get("prune_times", np.asarray([], dtype=int)),
                            "tab:orange",
                            None,
                        )

                    if source is not None:
                        sel_mask = series_p_win["selected"] == source
                        if np.any(sel_mask):
                            y_min, y_max = ax_loss.get_ylim()
                            y_tick = y_min + 0.03 * (y_max - y_min)
                            ax_loss.vlines(
                                series_p_win["times"][sel_mask],
                                y_min,
                                y_tick,
                                color="tab:purple",
                                alpha=0.4,
                                linewidth=0.8,
                                label=f"Selected {source}",
                            )
                    obs_mask = series_p_win["selected"] == target
                    if np.any(obs_mask):
                        y_min, y_max = ax_loss.get_ylim()
                        y_tick = y_min + 0.06 * (y_max - y_min)
                        ax_loss.vlines(
                            series_p_win["times"][obs_mask],
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
                    fig.subplots_adjust(
                        top=0.9, bottom=0.1, left=0.08, right=0.8, hspace=0.35
                    )
                    _save_fig(
                        fig,
                        out_dir,
                        "transfer_probe_window_partial",
                        save_png,
                        save_pdf,
                        show_plots,
                    )
