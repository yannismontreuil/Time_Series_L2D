import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Sequence, Tuple, Optional

from router_model import SLDSIMMRouter
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import LearningToDeferBaseline


def warm_start_router_to_time(
    router: SLDSIMMRouter,
    env: SyntheticTimeSeriesEnv,
    t0: int,
) -> None:
    """
    Reset router beliefs and run it on the environment up to time t0,
    using its configured feedback mode ("partial" or "full").
    """
    router.reset_beliefs()
    T = env.T
    t_max = min(max(int(t0), 0), T - 1)
    for t in range(1, t_max + 1):
        x_t = env.get_context(t)
        available_t = env.get_available_experts(t)
        r_t, cache = router.select_expert(x_t, available_t)
        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=None,
                available_experts=available_t,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=loss_all,
                available_experts=available_t,
                cache=cache,
            )


def eval_schedule_on_env(
    env: SyntheticTimeSeriesEnv,
    beta: np.ndarray,
    times: np.ndarray,
    schedule: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a fixed expert schedule over specified times, compute:
      - predictions y_hat_j(x_t) under the true environment,
      - incurred costs ℓ_{j,t} + β_j using true losses.
    """
    schedule = [int(j) for j in schedule]
    H = min(len(schedule), times.shape[0])
    preds = np.zeros(H, dtype=float)
    costs = np.zeros(H, dtype=float)
    for idx_h in range(H):
        t = int(times[idx_h])
        j = int(schedule[idx_h])
        x_t = env.get_context(t)
        preds[idx_h] = env.expert_predict(j, x_t)
        loss_all = env.losses(t)
        costs[idx_h] = float(loss_all[j] + beta[j])
    return preds, costs


def compute_oracle_and_baseline_schedules(
    env: SyntheticTimeSeriesEnv,
    beta: np.ndarray,
    t0: int,
    H: int,
) -> Tuple[np.ndarray, int, List[np.ndarray], List[int], List[int], int]:
    """
    From time t0, compute:
      - effective horizon H_eff (clipped to available data),
      - times = [t0+1,...,t0+H_eff],
      - availability sets per horizon step,
      - oracle schedule (per-step best expert given true losses),
      - constant-expert baseline schedule over experts always available
        on the horizon, and its expert index j_baseline.
    """
    T = env.T
    H_eff = min(max(int(H), 0), T - 1 - int(t0))
    if H_eff <= 0:
        return np.array([], dtype=int), 0, [], [], [], -1

    times = np.arange(int(t0) + 1, int(t0) + 1 + H_eff, dtype=int)

    # Availability sets for each future step t0+1,...,t0+H_eff
    avail_per_h: List[np.ndarray] = [
        env.get_available_experts(int(t)).copy() for t in times
    ]

    # Oracle "truth" schedule: best expert at each step given true losses.
    sched_oracle: List[int] = []
    for t, avail_t in zip(times, avail_per_h):
        loss_all = env.losses(int(t))
        costs_t = loss_all[avail_t] + beta[avail_t]
        idx_best = int(np.argmin(costs_t))
        sched_oracle.append(int(avail_t[idx_best]))

    # Baseline: best constant expert over this horizon among always-available experts.
    avail_window = env.availability[int(t0) + 1 : int(t0) + 1 + H_eff, :]
    always_avail_mask = avail_window.all(axis=0)
    candidate_experts = np.where(always_avail_mask)[0]
    if candidate_experts.size == 0:
        candidate_experts = np.arange(env.num_experts)

    avg_cost_per_expert: List[float] = []
    for j in candidate_experts:
        costs_j = []
        for t in times:
            loss_all = env.losses(int(t))
            costs_j.append(float(loss_all[j] + beta[j]))
        avg_cost_per_expert.append(float(np.mean(costs_j)))
    j_baseline = int(candidate_experts[int(np.argmin(avg_cost_per_expert))])
    sched_baseline = [j_baseline] * H_eff

    return times, H_eff, avail_per_h, sched_oracle, sched_baseline, j_baseline


def warm_start_l2d_to_time(
    baseline: LearningToDeferBaseline,
    env: SyntheticTimeSeriesEnv,
    t0: int,
) -> None:
    """
    Train the learning-to-defer baseline on the environment up to time t0,
    using its own selection rule and full-feedback updates.
    """
    T = env.T
    t_max = min(max(int(t0), 0), T - 1)
    for t in range(1, t_max + 1):
        x_t = env.get_context(t)
        available_t = env.get_available_experts(t)
        r_t = baseline.select_expert(x_t, available_t)
        loss_all = env.losses(t)
        baseline.update(x_t, loss_all, available_t)


def evaluate_horizon_planning(
    env: SyntheticTimeSeriesEnv,
    router_partial: SLDSIMMRouter,
    router_full: SLDSIMMRouter,
    beta: np.ndarray,
    t0: int,
    H: int,
    experts_predict: Sequence[Callable[[np.ndarray], float]],
    context_update: Callable[[np.ndarray, float], np.ndarray],
    l2d_baseline: Optional[LearningToDeferBaseline] = None,
) -> None:
    """
    Compare horizon-H planning from time t0 for:
      - oracle (truth) per-step best expert,
      - constant-expert baseline,
      - partial-feedback router horizon planner,
      - full-feedback router horizon planner.

    Produces summary printouts and plots of forecasts vs truth and
    per-step costs over the horizon, plus expert scheduling.
    """
    # Warm-start routers to time t0 under their feedback modes.
    warm_start_router_to_time(router_partial, env, t0)
    warm_start_router_to_time(router_full, env, t0)

    # Warm-start L2D baseline to time t0 if provided.
    if l2d_baseline is not None:
        warm_start_l2d_to_time(l2d_baseline, env, t0)

    # Compute oracle and baseline schedules and effective horizon.
    times, H_eff, avail_per_h, sched_oracle, sched_baseline, j_baseline = (
        compute_oracle_and_baseline_schedules(env, beta, t0, H)
    )
    if H_eff <= 0:
        print("Horizon too short after clipping; skipping horizon planning evaluation.")
        return

    # Horizon-H planning schedules from current beliefs
    t0_int = int(t0)
    x_now = env.get_context(t0_int)
    sched_partial, _, _ = router_partial.plan_horizon_schedule(
        x_t=x_now,
        H=H_eff,
        experts_predict=experts_predict,
        context_update=context_update,
        available_experts_per_h=[a.tolist() for a in avail_per_h],
    )
    sched_full, _, _ = router_full.plan_horizon_schedule(
        x_t=x_now,
        H=H_eff,
        experts_predict=experts_predict,
        context_update=context_update,
        available_experts_per_h=[a.tolist() for a in avail_per_h],
    )

    # Evaluate all schedules on the true environment.
    preds_oracle, cost_oracle = eval_schedule_on_env(env, beta, times, sched_oracle)
    preds_baseline, cost_baseline = eval_schedule_on_env(env, beta, times, sched_baseline)
    preds_partial_plan, cost_partial_plan = eval_schedule_on_env(
        env, beta, times, sched_partial
    )
    preds_full_plan, cost_full_plan = eval_schedule_on_env(
        env, beta, times, sched_full
    )

    # Constant-expert baselines for all experts (theoretical, ignore availability)
    const_preds = []
    const_costs = []
    for j in range(env.num_experts):
        sched_j = [j] * H_eff
        preds_j, costs_j = eval_schedule_on_env(env, beta, times, sched_j)
        const_preds.append(preds_j)
        const_costs.append(costs_j)

    # L2D baseline schedule and evaluation (if provided)
    if l2d_baseline is not None:
        sched_l2d: List[int] = []
        for t in times:
            x_t = env.get_context(int(t))
            available_t = env.get_available_experts(int(t))
            r_t = l2d_baseline.select_expert(x_t, available_t)
            sched_l2d.append(int(r_t))
        preds_l2d_plan, cost_l2d_plan = eval_schedule_on_env(
            env, beta, times, sched_l2d
        )
    else:
        sched_l2d = []
        preds_l2d_plan = np.array([], dtype=float)
        cost_l2d_plan = np.array([], dtype=float)

    y_window = env.y[t0_int + 1 : t0_int + 1 + H_eff]

    # Print horizon average costs
    print(f"\n=== Horizon-{H_eff} average costs from t={t0_int} ===")
    print(f"Oracle (truth):                {cost_oracle.mean():.4f}")
    print(f"Best constant expert (exp {j_baseline}): {cost_baseline.mean():.4f}")
    for j in range(env.num_experts):
        print(f"Const expert {j}:             {const_costs[j].mean():.4f}")
    if l2d_baseline is not None and cost_l2d_plan.size > 0:
        print(f"L2D baseline (horizon):       {cost_l2d_plan.mean():.4f}")
    print(f"H-plan (partial router):       {cost_partial_plan.mean():.4f}")
    print(f"H-plan (full router):          {cost_full_plan.mean():.4f}")

    # Plot forecasts vs truth and zoomed horizon window + scheduling
    fig_h, (ax_h_full, ax_h_zoom, ax_h_sched) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=False
    )

    # --- Top subplot: full history up to end of horizon ---
    all_times = np.arange(0, t0_int + 1 + H_eff, dtype=int)
    all_y = env.y[: t0_int + 1 + H_eff]
    ax_h_full.plot(all_times, all_y, label="True $y_t$", color="black", linewidth=2)

    # Anchor all forecast curves at t0 with the ground-truth value y_{t0}
    y_t0 = env.y[t0_int]
    times_forecast = np.concatenate(([t0_int], times))
    preds_oracle_plot = np.concatenate(([y_t0], preds_oracle))
    preds_baseline_plot = np.concatenate(([y_t0], preds_baseline))
    preds_partial_plot = np.concatenate(([y_t0], preds_partial_plan))
    preds_full_plot = np.concatenate(([y_t0], preds_full_plan))

    ax_h_full.plot(
        times_forecast,
        preds_oracle_plot,
        label="Oracle (truth)",
        linestyle="--",
        color="tab:gray",
    )
    # Do not plot a dedicated best-constant curve; individual const experts are shown below.
    ax_h_full.plot(times_forecast, preds_partial_plot, label="H-plan partial", alpha=0.8)
    ax_h_full.plot(times_forecast, preds_full_plot, label="H-plan full", alpha=0.8)
    # Plot constant-expert baselines (line + '*' markers for clarity)
    for j in range(env.num_experts):
        preds_j_plot = np.concatenate(([y_t0], const_preds[j]))
        ax_h_full.plot(
            times_forecast,
            preds_j_plot,
            linestyle=":",
            marker="*",
            markersize=6,
            alpha=0.7,
            label=f"Const expert {j}",
        )
    # Plot L2D baseline forecasts if available
    if l2d_baseline is not None and preds_l2d_plan.size > 0:
        preds_l2d_plot = np.concatenate(([y_t0], preds_l2d_plan))
        ax_h_full.plot(
            times_forecast,
            preds_l2d_plot,
            label="L2D baseline (horizon)",
            alpha=0.9,
        )
    ax_h_full.axvline(t0_int, linestyle=":", color="k", alpha=0.5)
    ax_h_full.set_xlabel("Time $t$")
    ax_h_full.set_ylabel("Value")
    ax_h_full.set_title(f"Horizon-{H_eff} forecasts from t={t0_int} (full history)")
    ax_h_full.legend(loc="upper left")

    # --- Middle subplot: zoomed view t in [t0, t0+H_eff] ---
    zoom_times = np.arange(t0_int, t0_int + 1 + H_eff, dtype=int)
    zoom_y = env.y[t0_int : t0_int + 1 + H_eff]
    ax_h_zoom.plot(zoom_times, zoom_y, label="True $y_t$", color="black", linewidth=2)
    ax_h_zoom.plot(
        times_forecast,
        preds_oracle_plot,
        label="Oracle (truth)",
        linestyle="--",
        color="tab:gray",
    )
    # Do not plot a dedicated best-constant curve in the zoom either.
    ax_h_zoom.plot(times_forecast, preds_partial_plot, label="H-plan partial", alpha=0.8)
    ax_h_zoom.plot(times_forecast, preds_full_plot, label="H-plan full", alpha=0.8)
    for j in range(env.num_experts):
        preds_j_plot = np.concatenate(([y_t0], const_preds[j]))
        ax_h_zoom.plot(
            times_forecast,
            preds_j_plot,
            linestyle=":",
            marker="*",
            markersize=6,
            alpha=0.7,
            label=f"Const expert {j}",
        )
    if l2d_baseline is not None and preds_l2d_plan.size > 0:
        preds_l2d_plot = np.concatenate(([y_t0], preds_l2d_plan))
        ax_h_zoom.plot(
            times_forecast,
            preds_l2d_plot,
            label="L2D baseline (horizon)",
            alpha=0.9,
        )
    ax_h_zoom.axvline(t0_int, linestyle=":", color="k", alpha=0.5)
    ax_h_zoom.set_xlim(t0_int, t0_int + H_eff)
    ax_h_zoom.set_xlabel("Time $t$")
    ax_h_zoom.set_ylabel("Value")
    ax_h_zoom.set_title(f"Horizon-{H_eff} forecasts from t={t0_int} (zoom)")
    ax_h_zoom.legend(loc="upper left")

    # --- Bottom subplot: expert scheduling over the horizon ---
    ax_h_sched.step(times, sched_oracle, where="post", label="Oracle", color="tab:gray")
    # Do not plot a separate schedule for the best constant expert; const schedules per expert follow.
    ax_h_sched.step(
        times, sched_partial, where="post", label="H-plan partial", color="tab:orange"
    )
    ax_h_sched.step(
        times, sched_full, where="post", label="H-plan full", color="tab:green"
    )
    # Constant-expert schedules
    for j in range(env.num_experts):
        sched_j = [j] * H_eff
        ax_h_sched.step(
            times,
            sched_j,
            where="post",
            linestyle=":",
            alpha=0.5,
            label=f"Const {j}",
        )
    # L2D schedule
    if l2d_baseline is not None and sched_l2d:
        ax_h_sched.step(
            times,
            sched_l2d,
            where="post",
            label="L2D baseline",
            color="tab:red",
        )
    ax_h_sched.set_xlabel("Time $t$")
    ax_h_sched.set_ylabel("Expert")
    ax_h_sched.set_yticks(np.arange(env.num_experts))
    ax_h_sched.set_xlim(t0_int, t0_int + H_eff)
    ax_h_sched.set_title("Expert scheduling over horizon")
    ax_h_sched.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
