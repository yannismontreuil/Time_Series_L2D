import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Sequence, Tuple, Optional

from models.router_model import SLDSIMMRouter
from models.router_model_corr import SLDSIMMRouter_Corr
from models.factorized_slds import FactorizedSLDS
from environment.synthetic_env import SyntheticTimeSeriesEnv
from models.l2d_baseline import L2D
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB
from plot_utils import get_expert_color, get_model_color
from matplotlib import lines as mlines, patches as mpatches


def _router_observes_residual(router) -> bool:
    return getattr(router, "observation_mode", "loss") == "residual"


def _require_available(
    available: Sequence[int],
    t: int,
    actor: str,
) -> np.ndarray:
    avail_arr = np.asarray(list(available), dtype=int)
    if avail_arr.size == 0:
        raise ValueError(f"{actor}: no available experts at t={t}.")
    return avail_arr


def _mask_feedback_vector(
    values: np.ndarray,
    available: np.ndarray,
    selected: int | None,
    full_feedback: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    masked = np.full(values.shape, np.nan, dtype=float)
    if full_feedback:
        masked[available] = values[available]
    else:
        if selected is None:
            raise ValueError("selected must be provided for partial feedback masking.")
        masked[int(selected)] = values[int(selected)]
    return masked


def _get_router_observation(
    router,
    env: SyntheticTimeSeriesEnv,
    t: int,
    x_t: np.ndarray,
    available: np.ndarray,
    r_t: int,
) -> Tuple[float, float, Optional[np.ndarray]]:
    if _router_observes_residual(router):
        y_t = float(env.y[t])
        preds = env.all_expert_predictions(x_t)
        residuals = preds - y_t
        residual_r = float(residuals[int(r_t)])
        loss_r = residual_r ** 2
        residuals_full = None
        if getattr(router, "feedback_mode", "partial") == "full":
            residuals_full = _mask_feedback_vector(
                residuals, available, r_t, full_feedback=True
            )
        return residual_r, loss_r, residuals_full

    loss_all = env.losses(t)
    loss_r = float(loss_all[r_t])
    losses_full = None
    if getattr(router, "feedback_mode", "partial") == "full":
        losses_full = _mask_feedback_vector(
            loss_all, available, r_t, full_feedback=True
        )
    return loss_r, loss_r, losses_full


def _simulate_value_scenarios_for_schedule(
    env: SyntheticTimeSeriesEnv,
    schedule: Sequence[int],
    times: np.ndarray,
    num_scenarios: int,
    scenario_generator_cfg: Optional[dict],
    scenarios: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate Monte Carlo trajectories of predicted values y_hat along a
    fixed expert schedule, using a Gaussian AR(1) perturbation model in
    context space. This is used purely for visualization to illustrate
    how, for a given chosen expert j_h, the predicted value could vary
    under context noise.

    For each scenario n and horizon step h we draw:
        x_{t+h}^{(n)} = x_hat_{t+h} + ξ_h^{(n)},
    where x_hat_{t+h} = env.get_context(times[h]) is the baseline
    context path and ξ_h follows an AR(1) recursion governed by the
    scenario_generator_cfg (rho, sigma0, q_scale). The expert's
    prediction y_hat^{(n)}_{h} is then
        y_hat^{(n)}_{h} = env.expert_predict(j_h, x_{t+h}^{(n)}).

    Returns
    -------
    values_scen : np.ndarray of shape (N_scen, H_eff)
        Scenario-wise predicted values along the schedule.
    """
    H_eff = int(len(times))
    N_scen = int(num_scenarios)
    if H_eff == 0 or N_scen <= 0:
        return np.zeros((max(N_scen, 0), max(H_eff, 0)), dtype=float)

    if scenarios is None:
        scenarios = _sample_context_scenarios(
            env, times, N_scen, scenario_generator_cfg
        )
    else:
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 3 or scenarios.shape[1] != H_eff:
            raise ValueError("scenarios must have shape (N_scen, H_eff, d).")
        N_scen = int(scenarios.shape[0])
    values_scen = np.zeros((N_scen, H_eff), dtype=float)
    for n in range(N_scen):
        for h in range(H_eff):
            j_h = int(schedule[h])
            x_scen = scenarios[n, h]
            if hasattr(env, "_last_t"):
                env._last_t = int(times[h])
            values_scen[n, h] = env.expert_predict(j_h, x_scen)

    return values_scen


def warm_start_router_to_time(
    router,
    env: SyntheticTimeSeriesEnv,
    t0: int,
    t_start: int = 1,
) -> None:
    """
    Reset router beliefs and run it on the environment from time t_start
    through t0 (inclusive), using its configured feedback mode
    ("partial" or "full").
    """
    router.reset_beliefs()
    T = env.T
    t_max = min(max(int(t0), 0), T - 1)
    t_start_int = max(int(t_start), 1)
    if t_start_int > t_max:
        return
    # Align internal step counters with the environment time index.
    if hasattr(router, "current_step"):
        router.current_step = max(0, t_start_int - 1)
    if hasattr(router, "_time"):
        router._time = max(0, t_start_int - 1)
    for t in range(t_start_int, t_max + 1):
        x_t = env.get_context(t)
        available_t = _require_available(env.get_available_experts(t), t, "router warm-start")
        r_t, cache = router.select_expert(x_t, available_t)
        if not np.any(available_t == int(r_t)):
            raise ValueError(
                f"router warm-start: selected expert {r_t} not in E_t at t={t}."
            )
        loss_obs, loss_r, losses_full = _get_router_observation(
            router, env, t, x_t, available_t, r_t
        )
        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=None,
                available_experts=available_t,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=losses_full,
                available_experts=available_t,
                cache=cache,
            )


def _get_working_registry(router, num_experts: int) -> List[int]:
    if hasattr(router, "registry"):
        registry = list(getattr(router, "registry") or [])
    elif hasattr(router, "_has_joined"):
        joined = np.asarray(getattr(router, "_has_joined"), dtype=bool)
        registry = list(np.where(joined)[0])
    else:
        registry = list(range(num_experts))
    return sorted({int(k) for k in registry})


def _resolve_warm_start_time(
    router,
    t0: int,
    online_start_t: Optional[int],
) -> int:
    if online_start_t is not None:
        return max(1, min(int(online_start_t), int(t0)))
    em_tk = getattr(router, "em_tk", None)
    if em_tk is not None:
        return max(1, min(int(em_tk), int(t0)))
    return 1


def _sample_context_scenarios(
    env: SyntheticTimeSeriesEnv,
    times: np.ndarray,
    num_scenarios: int,
    scenario_generator_cfg: Optional[dict],
) -> np.ndarray:
    """
    Sample context trajectories X^{(n)}_{t+1:t+H} from a scenario
    generator. For Gaussian AR(1), fit the autoregressive dynamics on
    historical contexts up to t0 and roll forward without using future
    contexts.
    """
    H_eff = int(len(times))
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be a positive integer.")
    cfg = scenario_generator_cfg or {}
    gen_type = str(cfg.get("type", "gaussian_ar1")).lower()
    if gen_type not in ("gaussian_ar1", "deterministic"):
        raise ValueError(
            f"Unsupported scenario generator type '{gen_type}'. "
            "Expected 'gaussian_ar1' or 'deterministic'."
        )
    if H_eff == 0:
        return np.zeros((int(num_scenarios), 0, 0), dtype=float)

    # Determine the planning time t0 from the horizon times.
    t0 = int(times[0]) - 1
    t0 = max(0, min(int(t0), int(env.T) - 1))

    # Historical contexts up to t0 (inclusive).
    hist_ctx = np.stack(
        [
            np.asarray(env.get_context(int(t)), dtype=float).reshape(-1)
            for t in range(t0 + 1)
        ],
        axis=0,
    )
    ctx_dim = int(hist_ctx.shape[1])
    x_last = hist_ctx[-1].copy()

    # If history is too short, fall back to a constant forecast.
    if hist_ctx.shape[0] < 2:
        base = np.repeat(x_last[None, :], H_eff, axis=0)
        return np.repeat(base[None, :, :], int(num_scenarios), axis=0)

    # Fit a per-dimension AR(1): x_t = c + rho * x_{t-1} + eps_t.
    x_prev = hist_ctx[:-1]
    x_next = hist_ctx[1:]
    mean_prev = x_prev.mean(axis=0)
    mean_next = x_next.mean(axis=0)
    denom = ((x_prev - mean_prev) ** 2).sum(axis=0)
    numer = ((x_prev - mean_prev) * (x_next - mean_next)).sum(axis=0)
    rho_hat = np.zeros(ctx_dim, dtype=float)
    valid = denom > 1e-12
    rho_hat[valid] = numer[valid] / denom[valid]
    # Use cfg rho as a stability cap on the fitted coefficient.
    rho_cap = float(cfg.get("rho", 0.999))
    rho = np.clip(rho_hat, -abs(rho_cap), abs(rho_cap))
    intercept = mean_next - rho * mean_prev
    resid = x_next - (intercept + rho * x_prev)
    resid_var = np.mean(resid ** 2, axis=0)
    resid_std = np.sqrt(np.maximum(resid_var, 0.0))

    sigma0_scale = float(cfg.get("sigma0", 1.0))
    q_scale = float(cfg.get("q_scale", 1.0))
    noise_scale = float(cfg.get("noise_scale", 1.0))
    sigma0_vec = noise_scale * sigma0_scale * resid_std
    q_vec = noise_scale * q_scale * resid_std

    seed = cfg.get("seed", None)
    rng = np.random.default_rng(seed)
    scenarios = np.zeros((int(num_scenarios), H_eff, ctx_dim), dtype=float)
    for n in range(int(num_scenarios)):
        x_prev_n = x_last.copy()
        for h in range(H_eff):
            if gen_type == "deterministic":
                eps = np.zeros(ctx_dim, dtype=float)
            else:
                noise_scale = sigma0_vec if h == 0 else q_vec
                eps = rng.normal(loc=0.0, scale=noise_scale, size=ctx_dim)
            x_next_n = intercept + rho * x_prev_n + eps
            scenarios[n, h] = x_next_n
            x_prev_n = x_next_n
    return scenarios


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
        _require_available(
            env.get_available_experts(int(t)),
            int(t),
            "oracle schedule",
        ).copy()
        for t in times
    ]

    # Oracle "truth" schedule: per-step best expert on the true
    # environment path (kept for reference; the dynamic planning
    # oracle below will refine this schedule using simulated context).
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


def compute_dynamic_oracle_schedule(
    env: SyntheticTimeSeriesEnv,
    beta: np.ndarray,
    t0: int,
    times: np.ndarray,
    avail_per_h: List[np.ndarray],
    experts_predict: Sequence[Callable[[np.ndarray], float]],
    context_update: Callable[[np.ndarray, float], np.ndarray],
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Deprecated helper (kept for reference): dynamic planning oracle
    that simulates the context forward. Currently unused; the horizon
    oracle in evaluate_horizon_planning is defined as the per-step
    clairvoyant best along the true environment path.
    """
    return [], np.array([], dtype=float), np.array([], dtype=float)


def warm_start_l2d_to_time(
    baseline: L2D,
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
        available_t = _require_available(env.get_available_experts(t), t, "L2D warm-start")
        r_t = baseline.select_expert(x_t, available_t)
        loss_all = env.losses(t)
        loss_masked = _mask_feedback_vector(
            loss_all, available_t, r_t, full_feedback=True
        )
        baseline.update(x_t, loss_masked, available_t, selected_expert=r_t)


def warm_start_linucb_to_time(
    baseline: LinUCB,
    env: SyntheticTimeSeriesEnv,
    t0: int,
) -> None:
    """
    Train a LinUCB baseline on the environment up to time t0 using its
    configured feedback mode.
    """
    T = env.T
    t_max = min(max(int(t0), 0), T - 1)
    for t in range(1, t_max + 1):
        x_t = env.get_context(t)
        available_t = _require_available(
            env.get_available_experts(t), t, "LinUCB warm-start"
        )
        loss_all = env.losses(t)
        r_t = baseline.select_expert(x_t, available_t)
        loss_masked = _mask_feedback_vector(
            loss_all,
            available_t,
            r_t,
            full_feedback=baseline.feedback_mode == "full",
        )
        baseline.update(x_t, loss_masked, available_t, selected_expert=r_t)


def warm_start_neuralucb_to_time(
    baseline: NeuralUCB,
    env: SyntheticTimeSeriesEnv,
    t0: int,
) -> None:
    """
    Train a NeuralUCB baseline on the environment up to time t0 using its
    configured feedback mode.
    """
    T = env.T
    t_max = min(max(int(t0), 0), T - 1)
    for t in range(1, t_max + 1):
        x_t = env.get_context(t)
        available_t = _require_available(
            env.get_available_experts(t), t, "NeuralUCB warm-start"
        )
        loss_all = env.losses(t)
        r_t = baseline.select_expert(x_t, available_t)
        loss_masked = _mask_feedback_vector(
            loss_all,
            available_t,
            r_t,
            full_feedback=baseline.feedback_mode == "full",
        )
        baseline.update(x_t, loss_masked, available_t, selected_expert=r_t)


def _policy_predicted_costs(
    policy,
    x_future: np.ndarray,
    num_experts: int,
) -> np.ndarray:
    """
    Compute planning-time predicted costs for all experts using the
    policy's internal prediction rule, without any measurement updates.
    """
    if isinstance(policy, L2D):
        # L2D selects argmax of its score logits; convert to costs.
        phi = policy._advance_and_get_phi(x_future)
        scores = np.asarray(policy._scores(phi), dtype=float).reshape(num_experts)
        return -scores
    if isinstance(policy, LinUCB):
        phi = policy._get_phi(x_future)
        costs = np.zeros(num_experts, dtype=float)
        for j in range(num_experts):
            mu_j, sigma_j = policy._theta_and_sigma(int(j), phi)
            costs[j] = mu_j - policy.alpha_ucb * sigma_j + policy.beta[j]
        return costs
    if isinstance(policy, NeuralUCB):
        phi = policy._phi(x_future)
        h, _ = policy._embed(phi)
        costs = np.zeros(num_experts, dtype=float)
        for j in range(num_experts):
            mu_j, sigma_j = policy._theta_and_sigma(int(j), h)
            costs[j] = mu_j - policy.alpha_ucb * sigma_j + policy.beta[j]
        return costs
    raise TypeError(f"Unsupported policy type '{type(policy).__name__}' for planning.")


def _plan_horizon_schedule_policy_monte_carlo(
    policy,
    env: SyntheticTimeSeriesEnv,
    times: np.ndarray,
    avail_per_h: List[np.ndarray],
    num_scenarios: int,
    delta: float,
    scenario_generator_cfg: Optional[dict],
    scenarios: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """
    Monte Carlo predictive scheduling for policies with a select_expert API,
    using scenario contexts without loss updates.
    """
    H_eff = int(times.shape[0])
    if H_eff == 0:
        empty = np.zeros((0, env.num_experts), dtype=float)
        return [], empty, empty, np.zeros((0, 0), dtype=float), []

    N_scen = int(num_scenarios)
    if scenarios is not None:
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 3 or scenarios.shape[1] != H_eff:
            raise ValueError("scenarios must have shape (N_scen, H_eff, d).")
        N_scen = int(scenarios.shape[0])
    if N_scen <= 0:
        raise ValueError("num_scenarios must be a positive integer.")

    N_experts = int(env.num_experts)
    avail_sets = [np.asarray(a, dtype=int) for a in avail_per_h]
    if scenarios is None:
        scenarios = _sample_context_scenarios(
            env, times, N_scen, scenario_generator_cfg
        )

    choices_scen = np.full((N_scen, H_eff), -1, dtype=int)
    scores_scen = np.full((N_scen, H_eff, N_experts), np.nan, dtype=float)
    for n in range(N_scen):
        policy_n = copy.deepcopy(policy)
        for h in range(H_eff):
            x_future = scenarios[n, h]
            avail = avail_sets[h]
            if avail.size == 0:
                raise ValueError(
                    f"policy planning: no available experts at horizon step {h}."
                )
            costs = _policy_predicted_costs(policy_n, x_future, N_experts)
            scores_scen[n, h, :] = costs
            best_k = min(
                avail.tolist(),
                key=lambda k: (float(costs[int(k)]), int(k)),
            )
            choices_scen[n, h] = int(best_k)

    rho_hat = np.zeros((H_eff, N_experts), dtype=float)
    for h in range(H_eff):
        for n in range(N_scen):
            j = int(choices_scen[n, h])
            if j >= 0:
                rho_hat[h, j] += 1.0 / float(N_scen)

    delta = float(np.clip(delta, 0.0, 0.999))
    active_sets: List[List[int]] = []
    schedule: List[int] = []
    for h in range(H_eff):
        avail = avail_sets[h]
        if avail.size == 0:
            raise ValueError(
                f"policy planning: no available experts at horizon step {h}."
            )
        sorted_k = sorted(avail.tolist(), key=lambda k: (-rho_hat[h, k], int(k)))
        cum = 0.0
        set_h: List[int] = []
        for k in sorted_k:
            set_h.append(int(k))
            cum += float(rho_hat[h, k])
            if cum >= 1.0 - delta:
                break
        active_sets.append(set_h)
        schedule.append(int(sorted_k[0]))

    J_sched = np.full((N_scen, H_eff), np.nan, dtype=float)
    for n in range(N_scen):
        for h in range(H_eff):
            j = schedule[h]
            if 0 <= j < N_experts:
                J_sched[n, h] = scores_scen[n, h, j]

    return schedule, rho_hat, rho_hat, J_sched, active_sets


def _plan_horizon_schedule_monte_carlo(
    router,
    env: SyntheticTimeSeriesEnv,
    beta: np.ndarray,
    times: np.ndarray,
    avail_per_h: List[np.ndarray],
    num_scenarios: int,
    delta: float,
    scenario_generator_cfg: Optional[dict],
    scenarios: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """
    Monte Carlo predictive scheduling (Section "Predictive Resource Allocation"):
      - sample context scenarios X_{t+1:t+H}^{(n)},
      - propagate the time-t belief forward without measurement updates,
      - compute planning-time predicted costs C_{t+h,k}^† for feasible experts,
      - take deterministic argmin per scenario and step,
      - estimate time-marginal demand ρ̂_{t,h}(k),
      - build coverage sets Ŝ_{t,h}(δ) with cumulative mass ≥ 1-δ.

    Returns:
      schedule: deterministic single-expert summary (argmax ρ̂ per step),
      p_avail: time-marginal demand (ρ̂),
      p_all: alias of time-marginal demand (ρ̂),
      J_sched: scenario-wise costs along the summary schedule,
      active_sets: coverage sets Ŝ_{t,h}(δ) for each h.
    """
    H_eff = int(times.shape[0])
    if H_eff == 0:
        empty = np.zeros((0, env.num_experts), dtype=float)
        return [], empty, empty, np.zeros((0, 0), dtype=float), []

    N_scen = int(num_scenarios)
    if scenarios is not None:
        scenarios = np.asarray(scenarios, dtype=float)
        if scenarios.ndim != 3 or scenarios.shape[1] != H_eff:
            raise ValueError("scenarios must have shape (N_scen, H_eff, d).")
        N_scen = int(scenarios.shape[0])
    if N_scen <= 0:
        raise ValueError("num_scenarios must be a positive integer.")

    N_experts = int(getattr(env, "num_experts", getattr(router, "N", 0)))
    registry = _get_working_registry(router, N_experts)
    avail_sets = [set(map(int, a)) for a in avail_per_h]
    if scenarios is None:
        scenarios = _sample_context_scenarios(
            env, times, N_scen, scenario_generator_cfg
        )

    # Scenario-wise choices and costs.
    choices_scen = np.full((N_scen, H_eff), -1, dtype=int)
    scores_scen = np.full((N_scen, H_eff, N_experts), np.nan, dtype=float)

    if isinstance(router, FactorizedSLDS):
        for n in range(N_scen):
            w = np.asarray(router.w, dtype=float).copy()
            mu_g = np.asarray(router.mu_g, dtype=float).copy()
            Sigma_g = np.asarray(router.Sigma_g, dtype=float).copy()
            mu_u = {
                int(k): (router.mu_u.get(int(k), router.u_mean0.copy())).copy()
                for k in registry
            }
            Sigma_u = {
                int(k): (router.Sigma_u.get(int(k), router.u_cov0.copy())).copy()
                for k in registry
            }

            for h in range(H_eff):
                x_future = scenarios[n, h]
                (
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                ) = router._interaction_and_time_update(
                    x_future, w, mu_g, Sigma_g, mu_u, Sigma_u
                )

                feas = [k for k in registry if k in avail_sets[h]]
                phi = router._compute_phi(x_future)
                costs, _ = router._score_experts(
                    phi,
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                    feas,
                )
                for k, cost in costs.items():
                    scores_scen[n, h, int(k)] = float(cost)
                if costs:
                    best_k = min(
                        costs.items(), key=lambda kv: (kv[1], int(kv[0]))
                    )[0]
                    choices_scen[n, h] = int(best_k)

                w, mu_g, Sigma_g, mu_u, Sigma_u = (
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                )
    else:
        b_list, m_list, P_list = router.precompute_horizon_states(H_eff)
        for n in range(N_scen):
            for h in range(H_eff):
                x_future = scenarios[n, h]
                b_h = b_list[h]
                m_h = m_list[h]
                P_h = P_list[h]

                if isinstance(router, SLDSIMMRouter_Corr):
                    phi_h, _ = router._compute_feature(x_future)
                    mean_ell, _, _, _ = router._predict_loss_distribution(
                        phi_h, b_h, m_h, P_h
                    )
                    scores = mean_ell + beta
                else:
                    phi_vec = np.asarray(router.feature_fn(x_future), dtype=float).reshape(
                        router.d
                    )
                    mu_k = np.zeros((router.M, N_experts), dtype=float)
                    S_k = np.zeros((router.M, N_experts), dtype=float)
                    for k in range(router.M):
                        for j in range(N_experts):
                            m_kj = m_h[k, j]
                            P_kj = P_h[k, j]
                            mu = float(phi_vec @ m_kj)
                            S_val = float(
                                phi_vec @ (P_kj @ phi_vec) + router.R[k, j]
                            )
                            mu_k[k, j] = mu
                            S_k[k, j] = max(S_val, router.eps)
                    mean_ell = (b_h.reshape(-1, 1) * mu_k).sum(axis=0)
                    scores = mean_ell + beta

                scores_scen[n, h, :] = scores

                feas = [k for k in registry if k in avail_sets[h]]
                if feas:
                    feas_scores = scores[np.asarray(feas, dtype=int)]
                    best_idx = int(np.argmin(feas_scores))
                    choices_scen[n, h] = int(feas[best_idx])

    # Time-marginal demand rho_hat.
    rho_hat = np.zeros((H_eff, N_experts), dtype=float)
    for h in range(H_eff):
        for n in range(N_scen):
            j = int(choices_scen[n, h])
            if j >= 0:
                rho_hat[h, j] += 1.0 / float(N_scen)

    # Coverage sets and deterministic schedule summary.
    delta = float(np.clip(delta, 0.0, 0.999))
    active_sets: List[List[int]] = []
    schedule: List[int] = []
    for h in range(H_eff):
        feas = [k for k in registry if k in avail_sets[h]]
        if not feas:
            active_sets.append([])
            if avail_sets[h]:
                schedule.append(int(sorted(avail_sets[h])[0]))
            else:
                schedule.append(-1)
            continue
        sorted_k = sorted(feas, key=lambda k: (-rho_hat[h, k], int(k)))
        cum = 0.0
        set_h: List[int] = []
        for k in sorted_k:
            set_h.append(int(k))
            cum += float(rho_hat[h, k])
            if cum >= 1.0 - delta:
                break
        active_sets.append(set_h)
        schedule.append(int(sorted_k[0]))

    # Scenario-wise scores along the summary schedule.
    J_sched = np.full((N_scen, H_eff), np.nan, dtype=float)
    for n in range(N_scen):
        for h in range(H_eff):
            j = schedule[h]
            if 0 <= j < N_experts:
                J_sched[n, h] = scores_scen[n, h, j]

    return schedule, rho_hat, rho_hat, J_sched, active_sets


def evaluate_horizon_planning(
    env: SyntheticTimeSeriesEnv,
    router_partial: SLDSIMMRouter,
    router_full: SLDSIMMRouter,
    beta: np.ndarray,
    t0: int,
    H: int,
    experts_predict: Sequence[Callable[[np.ndarray], float]],
    context_update: Callable[[np.ndarray, float], np.ndarray],
    router_factorial_partial: Optional[FactorizedSLDS] = None,
    router_factorial_full: Optional[FactorizedSLDS] = None,
    router_factorial_partial_linear: Optional[FactorizedSLDS] = None,
    router_factorial_full_linear: Optional[FactorizedSLDS] = None,
    l2d_baseline: Optional[L2D] = None,
    l2d_sw_baseline: Optional[L2D] = None,
    linucb_partial: Optional[LinUCB] = None,
    linucb_full: Optional[LinUCB] = None,
    neuralucb_partial: Optional[NeuralUCB] = None,
    neuralucb_full: Optional[NeuralUCB] = None,
    router_partial_corr: Optional[SLDSIMMRouter_Corr] = None,
    router_full_corr: Optional[SLDSIMMRouter_Corr] = None,
    router_partial_neural=None,
    router_full_neural=None,
    router_partial_corr_em: Optional[SLDSIMMRouter_Corr] = None,
    router_full_corr_em: Optional[SLDSIMMRouter_Corr] = None,
    planning_method: str = "regressive",
    scenario_generator_cfg: Optional[dict] = None,
    delta: float = 0.1,
    online_start_t: Optional[int] = None,
    factorized_label: str = "Factorized SLDS",
    factorized_linear_label: str = "Factorized SLDS linear",
) -> None:
    """
    Compare horizon-H planning from time t0 for:
      - oracle (truth) per-step best expert,
      - constant-expert baseline,
      - partial-feedback router horizon planner,
      - full-feedback router horizon planner.

    Produces summary printouts and plots of forecasts vs truth and
    per-step costs over the horizon, plus expert scheduling. In Monte
    Carlo mode, delta controls the active-set coverage threshold.
    online_start_t (if provided) overrides any router.em_tk warm start.
    """
    raw_method = str(planning_method)
    method = raw_method.lower()

    # Parse planning method, including N-scenario Monte Carlo variants.
    mode: str
    num_scenarios = 1
    if method == "regressive":
        mode = "regressive"
    elif method == "monte_carlo":
        mode = "monte_carlo"
        num_scenarios = 1
    elif method.endswith("_monte_carlo"):
        prefix = method[: -len("_monte_carlo")]
        try:
            num_scenarios = int(prefix)
        except ValueError as exc:
            raise ValueError(
                f"Invalid planning_method '{raw_method}': expected '{{N}}_monte_carlo' "
                "with integer N, or 'monte_carlo' / 'regressive'."
            ) from exc
        if num_scenarios <= 0:
            raise ValueError(
                f"Invalid number of scenarios in planning_method '{raw_method}': "
                "N must be a positive integer."
            )
        mode = "monte_carlo"
    else:
        raise ValueError(
            f"Unsupported planning_method '{raw_method}'. "
            "Expected 'regressive', 'monte_carlo', or '{N}_monte_carlo'."
        )

    # Warm-start routers to time t0 under their feedback modes.
    warm_start_router_to_time(
        router_partial,
        env,
        t0,
        t_start=_resolve_warm_start_time(router_partial, t0, online_start_t),
    )
    warm_start_router_to_time(
        router_full,
        env,
        t0,
        t_start=_resolve_warm_start_time(router_full, t0, online_start_t),
    )
    if router_factorial_partial is not None:
        warm_start_router_to_time(
            router_factorial_partial,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_factorial_partial, t0, online_start_t
            ),
        )
    if router_factorial_full is not None:
        warm_start_router_to_time(
            router_factorial_full,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_factorial_full, t0, online_start_t
            ),
        )
    if router_factorial_partial_linear is not None:
        warm_start_router_to_time(
            router_factorial_partial_linear,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_factorial_partial_linear, t0, online_start_t
            ),
        )
    if router_factorial_full_linear is not None:
        warm_start_router_to_time(
            router_factorial_full_linear,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_factorial_full_linear, t0, online_start_t
            ),
        )
    if router_partial_corr is not None:
        warm_start_router_to_time(
            router_partial_corr,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_partial_corr, t0, online_start_t
            ),
        )
    if router_full_corr is not None:
        warm_start_router_to_time(
            router_full_corr,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_full_corr, t0, online_start_t
            ),
        )
    if router_partial_corr_em is not None:
        warm_start_router_to_time(
            router_partial_corr_em,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_partial_corr_em, t0, online_start_t
            ),
        )
    if router_full_corr_em is not None:
        warm_start_router_to_time(
            router_full_corr_em,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_full_corr_em, t0, online_start_t
            ),
        )
    if router_partial_neural is not None:
        warm_start_router_to_time(
            router_partial_neural,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_partial_neural, t0, online_start_t
            ),
        )
    if router_full_neural is not None:
        warm_start_router_to_time(
            router_full_neural,
            env,
            t0,
            t_start=_resolve_warm_start_time(
                router_full_neural, t0, online_start_t
            ),
        )

    # Warm-start L2D baseline to time t0 if provided.
    if l2d_baseline is not None:
        warm_start_l2d_to_time(l2d_baseline, env, t0)
    if l2d_sw_baseline is not None:
        warm_start_l2d_to_time(l2d_sw_baseline, env, t0)
    if linucb_partial is not None:
        warm_start_linucb_to_time(linucb_partial, env, t0)
    if linucb_full is not None:
        warm_start_linucb_to_time(linucb_full, env, t0)
    if neuralucb_partial is not None:
        warm_start_neuralucb_to_time(neuralucb_partial, env, t0)
    if neuralucb_full is not None:
        warm_start_neuralucb_to_time(neuralucb_full, env, t0)

    # Compute oracle and baseline schedules and effective horizon.
    times, H_eff, avail_per_h, sched_oracle, sched_baseline, j_baseline = (
        compute_oracle_and_baseline_schedules(env, beta, t0, H)
    )
    if H_eff <= 0:
        print("Horizon too short after clipping; skipping horizon planning evaluation.")
        return

    shared_scenarios = None
    if mode == "monte_carlo":
        shared_scenarios = _sample_context_scenarios(
            env, times, num_scenarios, scenario_generator_cfg
        )

    print(
        f"\n[Horizon planning] method = {raw_method}, "
        f"parsed_mode = {mode}, num_scenarios = {num_scenarios}, "
        f"H_eff = {H_eff}, t0 = {int(t0)}"
    )

    # Horizon-H planning schedules from current beliefs
    t0_int = int(t0)
    x_now = env.get_context(t0_int)
    avail_lists = [a.tolist() for a in avail_per_h]
    policy_num_scenarios = num_scenarios if mode == "monte_carlo" else 1
    policy_scenario_cfg = (
        scenario_generator_cfg if mode == "monte_carlo" else {"type": "deterministic"}
    )

    if mode == "regressive":
        # Original regressive / router-influenced-context planning.
        if isinstance(router_partial, FactorizedSLDS):
            (
                sched_partial,
                _,
                _,
                _,
                _,
            ) = _plan_horizon_schedule_monte_carlo(
                router_partial,
                env,
                beta,
                times,
                avail_per_h,
                1,
                delta,
                {"type": "deterministic"},
            )
        else:
            sched_partial, _, _ = router_partial.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )
        if isinstance(router_full, FactorizedSLDS):
            (
                sched_full,
                _,
                _,
                _,
                _,
            ) = _plan_horizon_schedule_monte_carlo(
                router_full,
                env,
                beta,
                times,
                avail_per_h,
                1,
                delta,
                {"type": "deterministic"},
            )
        else:
            sched_full, _, _ = router_full.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )

        if router_partial_corr is not None:
            sched_partial_corr, _, _ = router_partial_corr.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )
        else:
            sched_partial_corr = []

        if router_full_corr is not None:
            sched_full_corr, _, _ = router_full_corr.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )
        else:
            sched_full_corr = []

        if router_partial_corr_em is not None:
            sched_partial_corr_em, _, _ = router_partial_corr_em.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )
        else:
            sched_partial_corr_em = []

        if router_full_corr_em is not None:
            sched_full_corr_em, _, _ = router_full_corr_em.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )
        else:
            sched_full_corr_em = []

        if router_partial_neural is not None:
            sched_partial_neural, _, _ = router_partial_neural.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )
        else:
            sched_partial_neural = []

        if router_full_neural is not None:
            sched_full_neural, _, _ = router_full_neural.plan_horizon_schedule(
                x_t=x_now,
                H=H_eff,
                experts_predict=experts_predict,
                context_update=context_update,
                available_experts_per_h=avail_lists,
            )
        else:
            sched_full_neural = []

        if router_factorial_partial is not None:
            (
                sched_factorial_partial,
                _,
                _,
                _,
                active_sets_factorial_partial,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_partial,
                env,
                beta,
                times,
                avail_per_h,
                policy_num_scenarios,
                delta,
                policy_scenario_cfg,
            )
        else:
            sched_factorial_partial = []
            active_sets_factorial_partial = []

        if router_factorial_full is not None:
            (
                sched_factorial_full,
                _,
                _,
                _,
                active_sets_factorial_full,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_full,
                env,
                beta,
                times,
                avail_per_h,
                policy_num_scenarios,
                delta,
                policy_scenario_cfg,
            )
        else:
            sched_factorial_full = []
            active_sets_factorial_full = []

        if router_factorial_partial_linear is not None:
            (
                sched_factorial_linear_partial,
                _,
                _,
                _,
                active_sets_factorial_linear_partial,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_partial_linear,
                env,
                beta,
                times,
                avail_per_h,
                policy_num_scenarios,
                delta,
                policy_scenario_cfg,
            )
        else:
            sched_factorial_linear_partial = []
            active_sets_factorial_linear_partial = []

        if router_factorial_full_linear is not None:
            (
                sched_factorial_linear_full,
                _,
                _,
                _,
                active_sets_factorial_linear_full,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_full_linear,
                env,
                beta,
                times,
                avail_per_h,
                policy_num_scenarios,
                delta,
                policy_scenario_cfg,
            )
        else:
            sched_factorial_linear_full = []
            active_sets_factorial_linear_full = []
    else:
        # Scenario-based Monte Carlo planning with exogenous contexts:
        # Section \ref{sec:staffing-mc} and Algorithm~\ref{alg:staffing-planning}.
        (
            sched_partial,
            p_avail_partial,
            p_all_partial,
            J_sched_partial,
            active_sets_partial,
        ) = _plan_horizon_schedule_monte_carlo(
            router_partial,
            env,
            beta,
            times,
            avail_per_h,
            num_scenarios,
            delta,
            scenario_generator_cfg,
            scenarios=shared_scenarios,
        )
        (
            sched_full,
            p_avail_full,
            p_all_full,
            J_sched_full,
            active_sets_full,
        ) = _plan_horizon_schedule_monte_carlo(
            router_full,
            env,
            beta,
            times,
            avail_per_h,
            num_scenarios,
            delta,
            scenario_generator_cfg,
            scenarios=shared_scenarios,
        )

        if router_partial_corr is not None:
            (
                sched_partial_corr,
                p_avail_partial_corr,
                p_all_partial_corr,
                J_sched_partial_corr,
                active_sets_partial_corr,
            ) = _plan_horizon_schedule_monte_carlo(
                router_partial_corr,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_partial_corr = []
            p_avail_partial_corr = None
            p_all_partial_corr = None
            J_sched_partial_corr = None
            active_sets_partial_corr = []

        if router_full_corr is not None:
            (
                sched_full_corr,
                p_avail_full_corr,
                p_all_full_corr,
                J_sched_full_corr,
                active_sets_full_corr,
            ) = _plan_horizon_schedule_monte_carlo(
                router_full_corr,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_full_corr = []
            p_avail_full_corr = None
            p_all_full_corr = None
            J_sched_full_corr = None
            active_sets_full_corr = []

        if router_partial_corr_em is not None:
            (
                sched_partial_corr_em,
                p_avail_partial_corr_em,
                p_all_partial_corr_em,
                J_sched_partial_corr_em,
                active_sets_partial_corr_em,
            ) = _plan_horizon_schedule_monte_carlo(
                router_partial_corr_em,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_partial_corr_em = []
            p_avail_partial_corr_em = None
            p_all_partial_corr_em = None
            J_sched_partial_corr_em = None
            active_sets_partial_corr_em = []

        if router_full_corr_em is not None:
            (
                sched_full_corr_em,
                p_avail_full_corr_em,
                p_all_full_corr_em,
                J_sched_full_corr_em,
                active_sets_full_corr_em,
            ) = _plan_horizon_schedule_monte_carlo(
                router_full_corr_em,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_full_corr_em = []
            p_avail_full_corr_em = None
            p_all_full_corr_em = None
            J_sched_full_corr_em = None
            active_sets_full_corr_em = []

        if router_factorial_partial is not None:
            (
                sched_factorial_partial,
                _,
                _,
                _,
                active_sets_factorial_partial,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_partial,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_factorial_partial = []
            active_sets_factorial_partial = []

        if router_factorial_full is not None:
            (
                sched_factorial_full,
                _,
                _,
                _,
                active_sets_factorial_full,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_full,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_factorial_full = []
            active_sets_factorial_full = []

        if router_factorial_partial_linear is not None:
            (
                sched_factorial_linear_partial,
                _,
                _,
                _,
                active_sets_factorial_linear_partial,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_partial_linear,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_factorial_linear_partial = []
            active_sets_factorial_linear_partial = []

        if router_factorial_full_linear is not None:
            (
                sched_factorial_linear_full,
                _,
                _,
                _,
                active_sets_factorial_linear_full,
            ) = _plan_horizon_schedule_monte_carlo(
                router_factorial_full_linear,
                env,
                beta,
                times,
                avail_per_h,
                num_scenarios,
                delta,
                scenario_generator_cfg,
                scenarios=shared_scenarios,
            )
        else:
            sched_factorial_linear_full = []
            active_sets_factorial_linear_full = []

        # For neural routers we do not have a generative SLDS model to
        # support IMM-based staffing planning, so we skip Monte Carlo
        # planning for them in this mode.
        sched_partial_neural = []
        sched_full_neural = []

    if l2d_baseline is not None:
        (
            sched_l2d,
            _,
            _,
            _,
            active_sets_l2d,
        ) = _plan_horizon_schedule_policy_monte_carlo(
            l2d_baseline,
            env,
            times,
            avail_per_h,
            policy_num_scenarios,
            delta,
            policy_scenario_cfg,
            scenarios=shared_scenarios,
        )
    else:
        sched_l2d = []
        active_sets_l2d = []

    if l2d_sw_baseline is not None:
        (
            sched_l2d_sw,
            _,
            _,
            _,
            active_sets_l2d_sw,
        ) = _plan_horizon_schedule_policy_monte_carlo(
            l2d_sw_baseline,
            env,
            times,
            avail_per_h,
            policy_num_scenarios,
            delta,
            policy_scenario_cfg,
            scenarios=shared_scenarios,
        )
    else:
        sched_l2d_sw = []
        active_sets_l2d_sw = []

    if linucb_partial is not None:
        (
            sched_linucb_partial,
            _,
            _,
            _,
            active_sets_linucb_partial,
        ) = _plan_horizon_schedule_policy_monte_carlo(
            linucb_partial,
            env,
            times,
            avail_per_h,
            policy_num_scenarios,
            delta,
            policy_scenario_cfg,
            scenarios=shared_scenarios,
        )
    else:
        sched_linucb_partial = []
        active_sets_linucb_partial = []

    if linucb_full is not None:
        (
            sched_linucb_full,
            _,
            _,
            _,
            active_sets_linucb_full,
        ) = _plan_horizon_schedule_policy_monte_carlo(
            linucb_full,
            env,
            times,
            avail_per_h,
            policy_num_scenarios,
            delta,
            policy_scenario_cfg,
            scenarios=shared_scenarios,
        )
    else:
        sched_linucb_full = []
        active_sets_linucb_full = []

    if neuralucb_partial is not None:
        (
            sched_neuralucb_partial,
            _,
            _,
            _,
            active_sets_neuralucb_partial,
        ) = _plan_horizon_schedule_policy_monte_carlo(
            neuralucb_partial,
            env,
            times,
            avail_per_h,
            policy_num_scenarios,
            delta,
            policy_scenario_cfg,
            scenarios=shared_scenarios,
        )
    else:
        sched_neuralucb_partial = []
        active_sets_neuralucb_partial = []

    if neuralucb_full is not None:
        (
            sched_neuralucb_full,
            _,
            _,
            _,
            active_sets_neuralucb_full,
        ) = _plan_horizon_schedule_policy_monte_carlo(
            neuralucb_full,
            env,
            times,
            avail_per_h,
            policy_num_scenarios,
            delta,
            policy_scenario_cfg,
            scenarios=shared_scenarios,
        )
    else:
        sched_neuralucb_full = []
        active_sets_neuralucb_full = []

    seed_sched = policy_scenario_cfg.get("seed", 0)
    rng_sched = np.random.default_rng(int(seed_sched))
    sched_random = []
    for h in range(H_eff):
        avail = avail_per_h[h]
        if avail.size == 0:
            raise ValueError(
                f"random schedule: no available experts at horizon step {h}."
            )
        else:
            sched_random.append(int(rng_sched.choice(avail)))

    # Evaluate oracle and all other schedules on the true environment
    # path (clairvoyant per-step best given true losses).
    preds_oracle, cost_oracle = eval_schedule_on_env(env, beta, times, sched_oracle)
    preds_baseline, cost_baseline = eval_schedule_on_env(env, beta, times, sched_baseline)
    preds_partial_plan, cost_partial_plan = eval_schedule_on_env(
        env, beta, times, sched_partial
    )
    preds_full_plan, cost_full_plan = eval_schedule_on_env(
        env, beta, times, sched_full
    )

    if sched_factorial_partial:
        preds_factorial_partial_plan, cost_factorial_partial_plan = eval_schedule_on_env(
            env, beta, times, sched_factorial_partial
        )
    else:
        preds_factorial_partial_plan = np.array([], dtype=float)
        cost_factorial_partial_plan = np.array([], dtype=float)

    if sched_factorial_full:
        preds_factorial_full_plan, cost_factorial_full_plan = eval_schedule_on_env(
            env, beta, times, sched_factorial_full
        )
    else:
        preds_factorial_full_plan = np.array([], dtype=float)
        cost_factorial_full_plan = np.array([], dtype=float)

    if sched_factorial_linear_partial:
        (
            preds_factorial_linear_partial_plan,
            cost_factorial_linear_partial_plan,
        ) = eval_schedule_on_env(env, beta, times, sched_factorial_linear_partial)
    else:
        preds_factorial_linear_partial_plan = np.array([], dtype=float)
        cost_factorial_linear_partial_plan = np.array([], dtype=float)

    if sched_factorial_linear_full:
        preds_factorial_linear_full_plan, cost_factorial_linear_full_plan = (
            eval_schedule_on_env(env, beta, times, sched_factorial_linear_full)
        )
    else:
        preds_factorial_linear_full_plan = np.array([], dtype=float)
        cost_factorial_linear_full_plan = np.array([], dtype=float)

    if sched_partial_corr:
        preds_partial_corr_plan, cost_partial_corr_plan = eval_schedule_on_env(
            env, beta, times, sched_partial_corr
        )
    else:
        preds_partial_corr_plan = np.array([], dtype=float)
        cost_partial_corr_plan = np.array([], dtype=float)

    if sched_full_corr:
        preds_full_corr_plan, cost_full_corr_plan = eval_schedule_on_env(
            env, beta, times, sched_full_corr
        )
    else:
        preds_full_corr_plan = np.array([], dtype=float)
        cost_full_corr_plan = np.array([], dtype=float)

    if sched_partial_corr_em:
        preds_partial_corr_em_plan, cost_partial_corr_em_plan = eval_schedule_on_env(
            env, beta, times, sched_partial_corr_em
        )
    else:
        preds_partial_corr_em_plan = np.array([], dtype=float)
        cost_partial_corr_em_plan = np.array([], dtype=float)

    if sched_full_corr_em:
        preds_full_corr_em_plan, cost_full_corr_em_plan = eval_schedule_on_env(
            env, beta, times, sched_full_corr_em
        )
    else:
        preds_full_corr_em_plan = np.array([], dtype=float)
        cost_full_corr_em_plan = np.array([], dtype=float)

    if sched_partial_neural:
        preds_partial_neural_plan, cost_partial_neural_plan = eval_schedule_on_env(
            env, beta, times, sched_partial_neural
        )
    else:
        preds_partial_neural_plan = np.array([], dtype=float)
        cost_partial_neural_plan = np.array([], dtype=float)

    if sched_full_neural:
        preds_full_neural_plan, cost_full_neural_plan = eval_schedule_on_env(
            env, beta, times, sched_full_neural
        )
    else:
        preds_full_neural_plan = np.array([], dtype=float)
        cost_full_neural_plan = np.array([], dtype=float)

    # Constant-expert baselines for all experts (theoretical, ignore availability)
    const_preds = []
    const_costs = []
    for j in range(env.num_experts):
        sched_j = [j] * H_eff
        preds_j, costs_j = eval_schedule_on_env(env, beta, times, sched_j)
        const_preds.append(preds_j)
        const_costs.append(costs_j)

    if sched_l2d:
        preds_l2d_plan, cost_l2d_plan = eval_schedule_on_env(
            env, beta, times, sched_l2d
        )
    else:
        preds_l2d_plan = np.array([], dtype=float)
        cost_l2d_plan = np.array([], dtype=float)

    if sched_l2d_sw:
        preds_l2d_sw_plan, cost_l2d_sw_plan = eval_schedule_on_env(
            env, beta, times, sched_l2d_sw
        )
    else:
        preds_l2d_sw_plan = np.array([], dtype=float)
        cost_l2d_sw_plan = np.array([], dtype=float)

    if sched_linucb_partial:
        preds_linucb_partial_plan, cost_linucb_partial_plan = eval_schedule_on_env(
            env, beta, times, sched_linucb_partial
        )
    else:
        preds_linucb_partial_plan = np.array([], dtype=float)
        cost_linucb_partial_plan = np.array([], dtype=float)

    if sched_linucb_full:
        preds_linucb_full_plan, cost_linucb_full_plan = eval_schedule_on_env(
            env, beta, times, sched_linucb_full
        )
    else:
        preds_linucb_full_plan = np.array([], dtype=float)
        cost_linucb_full_plan = np.array([], dtype=float)

    if sched_neuralucb_partial:
        (
            preds_neuralucb_partial_plan,
            cost_neuralucb_partial_plan,
        ) = eval_schedule_on_env(env, beta, times, sched_neuralucb_partial)
    else:
        preds_neuralucb_partial_plan = np.array([], dtype=float)
        cost_neuralucb_partial_plan = np.array([], dtype=float)

    if sched_neuralucb_full:
        preds_neuralucb_full_plan, cost_neuralucb_full_plan = eval_schedule_on_env(
            env, beta, times, sched_neuralucb_full
        )
    else:
        preds_neuralucb_full_plan = np.array([], dtype=float)
        cost_neuralucb_full_plan = np.array([], dtype=float)

    if sched_random:
        preds_random_plan, cost_random_plan = eval_schedule_on_env(
            env, beta, times, sched_random
        )
    else:
        preds_random_plan = np.array([], dtype=float)
        cost_random_plan = np.array([], dtype=float)

    plot_target = str(getattr(env, "plot_target", "y")).lower()
    if plot_target == "x":
        y_window = env.x[t0_int + 1 : t0_int + 1 + H_eff]
        true_label = "Context $x_t$ (lagged)"
    else:
        y_window = env.y[t0_int + 1 : t0_int + 1 + H_eff]
        true_label = "True $y_t$"

    # Visualization-only shift; positive values advance predictions (shift left).
    vis_shift = int(getattr(env, "plot_shift", 1))

    def _shift_vis_preds(preds: np.ndarray) -> np.ndarray:
        preds = np.asarray(preds, dtype=float)
        if preds.size == 0 or vis_shift <= 0:
            if vis_shift == 0:
                return preds
            if preds.size <= abs(vis_shift):
                return np.array([], dtype=float)
            return preds[:vis_shift]
        if preds.size <= vis_shift:
            return np.array([], dtype=float)
        return preds[vis_shift:]

    def _shift_vis_schedule(schedule: Sequence[int]) -> List[int]:
        if vis_shift <= 0:
            if vis_shift == 0:
                return [int(j) for j in schedule]
            if len(schedule) <= abs(vis_shift):
                return []
            return [int(j) for j in schedule[:vis_shift]]
        if len(schedule) <= vis_shift:
            return []
        return [int(j) for j in schedule[vis_shift:]]

    if vis_shift > 0:
        times_vis = times[:-vis_shift]
        y_window_vis = y_window[:-vis_shift]
    elif vis_shift < 0:
        times_vis = times[-vis_shift:]
        y_window_vis = y_window[-vis_shift:]
    else:
        times_vis = times
        y_window_vis = y_window

    preds_oracle_vis = _shift_vis_preds(preds_oracle)
    preds_partial_vis = _shift_vis_preds(preds_partial_plan)
    preds_full_vis = _shift_vis_preds(preds_full_plan)
    preds_factorial_partial_vis = _shift_vis_preds(preds_factorial_partial_plan)
    preds_factorial_full_vis = _shift_vis_preds(preds_factorial_full_plan)
    preds_factorial_linear_partial_vis = _shift_vis_preds(
        preds_factorial_linear_partial_plan
    )
    preds_factorial_linear_full_vis = _shift_vis_preds(
        preds_factorial_linear_full_plan
    )
    preds_partial_neural_vis = _shift_vis_preds(preds_partial_neural_plan)
    preds_full_neural_vis = _shift_vis_preds(preds_full_neural_plan)
    preds_partial_corr_vis = _shift_vis_preds(preds_partial_corr_plan)
    preds_full_corr_vis = _shift_vis_preds(preds_full_corr_plan)
    preds_partial_corr_em_vis = _shift_vis_preds(preds_partial_corr_em_plan)
    preds_full_corr_em_vis = _shift_vis_preds(preds_full_corr_em_plan)
    preds_l2d_vis = _shift_vis_preds(preds_l2d_plan)
    preds_l2d_sw_vis = _shift_vis_preds(preds_l2d_sw_plan)
    preds_linucb_partial_vis = _shift_vis_preds(preds_linucb_partial_plan)
    preds_linucb_full_vis = _shift_vis_preds(preds_linucb_full_plan)
    preds_neuralucb_partial_vis = _shift_vis_preds(preds_neuralucb_partial_plan)
    preds_neuralucb_full_vis = _shift_vis_preds(preds_neuralucb_full_plan)
    preds_random_vis = _shift_vis_preds(preds_random_plan)

    const_preds_vis = [_shift_vis_preds(p) for p in const_preds]

    base_partial_label = "Factorized SLDS w/o g_t (partial fb)"
    base_full_label = "Factorized SLDS w/o g_t (full fb)"
    factorized_partial_label = f"{factorized_label} (partial fb)"
    factorized_full_label = f"{factorized_label} (full fb)"
    factorized_linear_partial_label = f"{factorized_linear_label} (partial fb)"
    factorized_linear_full_label = f"{factorized_linear_label} (full fb)"
    corr_partial_label = "Router Corr (partial feedback)"
    corr_full_label = "Router Corr (full feedback)"
    corr_em_partial_label = "Router Corr EM (partial fb)"
    corr_em_full_label = "Router Corr EM (full fb)"
    neural_partial_label = "Neural router (partial fb)"
    neural_full_label = "Neural router (full fb)"
    l2d_label = "L2D (full feedback)"
    l2d_sw_label = "L2D_SW (full feedback)"
    linucb_partial_label = "LinUCB (partial feedback)"
    linucb_full_label = "LinUCB (full feedback)"
    neuralucb_partial_label = "NeuralUCB (partial feedback)"
    neuralucb_full_label = "NeuralUCB (full feedback)"
    random_label = "Random baseline"
    oracle_label = "Oracle baseline"

    # Print horizon average costs
    print(f"\n=== Horizon-{H_eff} average costs from t={t0_int} ===")
    print(f"{oracle_label}:                {cost_oracle.mean():.4f}")
    print(f"Best constant expert (exp {j_baseline}): {cost_baseline.mean():.4f}")
    for j in range(env.num_experts):
        print(f"Always using expert {j}:       {const_costs[j].mean():.4f}")
    print(f"{base_partial_label}:       {cost_partial_plan.mean():.4f}")
    print(f"{base_full_label}:          {cost_full_plan.mean():.4f}")
    if cost_factorial_partial_plan.size > 0:
        print(
            f"{factorized_partial_label}:   {cost_factorial_partial_plan.mean():.4f}"
        )
    if cost_factorial_full_plan.size > 0:
        print(
            f"{factorized_full_label}:      {cost_factorial_full_plan.mean():.4f}"
        )
    if cost_factorial_linear_partial_plan.size > 0:
        print(
            f"{factorized_linear_partial_label}: "
            f"{cost_factorial_linear_partial_plan.mean():.4f}"
        )
    if cost_factorial_linear_full_plan.size > 0:
        print(
            f"{factorized_linear_full_label}:    "
            f"{cost_factorial_linear_full_plan.mean():.4f}"
        )
    if cost_partial_neural_plan.size > 0:
        print(
            f"{neural_partial_label}: {cost_partial_neural_plan.mean():.4f}"
        )
    if cost_full_neural_plan.size > 0:
        print(
            f"{neural_full_label}:    {cost_full_neural_plan.mean():.4f}"
        )
    if cost_partial_corr_plan.size > 0:
        print(
            f"{corr_partial_label}: {cost_partial_corr_plan.mean():.4f}"
        )
    if cost_full_corr_plan.size > 0:
        print(f"{corr_full_label}:    {cost_full_corr_plan.mean():.4f}")
    if cost_partial_corr_em_plan.size > 0:
        print(
            f"{corr_em_partial_label}: {cost_partial_corr_em_plan.mean():.4f}"
        )
    if cost_full_corr_em_plan.size > 0:
        print(
            f"{corr_em_full_label}:    {cost_full_corr_em_plan.mean():.4f}"
        )
    if cost_l2d_plan.size > 0:
        print(f"{l2d_label}:       {cost_l2d_plan.mean():.4f}")
    if cost_l2d_sw_plan.size > 0:
        print(f"{l2d_sw_label}:    {cost_l2d_sw_plan.mean():.4f}")
    if cost_linucb_partial_plan.size > 0:
        print(
            f"{linucb_partial_label}:     {cost_linucb_partial_plan.mean():.4f}"
        )
    if cost_linucb_full_plan.size > 0:
        print(
            f"{linucb_full_label}:        {cost_linucb_full_plan.mean():.4f}"
        )
    if cost_neuralucb_partial_plan.size > 0:
        print(
            f"{neuralucb_partial_label}:  "
            f"{cost_neuralucb_partial_plan.mean():.4f}"
        )
    if cost_neuralucb_full_plan.size > 0:
        print(
            f"{neuralucb_full_label}:     "
            f"{cost_neuralucb_full_plan.mean():.4f}"
        )
    if cost_random_plan.size > 0:
        print(f"{random_label}:    {cost_random_plan.mean():.4f}")

    # ------------------------------------------------------------------
    # Heatmaps of selection probabilities (Monte Carlo planning only)
    # ------------------------------------------------------------------
    if mode == "monte_carlo":
        # For all routers included in this experiment (partial, full,
        # partial corr, full corr), print a small numeric slice of the
        # best-over-all selection probabilities and visualize them as
        # heatmaps over (horizon step, expert index).
        def plot_prob_heatmap(prob_matrix: np.ndarray, title: str) -> None:
            if prob_matrix.size == 0:
                return
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            im = ax.imshow(
                prob_matrix,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
            )
            ax.set_xlabel("Expert index j")
            ax.set_ylabel("Horizon step h (1,...,H)")
            ax.set_title(title)
            ax.set_xticks(np.arange(env.num_experts))
            fig.colorbar(im, ax=ax, label="Selection probability")
            plt.tight_layout()
            plt.show()

        prob_tables = [
            (p_all_partial, base_partial_label),
            (p_all_full, base_full_label),
        ]
        if p_all_partial_corr is not None:
            prob_tables.append((p_all_partial_corr, corr_partial_label))
        if p_all_full_corr is not None:
            prob_tables.append((p_all_full_corr, corr_full_label))

        print("\n[Monte Carlo] Best-over-all selection probability tables (first 3 steps):")
        for mat, label in prob_tables:
            if mat.size == 0:
                continue
            h_max = min(3, mat.shape[0])
            print(f"\n  {label} (p_all, steps 1..{h_max}):")
            print(mat[:h_max, :])
            plot_prob_heatmap(mat, f"Best-over-all selection probability ({label})")

        def print_active_sets(active_sets: List[List[int]], label: str) -> None:
            if not active_sets:
                return
            h_max = min(3, len(active_sets))
            print(
                f"\n  {label} coverage sets (delta={float(delta):.3f}, steps 1..{h_max}):"
            )
            for h in range(h_max):
                print(f"    h={h + 1}: {active_sets[h]}")

        print("\n[Monte Carlo] Coverage sets from time-marginals:")
        print_active_sets(active_sets_partial, base_partial_label)
        print_active_sets(active_sets_full, base_full_label)
        print_active_sets(active_sets_partial_corr, corr_partial_label)
        print_active_sets(active_sets_full_corr, corr_full_label)
        print_active_sets(active_sets_partial_corr_em, corr_em_partial_label)
        print_active_sets(active_sets_full_corr_em, corr_em_full_label)
        print_active_sets(active_sets_factorial_partial, factorized_partial_label)
        print_active_sets(active_sets_factorial_full, factorized_full_label)
        print_active_sets(
            active_sets_factorial_linear_partial, factorized_linear_partial_label
        )
        print_active_sets(
            active_sets_factorial_linear_full, factorized_linear_full_label
        )
        print_active_sets(active_sets_l2d, l2d_label)
        print_active_sets(active_sets_l2d_sw, l2d_sw_label)
        print_active_sets(active_sets_linucb_partial, linucb_partial_label)
        print_active_sets(active_sets_linucb_full, linucb_full_label)
        print_active_sets(active_sets_neuralucb_partial, neuralucb_partial_label)
        print_active_sets(active_sets_neuralucb_full, neuralucb_full_label)

        # --------------------------------------------------------------
        # New: Monte Carlo trajectories for oracle + factorized schedules
        # --------------------------------------------------------------
        if shared_scenarios is not None:
            fig_mc, ax_mc = plt.subplots(1, 1, figsize=(10, 4))
            ax_mc.plot(
                times_vis,
                y_window_vis,
                label=true_label,
                color=get_model_color("true"),
                linewidth=2,
            )

            def plot_mc_trajectories(
                label: str,
                color_key: str,
                schedule: Sequence[int],
                true_preds: Optional[np.ndarray],
            ) -> None:
                if not schedule:
                    return
                values_scen = _simulate_value_scenarios_for_schedule(
                    env,
                    schedule,
                    times,
                    num_scenarios,
                    scenario_generator_cfg,
                    scenarios=shared_scenarios,
                )
                values_scen = (
                    values_scen[:, vis_shift:]
                    if vis_shift > 0 and values_scen.shape[1] > vis_shift
                    else values_scen
                )
                color = get_model_color(color_key)
                for n in range(values_scen.shape[0]):
                    ax_mc.plot(
                        times_vis,
                        values_scen[n],
                        color=color,
                        alpha=0.15,
                        linewidth=1.0,
                    )
                if true_preds is not None and true_preds.size > 0:
                    plot_len = min(len(times_vis), int(true_preds.shape[0]))
                    if plot_len > 0:
                        ax_mc.plot(
                            times_vis[:plot_len],
                            true_preds[:plot_len],
                            color=color,
                            linewidth=2.0,
                            label=label,
                        )
                elif values_scen.size > 0:
                    mean_vals = values_scen.mean(axis=0)
                    ax_mc.plot(
                        times_vis,
                        mean_vals,
                        color=color,
                        linewidth=2.0,
                        label=label,
                    )

            plot_mc_trajectories(
                oracle_label,
                "oracle",
                sched_oracle,
                preds_oracle_vis,
            )
            plot_mc_trajectories(
                factorized_partial_label,
                "factorized_partial",
                sched_factorial_partial,
                preds_factorial_partial_vis,
            )
            plot_mc_trajectories(
                factorized_full_label,
                "factorized_full",
                sched_factorial_full,
                preds_factorial_full_vis,
            )

            ax_mc.set_xlabel("Time $t$")
            ax_mc.set_ylabel("Value")
            ax_mc.set_title(
                f"Horizon-{H_eff} MC trajectories (oracle + factorized)"
            )
            ax_mc.legend(loc="upper left")
            plt.tight_layout()
            plt.show()

        # --------------------------------------------------------------
        # New: number of planned experts per time step
        # --------------------------------------------------------------
        steps = np.arange(1, H_eff + 1, dtype=int)
        fig_counts, ax_counts = plt.subplots(1, 1, figsize=(10, 4))

        def plot_active_set_sizes(
            active_sets: List[List[int]],
            label: str,
            color_key: str,
        ):
            if not active_sets:
                return
            sizes = np.array([len(s) for s in active_sets], dtype=float)
            ax_counts.plot(
                steps,
                sizes,
                label=label,
                color=get_model_color(color_key),
                linewidth=1.8,
            )

        plot_active_set_sizes(active_sets_partial, base_partial_label, "partial")
        plot_active_set_sizes(active_sets_full, base_full_label, "full")
        plot_active_set_sizes(active_sets_partial_corr, corr_partial_label, "partial_corr")
        plot_active_set_sizes(active_sets_full_corr, corr_full_label, "full_corr")
        plot_active_set_sizes(
            active_sets_partial_corr_em, corr_em_partial_label, "partial_corr_em"
        )
        plot_active_set_sizes(
            active_sets_full_corr_em, corr_em_full_label, "full_corr_em"
        )
        plot_active_set_sizes(
            active_sets_factorial_partial, factorized_partial_label, "factorized_partial"
        )
        plot_active_set_sizes(
            active_sets_factorial_full, factorized_full_label, "factorized_full"
        )
        plot_active_set_sizes(
            active_sets_factorial_linear_partial,
            factorized_linear_partial_label,
            "factorized_linear_partial",
        )
        plot_active_set_sizes(
            active_sets_factorial_linear_full,
            factorized_linear_full_label,
            "factorized_linear_full",
        )
        plot_active_set_sizes(active_sets_l2d, l2d_label, "l2d")
        plot_active_set_sizes(active_sets_l2d_sw, l2d_sw_label, "l2d_sw")
        plot_active_set_sizes(active_sets_linucb_partial, linucb_partial_label, "linucb_partial")
        plot_active_set_sizes(active_sets_linucb_full, linucb_full_label, "linucb_full")
        plot_active_set_sizes(
            active_sets_neuralucb_partial, neuralucb_partial_label, "neuralucb"
        )
        plot_active_set_sizes(
            active_sets_neuralucb_full, neuralucb_full_label, "neuralucb"
        )

        ax_counts.set_xlabel("Horizon step $h$")
        ax_counts.set_ylabel("Num experts in active set")
        ax_counts.set_title("Planned expert set sizes over the horizon")
        ax_counts.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------------
        # New: truth y_t and correlated-router forecasts with MC value bands
        # ------------------------------------------------------------------
        if cost_full_corr_plan.size > 0 or cost_partial_corr_plan.size > 0:
            fig_corr, ax_y = plt.subplots(1, 1, figsize=(10, 4))

            # True series on the horizon window.
            ax_y.plot(
                times,
                y_window,
                label=true_label,
                color=get_model_color("true"),
                linewidth=2,
            )

            # Monte Carlo value bands for correlated routers (if schedules exist).
            if sched_full_corr:
                values_full_corr_scen = _simulate_value_scenarios_for_schedule(
                    env,
                    sched_full_corr,
                    times,
                    num_scenarios,
                    scenario_generator_cfg,
                )
                mean_full = values_full_corr_scen.mean(axis=0)
                std_full = values_full_corr_scen.std(axis=0)
                lower_full = mean_full - std_full
                upper_full = mean_full + std_full
                color_full = get_model_color("full_corr")
                ax_y.fill_between(
                    times,
                    lower_full,
                    upper_full,
                    color=color_full,
                    alpha=0.15,
                    label="Corr full (MC band, value)",
                )
            if sched_partial_corr:
                values_partial_corr_scen = _simulate_value_scenarios_for_schedule(
                    env,
                    sched_partial_corr,
                    times,
                    num_scenarios,
                    scenario_generator_cfg,
                )
                mean_part = values_partial_corr_scen.mean(axis=0)
                std_part = values_partial_corr_scen.std(axis=0)
                lower_part = mean_part - std_part
                upper_part = mean_part + std_part
                color_part = get_model_color("partial_corr")
                ax_y.fill_between(
                    times,
                    lower_part,
                    upper_part,
                    color=color_part,
                    alpha=0.15,
                    label="Corr partial (MC band, value)",
                )

            # Correlated-router forecasts on the same horizon.
            if cost_full_corr_plan.size > 0 and preds_full_corr_plan.size > 0:
                ax_y.plot(
                    times,
                    preds_full_corr_plan,
                    label=corr_full_label,
                    color=get_model_color("full_corr"),
                    linewidth=1.8,
                    marker="o",
                    markersize=4,
                    markerfacecolor="none",
                )
            if cost_partial_corr_plan.size > 0 and preds_partial_corr_plan.size > 0:
                ax_y.plot(
                    times,
                    preds_partial_corr_plan,
                    label=corr_partial_label,
                    color=get_model_color("partial_corr"),
                    linewidth=1.8,
                    marker="o",
                    markersize=4,
                    markerfacecolor="none",
                )
            # Oracle trajectory on the same horizon, for fair comparison.
            if preds_oracle.size > 0:
                ax_y.plot(
                    times,
                    preds_oracle,
                    label=oracle_label,
                    color=get_model_color("oracle"),
                    linewidth=1.8,
                )
            ax_y.set_xlabel("Time $t$")
            ax_y.set_ylabel("Value")
            ax_y.set_title(
                f"True series and correlated-router forecasts "
                f"on horizon [{times[0]}, {times[-1]}]"
            )
            ax_y.legend(loc="upper left")
            plt.tight_layout()
            plt.show()

        # ------------------------------------------------------------------
        # New: expert selection sets for correlated routers
        # ------------------------------------------------------------------
        if sched_full_corr or sched_partial_corr:
            fig_sel, ax_sel = plt.subplots(1, 1, figsize=(10, 4))

            # Deterministic schedules (no-noise planning outcome)
            # Add the clairvoyant oracle schedule for comparison.
            ax_sel.step(
                times,
                sched_oracle,
                where="post",
                label=oracle_label,
                color=get_model_color("oracle"),
                linewidth=2,
                linestyle=":",
            )
            if sched_full_corr:
                ax_sel.step(
                    times,
                    sched_full_corr,
                    where="post",
                    label=corr_full_label,
                    color=get_model_color("full_corr"),
                    linewidth=2,
                )
            if sched_partial_corr:
                ax_sel.step(
                    times,
                    sched_partial_corr,
                    where="post",
                    label=corr_partial_label,
                    color=get_model_color("partial_corr"),
                    linewidth=2,
                    linestyle="--",
                )

            # For each horizon step, explicitly mark which experts could
            # be optimal under noise. If, for a given h, the Monte Carlo
            # runs ever select experts e1, e2, e3 as best-over-all, we
            # draw markers at (t_h, e1), (t_h, e2), (t_h, e3).
            eps_prob = 1e-12

            # Helper to scatter support of p_all_*.
            def scatter_expert_support(
                prob_matrix: np.ndarray,
                color_key: str,
                label: str,
                marker: str,
            ) -> None:
                if prob_matrix.size == 0:
                    return
                color = get_model_color(color_key)
                plotted = False
                for j in range(env.num_experts):
                    mask_h = prob_matrix[:, j] > eps_prob
                    if not np.any(mask_h):
                        continue
                    # Times where expert j is selected in at least one scenario.
                    t_support = times[mask_h]
                    y_j = np.full(t_support.shape[0], j, dtype=float)
                    ax_sel.scatter(
                    t_support,
                    y_j,
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    alpha=0.6,
                    s=30,
                        label=label if not plotted else None,
                    )
                    plotted = True

            if p_all_full_corr is not None:
                scatter_expert_support(
                    p_all_full_corr,
                    "full_corr",
                    f"{corr_full_label} (experts possible under noise)",
                    marker="o",
                )
            if p_all_partial_corr is not None:
                scatter_expert_support(
                    p_all_partial_corr,
                    "partial_corr",
                    f"{corr_partial_label} (experts possible under noise)",
                    marker="o",
                )

            ax_sel.set_xlabel("Time $t$")
            ax_sel.set_ylabel("Expert index")
            ax_sel.set_yticks(np.arange(env.num_experts))
            ax_sel.set_title(
                "Expert selection under noise (schedules and possible experts)"
            )
            ax_sel.legend(loc="upper left")
            plt.tight_layout()
            plt.show()

            # Separate figure: number of experts possible per time.
            fig_count, ax_count = plt.subplots(1, 1, figsize=(10, 3))
            if p_all_full_corr is not None and p_all_full_corr.size > 0:
                n_full = (p_all_full_corr > eps_prob).sum(axis=1)
                ax_count.plot(
                    times,
                    n_full,
                    color=get_model_color("full_corr"),
                    linestyle=":",
                    marker="o",
                    markerfacecolor="none",
                    linewidth=1.5,
                    label=f"{corr_full_label} (num experts possible)",
                )
            if p_all_partial_corr is not None and p_all_partial_corr.size > 0:
                n_part = (p_all_partial_corr > eps_prob).sum(axis=1)
                ax_count.plot(
                    times,
                    n_part,
                    color=get_model_color("partial_corr"),
                    linestyle=":",
                    marker="o",
                    markerfacecolor="none",
                    linewidth=1.5,
                    label=f"{corr_partial_label} (num experts possible)",
                )
            ax_count.set_xlabel("Time $t$")
            ax_count.set_ylabel("num experts possible")
            ax_count.set_title("Number of experts possible under noise")
            ax_count.legend(loc="upper left")
            plt.tight_layout()
            plt.show()

    # Plot forecasts vs truth and zoomed horizon window + scheduling
    fig_h, (ax_h_full, ax_h_zoom, ax_h_sched) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=False
    )

    # --- Top subplot: full history up to end of horizon ---
    all_times = np.arange(0, t0_int + 1 + H_eff, dtype=int)
    if plot_target == "x":
        all_y = env.x[: t0_int + 1 + H_eff]
    else:
        all_y = env.y[: t0_int + 1 + H_eff]
    ax_h_full.plot(
        all_times,
        all_y,
        label=true_label,
        color=get_model_color("true"),
        linewidth=2,
    )

    # Anchor all forecast curves at t0 with the selected plot target.
    y_t0 = env.x[t0_int] if plot_target == "x" else env.y[t0_int]
    times_forecast_plot = np.concatenate(([t0_int], times_vis))
    preds_oracle_plot = np.concatenate(([y_t0], preds_oracle_vis))
    preds_baseline_plot = np.concatenate(([y_t0], _shift_vis_preds(preds_baseline)))
    preds_partial_plot = np.concatenate(([y_t0], preds_partial_vis))
    preds_full_plot = np.concatenate(([y_t0], preds_full_vis))
    preds_factorial_partial_plot = (
        np.concatenate(([y_t0], preds_factorial_partial_vis))
        if preds_factorial_partial_vis.size > 0
        else None
    )
    preds_factorial_full_plot = (
        np.concatenate(([y_t0], preds_factorial_full_vis))
        if preds_factorial_full_vis.size > 0
        else None
    )
    preds_factorial_linear_partial_plot = (
        np.concatenate(([y_t0], preds_factorial_linear_partial_vis))
        if preds_factorial_linear_partial_vis.size > 0
        else None
    )
    preds_factorial_linear_full_plot = (
        np.concatenate(([y_t0], preds_factorial_linear_full_vis))
        if preds_factorial_linear_full_vis.size > 0
        else None
    )
    preds_partial_neural_plot = (
        np.concatenate(([y_t0], preds_partial_neural_vis))
        if preds_partial_neural_vis.size > 0
        else None
    )
    preds_full_neural_plot = (
        np.concatenate(([y_t0], preds_full_neural_vis))
        if preds_full_neural_vis.size > 0
        else None
    )
    preds_partial_corr_plot = (
        np.concatenate(([y_t0], preds_partial_corr_vis))
        if preds_partial_corr_vis.size > 0
        else None
    )
    preds_full_corr_plot = (
        np.concatenate(([y_t0], preds_full_corr_vis))
        if preds_full_corr_vis.size > 0
        else None
    )
    preds_partial_corr_em_plot = (
        np.concatenate(([y_t0], preds_partial_corr_em_vis))
        if preds_partial_corr_em_vis.size > 0
        else None
    )
    preds_full_corr_em_plot = (
        np.concatenate(([y_t0], preds_full_corr_em_vis))
        if preds_full_corr_em_vis.size > 0
        else None
    )
    preds_l2d_plot = (
        np.concatenate(([y_t0], preds_l2d_vis))
        if preds_l2d_vis.size > 0
        else None
    )
    preds_l2d_sw_plot = (
        np.concatenate(([y_t0], preds_l2d_sw_vis))
        if preds_l2d_sw_vis.size > 0
        else None
    )
    preds_linucb_partial_plot = (
        np.concatenate(([y_t0], preds_linucb_partial_vis))
        if preds_linucb_partial_vis.size > 0
        else None
    )
    preds_linucb_full_plot = (
        np.concatenate(([y_t0], preds_linucb_full_vis))
        if preds_linucb_full_vis.size > 0
        else None
    )
    preds_neuralucb_partial_plot = (
        np.concatenate(([y_t0], preds_neuralucb_partial_vis))
        if preds_neuralucb_partial_vis.size > 0
        else None
    )
    preds_neuralucb_full_plot = (
        np.concatenate(([y_t0], preds_neuralucb_full_vis))
        if preds_neuralucb_full_vis.size > 0
        else None
    )

    ax_h_full.plot(
        times_forecast_plot,
        preds_oracle_plot,
        label=oracle_label,
        linestyle="-",
        color=get_model_color("oracle"),
    )
    # Do not plot a dedicated best-constant curve; individual const experts are shown below.
    ax_h_full.plot(
        times_forecast_plot,
        preds_partial_plot,
        label=base_partial_label,
        color=get_model_color("partial"),
        alpha=0.8,
    )
    ax_h_full.plot(
        times_forecast_plot,
        preds_full_plot,
        label=base_full_label,
        color=get_model_color("full"),
        alpha=0.8,
    )
    if preds_partial_neural_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_partial_neural_plot,
            label=neural_partial_label,
            color=get_model_color("neural_partial"),
            alpha=0.8,
        )
    if preds_full_neural_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_full_neural_plot,
            label=neural_full_label,
            color=get_model_color("neural_full"),
            alpha=0.8,
        )
    if preds_partial_corr_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_partial_corr_plot,
            label=corr_partial_label,
            color=get_model_color("partial_corr"),
            alpha=0.8,
        )
    if preds_full_corr_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_full_corr_plot,
            label=corr_full_label,
            color=get_model_color("full_corr"),
            alpha=0.8,
        )
    if preds_partial_corr_em_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_partial_corr_em_plot,
            label=corr_em_partial_label,
            color=get_model_color("partial_corr_em"),
            alpha=0.8,
        )
    if preds_full_corr_em_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_full_corr_em_plot,
            label=corr_em_full_label,
            color=get_model_color("full_corr_em"),
            alpha=0.8,
        )
    if preds_factorial_partial_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_factorial_partial_plot,
            label=factorized_partial_label,
            color=get_model_color("factorized_partial"),
            alpha=0.8,
        )
    if preds_factorial_full_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_factorial_full_plot,
            label=factorized_full_label,
            color=get_model_color("factorized_full"),
            alpha=0.8,
        )
    if preds_factorial_linear_partial_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_factorial_linear_partial_plot,
            label=factorized_linear_partial_label,
            color=get_model_color("factorized_linear_partial"),
            alpha=0.8,
        )
    if preds_factorial_linear_full_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_factorial_linear_full_plot,
            label=factorized_linear_full_label,
            color=get_model_color("factorized_linear_full"),
            alpha=0.8,
        )
    # Plot constant-expert baselines (line + '*' markers for clarity)
    for j in range(env.num_experts):
        preds_j_plot = np.concatenate(([y_t0], const_preds_vis[j]))
        ax_h_full.plot(
            times_forecast_plot,
            preds_j_plot,
            color=get_expert_color(j),
            linestyle="--",
            marker="o",
            markerfacecolor="none",
            markersize=6,
            alpha=0.7,
            label=f"Always using expert {j}",
        )
    if preds_l2d_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_l2d_plot,
            label=l2d_label,
            color=get_model_color("l2d"),
            alpha=0.9,
        )
    if preds_l2d_sw_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_l2d_sw_plot,
            label=l2d_sw_label,
            color=get_model_color("l2d_sw"),
            alpha=0.9,
        )
    if preds_linucb_partial_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_linucb_partial_plot,
            label=linucb_partial_label,
            color=get_model_color("linucb_partial"),
            alpha=0.8,
        )
    if preds_linucb_full_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_linucb_full_plot,
            label=linucb_full_label,
            color=get_model_color("linucb_full"),
            alpha=0.8,
        )
    if preds_neuralucb_partial_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_neuralucb_partial_plot,
            label=neuralucb_partial_label,
            color=get_model_color("neuralucb"),
            alpha=0.8,
        )
    if preds_neuralucb_full_plot is not None:
        ax_h_full.plot(
            times_forecast_plot,
            preds_neuralucb_full_plot,
            label=neuralucb_full_label,
            color=get_model_color("neuralucb"),
            linestyle="--",
            alpha=0.8,
        )
    ax_h_full.axvline(t0_int, linestyle=":", color="k", alpha=0.5)
    ax_h_full.set_xlabel("Time $t$")
    ax_h_full.set_ylabel("Value")
    ax_h_full.set_title(f"Horizon-{H_eff} forecasts from t={t0_int} (full history)")
    ax_h_full.legend(loc="upper left")

    # --- Middle subplot: zoomed view t in [t0, t0+H_eff] ---
    zoom_times = np.concatenate(([t0_int], times_vis))
    zoom_y = np.concatenate(([y_t0], y_window_vis))
    ax_h_zoom.plot(
        zoom_times,
        zoom_y,
        label=true_label,
        color=get_model_color("true"),
        linewidth=2,
    )
    ax_h_zoom.plot(
        times_forecast_plot,
        preds_oracle_plot,
        label=oracle_label,
        linestyle="-",
        color=get_model_color("oracle"),
    )
    # Do not plot a dedicated best-constant curve in the zoom either.
    ax_h_zoom.plot(
        times_forecast_plot,
        preds_partial_plot,
        label=base_partial_label,
        color=get_model_color("partial"),
        alpha=0.8,
    )
    ax_h_zoom.plot(
        times_forecast_plot,
        preds_full_plot,
        label=base_full_label,
        color=get_model_color("full"),
        alpha=0.8,
    )
    if preds_partial_neural_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_partial_neural_plot,
            label=neural_partial_label,
            color=get_model_color("neural_partial"),
            alpha=0.8,
        )
    if preds_full_neural_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_full_neural_plot,
            label=neural_full_label,
            color=get_model_color("neural_full"),
            alpha=0.8,
        )
    if preds_partial_corr_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_partial_corr_plot,
            label=corr_partial_label,
            color=get_model_color("partial_corr"),
            alpha=0.8,
        )
    if preds_full_corr_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_full_corr_plot,
            label=corr_full_label,
            color=get_model_color("full_corr"),
            alpha=0.8,
        )
    if preds_partial_corr_em_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_partial_corr_em_plot,
            label=corr_em_partial_label,
            color=get_model_color("partial_corr_em"),
            alpha=0.8,
        )
    if preds_full_corr_em_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_full_corr_em_plot,
            label=corr_em_full_label,
            color=get_model_color("full_corr_em"),
            alpha=0.8,
        )
    if preds_factorial_partial_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_factorial_partial_plot,
            label=factorized_partial_label,
            color=get_model_color("factorized_partial"),
            alpha=0.8,
        )
    if preds_factorial_full_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_factorial_full_plot,
            label=factorized_full_label,
            color=get_model_color("factorized_full"),
            alpha=0.8,
        )
    if preds_factorial_linear_partial_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_factorial_linear_partial_plot,
            label=factorized_linear_partial_label,
            color=get_model_color("factorized_linear_partial"),
            alpha=0.8,
        )
    if preds_factorial_linear_full_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_factorial_linear_full_plot,
            label=factorized_linear_full_label,
            color=get_model_color("factorized_linear_full"),
            alpha=0.8,
        )
    for j in range(env.num_experts):
        preds_j_plot = np.concatenate(([y_t0], const_preds_vis[j]))
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_j_plot,
            color=get_expert_color(j),
            linestyle="--",
            marker="o",
            markerfacecolor="none",
            markersize=6,
            alpha=0.7,
            label=f"Always using expert {j}",
        )
    if preds_l2d_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_l2d_plot,
            label=l2d_label,
            color=get_model_color("l2d"),
            alpha=0.9,
        )
    if preds_l2d_sw_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_l2d_sw_plot,
            label=l2d_sw_label,
            color=get_model_color("l2d_sw"),
            alpha=0.9,
        )
    if preds_linucb_partial_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_linucb_partial_plot,
            label=linucb_partial_label,
            color=get_model_color("linucb_partial"),
            alpha=0.8,
        )
    if preds_linucb_full_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_linucb_full_plot,
            label=linucb_full_label,
            color=get_model_color("linucb_full"),
            alpha=0.8,
        )
    if preds_neuralucb_partial_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_neuralucb_partial_plot,
            label=neuralucb_partial_label,
            color=get_model_color("neuralucb"),
            alpha=0.8,
        )
    if preds_neuralucb_full_plot is not None:
        ax_h_zoom.plot(
            times_forecast_plot,
            preds_neuralucb_full_plot,
            label=neuralucb_full_label,
            color=get_model_color("neuralucb"),
            linestyle="--",
            alpha=0.8,
        )
    ax_h_zoom.axvline(t0_int, linestyle=":", color="k", alpha=0.5)
    ax_h_zoom.set_xlim(t0_int, t0_int + H_eff)
    ax_h_zoom.set_xlabel("Time $t$")
    ax_h_zoom.set_ylabel("Value")
    ax_h_zoom.set_title(f"Horizon-{H_eff} forecasts from t={t0_int} (zoom)")
    ax_h_zoom.legend(loc="upper left")

    # --- Bottom subplot: expert scheduling over the horizon ---
    ax_h_sched.step(
        times,
        sched_oracle,
        where="post",
        label=oracle_label,
        color=get_model_color("oracle"),
    )
    # Do not plot a separate schedule for the best constant expert; const schedules per expert follow.
    ax_h_sched.step(
        times,
        sched_partial,
        where="post",
        label=base_partial_label,
        color=get_model_color("partial"),
    )
    ax_h_sched.step(
        times,
        sched_full,
        where="post",
        label=base_full_label,
        color=get_model_color("full"),
    )
    if sched_factorial_partial:
        ax_h_sched.step(
            times,
            sched_factorial_partial,
            where="post",
            label=factorized_partial_label,
            color=get_model_color("factorized_partial"),
        )
    if sched_factorial_full:
        ax_h_sched.step(
            times,
            sched_factorial_full,
            where="post",
            label=factorized_full_label,
            color=get_model_color("factorized_full"),
        )
    if sched_factorial_linear_partial:
        ax_h_sched.step(
            times,
            sched_factorial_linear_partial,
            where="post",
            label=factorized_linear_partial_label,
            color=get_model_color("factorized_linear_partial"),
        )
    if sched_factorial_linear_full:
        ax_h_sched.step(
            times,
            sched_factorial_linear_full,
            where="post",
            label=factorized_linear_full_label,
            color=get_model_color("factorized_linear_full"),
        )
    if sched_partial_neural:
        ax_h_sched.step(
            times,
            sched_partial_neural,
            where="post",
            label=neural_partial_label,
            color=get_model_color("neural_partial"),
        )
    if sched_full_neural:
        ax_h_sched.step(
            times,
            sched_full_neural,
            where="post",
            label=neural_full_label,
            color=get_model_color("neural_full"),
        )
    if sched_partial_corr:
        ax_h_sched.step(
            times,
            sched_partial_corr,
            where="post",
            label=corr_partial_label,
            color=get_model_color("partial_corr"),
        )
    if sched_full_corr:
        ax_h_sched.step(
            times,
            sched_full_corr,
            where="post",
            label=corr_full_label,
            color=get_model_color("full_corr"),
        )
    if sched_partial_corr_em:
        ax_h_sched.step(
            times,
            sched_partial_corr_em,
            where="post",
            label=corr_em_partial_label,
            color=get_model_color("partial_corr_em"),
        )
    if sched_full_corr_em:
        ax_h_sched.step(
            times,
            sched_full_corr_em,
            where="post",
            label=corr_em_full_label,
            color=get_model_color("full_corr_em"),
        )
    if sched_l2d:
        ax_h_sched.step(
            times,
            sched_l2d,
            where="post",
            label=l2d_label,
            color=get_model_color("l2d"),
        )
    if sched_l2d_sw:
        ax_h_sched.step(
            times,
            sched_l2d_sw,
            where="post",
            label=l2d_sw_label,
            color=get_model_color("l2d_sw"),
        )
    if sched_linucb_partial:
        ax_h_sched.step(
            times,
            sched_linucb_partial,
            where="post",
            label=linucb_partial_label,
            color=get_model_color("linucb_partial"),
        )
    if sched_linucb_full:
        ax_h_sched.step(
            times,
            sched_linucb_full,
            where="post",
            label=linucb_full_label,
            color=get_model_color("linucb_full"),
        )
    if sched_neuralucb_partial:
        ax_h_sched.step(
            times,
            sched_neuralucb_partial,
            where="post",
            label=neuralucb_partial_label,
            color=get_model_color("neuralucb"),
        )
    if sched_neuralucb_full:
        ax_h_sched.step(
            times,
            sched_neuralucb_full,
            where="post",
            label=neuralucb_full_label,
            color=get_model_color("neuralucb"),
            linestyle="--",
        )
    # Constant-expert schedules
    for j in range(env.num_experts):
        sched_j = [j] * H_eff
        ax_h_sched.step(
            times,
            sched_j,
            where="post",
            color=get_expert_color(j),
            linestyle="--",
            alpha=0.5,
            label=f"Always using expert {j}",
        )
    ax_h_sched.set_xlabel("Time $t$")
    ax_h_sched.set_ylabel("Expert")
    ax_h_sched.set_yticks(np.arange(env.num_experts))
    ax_h_sched.set_xlim(t0_int, t0_int + H_eff)
    ax_h_sched.set_title("Expert scheduling over horizon")
    ax_h_sched.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # New plot: horizon forecasts with expert-colored points
    # --------------------------------------------------------
    fig_forecast, ax_forecast = plt.subplots(1, 1, figsize=(10, 5))

    # True series on the horizon window (line, model color)
    ax_forecast.plot(
        times_vis,
        y_window_vis,
        label=true_label,
        color=get_model_color("true"),
        linewidth=2,
    )

    # Helper to plot method lines with consistent model colors
    def plot_method_line(model_key: str, label: str, preds_arr: np.ndarray):
        if preds_arr.size == 0:
            return
        ax_forecast.plot(
            times_vis,
            preds_arr,
            label=label,
            color=get_model_color(model_key),
            linewidth=1.8,
        )

    # Lines for each method (colors coherent with other plots)
    plot_method_line("oracle", oracle_label, preds_oracle_vis)
    plot_method_line("partial", base_partial_label, preds_partial_vis)
    plot_method_line("full", base_full_label, preds_full_vis)
    if preds_factorial_partial_vis.size > 0:
        plot_method_line(
            "factorized_partial", factorized_partial_label, preds_factorial_partial_vis
        )
    if preds_factorial_full_vis.size > 0:
        plot_method_line(
            "factorized_full", factorized_full_label, preds_factorial_full_vis
        )
    if preds_factorial_linear_partial_vis.size > 0:
        plot_method_line(
            "factorized_linear_partial",
            factorized_linear_partial_label,
            preds_factorial_linear_partial_vis,
        )
    if preds_factorial_linear_full_vis.size > 0:
        plot_method_line(
            "factorized_linear_full",
            factorized_linear_full_label,
            preds_factorial_linear_full_vis,
        )
    if preds_partial_neural_vis.size > 0:
        plot_method_line(
            "neural_partial", neural_partial_label, preds_partial_neural_vis
        )
    if preds_full_neural_vis.size > 0:
        plot_method_line(
            "neural_full", neural_full_label, preds_full_neural_vis
        )
    if preds_partial_corr_vis.size > 0:
        plot_method_line("partial_corr", corr_partial_label, preds_partial_corr_vis)
    if preds_full_corr_vis.size > 0:
        plot_method_line("full_corr", corr_full_label, preds_full_corr_vis)
    if preds_partial_corr_em_vis.size > 0:
        plot_method_line(
            "partial_corr_em",
            corr_em_partial_label,
            preds_partial_corr_em_vis,
        )
    if preds_full_corr_em_vis.size > 0:
        plot_method_line(
            "full_corr_em",
            corr_em_full_label,
            preds_full_corr_em_vis,
        )
    if preds_l2d_vis.size > 0:
        plot_method_line("l2d", l2d_label, preds_l2d_vis)
    if preds_l2d_sw_vis.size > 0:
        plot_method_line("l2d_sw", l2d_sw_label, preds_l2d_sw_vis)
    if preds_linucb_partial_vis.size > 0:
        plot_method_line(
            "linucb_partial", linucb_partial_label, preds_linucb_partial_vis
        )
    if preds_linucb_full_vis.size > 0:
        plot_method_line("linucb_full", linucb_full_label, preds_linucb_full_vis)
    if preds_neuralucb_partial_vis.size > 0:
        plot_method_line(
            "neuralucb", neuralucb_partial_label, preds_neuralucb_partial_vis
        )
    if preds_neuralucb_full_vis.size > 0:
        plot_method_line(
            "neuralucb", neuralucb_full_label, preds_neuralucb_full_vis
        )

    # Helper to scatter predictions with color = expert, marker = method
    method_markers = {
        oracle_label: "o",
        base_partial_label: "s",
        base_full_label: "^",
        factorized_partial_label: "s",
        factorized_full_label: "^",
        factorized_linear_partial_label: "s",
        factorized_linear_full_label: "^",
        neural_partial_label: "P",
        neural_full_label: "*",
        corr_partial_label: "D",
        corr_full_label: "v",
        corr_em_partial_label: "D",
        corr_em_full_label: "^",
        l2d_label: "x",
        l2d_sw_label: "x",
        linucb_partial_label: "o",
        linucb_full_label: "o",
        neuralucb_partial_label: "P",
        neuralucb_full_label: "P",
    }

    sched_oracle_vis = _shift_vis_schedule(sched_oracle)
    sched_partial_vis = _shift_vis_schedule(sched_partial)
    sched_full_vis = _shift_vis_schedule(sched_full)
    sched_factorial_partial_vis = _shift_vis_schedule(sched_factorial_partial)
    sched_factorial_full_vis = _shift_vis_schedule(sched_factorial_full)
    sched_factorial_linear_partial_vis = _shift_vis_schedule(
        sched_factorial_linear_partial
    )
    sched_factorial_linear_full_vis = _shift_vis_schedule(
        sched_factorial_linear_full
    )
    sched_partial_neural_vis = _shift_vis_schedule(sched_partial_neural)
    sched_full_neural_vis = _shift_vis_schedule(sched_full_neural)
    sched_partial_corr_vis = _shift_vis_schedule(sched_partial_corr)
    sched_full_corr_vis = _shift_vis_schedule(sched_full_corr)
    sched_partial_corr_em_vis = _shift_vis_schedule(sched_partial_corr_em)
    sched_full_corr_em_vis = _shift_vis_schedule(sched_full_corr_em)
    sched_l2d_vis = _shift_vis_schedule(sched_l2d)
    sched_l2d_sw_vis = _shift_vis_schedule(sched_l2d_sw)
    sched_linucb_partial_vis = _shift_vis_schedule(sched_linucb_partial)
    sched_linucb_full_vis = _shift_vis_schedule(sched_linucb_full)
    sched_neuralucb_partial_vis = _shift_vis_schedule(sched_neuralucb_partial)
    sched_neuralucb_full_vis = _shift_vis_schedule(sched_neuralucb_full)

    def scatter_method(label: str, preds_arr, sched_arr):
        if preds_arr.size == 0 or len(sched_arr) == 0:
            return
        times_arr = np.asarray(times_vis, dtype=int)
        preds_arr = np.asarray(preds_arr, dtype=float)
        sched_arr = np.asarray(sched_arr, dtype=int)
        marker = method_markers.get(label, "o")
        for j in range(env.num_experts):
            mask = sched_arr == j
            if not np.any(mask):
                continue
            ax_forecast.scatter(
                times_arr[mask],
                preds_arr[mask],
                color=get_expert_color(j),
                marker=marker,
                alpha=0.8,
            )

    # Scatter forecasts for each method (points colored by expert)
    scatter_method(oracle_label, preds_oracle_vis, sched_oracle_vis)
    scatter_method(
        base_partial_label, preds_partial_vis, sched_partial_vis
    )
    scatter_method(base_full_label, preds_full_vis, sched_full_vis)
    if preds_factorial_partial_vis.size > 0 and sched_factorial_partial_vis:
        scatter_method(
            factorized_partial_label,
            preds_factorial_partial_vis,
            sched_factorial_partial_vis,
        )
    if preds_factorial_full_vis.size > 0 and sched_factorial_full_vis:
        scatter_method(
            factorized_full_label,
            preds_factorial_full_vis,
            sched_factorial_full_vis,
        )
    if (
        preds_factorial_linear_partial_vis.size > 0
        and sched_factorial_linear_partial_vis
    ):
        scatter_method(
            factorized_linear_partial_label,
            preds_factorial_linear_partial_vis,
            sched_factorial_linear_partial_vis,
        )
    if preds_factorial_linear_full_vis.size > 0 and sched_factorial_linear_full_vis:
        scatter_method(
            factorized_linear_full_label,
            preds_factorial_linear_full_vis,
            sched_factorial_linear_full_vis,
        )
    if preds_partial_neural_vis.size > 0 and sched_partial_neural_vis:
        scatter_method(
            neural_partial_label,
            preds_partial_neural_vis,
            sched_partial_neural_vis,
        )
    if preds_full_neural_vis.size > 0 and sched_full_neural_vis:
        scatter_method(
            neural_full_label,
            preds_full_neural_vis,
            sched_full_neural_vis,
        )
    if preds_partial_corr_vis.size > 0 and sched_partial_corr_vis:
        scatter_method(corr_partial_label, preds_partial_corr_vis, sched_partial_corr_vis)
    if preds_full_corr_vis.size > 0 and sched_full_corr_vis:
        scatter_method(corr_full_label, preds_full_corr_vis, sched_full_corr_vis)
    if preds_partial_corr_em_vis.size > 0 and sched_partial_corr_em_vis:
        scatter_method(
            corr_em_partial_label,
            preds_partial_corr_em_vis,
            sched_partial_corr_em_vis,
        )
    if preds_full_corr_em_vis.size > 0 and sched_full_corr_em_vis:
        scatter_method(
            corr_em_full_label,
            preds_full_corr_em_vis,
            sched_full_corr_em_vis,
        )
    if preds_l2d_vis.size > 0 and sched_l2d_vis:
        scatter_method(l2d_label, preds_l2d_vis, sched_l2d_vis)
    if preds_l2d_sw_vis.size > 0 and sched_l2d_sw_vis:
        scatter_method(l2d_sw_label, preds_l2d_sw_vis, sched_l2d_sw_vis)
    if preds_linucb_partial_vis.size > 0 and sched_linucb_partial_vis:
        scatter_method(
            linucb_partial_label, preds_linucb_partial_vis, sched_linucb_partial_vis
        )
    if preds_linucb_full_vis.size > 0 and sched_linucb_full_vis:
        scatter_method(linucb_full_label, preds_linucb_full_vis, sched_linucb_full_vis)
    if preds_neuralucb_partial_vis.size > 0 and sched_neuralucb_partial_vis:
        scatter_method(
            neuralucb_partial_label,
            preds_neuralucb_partial_vis,
            sched_neuralucb_partial_vis,
        )
    if preds_neuralucb_full_vis.size > 0 and sched_neuralucb_full_vis:
        scatter_method(
            neuralucb_full_label,
            preds_neuralucb_full_vis,
            sched_neuralucb_full_vis,
        )

    # Legends: methods (line+marker, colored by model) and experts (color)
    method_handles = []
    legend_specs = [
        ("true", true_label),
        ("oracle", oracle_label),
        ("partial", base_partial_label),
        ("full", base_full_label),
    ]
    if preds_factorial_partial_vis.size > 0:
        legend_specs.append(("factorized_partial", factorized_partial_label))
    if preds_factorial_full_vis.size > 0:
        legend_specs.append(("factorized_full", factorized_full_label))
    if preds_factorial_linear_partial_vis.size > 0:
        legend_specs.append(("factorized_linear_partial", factorized_linear_partial_label))
    if preds_factorial_linear_full_vis.size > 0:
        legend_specs.append(("factorized_linear_full", factorized_linear_full_label))
    if preds_partial_neural_vis.size > 0:
        legend_specs.append(("neural_partial", neural_partial_label))
    if preds_full_neural_vis.size > 0:
        legend_specs.append(("neural_full", neural_full_label))
    if preds_partial_corr_vis.size > 0:
        legend_specs.append(("partial_corr", corr_partial_label))
    if preds_full_corr_vis.size > 0:
        legend_specs.append(("full_corr", corr_full_label))
    if preds_partial_corr_em_vis.size > 0:
        legend_specs.append(("partial_corr_em", corr_em_partial_label))
    if preds_full_corr_em_vis.size > 0:
        legend_specs.append(("full_corr_em", corr_em_full_label))
    if preds_l2d_vis.size > 0:
        legend_specs.append(("l2d", l2d_label))
    if preds_l2d_sw_vis.size > 0:
        legend_specs.append(("l2d_sw", l2d_sw_label))
    if preds_linucb_partial_vis.size > 0:
        legend_specs.append(("linucb_partial", linucb_partial_label))
    if preds_linucb_full_vis.size > 0:
        legend_specs.append(("linucb_full", linucb_full_label))
    if preds_neuralucb_partial_vis.size > 0:
        legend_specs.append(("neuralucb", neuralucb_partial_label))
    if preds_neuralucb_full_vis.size > 0:
        legend_specs.append(("neuralucb", neuralucb_full_label))
    for key, label in legend_specs:
        color = get_model_color(key)
        marker = "" if key == "true" else method_markers.get(label, "o")
        handle = mlines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="-",
            label=label,
        )
        method_handles.append(handle)
    expert_handles = [
        mpatches.Patch(color=get_expert_color(j), label=f"Expert {j}")
        for j in range(env.num_experts)
    ]

    legend_methods = ax_forecast.legend(
        handles=method_handles,
        title="Methods",
        loc="upper left",
    )
    ax_forecast.add_artist(legend_methods)
    ax_forecast.legend(
        handles=expert_handles,
        title="Experts",
        loc="upper right",
    )

    ax_forecast.set_xlabel("Time $t$")
    ax_forecast.set_ylabel(f"Forecast / {true_label}")
    ax_forecast.set_title(
        f"Horizon-{H_eff} forecasts from t={t0_int} (all baselines)"
    )
    plt.tight_layout()
    plt.show()
