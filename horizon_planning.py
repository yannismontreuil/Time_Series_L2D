import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Sequence, Tuple, Optional

from router_model import SLDSIMMRouter
from router_model_corr import SLDSIMMRouter_Corr
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import L2D
from plot_utils import get_expert_color, get_model_color
from matplotlib import lines as mlines, patches as mpatches


def _simulate_value_scenarios_for_schedule(
    env: SyntheticTimeSeriesEnv,
    schedule: Sequence[int],
    times: np.ndarray,
    num_scenarios: int,
    scenario_generator_cfg: Optional[dict],
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

    cfg = scenario_generator_cfg or {}
    rho = float(cfg.get("rho", 0.0))
    sigma0 = float(cfg.get("sigma0", 0.0))
    q_scale = float(cfg.get("q_scale", 0.0))

    # Baseline context path x_hat_{t+h}.
    x_hat = np.zeros(H_eff, dtype=float)
    for h in range(H_eff):
        x_hat[h] = float(env.get_context(int(times[h]))[0])

    values_scen = np.zeros((N_scen, H_eff), dtype=float)
    rng = np.random.default_rng()

    for n in range(N_scen):
        # AR(1) residual for scalar context.
        if sigma0 > 0.0:
            xi_prev = float(rng.normal(loc=0.0, scale=sigma0))
        else:
            xi_prev = 0.0
        for h in range(H_eff):
            if h == 0:
                xi_h = xi_prev
            else:
                if q_scale > 0.0:
                    eta_h = float(rng.normal(loc=0.0, scale=q_scale))
                else:
                    eta_h = 0.0
                xi_h = rho * xi_prev + eta_h
                xi_prev = xi_h
            x_scen = x_hat[h] + xi_h
            j_h = int(schedule[h])
            values_scen[n, h] = env.expert_predict(j_h, np.array([x_scen], dtype=float))

    return values_scen


def warm_start_router_to_time(
    router,
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
        available_t = env.get_available_experts(t)
        r_t = baseline.select_expert(x_t, available_t)
        loss_all = env.losses(t)
        baseline.update(x_t, loss_all, available_t)


def _plan_horizon_schedule_monte_carlo(
    router,
    env: SyntheticTimeSeriesEnv,
    beta: np.ndarray,
    times: np.ndarray,
    avail_per_h: List[np.ndarray],
    num_scenarios: int,
    scenario_generator_cfg: Optional[dict],
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    N-scenario Monte Carlo / scenario-based planning as in
    Section "Extension: Staffing/Availability Planning over a Horizon":
      - open-loop IMM prediction (no future measurements),
      - context treated as exogenous (here, scenarios are taken from
        env.get_context at the true future times),
      - risk-adjusted scores J_{j,h}(x) built from mixture mean/variance,
      - planning-time scores c_h(j) ≈ (1/N) Σ_n J_{j,h}(x_{t+h}^{(n)}),
      - time-separable trajectory with capacity m=1:
            j_h^* ∈ argmin_j c_h(j) over available experts,
      - probability tables:
            p_avail[j,h]  ≈ P(j is best among available A_{t+h}),
            p_all[j,h]    ≈ P(j is best if all experts are available).

    The current implementation uses the realized future context
    x_{t+h} = env.get_context(t+h) for all scenarios n=1..N; if a more
    sophisticated scenario generator is available, it can be wired in
    here without changing the rest of the logic.
    """
    H_eff = int(times.shape[0])
    if H_eff == 0:
        empty = np.zeros((0, env.num_experts), dtype=float)
        return [], empty, empty, empty

    N_scen = int(num_scenarios)
    if N_scen <= 0:
        raise ValueError("num_scenarios must be a positive integer.")

    # Open-loop IMM prediction: (b_{t+h|t}, m_{t+h|t}, P_{t+h|t})
    b_list, m_list, P_list = router.precompute_horizon_states(H_eff)

    # Determine feature dimension for the emission feature φ.
    if isinstance(router, SLDSIMMRouter_Corr):
        d_feat = router.du
    else:
        d_feat = router.d

    # Baseline feature forecasts φ̂_{t+h|t}. For simplicity, we take
    # the realized future context as the baseline path and map it to
    # features via the router's emission feature map.
    phi_hat = np.zeros((H_eff, d_feat), dtype=float)
    for h in range(H_eff):
        t_future = int(times[h])
        x_future = env.get_context(t_future)
        if isinstance(router, SLDSIMMRouter_Corr):
            phi_h, _ = router._compute_feature(x_future)
            phi_hat[h] = np.asarray(phi_h, dtype=float).reshape(d_feat)
        else:
            phi_h = router.feature_fn(x_future)
            phi_hat[h] = np.asarray(phi_h, dtype=float).reshape(d_feat)

    # Parse Gaussian AR(1) scenario generator parameters.
    cfg = scenario_generator_cfg or {}
    gen_type = str(cfg.get("type", "gaussian_ar1")).lower()
    if gen_type != "gaussian_ar1":
        raise ValueError(
            f"Unsupported scenario generator type '{gen_type}'. "
            "Expected 'gaussian_ar1' as in the Gaussian perturbation generator."
        )
    rho = float(cfg.get("rho", 0.0))
    sigma0 = float(cfg.get("sigma0", 0.0))
    q_scale = float(cfg.get("q_scale", 0.0))

    # Scenario scores J_{j,h}(x_{t+h}^{(n)}) with shape (N_scen, H_eff, N_experts)
    N_experts = router.N
    J_scen = np.zeros((N_scen, H_eff, N_experts), dtype=float)

    schedule: List[int] = []
    # For probability tables:
    #   p_avail[h, j] ≈ P(j is best among available A_{t+h}),
    #   p_all[h, j]   ≈ P(j is best among all experts),
    # with tie mass split uniformly among minimizers.
    p_avail = np.zeros((H_eff, N_experts), dtype=float)
    p_all = np.zeros((H_eff, N_experts), dtype=float)

    rng = np.random.default_rng()

    # Evaluate scenario scores with Gaussian AR(1) feature perturbations.
    for n in range(N_scen):
        # Sample AR(1) residual path ξ^{(n)}_{1:H_eff} in feature space.
        if sigma0 > 0.0:
            xi_prev = rng.normal(loc=0.0, scale=sigma0, size=d_feat)
        else:
            xi_prev = np.zeros(d_feat, dtype=float)

        for h in range(H_eff):
            if h == 0:
                xi_h = xi_prev
            else:
                if q_scale > 0.0:
                    eta_h = rng.normal(loc=0.0, scale=q_scale, size=d_feat)
                else:
                    eta_h = np.zeros(d_feat, dtype=float)
                xi_h = rho * xi_prev + eta_h
                xi_prev = xi_h

            # Feature scenario at step h: φ^{(n)}_{t+h} = φ̂_{t+h|t} + ξ^{(n)}_h
            phi_future = phi_hat[h] + xi_h

            b_h = b_list[h]
            m_h = m_list[h]
            P_h = P_list[h]

            if isinstance(router, SLDSIMMRouter_Corr):
                # Correlated router: use its loss-distribution helper.
                mean_ell, var_ell, _, _ = router._predict_loss_distribution(
                    phi_future, b_h, m_h, P_h
                )
            else:
                # Independent-expert SLDS router.
                phi_future_vec = np.asarray(phi_future, dtype=float).reshape(router.d)
                M = router.M
                mu_k = np.zeros((M, router.N), dtype=float)
                S_k = np.zeros((M, router.N), dtype=float)
                for k in range(M):
                    for j in range(router.N):
                        m_kj = m_h[k, j]
                        P_kj = P_h[k, j]
                        mu = float(phi_future_vec @ m_kj)
                        S_val = float(phi_future_vec @ (P_kj @ phi_future_vec) + router.R[k, j])
                        mu_k[k, j] = mu
                        S_k[k, j] = max(S_val, router.eps)
                mean_ell = (b_h.reshape(-1, 1) * mu_k).sum(axis=0)
                var_within = (b_h.reshape(-1, 1) * S_k).sum(axis=0)
                diff = mu_k - mean_ell
                var_between = (b_h.reshape(-1, 1) * (diff**2)).sum(axis=0)
                var_ell = np.maximum(var_within + var_between, router.eps)

            # Planning-time risk parameter: risk-neutral (λ_plan = 0),
            # independent of the router's online λ used for routing.
            scores = mean_ell + beta
            J_scen[n, h, :] = scores

    # Planning-time scores: c_h(j) ≈ 1/N Σ_n J_{j,h}(x_{t+h}^{(n)}).
    c_hat = J_scen.mean(axis=0)  # shape (H_eff, N)

    # Build trajectory j_h^* and probability tables.
    for h in range(H_eff):
        avail = np.asarray(avail_per_h[h], dtype=int)
        if avail.size == 0:
            avail = np.arange(N_experts, dtype=int)

        # Trajectory (capacity m=1) based on c_hat.
        scores_avail = c_hat[h, avail]
        j_star = int(avail[int(np.argmin(scores_avail))])
        schedule.append(j_star)

        # Probability tables from scenario-wise argmins.
        for n in range(N_scen):
            scores_n = J_scen[n, h, :]

            # Best-over-all probability p_all.
            min_all = float(np.min(scores_n))
            all_min_indices = np.flatnonzero(np.isclose(scores_n, min_all))
            if all_min_indices.size > 0:
                mass_all = 1.0 / (N_scen * float(all_min_indices.size))
                p_all[h, all_min_indices] += mass_all

            # Best-over-available probability p_avail (A_{t+h}).
            scores_av = scores_n[avail]
            min_av = float(np.min(scores_av))
            av_min_local = np.flatnonzero(np.isclose(scores_av, min_av))
            if av_min_local.size > 0:
                mass_av = 1.0 / (N_scen * float(av_min_local.size))
                p_avail[h, avail[av_min_local]] += mass_av

    # Scenario-wise scores along the planned trajectory:
    # J_sched[n, h] = J_{j_h^*, h}(x^{(n)}_{t+h}).
    if H_eff > 0:
        schedule_arr = np.asarray(schedule, dtype=int).reshape(1, H_eff)
        h_idx = np.arange(H_eff, dtype=int).reshape(1, H_eff)
        n_idx = np.arange(N_scen, dtype=int).reshape(N_scen, 1)
        J_sched = J_scen[n_idx, h_idx, schedule_arr]
    else:
        J_sched = np.zeros((N_scen, 0), dtype=float)

    return schedule, p_avail, p_all, J_sched


def evaluate_horizon_planning(
    env: SyntheticTimeSeriesEnv,
    router_partial: SLDSIMMRouter,
    router_full: SLDSIMMRouter,
    beta: np.ndarray,
    t0: int,
    H: int,
    experts_predict: Sequence[Callable[[np.ndarray], float]],
    context_update: Callable[[np.ndarray, float], np.ndarray],
    l2d_baseline: Optional[L2D] = None,
    router_partial_corr: Optional[SLDSIMMRouter_Corr] = None,
    router_full_corr: Optional[SLDSIMMRouter_Corr] = None,
    router_partial_neural=None,
    router_full_neural=None,
    planning_method: str = "regressive",
    scenario_generator_cfg: Optional[dict] = None,
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
    warm_start_router_to_time(router_partial, env, t0)
    warm_start_router_to_time(router_full, env, t0)
    if router_partial_corr is not None:
        warm_start_router_to_time(router_partial_corr, env, t0)
    if router_full_corr is not None:
        warm_start_router_to_time(router_full_corr, env, t0)
    if router_partial_neural is not None:
        warm_start_router_to_time(router_partial_neural, env, t0)
    if router_full_neural is not None:
        warm_start_router_to_time(router_full_neural, env, t0)

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

    print(
        f"\n[Horizon planning] method = {raw_method}, "
        f"parsed_mode = {mode}, num_scenarios = {num_scenarios}, "
        f"H_eff = {H_eff}, t0 = {int(t0)}"
    )

    # Horizon-H planning schedules from current beliefs
    t0_int = int(t0)
    x_now = env.get_context(t0_int)
    avail_lists = [a.tolist() for a in avail_per_h]

    if mode == "regressive":
        # Original regressive / router-influenced-context planning.
        sched_partial, _, _ = router_partial.plan_horizon_schedule(
            x_t=x_now,
            H=H_eff,
            experts_predict=experts_predict,
            context_update=context_update,
            available_experts_per_h=avail_lists,
        )
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
    else:
        # Scenario-based Monte Carlo planning with exogenous contexts:
        # Section \ref{sec:staffing-mc} and Algorithm~\ref{alg:staffing-planning}.
        sched_partial, p_avail_partial, p_all_partial, J_sched_partial = _plan_horizon_schedule_monte_carlo(
            router_partial,
            env,
            beta,
            times,
            avail_per_h,
            num_scenarios,
            scenario_generator_cfg,
        )
        sched_full, p_avail_full, p_all_full, J_sched_full = _plan_horizon_schedule_monte_carlo(
            router_full,
            env,
            beta,
            times,
            avail_per_h,
            num_scenarios,
            scenario_generator_cfg,
        )

        if router_partial_corr is not None:
            sched_partial_corr, p_avail_partial_corr, p_all_partial_corr, J_sched_partial_corr = (
                _plan_horizon_schedule_monte_carlo(
                    router_partial_corr,
                    env,
                    beta,
                    times,
                    avail_per_h,
                    num_scenarios,
                    scenario_generator_cfg,
                )
            )
        else:
            sched_partial_corr = []
            p_avail_partial_corr = np.zeros((H_eff, env.num_experts), dtype=float)
            p_all_partial_corr = np.zeros((H_eff, env.num_experts), dtype=float)
            J_sched_partial_corr = np.zeros((num_scenarios, H_eff), dtype=float)

        if router_full_corr is not None:
            sched_full_corr, p_avail_full_corr, p_all_full_corr, J_sched_full_corr = (
                _plan_horizon_schedule_monte_carlo(
                    router_full_corr,
                    env,
                    beta,
                    times,
                    avail_per_h,
                    num_scenarios,
                    scenario_generator_cfg,
                )
            )
        else:
            sched_full_corr = []
            p_avail_full_corr = np.zeros((H_eff, env.num_experts), dtype=float)
            p_all_full_corr = np.zeros((H_eff, env.num_experts), dtype=float)
            J_sched_full_corr = np.zeros((num_scenarios, H_eff), dtype=float)

        # For neural routers we do not have a generative SLDS model to
        # support IMM-based staffing planning, so we skip Monte Carlo
        # planning for them in this mode.
        sched_partial_neural = []
        sched_full_neural = []

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
    if cost_partial_neural_plan.size > 0:
        print(
            f"H-plan (neural partial router): {cost_partial_neural_plan.mean():.4f}"
        )
    if cost_full_neural_plan.size > 0:
        print(
            f"H-plan (neural full router):    {cost_full_neural_plan.mean():.4f}"
        )
    if cost_partial_corr_plan.size > 0:
        print(
            f"H-plan (partial corr router): {cost_partial_corr_plan.mean():.4f}"
        )
    if cost_full_corr_plan.size > 0:
        print(f"H-plan (full corr router):    {cost_full_corr_plan.mean():.4f}")

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
            (p_all_partial, "Router partial"),
            (p_all_full, "Router full"),
            (p_all_partial_corr, "Router corr partial"),
            (p_all_full_corr, "Router corr full"),
        ]

        print("\n[Monte Carlo] Best-over-all selection probability tables (first 3 steps):")
        for mat, label in prob_tables:
            if mat.size == 0:
                continue
            h_max = min(3, mat.shape[0])
            print(f"\n  {label} (p_all, steps 1..{h_max}):")
            print(mat[:h_max, :])
            plot_prob_heatmap(mat, f"Best-over-all selection probability ({label})")

        # ------------------------------------------------------------------
        # New: truth y_t and correlated-router forecasts with MC value bands
        # ------------------------------------------------------------------
        if cost_full_corr_plan.size > 0 or cost_partial_corr_plan.size > 0:
            fig_corr, ax_y = plt.subplots(1, 1, figsize=(10, 4))

            # True series on the horizon window.
            ax_y.plot(
                times,
                y_window,
                label="True $y_t$",
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
                    label="H-plan corr full (forecast)",
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
                    label="H-plan corr partial (forecast)",
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
                    label="Oracle (schedule forecast)",
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
                label="Oracle schedule",
                color=get_model_color("oracle"),
                linewidth=2,
                linestyle=":",
            )
            if sched_full_corr:
                ax_sel.step(
                    times,
                    sched_full_corr,
                    where="post",
                    label="Corr full schedule",
                    color=get_model_color("full_corr"),
                    linewidth=2,
                )
            if sched_partial_corr:
                ax_sel.step(
                    times,
                    sched_partial_corr,
                    where="post",
                    label="Corr partial schedule",
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

            scatter_expert_support(
                p_all_full_corr,
                "full_corr",
                "Corr full (experts possible under noise)",
                marker="o",
            )
            scatter_expert_support(
                p_all_partial_corr,
                "partial_corr",
                "Corr partial (experts possible under noise)",
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
            if p_all_full_corr.size > 0:
                n_full = (p_all_full_corr > eps_prob).sum(axis=1)
                ax_count.plot(
                    times,
                    n_full,
                    color=get_model_color("full_corr"),
                    linestyle=":",
                    marker="o",
                    markerfacecolor="none",
                    linewidth=1.5,
                    label="Corr full (num experts possible)",
                )
            if p_all_partial_corr.size > 0:
                n_part = (p_all_partial_corr > eps_prob).sum(axis=1)
                ax_count.plot(
                    times,
                    n_part,
                    color=get_model_color("partial_corr"),
                    linestyle=":",
                    marker="o",
                    markerfacecolor="none",
                    linewidth=1.5,
                    label="Corr partial (num experts possible)",
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
    all_y = env.y[: t0_int + 1 + H_eff]
    ax_h_full.plot(
        all_times,
        all_y,
        label="True $y_t$",
        color=get_model_color("true"),
        linewidth=2,
    )

    # Anchor all forecast curves at t0 with the ground-truth value y_{t0}
    y_t0 = env.y[t0_int]
    times_forecast = np.concatenate(([t0_int], times))
    preds_oracle_plot = np.concatenate(([y_t0], preds_oracle))
    preds_baseline_plot = np.concatenate(([y_t0], preds_baseline))
    preds_partial_plot = np.concatenate(([y_t0], preds_partial_plan))
    preds_full_plot = np.concatenate(([y_t0], preds_full_plan))
    preds_partial_neural_plot = (
        np.concatenate(([y_t0], preds_partial_neural_plan))
        if preds_partial_neural_plan.size > 0
        else None
    )
    preds_full_neural_plot = (
        np.concatenate(([y_t0], preds_full_neural_plan))
        if preds_full_neural_plan.size > 0
        else None
    )
    preds_partial_corr_plot = (
        np.concatenate(([y_t0], preds_partial_corr_plan))
        if preds_partial_corr_plan.size > 0
        else None
    )
    preds_full_corr_plot = (
        np.concatenate(([y_t0], preds_full_corr_plan))
        if preds_full_corr_plan.size > 0
        else None
    )

    ax_h_full.plot(
        times_forecast,
        preds_oracle_plot,
        label="Oracle (truth)",
        linestyle="-",
        color=get_model_color("oracle"),
    )
    # Do not plot a dedicated best-constant curve; individual const experts are shown below.
    ax_h_full.plot(
        times_forecast,
        preds_partial_plot,
        label="H-plan partial",
        color=get_model_color("partial"),
        alpha=0.8,
    )
    ax_h_full.plot(
        times_forecast,
        preds_full_plot,
        label="H-plan full",
        color=get_model_color("full"),
        alpha=0.8,
    )
    if preds_partial_neural_plot is not None:
        ax_h_full.plot(
            times_forecast,
            preds_partial_neural_plot,
            label="H-plan neural partial",
            color=get_model_color("neural_partial"),
            alpha=0.8,
        )
    if preds_full_neural_plot is not None:
        ax_h_full.plot(
            times_forecast,
            preds_full_neural_plot,
            label="H-plan neural full",
            color=get_model_color("neural_full"),
            alpha=0.8,
        )
    if preds_partial_corr_plot is not None:
        ax_h_full.plot(
            times_forecast,
            preds_partial_corr_plot,
            label="H-plan partial corr",
            color=get_model_color("partial_corr"),
            alpha=0.8,
        )
    if preds_full_corr_plot is not None:
        ax_h_full.plot(
            times_forecast,
            preds_full_corr_plot,
            label="H-plan full corr",
            color=get_model_color("full_corr"),
            alpha=0.8,
        )
    # Plot constant-expert baselines (line + '*' markers for clarity)
    for j in range(env.num_experts):
        preds_j_plot = np.concatenate(([y_t0], const_preds[j]))
        ax_h_full.plot(
            times_forecast,
            preds_j_plot,
            color=get_expert_color(j),
            linestyle="--",
            marker="o",
            markerfacecolor="none",
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
            color=get_model_color("l2d"),
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
    ax_h_zoom.plot(
        zoom_times,
        zoom_y,
        label="True $y_t$",
        color=get_model_color("true"),
        linewidth=2,
    )
    ax_h_zoom.plot(
        times_forecast,
        preds_oracle_plot,
        label="Oracle (truth)",
        linestyle="-",
        color=get_model_color("oracle"),
    )
    # Do not plot a dedicated best-constant curve in the zoom either.
    ax_h_zoom.plot(
        times_forecast,
        preds_partial_plot,
        label="H-plan partial",
        color=get_model_color("partial"),
        alpha=0.8,
    )
    ax_h_zoom.plot(
        times_forecast,
        preds_full_plot,
        label="H-plan full",
        color=get_model_color("full"),
        alpha=0.8,
    )
    if preds_partial_neural_plot is not None:
        ax_h_zoom.plot(
            times_forecast,
            preds_partial_neural_plot,
            label="H-plan neural partial",
            color=get_model_color("neural_partial"),
            alpha=0.8,
        )
    if preds_full_neural_plot is not None:
        ax_h_zoom.plot(
            times_forecast,
            preds_full_neural_plot,
            label="H-plan neural full",
            color=get_model_color("neural_full"),
            alpha=0.8,
        )
    if preds_partial_corr_plot is not None:
        ax_h_zoom.plot(
            times_forecast,
            preds_partial_corr_plot,
            label="H-plan partial corr",
            color=get_model_color("partial_corr"),
            alpha=0.8,
        )
    if preds_full_corr_plot is not None:
        ax_h_zoom.plot(
            times_forecast,
            preds_full_corr_plot,
            label="H-plan full corr",
            color=get_model_color("full_corr"),
            alpha=0.8,
        )
    for j in range(env.num_experts):
        preds_j_plot = np.concatenate(([y_t0], const_preds[j]))
        ax_h_zoom.plot(
            times_forecast,
            preds_j_plot,
            color=get_expert_color(j),
            linestyle="--",
            marker="o",
            markerfacecolor="none",
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
            color=get_model_color("l2d"),
            alpha=0.9,
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
        label="Oracle",
        color=get_model_color("oracle"),
    )
    # Do not plot a separate schedule for the best constant expert; const schedules per expert follow.
    ax_h_sched.step(
        times,
        sched_partial,
        where="post",
        label="H-plan partial",
        color=get_model_color("partial"),
    )
    ax_h_sched.step(
        times,
        sched_full,
        where="post",
        label="H-plan full",
        color=get_model_color("full"),
    )
    if sched_partial_neural:
        ax_h_sched.step(
            times,
            sched_partial_neural,
            where="post",
            label="H-plan neural partial",
            color=get_model_color("neural_partial"),
        )
    if sched_full_neural:
        ax_h_sched.step(
            times,
            sched_full_neural,
            where="post",
            label="H-plan neural full",
            color=get_model_color("neural_full"),
        )
    if sched_partial_corr:
        ax_h_sched.step(
            times,
            sched_partial_corr,
            where="post",
            label="H-plan partial corr",
            color=get_model_color("partial_corr"),
        )
    if sched_full_corr:
        ax_h_sched.step(
            times,
            sched_full_corr,
            where="post",
            label="H-plan full corr",
            color=get_model_color("full_corr"),
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
            label=f"Const {j}",
        )
    # L2D schedule
    if l2d_baseline is not None and sched_l2d:
        ax_h_sched.step(
            times,
            sched_l2d,
            where="post",
            label="L2D baseline",
            color=get_model_color("l2d"),
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
        times,
        y_window,
        label="True $y_t$",
        color=get_model_color("true"),
        linewidth=2,
    )

    # Helper to plot method lines with consistent model colors
    def plot_method_line(model_key: str, label: str, preds_arr: np.ndarray):
        if preds_arr.size == 0:
            return
        ax_forecast.plot(
            times,
            preds_arr,
            label=label,
            color=get_model_color(model_key),
            linewidth=1.8,
        )

    # Lines for each method (colors coherent with other plots)
    plot_method_line("oracle", "Oracle", preds_oracle)
    plot_method_line("partial", "Router (partial)", preds_partial_plan)
    plot_method_line("full", "Router (full)", preds_full_plan)
    if preds_partial_neural_plan.size > 0:
        plot_method_line(
            "neural_partial", "Neural router (partial)", preds_partial_neural_plan
        )
    if preds_full_neural_plan.size > 0:
        plot_method_line(
            "neural_full", "Neural router (full)", preds_full_neural_plan
        )
    if preds_partial_corr_plan.size > 0:
        plot_method_line("partial_corr", "Router Corr (partial)", preds_partial_corr_plan)
    if preds_full_corr_plan.size > 0:
        plot_method_line("full_corr", "Router Corr (full)", preds_full_corr_plan)
    if l2d_baseline is not None and preds_l2d_plan.size > 0:
        plot_method_line("l2d", "L2D", preds_l2d_plan)

    # Helper to scatter predictions with color = expert, marker = method
    method_markers = {
        "Oracle": "o",
        "Router (partial)": "s",
        "Router (full)": "^",
        "Neural router (partial)": "P",
        "Neural router (full)": "*",
        "Router Corr (partial)": "D",
        "Router Corr (full)": "v",
        "L2D": "x",
    }

    def scatter_method(label: str, preds_arr, sched_arr):
        if preds_arr.size == 0 or len(sched_arr) == 0:
            return
        times_arr = np.asarray(times, dtype=int)
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
    scatter_method("Oracle", preds_oracle, sched_oracle)
    scatter_method("Router (partial)", preds_partial_plan, sched_partial)
    scatter_method("Router (full)", preds_full_plan, sched_full)
    if preds_partial_neural_plan.size > 0 and sched_partial_neural:
        scatter_method(
            "Neural router (partial)",
            preds_partial_neural_plan,
            sched_partial_neural,
        )
    if preds_full_neural_plan.size > 0 and sched_full_neural:
        scatter_method(
            "Neural router (full)",
            preds_full_neural_plan,
            sched_full_neural,
        )
    if preds_partial_corr_plan.size > 0 and sched_partial_corr:
        scatter_method("Router Corr (partial)", preds_partial_corr_plan, sched_partial_corr)
    if preds_full_corr_plan.size > 0 and sched_full_corr:
        scatter_method("Router Corr (full)", preds_full_corr_plan, sched_full_corr)
    if l2d_baseline is not None and preds_l2d_plan.size > 0 and sched_l2d:
        scatter_method("L2D", preds_l2d_plan, sched_l2d)

    # Legends: methods (line+marker, colored by model) and experts (color)
    method_handles = []
    legend_specs = [
        ("true", "True"),
        ("oracle", "Oracle"),
        ("partial", "Router (partial)"),
        ("full", "Router (full)"),
        ("neural_partial", "Neural router (partial)"),
        ("neural_full", "Neural router (full)"),
        ("partial_corr", "Router Corr (partial)"),
        ("full_corr", "Router Corr (full)"),
        ("l2d", "L2D"),
    ]
    for key, label in legend_specs:
        color = get_model_color(key if label != "True" else "true")
        marker = method_markers.get(label, "o") if label != "True" else ""
        handle = mlines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="-",
            label="True $y_t$" if label == "True" else label,
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
    ax_forecast.set_ylabel("Forecast / True $y_t$")
    ax_forecast.set_title(
        f"Horizon-{H_eff} forecasts from t={t0_int} (Oracle, Partial, Full, Corr, L2D)"
    )
    plt.tight_layout()
    plt.show()
