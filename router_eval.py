import numpy as np
from typing import Optional, Tuple

from models.factorized_slds import FactorizedSLDS
from models.router_model import SLDSIMMRouter
from environment.synthetic_env import SyntheticTimeSeriesEnv
from environment.etth1_env import ETTh1TimeSeriesEnv
from models.l2d_baseline import L2D
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB

TimeSeriesEnv = SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv


def _train_router_offline(
    router: SLDSIMMRouter,
    env: TimeSeriesEnv,
    t_train_end: int,
    sliding_window: bool = False,
    sliding_window_size: Optional[int] = None,
) -> None:
    """
    Offline training phase for expanding-window / periodic retraining.

    This function replays the environment up to time t_train_end,
    allowing routers with a `training_mode` attribute (e.g.,
    correlated SLDS, DPF/Neural router) to update their internal
    parameters while tracking the posterior belief at t_train_end.
    """
    T = env.T
    if t_train_end <= 1 or t_train_end >= T:
        return

    # Determine training window start index for sliding vs expanding.
    if sliding_window and sliding_window_size is not None:
        t_start = max(1, t_train_end - int(sliding_window_size) + 1)
    else:
        t_start = 1

    router_name = getattr(router, "__class__", type(router)).__name__
    print(
        f"[offline-train] Router={router_name}, "
        f"window=[{t_start},{t_train_end}] "
        f"({'sliding' if sliding_window else 'expanding'})"
    )

    # Routers are expected to handle reset_beliefs and internal time
    # counters; we fully recompute the belief state on the training
    # window so that the final posterior at t_train_end is consistent
    # with the updated parameters.
    router.reset_beliefs()

    has_training_mode = hasattr(router, "training_mode")
    if has_training_mode:
        old_mode = bool(getattr(router, "training_mode"))
        router.training_mode = True  # enable parameter updates

    T_env = env.T
    t_end_clipped = min(t_train_end, T_env - 1)

    for t in range(1, t_end_clipped + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)
        r_t, cache = router.select_expert(x_t, available)
        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])

        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=None,
                available_experts=available,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=loss_all,
                available_experts=available,
                cache=cache,
            )

    if has_training_mode:
        router.training_mode = old_mode


def _run_router_online_window(
    router: SLDSIMMRouter,
    env: TimeSeriesEnv,
    t_start: int,
    t_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the router on a specified time window [t_start, t_end],
    recording costs and choices while treating parameters as frozen.
    """
    T = env.T
    t_start = max(int(t_start), 1)
    t_end = min(int(t_end), T - 1)
    if t_start > t_end:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    costs: list[float] = []
    choices: list[int] = []

    has_training_mode = hasattr(router, "training_mode")
    if has_training_mode:
        old_mode = bool(getattr(router, "training_mode"))
        router.training_mode = False  # freeze parameters

    router_name = getattr(router, "__class__", type(router)).__name__
    print(
        f"[online-run] Router={router_name}, "
        f"eval_window=[{t_start},{t_end}] "
        f"(training_mode={'off' if has_training_mode else 'n/a'})"
    )

    for t in range(t_start, t_end + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)
        r_t, cache = router.select_expert(x_t, available)

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + router.beta[r_t]

        costs.append(cost_t)
        choices.append(int(r_t))

        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=None,
                available_experts=available,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=loss_all,
                available_experts=available,
                cache=cache,
            )

    if has_training_mode:
        router.training_mode = old_mode

    return np.asarray(costs, dtype=float), np.asarray(choices, dtype=int)


def run_router_on_env_expanding(
    router: SLDSIMMRouter,
    env: TimeSeriesEnv,
    training_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expanding-window / periodic retraining evaluation protocol.

    Configuration (training_cfg):
      use_expanding_window : bool
      initial_window       : int  (N_init)
      retrain_interval     : int  (N_retrain)
      sliding_window       : bool
      sliding_window_size  : int  (used when sliding_window is True)

    For routers without any training hooks (no training_mode attribute),
    this function falls back to a single-pass run.
    """
    use_ew = bool(training_cfg.get("use_expanding_window", False))
    if not use_ew:
        return run_router_on_env(router, env)

    T = env.T
    N_init = int(training_cfg.get("initial_window", 300))
    N_retrain = int(training_cfg.get("retrain_interval", 100))
    sliding = bool(training_cfg.get("sliding_window", False))
    sliding_size_cfg = training_cfg.get("sliding_window_size", None)
    sliding_size = int(sliding_size_cfg) if sliding and sliding_size_cfg is not None else None

    has_training_mode = hasattr(router, "training_mode")
    # If the router cannot be trained, revert to single-pass evaluation.
    if not has_training_mode:
        return run_router_on_env(router, env)

    # Phase 1: initial offline training on [1, N_init].
    t_train_end = min(max(N_init, 1), T - 1)
    _train_router_offline(
        router,
        env,
        t_train_end=t_train_end,
        sliding_window=sliding,
        sliding_window_size=sliding_size,
    )

    costs_all: list[float] = []
    choices_all: list[int] = []

    t = t_train_end + 1
    while t < T:
        # Phase 2: online deployment over next interval.
        eval_end = min(t + N_retrain - 1, T - 1)
        costs_win, choices_win = _run_router_online_window(router, env, t, eval_end)
        if costs_win.size > 0:
            costs_all.extend(costs_win.tolist())
            choices_all.extend(choices_win.tolist())

        t_train_end = eval_end
        t = eval_end + 1

        if t >= T:
            break

        # Phase 3: offline retraining on expanded (or sliding) window
        # up to t_train_end.
        _train_router_offline(
            router,
            env,
            t_train_end=t_train_end,
            sliding_window=sliding,
            sliding_window_size=sliding_size,
        )

    return np.asarray(costs_all, dtype=float), np.asarray(choices_all, dtype=int)
def run_router_on_env(
    router: SLDSIMMRouter,
    env: TimeSeriesEnv,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the router on the synthetic environment.

    At each time t (1,...,T-1):
      - context x_t = env.get_context(t)
      - select expert r_t
      - environment reveals losses at time t
      - update beliefs using partial or full feedback
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}

    Returns
    -------
    costs : np.ndarray of shape (T-1,)
    choices : np.ndarray of shape (T-1,)
    """
    T = env.T
    N = env.num_experts

    costs = []
    choices = []

    for t in range(1, T):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)

        # Router decision for step t (select expert for loss at time t)
        r_t, cache = router.select_expert(x_t, available)

        # Environment produces true losses at time t
        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + router.beta[r_t]

        costs.append(cost_t)
        choices.append(r_t)

        # Belief update
        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=None,
                available_experts=available,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_r,
                losses_full=loss_all,
                available_experts=available,
                cache=cache,
            )

    return np.array(costs), np.array(choices)

def run_factored_router_on_env(
    router: FactorizedSLDS,
    env: TimeSeriesEnv,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the factored SLDS router on the synthetic environment.

    At each time t (1,...,T-1):
      - context x_t = env.get_context(t)
      - select expert r_t
      - environment reveals losses at time t
      - update beliefs using partial or full feedback
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}

    Returns
    -------
    costs : np.ndarray of shape (T-1,)
    choices : np.ndarray of shape (T-1,)
    """
    T = env.T
    N = env.num_experts

    costs = []
    choices = []

    for t in range(1, T):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)

        # Manage Registry state for factorized SLDS
        router.manage_registry(available)
        # IMM
        router.predict_step(x_t)
        # Predict, score and select expert
        I_t = router.select_action(x_t, available)

        loss_all = env.losses(t)
        loss_I_t = float(loss_all[I_t])
        cost_t = loss_I_t + router.beta[I_t]
        costs.append(cost_t)
        choices.append(I_t)

        # Correct belief with observed loss
        router.update_step(I_t, loss_I_t, x_t)

    return np.array(costs), np.array(choices)


def run_l2d_on_env(
    baseline: L2D,
    env: TimeSeriesEnv,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the usual learning-to-defer baseline on the synthetic environment.

    At each time t (1,...,T-1):
      - context x_t = env.get_context(t)
      - baseline selects expert r_t via π
      - environment reveals squared losses at time t
      - baseline updates π using full-feedback surrogate
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}

    Returns
    -------
    costs : np.ndarray of shape (T-1,)
    choices : np.ndarray of shape (T-1,)
    """
    T = env.T
    N = env.num_experts

    costs = []
    choices = []

    for t in range(1, T):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)

        r_t = baseline.select_expert(x_t, available)

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + baseline.beta[r_t]

        costs.append(cost_t)
        choices.append(r_t)

        # Full-feedback update on all experts' losses for available experts
        baseline.update(x_t, loss_all, available)

    return np.array(costs), np.array(choices)


def run_linucb_on_env(
    baseline: LinUCB,
    env: TimeSeriesEnv,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a LinUCB baseline on the synthetic environment.

    At each time t (1,...,T-1):
      - context x_t = env.get_context(t)
      - policy selects expert r_t via LinUCB
      - environment reveals squared losses at time t
      - policy updates its per-expert linear models
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}
    """
    T = env.T
    N = env.num_experts

    costs: list[float] = []
    choices: list[int] = []

    for t in range(1, T):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)

        r_t = baseline.select_expert(x_t, available)

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + baseline.beta[r_t]

        costs.append(cost_t)
        choices.append(int(r_t))

        baseline.update(x_t, loss_all, available)

    return np.array(costs), np.array(choices)


def run_neuralucb_on_env(
    baseline: NeuralUCB,
    env: TimeSeriesEnv,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a NeuralUCB baseline on the synthetic environment.

    At each time t (1,...,T-1):
      - context x_t = env.get_context(t)
      - policy selects expert r_t via NeuralUCB
      - environment reveals squared losses at time t
      - policy updates its neural embedding + linear heads
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}
    """
    T = env.T
    N = env.num_experts

    costs: list[float] = []
    choices: list[int] = []

    for t in range(1, T):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)

        r_t = baseline.select_expert(x_t, available)

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + baseline.beta[r_t]

        costs.append(cost_t)
        choices.append(int(r_t))

        baseline.update(x_t, loss_all, available)

    return np.array(costs), np.array(choices)


def run_random_on_env(
    env: TimeSeriesEnv,
    beta: np.ndarray,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random baseline that selects an available expert uniformly at random
    at each time step and incurs cost ℓ_{r_t,t} + β_{r_t}.

    Parameters
    ----------
    env : SyntheticTimeSeriesEnv
        Environment providing contexts, expert predictions, and losses.
    beta : np.ndarray
        Consultation costs β_j, shape (N,).
    seed : int
        Random seed for reproducible choices.

    Returns
    -------
    costs : np.ndarray of shape (T-1,)
        Per-step costs.
    choices : np.ndarray of shape (T-1,)
        Selected expert index at each time t.
    """
    T = env.T
    N = env.num_experts

    beta = np.asarray(beta, dtype=float).reshape(N)
    rng = np.random.default_rng(seed)

    costs = np.zeros(T - 1, dtype=float)
    choices = np.zeros(T - 1, dtype=int)

    for t in range(1, T):
        available = env.get_available_experts(t)
        if available.size == 0:
            available = np.arange(N, dtype=int)

        r_t = int(rng.choice(available))

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        costs[t - 1] = loss_r + beta[r_t]
        choices[t - 1] = r_t

    return costs, choices


def run_oracle_on_env(
    env: TimeSeriesEnv,
    beta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oracle baseline that, at each time step, selects the available expert
    with minimum instantaneous cost ℓ_{j,t} + β_j (clairvoyant per-step best).

    Parameters
    ----------
    env : SyntheticTimeSeriesEnv
        Environment providing expert losses.
    beta : np.ndarray
        Consultation costs β_j, shape (N,).

    Returns
    -------
    costs : np.ndarray of shape (T-1,)
        Per-step oracle costs.
    choices : np.ndarray of shape (T-1,)
        Oracle-selected expert index at each time t.
    """
    T = env.T
    N = env.num_experts

    beta = np.asarray(beta, dtype=float).reshape(N)

    costs = np.zeros(T - 1, dtype=float)
    choices = np.zeros(T - 1, dtype=int)

    for t in range(1, T):
        loss_all = env.losses(t)
        total_costs = loss_all + beta

        available = env.get_available_experts(t)
        if available.size == 0:
            available = np.arange(N, dtype=int)

        avail_costs = total_costs[available]
        r_t = int(available[int(np.argmin(avail_costs))])

        costs[t - 1] = float(total_costs[r_t])
        choices[t - 1] = r_t

    return costs, choices


def compute_predictions_from_choices(
    env: TimeSeriesEnv,
    choices: np.ndarray,
) -> np.ndarray:
    """
    Given a sequence of selected experts (choices), compute the corresponding
    expert predictions at each time step t = 1,...,T-1.

    Parameters
    ----------
    env : SyntheticTimeSeriesEnv
        Environment providing contexts and expert predictors.
    choices : np.ndarray of shape (T-1,)
        choices[t-1] is the expert index selected for time t.

    Returns
    -------
    preds : np.ndarray of shape (T-1,)
        Predicted y_t from the selected expert at each time t.
    """
    T = env.T
    choices = np.asarray(choices, dtype=int)
    assert choices.shape[0] == T - 1

    preds = np.zeros(T - 1, dtype=float)
    for t in range(1, T):
        x_t = env.get_context(t)
        j = int(choices[t - 1])
        preds[t - 1] = env.expert_predict(j, x_t)
    return preds
