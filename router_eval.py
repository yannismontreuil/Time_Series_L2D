import copy
import numpy as np
from typing import Optional, Sequence, Tuple

from models.factorized_slds import FactorizedSLDS
from models.router_model import SLDSIMMRouter
from environment.synthetic_env import SyntheticTimeSeriesEnv
from environment.etth1_env import ETTh1TimeSeriesEnv
from models.l2d_baseline import L2D
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB

TimeSeriesEnv = SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv

_TRANSITION_LOG_CFG: Optional[dict] = None
_TRANSITION_LOG_LABELS: dict[int, str] = {}
_TRANSITION_LOG_STORE: dict[str, list[tuple[int, Optional[np.ndarray]]]] = {}


def _maybe_snapshot_router(
    router: SLDSIMMRouter | FactorizedSLDS | L2D | LinUCB | NeuralUCB,
    t: int,
    snapshot_at_t: Optional[int],
    snapshot_dict: Optional[dict],
    snapshot_key: Optional[str],
) -> None:
    if snapshot_at_t is None or snapshot_dict is None or snapshot_key is None:
        return
    if int(t) != int(snapshot_at_t):
        return
    if snapshot_key in snapshot_dict:
        return
    snapshot_dict[snapshot_key] = copy.deepcopy(router)


def set_transition_log_config(cfg: Optional[dict]) -> None:
    global _TRANSITION_LOG_CFG
    if cfg is None or not isinstance(cfg, dict):
        _TRANSITION_LOG_CFG = None
        return
    _TRANSITION_LOG_CFG = cfg


def register_transition_log_label(obj: object, label: str) -> None:
    if obj is None:
        return
    _TRANSITION_LOG_LABELS[id(obj)] = str(label)

def get_transition_log_config() -> Optional[dict]:
    cfg = _get_transition_log_cfg()
    return None if cfg is None else dict(cfg)


def get_transition_log_store(reset: bool = False) -> dict[str, list[tuple[int, Optional[np.ndarray]]]]:
    snapshot: dict[str, list[tuple[int, Optional[np.ndarray]]]] = {}
    for label, series in _TRANSITION_LOG_STORE.items():
        snapshot[label] = [
            (int(t), None if mat is None else np.array(mat, copy=True))
            for t, mat in series
        ]
    if reset:
        _TRANSITION_LOG_STORE.clear()
    return snapshot


def _get_transition_log_cfg() -> Optional[dict]:
    cfg = _TRANSITION_LOG_CFG
    if not isinstance(cfg, dict):
        return None
    if not cfg.get("enabled", False):
        return None
    return cfg


def _transition_log_precision(cfg: dict) -> int:
    try:
        return int(cfg.get("precision", 4))
    except (TypeError, ValueError):
        return 4


def _should_log_transition(t: int, cfg: dict) -> bool:
    try:
        stride = int(cfg.get("stride", 1))
    except (TypeError, ValueError):
        stride = 1
    if stride <= 1:
        stride = 1
    start_t = cfg.get("start_t", None)
    end_t = cfg.get("end_t", None)
    if start_t is not None:
        try:
            if t < int(start_t):
                return False
        except (TypeError, ValueError):
            pass
    if end_t is not None:
        try:
            if t > int(end_t):
                return False
        except (TypeError, ValueError):
            pass
    if stride == 1:
        return True
    return ((t - 1) % stride) == 0


def _format_matrix(mat: np.ndarray, precision: int) -> str:
    return np.array2string(
        mat,
        precision=precision,
        floatmode="fixed",
        separator=", ",
    )


def _resolve_transition_label(obj: object, fallback: Optional[str]) -> str:
    if obj is not None:
        label = _TRANSITION_LOG_LABELS.get(id(obj))
        if label:
            return label
    if fallback:
        return str(fallback)
    if obj is None:
        return "unknown"
    name = getattr(obj, "__class__", type(obj)).__name__
    extras = []
    feedback_mode = getattr(obj, "feedback_mode", None)
    if feedback_mode:
        extras.append(str(feedback_mode))
    if isinstance(obj, FactorizedSLDS):
        extras.append(str(getattr(obj, "transition_mode", "transition")))
        if getattr(obj, "d_g", None) == 0:
            extras.append("no_g")
    if extras:
        return f"{name}({', '.join(extras)})"
    return name


def _record_transition(label: str, t: int, mat: Optional[np.ndarray]) -> None:
    if label not in _TRANSITION_LOG_STORE:
        _TRANSITION_LOG_STORE[label] = []
    if mat is None:
        _TRANSITION_LOG_STORE[label].append((int(t), None))
    else:
        _TRANSITION_LOG_STORE[label].append((int(t), np.array(mat, copy=True)))


def _transition_matrix_for(obj: object, x_t: np.ndarray) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, FactorizedSLDS):
        return obj._context_transition(x_t)
    if hasattr(obj, "Pi"):
        return np.asarray(getattr(obj, "Pi"), dtype=float)
    return None


def _log_transition(
    obj: object,
    t: int,
    x_t: np.ndarray,
    fallback_label: Optional[str] = None,
) -> None:
    cfg = _get_transition_log_cfg()
    if cfg is None:
        return
    if not _should_log_transition(t, cfg):
        return
    label = _resolve_transition_label(obj, fallback_label)
    mat = _transition_matrix_for(obj, x_t)
    if bool(cfg.get("plot", False)) or bool(cfg.get("collect", False)):
        _record_transition(label, t, mat)
    if not bool(cfg.get("print", True)):
        return
    if mat is None:
        print(f"[transition] {label} t={t} Pi=N/A")
        return
    precision = _transition_log_precision(cfg)
    mat_str = _format_matrix(np.asarray(mat, dtype=float), precision)
    print(f"[transition] {label} t={t} Pi={mat_str}")


def _router_observes_residual(router: SLDSIMMRouter) -> bool:
    return getattr(router, "observation_mode", "loss") == "residual"


def _require_available(
    available: np.ndarray | Sequence[int],
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
    router: SLDSIMMRouter,
    env: TimeSeriesEnv,
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
        if router.feedback_mode == "full":
            residuals_full = _mask_feedback_vector(
                residuals, available, r_t, full_feedback=True
            )
        return residual_r, loss_r, residuals_full

    loss_all = env.losses(t)
    loss_r = float(loss_all[r_t])
    losses_full = None
    if router.feedback_mode == "full":
        losses_full = _mask_feedback_vector(
            loss_all, available, r_t, full_feedback=True
        )
    return loss_r, loss_r, losses_full


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
        _log_transition(router, t, x_t)
        available = _require_available(env.get_available_experts(t), t, "router offline-train")
        r_t, cache = router.select_expert(x_t, available)
        if not np.any(available == int(r_t)):
            raise ValueError(
                f"router offline-train: selected expert {r_t} not in E_t at t={t}."
            )
        loss_obs, loss_r, losses_full = _get_router_observation(
            router, env, t, x_t, available, r_t
        )

        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=None,
                available_experts=available,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=losses_full,
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
    snapshot_at_t: Optional[int] = None,
    snapshot_dict: Optional[dict] = None,
    snapshot_key: Optional[str] = None,
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
        _log_transition(router, t, x_t)
        available = _require_available(env.get_available_experts(t), t, "router online-run")
        r_t, cache = router.select_expert(x_t, available)
        if not np.any(available == int(r_t)):
            raise ValueError(
                f"router online-run: selected expert {r_t} not in E_t at t={t}."
            )

        loss_obs, loss_r, losses_full = _get_router_observation(
            router, env, t, x_t, available, r_t
        )
        cost_t = loss_r + router.beta[r_t]

        costs.append(cost_t)
        choices.append(int(r_t))

        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=None,
                available_experts=available,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=losses_full,
                available_experts=available,
                cache=cache,
            )
        _maybe_snapshot_router(
            router, t, snapshot_at_t, snapshot_dict, snapshot_key
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
    snapshot_at_t: Optional[int] = None,
    snapshot_dict: Optional[dict] = None,
    snapshot_key: Optional[str] = None,
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
        _log_transition(router, t, x_t)
        available = _require_available(env.get_available_experts(t), t, "router run")

        # Router decision for step t (select expert for loss at time t)
        r_t, cache = router.select_expert(x_t, available)
        if not np.any(available == int(r_t)):
            raise ValueError(f"router run: selected expert {r_t} not in E_t at t={t}.")

        # Environment produces observation (loss or residual) at time t
        loss_obs, loss_r, losses_full = _get_router_observation(
            router, env, t, x_t, available, r_t
        )
        cost_t = loss_r + router.beta[r_t]

        costs.append(cost_t)
        choices.append(r_t)

        # Belief update
        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=None,
                available_experts=available,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=losses_full,
                available_experts=available,
                cache=cache,
            )
        _maybe_snapshot_router(
            router, t, snapshot_at_t, snapshot_dict, snapshot_key
        )

    return np.array(costs), np.array(choices)


def run_router_on_env_training_window(
    router: SLDSIMMRouter,
    env: TimeSeriesEnv,
    t_train_end: int,
    snapshot_at_t: Optional[int] = None,
    snapshot_dict: Optional[dict] = None,
    snapshot_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the router on a training window [1, t_train_end], recording costs
    and choices while allowing parameter updates.
    """
    T = env.T
    t_train_end = min(int(t_train_end), T - 1)
    if t_train_end <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    costs: list[float] = []
    choices: list[int] = []

    router.reset_beliefs()

    has_training_mode = hasattr(router, "training_mode")
    if has_training_mode:
        old_mode = bool(getattr(router, "training_mode"))
        router.training_mode = True

    for t in range(1, t_train_end + 1):
        x_t = env.get_context(t)
        _log_transition(router, t, x_t)
        available = _require_available(env.get_available_experts(t), t, "router training window")
        r_t, cache = router.select_expert(x_t, available)
        if not np.any(available == int(r_t)):
            raise ValueError(
                f"router training window: selected expert {r_t} not in E_t at t={t}."
            )
        loss_obs, loss_r, losses_full = _get_router_observation(
            router, env, t, x_t, available, r_t
        )
        cost_t = loss_r + router.beta[r_t]

        costs.append(cost_t)
        choices.append(int(r_t))

        if router.feedback_mode == "partial":
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=None,
                available_experts=available,
                cache=cache,
            )
        else:
            router.update_beliefs(
                r_t=r_t,
                loss_obs=loss_obs,
                losses_full=losses_full,
                available_experts=available,
                cache=cache,
            )
        _maybe_snapshot_router(
            router, t, snapshot_at_t, snapshot_dict, snapshot_key
        )

    if has_training_mode:
        router.training_mode = old_mode

    return np.asarray(costs, dtype=float), np.asarray(choices, dtype=int)


def run_router_on_env_em_split(
    router: SLDSIMMRouter,
    env: TimeSeriesEnv,
    t_train_end: int,
    snapshot_at_t: Optional[int] = None,
    snapshot_dict: Optional[dict] = None,
    snapshot_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run EM-style training on [1, t_train_end] (if applicable), then run
    the online phase on [t_train_end + 1, T - 1], returning full-length
    cost and choice arrays.
    """
    T = env.T
    t_train_end = min(int(t_train_end), T - 1)
    if t_train_end <= 0:
        return run_router_on_env(router, env)

    train_costs, train_choices = run_router_on_env_training_window(
        router,
        env,
        t_train_end,
        snapshot_at_t=snapshot_at_t,
        snapshot_dict=snapshot_dict,
        snapshot_key=snapshot_key,
    )
    online_costs, online_choices = _run_router_online_window(
        router,
        env,
        t_train_end + 1,
        T - 1,
        snapshot_at_t=snapshot_at_t,
        snapshot_dict=snapshot_dict,
        snapshot_key=snapshot_key,
    )

    costs = (
        np.concatenate([train_costs, online_costs], axis=0)
        if train_costs.size or online_costs.size
        else np.zeros(0, dtype=float)
    )
    choices = (
        np.concatenate([train_choices, online_choices], axis=0)
        if train_choices.size or online_choices.size
        else np.zeros(0, dtype=int)
    )
    return costs, choices

def run_factored_router_on_env(
    router: FactorizedSLDS,
    env: TimeSeriesEnv,
    snapshot_at_t: Optional[int] = None,
    snapshot_dict: Optional[dict] = None,
    snapshot_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper around run_router_on_env for FactorizedSLDS.
    """
    return run_router_on_env(
        router,
        env,
        snapshot_at_t=snapshot_at_t,
        snapshot_dict=snapshot_dict,
        snapshot_key=snapshot_key,
    )


def run_l2d_on_env(
    baseline: L2D,
    env: TimeSeriesEnv,
    t_start: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the usual learning-to-defer baseline on the synthetic environment.

    At each time t (t_start,...,T-1):
      - context x_t = env.get_context(t)
      - baseline selects expert r_t via π
      - environment reveals squared losses at time t
      - baseline updates π using full-feedback surrogate
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}

    Returns
    -------
    costs : np.ndarray of shape (T-1,)
    choices : np.ndarray of shape (T-1,)
        Entries before t_start are NaN in costs and zeros in choices.
    """
    T = env.T
    N = env.num_experts
    t_start = max(1, int(t_start))

    costs = np.full(T - 1, np.nan, dtype=float)
    choices = np.zeros(T - 1, dtype=int)

    for t in range(t_start, T):
        x_t = env.get_context(t)
        _log_transition(baseline, t, x_t)
        available = _require_available(env.get_available_experts(t), t, "L2D")

        r_t = baseline.select_expert(x_t, available)
        if not np.any(available == int(r_t)):
            raise ValueError(f"L2D: selected expert {r_t} not in E_t at t={t}.")

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + baseline.beta[r_t]
        loss_masked = _mask_feedback_vector(
            loss_all, available, r_t, full_feedback=True
        )

        costs[t - 1] = cost_t
        choices[t - 1] = int(r_t)

        # Full-feedback update on all experts' losses for available experts
        baseline.update(x_t, loss_masked, available, selected_expert=r_t)

    return costs, choices


def run_linucb_on_env(
    baseline: LinUCB,
    env: TimeSeriesEnv,
    t_start: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a LinUCB baseline on the synthetic environment.

    At each time t (t_start,...,T-1):
      - context x_t = env.get_context(t)
      - policy selects expert r_t via LinUCB
      - environment reveals squared losses at time t
      - policy updates its per-expert linear models
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}
    """
    T = env.T
    N = env.num_experts
    t_start = max(1, int(t_start))

    costs = np.full(T - 1, np.nan, dtype=float)
    choices = np.zeros(T - 1, dtype=int)

    for t in range(t_start, T):
        x_t = env.get_context(t)
        _log_transition(baseline, t, x_t)
        available = _require_available(env.get_available_experts(t), t, "LinUCB")

        r_t = baseline.select_expert(x_t, available)
        if not np.any(available == int(r_t)):
            raise ValueError(f"LinUCB: selected expert {r_t} not in E_t at t={t}.")

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + baseline.beta[r_t]
        loss_masked = _mask_feedback_vector(
            loss_all,
            available,
            r_t,
            full_feedback=baseline.feedback_mode == "full",
        )

        costs[t - 1] = cost_t
        choices[t - 1] = int(r_t)

        baseline.update(x_t, loss_masked, available, selected_expert=r_t)

    return costs, choices


def run_neuralucb_on_env(
    baseline: NeuralUCB,
    env: TimeSeriesEnv,
    t_start: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a NeuralUCB baseline on the synthetic environment.

    At each time t (t_start,...,T-1):
      - context x_t = env.get_context(t)
      - policy selects expert r_t via NeuralUCB
      - environment reveals squared losses at time t
      - policy updates its neural embedding + linear heads
      - record incurred cost: ℓ_{r_t,t} + β_{r_t}
    """
    T = env.T
    N = env.num_experts
    t_start = max(1, int(t_start))

    costs = np.full(T - 1, np.nan, dtype=float)
    choices = np.zeros(T - 1, dtype=int)

    for t in range(t_start, T):
        x_t = env.get_context(t)
        _log_transition(baseline, t, x_t)
        available = _require_available(env.get_available_experts(t), t, "NeuralUCB")

        r_t = baseline.select_expert(x_t, available)
        if not np.any(available == int(r_t)):
            raise ValueError(f"NeuralUCB: selected expert {r_t} not in E_t at t={t}.")

        loss_all = env.losses(t)
        loss_r = float(loss_all[r_t])
        cost_t = loss_r + baseline.beta[r_t]
        loss_masked = _mask_feedback_vector(
            loss_all,
            available,
            r_t,
            full_feedback=baseline.feedback_mode == "full",
        )

        costs[t - 1] = cost_t
        choices[t - 1] = int(r_t)

        baseline.update(x_t, loss_masked, available, selected_expert=r_t)

    return costs, choices


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
        _log_transition(None, t, np.asarray([]), fallback_label="Random")
        available = _require_available(env.get_available_experts(t), t, "Random")

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
        _log_transition(None, t, np.asarray([]), fallback_label="Oracle")
        loss_all = env.losses(t)
        total_costs = loss_all + beta

        available = _require_available(env.get_available_experts(t), t, "Oracle")

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
