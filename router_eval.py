import numpy as np
from typing import Optional, Sequence, Tuple

from router_model import SLDSIMMRouter
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import L2D
from linucb_baseline import LinUCB
from neuralucb_baseline import NeuralUCB


def run_router_on_env(
    router: SLDSIMMRouter,
    env: SyntheticTimeSeriesEnv,
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


def run_l2d_on_env(
    baseline: L2D,
    env: SyntheticTimeSeriesEnv,
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
    env: SyntheticTimeSeriesEnv,
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
    env: SyntheticTimeSeriesEnv,
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
    env: SyntheticTimeSeriesEnv,
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
    env: SyntheticTimeSeriesEnv,
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
    env: SyntheticTimeSeriesEnv,
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
