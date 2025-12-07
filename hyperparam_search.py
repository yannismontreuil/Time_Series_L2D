import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, List, Any

import numpy as np

from router_model_corr import SLDSIMMRouter_Corr, feature_phi
from synthetic_env import SyntheticTimeSeriesEnv
from etth1_env import ETTh1TimeSeriesEnv
from router_eval import run_router_on_env

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

# Simple in-process cache for environments keyed by configuration and
# seed. This avoids retraining ETTh1 neural-network experts and
# regenerating synthetic time series for every hyperparameter
# configuration in a single worker process.
_ENV_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _load_config(path: str = "config.yaml") -> Dict:
    """
    Load YAML/JSON configuration (same convention as slds_imm_router.py).
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Top-level configuration must be a mapping.")
    return data


def _resolve_vector(value, default_scalar: float, length: int) -> np.ndarray:
    """
    Utility: broadcast scalars / validate vectors of length `length`.
    """
    if value is None:
        return np.full(length, default_scalar, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return np.full(length, float(arr), dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"Expected vector of length {length}, got shape {arr.shape}.")
    return arr


def _get_dims_from_config(cfg: Dict, override_M: int | None = None) -> Tuple[int, int, int, str]:
    """
    Extract (N, d, M, setting) from the config, matching slds_imm_router.py.

    If override_M is provided, it is used as the number of regimes
    irrespective of the environment setting (useful for hyperparameter
    search over num_regimes).
    """
    env_cfg = cfg.get("environment", {})
    setting = env_cfg.get("setting", "easy_setting")

    # Experts and state dimension
    N = int(env_cfg.get("num_experts", 5))
    d = int(env_cfg.get("state_dim", 2))

    if override_M is not None:
        M = int(override_M)
    else:
        # Regimes (respect noisy_forgetting convention: at least 5 regimes)
        if setting == "noisy_forgetting":
            default_M = 5
            raw_M = env_cfg.get("num_regimes", None)
            if raw_M is None:
                M = default_M
            else:
                M = max(int(raw_M), default_M)
        else:
            M = int(env_cfg.get("num_regimes", 2))

    return N, d, M, setting


def build_environment(seed: int, cfg: Dict, num_regimes: int | None = None):
    """
    Environment configuration mirroring the main experiment in
    slds_imm_router.py. Supports both the synthetic environment and the
    ETTh1 dataset, depending on `environment.data_source`.
    """
    env_cfg = cfg.get("environment", {})
    data_source = env_cfg.get("data_source", "synthetic")
    N, _, M, setting = _get_dims_from_config(cfg, override_M=num_regimes)

    # Build a cache key that captures the aspects of the environment
    # that affect its dynamics and expert structure. This allows reuse
    # across different hyperparameter configurations within a single
    # worker process.
    if data_source == "etth1":
        T_raw = env_cfg.get("T", None)
        T_env = None if T_raw is None else int(T_raw)
        csv_path = env_cfg.get("csv_path", "Data/ETTh1.csv")
        target_column = env_cfg.get("target_column", "OT")
        unavailable_expert_idx = env_cfg.get("unavailable_expert_idx", None)
        unavailable_intervals = env_cfg.get("unavailable_intervals", None)
        arrival_expert_idx = env_cfg.get("arrival_expert_idx", None)
        arrival_intervals = env_cfg.get("arrival_intervals", None)

        # Canonicalize interval lists for the cache key.
        def _canon_intervals(intv):
            if intv is None:
                return None
            return tuple(tuple(int(v) for v in pair) for pair in intv)

        key: Tuple[Any, ...] = (
            "etth1",
            int(seed),
            int(N),
            int(M),
            T_env,
            csv_path,
            target_column,
            unavailable_expert_idx,
            _canon_intervals(unavailable_intervals),
            arrival_expert_idx,
            _canon_intervals(arrival_intervals),
        )
        if key in _ENV_CACHE:
            return _ENV_CACHE[key]

        env = ETTh1TimeSeriesEnv(
            csv_path=csv_path,
            target_column=target_column,
            num_experts=N,
            num_regimes=M,
            T=T_env,
            seed=int(seed),
            unavailable_expert_idx=unavailable_expert_idx,
            unavailable_intervals=unavailable_intervals,
            arrival_expert_idx=arrival_expert_idx,
            arrival_intervals=arrival_intervals,
        )
        _ENV_CACHE[key] = env
        return env

    # Synthetic environment (default)
    T_env = int(env_cfg.get("T", 300))
    unavailable_expert_idx = env_cfg.get("unavailable_expert_idx", 1)
    unavailable_intervals = env_cfg.get(
        "unavailable_intervals", [[10, 50], [200, 250]]
    )
    arrival_expert_idx = env_cfg.get("arrival_expert_idx", 4)
    arrival_intervals = env_cfg.get("arrival_intervals", [[120, 200]])
    noise_scale = env_cfg.get("noise_scale", None)

    def _canon_intv_synth(intv):
        if intv is None:
            return None
        return tuple(tuple(int(v) for v in pair) for pair in intv)

    key_synth: Tuple[Any, ...] = (
        "synthetic",
        int(seed),
        int(N),
        int(M),
        T_env,
        int(unavailable_expert_idx) if unavailable_expert_idx is not None else None,
        _canon_intv_synth(unavailable_intervals),
        int(arrival_expert_idx) if arrival_expert_idx is not None else None,
        _canon_intv_synth(arrival_intervals),
        setting,
        noise_scale,
    )
    if key_synth in _ENV_CACHE:
        return _ENV_CACHE[key_synth]

    env = SyntheticTimeSeriesEnv(
        num_experts=N,
        num_regimes=M,
        T=T_env,
        seed=int(seed),
        unavailable_expert_idx=int(unavailable_expert_idx)
        if unavailable_expert_idx is not None
        else None,
        unavailable_intervals=unavailable_intervals,
        arrival_expert_idx=int(arrival_expert_idx)
        if arrival_expert_idx is not None
        else None,
        arrival_intervals=arrival_intervals,
        setting=setting,
        noise_scale=noise_scale,
    )
    _ENV_CACHE[key_synth] = env
    return env


def build_correlated_routers(
    lambda_risk: float,
    q_g_scale: float,
    q_u_scale: float,
    r_scale: float,
    cfg: Dict,
    num_regimes: int | None = None,
    feature_mode: str | None = None,
    feature_learning_rate: float | None = None,
    feature_arch: str | None = None,
    shared_dim: int | None = None,
    idiosyncratic_dim: int | None = None,
) -> Tuple[SLDSIMMRouter_Corr, SLDSIMMRouter_Corr]:
    """
    Construct partial- and full-feedback SLDSIMMRouter_Corr instances
    with given hyperparameters, using the same base configuration as
    the main experiment in slds_imm_router.py.

    Hyperparameters:
      - lambda_risk: scalar risk sensitivity (overrides config value)
      - q_scale: global scaling applied to both Q_g and Q_u
      - r_scale: global scaling applied to observation noise R
    """
    env_cfg = cfg.get("environment", {})
    routers_cfg = cfg.get("routers", {})
    slds_cfg = routers_cfg.get("slds_imm", {}) or {}
    slds_corr_cfg = routers_cfg.get("slds_imm_corr", {}) or {}

    N, d, M, _ = _get_dims_from_config(cfg, override_M=num_regimes)

    # Consultation costs (shared across experts)
    beta = _resolve_vector(routers_cfg.get("beta", None), 0.0, N)

    # Dimensions for correlated router (can be overridden).
    d_g_cfg = slds_corr_cfg.get("shared_dim", 1)
    d_u_cfg = slds_corr_cfg.get("idiosyncratic_dim", d)
    d_g = int(d_g_cfg if shared_dim is None else shared_dim)
    d_u = int(d_u_cfg if idiosyncratic_dim is None else idiosyncratic_dim)

    # Observation noise R for the correlated router: we treat r_scale
    # as an absolute value and build R_{k,j} = r_scale for all
    # regimes/experts. This decouples the hyperparameter search from
    # any baseline R specified in the config and is more robust across
    # different problems.
    R = np.full((M, N), float(r_scale), dtype=float)

    # Regime transition matrix Π
    Pi_cfg = slds_cfg.get("Pi", None)
    if Pi_cfg is not None:
        Pi_raw = np.asarray(Pi_cfg, dtype=float)
        if Pi_raw.shape == (M, M):
            Pi = Pi_raw
        else:
            # If the configured Pi does not match the current number of
            # regimes (e.g., when sweeping over M in hyperparameter
            # search), fall back to a simple uniform transition matrix.
            Pi = np.full((M, M), 1.0 / M, dtype=float)
    else:
        if M == 2:
            Pi = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
        else:
            Pi = np.full((M, M), 1.0 / M, dtype=float)

    # Joint dynamics for correlated router
    # A_g: per-regime shared-factor dynamics
    A_g_cfg = slds_corr_cfg.get("A_g", None)
    if A_g_cfg is not None:
        A_g = np.asarray(A_g_cfg, dtype=float)
    else:
        A_g = np.tile(np.eye(d_g, dtype=float)[None, :, :], (M, 1, 1))

    # Q_g: per-regime shared-factor process noise.
    # For hyperparameter search, we parameterize Q_g directly via the
    # scalar q_g_scale, rather than only scaling a fixed template from
    # the config. By default we use Q_g(k) = q_g_scale * I for all k.
    Q_g_cfg = slds_corr_cfg.get("Q_g", None)
    if Q_g_cfg is not None:
        Q_g_base = np.asarray(Q_g_cfg, dtype=float)
        Q_g = Q_g_base * float(q_g_scale)
    else:
        q_val_g = float(q_g_scale)
        Q_g = np.zeros((M, d_g, d_g), dtype=float)
        for k in range(M):
            Q_g[k] = q_val_g * np.eye(d_g, dtype=float)

    # A_u: per-regime idiosyncratic dynamics
    A_u_cfg = slds_corr_cfg.get("A_u", None)
    if A_u_cfg is not None:
        A_u = np.asarray(A_u_cfg, dtype=float)
    else:
        A_u = np.tile(np.eye(d_u, dtype=float)[None, :, :], (M, 1, 1))

    # Q_u: per-regime idiosyncratic process noise, parameterized
    # directly via q_u_scale in the same way.
    Q_u_cfg = slds_corr_cfg.get("Q_u", None)
    if Q_u_cfg is not None:
        Q_u_base = np.asarray(Q_u_cfg, dtype=float)
        Q_u = Q_u_base * float(q_u_scale)
    else:
        q_val_u = float(q_u_scale)
        Q_u = np.zeros((M, d_u, d_u), dtype=float)
        for k in range(M):
            Q_u[k] = q_val_u * np.eye(d_u, dtype=float)

    # Shared-factor loadings B
    B_cfg = slds_corr_cfg.get("B", None)
    if B_cfg is not None:
        B = np.asarray(B_cfg, dtype=float)
    else:
        load = float(slds_corr_cfg.get("B_intercept_load", 1.0))
        B = np.zeros((N, d_u, d_g), dtype=float)
        for j in range(N):
            B[j, 0, 0] = load

    # Priors and numerical stabilizer
    eps_corr = float(slds_corr_cfg.get("eps", 1e-8))
    g_mean0_cfg = slds_corr_cfg.get("g_mean0", None)
    g_cov0_cfg = slds_corr_cfg.get("g_cov0", None)
    u_mean0_cfg = slds_corr_cfg.get("u_mean0", None)
    u_cov0_cfg = slds_corr_cfg.get("u_cov0", None)

    g_mean0 = (
        np.asarray(g_mean0_cfg, dtype=float) if g_mean0_cfg is not None else None
    )
    g_cov0 = (
        np.asarray(g_cov0_cfg, dtype=float) if g_cov0_cfg is not None else None
    )
    u_mean0 = (
        np.asarray(u_mean0_cfg, dtype=float) if u_mean0_cfg is not None else None
    )
    u_cov0 = (
        np.asarray(u_cov0_cfg, dtype=float) if u_cov0_cfg is not None else None
    )

    # If dimensionality overrides are in effect or prior shapes do not
    # match the chosen dims, fall back to default priors inside
    # SLDSIMMRouter_Corr by passing None for means/covariances.
    if g_mean0 is not None and g_mean0.shape != (d_g,):
        g_mean0 = None
        g_cov0 = None
    if g_cov0 is not None and g_cov0.shape != (d_g, d_g):
        g_mean0 = None
        g_cov0 = None
    if u_mean0 is not None and u_mean0.shape != (d_u,):
        u_mean0 = None
        u_cov0 = None
    if u_cov0 is not None and u_cov0.shape != (d_u, d_u):
        u_mean0 = None
        u_cov0 = None

    staleness_threshold_cfg = routers_cfg.get("staleness_threshold", None)
    staleness_threshold = (
        int(staleness_threshold_cfg) if staleness_threshold_cfg is not None else None
    )

    # Exploration and feature-learning configuration for correlated router.
    exploration_mode = slds_corr_cfg.get("exploration_mode", "greedy")
    feature_mode_cfg = slds_corr_cfg.get("feature_mode", "fixed")
    feature_mode_eff = feature_mode_cfg if feature_mode is None else str(feature_mode)

    feature_lr_cfg = float(slds_corr_cfg.get("feature_learning_rate", 0.0))
    if feature_learning_rate is None:
        feature_lr_eff = feature_lr_cfg
    else:
        feature_lr_eff = float(feature_learning_rate)

    feature_freeze_after = slds_corr_cfg.get("feature_freeze_after", None)
    feature_log_interval = slds_corr_cfg.get("feature_log_interval", None)

    feature_arch_cfg = slds_corr_cfg.get("feature_arch", "linear")
    feature_arch_eff = feature_arch_cfg if feature_arch is None else str(feature_arch)

    feature_hidden_dim = slds_corr_cfg.get("feature_hidden_dim", None)
    feature_activation = slds_corr_cfg.get("feature_activation", "tanh")

    # Scalar lambda_risk used for both routers in the search
    lambda_scalar = float(lambda_risk)

    router_partial_corr = SLDSIMMRouter_Corr(
        num_experts=N,
        num_regimes=M,
        shared_dim=d_g,
        idiosyncratic_dim=d_u,
        feature_fn=feature_phi,
        A_g=A_g,
        Q_g=Q_g,
        A_u=A_u,
        Q_u=Q_u,
        B=B,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=lambda_scalar,
        staleness_threshold=staleness_threshold,
        exploration_mode=exploration_mode,
        feature_mode=feature_mode_eff,
        feature_learning_rate=feature_lr_eff,
        feature_freeze_after=feature_freeze_after,
        feature_log_interval=feature_log_interval,
        feedback_mode="partial",
        eps=eps_corr,
        g_mean0=g_mean0,
        g_cov0=g_cov0,
        u_mean0=u_mean0,
        u_cov0=u_cov0,
    )

    router_full_corr = SLDSIMMRouter_Corr(
        num_experts=N,
        num_regimes=M,
        shared_dim=d_g,
        idiosyncratic_dim=d_u,
        feature_fn=feature_phi,
        A_g=A_g,
        Q_g=Q_g,
        A_u=A_u,
        Q_u=Q_u,
        B=B,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=lambda_scalar,
        staleness_threshold=staleness_threshold,
        exploration_mode=exploration_mode,
        feature_mode=feature_mode_eff,
        feature_learning_rate=feature_lr_eff,
        feature_freeze_after=feature_freeze_after,
        feature_log_interval=feature_log_interval,
        feedback_mode="full",
        eps=eps_corr,
        g_mean0=g_mean0,
        g_cov0=g_cov0,
        u_mean0=u_mean0,
        u_cov0=u_cov0,
        feature_arch=feature_arch_eff,
        feature_hidden_dim=feature_hidden_dim,
        feature_activation=feature_activation,
    )

    return router_partial_corr, router_full_corr


def evaluate_config(
    lambda_risk: float,
    q_g_scale: float,
    q_u_scale: float,
    r_scale: float,
    num_regimes: int,
    seeds: List[int],
    cfg: Dict,
    feature_mode: str | None = None,
    feature_learning_rate: float | None = None,
    feature_arch: str | None = None,
    shared_dim: int | None = None,
    idiosyncratic_dim: int | None = None,
) -> Tuple[float, float]:
    """
    Average correlated-router costs over a list of environment seeds
    for one hyperparameter configuration.

    Returns (avg_cost_partial_corr, avg_cost_full_corr).
    """
    costs_partial = []
    costs_full = []

    for seed in seeds:
        env = build_environment(seed, cfg, num_regimes=num_regimes)
        router_partial_corr, router_full_corr = build_correlated_routers(
            lambda_risk=lambda_risk,
            q_g_scale=q_g_scale,
            q_u_scale=q_u_scale,
            r_scale=r_scale,
            cfg=cfg,
            num_regimes=num_regimes,
            feature_mode=feature_mode,
            feature_learning_rate=feature_learning_rate,
            feature_arch=feature_arch,
            shared_dim=shared_dim,
            idiosyncratic_dim=idiosyncratic_dim,
        )

        c_partial, _ = run_router_on_env(router_partial_corr, env)
        c_full, _ = run_router_on_env(router_full_corr, env)

        costs_partial.append(float(c_partial.mean()))
        costs_full.append(float(c_full.mean()))

    return float(np.mean(costs_partial)), float(np.mean(costs_full))


def _eval_hyperparam_task(
    task: Tuple[
        int,
        float,
        float,
        float,
        float,
        str,
        float,
        str,
        int,
        int,
        List[int],
        Dict,
    ],
) -> Dict[str, float]:
    """
    Worker function for parallel hyperparameter evaluation.

    Parameters
    ----------
    task : (M, lambda_risk, q_g_scale, q_u_scale, r_scale,
            feature_mode, feature_lr, feature_arch,
            shared_dim, idiosyncratic_dim, seeds, cfg)

    Returns
    -------
    dict with keys:
      - num_regimes, lambda_risk, q_g_scale, q_u_scale, r_scale,
      - feature_mode, feature_learning_rate, feature_arch,
      - shared_dim, idiosyncratic_dim,
      - avg_cost_partial, avg_cost_full
    """
    (
        M,
        lambda_risk,
        q_g_scale,
        q_u_scale,
        r_scale,
        feature_mode,
        feature_lr,
        feature_arch,
        shared_dim,
        id_dim,
        seeds,
        cfg,
    ) = task
    avg_partial, avg_full = evaluate_config(
        lambda_risk=lambda_risk,
        q_g_scale=q_g_scale,
        q_u_scale=q_u_scale,
        r_scale=r_scale,
        num_regimes=M,
        seeds=seeds,
        cfg=cfg,
        feature_mode=feature_mode,
        feature_learning_rate=feature_lr,
        feature_arch=feature_arch,
        shared_dim=shared_dim,
        idiosyncratic_dim=id_dim,
    )
    return {
        "num_regimes": int(M),
        "lambda_risk": float(lambda_risk),
        "q_g_scale": float(q_g_scale),
        "q_u_scale": float(q_u_scale),
        "r_scale": float(r_scale),
        "feature_mode": str(feature_mode),
        "feature_learning_rate": float(feature_lr),
        "feature_arch": str(feature_arch),
        "shared_dim": int(shared_dim),
        "idiosyncratic_dim": int(id_dim),
        "avg_cost_partial": float(avg_partial),
        "avg_cost_full": float(avg_full),
    }


def run_hyperparam_search(config_path: str = "config.yaml") -> None:
    """
    Grid search over a small, tailored set of SLDS+IMM correlated-router
    hyperparameters, using the experiment defined in `config.yaml`.

    Hyperparameters:
      - num_regimes: number of discrete regimes M
      - lambda_risk: scalar risk sensitivity in the router score
      - q_scale: global scaling of Q_g and Q_u (process noise)
      - r_scale: global scaling of R (observation noise)

    The grids can be customized via an optional `hyperparam_search`
    section in the config, e.g.:

      hyperparam_search:
        num_regimes_grid: [2, 3, 5]
        lambda_risk_grid: [-0.5, -0.2, 0.0, 0.2]
        q_scale_grid: [0.5, 1.0, 2.0]
        r_scale_grid: [0.5, 1.0, 2.0]
    """
    cfg = _load_config(config_path)
    search_cfg = cfg.get("hyperparam_search", {})

    # Base number of regimes and state dimension from the experiment config
    _, d_base, M_base, _ = _get_dims_from_config(cfg)

    routers_cfg = cfg.get("routers", {})
    slds_corr_cfg = routers_cfg.get("slds_imm_corr", {}) or {}

    # Grids for hyperparameters (can be overridden via config). We treat
    # q_g_scale, q_u_scale, and r_scale as absolute values (typically
    # variances) and use log-spaced defaults for robustness.
    lambda_risk_grid = search_cfg.get(
        "lambda_risk_grid", [-1.0, -0.5, 0.0, 0.5]
    )
    # Global default value grid; can be shared by q_g and q_u if
    # component-specific grids are not provided.
    q_value_grid_default = search_cfg.get(
        "q_scale_grid", [1e-3, 1e-2, 1e-1, 1.0]
    )
    q_g_scale_grid = search_cfg.get("q_g_scale_grid", q_value_grid_default)
    q_u_scale_grid = search_cfg.get("q_u_scale_grid", q_value_grid_default)
    r_scale_grid = search_cfg.get("r_scale_grid", [1e-3, 1e-2, 1e-1, 1.0])

    # Grids for feature and architecture hyperparameters (optional).
    feature_mode_default = slds_corr_cfg.get("feature_mode", "fixed")
    feature_mode_grid = search_cfg.get(
        "feature_mode_grid", [feature_mode_default]
    )
    feature_lr_default = float(slds_corr_cfg.get("feature_learning_rate", 0.0))
    feature_lr_grid = search_cfg.get(
        "feature_learning_rate_grid", [feature_lr_default]
    )
    feature_arch_default = slds_corr_cfg.get("feature_arch", "linear")
    feature_arch_grid = search_cfg.get(
        "feature_arch_grid", [feature_arch_default]
    )

    shared_dim_default = int(slds_corr_cfg.get("shared_dim", 1))
    shared_dim_grid = search_cfg.get(
        "shared_dim_grid", [shared_dim_default]
    )
    id_dim_default = int(slds_corr_cfg.get("idiosyncratic_dim", d_base))
    id_dim_grid = search_cfg.get(
        "idiosyncratic_dim_grid", [id_dim_default]
    )

    # Regime-count grid: if not provided, search over a small set around M_base.
    num_regimes_grid_raw = search_cfg.get("num_regimes_grid", None)
    if num_regimes_grid_raw is None:
        candidates = {max(2, M_base)}
        if M_base > 2:
            candidates.add(max(2, M_base - 1))
        else:
            candidates.add(M_base + 1)
        num_regimes_grid = sorted(candidates)
    else:
        num_regimes_grid = sorted(
            {max(2, int(m)) for m in num_regimes_grid_raw}
        )
        if not num_regimes_grid:
            num_regimes_grid = [max(2, M_base)]

    # Seeds for averaging: can be overridden via hyperparam_search.seeds.
    seeds_cfg = search_cfg.get("seeds", None)
    if seeds_cfg is None:
        seeds = [42, 43, 44, 45, 46]
    else:
        seeds = [int(s) for s in seeds_cfg]

    # Optional number of worker processes for parallel evaluation.
    # If num_workers <= 0 or omitted, use all available CPUs.
    num_workers_cfg = search_cfg.get("num_workers", None)
    if num_workers_cfg is None:
        num_workers = os.cpu_count() or 1
    else:
        try:
            nw_raw = int(num_workers_cfg)
        except (TypeError, ValueError):
            nw_raw = 0
        if nw_raw <= 0:
            num_workers = os.cpu_count() or 1
        else:
            num_workers = nw_raw

    # Prepare all tasks
    tasks: List[Tuple[
        int,
        float,
        float,
        float,
        float,
        str,
        float,
        str,
        int,
        int,
        List[int],
        Dict,
    ]] = []
    for M in num_regimes_grid:
        for (
            lambda_risk,
            q_g_scale,
            q_u_scale,
            r_scale,
            feature_mode,
            feature_lr,
            feature_arch,
            shared_dim,
            id_dim,
        ) in itertools.product(
            lambda_risk_grid,
            q_g_scale_grid,
            q_u_scale_grid,
            r_scale_grid,
            feature_mode_grid,
            feature_lr_grid,
            feature_arch_grid,
            shared_dim_grid,
            id_dim_grid,
        ):
            # Skip incompatible combinations: in "fixed" feature_mode we
            # require idiosyncratic_dim to equal the base state_dim so
            # that φ(x) and u_{j,t} have matching dimensions.
            if str(feature_mode) == "fixed" and int(id_dim) != int(d_base):
                continue
            tasks.append(
                (
                    M,
                    float(lambda_risk),
                    float(q_g_scale),
                    float(q_u_scale),
                    float(r_scale),
                     str(feature_mode),
                     float(feature_lr),
                     str(feature_arch),
                     int(shared_dim),
                     int(id_dim),
                    seeds,
                    cfg,
                )
            )

    results: List[Dict[str, float]] = []

    if not tasks:
        print("No valid hyperparameter combinations to evaluate.")
        return

    total_tasks = len(tasks)

    # Parallel or sequential evaluation across all hyperparameter
    # combinations, depending on num_workers. We print each result as
    # soon as it is available so long runs do not appear stuck.
    if num_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            result = _eval_hyperparam_task(task)
            results.append(result)
            print(
                f"[{idx}/{total_tasks}] Config:",
                f"M={int(result['num_regimes']):d},",
                f"lambda_risk={result['lambda_risk']:+.2f},",
                f"q_g_scale={result['q_g_scale']:.4f},",
                f"q_u_scale={result['q_u_scale']:.4f},",
                f"r_scale={result['r_scale']:.4f},",
                f"feature_mode={result['feature_mode']},",
                f"feature_lr={result['feature_learning_rate']:.4e},",
                f"feature_arch={result['feature_arch']},",
                f"shared_dim={result['shared_dim']},",
                f"id_dim={result['idiosyncratic_dim']}",
                "|",
                f"partial_corr={result['avg_cost_partial']:.4f}, "
                f"full_corr={result['avg_cost_full']:.4f}",
            )
    else:
        max_workers = min(num_workers, total_tasks)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, result in enumerate(
                executor.map(_eval_hyperparam_task, tasks), start=1
            ):
                results.append(result)
                print(
                    f"[{idx}/{total_tasks}] Config:",
                    f"M={int(result['num_regimes']):d},",
                    f"lambda_risk={result['lambda_risk']:+.2f},",
                    f"q_g_scale={result['q_g_scale']:.4f},",
                    f"q_u_scale={result['q_u_scale']:.4f},",
                    f"r_scale={result['r_scale']:.4f},",
                    f"feature_mode={result['feature_mode']},",
                    f"feature_lr={result['feature_learning_rate']:.4e},",
                    f"feature_arch={result['feature_arch']},",
                    f"shared_dim={result['shared_dim']},",
                    f"id_dim={result['idiosyncratic_dim']}",
                    "|",
                    f"partial_corr={result['avg_cost_partial']:.4f}, "
                    f"full_corr={result['avg_cost_full']:.4f}",
                )

    # Sort results by partial-feedback correlated router performance
    results_sorted_partial = sorted(
        results,
        key=lambda r: r["avg_cost_partial"],
    )
    best_partial = results_sorted_partial[0]

    # Sort results by full-feedback correlated router performance
    results_sorted_full = sorted(
        results,
        key=lambda r: r["avg_cost_full"],
    )
    best_full = results_sorted_full[0]

    print("\n=== Best config (correlated, partial feedback) ===")
    print(
        f"M={int(best_partial['num_regimes'])}, "
        f"lambda_risk={best_partial['lambda_risk']:+.2f}, "
        f"q_g_scale={best_partial['q_g_scale']:.4f}, "
        f"q_u_scale={best_partial['q_u_scale']:.4f}, "
        f"r_scale={best_partial['r_scale']:.4f}, "
        f"feature_mode={best_partial['feature_mode']}, "
        f"feature_lr={best_partial['feature_learning_rate']:.4e}, "
        f"feature_arch={best_partial['feature_arch']}, "
        f"shared_dim={best_partial['shared_dim']}, "
        f"idiosyncratic_dim={best_partial['idiosyncratic_dim']} | "
        f"avg_cost_partial_corr={best_partial['avg_cost_partial']:.4f}"
    )

    print("\n=== Best config (correlated, full feedback) ===")
    print(
        f"M={int(best_full['num_regimes'])}, "
        f"lambda_risk={best_full['lambda_risk']:+.2f}, "
        f"q_g_scale={best_full['q_g_scale']:.4f}, "
        f"q_u_scale={best_full['q_u_scale']:.4f}, "
        f"r_scale={best_full['r_scale']:.4f}, "
        f"feature_mode={best_full['feature_mode']}, "
        f"feature_lr={best_full['feature_learning_rate']:.4e}, "
        f"feature_arch={best_full['feature_arch']}, "
        f"shared_dim={best_full['shared_dim']}, "
        f"idiosyncratic_dim={best_full['idiosyncratic_dim']} | "
        f"avg_cost_full_corr={best_full['avg_cost_full']:.4f}"
    )

    # Save best hyperparameters to a dedicated file for later use.
    best_params = {
        "partial": {
            "num_regimes": int(best_partial["num_regimes"]),
            "lambda_risk": float(best_partial["lambda_risk"]),
            "q_g": float(best_partial["q_g_scale"]),
            "q_u": float(best_partial["q_u_scale"]),
            "r": float(best_partial["r_scale"]),
            "feature_mode": str(best_partial["feature_mode"]),
            "feature_learning_rate": float(best_partial["feature_learning_rate"]),
            "feature_arch": str(best_partial["feature_arch"]),
            "shared_dim": int(best_partial["shared_dim"]),
            "idiosyncratic_dim": int(best_partial["idiosyncratic_dim"]),
            "avg_cost": float(best_partial["avg_cost_partial"]),
        },
        "full": {
            "num_regimes": int(best_full["num_regimes"]),
            "lambda_risk": float(best_full["lambda_risk"]),
            "q_g": float(best_full["q_g_scale"]),
            "q_u": float(best_full["q_u_scale"]),
            "r": float(best_full["r_scale"]),
            "feature_mode": str(best_full["feature_mode"]),
            "feature_learning_rate": float(best_full["feature_learning_rate"]),
            "feature_arch": str(best_full["feature_arch"]),
            "shared_dim": int(best_full["shared_dim"]),
            "idiosyncratic_dim": int(best_full["idiosyncratic_dim"]),
            "avg_cost": float(best_full["avg_cost_full"]),
        },
    }
    os.makedirs("param", exist_ok=True)
    out_path = os.path.join("param", "best_corr_hyperparams.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nSaved best hyperparameters to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter search for the correlated SLDS-IMM router "
            "using the experiment defined in a config file."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML/JSON configuration file (default: config.yaml).",
    )
    args = parser.parse_args()
    run_hyperparam_search(args.config)
