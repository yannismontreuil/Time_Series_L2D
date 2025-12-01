import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, List

# Limit internal BLAS/OpenMP threading so that using multiple Python
# processes actually speeds things up rather than oversubscribing CPU
# cores. These must be set before importing NumPy / SciPy / PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from router_model_corr import SLDSIMMRouter_Corr, feature_phi
from synthetic_env import SyntheticTimeSeriesEnv
from etth1_env import ETTh1TimeSeriesEnv
from router_eval import run_router_on_env

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


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

    if data_source == "etth1":
        # Real-world ETTh1 experiment (oil temperature as target).
        T_raw = env_cfg.get("T", None)
        T_env = None if T_raw is None else int(T_raw)
        csv_path = env_cfg.get("csv_path", "Data/ETTh1.csv")
        target_column = env_cfg.get("target_column", "OT")

        return ETTh1TimeSeriesEnv(
            csv_path=csv_path,
            target_column=target_column,
            num_experts=N,
            num_regimes=M,
            T=T_env,
            seed=int(seed),
            unavailable_expert_idx=env_cfg.get("unavailable_expert_idx", None),
            unavailable_intervals=env_cfg.get("unavailable_intervals", None),
            arrival_expert_idx=env_cfg.get("arrival_expert_idx", None),
            arrival_intervals=env_cfg.get("arrival_intervals", None),
        )

    # Synthetic environment (default)
    return SyntheticTimeSeriesEnv(
        num_experts=N,
        num_regimes=M,
        T=int(env_cfg.get("T", 300)),
        seed=int(seed),
        unavailable_expert_idx=int(env_cfg.get("unavailable_expert_idx", 1)),
        unavailable_intervals=env_cfg.get(
            "unavailable_intervals", [[10, 50], [200, 250]]
        ),
        arrival_expert_idx=int(env_cfg.get("arrival_expert_idx", 4)),
        arrival_intervals=env_cfg.get("arrival_intervals", [[120, 200]]),
        setting=setting,
        noise_scale=env_cfg.get("noise_scale", None),
    )


def build_correlated_routers(
    lambda_risk: float,
    q_g_scale: float,
    q_u_scale: float,
    r_scale: float,
    cfg: Dict,
    num_regimes: int | None = None,
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

    # Dimensions for correlated router
    d_g = int(slds_corr_cfg.get("shared_dim", 1))
    d_u = int(slds_corr_cfg.get("idiosyncratic_dim", d))

    # Independent-router R (shared with correlated router)
    R_cfg = slds_cfg.get("R", None)
    if R_cfg is not None:
        R_base = np.asarray(R_cfg, dtype=float)
    else:
        r_scalar = float(slds_cfg.get("R_scalar", 0.5))
        R_base = np.full((M, N), r_scalar, dtype=float)
    R = R_base * float(r_scale)

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

    staleness_threshold_cfg = routers_cfg.get("staleness_threshold", None)
    staleness_threshold = (
        int(staleness_threshold_cfg) if staleness_threshold_cfg is not None else None
    )

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
        feedback_mode="full",
        eps=eps_corr,
        g_mean0=g_mean0,
        g_cov0=g_cov0,
        u_mean0=u_mean0,
        u_cov0=u_cov0,
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
        )

        c_partial, _ = run_router_on_env(router_partial_corr, env)
        c_full, _ = run_router_on_env(router_full_corr, env)

        costs_partial.append(float(c_partial.mean()))
        costs_full.append(float(c_full.mean()))

    return float(np.mean(costs_partial)), float(np.mean(costs_full))


def _eval_hyperparam_task(
    task: Tuple[int, float, float, float, float, List[int], Dict],
) -> Dict[str, float]:
    """
    Worker function for parallel hyperparameter evaluation.

    Parameters
    ----------
    task : (M, lambda_risk, q_g_scale, q_u_scale, r_scale, seeds, cfg)

    Returns
    -------
    dict with keys:
      - num_regimes, lambda_risk, q_scale, r_scale,
      - avg_cost_partial, avg_cost_full
    """
    M, lambda_risk, q_g_scale, q_u_scale, r_scale, seeds, cfg = task
    avg_partial, avg_full = evaluate_config(
        lambda_risk=lambda_risk,
        q_g_scale=q_g_scale,
        q_u_scale=q_u_scale,
        r_scale=r_scale,
        num_regimes=M,
        seeds=seeds,
        cfg=cfg,
    )
    return {
        "num_regimes": int(M),
        "lambda_risk": float(lambda_risk),
        "q_g_scale": float(q_g_scale),
        "q_u_scale": float(q_u_scale),
        "r_scale": float(r_scale),
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

    # Base number of regimes from the experiment config
    _, _, M_base, _ = _get_dims_from_config(cfg)

    # Grids for hyperparameters (can be overridden via config)
    lambda_risk_grid = search_cfg.get(
        "lambda_risk_grid", [-0.5, -0.2, 0.0, 0.2]
    )
    # Global default scale grid; can be shared by q_g and q_u if
    # component-specific grids are not provided.
    q_scale_grid_default = search_cfg.get("q_scale_grid", [0.5, 1.0, 2.0])
    q_g_scale_grid = search_cfg.get("q_g_scale_grid", q_scale_grid_default)
    q_u_scale_grid = search_cfg.get("q_u_scale_grid", q_scale_grid_default)
    r_scale_grid = search_cfg.get("r_scale_grid", [0.5, 1.0, 2.0])

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

    seeds = [42, 43, 44, 45, 46]

    # Optional number of worker processes for parallel evaluation.
    # If num_workers <= 0 or omitted, use all available CPUs.
    num_workers_cfg = search_cfg.get("num_workers", None)
    if num_workers_cfg is None:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = max(1, int(num_workers_cfg))

    # Prepare all tasks
    tasks: List[Tuple[int, float, float, float, float, List[int], Dict]] = []
    for M in num_regimes_grid:
        for lambda_risk, q_g_scale, q_u_scale, r_scale in itertools.product(
            lambda_risk_grid,
            q_g_scale_grid,
            q_u_scale_grid,
            r_scale_grid,
        ):
            tasks.append(
                (
                    M,
                    float(lambda_risk),
                    float(q_g_scale),
                    float(q_u_scale),
                    float(r_scale),
                    seeds,
                    cfg,
                )
            )

    results: List[Dict[str, float]] = []

    # Parallel evaluation across CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_eval_hyperparam_task, task) for task in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)

            print(
                "Config:",
                f"M={int(result['num_regimes']):d},",
                f"lambda_risk={result['lambda_risk']:+.2f},",
                f"q_g_scale={result['q_g_scale']:.2f},",
                f"q_u_scale={result['q_u_scale']:.2f},",
                f"r_scale={result['r_scale']:.2f}",
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
        f"q_g_scale={best_partial['q_g_scale']:.2f}, "
        f"q_u_scale={best_partial['q_u_scale']:.2f}, "
        f"r_scale={best_partial['r_scale']:.2f} | "
        f"avg_cost_partial_corr={best_partial['avg_cost_partial']:.4f}"
    )

    print("\n=== Best config (correlated, full feedback) ===")
    print(
        f"M={int(best_full['num_regimes'])}, "
        f"lambda_risk={best_full['lambda_risk']:+.2f}, "
        f"q_g_scale={best_full['q_g_scale']:.2f}, "
        f"q_u_scale={best_full['q_u_scale']:.2f}, "
        f"r_scale={best_full['r_scale']:.2f} | "
        f"avg_cost_full_corr={best_full['avg_cost_full']:.4f}"
    )

    print(
        "\nYou can adapt your YAML by, e.g.,\n"
        "  environment:\n"
        f"    num_regimes: {int(best_full['num_regimes'])}\n"
        "  routers:\n"
        "    lambda_risk: "
        f"{best_full['lambda_risk']:+.2f}\n"
        "    slds_imm:\n"
        "      # multiply existing Q_scales / R_scalar by q_scale / r_scale\n"
        f"      # recommended q_g_scale = {best_full['q_g_scale']:.2f}, "
        f"q_u_scale = {best_full['q_u_scale']:.2f}, "
        f"r_scale = {best_full['r_scale']:.2f}"
    )


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
