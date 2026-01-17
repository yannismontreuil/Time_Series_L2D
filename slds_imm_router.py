# %%
# %matplotlib widget

import argparse
import copy
import json
import os
import random
import multiprocessing as mp
import subprocess
import sys
from typing import Optional
import numpy as np

from environment.etth1_env import ETTh1TimeSeriesEnv

from models.router_model import SLDSIMMRouter, feature_phi
from models.router_model_corr import SLDSIMMRouter_Corr, RecurrentSLDSIMMRouter_Corr
from models.router_model_corr_em import SLDSIMMRouter_Corr_EM
from environment.synthetic_env import SyntheticTimeSeriesEnv
from models.l2d_baseline import L2D, L2D_SW
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB
from models.factorized_slds import FactorizedSLDS

from router_eval import set_transition_log_config, register_transition_log_label

from plot_utils import (
    evaluate_routers_and_baselines,
    analysis_late_arrival,
    plot_time_series,
)
from horizon_planning import evaluate_horizon_planning

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _load_config(path: str = "config/config.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f: # "r" means read mode
        text = f.read() # by default reads config.yaml
    if yaml is not None:
        data = yaml.safe_load(text)
        print(f"Loaded configuration from {path} using YAML parser.")
    else:
        data = json.loads(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Top-level configuration must be a mapping.")
    return data


def _resolve_vector(value, default_scalar: float, length: int) -> np.ndarray:
    """
    Turn vatious types of configuration input into a Numpy vector with correct length.
    """
    if value is None:
        return np.full(length, default_scalar, dtype=float)
    # value provided: parse and validate
    arr = np.asarray(value, dtype=float)
    if arr.shape == (): # scalar
        return np.full(length, float(arr), dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"Expected vector of length {length}, got shape {arr.shape}.")
    return arr


def _collect_factorized_em_data(
    router: FactorizedSLDS,
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    t_end: int,
    force_full_feedback: bool = True,
):
    contexts = []
    available_sets = []
    actions = []
    residuals = []
    residuals_full = [] if force_full_feedback else None

    prev_suspend = getattr(router, "_online_em_suspended", False)
    prev_feedback_mode = router.feedback_mode
    prev_exploration = router.exploration
    prev_sampling_det = router.exploration_sampling_deterministic
    prev_rng_state = copy.deepcopy(router._rng.bit_generator.state)
    if hasattr(router, "suspend_online_em"):
        router.suspend_online_em(True)
    router.reset_beliefs()
    # Force deterministic policy during warmup so that partial/full
    # routers collect identical datasets.
    if force_full_feedback:
        router.feedback_mode = "full"
    router.exploration = "g"
    router.exploration_sampling_deterministic = True
    # Reset RNG state so diagnostics are consistent across routers.
    router._rng.bit_generator.state = np.random.default_rng(0).bit_generator.state
    t_end = min(int(t_end), env.T - 1)
    try:
        for t in range(1, t_end + 1):
            x_t = env.get_context(t)
            available = np.asarray(env.get_available_experts(t), dtype=int)
            if available.size == 0:
                raise ValueError(f"EM warmup: no available experts at t={t}.")
            r_t = int(np.min(available))

            preds = env.all_expert_predictions(x_t)
            residuals_all = preds - float(env.y[t])
            residual = float(residuals_all[int(r_t)])
            if force_full_feedback:
                residuals_masked = np.full(residuals_all.shape, np.nan, dtype=float)
                residuals_masked[available] = residuals_all[available]
            else:
                residuals_masked = None

            contexts.append(x_t)
            available_sets.append(list(available))
            actions.append(int(r_t))
            residuals.append(residual)
            if force_full_feedback:
                residuals_full.append(residuals_masked)
    finally:
        router.feedback_mode = prev_feedback_mode
        router.exploration = prev_exploration
        router.exploration_sampling_deterministic = prev_sampling_det
        router._rng.bit_generator.state = prev_rng_state
        if hasattr(router, "suspend_online_em"):
            router.suspend_online_em(prev_suspend)

    return contexts, available_sets, actions, residuals, residuals_full


def _clone_factorized_em_data(
    data: tuple[
        list[np.ndarray],
        list[list[int]],
        list[int],
        list[float],
        Optional[list[np.ndarray]],
    ],
):
    ctx, avail, actions, residuals, residuals_full = data
    ctx_out = [np.asarray(x, dtype=float).copy() for x in ctx]
    avail_out = [list(a) for a in avail]
    actions_out = [int(a) for a in actions]
    residuals_out = [float(r) for r in residuals]
    if residuals_full is None:
        residuals_full_out = None
    else:
        residuals_full_out = [np.asarray(r, dtype=float).copy() for r in residuals_full]
    return ctx_out, avail_out, actions_out, residuals_out, residuals_full_out


def _evaluate_factorized_run_worker(payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    transition_cfg = payload.get("transition_log_cfg", None)
    if transition_cfg is not None:
        transition_cfg = dict(transition_cfg)
        transition_cfg["plot_show"] = False
    set_transition_log_config(transition_cfg)
    evaluate_routers_and_baselines(
        payload["env"],
        payload["router_partial"],
        payload["router_full"],
        payload["router_factorial_partial"],
        payload["router_factorial_full"],
        factorized_label=payload["factorized_label"],
        router_factorial_partial_linear=payload["router_factorial_partial_linear"],
        router_factorial_full_linear=payload["router_factorial_full_linear"],
        factorized_linear_label=payload["factorized_linear_label"],
        l2d_baseline=payload.get("l2d_baseline", None),
        l2d_sw_baseline=payload.get("l2d_sw_baseline", None),
        linucb_partial=payload.get("linucb_partial", None),
        linucb_full=payload.get("linucb_full", None),
        neuralucb_partial=payload.get("neuralucb_partial", None),
        neuralucb_full=payload.get("neuralucb_full", None),
        seed=int(payload.get("seed", 0)),
    )


def _run_pipeline_subprocess(payload: dict) -> None:
    config_path = payload["config_path"]
    exploration_mode = payload["exploration_mode"]
    disable_show = payload.get("disable_show", True)
    env = os.environ.copy()
    env["FACTOR_EXPLORATION_MODE"] = exploration_mode
    if disable_show:
        env["FACTOR_DISABLE_PLOT_SHOW"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--config", config_path]
    subprocess.run(cmd, check=True, env=env)


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Example:
        python slds_imm_router.py --config path/to/config.yaml
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run SLDS-IMM routers and baselines on either the synthetic "
            "environment or the ETTh1 real-world dataset, depending on the "
            "provided configuration."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to YAML/JSON configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--late-arrival-analysis",
        action="store_true",
        help=(
            "If set, run late-arrival analysis for the configured "
            "arrival_expert_idx (if any), printing adoption metrics "
            "and plotting reaction to the new expert."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = _load_config(args.config)
    transition_log_cfg = cfg.get("transition_log", None)
    if transition_log_cfg is None:
        transition_log_cfg = cfg.get("debug", {}).get("transition_log", None)
    if os.environ.get("FACTOR_DISABLE_PLOT_SHOW") == "1" and isinstance(
        transition_log_cfg, dict
    ):
        transition_log_cfg = dict(transition_log_cfg)
        transition_log_cfg["plot_show"] = False
    if isinstance(transition_log_cfg, dict) and transition_log_cfg.get("enabled", False):
        set_transition_log_config(transition_log_cfg)
    else:
        transition_log_cfg = None
    env_cfg = cfg.get("environment", {})
    seed_cfg = env_cfg.get("seed", 0)
    seed = int(seed_cfg) if seed_cfg is not None else 0
    np.random.seed(seed)
    random.seed(seed)
    try:  # pragma: no cover - torch is optional
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    routers_cfg = cfg.get("routers", {})
    slds_cfg = routers_cfg.get("slds_imm", {}) or {}
    slds_corr_cfg = routers_cfg.get("slds_imm_corr", {}) or {}
    slds_corr_enabled = bool(slds_corr_cfg.get("enabled", True))
    factorized_slds_cfg_raw = routers_cfg.get("factorized_slds", None)
    if factorized_slds_cfg_raw is None:
        factorized_slds_cfg = {}
    else:
        if not isinstance(factorized_slds_cfg_raw, dict):
            raise ValueError("routers.factorized_slds must be a mapping when provided.")
        factorized_slds_cfg = factorized_slds_cfg_raw
    factorized_slds_enabled = bool(factorized_slds_cfg.get("enabled", True))
    exploration_override = os.environ.get("FACTOR_EXPLORATION_MODE", None)
    if exploration_override:
        factorized_slds_cfg = dict(factorized_slds_cfg)
        factorized_slds_cfg["exploration"] = exploration_override

    if factorized_slds_enabled and not exploration_override:
        parallel_pipeline = bool(factorized_slds_cfg.get("parallel_pipeline", False))
        if parallel_pipeline:
            exploration_cfg = factorized_slds_cfg.get("exploration", "g")
            if isinstance(exploration_cfg, (list, tuple)):
                exploration_modes = [str(x).lower() for x in exploration_cfg]
            else:
                exploration_modes = [str(exploration_cfg).lower()]
            exploration_modes = [m for m in exploration_modes if m]
            exploration_modes = exploration_modes or ["g"]
            if len(exploration_modes) > 1:
                workers_cfg = factorized_slds_cfg.get("parallel_pipeline_workers", None)
                disable_show = bool(
                    factorized_slds_cfg.get("parallel_pipeline_disable_plot_show", True)
                )
                max_workers = (
                    min(len(exploration_modes), int(workers_cfg))
                    if workers_cfg is not None
                    else min(len(exploration_modes), os.cpu_count() or 1)
                )
                payloads = [
                    {
                        "config_path": args.config,
                        "exploration_mode": mode,
                        "disable_show": disable_show,
                    }
                    for mode in exploration_modes
                ]
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=max_workers) as pool:
                    pool.map(_run_pipeline_subprocess, payloads)
                sys.exit(0)

    baselines_cfg = cfg.get("baselines", {})
    l2d_cfg = baselines_cfg.get("l2d", {})
    l2d_sw_cfg = baselines_cfg.get("l2d_sw", {})
    linucb_cfg = baselines_cfg.get("linucb", {})
    neuralucb_cfg = baselines_cfg.get("neural_ucb", {})
    horizon_cfg = cfg.get("horizon_planning", {})

    # Model dimensions and core hyperparameters
    setting = env_cfg.get("setting", "easy_setting")
    data_source = env_cfg.get("data_source", "synthetic")
    # Default: universe of 5 experts indexed j=0,...,4.
    N = int(env_cfg.get("num_experts"))   # experts
    # State dimension (= dim φ(x)); feature map in router_model.py currently
    # returns a 2D feature, so d must be compatible with that.
    d = int(env_cfg.get("state_dim"))

    # Number of regimes M. For the "noisy_forgetting" setting we
    # automatically use at least 5 regimes to create a more challenging
    # Markovian pattern, unless the config explicitly requests more.
    if setting == "noisy_forgetting":
        default_M = 5
        raw_M = env_cfg.get("num_regimes", None)
        if raw_M is None:
            M = default_M
        else:
            M = max(int(raw_M), default_M)
    else:
        M = int(env_cfg.get("num_regimes", 2))
    print(f"Using num_regimes = {M}.")

    # Risk sensitivity λ; can be scalar or length-M vector. If a
    # 2-element vector is provided while num_regimes > 2, we interpret
    # it as [λ_0, λ_other] and broadcast λ_other to regimes 1,...,M-1.
    lambda_cfg = routers_cfg.get("lambda_risk", -0.2)
    lambda_arr = np.asarray(lambda_cfg, dtype=float)
    if lambda_arr.shape == ():
        lambda_risk = float(lambda_arr)
    elif lambda_arr.shape == (M,): # one dimension array with length M
        lambda_risk = lambda_arr
    elif lambda_arr.size == 2 and M > 2:
        lambda_broadcast = np.empty(M, dtype=float)
        lambda_broadcast[0] = float(lambda_arr[0])
        lambda_broadcast[1:] = float(lambda_arr[1])
        lambda_risk = lambda_broadcast
    else:
        # Fallback: treat any other shape as scalar by averaging.
        lambda_risk = float(lambda_arr.mean())

    # Consultation costs for routers (shared across experts by default)
    beta = _resolve_vector(routers_cfg.get("beta", None), 0.0, N)

    # --------------------------------------------------------
    # SLDS parameters (independent experts, configurable via YAML)
    # --------------------------------------------------------

    # Transition matrices A_k: either provided explicitly or default to
    # identity for each regime.
    A_cfg = slds_cfg.get("A", None)
    if A_cfg is not None:
        A = np.asarray(A_cfg, dtype=float)
    else:
        A = np.tile(np.eye(d, dtype=float)[None, :, :], (M, 1, 1))
        # shape (M, d, d)
        # identity matrix, the next is about the same as the current state

    # Process noise covariances Q_k: use full matrix if given; otherwise
    # build **diagonal** covariances from per-regime scales Q_scales.
    Q_cfg = slds_cfg.get("Q", None)
    if Q_cfg is not None:
        Q = np.asarray(Q_cfg, dtype=float)
    else:
        q_scales_cfg = slds_cfg.get("Q_scales", None)
        if q_scales_cfg is None:
            if M == 2:
                q_scales_cfg = [0.01, 0.1]
            else:
                q_scales_cfg = 0.01
        q_arr = np.asarray(q_scales_cfg, dtype=float)
        if q_arr.shape == ():
            q_arr = np.full(M, float(q_arr), dtype=float)
        elif q_arr.shape == (M,):
            pass
        elif q_arr.size == 2 and M > 2:
            # Interpret as [q_0, q_other] and broadcast to regimes 1..M-1.
            q_broadcast = np.empty(M, dtype=float)
            q_broadcast[0] = float(q_arr[0])
            q_broadcast[1:] = float(q_arr[1])
            q_arr = q_broadcast
        else:
            raise ValueError(
                "routers.slds_imm.Q_scales must be a scalar, a list of length "
                "num_regimes, or a length-2 list [q_0, q_other] when "
                "num_regimes > 2."
            )
        Q = np.zeros((M, d, d), dtype=float)
        for k in range(M):
            Q[k] = q_arr[k] * np.eye(d, dtype=float)
            #**diagonal** covariances from per-regime scales Q_scales.

    # Observation noise R_{k,j}: observation noise variance for expert j in regime k
    # use full matrix if given;
    # otherwise use a single scalar (broadcast to all regimes/experts).
    R_cfg = slds_cfg.get("R", None)
    if R_cfg is not None:
        R = np.asarray(R_cfg, dtype=float)
    else:
        r_scalar = float(slds_cfg.get("R_scalar", 0.5))
        R = np.full((M, N), r_scalar, dtype=float)
        # M for regimes, N for experts

    # Regime transition matrix Π: if omitted, use the original 2-regime
    # example or a uniform transition for general M.
    Pi_cfg = slds_cfg.get("Pi", None)
    if Pi_cfg is not None:
        Pi_raw = np.asarray(Pi_cfg, dtype=float)
        if Pi_raw.shape == (M, M):
            Pi = Pi_raw
        elif M == 2:
            raise ValueError(
                "routers.slds_imm.Pi must have shape [num_regimes, num_regimes]."
            )
        else:
            # For num_regimes > 2 and mismatched Pi, fall back to a
            # simple uniform transition.
            print("Warning: using uniform transition matrix Pi.")
            Pi = np.full((M, M), 1.0 / M, dtype=float)
    else: # Pi is not provided
        print("Warning: using default transition matrix Pi.")
        if M == 2:
            Pi = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
        else:
            Pi = np.full((M, M), 1.0 / M, dtype=float)

    # Routers for partial and full feedback (independent experts)
    # Optional priors and numerical stabilizer for SLDSIMMRouter
    pop_mean_cfg = slds_cfg.get("pop_mean", None)
    pop_cov_cfg = slds_cfg.get("pop_cov", None)
    eps_slds = float(slds_cfg.get("eps", 1e-8))

    pop_mean = (
        np.asarray(pop_mean_cfg, dtype=float) if pop_mean_cfg is not None else None
    )
    pop_cov = (
        np.asarray(pop_cov_cfg, dtype=float) if pop_cov_cfg is not None else None
    )

    router_partial = SLDSIMMRouter(
        num_experts=N,
        num_regimes=M,
        state_dim=d,
        feature_fn=feature_phi,
        A=A,
        Q=Q,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=lambda_risk,
        feedback_mode="partial",
        pop_mean=pop_mean,
        pop_cov=pop_cov,
        eps=eps_slds,
    )

    router_full = SLDSIMMRouter(
        num_experts=N,
        num_regimes=M,
        state_dim=d,
        feature_fn=feature_phi,
        A=A,
        Q=Q,
        R=R,
        Pi=Pi,
        beta=beta,
        lambda_risk=lambda_risk,
        feedback_mode="full",
        pop_mean=pop_mean,
        pop_cov=pop_cov,
        eps=eps_slds,
    )

    # --------------------------------------------------------
    # Correlated-expert SLDS-IMM routers (shared factor model)
    # --------------------------------------------------------

    staleness_threshold_cfg = routers_cfg.get("staleness_threshold", None)
    staleness_threshold = (
        int(staleness_threshold_cfg) if staleness_threshold_cfg is not None else None
    )
    # Allow mode-specific overrides for the correlated router:
    # routers.slds_imm_corr.partial_overrides and
    # routers.slds_imm_corr.full_overrides.
    # If these keys are absent, both partial and full routers share the same hyperparameters as
    # in the original implementation.
    slds_corr_partial_overrides = slds_corr_cfg.get("partial_overrides", {}) or {}
    slds_corr_full_overrides = slds_corr_cfg.get("full_overrides", {}) or {}

    '''
    See Section 5: Parameter Optimization with Expectation-Maximization (EM)
    E-step: Compute the expected sufficient statistics of the latent variables given the current parameters.
    M-step: Maximize the expected complete-data log-likelihood with respect to the parameters 
            using the statistics from the E-step.
    '''
    # Optional EM-capable correlated router configuration. If present,
    # we build an additional pair of correlated routers that perform an
    # EM-style update of dynamics/noise over an initial window.
    slds_corr_em_cfg = routers_cfg.get("slds_imm_corr_em", {}) or {}
    slds_corr_em_enabled = bool(slds_corr_em_cfg.get("enabled", True))
    slds_corr_em_partial_overrides = slds_corr_em_cfg.get("partial_overrides", {}) or {}
    slds_corr_em_full_overrides = slds_corr_em_cfg.get("full_overrides", {}) or {}

    def _build_corr_router(
        corr_base_cfg: dict,
        overrides: dict,
        feedback_mode: str,
    ) -> SLDSIMMRouter_Corr:
        # Merge base config with mode-specific overrides, ignoring the
        # override containers themselves to avoid accidental reuse.
        cfg_local = dict(corr_base_cfg)
        cfg_local.update(overrides or {})
        cfg_local.pop("partial_overrides", None)
        cfg_local.pop("full_overrides", None)

        # Dimensions can be overridden under routers.slds_imm_corr[…].
        d_g_local = int(cfg_local.get("shared_dim", 1))   # shared-factor dimension
        d_u_local = int(cfg_local.get("idiosyncratic_dim", d))  # idiosyncratic dim

        # Exploration mode for correlated router: "greedy" (risk-adjusted)
        # or "ids" (Information-Directed Sampling). Default: "greedy".
        corr_exploration_mode_local = cfg_local.get("exploration_mode", "greedy")
        # Feature mode for correlated router: "fixed" (use feature_phi) or
        # "learnable" (online feature adaptation).
        corr_feature_mode_local = cfg_local.get("feature_mode", "fixed")
        corr_feature_lr_local = float(cfg_local.get("feature_learning_rate", 0.0))
        corr_feature_freeze_after_cfg = cfg_local.get("feature_freeze_after", None)
        corr_feature_freeze_after_local = (
            int(corr_feature_freeze_after_cfg)
            if corr_feature_freeze_after_cfg is not None
            else None
        )
        corr_feature_log_interval_cfg = cfg_local.get("feature_log_interval", None)
        corr_feature_log_interval_local = (
            int(corr_feature_log_interval_cfg)
            if corr_feature_log_interval_cfg is not None
            else None
        )
        # Optional architecture for learnable features: "linear" or "mlp".
        corr_feature_arch_local = cfg_local.get("feature_arch", "linear")
        corr_feature_hidden_dim_cfg = cfg_local.get("feature_hidden_dim", None)
        corr_feature_hidden_dim_local = (
            int(corr_feature_hidden_dim_cfg)
            if corr_feature_hidden_dim_cfg is not None
            else None
        )
        corr_feature_activation_local = cfg_local.get("feature_activation", "tanh")

        # Joint dynamics for correlated router:
        # - A_gk, A_uk default to identity per regime (unless overridden),
        # - Q_gk, Q_uk built from per-regime scales if full matrices not given.

        A_g_cfg = cfg_local.get("A_g", None)
        if A_g_cfg is not None:
            A_g_local = np.asarray(A_g_cfg, dtype=float)
        else:
            A_g_local = np.tile(np.eye(d_g_local, dtype=float)[None, :, :], (M, 1, 1))

        Q_g_cfg = cfg_local.get("Q_g", None)
        if Q_g_cfg is not None:
            Q_g_local = np.asarray(Q_g_cfg, dtype=float)
        else:
            q_g_scales_cfg = cfg_local.get("Q_g_scales", None)
            if q_g_scales_cfg is None:
                if M == 2:
                    q_g_scales_cfg = [0.01, 0.05]
                else:
                    q_g_scales_cfg = 0.01
            q_g_arr = np.asarray(q_g_scales_cfg, dtype=float)
            if q_g_arr.shape == ():
                q_g_arr = np.full(M, float(q_g_arr), dtype=float)
            elif q_g_arr.shape == (M,):
                pass
            elif q_g_arr.size == 2 and M > 2:
                qg_broadcast = np.empty(M, dtype=float)
                qg_broadcast[0] = float(q_g_arr[0])
                qg_broadcast[1:] = float(q_g_arr[1])
                q_g_arr = qg_broadcast
            else:
                raise ValueError(
                    "routers.slds_imm_corr.Q_g_scales must be a scalar, a list "
                    "of length num_regimes, or a length-2 list [qg_0, qg_other] "
                    "when num_regimes > 2."
                )
            Q_g_local = np.zeros((M, d_g_local, d_g_local), dtype=float)
            for k in range(M):
                Q_g_local[k] = q_g_arr[k] * np.eye(d_g_local, dtype=float)

        A_u_cfg = cfg_local.get("A_u", None)
        if A_u_cfg is not None:
            A_u_local = np.asarray(A_u_cfg, dtype=float)
        else:
            A_u_local = np.tile(np.eye(d_u_local, dtype=float)[None, :, :], (M, 1, 1))

        Q_u_cfg = cfg_local.get("Q_u", None)
        if Q_u_cfg is not None:
            Q_u_local = np.asarray(Q_u_cfg, dtype=float)
        else:
            q_u_scales_cfg = cfg_local.get("Q_u_scales", None)
            if q_u_scales_cfg is None:
                if M == 2:
                    q_u_scales_cfg = [0.01, 0.1]
                else:
                    q_u_scales_cfg = 0.01
            q_u_arr = np.asarray(q_u_scales_cfg, dtype=float)
            if q_u_arr.shape == ():
                q_u_arr = np.full(M, float(q_u_arr), dtype=float)
            elif q_u_arr.shape == (M,):
                pass
            elif q_u_arr.size == 2 and M > 2:
                qu_broadcast = np.empty(M, dtype=float)
                qu_broadcast[0] = float(q_u_arr[0])
                qu_broadcast[1:] = float(q_u_arr[1])
                q_u_arr = qu_broadcast
            else:
                raise ValueError(
                    "routers.slds_imm_corr.Q_u_scales must be a scalar, a list "
                    "of length num_regimes, or a length-2 list [qu_0, qu_other] "
                    "when num_regimes > 2."
                )
            Q_u_local = np.zeros((M, d_u_local, d_u_local), dtype=float)
            for k in range(M):
                Q_u_local[k] = q_u_arr[k] * np.eye(d_u_local, dtype=float)

        # Shared-factor loadings B: either provided explicitly as a full
        # tensor or generated from a single intercept loading value.
        B_cfg = cfg_local.get("B", None)
        if B_cfg is not None:
            B_local = np.asarray(B_cfg, dtype=float)
        else:
            load = float(cfg_local.get("B_intercept_load", 1.0))
            B_local = np.zeros((N, d_u_local, d_g_local), dtype=float)
            for j in range(N):
                B_local[j, 0, 0] = load

        # Optional priors and numerical stabilizer for SLDSIMMRouter_Corr
        eps_corr_local = float(cfg_local.get("eps", 1e-8))
        g_mean0_cfg = cfg_local.get("g_mean0", None)
        g_cov0_cfg = cfg_local.get("g_cov0", None)
        u_mean0_cfg = cfg_local.get("u_mean0", None)
        u_cov0_cfg = cfg_local.get("u_cov0", None)

        g_mean0_local = (
            np.asarray(g_mean0_cfg, dtype=float) if g_mean0_cfg is not None else None
        )
        g_cov0_local = (
            np.asarray(g_cov0_cfg, dtype=float) if g_cov0_cfg is not None else None
        )
        u_mean0_local = (
            np.asarray(u_mean0_cfg, dtype=float) if u_mean0_cfg is not None else None
        )
        u_cov0_local = (
            np.asarray(u_cov0_cfg, dtype=float) if u_cov0_cfg is not None else None
        )

        # Observation noise: optionally override the base SLDS R with
        # a correlated-router-specific scalar or full matrix.
        R_cfg_local = cfg_local.get("R", None)
        if R_cfg_local is not None:
            R_local = np.asarray(R_cfg_local, dtype=float)
        else:
            r_scalar_local = cfg_local.get("R_scalar", None)
            if r_scalar_local is not None:
                R_local = np.full((M, N), float(r_scalar_local), dtype=float)
            else:
                R_local = R

        return SLDSIMMRouter_Corr(
            num_experts=N,
            num_regimes=M,
            shared_dim=d_g_local,
            idiosyncratic_dim=d_u_local,
            feature_fn=feature_phi,
            A_g=A_g_local,
            Q_g=Q_g_local,
            A_u=A_u_local,
            Q_u=Q_u_local,
            B=B_local,
            R=R_local,
            Pi=Pi,
            beta=beta,
            lambda_risk=lambda_risk,
            staleness_threshold=staleness_threshold,
            exploration_mode=corr_exploration_mode_local,
            feature_mode=corr_feature_mode_local,
            feature_learning_rate=corr_feature_lr_local,
            feature_freeze_after=corr_feature_freeze_after_local,
            feature_log_interval=corr_feature_log_interval_local,
            feedback_mode=feedback_mode,
            eps=eps_corr_local,
            g_mean0=g_mean0_local,
            g_cov0=g_cov0_local,
            u_mean0=u_mean0_local,
            u_cov0=u_cov0_local,
            feature_arch=corr_feature_arch_local,
            feature_hidden_dim=corr_feature_hidden_dim_local,
            feature_activation=corr_feature_activation_local,
            seed=seed,
        )
    
    # --------------------------------------------------------
    # EM-capable Correlated-expert SLDS-IMM routers
    # --------------------------------------------------------

    def _build_corr_router_em(
        corr_base_cfg: dict,
        overrides: dict,
        feedback_mode: str,
    ) -> SLDSIMMRouter_Corr_EM:
        """
        Build an EM-capable correlated router. Configuration follows the
        same structure as routers.slds_imm_corr[…], with additional
        keys:
          - em_tk: cutoff time index for EM window (required),
          - em_min_weight: minimum effective weight per regime,
          - em_verbose: whether to log after the M-step.
        """
        cfg_local = dict(corr_base_cfg)
        cfg_local.update(overrides or {})
        cfg_local.pop("partial_overrides", None)
        cfg_local.pop("full_overrides", None)

        d_g_local = int(cfg_local.get("shared_dim", 1))
        d_u_local = int(cfg_local.get("idiosyncratic_dim", d))

        corr_exploration_mode_local = cfg_local.get("exploration_mode", "greedy")
        corr_feature_mode_local = cfg_local.get("feature_mode", "fixed")
        corr_feature_lr_local = float(cfg_local.get("feature_learning_rate", 0.0))
        corr_feature_freeze_after_cfg = cfg_local.get("feature_freeze_after", None)
        corr_feature_freeze_after_local = (
            int(corr_feature_freeze_after_cfg)
            if corr_feature_freeze_after_cfg is not None
            else None
        )
        corr_feature_log_interval_cfg = cfg_local.get("feature_log_interval", None)
        corr_feature_log_interval_local = (
            int(corr_feature_log_interval_cfg)
            if corr_feature_log_interval_cfg is not None
            else None
        )
        corr_feature_arch_local = cfg_local.get("feature_arch", "linear")
        corr_feature_hidden_dim_cfg = cfg_local.get("feature_hidden_dim", None)
        corr_feature_hidden_dim_local = (
            int(corr_feature_hidden_dim_cfg)
            if corr_feature_hidden_dim_cfg is not None
            else None
        )
        corr_feature_activation_local = cfg_local.get("feature_activation", "tanh")

        A_g_cfg = cfg_local.get("A_g", None)
        if A_g_cfg is not None:
            A_g_local = np.asarray(A_g_cfg, dtype=float)
        else:
            A_g_local = np.tile(np.eye(d_g_local, dtype=float)[None, :, :], (M, 1, 1))

        Q_g_cfg = cfg_local.get("Q_g", None)
        if Q_g_cfg is not None:
            Q_g_local = np.asarray(Q_g_cfg, dtype=float)
        else:
            q_g_scales_cfg = cfg_local.get("Q_g_scales", None)
            if q_g_scales_cfg is None:
                if M == 2:
                    q_g_scales_cfg = [0.01, 0.05]
                else:
                    q_g_scales_cfg = 0.01
            q_g_arr = np.asarray(q_g_scales_cfg, dtype=float)
            if q_g_arr.shape == ():
                q_g_arr = np.full(M, float(q_g_arr), dtype=float)
            elif q_g_arr.shape == (M,):
                pass
            elif q_g_arr.size == 2 and M > 2:
                qg_broadcast = np.empty(M, dtype=float)
                qg_broadcast[0] = float(q_g_arr[0])
                qg_broadcast[1:] = float(q_g_arr[1])
                q_g_arr = qg_broadcast
            else:
                raise ValueError(
                    "routers.slds_imm_corr_em.Q_g_scales must be a scalar, a list "
                    "of length num_regimes, or a length-2 list [qg_0, qg_other] "
                    "when num_regimes > 2."
                )
            Q_g_local = np.zeros((M, d_g_local, d_g_local), dtype=float)
            for k in range(M):
                Q_g_local[k] = q_g_arr[k] * np.eye(d_g_local, dtype=float)

        A_u_cfg = cfg_local.get("A_u", None)
        if A_u_cfg is not None:
            A_u_local = np.asarray(A_u_cfg, dtype=float)
        else:
            A_u_local = np.tile(np.eye(d_u_local, dtype=float)[None, :, :], (M, 1, 1))

        Q_u_cfg = cfg_local.get("Q_u", None)
        if Q_u_cfg is not None:
            Q_u_local = np.asarray(Q_u_cfg, dtype=float)
        else:
            q_u_scales_cfg = cfg_local.get("Q_u_scales", None)
            if q_u_scales_cfg is None:
                if M == 2:
                    q_u_scales_cfg = [0.01, 0.1]
                else:
                    q_u_scales_cfg = 0.01
            q_u_arr = np.asarray(q_u_scales_cfg, dtype=float)
            if q_u_arr.shape == ():
                q_u_arr = np.full(M, float(q_u_arr), dtype=float)
            elif q_u_arr.shape == (M,):
                pass
            elif q_u_arr.size == 2 and M > 2:
                qu_broadcast = np.empty(M, dtype=float)
                qu_broadcast[0] = float(q_u_arr[0])
                qu_broadcast[1:] = float(q_u_arr[1])
                q_u_arr = qu_broadcast
            else:
                raise ValueError(
                    "routers.slds_imm_corr_em.Q_u_scales must be a scalar, a list "
                    "of length num_regimes, or a length-2 list [qu_0, qu_other] "
                    "when num_regimes > 2."
                )
            Q_u_local = np.zeros((M, d_u_local, d_u_local), dtype=float)
            for k in range(M):
                Q_u_local[k] = q_u_arr[k] * np.eye(d_u_local, dtype=float)

        B_cfg = cfg_local.get("B", None)
        if B_cfg is not None:
            B_local = np.asarray(B_cfg, dtype=float)
        else:
            load = float(cfg_local.get("B_intercept_load", 1.0))
            B_local = np.zeros((N, d_u_local, d_g_local), dtype=float)
            for j in range(N):
                B_local[j, 0, 0] = load

        eps_corr_local = float(cfg_local.get("eps", 1e-8))
        g_mean0_cfg = cfg_local.get("g_mean0", None)
        g_cov0_cfg = cfg_local.get("g_cov0", None)
        u_mean0_cfg = cfg_local.get("u_mean0", None)
        u_cov0_cfg = cfg_local.get("u_cov0", None)

        g_mean0_local = (
            np.asarray(g_mean0_cfg, dtype=float) if g_mean0_cfg is not None else None
        )
        g_cov0_local = (
            np.asarray(g_cov0_cfg, dtype=float) if g_cov0_cfg is not None else None
        )
        u_mean0_local = (
            np.asarray(u_mean0_cfg, dtype=float) if u_mean0_cfg is not None else None
        )
        u_cov0_local = (
            np.asarray(u_cov0_cfg, dtype=float) if u_cov0_cfg is not None else None
        )

        R_cfg_local = cfg_local.get("R", None)
        if R_cfg_local is not None:
            R_local = np.asarray(R_cfg_local, dtype=float)
        else:
            r_scalar_local = cfg_local.get("R_scalar", None)
            if r_scalar_local is not None:
                R_local = np.full((M, N), float(r_scalar_local), dtype=float)
            else:
                R_local = R

        em_tk_cfg = cfg_local.get("em_tk", None)
        em_tk_local = int(em_tk_cfg) if em_tk_cfg is not None else None
        em_min_weight_local = float(cfg_local.get("em_min_weight", 1e-6))
        em_verbose_local = bool(cfg_local.get("em_verbose", False))

        # Optional feature-learning schedule across phases. Any of these
        # keys can be omitted to fall back to the defaults coded inside
        # SLDSIMMRouter_Corr_EM.
        phase0_t_end_cfg = cfg_local.get("phase0_t_end", None)
        phase0_t_end_local = (
            int(phase0_t_end_cfg) if phase0_t_end_cfg is not None else None
        )
        feature_lr_phase0_cfg = cfg_local.get("feature_lr_phase0", None)
        feature_lr_phase0_local = (
            float(feature_lr_phase0_cfg)
            if feature_lr_phase0_cfg is not None
            else None
        )
        feature_lr_phase1_cfg = cfg_local.get("feature_lr_phase1", None)
        feature_lr_phase1_local = (
            float(feature_lr_phase1_cfg)
            if feature_lr_phase1_cfg is not None
            else None
        )
        feature_lr_phase2_cfg = cfg_local.get("feature_lr_phase2", None)
        feature_lr_phase2_local = (
            float(feature_lr_phase2_cfg)
            if feature_lr_phase2_cfg is not None
            else 0.0
        )

        if em_tk_local is None:
            raise ValueError(
                "routers.slds_imm_corr_em requires an 'em_tk' key specifying "
                "the cutoff time index for EM learning."
            )

        router_em = SLDSIMMRouter_Corr_EM(
            num_experts=N,
            num_regimes=M,
            shared_dim=d_g_local,
            idiosyncratic_dim=d_u_local,
            feature_fn=feature_phi,
            A_g=A_g_local,
            Q_g=Q_g_local,
            A_u=A_u_local,
            Q_u=Q_u_local,
            B=B_local,
            R=R_local,
            Pi=Pi,
            beta=beta,
            lambda_risk=lambda_risk,
            staleness_threshold=staleness_threshold,
            exploration_mode=corr_exploration_mode_local,
            feature_mode=corr_feature_mode_local,
            feature_learning_rate=corr_feature_lr_local,
            feature_freeze_after=corr_feature_freeze_after_local,
            feature_log_interval=corr_feature_log_interval_local,
            feedback_mode=feedback_mode,
            eps=eps_corr_local,
            g_mean0=g_mean0_local,
            g_cov0=g_cov0_local,
            u_mean0=u_mean0_local,
            u_cov0=u_cov0_local,
            feature_arch=corr_feature_arch_local,
            feature_hidden_dim=corr_feature_hidden_dim_local,
            feature_activation=corr_feature_activation_local,
            seed=seed,
            em_tk=em_tk_local,
            em_min_weight=em_min_weight_local,
            em_verbose=em_verbose_local,
            phase0_t_end=phase0_t_end_local,
            feature_lr_phase0=feature_lr_phase0_local,
            feature_lr_phase1=feature_lr_phase1_local,
            feature_lr_phase2=feature_lr_phase2_local,
        )
        # Enable EM accumulation from the start of the run; for
        # expanding-window protocols, router_eval will toggle this flag.
        router_em.training_mode = True
        return router_em

    # Build mode-specific correlated routers. If no overrides are
    # provided, both fall back to the same configuration.
    router_partial_corr = None
    router_full_corr = None
    if slds_corr_enabled:
        router_partial_corr = _build_corr_router(
            slds_corr_cfg, slds_corr_partial_overrides, feedback_mode="partial"
        )
        router_full_corr = _build_corr_router(
            slds_corr_cfg, slds_corr_full_overrides, feedback_mode="full"
        )

    # Optional EM-style correlated routers (distinct from the base
    # correlated routers above). If routers.slds_imm_corr_em is empty,
    # these remain None and are omitted from evaluation.
    router_partial_corr_em = None
    router_full_corr_em = None
    if slds_corr_em_cfg and slds_corr_em_enabled:
        router_partial_corr_em = _build_corr_router_em(
            slds_corr_em_cfg,
            slds_corr_em_partial_overrides,
            feedback_mode="partial",
        )
        router_full_corr_em = _build_corr_router_em(
            slds_corr_em_cfg,
            slds_corr_em_full_overrides,
            feedback_mode="full",
        )

    # L2D baselines (configurable MLP/RNN, with and without sliding window)
    alpha_l2d = _resolve_vector(l2d_cfg.get("alpha", 1.0), 1.0, N)
    beta_l2d_cfg = l2d_cfg.get("beta", None)
    beta_l2d = beta.copy() if beta_l2d_cfg is None else _resolve_vector(
        beta_l2d_cfg, 0.0, N
    )
    lr_l2d = float(l2d_cfg.get("learning_rate", 1e-2))
    arch_l2d = str(l2d_cfg.get("arch", "mlp")).lower()
    hidden_dim_l2d = int(l2d_cfg.get("hidden_dim", 8))

    l2d_baseline = L2D(
        num_experts=N,
        feature_fn=feature_phi,
        alpha=alpha_l2d,
        beta=beta_l2d,
        learning_rate=lr_l2d,
        arch=arch_l2d,
        hidden_dim=hidden_dim_l2d,
        window_size=int(l2d_cfg.get("window_size", 1)),
        seed=seed,
    )

    l2d_sw_baseline = None
    if l2d_sw_cfg: # overridden with RNN architecture with sliding window
        alpha_l2d_sw = _resolve_vector(l2d_sw_cfg.get("alpha", 1.0), 1.0, N)
        beta_l2d_sw_cfg = l2d_sw_cfg.get("beta", None)
        beta_l2d_sw = beta.copy() if beta_l2d_sw_cfg is None else _resolve_vector(
            beta_l2d_sw_cfg, 0.0, N
        )
        lr_l2d_sw = float(l2d_sw_cfg.get("learning_rate", lr_l2d))
        arch_l2d_sw = str(l2d_sw_cfg.get("arch", arch_l2d)).lower()
        hidden_dim_l2d_sw = int(l2d_sw_cfg.get("hidden_dim", hidden_dim_l2d))
        window_size_sw = int(l2d_sw_cfg.get("window_size", 5))

        l2d_sw_baseline = L2D_SW(
            num_experts=N,
            feature_fn=feature_phi,
            alpha=alpha_l2d_sw,
            beta=beta_l2d_sw,
            learning_rate=lr_l2d_sw,
            arch=arch_l2d_sw,
            hidden_dim=hidden_dim_l2d_sw,
            window_size=window_size_sw,
            seed=seed,
        )

    # --------------------------------------------------------
    # Four UCB-style baselines routers
    # --------------------------------------------------------

    # LinUCB baselines (partial and full feedback)
    linucb_partial = None
    linucb_full = None
    if linucb_cfg:
        alpha_ucb = float(linucb_cfg.get("alpha_ucb", 1.0))
        lambda_reg = float(linucb_cfg.get("lambda_reg", 1.0))

        linucb_partial = LinUCB(
            num_experts=N,
            feature_fn=feature_phi,
            alpha_ucb=alpha_ucb,
            lambda_reg=lambda_reg,
            beta=beta,
            feedback_mode="partial",
        )
        linucb_full = LinUCB(
            num_experts=N,
            feature_fn=feature_phi,
            alpha_ucb=alpha_ucb,
            lambda_reg=lambda_reg,
            beta=beta,
            feedback_mode="full",
        )

    # NeuralUCB baseline (single policy; partial feedback by default)
    neuralucb_partial = None
    neuralucb_full = None
    if neuralucb_cfg:
        alpha_ucb_nn = float(neuralucb_cfg.get("alpha_ucb", 1.0))
        lambda_reg_nn = float(neuralucb_cfg.get("lambda_reg", 1.0))
        hidden_dim_nn = int(neuralucb_cfg.get("hidden_dim", 16))
        nn_lr = float(neuralucb_cfg.get("nn_learning_rate", 1e-3))
        neuralucb_partial = NeuralUCB(
            num_experts=N,
            feature_fn=feature_phi,
            alpha_ucb=alpha_ucb_nn,
            lambda_reg=lambda_reg_nn,
            beta=beta,
            hidden_dim=hidden_dim_nn,
            nn_learning_rate=nn_lr,
            feedback_mode="partial",
            seed=seed,
        )
        neuralucb_full = NeuralUCB(
            num_experts=N,
            feature_fn=feature_phi,
            alpha_ucb=alpha_ucb_nn,
            lambda_reg=lambda_reg_nn,
            beta=beta,
            hidden_dim=hidden_dim_nn,
            nn_learning_rate=nn_lr,
            feedback_mode="full",
            seed=seed,
        )

    # Copies for horizon planning (avoid contamination from full-run evaluation).
    l2d_baseline_horizon = copy.deepcopy(l2d_baseline)
    l2d_sw_baseline_horizon = (
        copy.deepcopy(l2d_sw_baseline) if l2d_sw_baseline is not None else None
    )
    linucb_partial_horizon = (
        copy.deepcopy(linucb_partial) if linucb_partial is not None else None
    )
    linucb_full_horizon = (
        copy.deepcopy(linucb_full) if linucb_full is not None else None
    )
    neuralucb_partial_horizon = (
        copy.deepcopy(neuralucb_partial) if neuralucb_partial is not None else None
    )
    neuralucb_full_horizon = (
        copy.deepcopy(neuralucb_full) if neuralucb_full is not None else None
    )

    # --------------------------------------------------------
    # Factorized Switching Linear Dynamical System
    # --------------------------------------------------------
    # Wire the FactorizedSLDS into the evaluation if configured
    fact_router_partial = None
    fact_router_full = None
    fact_router_partial_linear = None
    fact_router_full_linear = None
    router_partial_no_g = None
    router_full_no_g = None
    factorized_runs: list[dict] = []
    factorized_label = "Factorized SLDS"
    factorized_linear_label = "Factorized SLDS linear"
    base_transition_mode = None
    extra_transition_mode = None
    if factorized_slds_enabled:
        M_fact = int(factorized_slds_cfg.get("num_regimes", M))
        # Use shared dim from config or same logic as correlated router
        d_g_std = 1
        d_g_fact_cfg = factorized_slds_cfg.get("shared_dim", d_g_std)
        d_g_fact = int(d_g_fact_cfg)
        d_phi_fact = int(factorized_slds_cfg.get("idiosyncratic_dim", d))
        delta_max_fact = int(factorized_slds_cfg.get("delta_max", 50))
        B_dict_fact = factorized_slds_cfg.get("B_dict", None)
        r_scalar_fact_cfg = factorized_slds_cfg.get("R_scalar", None)
        if r_scalar_fact_cfg is None:
            r_scalar_fact = float(slds_cfg.get("R_scalar", 0.5))
        else:
            r_scalar_fact = float(r_scalar_fact_cfg)
        B_intercept_load_fact = float(
            factorized_slds_cfg.get("B_intercept_load", 1.0)
        )
        attn_dim_fact = factorized_slds_cfg.get("attn_dim", None)
        g_mean0_fact = factorized_slds_cfg.get("g_mean0", None)
        g_cov0_fact = factorized_slds_cfg.get("g_cov0", None)
        u_mean0_fact = factorized_slds_cfg.get("u_mean0", None)
        u_cov0_fact = factorized_slds_cfg.get("u_cov0", None)
        eps_fact = float(factorized_slds_cfg.get("eps", 1e-8))
        observation_mode_fact = str(
            factorized_slds_cfg.get("observation_mode", "residual")
        )
        transition_hidden_dims_cfg = factorized_slds_cfg.get(
            "transition_hidden_dims", None
        )
        if transition_hidden_dims_cfg is None:
            transition_hidden_dims = None
        elif isinstance(transition_hidden_dims_cfg, (list, tuple)):
            transition_hidden_dims = [int(x) for x in transition_hidden_dims_cfg]
        else:
            transition_hidden_dims = [int(transition_hidden_dims_cfg)]
        transition_activation = str(
            factorized_slds_cfg.get("transition_activation", "tanh")
        )
        transition_device = factorized_slds_cfg.get("transition_device", None)
        transition_mode_cfg = str(
            factorized_slds_cfg.get("transition_mode", "attention")
        ).lower()
        if transition_mode_cfg == "both":
            base_transition_mode = "attention"
            extra_transition_mode = "linear"
        elif transition_mode_cfg in ("attention", "linear"):
            base_transition_mode = transition_mode_cfg
            extra_transition_mode = None
        else:
            raise ValueError(
                "routers.factorized_slds.transition_mode must be "
                "'attention', 'linear', or 'both'."
            )

        exploration_cfg = factorized_slds_cfg.get("exploration", "g")
        if isinstance(exploration_cfg, (list, tuple)):
            exploration_modes = [str(x).lower() for x in exploration_cfg]
        else:
            exploration_modes = [str(exploration_cfg).lower()]
        exploration_modes = [mode for mode in exploration_modes if mode]
        if not exploration_modes:
            exploration_modes = ["g"]
        valid_exploration = {"g", "g_z", "ucb", "sampling"}
        for mode in exploration_modes:
            if mode not in valid_exploration:
                raise ValueError(
                    "routers.factorized_slds.exploration must be one of "
                    "'g', 'g_z', 'ucb', or 'sampling'."
                )

        exploration_mc_samples = int(
            factorized_slds_cfg.get("exploration_mc_samples", 25)
        )
        exploration_ucb_samples = int(
            factorized_slds_cfg.get("exploration_ucb_samples", 200)
        )
        exploration_ucb_alpha = factorized_slds_cfg.get("exploration_ucb_alpha", None)
        exploration_ucb_schedule = str(
            factorized_slds_cfg.get("exploration_ucb_schedule", "inverse_t")
        )
        exploration_sampling_deterministic = bool(
            factorized_slds_cfg.get("exploration_sampling_deterministic", False)
        )
        exploration_diag_enabled = bool(
            factorized_slds_cfg.get("exploration_diag_enabled", False)
        )
        exploration_diag_stride = int(
            factorized_slds_cfg.get("exploration_diag_stride", 100)
        )
        exploration_diag_samples = int(
            factorized_slds_cfg.get("exploration_diag_samples", 50)
        )
        exploration_diag_print = bool(
            factorized_slds_cfg.get("exploration_diag_print", True)
        )
        exploration_diag_max_records = int(
            factorized_slds_cfg.get("exploration_diag_max_records", 2000)
        )
        parallel_exploration = bool(
            factorized_slds_cfg.get("parallel_exploration", False)
        )
        parallel_workers_cfg = factorized_slds_cfg.get("parallel_workers", None)
        parallel_workers = (
            None if parallel_workers_cfg is None else int(parallel_workers_cfg)
        )

        A_g_fact = factorized_slds_cfg.get("A_g", None)
        A_u_fact = factorized_slds_cfg.get("A_u", None)
        Q_g_fact = factorized_slds_cfg.get("Q_g", None)
        Q_u_fact = factorized_slds_cfg.get("Q_u", None)
        if Q_g_fact is None:
            q_g_scales_cfg = factorized_slds_cfg.get("Q_g_scales", None)
            if q_g_scales_cfg is not None:
                q_g_arr = np.asarray(q_g_scales_cfg, dtype=float)
                if q_g_arr.shape == ():
                    q_g_arr = np.full(M_fact, float(q_g_arr), dtype=float)
                elif q_g_arr.shape == (M_fact,):
                    pass
                elif q_g_arr.size == 2 and M_fact > 2:
                    qg_broadcast = np.empty(M_fact, dtype=float)
                    qg_broadcast[0] = float(q_g_arr[0])
                    qg_broadcast[1:] = float(q_g_arr[1])
                    q_g_arr = qg_broadcast
                else:
                    raise ValueError(
                        "routers.factorized_slds.Q_g_scales must be a scalar, a list "
                        "of length num_regimes, or a length-2 list [qg_0, qg_other] "
                        "when num_regimes > 2."
                    )
                Q_g_fact = np.zeros((M_fact, d_g_fact, d_g_fact), dtype=float)
                for k in range(M_fact):
                    Q_g_fact[k] = q_g_arr[k] * np.eye(d_g_fact, dtype=float)
        if Q_u_fact is None:
            q_u_scales_cfg = factorized_slds_cfg.get("Q_u_scales", None)
            if q_u_scales_cfg is not None:
                q_u_arr = np.asarray(q_u_scales_cfg, dtype=float)
                if q_u_arr.shape == ():
                    q_u_arr = np.full(M_fact, float(q_u_arr), dtype=float)
                elif q_u_arr.shape == (M_fact,):
                    pass
                elif q_u_arr.size == 2 and M_fact > 2:
                    qu_broadcast = np.empty(M_fact, dtype=float)
                    qu_broadcast[0] = float(q_u_arr[0])
                    qu_broadcast[1:] = float(q_u_arr[1])
                    q_u_arr = qu_broadcast
                else:
                    raise ValueError(
                        "routers.factorized_slds.Q_u_scales must be a scalar, a list "
                        "of length num_regimes, or a length-2 list [qu_0, qu_other] "
                        "when num_regimes > 2."
                    )
                Q_u_fact = np.zeros((M_fact, d_phi_fact, d_phi_fact), dtype=float)
                for k in range(M_fact):
                    Q_u_fact[k] = q_u_arr[k] * np.eye(d_phi_fact, dtype=float)
        if B_dict_fact is None:
            tri_cycle_cfg = env_cfg.get("tri_cycle", None)
            if isinstance(tri_cycle_cfg, dict):
                if str(tri_cycle_cfg.get("shared_noise_mode", "")).lower() == "model_match":
                    loadings = tri_cycle_cfg.get("shared_loadings", None)
                    if loadings is not None:
                        loadings_arr = np.asarray(loadings, dtype=float)
                        if loadings_arr.shape == (N, d_g_fact):
                            B_dict_fact = {
                                int(k): loadings_arr[int(k)].reshape(d_phi_fact, d_g_fact)
                                for k in range(N)
                            }

        def _build_factorized_router(
            feedback_mode: str,
            transition_mode_local: str,
            exploration_mode: str,
        ) -> FactorizedSLDS:
            return FactorizedSLDS(
                M=M_fact,
                d_g=d_g_fact,
                d_phi=d_phi_fact,
                feature_fn=feature_phi,
                B_dict=B_dict_fact,
                beta=beta,
                Delta_max=delta_max_fact,
                R=r_scalar_fact,
                num_experts=N,
                B_intercept_load=B_intercept_load_fact,
                attn_dim=attn_dim_fact,
                g_mean0=g_mean0_fact,
                g_cov0=g_cov0_fact,
                u_mean0=u_mean0_fact,
                u_cov0=u_cov0_fact,
                A_g=A_g_fact,
                A_u=A_u_fact,
                Q_g=Q_g_fact,
                Q_u=Q_u_fact,
                eps=eps_fact,
                exploration=exploration_mode,
                exploration_mc_samples=exploration_mc_samples,
                exploration_ucb_samples=exploration_ucb_samples,
                exploration_ucb_alpha=exploration_ucb_alpha,
                exploration_ucb_schedule=exploration_ucb_schedule,
                exploration_sampling_deterministic=exploration_sampling_deterministic,
                exploration_diag_enabled=exploration_diag_enabled,
                exploration_diag_stride=exploration_diag_stride,
                exploration_diag_samples=exploration_diag_samples,
                exploration_diag_print=exploration_diag_print,
                exploration_diag_max_records=exploration_diag_max_records,
                observation_mode=observation_mode_fact,
                transition_hidden_dims=transition_hidden_dims,
                transition_activation=transition_activation,
                transition_device=transition_device,
                transition_mode=transition_mode_local,
                feedback_mode=feedback_mode,
                seed=seed,
            )

        def _build_factorized_router_no_g(
            feedback_mode: str,
            transition_mode_local: str,
            exploration_mode: str,
        ) -> FactorizedSLDS:
            # d_g=0 disables all shared-factor g_t dynamics/emission terms.
            return FactorizedSLDS(
                M=M_fact,
                d_g=0,
                d_phi=d_phi_fact,
                feature_fn=feature_phi,
                beta=beta,
                Delta_max=delta_max_fact,
                R=r_scalar_fact,
                num_experts=N,
                B_intercept_load=B_intercept_load_fact,
                attn_dim=attn_dim_fact,
                g_mean0=None,
                g_cov0=None,
                u_mean0=u_mean0_fact,
                u_cov0=u_cov0_fact,
                A_g=None,
                A_u=A_u_fact,
                Q_g=None,
                Q_u=Q_u_fact,
                eps=eps_fact,
                exploration=exploration_mode,
                exploration_mc_samples=exploration_mc_samples,
                exploration_ucb_samples=exploration_ucb_samples,
                exploration_ucb_alpha=exploration_ucb_alpha,
                exploration_ucb_schedule=exploration_ucb_schedule,
                exploration_sampling_deterministic=exploration_sampling_deterministic,
                exploration_diag_enabled=exploration_diag_enabled,
                exploration_diag_stride=exploration_diag_stride,
                exploration_diag_samples=exploration_diag_samples,
                exploration_diag_print=exploration_diag_print,
                exploration_diag_max_records=exploration_diag_max_records,
                observation_mode=observation_mode_fact,
                transition_hidden_dims=transition_hidden_dims,
                transition_activation=transition_activation,
                transition_device=transition_device,
                transition_mode=transition_mode_local,
                feedback_mode=feedback_mode,
                seed=seed,
            )

        factorized_runs = []
        primary_run = None

        for exploration_mode in exploration_modes:
            print(
                f"\n--- Running FactorizedSLDS Router (partial, {base_transition_mode}, {exploration_mode}) ---"
            )
            fact_router_partial = _build_factorized_router(
                "partial", base_transition_mode, exploration_mode
            )
            print(
                f"\n--- Running FactorizedSLDS Router (full, {base_transition_mode}, {exploration_mode}) ---"
            )
            fact_router_full = _build_factorized_router(
                "full", base_transition_mode, exploration_mode
            )
            factorized_label_local = (
                f"Factorized SLDS {base_transition_mode} ({exploration_mode})"
            )

            fact_router_partial_linear = None
            fact_router_full_linear = None
            factorized_linear_label_local = factorized_linear_label
            if extra_transition_mode is not None:
                print(
                    f"\n--- Running FactorizedSLDS Router (partial, {extra_transition_mode}, {exploration_mode}) ---"
                )
                fact_router_partial_linear = _build_factorized_router(
                    "partial", extra_transition_mode, exploration_mode
                )
                print(
                    f"\n--- Running FactorizedSLDS Router (full, {extra_transition_mode}, {exploration_mode}) ---"
                )
                fact_router_full_linear = _build_factorized_router(
                    "full", extra_transition_mode, exploration_mode
                )
                factorized_linear_label_local = (
                    f"Factorized SLDS {extra_transition_mode} ({exploration_mode})"
                )

            router_partial_no_g = _build_factorized_router_no_g(
                "partial", base_transition_mode, exploration_mode
            )
            router_full_no_g = _build_factorized_router_no_g(
                "full", base_transition_mode, exploration_mode
            )

            run_bundle = {
                "exploration_mode": exploration_mode,
                "router_partial_no_g": router_partial_no_g,
                "router_full_no_g": router_full_no_g,
                "fact_router_partial": fact_router_partial,
                "fact_router_full": fact_router_full,
                "fact_router_partial_linear": fact_router_partial_linear,
                "fact_router_full_linear": fact_router_full_linear,
                "factorized_label": factorized_label_local,
                "factorized_linear_label": factorized_linear_label_local,
            }
            factorized_runs.append(run_bundle)
            if primary_run is None:
                primary_run = run_bundle

        if primary_run is not None:
            router_partial_no_g = primary_run["router_partial_no_g"]
            router_full_no_g = primary_run["router_full_no_g"]
            fact_router_partial = primary_run["fact_router_partial"]
            fact_router_full = primary_run["fact_router_full"]
            fact_router_partial_linear = primary_run["fact_router_partial_linear"]
            fact_router_full_linear = primary_run["fact_router_full_linear"]
            factorized_label = primary_run["factorized_label"]
            factorized_linear_label = primary_run["factorized_linear_label"]
            router_partial = router_partial_no_g
            router_full = router_full_no_g

    # --------------------------------------------------------
    # Environment and L2D baselines
    # --------------------------------------------------------
    
    # Environment: either synthetic or ETTh1, depending on env_cfg.
    if data_source == "etth1":
        # Real-world ETTh1 experiment (oil temperature as target).
        T_raw = env_cfg.get("T", None)
        T_env = None if T_raw is None else int(T_raw)
        csv_path = env_cfg.get("csv_path", "data/ETTh1.csv")
        target_column = env_cfg.get("target_column", "OT")

        env = ETTh1TimeSeriesEnv(
            csv_path=csv_path,
            target_column=target_column,
            num_experts=N,
            num_regimes=M,
            T=T_env,
            seed=int(env_cfg.get("seed", 42)),
            unavailable_expert_idx=env_cfg.get("unavailable_expert_idx", None),
            unavailable_intervals=env_cfg.get("unavailable_intervals", None),
            arrival_expert_idx=env_cfg.get("arrival_expert_idx", None),
            arrival_intervals=env_cfg.get("arrival_intervals", None),
        )
    else:
        # Synthetic environment with dynamic expert availability.
        # - Expert 1: unavailable on [10, 50] and [200, 250] (inclusive).
        # - Expert 4: arrives after t=100 and leaves at t=150, i.e.
        #   available on [101, 150] and unavailable outside that window.
        env = SyntheticTimeSeriesEnv(
            num_experts=N,
            num_regimes=M,
            T=int(env_cfg.get("T", 300)),
            seed=int(env_cfg.get("seed", 42)),
            unavailable_expert_idx=int(env_cfg.get("unavailable_expert_idx", 1)),
            unavailable_intervals=env_cfg.get("unavailable_intervals", [(10, 50), (200, 250)]),
            arrival_expert_idx=int(env_cfg.get("arrival_expert_idx", 4)),
            arrival_intervals=env_cfg.get("arrival_intervals", [(120, 200)]),
            setting=setting,
            noise_scale=env_cfg.get("noise_scale", None),
            tri_cycle_cfg=env_cfg.get("tri_cycle", None),
        )
    # Visualization-only settings for plots.
    env.plot_shift = int(cfg.get("plot_shift", 1))
    env.plot_target = str(cfg.get("plot_target", "y")).lower()
    if bool(cfg.get("plot_synth_preview", False)):
        plot_time_series(env)

    em_data_cache = {}

    def _run_factorized_em(router: Optional[FactorizedSLDS], label: str) -> None:
        if router is None:
            return
        em_enabled = bool(factorized_slds_cfg.get("em_enabled", True))
        em_force_full_feedback = bool(
            factorized_slds_cfg.get("em_offline_full_feedback", True)
        )
        em_tk_cfg = factorized_slds_cfg.get("em_tk", None)
        if em_tk_cfg is None:
            em_tk = min(500, env.T - 1)
        else:
            em_tk = int(em_tk_cfg)
        if not em_enabled or em_tk <= 1:
            return

        n_em = int(factorized_slds_cfg.get("em_n", 5))
        n_samples = int(factorized_slds_cfg.get("em_samples", 10))
        burn_in = int(factorized_slds_cfg.get("em_burn_in", 5))
        val_fraction = float(factorized_slds_cfg.get("em_val_fraction", 0.2))
        val_len_cfg = factorized_slds_cfg.get("em_val_len", None)
        val_len = int(val_len_cfg) if val_len_cfg is not None else None
        val_strategy = str(factorized_slds_cfg.get("em_val_strategy", "tail"))
        val_roll_splits_cfg = factorized_slds_cfg.get("em_val_roll_splits", None)
        val_roll_splits = (
            int(val_roll_splits_cfg) if val_roll_splits_cfg is not None else None
        )
        val_roll_len_cfg = factorized_slds_cfg.get("em_val_roll_len", None)
        val_roll_len = int(val_roll_len_cfg) if val_roll_len_cfg is not None else None
        val_roll_stride_cfg = factorized_slds_cfg.get("em_val_roll_stride", None)
        val_roll_stride = (
            int(val_roll_stride_cfg) if val_roll_stride_cfg is not None else None
        )
        theta_lr = float(factorized_slds_cfg.get("em_theta_lr", 1e-2))
        theta_steps = int(factorized_slds_cfg.get("em_theta_steps", 1))
        em_seed_cfg = factorized_slds_cfg.get("em_seed", env_cfg.get("seed", 0))
        em_seed = int(em_seed_cfg) if em_seed_cfg is not None else 0
        em_priors = factorized_slds_cfg.get("em_priors", None)
        print_val_loss = bool(factorized_slds_cfg.get("em_print_val_loss", True))
        use_validation_cfg = factorized_slds_cfg.get("em_use_validation", None)
        if use_validation_cfg is None:
            use_validation = bool(
                val_fraction > 0.0
                or val_len is not None
                or str(val_strategy).lower() == "rolling"
            )
        else:
            use_validation = bool(use_validation_cfg)
        em_eps_n = float(factorized_slds_cfg.get("em_eps_n", 0.0))

        print(f"\n--- FactorizedSLDS EM ({label}) (t=1..{em_tk}, n_em={n_em}) ---")
        force_full = bool(em_force_full_feedback or router.feedback_mode == "full")
        cache_key = (int(em_tk), force_full)
        em_data = em_data_cache.get(cache_key)
        if em_data is None:
            em_data = _collect_factorized_em_data(
                router,
                env,
                em_tk,
                force_full_feedback=force_full,
            )
            em_data_cache[cache_key] = em_data
        ctx_em, avail_em, actions_em, resid_em, resid_full_em = _clone_factorized_em_data(
            em_data
        )
        if ctx_em:
            context_dim = int(np.asarray(ctx_em[0], dtype=float).reshape(-1).shape[0])
            needs_init = False
            if router.transition_model is None and router.transition_hidden_dims is None:
                if router.transition_mode == "linear":
                    needs_init = router.W_lin is None or router.b_lin is None
                else:
                    needs_init = router.W_q is None or router.W_k is None
            if needs_init:
                prev_rng_state = copy.deepcopy(router._rng.bit_generator.state)
                router._rng.bit_generator.state = (
                    np.random.default_rng(em_seed).bit_generator.state
                )
                router._init_transition_params(context_dim)
                router._rng.bit_generator.state = prev_rng_state
        prev_feedback_mode = router.feedback_mode
        if em_force_full_feedback:
            router.feedback_mode = "full"
        try:
            router.fit_em(
                contexts=ctx_em,
                available_sets=avail_em,
                actions=actions_em,
                residuals=resid_em,
                residuals_full=resid_full_em
                if router.feedback_mode == "full"
                else None,
                n_em=n_em,
                n_samples=n_samples,
                burn_in=burn_in,
                val_fraction=val_fraction,
                val_len=val_len,
                val_strategy=val_strategy,
                val_roll_splits=val_roll_splits,
                val_roll_len=val_roll_len,
                val_roll_stride=val_roll_stride,
                priors=em_priors,
                theta_lr=theta_lr,
                theta_steps=theta_steps,
                seed=em_seed,
                print_val_loss=print_val_loss,
                epsilon_N=em_eps_n,
                use_validation=use_validation,
                transition_log_stage="post_offline_em",
            )
        finally:
            router.feedback_mode = prev_feedback_mode
        router.em_tk = int(em_tk)
        router.reset_beliefs()

    def _configure_factorized_online_em(
        router: Optional[FactorizedSLDS],
    ) -> None:
        if router is None:
            return
        online_enabled = bool(factorized_slds_cfg.get("em_online_enabled", False))
        if not online_enabled:
            return

        window_cfg = factorized_slds_cfg.get("em_online_window", None)
        window = int(window_cfg) if window_cfg is not None else 0
        if window <= 0:
            return
        period = int(factorized_slds_cfg.get("em_online_period", 1))
        n_em = int(
            factorized_slds_cfg.get(
                "em_online_n", factorized_slds_cfg.get("em_n", 1)
            )
        )
        n_samples = int(
            factorized_slds_cfg.get(
                "em_online_samples", factorized_slds_cfg.get("em_samples", 10)
            )
        )
        burn_in = int(
            factorized_slds_cfg.get(
                "em_online_burn_in", factorized_slds_cfg.get("em_burn_in", 5)
            )
        )
        theta_lr = float(
            factorized_slds_cfg.get(
                "em_online_theta_lr", factorized_slds_cfg.get("em_theta_lr", 1e-2)
            )
        )
        theta_steps = int(
            factorized_slds_cfg.get(
                "em_online_theta_steps", factorized_slds_cfg.get("em_theta_steps", 1)
            )
        )
        eps_n = float(
            factorized_slds_cfg.get(
                "em_online_eps_n", factorized_slds_cfg.get("em_eps_n", 0.0)
            )
        )
        start_t_cfg = factorized_slds_cfg.get("em_online_start_t", None)
        if start_t_cfg is None:
            em_tk_local = getattr(router, "em_tk", None)
            start_t = None if em_tk_local is None else int(em_tk_local) + 1
        else:
            start_t = int(start_t_cfg)
        seed_cfg = factorized_slds_cfg.get(
            "em_online_seed", factorized_slds_cfg.get("em_seed", None)
        )
        seed = None if seed_cfg is None else int(seed_cfg)
        online_priors = factorized_slds_cfg.get("em_online_priors", None)
        if online_priors is None:
            online_priors = factorized_slds_cfg.get("em_priors", None)
        print_val_loss = bool(
            factorized_slds_cfg.get(
                "em_online_print_val_loss",
                factorized_slds_cfg.get("em_print_val_loss", True),
            )
        )
        use_state_priors = bool(
            factorized_slds_cfg.get("em_online_use_state_priors", True)
        )
        router.configure_online_em(
            enabled=True,
            window=window,
            period=period,
            n_em=n_em,
            n_samples=n_samples,
            burn_in=burn_in,
            epsilon_N=eps_n,
            theta_lr=theta_lr,
            theta_steps=theta_steps,
            priors=online_priors,
            start_t=start_t,
            seed=seed,
            print_val_loss=print_val_loss,
            use_state_priors=use_state_priors,
        )

    if transition_log_cfg is not None:
        def _register_transition_logger(obj: object, label: str) -> None:
            if obj is None:
                return
            register_transition_log_label(obj, label)
            if isinstance(obj, FactorizedSLDS):
                obj.transition_log_cfg = transition_log_cfg
                obj.transition_log_label = label

        if router_partial is not None and router_partial is not router_partial_no_g:
            _register_transition_logger(router_partial, "SLDS-IMM partial")
        if router_full is not None and router_full is not router_full_no_g:
            _register_transition_logger(router_full, "SLDS-IMM full")
        if router_partial_corr is not None:
            _register_transition_logger(router_partial_corr, "Corr SLDS-IMM partial")
        if router_full_corr is not None:
            _register_transition_logger(router_full_corr, "Corr SLDS-IMM full")
        if router_partial_corr_em is not None:
            _register_transition_logger(
                router_partial_corr_em, "Corr SLDS-IMM EM partial"
            )
        if router_full_corr_em is not None:
            _register_transition_logger(
                router_full_corr_em, "Corr SLDS-IMM EM full"
            )
        for run in factorized_runs:
            rp_no_g = run.get("router_partial_no_g")
            rf_no_g = run.get("router_full_no_g")
            fact_partial = run.get("fact_router_partial")
            fact_full = run.get("fact_router_full")
            fact_partial_linear = run.get("fact_router_partial_linear")
            fact_full_linear = run.get("fact_router_full_linear")
            label = run.get("factorized_label", factorized_label)
            linear_label = run.get("factorized_linear_label", factorized_linear_label)
            if rp_no_g is not None:
                _register_transition_logger(
                    rp_no_g, f"{label} no-g partial"
                )
            if rf_no_g is not None:
                _register_transition_logger(
                    rf_no_g, f"{label} no-g full"
                )
            if fact_partial is not None:
                _register_transition_logger(
                    fact_partial, f"{label} partial"
                )
            if fact_full is not None:
                _register_transition_logger(
                    fact_full, f"{label} full"
                )
            if fact_partial_linear is not None:
                _register_transition_logger(
                    fact_partial_linear, f"{linear_label} partial"
                )
            if fact_full_linear is not None:
                _register_transition_logger(
                    fact_full_linear, f"{linear_label} full"
                )
        if l2d_baseline is not None:
            _register_transition_logger(l2d_baseline, "L2D (full feedback)")
        if l2d_sw_baseline is not None:
            _register_transition_logger(l2d_sw_baseline, "L2D_SW (full feedback)")
        if linucb_partial is not None:
            _register_transition_logger(linucb_partial, "LinUCB partial")
        if linucb_full is not None:
            _register_transition_logger(linucb_full, "LinUCB full")
        if neuralucb_partial is not None:
            _register_transition_logger(neuralucb_partial, "NeuralUCB partial")
        if neuralucb_full is not None:
            _register_transition_logger(neuralucb_full, "NeuralUCB full")

    if base_transition_mode is not None:
        for run in factorized_runs:
            label = run.get("factorized_label", factorized_label)
            _run_factorized_em(
                run.get("fact_router_partial"), f"{label} partial"
            )
            _run_factorized_em(
                run.get("fact_router_full"), f"{label} full"
            )
            _run_factorized_em(
                run.get("router_partial_no_g"), f"{label} no-g partial"
            )
            _run_factorized_em(
                run.get("router_full_no_g"), f"{label} no-g full"
            )
    if extra_transition_mode is not None:
        for run in factorized_runs:
            linear_label = run.get("factorized_linear_label", factorized_linear_label)
            _run_factorized_em(
                run.get("fact_router_partial_linear"), f"{linear_label} partial"
            )
            _run_factorized_em(
                run.get("fact_router_full_linear"), f"{linear_label} full"
            )

    for run in factorized_runs:
        _configure_factorized_online_em(run.get("fact_router_partial"))
        _configure_factorized_online_em(run.get("fact_router_full"))
        _configure_factorized_online_em(run.get("fact_router_partial_linear"))
        _configure_factorized_online_em(run.get("fact_router_full_linear"))
        _configure_factorized_online_em(run.get("router_partial_no_g"))
        _configure_factorized_online_em(run.get("router_full_no_g"))

    # Plot the true series and expert predictions
    # plot_time_series(env)

    # Evaluate routers, L2D baseline, and constant-expert baselines,
    # and plot their induced prediction time series.
    if factorized_runs:
        if parallel_exploration and len(factorized_runs) > 1:
            transition_cfg = get_transition_log_config()
            payloads = []
            for run in factorized_runs:
                payloads.append(
                    {
                        "env": env,
                        "router_partial": run["router_partial_no_g"],
                        "router_full": run["router_full_no_g"],
                        "router_factorial_partial": run["fact_router_partial"],
                        "router_factorial_full": run["fact_router_full"],
                        "factorized_label": run["factorized_label"],
                        "router_factorial_partial_linear": run["fact_router_partial_linear"],
                        "router_factorial_full_linear": run["fact_router_full_linear"],
                        "factorized_linear_label": run["factorized_linear_label"],
                        "l2d_baseline": l2d_baseline,
                        "l2d_sw_baseline": l2d_sw_baseline,
                        "linucb_partial": linucb_partial,
                        "linucb_full": linucb_full,
                        "neuralucb_partial": neuralucb_partial,
                        "neuralucb_full": neuralucb_full,
                        "seed": seed,
                        "transition_log_cfg": transition_cfg,
                    }
                )
            workers = parallel_workers
            if workers is None or workers <= 0:
                workers = min(len(payloads), os.cpu_count() or 1)
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                pool.map(_evaluate_factorized_run_worker, payloads)
        else:
            for run in factorized_runs:
                l2d_run = (
                    copy.deepcopy(l2d_baseline) if l2d_baseline is not None else None
                )
                l2d_sw_run = (
                    copy.deepcopy(l2d_sw_baseline)
                    if l2d_sw_baseline is not None
                    else None
                )
                linucb_partial_run = (
                    copy.deepcopy(linucb_partial)
                    if linucb_partial is not None
                    else None
                )
                linucb_full_run = (
                    copy.deepcopy(linucb_full) if linucb_full is not None else None
                )
                neuralucb_partial_run = (
                    copy.deepcopy(neuralucb_partial)
                    if neuralucb_partial is not None
                    else None
                )
                neuralucb_full_run = (
                    copy.deepcopy(neuralucb_full) if neuralucb_full is not None else None
                )

                evaluate_routers_and_baselines(
                    env,
                    run["router_partial_no_g"],
                    run["router_full_no_g"],
                    run["fact_router_partial"],
                    run["fact_router_full"],
                    factorized_label=run["factorized_label"],
                    router_factorial_partial_linear=run["fact_router_partial_linear"],
                    router_factorial_full_linear=run["fact_router_full_linear"],
                    factorized_linear_label=run["factorized_linear_label"],
                    l2d_baseline=l2d_run,
                    l2d_sw_baseline=l2d_sw_run,
                    linucb_partial=linucb_partial_run,
                    linucb_full=linucb_full_run,
                    neuralucb_partial=neuralucb_partial_run,
                    neuralucb_full=neuralucb_full_run,
                    seed=seed,
                )
    else:
        evaluate_routers_and_baselines(
            env,
            router_partial,
            router_full,
            fact_router_partial,
            fact_router_full,
            factorized_label=factorized_label,
            router_factorial_partial_linear=fact_router_partial_linear,
            router_factorial_full_linear=fact_router_full_linear,
            factorized_linear_label=factorized_linear_label,
            l2d_baseline=l2d_baseline,
            l2d_sw_baseline=l2d_sw_baseline,
            linucb_partial=linucb_partial,
            linucb_full=linucb_full,
            neuralucb_partial=neuralucb_partial,
            neuralucb_full=neuralucb_full,
            seed=seed,
        )


    # Optional: analyze reaction to a late-arriving expert. This can be
    '''
    # or via the top-level config key `late_arrival_analysis: true`.
    do_late_arrival = bool(cfg.get("late_arrival_analysis", False)) or getattr(
        args, "late_arrival_analysis", False
    )
    # In config.yaml, disabled
    if do_late_arrival:
        arrival_idx_cfg = env_cfg.get("arrival_expert_idx", None)
        if arrival_idx_cfg is None:
            print(
                "[analysis_late_arrival] No arrival_expert_idx configured in "
                "environment; skipping late-arrival analysis."
            )
        else:
            try:
                j_new = int(arrival_idx_cfg)
            except (TypeError, ValueError):
                print(
                    "[analysis_late_arrival] arrival_expert_idx in config is "
                    "not an integer; skipping late-arrival analysis."
                )
            else:
                window_cfg = env_cfg.get("analysis_window", 500)
                adopt_cfg = env_cfg.get("analysis_adoption_threshold", 0.5)
                try:
                    window = int(window_cfg)
                except (TypeError, ValueError):
                    window = 500
                try:
                    adoption_threshold = float(adopt_cfg)
                except (TypeError, ValueError):
                    adoption_threshold = 0.5

                analysis_late_arrival(
                    env,
                    router_partial,
                    router_full,
                    l2d_baseline=l2d_baseline,
                    router_partial_corr=router_partial_corr,
                    router_full_corr=router_full_corr,
                    router_partial_corr_em=router_partial_corr_em,
                    router_full_corr_em=router_full_corr_em,
                    l2d_sw_baseline=l2d_sw_baseline,
                    linucb_partial=linucb_partial,
                    linucb_full=linucb_full,
                    neuralucb_partial=neuralucb_partial,
                    neuralucb_full=neuralucb_full,
                    new_expert_idx=j_new,
                    window=window,
                    adoption_threshold=adoption_threshold,
                )         
    '''
    # --------------------------------------------------------
    # Horizon-H planning from a given time t
    # --------------------------------------------------------

    # Build expert prediction functions for planning
    def experts_predict_factory(env_):
        def f(j: int):
            return lambda x: env_.expert_predict(j, x)
        return [f(j) for j in range(env_.num_experts)]

    experts_predict = experts_predict_factory(env)

    # Simple context update: x_{t+1} := y_hat (recursive forecasting)
    def context_update(x: np.ndarray, y_hat: float) -> np.ndarray:
        return np.array([y_hat], dtype=float)

    # Take current context at t0 and plan H steps ahead, and evaluate.
    t0_cfg = int(horizon_cfg.get("t0", 175))
    H = int(horizon_cfg.get("H", 5))
    planning_method = str(horizon_cfg.get("method", "regressive"))
    scenario_generator_cfg = horizon_cfg.get("scenario_generator", {}) or {}
    if "seed" not in scenario_generator_cfg:
        scenario_generator_cfg = dict(scenario_generator_cfg)
        scenario_generator_cfg["seed"] = seed
    delta = float(horizon_cfg.get("delta", 0.1))
    online_start_t = horizon_cfg.get("online_start_t", None)

    em_tk_candidates = []
    for r in (
        router_partial,
        router_full,
        fact_router_partial,
        fact_router_full,
        fact_router_partial_linear,
        fact_router_full_linear,
        router_partial_corr_em,
        router_full_corr_em,
    ):
        if r is None:
            continue
        em_val = getattr(r, "em_tk", None)
        if em_val is not None:
            em_tk_candidates.append(int(em_val))
    em_tk_anchor = max(em_tk_candidates) if em_tk_candidates else None

    t0 = int(t0_cfg)
    if em_tk_anchor is not None:
        if t0 <= em_tk_anchor:
            t0 = int(em_tk_anchor + 1)
        if online_start_t is None:
            online_start_t = int(em_tk_anchor)
    if t0 != int(t0_cfg):
        print(
            f"[Horizon planning] Adjusted t0 from {t0_cfg} to {t0} "
            f"to start after EM (em_tk={em_tk_anchor})."
        )

    evaluate_horizon_planning(
        env=env,
        router_partial=router_partial,
        router_full=router_full,
        router_factorial_partial=fact_router_partial,
        router_factorial_full=fact_router_full,
        router_factorial_partial_linear=fact_router_partial_linear,
        router_factorial_full_linear=fact_router_full_linear,
        beta=beta,
        t0=t0,
        H=H,
        experts_predict=experts_predict,
        context_update=context_update,
        l2d_baseline=l2d_baseline_horizon,
        l2d_sw_baseline=l2d_sw_baseline_horizon,
        linucb_partial=linucb_partial_horizon,
        linucb_full=linucb_full_horizon,
        neuralucb_partial=neuralucb_partial_horizon,
        neuralucb_full=neuralucb_full_horizon,
        planning_method=planning_method,
        scenario_generator_cfg=scenario_generator_cfg,
        delta=delta,
        online_start_t=online_start_t,
        factorized_label=factorized_label,
        factorized_linear_label=factorized_linear_label,
    )
