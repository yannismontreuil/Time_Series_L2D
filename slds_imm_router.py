# %%
# %matplotlib widget

import argparse
import json
import os
import numpy as np

from etth1_env import ETTh1TimeSeriesEnv

from router_model import SLDSIMMRouter, feature_phi
from router_model_recurrent import RecurrentSLDSRouter
from router_model_corr import SLDSIMMRouter_Corr, RecurrentSLDSIMMRouter_Corr
from router_model_corr_em import SLDSIMMRouter_Corr_EM
from synthetic_env import SyntheticTimeSeriesEnv
from l2d_baseline import L2D, L2D_SW
from linucb_baseline import LinUCB
from neuralucb_baseline import NeuralUCB

from plot_utils import (
    plot_time_series,
    evaluate_routers_and_baselines,
    analysis_late_arrival,
)
from horizon_planning import evaluate_horizon_planning

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _load_config(path: str = "config.yaml") -> dict:
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
        default="config.yaml",
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
    env_cfg = cfg.get("environment", {})

    routers_cfg = cfg.get("routers", {})
    slds_cfg = routers_cfg.get("slds_imm", {}) or {}
    slds_corr_cfg = routers_cfg.get("slds_imm_corr", {}) or {}
    slds_corr_rec_cfg = routers_cfg.get("slds_imm_corr_recurrent", {}) or {}

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
    N = int(env_cfg.get("num_experts", 5))   # experts
    # State dimension (= dim φ(x)); feature map in router_model.py currently
    # returns a 2D feature, so d must be compatible with that.
    d = int(env_cfg.get("state_dim", 2))

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
    # Recurrent SLDS (r-SLDS) routers with stick-breaking gating
    # --------------------------------------------------------

    slds_recurrent_cfg = routers_cfg.get("slds_imm_recurrent", {}) or {}
    rec_partial_overrides = slds_recurrent_cfg.get("partial_overrides", {}) or {}
    rec_full_overrides = slds_recurrent_cfg.get("full_overrides", {}) or {}

    C_cfg = slds_recurrent_cfg.get("C", None)
    if C_cfg is not None:
        C_rec_base = np.asarray(C_cfg, dtype=float)
    else:
        C_rec_base = np.zeros((M, N, d), dtype=float)

    stick_gamma_cfg = slds_recurrent_cfg.get("stick_gamma", None)
    stick_kappa_cfg = slds_recurrent_cfg.get("stick_kappa", None)
    stick_gamma_base = (
        np.asarray(stick_gamma_cfg, dtype=float)
        if stick_gamma_cfg is not None
        else None
    )
    stick_kappa_base = (
        np.asarray(stick_kappa_cfg, dtype=float)
        if stick_kappa_cfg is not None
        else None
    )

    stick_gamma_rows_cfg = slds_recurrent_cfg.get("stick_gamma_rows", None)
    stick_kappa_rows_cfg = slds_recurrent_cfg.get("stick_kappa_rows", None)
    stick_gamma_rows_base = (
        np.asarray(stick_gamma_rows_cfg, dtype=float)
        if stick_gamma_rows_cfg is not None
        else None
    )
    stick_kappa_rows_base = (
        np.asarray(stick_kappa_rows_cfg, dtype=float)
        if stick_kappa_rows_cfg is not None
        else None
    )

    def _build_rec_router(overrides: dict, feedback_mode: str) -> RecurrentSLDSRouter:
        q_scale = float(overrides.get("q", 1.0))
        r_scale = float(overrides.get("r", 1.0))
        lambda_local = float(overrides.get("lambda_risk", lambda_risk))
        C_scale = float(overrides.get("C_scale", 1.0))
        stick_scale = float(overrides.get("stick_gamma_scale", 1.0))

        C_rec = C_rec_base * C_scale

        stick_gamma = (
            stick_gamma_base * stick_scale if stick_gamma_base is not None else None
        )
        stick_kappa = stick_kappa_base.copy() if stick_kappa_base is not None else None
        stick_gamma_rows = (
            stick_gamma_rows_base * stick_scale
            if stick_gamma_rows_base is not None
            else None
        )
        stick_kappa_rows = (
            stick_kappa_rows_base.copy() if stick_kappa_rows_base is not None else None
        )

        return RecurrentSLDSRouter(
            num_experts=N,
            num_regimes=M,
            state_dim=d,
            feature_fn=feature_phi,
            A=A,
            Q=Q * q_scale,
            R=R * r_scale,
            Pi=Pi,
            C=C_rec,
            beta=beta,
            lambda_risk=lambda_local,
            feedback_mode=feedback_mode,
            pop_mean=pop_mean,
            pop_cov=pop_cov,
            eps=eps_slds,
            stick_gamma=stick_gamma,
            stick_kappa=stick_kappa,
            stick_gamma_rows=stick_gamma_rows,
            stick_kappa_rows=stick_kappa_rows,
        )

    router_partial_rec = _build_rec_router(rec_partial_overrides, feedback_mode="partial")
    router_full_rec = _build_rec_router(rec_full_overrides, feedback_mode="full")

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

    slds_corr_rec_partial_overrides = slds_corr_rec_cfg.get("partial_overrides", {}) or {}
    slds_corr_rec_full_overrides = slds_corr_rec_cfg.get("full_overrides", {}) or {}

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
        )
    
    # --------------------------------------------------------
    # Recurrent + Correlated-expert SLDS-IMM routers
    # --------------------------------------------------------

    def _build_corr_router_rec(
        corr_base_cfg: dict,
        overrides: dict,
        feedback_mode: str,
    ) -> RecurrentSLDSIMMRouter_Corr:
        """
        Build the recurrent + correlated router. Mirrors _build_corr_router
        but injects a bias tensor C_u[k, j, :] on u_{j,t} when expert j was
        chosen at t-1. Config shape: (M, N, d_u).
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
                    "routers.slds_imm_corr_recurrent.Q_g_scales must be a scalar, a list "
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
                    "routers.slds_imm_corr_recurrent.Q_u_scales must be a scalar, a list "
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

        # Optional recurrent bias; allow scaling via C_scale (default 1.0).
        C_u_cfg = cfg_local.get("C_u", None)
        C_scale_local = float(cfg_local.get("C_scale", 1.0))
        if C_u_cfg is not None:
            C_u_local = np.asarray(C_u_cfg, dtype=float)
            if C_u_local.shape != (M, N, d_u_local):
                raise ValueError("routers.slds_imm_corr_recurrent.C_u must have shape (M, N, d_u)")
        else:
            C_u_local = np.zeros((M, N, d_u_local), dtype=float)
        C_u_local = C_u_local * C_scale_local

        return RecurrentSLDSIMMRouter_Corr(
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
            C_u=C_u_local,
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
    router_partial_corr = _build_corr_router(
        slds_corr_cfg, slds_corr_partial_overrides, feedback_mode="partial"
    )
    router_full_corr = _build_corr_router(
        slds_corr_cfg, slds_corr_full_overrides, feedback_mode="full"
    )

    # Merge base correlated overrides with recurrent-specific overrides so that
    # users can specify partial/full overrides in both blocks; recurrent-specific
    # keys take precedence.
    merged_corr_rec_partial_overrides = {
        **slds_corr_partial_overrides,
        **slds_corr_rec_partial_overrides,
    }
    merged_corr_rec_full_overrides = {
        **slds_corr_full_overrides,
        **slds_corr_rec_full_overrides,
    }

    router_partial_corr_rec = _build_corr_router_rec(
        slds_corr_rec_cfg, merged_corr_rec_partial_overrides, feedback_mode="partial"
    )
    router_full_corr_rec = _build_corr_router_rec(
        slds_corr_rec_cfg, merged_corr_rec_full_overrides, feedback_mode="full"
    )

    # Optional EM-style correlated routers (distinct from the base
    # correlated routers above). If routers.slds_imm_corr_em is empty,
    # these remain None and are omitted from evaluation.
    router_partial_corr_em = None
    router_full_corr_em = None
    if slds_corr_em_cfg:
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
        )

    # --------------------------------------------------------
    # Environment and L2D baselines
    # --------------------------------------------------------
    
    # Environment: either synthetic or ETTh1, depending on env_cfg.
    if data_source == "etth1":
        # Real-world ETTh1 experiment (oil temperature as target).
        T_raw = env_cfg.get("T", None)
        T_env = None if T_raw is None else int(T_raw)
        csv_path = env_cfg.get("csv_path", "Data/ETTh1.csv")
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
            unavailable_intervals=env_cfg.get(
                "unavailable_intervals", [[10, 50], [200, 250]]
            ),
            arrival_expert_idx=int(env_cfg.get("arrival_expert_idx", 4)),
            arrival_intervals=env_cfg.get("arrival_intervals", [[120, 200]]),
            setting=setting,
            noise_scale=env_cfg.get("noise_scale", None),
        )

    # Plot the true series and expert predictions
    # plot_time_series(env)

    # Evaluate routers, L2D baseline, and constant-expert baselines,
    # and plot their induced prediction time series.
    evaluate_routers_and_baselines(
        env,
        router_partial,
        router_full,
        l2d_baseline,
        router_partial_rec=router_partial_rec,
        router_full_rec=router_full_rec,
        router_partial_corr=router_partial_corr,
        router_full_corr=router_full_corr,
        router_partial_corr_rec=router_partial_corr_rec,
        router_full_corr_rec=router_full_corr_rec,
        router_partial_corr_em=router_partial_corr_em,
        router_full_corr_em=router_full_corr_em,
        l2d_sw_baseline=l2d_sw_baseline,
        linucb_partial=linucb_partial,
        linucb_full=linucb_full,
        neuralucb_partial=neuralucb_partial,
        neuralucb_full=neuralucb_full,
        # router_partial_neural=router_partial_neural,
        # router_full_neural=router_full_neural,
    )

    
    # Optional: analyze reaction to a late-arriving expert. This can be
    # enabled either via the command-line flag --late-arrival-analysis
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
                    router_partial_rec=router_partial_rec,
                    router_full_rec=router_full_rec,
                    router_partial_corr=router_partial_corr,
                    router_full_corr=router_full_corr,
                    router_partial_corr_rec=router_partial_corr_rec,
                    router_full_corr_rec=router_full_corr_rec,
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

    # --------------------------------------------------------
    # Example: horizon-H planning from a given time t
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
    t0 = int(horizon_cfg.get("t0", 175))
    H = int(horizon_cfg.get("H", 5))
    planning_method = str(horizon_cfg.get("method", "regressive"))
    scenario_generator_cfg = horizon_cfg.get("scenario_generator", {})
    # Separate L2D baseline instance for horizon-only evaluation (trained up to t0)
    l2d_baseline_horizon = L2D(
        num_experts=N,
        feature_fn=feature_phi,
        alpha=np.ones(N, dtype=float),
        beta=beta,
        learning_rate=1e-2,
        arch="mlp",
        hidden_dim=8,
        window_size=1,
    )

    # evaluate_horizon_planning(
    #     env=env,
    #     router_partial=router_partial,
    #     router_full=router_full,
    #     beta=beta,
    #     t0=t0,
    #     H=H,
    #     experts_predict=experts_predict,
    #     context_update=context_update,
    #     l2d_baseline=l2d_baseline_horizon,
    #     router_partial_corr=router_partial_corr,
    #     router_full_corr=router_full_corr,
    #     router_partial_corr_em=router_partial_corr_em,
    #     router_full_corr_em=router_full_corr_em,
    #     # router_partial_neural=router_partial_neural,
    #     # router_full_neural=router_full_neural,
    #     planning_method=planning_method,
    #     scenario_generator_cfg=scenario_generator_cfg,
    # )
    
    # NeuralUCB baseline (single policy; partial feedback by default)
    # neuralucb_baseline = None
    # if neuralucb_cfg:
    #     alpha_ucb_nn = float(neuralucb_cfg.get("alpha_ucb", 1.0))
    #     lambda_reg_nn = float(neuralucb_cfg.get("lambda_reg", 1.0))
    #     hidden_dim_nn = int(neuralucb_cfg.get("hidden_dim", 16))
    #     nn_lr = float(neuralucb_cfg.get("nn_learning_rate", 1e-3))
    #     neuralucb_baseline = NeuralUCB(
    #         num_experts=N,
    #         feature_fn=feature_phi,
    #         alpha_ucb=alpha_ucb_nn,
    #         lambda_reg=lambda_reg_nn,
    #         beta=beta,
    #         hidden_dim=hidden_dim_nn,
    #         nn_learning_rate=nn_lr,
    #         feedback_mode="partial",
    #     )
