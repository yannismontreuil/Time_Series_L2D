# %%
# %matplotlib widget

import argparse
import json
import os
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

from plot_utils import (
    evaluate_routers_and_baselines,
    analysis_late_arrival,
)

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
):
    contexts = []
    available_sets = []
    actions = []
    residuals = []
    residuals_full = []

    router.reset_beliefs()
    t_end = min(int(t_end), env.T - 1)
    for t in range(1, t_end + 1):
        x_t = env.get_context(t)
        available = env.get_available_experts(t)
        r_t, cache = router.select_expert(x_t, available)

        preds = env.all_expert_predictions(x_t)
        residuals_all = preds - float(env.y[t])
        residual = float(residuals_all[int(r_t)])

        contexts.append(x_t)
        available_sets.append(list(available))
        actions.append(int(r_t))
        residuals.append(residual)
        residuals_full.append(residuals_all)

        losses_full = residuals_all if router.feedback_mode == "full" else None
        router.update_beliefs(
            r_t=r_t,
            loss_obs=residual,
            losses_full=losses_full,
            available_experts=available,
            cache=cache,
        )

    return contexts, available_sets, actions, residuals, residuals_full


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
    env_cfg = cfg.get("environment", {})

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
    # Factorized Switching Linear Dynamical System
    # --------------------------------------------------------
    # Wire the FactorizedSLDS into the evaluation if configured
    fact_router_partial = None
    fact_router_full = None
    fact_router_partial_linear = None
    fact_router_full_linear = None
    router_partial_no_g = None
    router_full_no_g = None
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

        def _build_factorized_router(
            feedback_mode: str,
            transition_mode_local: str,
        ) -> FactorizedSLDS:
            return FactorizedSLDS(
                M=M_fact,
                d_g=d_g_fact,
                d_phi=d_phi_fact,
                feature_fn=feature_phi,
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
                observation_mode=observation_mode_fact,
                transition_hidden_dims=transition_hidden_dims,
                transition_activation=transition_activation,
                transition_device=transition_device,
                transition_mode=transition_mode_local,
                feedback_mode=feedback_mode,
            )

        def _build_factorized_router_no_g(
            feedback_mode: str,
            transition_mode_local: str,
        ) -> FactorizedSLDS:
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
                observation_mode=observation_mode_fact,
                transition_hidden_dims=transition_hidden_dims,
                transition_activation=transition_activation,
                transition_device=transition_device,
                transition_mode=transition_mode_local,
                feedback_mode=feedback_mode,
            )

        print(
            f"\n--- Running FactorizedSLDS Router (partial, {base_transition_mode}) ---"
        )
        fact_router_partial = _build_factorized_router(
            "partial", base_transition_mode
        )
        print(
            f"\n--- Running FactorizedSLDS Router (full, {base_transition_mode}) ---"
        )
        fact_router_full = _build_factorized_router("full", base_transition_mode)
        factorized_label = f"Factorized SLDS {base_transition_mode}"

        if extra_transition_mode is not None:
            print(
                f"\n--- Running FactorizedSLDS Router (partial, {extra_transition_mode}) ---"
            )
            fact_router_partial_linear = _build_factorized_router(
                "partial", extra_transition_mode
            )
            print(
                f"\n--- Running FactorizedSLDS Router (full, {extra_transition_mode}) ---"
            )
            fact_router_full_linear = _build_factorized_router(
                "full", extra_transition_mode
            )
            factorized_linear_label = f"Factorized SLDS {extra_transition_mode}"

        router_partial_no_g = _build_factorized_router_no_g(
            "partial", base_transition_mode
        )
        router_full_no_g = _build_factorized_router_no_g(
            "full", base_transition_mode
        )
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
        )

    def _run_factorized_em(router: Optional[FactorizedSLDS], label: str) -> None:
        if router is None:
            return
        em_enabled = bool(factorized_slds_cfg.get("em_enabled", True))
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
        theta_lr = float(factorized_slds_cfg.get("em_theta_lr", 1e-2))
        theta_steps = int(factorized_slds_cfg.get("em_theta_steps", 1))
        em_seed_cfg = factorized_slds_cfg.get("em_seed", env_cfg.get("seed", 0))
        em_seed = int(em_seed_cfg) if em_seed_cfg is not None else 0
        em_priors = factorized_slds_cfg.get("em_priors", None)
        print_val_loss = bool(factorized_slds_cfg.get("em_print_val_loss", True))

        print(f"\n--- FactorizedSLDS EM ({label}) (t=1..{em_tk}, n_em={n_em}) ---")
        ctx_em, avail_em, actions_em, resid_em, resid_full_em = _collect_factorized_em_data(
            router, env, em_tk
        )
        router.fit_em(
            contexts=ctx_em,
            available_sets=avail_em,
            actions=actions_em,
            residuals=resid_em,
            residuals_full=resid_full_em if router.feedback_mode == "full" else None,
            n_em=n_em,
            n_samples=n_samples,
            burn_in=burn_in,
            val_fraction=val_fraction,
            priors=em_priors,
            theta_lr=theta_lr,
            theta_steps=theta_steps,
            seed=em_seed,
            print_val_loss=print_val_loss,
        )
        router.em_tk = int(em_tk)
        router.reset_beliefs()

    if base_transition_mode is not None:
        _run_factorized_em(
            fact_router_partial, f"{base_transition_mode} partial"
        )
        _run_factorized_em(fact_router_full, f"{base_transition_mode} full")
        _run_factorized_em(
            router_partial_no_g, f"{base_transition_mode} no-g partial"
        )
        _run_factorized_em(
            router_full_no_g, f"{base_transition_mode} no-g full"
        )
    if extra_transition_mode is not None:
        _run_factorized_em(
            fact_router_partial_linear, f"{extra_transition_mode} partial"
        )
        _run_factorized_em(
            fact_router_full_linear, f"{extra_transition_mode} full"
        )

    # Plot the true series and expert predictions
    # plot_time_series(env)

    # Evaluate routers, L2D baseline, and constant-expert baselines,
    # and plot their induced prediction time series.
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
        # router_partial_neural=router_partial_neural,
        # router_full_neural=router_full_neural,
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

    # --------------------------------------------------------
    '''
    # Example: horizon-H planning from a given time t
    # --------------------------------------------------------

    '''
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
    '''

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
