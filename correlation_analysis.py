# %%
# %matplotlib widget

import argparse
import copy
import json
import os
import random
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

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
from horizon_planning import evaluate_horizon_planning
from router_eval import run_f_router_on_env_return_idiosyncratic_factor

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _load_config(path: str = "config/config_synth_paper.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:  # "r" means read mode
        text = f.read()  # by default reads config.yaml
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
    if arr.shape == ():  # scalar
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
        default="config/config_synth_paper.yaml",
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


def analysis_correlations_in_experts(
    env: SyntheticTimeSeriesEnv | ETTh1TimeSeriesEnv,
    router_partial_no_g: SLDSIMMRouter,
    router_full_no_g: SLDSIMMRouter,
    router_factorial_partial: Optional[FactorizedSLDS],
    router_factorial_full: Optional[FactorizedSLDS],
    factorized_label: str = "Factorized SLDS",
    router_factorial_partial_linear: Optional[FactorizedSLDS] = None,
    router_factorial_full_linear: Optional[FactorizedSLDS] = None,
    factorized_linear_label: str = "Factorized SLDS linear",
    seed: int = 0,
) -> None:
    """
    Plot the underlying latent u_k for each expert k=0,...,N-1 over time, plot the mean and variance bar
    One plot for each router: partial no-g, full no-g, factorized partial, factorized full

    the correlated structure in synthetic environment:
        - "theoretical_trap": Phase 1 in regime 0 (good times),
                              Phase 2 in regime 1 (bad times)
                phase1_end = min(1000, T - 1)

            # Stale Prior trap experiment:
            # Regime 0 (good times): experts 0 and 1 ~ 0; expert 2 ~ 2.
            # Regime 1 (bad times):  experts 0 and 1 ~ 4 on average,
            # with enough variance that they are occasionally better
            # than expert 2, which remains ~2. This preserves
            # μ_hist < μ_safe < μ_latent in expectation, but makes
            # Regime 1 decisions non-trivial.

            # In the bad regime (Regime 2), draw expert-0/1 losses from a
            # broader distribution centered at 4.0 so that they
            # sometimes fall below 2.0.

            This design makes the theoretical trap particularly challenging
            because the router cannot easily distinguish between the highly correlated Experts 0 & 1,
    """
    # Note: not consider difference between attention and linear
    def _run_factorized_router_collect_u_k(router: FactorizedSLDS):
        u_ks = []
        router.reset_beliefs()
        idiosyncratic_states = run_f_router_on_env_return_idiosyncratic_factor(router, env)

        print(f"idiosyncratic_means: {idiosyncratic_states.mean}")
        N = router._get_N()
        for expert_idx in range(N):
            u_k = idiosyncratic_states[expert_idx]
            u_ks.append(u_k)
        # each u_ks[i] is array of shape (T, d_u)
        N = len(u_ks)
        T, d_r, d_c = u_ks[0].shape  # (T, 2, 2)

        for k in range(N):
            plt.figure()
            for i in range(d_r):
                for j in range(d_c):
                    plt.plot(
                        u_ks[k][:, i, j],
                        label=f"({i},{j})"
                    )

            plt.title(f"Expert {k}: entries of $u_k(t)$")
            plt.xlabel("Time t")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.show()

    if router_factorial_partial is not None:
        print(f"\n--- Analyzing idiosyncratic factors for {factorized_label} (partial) ---")
        _run_factorized_router_collect_u_k(router_factorial_partial)
    if router_factorial_full is not None:
        print(f"\n--- Analyzing idiosyncratic factors for {factorized_label} (full) ---")
        _run_factorized_router_collect_u_k(router_factorial_full)
    if router_factorial_partial_linear is not None:
        print(f"\n--- Analyzing idiosyncratic factors for {factorized_linear_label} (partial) ---")
        _run_factorized_router_collect_u_k(router_factorial_partial_linear)
    if router_factorial_full_linear is not None:
        print(f"\n--- Analyzing idiosyncratic factors for {factorized_linear_label} (full) ---")
        _run_factorized_router_collect_u_k(router_factorial_full_linear)






if __name__ == "__main__":
    args = _parse_args()
    cfg = _load_config(args.config)
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
    N = int(env_cfg.get("num_experts"))  # experts
    # State dimension (= dim φ(x)); feature map in router_model.py currently
    # returns a 2D feature, so d must be compatible with that.
    d = int(env_cfg.get("state_dim"))

    M = int(env_cfg.get("num_regimes"))

    print(f"Using num_regimes = {M}.")

    # Risk sensitivity λ; can be scalar or length-M vector. If a
    # 2-element vector is provided while num_regimes > 2, we interpret
    # it as [λ_0, λ_other] and broadcast λ_other to regimes 1,...,M-1.
    lambda_cfg = routers_cfg.get("lambda_risk", -0.2)
    lambda_arr = np.asarray(lambda_cfg, dtype=float)
    if lambda_arr.shape == ():
        lambda_risk = float(lambda_arr)
    elif lambda_arr.shape == (M,):  # one dimension array with length M
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

    staleness_threshold_cfg = routers_cfg.get("staleness_threshold", None)
    staleness_threshold = (
        int(staleness_threshold_cfg) if staleness_threshold_cfg is not None else None
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
    if l2d_sw_cfg:  # overridden with RNN architecture with sliding window
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
                seed=seed,
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
                seed=seed,
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
    # Visualization-only settings for plots.
    env.plot_shift = int(cfg.get("plot_shift", 1))
    env.plot_target = str(cfg.get("plot_target", "y")).lower()


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
        _run_factorized_em(
            fact_router_full, f"{base_transition_mode} full"
        )
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
    analysis_correlations_in_experts(
        env,
        router_partial_no_g,
        router_full_no_g,
        fact_router_partial,
        fact_router_full,
        factorized_label=factorized_label,
        router_factorial_partial_linear=fact_router_partial_linear,
        router_factorial_full_linear=fact_router_full_linear,
        factorized_linear_label=factorized_linear_label,
        seed=seed,
    )


