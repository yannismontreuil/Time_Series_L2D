import copy
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.etth1_env import ETTh1TimeSeriesEnv
from models.factorized_slds import FactorizedSLDS
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB
from models.router_model import feature_phi
from models.shared_linear_bandits import (
    LinearEnsembleSampling,
    LinearThompsonSampling,
    SharedLinUCB,
)
from router_eval import (
    run_ensemble_sampling_on_env,
    run_factored_router_on_env,
    run_lin_ts_on_env,
    run_linucb_on_env,
    run_neuralucb_on_env,
    run_random_on_env,
    run_shared_linucb_on_env,
)
from scripts.measure_jena_runtime import _apply_factorized_init


def _diag_stack_broadcast(scales, d: int, M: int) -> np.ndarray:
    arr = np.asarray(scales, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(M, float(arr[0]), dtype=float)
    elif arr.size != M:
        raise ValueError(f"Expected scalar or length-{M} scales, got shape {arr.shape}.")
    out = np.zeros((M, d, d), dtype=float)
    for m, val in enumerate(arr):
        out[m] = float(val) * np.eye(d, dtype=float)
    return out


def _build_factorized_pair(cfg: dict, env: ETTh1TimeSeriesEnv):
    env_cfg = cfg["environment"]
    routers_cfg = cfg["routers"]
    factor_cfg = routers_cfg["factorized_slds"]
    N = int(env_cfg["num_experts"])
    M = int(factor_cfg["num_regimes"])
    d_phi = int(factor_cfg["idiosyncratic_dim"])
    d_g = int(factor_cfg["shared_dim"])
    beta = np.zeros(N, dtype=float)
    A_g = np.asarray(factor_cfg["A_g"], dtype=float)
    A_u = _diag_stack_broadcast(factor_cfg["A_u_scale"], d_phi, M)
    Q_g = _diag_stack_broadcast(factor_cfg["Q_g_scales"], d_g, M)
    Q_u = _diag_stack_broadcast(factor_cfg["Q_u_scales"], d_phi, M)
    B_dict_cfg = factor_cfg.get("B_dict", {})
    B_dict = {
        int(k): np.asarray(v, dtype=float)
        for k, v in B_dict_cfg.items()
    }
    common = dict(
        M=M,
        d_phi=d_phi,
        feature_fn=feature_phi,
        beta=beta,
        Delta_max=int(factor_cfg["delta_max"]),
        R=float(factor_cfg.get("R_scalar", 1.0)),
        R_mode="scalar",
        num_experts=N,
        B_dict=B_dict,
        B_intercept_load=float(factor_cfg.get("B_intercept_load", 1.0)),
        attn_dim=int(factor_cfg.get("attn_dim", d_phi)),
        A_u=A_u,
        Q_u=Q_u,
        eps=float(factor_cfg.get("eps", 1e-8)),
        exploration=str(factor_cfg.get("exploration", ["g"])[0]),
        exploration_mc_samples=int(factor_cfg.get("exploration_mc_samples", 25)),
        exploration_ucb_samples=int(factor_cfg.get("exploration_ucb_samples", 200)),
        exploration_ucb_alpha=factor_cfg.get("exploration_ucb_alpha", None),
        exploration_ucb_schedule=str(factor_cfg.get("exploration_ucb_schedule", "inverse_t")),
        exploration_sampling_deterministic=bool(
            factor_cfg.get("exploration_sampling_deterministic", False)
        ),
        exploration_diag_enabled=False,
        exploration_diag_stride=int(factor_cfg.get("exploration_diag_stride", 100)),
        exploration_diag_samples=int(factor_cfg.get("exploration_diag_samples", 50)),
        exploration_diag_print=False,
        exploration_diag_max_records=int(factor_cfg.get("exploration_diag_max_records", 2000)),
        observation_mode="residual",
        transition_init=str(factor_cfg.get("transition_init", "uniform")),
        transition_mode=str(factor_cfg.get("transition_mode", "attention")),
        feedback_mode="partial",
        seed=int(factor_cfg.get("seed", 42)),
    )
    ours = FactorizedSLDS(
        d_g=d_g,
        A_g=A_g,
        Q_g=Q_g,
        **common,
    )
    no_g = FactorizedSLDS(
        d_g=0,
        A_g=None,
        Q_g=None,
        **common,
    )
    _apply_factorized_init(env, [ours, no_g], factor_cfg)
    return ours, no_g


def _load_cfg(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_env(cfg: dict) -> ETTh1TimeSeriesEnv:
    env_cfg = cfg["environment"]
    T_raw = env_cfg.get("T", None)
    T_env = None if T_raw is None else int(T_raw)
    return ETTh1TimeSeriesEnv(
        csv_path=env_cfg.get("csv_path", "data/FRED_DGS10.csv"),
        target_column=env_cfg.get("target_column", "DGS10"),
        num_experts=int(env_cfg["num_experts"]),
        num_regimes=int(env_cfg["num_regimes"]),
        T=T_env,
        seed=int(env_cfg.get("seed", 42)),
        data_seed=env_cfg.get("data_seed", None),
        unavailable_expert_idx=env_cfg.get("unavailable_expert_idx", None),
        unavailable_intervals=env_cfg.get("unavailable_intervals", None),
        arrival_expert_idx=env_cfg.get("arrival_expert_idx", None),
        arrival_intervals=env_cfg.get("arrival_intervals", None),
        context_columns=env_cfg.get("context_columns", None),
        context_lags=env_cfg.get("context_lags", None),
        row_stride=env_cfg.get("row_stride", 1),
        include_time_features=env_cfg.get("include_time_features", False),
        time_features=env_cfg.get("time_features", None),
        normalize_context=env_cfg.get("normalize_context", False),
        normalization_window=env_cfg.get("normalization_window", None),
        normalization_eps=env_cfg.get("normalization_eps", 1e-6),
        normalization_mode=env_cfg.get("normalization_mode", "zscore"),
        feature_expansions=env_cfg.get("feature_expansions", None),
        lag_diff_pairs=env_cfg.get("lag_diff_pairs", None),
        expert_archs=env_cfg.get("expert_archs", None),
        nn_expert_type=env_cfg.get("nn_expert_type", None),
        rnn_hidden_dim=env_cfg.get("rnn_hidden_dim", 8),
        rnn_spectral_radius=env_cfg.get("rnn_spectral_radius", 0.9),
        rnn_ridge=env_cfg.get("rnn_ridge", 1e-3),
        rnn_washout=env_cfg.get("rnn_washout", 5),
        rnn_input_scale=env_cfg.get("rnn_input_scale", 0.5),
        rnn_share_reservoir=env_cfg.get("rnn_share_reservoir", True),
        rnn_hidden_dims=env_cfg.get("rnn_hidden_dims", None),
        rnn_spectral_radii=env_cfg.get("rnn_spectral_radii", None),
        rnn_ridges=env_cfg.get("rnn_ridges", None),
        rnn_washouts=env_cfg.get("rnn_washouts", None),
        rnn_input_scales=env_cfg.get("rnn_input_scales", None),
        expert_pred_noise_std=env_cfg.get("expert_pred_noise_std", None),
        expert_train_ranges=env_cfg.get("expert_train_ranges", None),
        expert_train_date_ranges=env_cfg.get("expert_train_date_ranges", None),
        arima_lags=env_cfg.get("arima_lags", None),
        arima_diff_order=env_cfg.get("arima_diff_order", 0),
    )


def _build_baselines(cfg: dict, env: ETTh1TimeSeriesEnv):
    env_cfg = cfg["environment"]
    base_cfg = cfg["baselines"]
    N = int(env_cfg["num_experts"])
    d = int(env_cfg["state_dim"])
    beta = np.zeros(N, dtype=float)
    linucb = LinUCB(
        num_experts=N,
        feature_fn=feature_phi,
        alpha_ucb=float(base_cfg["linucb"]["alpha_ucb"]),
        lambda_reg=float(base_cfg["linucb"].get("lambda_reg", base_cfg["linucb"].get("l2_reg", 1.0))),
        beta=beta,
        feedback_mode=str(base_cfg["linucb"].get("feedback_mode", "partial")),
        context_dim=d,
    )
    shared = SharedLinUCB(
        num_experts=N,
        feature_fn=feature_phi,
        alpha_ucb=float(base_cfg["shared_linucb"]["alpha_ucb"]),
        lambda_reg=float(base_cfg["shared_linucb"]["lambda_reg"]),
        beta=beta,
        feedback_mode=str(base_cfg["shared_linucb"].get("feedback_mode", "partial")),
        context_dim=d,
        include_shared_context=bool(base_cfg["shared_linucb"].get("include_shared_context", True)),
        include_arm_bias=bool(base_cfg["shared_linucb"].get("include_arm_bias", True)),
        include_arm_interactions=bool(base_cfg["shared_linucb"].get("include_arm_interactions", True)),
        seed=int(cfg["environment"].get("seed", 13)),
    )
    neural = NeuralUCB(
        num_experts=N,
        feature_fn=feature_phi,
        alpha_ucb=float(base_cfg["neural_ucb"]["alpha_ucb"]),
        lambda_reg=float(base_cfg["neural_ucb"]["lambda_reg"]),
        beta=beta,
        hidden_dim=int(base_cfg["neural_ucb"]["hidden_dim"]),
        nn_learning_rate=float(base_cfg["neural_ucb"]["nn_learning_rate"]),
        feedback_mode=str(base_cfg["neural_ucb"].get("feedback_mode", "partial")),
        seed=int(cfg["environment"].get("seed", 13)),
        context_dim=d,
    )
    lints = LinearThompsonSampling(
        num_experts=N,
        feature_fn=feature_phi,
        lambda_reg=float(base_cfg["lin_ts"]["lambda_reg"]),
        beta=beta,
        feedback_mode=str(base_cfg["lin_ts"].get("feedback_mode", "partial")),
        context_dim=d,
        posterior_scale=float(base_cfg["lin_ts"]["posterior_scale"]),
        include_shared_context=bool(base_cfg["lin_ts"].get("include_shared_context", True)),
        include_arm_bias=bool(base_cfg["lin_ts"].get("include_arm_bias", True)),
        include_arm_interactions=bool(base_cfg["lin_ts"].get("include_arm_interactions", True)),
        seed=int(cfg["environment"].get("seed", 13)),
    )
    ensemble = LinearEnsembleSampling(
        num_experts=N,
        feature_fn=feature_phi,
        ensemble_size=int(base_cfg["ensemble_sampling"]["ensemble_size"]),
        lambda_reg=float(base_cfg["ensemble_sampling"]["lambda_reg"]),
        obs_noise_std=float(base_cfg["ensemble_sampling"]["obs_noise_std"]),
        beta=beta,
        feedback_mode=str(base_cfg["ensemble_sampling"].get("feedback_mode", "partial")),
        context_dim=d,
        include_shared_context=bool(base_cfg["ensemble_sampling"].get("include_shared_context", True)),
        include_arm_bias=bool(base_cfg["ensemble_sampling"].get("include_arm_bias", True)),
        include_arm_interactions=bool(base_cfg["ensemble_sampling"].get("include_arm_interactions", True)),
        seed=int(cfg["environment"].get("seed", 13)),
    )
    return linucb, shared, neural, lints, ensemble, beta


def _measure(method_name: str, obj, runner, env, repeats: int = 2):
    elapsed = []
    costs_out = None
    for _ in range(repeats):
        if hasattr(obj, "reset_beliefs"):
            obj.reset_beliefs()
        if hasattr(obj, "reset_state"):
            obj.reset_state()
        t0 = time.perf_counter()
        costs, _ = runner(obj, env)
        t1 = time.perf_counter()
        elapsed.append(t1 - t0)
        if costs_out is None:
            costs_out = costs
    mean_cost = float(np.nanmean(costs_out))
    ms_per_step = 1000.0 * float(np.mean(elapsed)) / float(env.T - 1)
    return method_name, mean_cost, ms_per_step


def _measure_random(beta: np.ndarray, env: ETTh1TimeSeriesEnv, seed: int, repeats: int = 2):
    elapsed = []
    costs_out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        costs, _ = run_random_on_env(env, beta=beta, seed=seed)
        t1 = time.perf_counter()
        elapsed.append(t1 - t0)
        if costs_out is None:
            costs_out = costs
    mean_cost = float(np.nanmean(costs_out))
    ms_per_step = 1000.0 * float(np.mean(elapsed)) / float(env.T - 1)
    return "Random", mean_cost, ms_per_step


def main() -> None:
    cfg = _load_cfg(ROOT / "config" / "exp_FRED_rebuttal.yaml")
    env = _build_env(cfg)
    ours, ablation = _build_factorized_pair(cfg, env)
    linucb, shared, neural, lints, ensemble, beta = _build_baselines(cfg, env)

    rows = []
    rows.append(_measure("L2D-SLDS", ours, run_factored_router_on_env, env))
    rows.append(_measure("L2D-SLDS w/o $g_t$", ablation, run_factored_router_on_env, env))
    rows.append(_measure("LinUCB", linucb, run_linucb_on_env, env))
    rows.append(_measure("SharedLinUCB", shared, run_shared_linucb_on_env, env))
    rows.append(_measure("NeuralUCB", neural, run_neuralucb_on_env, env))
    rows.append(_measure("LinTS", lints, run_lin_ts_on_env, env))
    rows.append(_measure("EnsembleSampling", ensemble, run_ensemble_sampling_on_env, env))
    rows.append(_measure_random(beta, env, seed=int(cfg["environment"].get("seed", 13))))

    print("Method | Mean Cost | Avg Time / Step (ms)")
    for name, mean_cost, ms_per_step in rows:
        print(f"{name} | {mean_cost:.6f} | {ms_per_step:.3f}")


if __name__ == "__main__":
    main()
