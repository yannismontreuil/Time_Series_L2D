import copy
import os
import pathlib
import tempfile
import time
from argparse import ArgumentParser
from typing import Any, Callable

import yaml

from environment.etth1_env import ETTh1TimeSeriesEnv, ensure_daily_temp_csv
from models.factorized_slds import FactorizedSLDS
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB
from models.shared_linear_bandits import (
    LinearEnsembleSampling,
    LinearThompsonSampling,
    SharedLinUCB,
)
from router_eval import (
    run_ensemble_sampling_on_env,
    run_lin_ts_on_env,
    run_linucb_on_env,
    run_neuralucb_on_env,
    run_router_on_env,
    run_shared_linucb_on_env,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "config_melbourne_review.yaml"


def _load_cfg(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_args():
    parser = ArgumentParser(description="Measure Melbourne online runtime per step.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=CFG_PATH,
        help="Config path to evaluate. Defaults to config_melbourne_review.yaml.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional horizon override for faster runtime checks.",
    )
    return parser.parse_args()


def _make_temp_cfg(base_cfg: dict[str, Any], horizon: int | None = None) -> pathlib.Path:
    cfg = copy.deepcopy(base_cfg)
    if horizon is not None:
        cfg["environment"]["T"] = horizon
    cfg["analysis"]["tri_cycle_corr"]["enabled"] = False
    cfg["analysis"]["pruning"]["enabled"] = False
    cfg["plot_time_series_pdf"] = False
    cfg["plot_time_series_png"] = False
    tmp = tempfile.NamedTemporaryFile(
        "w",
        suffix=".yaml",
        prefix="melbourne_runtime_",
        delete=False,
        dir=str(ROOT / "config"),
        encoding="utf-8",
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return pathlib.Path(tmp.name)


def _build_env(cfg: dict[str, Any]) -> ETTh1TimeSeriesEnv:
    env_cfg = cfg["environment"]
    csv_path = env_cfg.get("csv_path", "data/daily_temp_melbourne.csv")
    if not os.path.exists(csv_path):
        csv_path = ensure_daily_temp_csv(csv_path)
    target_column = env_cfg.get("target_column", "temp")
    T_raw = env_cfg.get("T", None)
    T_env = None if T_raw is None else int(T_raw)
    return ETTh1TimeSeriesEnv(
        csv_path=csv_path,
        target_column=target_column,
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
        analysis_window=env_cfg.get("analysis_window", None),
    )


def _measure(
    name: str,
    runner: Callable[..., Any],
    actor: Any,
    env: ETTh1TimeSeriesEnv,
) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    if runner is run_router_on_env:
        _, costs = runner(env, actor, feedback_mode="partial")
    else:
        _, costs = runner(env, actor)
    elapsed = time.perf_counter() - t0
    avg_cost = float(sum(costs) / len(costs))
    ms_per_step = 1000.0 * elapsed / len(costs)
    return name, avg_cost, ms_per_step


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config.resolve())
    tmp_path = _make_temp_cfg(cfg, args.horizon)
    try:
        cfg_runtime = _load_cfg(tmp_path)
        env = _build_env(cfg_runtime)
        routers = cfg_runtime["routers"]
        baselines = cfg_runtime["baselines"]
        factor_cfg = routers["factorized_slds"]

        ours = FactorizedSLDS(
            num_experts=env.n_experts,
            context_dim=env.context_dim,
            shared_dim=factor_cfg["shared_dim"],
            idiosyncratic_dim=factor_cfg["idiosyncratic_dim"],
            num_regimes=factor_cfg["num_regimes"],
            include_g=True,
            feedback_mode=factor_cfg["feedback_mode"],
            delta_max=factor_cfg["delta_max"],
            R_scalar=factor_cfg["R_scalar"],
            A_g=factor_cfg["A_g"],
            Q_g_scales=factor_cfg["Q_g_scales"],
            A_u_scale=factor_cfg["A_u_scale"],
            Q_u_scales=factor_cfg["Q_u_scales"],
            exploration=factor_cfg["exploration"],
            exploration_diag_enabled=factor_cfg.get("exploration_diag_enabled", False),
            init_R_mode=factor_cfg["init_R_mode"],
            init_R_window=factor_cfg["init_R_window"],
            init_R_use_available=factor_cfg["init_R_use_available"],
            init_R_floor=factor_cfg["init_R_floor"],
            init_R_print=factor_cfg["init_R_print"],
            init_state_from_residuals=factor_cfg["init_state_from_residuals"],
            init_state_window=factor_cfg["init_state_window"],
            init_state_u_scale=factor_cfg["init_state_u_scale"],
            init_state_g_scale=factor_cfg["init_state_g_scale"],
            init_state_print=factor_cfg["init_state_print"],
            transition_mode=factor_cfg["transition_mode"],
            transition_init=factor_cfg.get("transition_init", "random"),
            seed=factor_cfg["seed"],
        )
        no_g = FactorizedSLDS(
            num_experts=env.n_experts,
            context_dim=env.context_dim,
            shared_dim=factor_cfg["shared_dim"],
            idiosyncratic_dim=factor_cfg["idiosyncratic_dim"],
            num_regimes=factor_cfg["num_regimes"],
            include_g=False,
            feedback_mode=factor_cfg["feedback_mode"],
            delta_max=factor_cfg["delta_max"],
            R_scalar=factor_cfg["R_scalar"],
            A_g=factor_cfg["A_g"],
            Q_g_scales=factor_cfg["Q_g_scales"],
            A_u_scale=factor_cfg["A_u_scale"],
            Q_u_scales=factor_cfg["Q_u_scales"],
            exploration=factor_cfg["exploration"],
            exploration_diag_enabled=factor_cfg.get("exploration_diag_enabled", False),
            init_R_mode=factor_cfg["init_R_mode"],
            init_R_window=factor_cfg["init_R_window"],
            init_R_use_available=factor_cfg["init_R_use_available"],
            init_R_floor=factor_cfg["init_R_floor"],
            init_R_print=factor_cfg["init_R_print"],
            init_state_from_residuals=factor_cfg["init_state_from_residuals"],
            init_state_window=factor_cfg["init_state_window"],
            init_state_u_scale=factor_cfg["init_state_u_scale"],
            init_state_g_scale=factor_cfg["init_state_g_scale"],
            init_state_print=factor_cfg["init_state_print"],
            transition_mode=factor_cfg["transition_mode"],
            transition_init=factor_cfg.get("transition_init", "random"),
            seed=factor_cfg["seed"],
        )

        linucb = LinUCB(
            n_experts=env.n_experts,
            context_dim=env.context_dim,
            alpha_ucb=baselines["linucb"]["alpha_ucb"],
            l2_reg=baselines["linucb"]["l2_reg"],
            feedback_mode=baselines["linucb"]["feedback_mode"],
        )
        shared = SharedLinUCB(
            n_experts=env.n_experts,
            context_dim=env.context_dim,
            alpha_ucb=baselines["shared_linucb"]["alpha_ucb"],
            lambda_reg=baselines["shared_linucb"]["lambda_reg"],
            feedback_mode=baselines["shared_linucb"]["feedback_mode"],
            include_shared_context=baselines["shared_linucb"]["include_shared_context"],
            include_arm_bias=baselines["shared_linucb"]["include_arm_bias"],
            include_arm_interactions=baselines["shared_linucb"]["include_arm_interactions"],
        )
        neural = NeuralUCB(
            n_experts=env.n_experts,
            context_dim=env.context_dim,
            alpha_ucb=baselines["neural_ucb"]["alpha_ucb"],
            lambda_reg=baselines["neural_ucb"]["lambda_reg"],
            hidden_dim=baselines["neural_ucb"]["hidden_dim"],
            nn_learning_rate=baselines["neural_ucb"]["nn_learning_rate"],
            feedback_mode=baselines["neural_ucb"]["feedback_mode"],
        )
        lints = LinearThompsonSampling(
            n_experts=env.n_experts,
            context_dim=env.context_dim,
            lambda_reg=baselines["lin_ts"]["lambda_reg"],
            posterior_scale=baselines["lin_ts"]["posterior_scale"],
            feedback_mode=baselines["lin_ts"]["feedback_mode"],
            include_shared_context=baselines["lin_ts"]["include_shared_context"],
            include_arm_bias=baselines["lin_ts"]["include_arm_bias"],
            include_arm_interactions=baselines["lin_ts"]["include_arm_interactions"],
        )
        ensemble = LinearEnsembleSampling(
            n_experts=env.n_experts,
            context_dim=env.context_dim,
            ensemble_size=baselines["ensemble_sampling"]["ensemble_size"],
            lambda_reg=baselines["ensemble_sampling"]["lambda_reg"],
            obs_noise_std=baselines["ensemble_sampling"]["obs_noise_std"],
            feedback_mode=baselines["ensemble_sampling"]["feedback_mode"],
            include_shared_context=baselines["ensemble_sampling"]["include_shared_context"],
            include_arm_bias=baselines["ensemble_sampling"]["include_arm_bias"],
            include_arm_interactions=baselines["ensemble_sampling"]["include_arm_interactions"],
        )

        rows = [
            _measure("L2D-SLDS", run_router_on_env, ours, env),
            _measure("L2D-SLDS w/o $g_t$", run_router_on_env, no_g, env),
            _measure("LinUCB", run_linucb_on_env, linucb, env),
            _measure("SharedLinUCB", run_shared_linucb_on_env, shared, env),
            _measure("NeuralUCB", run_neuralucb_on_env, neural, env),
            _measure("LinTS", run_lin_ts_on_env, lints, env),
            _measure("EnsembleSampling", run_ensemble_sampling_on_env, ensemble, env),
        ]
        for name, avg_cost, ms_per_step in rows:
            print(f"{name:18s} | {avg_cost:.4f} | {ms_per_step:.3f} ms/step")
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
