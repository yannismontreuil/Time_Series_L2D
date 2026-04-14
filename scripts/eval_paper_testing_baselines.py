import argparse
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.etth1_env import ETTh1TimeSeriesEnv, ensure_daily_temp_csv, ensure_jena_climate_csv
from models.factorized_slds import FactorizedSLDS
from models.neuralucb_baseline import NeuralUCB
from models.nonstationary_linear_bandits import (
    CUSUMLinUCB,
    DiscountedLinUCB,
    GLRLinUCB,
    SlidingWindowNeuralUCB,
)
from models.router_model import feature_phi
from router_eval import run_factored_router_on_env, run_neuralucb_on_env


def _load_cfg(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _diag_stack(scales: list[float], d: int) -> np.ndarray:
    arr = np.asarray(scales, dtype=float).reshape(-1)
    out = np.zeros((arr.shape[0], d, d), dtype=float)
    for m, val in enumerate(arr):
        out[m] = float(val) * np.eye(d, dtype=float)
    return out


def _build_env(cfg: dict) -> ETTh1TimeSeriesEnv:
    env_cfg = cfg["environment"]
    csv_path = env_cfg.get("csv_path")
    source = str(env_cfg.get("data_source", "")).lower()
    if source == "jena" and not pathlib.Path(csv_path).exists():
        csv_path = ensure_jena_climate_csv(csv_path)
    if source in ("melbourne", "merlbourne") and not pathlib.Path(csv_path).exists():
        csv_path = ensure_daily_temp_csv(csv_path)
    return ETTh1TimeSeriesEnv(
        csv_path=csv_path,
        target_column=env_cfg.get("target_column"),
        num_experts=int(env_cfg["num_experts"]),
        num_regimes=int(env_cfg["num_regimes"]),
        T=env_cfg.get("T", None),
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
        arima_diff_order=env_cfg.get("arima_diff_order", 1),
        nn_expert_device=env_cfg.get("nn_expert_device", None),
        nn_expert_deterministic=env_cfg.get("nn_expert_deterministic", None),
    )


def _estimate_residual_mse(env: ETTh1TimeSeriesEnv, t_end: int, available_only: bool):
    t_end = max(1, min(int(t_end), env.T - 1))
    sums = np.zeros(env.num_experts, dtype=float)
    counts = np.zeros(env.num_experts, dtype=float)
    for t in range(1, t_end + 1):
        x_t = env.get_context(t)
        preds = np.asarray(env.all_expert_predictions(x_t), dtype=float)
        y_t = float(env.y[t])
        resid = preds - y_t
        avail = env.get_available_experts(t) if available_only else np.arange(env.num_experts, dtype=int)
        for k in avail:
            r = float(resid[int(k)])
            if np.isfinite(r):
                sums[int(k)] += r * r
                counts[int(k)] += 1.0
    return sums, counts


def _estimate_phi_norm2(env: ETTh1TimeSeriesEnv, t_end: int) -> float:
    t_end = max(1, min(int(t_end), env.T - 1))
    total = 0.0
    count = 0
    for t in range(1, t_end + 1):
        phi = np.asarray(feature_phi(env.get_context(t)), dtype=float).reshape(-1)
        if phi.size == 0:
            continue
        total += float(phi @ phi)
        count += 1
    return max(total / max(count, 1), 1e-8)


def _apply_factorized_init(env: ETTh1TimeSeriesEnv, router: FactorizedSLDS, factor_cfg: dict) -> None:
    init_mode = str(factor_cfg.get("init_R_mode", "none")).lower()
    if init_mode not in ("none", "off", "false"):
        window = int(factor_cfg.get("init_R_window", min(2000, env.T - 1)))
        available_only = bool(factor_cfg.get("init_R_use_available", True))
        floor = float(factor_cfg.get("init_R_floor", 1e-4))
        sums, counts = _estimate_residual_mse(env, window, available_only)
        counts_safe = np.maximum(counts, 1.0)
        mse = sums / counts_safe
        global_mse = float(np.sum(sums) / max(np.sum(counts), 1.0))
        mse[counts <= 0] = global_mse
        mse = np.maximum(mse, floor)
        arr = np.asarray(mse, dtype=float).reshape(1, -1)
        router.R = np.repeat(arr, router.M, axis=0)
        router.R_mode = "scalar"

    mode = str(factor_cfg.get("init_state_from_residuals", "none")).lower()
    if mode in ("none", "off", "false"):
        router.reset_beliefs()
        return
    window = int(factor_cfg.get("init_state_window", factor_cfg.get("init_R_window", min(2000, env.T - 1))))
    phi_norm2 = _estimate_phi_norm2(env, window)
    r_mean = float(np.mean(router.R)) if np.ndim(router.R) > 0 else float(router.R)
    u_scale = float(factor_cfg.get("init_state_u_scale", 1.0))
    g_scale = float(factor_cfg.get("init_state_g_scale", 0.3))
    if router.d_phi > 0 and mode in ("u", "g_and_u", "ug", "gu"):
        u_var = max(u_scale * r_mean / phi_norm2, 1e-8)
        router.u_cov0 = np.stack([np.eye(router.d_phi, dtype=float) * u_var for _ in range(router.M)])
    if router.d_g > 0 and mode in ("g", "g_and_u", "ug", "gu"):
        g_var = max(g_scale * r_mean / phi_norm2, 1e-8)
        router.g_cov0 = np.stack([np.eye(router.d_g, dtype=float) * g_var for _ in range(router.M)])
    router.reset_beliefs()


def _build_factorized_router(cfg: dict, env: ETTh1TimeSeriesEnv) -> FactorizedSLDS:
    env_cfg = cfg["environment"]
    factor_cfg = cfg["routers"]["factorized_slds"]
    n = int(env_cfg["num_experts"])
    m = int(factor_cfg["num_regimes"])
    d_phi = int(factor_cfg["idiosyncratic_dim"])
    d_g = int(factor_cfg["shared_dim"])
    beta = np.zeros(n, dtype=float)
    router = FactorizedSLDS(
        M=m,
        d_g=d_g,
        d_phi=d_phi,
        A_g=np.asarray(factor_cfg["A_g"], dtype=float),
        A_u=_diag_stack(factor_cfg["A_u_scale"], d_phi),
        Q_g=_diag_stack(factor_cfg["Q_g_scales"], d_g),
        Q_u=_diag_stack(factor_cfg["Q_u_scales"], d_phi),
        feature_fn=feature_phi,
        beta=beta,
        Delta_max=int(factor_cfg["delta_max"]),
        R=float(factor_cfg.get("R_scalar", 1.0)),
        R_mode="scalar",
        num_experts=n,
        B_intercept_load=float(factor_cfg.get("B_intercept_load", 1.0)),
        attn_dim=int(factor_cfg.get("attn_dim", d_phi)),
        eps=float(factor_cfg.get("eps", 1e-8)),
        exploration=str(factor_cfg.get("exploration", ["g"])[0]),
        exploration_mc_samples=int(factor_cfg.get("exploration_mc_samples", 25)),
        exploration_ucb_samples=int(factor_cfg.get("exploration_ucb_samples", 200)),
        exploration_ucb_alpha=factor_cfg.get("exploration_ucb_alpha", None),
        exploration_ucb_schedule=str(factor_cfg.get("exploration_ucb_schedule", "inverse_t")),
        exploration_sampling_deterministic=bool(factor_cfg.get("exploration_sampling_deterministic", False)),
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
    _apply_factorized_init(env, router, factor_cfg)
    return router


def _run_simple_bandit_on_env(baseline, env: ETTh1TimeSeriesEnv, t_start: int = 1):
    t_start = max(1, int(t_start))
    costs = np.full(env.T - 1, np.nan, dtype=float)
    choices = np.full(env.T - 1, -1, dtype=int)
    for t in range(t_start, env.T):
        x_t = env.get_context(t)
        available = np.asarray(env.get_available_experts(t), dtype=int)
        r_t = baseline.select_expert(x_t, available)
        loss_all = env.losses(t)
        cost_t = float(loss_all[r_t]) + float(baseline.beta[r_t])
        loss_masked = np.full(loss_all.shape, np.nan, dtype=float)
        if getattr(baseline, "feedback_mode", "partial") == "full":
            loss_masked[available] = loss_all[available]
        else:
            loss_masked[r_t] = loss_all[r_t]
        costs[t - 1] = cost_t
        choices[t - 1] = int(r_t)
        baseline.update(x_t, loss_masked, available, selected_expert=r_t)
    return costs, choices


def _time_run(fn, *args, **kwargs):
    start = time.perf_counter()
    costs, choices = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    valid = np.isfinite(costs)
    denom = max(int(np.sum(valid)), 1)
    return {
        "mean_cost": float(np.nanmean(costs)),
        "cum_cost": float(np.nansum(costs)),
        "runtime_ms_per_step": 1000.0 * elapsed / denom,
        "choices": choices,
    }


def _build_baselines(cfg: dict, env: ETTh1TimeSeriesEnv, method_filter: str = "all") -> dict[str, object]:
    env_cfg = cfg["environment"]
    base_cfg = cfg["baselines"]
    n = int(env_cfg["num_experts"])
    d = int(env_cfg["state_dim"])
    beta = np.zeros(n, dtype=float)
    seed = int(env_cfg.get("seed", 42))
    neural_cfg = base_cfg["neural_ucb"]
    dlin_cfg = base_cfg.get("d_linucb", {})
    cusum_cfg = base_cfg.get("cusum_linucb", {})
    glr_cfg = base_cfg.get("glr_linucb", {})
    sw_neural_cfg = base_cfg.get("sw_neural_ucb", {})

    methods: dict[str, object] = {}
    if method_filter in ("all", "neuralucb"):
        methods["NeuralUCB"] = NeuralUCB(
            num_experts=n,
            feature_fn=feature_phi,
            alpha_ucb=float(neural_cfg.get("alpha_ucb", 5.0)),
            lambda_reg=float(neural_cfg.get("lambda_reg", 1.0)),
            beta=beta,
            hidden_dim=int(neural_cfg.get("hidden_dim", 16)),
            nn_learning_rate=float(neural_cfg.get("nn_learning_rate", 1e-3)),
            feedback_mode=str(neural_cfg.get("feedback_mode", "partial")),
            seed=seed,
            context_dim=d,
        )
    if method_filter in ("all", "sw-neuralucb"):
        methods["SW-NeuralUCB"] = SlidingWindowNeuralUCB(
            num_experts=n,
            feature_fn=feature_phi,
            alpha_ucb=float(sw_neural_cfg.get("alpha_ucb", neural_cfg.get("alpha_ucb", 5.0))),
            lambda_reg=float(sw_neural_cfg.get("lambda_reg", neural_cfg.get("lambda_reg", 1.0))),
            beta=beta,
            hidden_dim=int(sw_neural_cfg.get("hidden_dim", neural_cfg.get("hidden_dim", 16))),
            nn_learning_rate=float(sw_neural_cfg.get("nn_learning_rate", neural_cfg.get("nn_learning_rate", 1e-3))),
            feedback_mode=str(sw_neural_cfg.get("feedback_mode", neural_cfg.get("feedback_mode", "partial"))),
            seed=seed,
            context_dim=d,
            window_size=int(sw_neural_cfg.get("window_size", 90)),
        )
    if method_filter in ("all", "d-linucb"):
        methods["D-LinUCB"] = DiscountedLinUCB(
            num_experts=n,
            feature_fn=feature_phi,
            discount_gamma=float(dlin_cfg.get("discount_gamma", 0.995)),
            lambda_reg=float(dlin_cfg.get("lambda_reg", 1.0)),
            beta=beta,
            feedback_mode=str(dlin_cfg.get("feedback_mode", "partial")),
            context_dim=d,
            param_norm_bound=float(dlin_cfg.get("param_norm_bound", 1.0)),
            noise_std=float(dlin_cfg.get("noise_std", 1.0)),
            delta_confidence=float(dlin_cfg.get("delta_confidence", 0.05)),
            seed=seed,
        )
    if method_filter in ("all", "cusum-linucb"):
        methods["CUSUM-LinUCB"] = CUSUMLinUCB(
            num_experts=n,
            feature_fn=feature_phi,
            alpha_ucb=float(cusum_cfg.get("alpha_ucb", base_cfg.get("linucb", {}).get("alpha_ucb", 5.0))),
            lambda_reg=float(cusum_cfg.get("lambda_reg", base_cfg.get("linucb", {}).get("l2_reg", 1.0))),
            beta=beta,
            feedback_mode=str(cusum_cfg.get("feedback_mode", "partial")),
            context_dim=d,
            detector_warmup=int(cusum_cfg.get("detector_warmup", 25)),
            detector_epsilon=float(cusum_cfg.get("detector_epsilon", 0.02)),
            detector_threshold=float(cusum_cfg.get("detector_threshold", 0.25)),
            detector_scale=float(cusum_cfg.get("detector_scale", 1.0)),
            random_explore_prob=float(cusum_cfg.get("random_explore_prob", 0.05)),
            seed=seed,
        )
    if method_filter in ("all", "glr-linucb"):
        methods["GLR-LinUCB"] = GLRLinUCB(
            num_experts=n,
            feature_fn=feature_phi,
            alpha_ucb=float(glr_cfg.get("alpha_ucb", base_cfg.get("linucb", {}).get("alpha_ucb", 5.0))),
            lambda_reg=float(glr_cfg.get("lambda_reg", base_cfg.get("linucb", {}).get("l2_reg", 1.0))),
            beta=beta,
            feedback_mode=str(glr_cfg.get("feedback_mode", "partial")),
            context_dim=d,
            detector_delta=float(glr_cfg.get("detector_delta", 0.05)),
            detector_min_window=int(glr_cfg.get("detector_min_window", 20)),
            detector_scale=float(glr_cfg.get("detector_scale", 1.0)),
            forced_exploration_prob=float(glr_cfg.get("forced_exploration_prob", 0.1)),
            seed=seed,
        )
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--method", type=str, default="all")
    parser.add_argument("--dlin-gamma", type=float, default=None)
    parser.add_argument("--dlin-lambda", type=float, default=None)
    parser.add_argument("--sw-neural-window", type=int, default=None)
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = _load_cfg(cfg_path)
    env = _build_env(cfg)
    method = str(args.method).lower()
    router = _build_factorized_router(cfg, env) if method in ("all", "ours", "l2d-slds") else None
    baselines = _build_baselines(cfg, env, method_filter=method)
    if args.dlin_gamma is not None:
        dlin = baselines.get("D-LinUCB")
        if dlin is not None:
            dlin.discount_gamma = float(args.dlin_gamma)
            dlin.reset_state()
    if args.dlin_lambda is not None:
        dlin = baselines.get("D-LinUCB")
        if dlin is not None:
            dlin.lambda_reg = float(args.dlin_lambda)
            dlin.reset_state()
    if args.sw_neural_window is not None:
        sw_neural = baselines.get("SW-NeuralUCB")
        if sw_neural is not None:
            sw_neural.window_size = int(args.sw_neural_window)
            sw_neural.reset_state()
    t_start = 1

    results: list[dict[str, float | str]] = []

    if method in ("all", "ours", "l2d-slds"):
        ours = _time_run(run_factored_router_on_env, router, env, t_start)
        results.append({
            "method": "L2D-SLDS",
            "mean_cost": ours["mean_cost"],
            "cum_cost": ours["cum_cost"],
            "runtime_ms_per_step": ours["runtime_ms_per_step"],
        })

    for name, model in baselines.items():
        if method not in ("all", name.lower()):
            continue
        runner = run_neuralucb_on_env if name in ("NeuralUCB", "SW-NeuralUCB") else _run_simple_bandit_on_env
        metrics = _time_run(runner, model, env, t_start)
        results.append({
            "method": name,
            "mean_cost": metrics["mean_cost"],
            "cum_cost": metrics["cum_cost"],
            "runtime_ms_per_step": metrics["runtime_ms_per_step"],
        })

    out_path = pathlib.Path(args.out) if args.out is not None else ROOT / "out" / "paper_testing_baselines.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("method,mean_cost,cum_cost,runtime_ms_per_step\n")
        for row in results:
            f.write(
                f"{row['method']},{float(row['mean_cost']):.6f},{float(row['cum_cost']):.6f},{float(row['runtime_ms_per_step']):.6f}\n"
            )

    print(f"Config: {cfg_path}")
    for row in results:
        print(
            f"{row['method']}: mean_cost={float(row['mean_cost']):.6f}, "
            f"cum_cost={float(row['cum_cost']):.2f}, "
            f"runtime={float(row['runtime_ms_per_step']):.3f} ms/step"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
