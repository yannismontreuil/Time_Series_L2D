import argparse
import copy
import csv
import pathlib
import sys
from typing import Any

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.synthetic_env import SyntheticTimeSeriesEnv
from models.factorized_slds import FactorizedSLDS
from models.router_model import feature_phi
from router_eval import run_factored_router_on_env
from slds_imm_router import _clone_factorized_em_data, _collect_factorized_em_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic misspecification-vs-EM ablation on the paper's tri_cycle_corr "
            "setting. The experiment fits EM on a warmup prefix and reports only the "
            "post-prefix horizon."
        )
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOT / "config" / "exp_synthetic_1.yaml",
        help="Base synthetic config from the paper.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 12, 13, 14, 15],
        help="Learning seeds to evaluate.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=ROOT / "out" / "synthetic_em_ablation",
        help="Directory for CSV summaries.",
    )
    parser.add_argument(
        "--em-iters",
        type=int,
        default=None,
        help="Optional override for the number of offline EM iterations.",
    )
    parser.add_argument(
        "--em-samples",
        type=int,
        default=None,
        help="Optional override for the number of posterior samples per EM iteration.",
    )
    parser.add_argument(
        "--em-burn-in",
        type=int,
        default=None,
        help="Optional override for the burn-in used inside EM.",
    )
    parser.add_argument(
        "--em-tk",
        type=int,
        default=None,
        help="Optional override for the offline EM warmup prefix length.",
    )
    parser.add_argument(
        "--em-use-validation",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Optional override for EM validation usage.",
    )
    parser.add_argument(
        "--em-val-strategy",
        type=str,
        default=None,
        choices=["tail", "rolling"],
        help="Optional override for the EM validation split strategy.",
    )
    parser.add_argument(
        "--misspec-profile",
        type=str,
        default="strong",
        choices=["mild", "strong"],
        help="Misspecification profile for dynamics/noise initialization.",
    )
    parser.add_argument(
        "--print-per-seed",
        action="store_true",
        help="Print per-seed summaries in addition to the aggregate summary.",
    )
    return parser.parse_args()


def _load_cfg(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a mapping.")
    return cfg


def _diag_stack_from_list(values: list[float], d: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = np.zeros((arr.shape[0], d, d), dtype=float)
    eye = np.eye(d, dtype=float)
    for m, val in enumerate(arr):
        out[m] = float(val) * eye
    return out


def _diag_stack_from_scalar_or_list(value: Any, d: int, M: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        scalars = np.full(M, float(arr), dtype=float)
    else:
        scalars = arr.reshape(-1)
        if scalars.size == 1:
            scalars = np.full(M, float(scalars[0]), dtype=float)
        elif scalars.size != M:
            raise ValueError(f"Expected {M} scalars, got {scalars.size}.")
    return _diag_stack_from_list(scalars.tolist(), d)


def _build_env(cfg: dict[str, Any]) -> SyntheticTimeSeriesEnv:
    env_cfg = copy.deepcopy(cfg["environment"])
    return SyntheticTimeSeriesEnv(
        num_experts=int(env_cfg["num_experts"]),
        num_regimes=int(env_cfg["num_regimes"]),
        T=int(env_cfg["T"]),
        seed=int(env_cfg.get("seed", 0)),
        data_seed=int(env_cfg.get("data_seed", 42)),
        unavailable_expert_idx=env_cfg.get("unavailable_expert_idx", None),
        unavailable_start_t=env_cfg.get("unavailable_start_t", None),
        unavailable_intervals=env_cfg.get("unavailable_intervals", None),
        arrival_expert_idx=env_cfg.get("arrival_expert_idx", None),
        arrival_intervals=env_cfg.get("arrival_intervals", None),
        setting=str(env_cfg.get("setting", "easy_setting")),
        noise_scale=env_cfg.get("noise_scale", None),
        tri_cycle_cfg=env_cfg.get("tri_cycle", None),
    )


def _build_factorized_router(cfg: dict[str, Any], env: SyntheticTimeSeriesEnv, seed: int) -> FactorizedSLDS:
    factor_cfg = cfg["routers"]["factorized_slds"]
    env_cfg = cfg["environment"]
    tri_cycle_cfg = env_cfg.get("tri_cycle", {})
    M = int(factor_cfg["num_regimes"])
    d_g = int(factor_cfg["shared_dim"])
    d_phi = int(factor_cfg["idiosyncratic_dim"])
    N = int(env.num_experts)

    loadings = tri_cycle_cfg.get("shared_loadings", None)
    B_dict = None
    if loadings is not None and d_g > 0:
        load_arr = np.asarray(loadings, dtype=float)
        if load_arr.shape == (N, d_g):
            B_dict = {
                int(k): load_arr[int(k)].reshape(d_phi, d_g)
                for k in range(N)
            }

    beta = np.zeros(N, dtype=float)
    A_g = None if d_g <= 0 else np.asarray(factor_cfg["A_g"], dtype=float)
    Q_g = None
    if d_g > 0:
        if "Q_g" in factor_cfg:
            Q_g = np.asarray(factor_cfg["Q_g"], dtype=float)
        else:
            Q_g = _diag_stack_from_list(factor_cfg["Q_g_scales"], d_g)
    if "A_u" in factor_cfg:
        A_u = np.asarray(factor_cfg["A_u"], dtype=float)
    else:
        A_u = _diag_stack_from_scalar_or_list(factor_cfg["A_u_scale"], d_phi, M)
    Q_u = _diag_stack_from_list(factor_cfg["Q_u_scales"], d_phi)

    router = FactorizedSLDS(
        M=M,
        d_g=d_g,
        d_phi=d_phi,
        feature_fn=feature_phi,
        B_dict=B_dict,
        beta=beta,
        Delta_max=int(factor_cfg["delta_max"]),
        R=float(factor_cfg.get("R_scalar", 1.0)),
        R_mode="scalar",
        num_experts=N,
        B_intercept_load=float(factor_cfg.get("B_intercept_load", 1.0)),
        attn_dim=int(factor_cfg.get("attn_dim", d_phi)),
        g_mean0=None,
        g_cov0=None,
        u_mean0=None,
        u_cov0=None,
        A_g=A_g,
        A_u=A_u,
        Q_g=Q_g,
        Q_u=Q_u,
        eps=float(factor_cfg.get("eps", 1e-8)),
        exploration=str(factor_cfg.get("exploration", ["g_z"])[0]),
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
        transition_hidden_dims=factor_cfg.get("transition_hidden_dims", None),
        transition_activation=str(factor_cfg.get("transition_activation", "relu")),
        transition_device=str(factor_cfg.get("transition_device", "cpu")),
        transition_init=str(factor_cfg.get("transition_init", "uniform")),
        transition_mode=str(factor_cfg.get("transition_mode", "attention")),
        feedback_mode=str(factor_cfg.get("feedback_mode", "partial")),
        seed=int(seed),
    )
    router.reset_beliefs()
    return router


def _apply_misspecification(router: FactorizedSLDS, profile: str) -> None:
    if profile == "mild":
        if router.d_g > 0:
            router.A_g[:] = np.eye(router.d_g, dtype=float)[None, :, :] * 0.80
            router.Q_g[:] = np.eye(router.d_g, dtype=float)[None, :, :] * 0.08
        router.A_u[:] = np.eye(router.d_phi, dtype=float)[None, :, :] * 0.80
        router.Q_u[:] = np.eye(router.d_phi, dtype=float)[None, :, :] * 0.04
        router.R = 0.01
    elif profile == "strong":
        if router.d_g > 0:
            router.A_g[:] = np.eye(router.d_g, dtype=float)[None, :, :] * 0.55
            router.Q_g[:] = np.eye(router.d_g, dtype=float)[None, :, :] * 0.12
        router.A_u[:] = np.eye(router.d_phi, dtype=float)[None, :, :] * 0.60
        router.Q_u[:] = np.eye(router.d_phi, dtype=float)[None, :, :] * 0.06
        router.R = 0.02
    else:
        raise ValueError(f"Unknown misspecification profile: {profile}")

    router._A_g_diag = None
    router._A_u_diag = None
    router._A_g_outer = None
    router._A_u_outer = None
    if router.d_g > 0:
        A_g_diag = np.diagonal(router.A_g, axis1=1, axis2=2)
        if np.allclose(router.A_g, np.eye(router.d_g)[None, :, :] * A_g_diag[:, None, :]):
            router._A_g_diag = A_g_diag.copy()
            router._A_g_outer = router._A_g_diag[:, :, None] * router._A_g_diag[:, None, :]
    A_u_diag = np.diagonal(router.A_u, axis1=1, axis2=2)
    if np.allclose(router.A_u, np.eye(router.d_phi)[None, :, :] * A_u_diag[:, None, :]):
        router._A_u_diag = A_u_diag.copy()
        router._A_u_outer = router._A_u_diag[:, :, None] * router._A_u_diag[:, None, :]


def _true_params_from_cfg(cfg: dict[str, Any], env: SyntheticTimeSeriesEnv) -> dict[str, Any]:
    factor_cfg = cfg["routers"]["factorized_slds"]
    env_cfg = cfg["environment"]
    tri_cycle_cfg = env_cfg.get("tri_cycle", {})
    M = int(factor_cfg["num_regimes"])
    d_g = int(factor_cfg["shared_dim"])
    d_phi = int(factor_cfg["idiosyncratic_dim"])
    N = int(env.num_experts)

    if "A_u" in factor_cfg:
        A_u = np.asarray(factor_cfg["A_u"], dtype=float)
    else:
        A_u = _diag_stack_from_scalar_or_list(factor_cfg["A_u_scale"], d_phi, M)
    if d_g > 0:
        if "Q_g" in factor_cfg:
            Q_g = np.asarray(factor_cfg["Q_g"], dtype=float)
        else:
            Q_g = _diag_stack_from_list(factor_cfg["Q_g_scales"], d_g)
        A_g = np.asarray(factor_cfg["A_g"], dtype=float)
    else:
        A_g = None
        Q_g = None
    Q_u = _diag_stack_from_list(factor_cfg["Q_u_scales"], d_phi)

    loadings = tri_cycle_cfg.get("shared_loadings", None)
    B_dict = {}
    if loadings is not None and d_g > 0:
        load_arr = np.asarray(loadings, dtype=float)
        if load_arr.shape == (N, d_g):
            B_dict = {
                int(k): load_arr[int(k)].reshape(d_phi, d_g)
                for k in range(N)
            }

    return {
        "A_g": A_g,
        "Q_g": Q_g,
        "A_u": A_u,
        "Q_u": Q_u,
        "R": float(factor_cfg.get("R_scalar", 1.0)),
        "B_dict": B_dict,
    }


def _param_rmse(params: dict[str, Any], truth: dict[str, Any]) -> float:
    diffs: list[float] = []
    for key in ("A_g", "Q_g", "A_u", "Q_u"):
        p = params.get(key, None)
        q = truth.get(key, None)
        if p is None or q is None:
            continue
        arr_p = np.asarray(p, dtype=float)
        arr_q = np.asarray(q, dtype=float)
        diffs.append(float(np.mean((arr_p - arr_q) ** 2)))
    if "R" in params and "R" in truth:
        r_p = np.asarray(params["R"], dtype=float)
        r_q = np.asarray(truth["R"], dtype=float)
        if r_p.shape == () and r_q.shape == ():
            diffs.append((float(r_p) - float(r_q)) ** 2)
        else:
            if r_q.shape == ():
                r_q = np.full(r_p.shape, float(r_q), dtype=float)
            diffs.append(float(np.mean((r_p - r_q) ** 2)))
    if "B_dict" in params and "B_dict" in truth and truth["B_dict"]:
        vals = []
        for k, q in truth["B_dict"].items():
            if k not in params["B_dict"]:
                continue
            p = np.asarray(params["B_dict"][k], dtype=float)
            vals.append(float(np.mean((p - np.asarray(q, dtype=float)) ** 2)))
        if vals:
            diffs.append(float(np.mean(vals)))
    if not diffs:
        return 0.0
    return float(np.sqrt(np.mean(diffs)))


def _masked_summary(costs: np.ndarray, em_tk: int) -> tuple[float, float]:
    tail = np.asarray(costs, dtype=float)
    if em_tk > 0:
        tail = tail[int(em_tk):]
    return float(np.nanmean(tail)), float(np.nansum(tail))


def _fit_em_from_prefix(
    router: FactorizedSLDS,
    env: SyntheticTimeSeriesEnv,
    cfg: dict[str, Any],
    em_data_cache: dict[tuple[int, bool], tuple],
    em_seed: int,
) -> None:
    factor_cfg = cfg["routers"]["factorized_slds"]
    em_tk = int(factor_cfg.get("em_tk", min(500, env.T - 1)))
    force_full = bool(factor_cfg.get("em_offline_full_feedback", True))
    cache_key = (em_tk, force_full)
    em_data = em_data_cache.get(cache_key)
    if em_data is None:
        em_data = _collect_factorized_em_data(router, env, em_tk, force_full_feedback=force_full)
        em_data_cache[cache_key] = em_data
    ctx_em, avail_em, actions_em, resid_em, resid_full_em = _clone_factorized_em_data(em_data)

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
            router._rng.bit_generator.state = np.random.default_rng(em_seed).bit_generator.state
            router._init_transition_params(context_dim)
            router._rng.bit_generator.state = prev_rng_state

    prev_feedback_mode = router.feedback_mode
    router.feedback_mode = "full" if force_full else prev_feedback_mode
    try:
        router.fit_em(
            contexts=ctx_em,
            available_sets=avail_em,
            actions=actions_em,
            residuals=resid_em,
            residuals_full=resid_full_em if router.feedback_mode == "full" else None,
            n_em=int(factor_cfg.get("em_n", 5)),
            n_samples=int(factor_cfg.get("em_samples", 10)),
            burn_in=int(factor_cfg.get("em_burn_in", 5)),
            val_fraction=float(factor_cfg.get("em_val_fraction", 0.0)),
            val_len=factor_cfg.get("em_val_len", None),
            val_strategy=str(factor_cfg.get("em_val_strategy", "tail")),
            val_roll_splits=factor_cfg.get("em_val_roll_splits", None),
            val_roll_len=factor_cfg.get("em_val_roll_len", None),
            val_roll_stride=factor_cfg.get("em_val_roll_stride", None),
            theta_lr=float(factor_cfg.get("em_theta_lr", 1e-2)),
            theta_steps=int(factor_cfg.get("em_theta_steps", 1)),
            seed=int(em_seed),
            priors=factor_cfg.get("em_priors", None),
            use_validation=bool(factor_cfg.get("em_use_validation", False)),
            print_val_loss=bool(factor_cfg.get("em_print_val_loss", True)),
            check_finite=bool(factor_cfg.get("em_check_finite", True)),
            warn_nll_increase=bool(factor_cfg.get("em_warn_nll_increase", True)),
            nll_increase_tol=float(factor_cfg.get("em_nll_increase_tol", 1e-6)),
            sanitize_cov=bool(factor_cfg.get("em_sanitize_cov", True)),
            epsilon_N=float(factor_cfg.get("em_eps_n", 0.0)),
        )
    finally:
        router.feedback_mode = prev_feedback_mode
    router.reset_beliefs()


def _maybe_override_em_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    factor_cfg = cfg["routers"]["factorized_slds"]
    factor_cfg["em_enabled"] = True
    if args.em_iters is not None:
        factor_cfg["em_n"] = int(args.em_iters)
    if args.em_samples is not None:
        factor_cfg["em_samples"] = int(args.em_samples)
    if args.em_burn_in is not None:
        factor_cfg["em_burn_in"] = int(args.em_burn_in)
    if args.em_tk is not None:
        factor_cfg["em_tk"] = int(args.em_tk)
    if args.em_use_validation is not None:
        factor_cfg["em_use_validation"] = bool(str(args.em_use_validation).lower() == "true")
    if args.em_val_strategy is not None:
        factor_cfg["em_val_strategy"] = str(args.em_val_strategy)


def _evaluate_case(
    cfg: dict[str, Any],
    env: SyntheticTimeSeriesEnv,
    seed: int,
    truth: dict[str, Any],
    em_data_cache: dict[tuple[int, bool], tuple],
    misspec_profile: str,
) -> list[dict[str, Any]]:
    factor_cfg = cfg["routers"]["factorized_slds"]
    em_tk = int(factor_cfg.get("em_tk", 0))
    rows: list[dict[str, Any]] = []

    matched = _build_factorized_router(cfg, env, seed)
    matched_params = matched._snapshot_params()
    matched_costs, _ = run_factored_router_on_env(matched, env)
    matched_avg, matched_cum = _masked_summary(matched_costs, em_tk)
    rows.append(
        {
            "seed": seed,
            "variant": "matched_no_em",
            "avg_cost_tail": matched_avg,
            "cum_cost_tail": matched_cum,
            "param_rmse_before": _param_rmse(matched_params, truth),
            "param_rmse_after": _param_rmse(matched_params, truth),
        }
    )

    miss_no_em = _build_factorized_router(cfg, env, seed)
    _apply_misspecification(miss_no_em, misspec_profile)
    miss_before = miss_no_em._snapshot_params()
    miss_costs, _ = run_factored_router_on_env(miss_no_em, env)
    miss_avg, miss_cum = _masked_summary(miss_costs, em_tk)
    rows.append(
        {
            "seed": seed,
            "variant": "misspecified_no_em",
            "avg_cost_tail": miss_avg,
            "cum_cost_tail": miss_cum,
            "param_rmse_before": _param_rmse(miss_before, truth),
            "param_rmse_after": _param_rmse(miss_before, truth),
        }
    )

    miss_em = _build_factorized_router(cfg, env, seed)
    _apply_misspecification(miss_em, misspec_profile)
    em_before = miss_em._snapshot_params()
    _fit_em_from_prefix(miss_em, env, cfg, em_data_cache, em_seed=seed)
    em_after = miss_em._snapshot_params()
    em_costs, _ = run_factored_router_on_env(miss_em, env)
    em_avg, em_cum = _masked_summary(em_costs, em_tk)
    rows.append(
        {
            "seed": seed,
            "variant": "misspecified_plus_em",
            "avg_cost_tail": em_avg,
            "cum_cost_tail": em_cum,
            "param_rmse_before": _param_rmse(em_before, truth),
            "param_rmse_after": _param_rmse(em_after, truth),
        }
    )
    return rows


def _write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "variant",
        "avg_cost_tail",
        "cum_cost_tail",
        "param_rmse_before",
        "param_rmse_after",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    out = []
    for variant, items in grouped.items():
        avg = np.asarray([float(x["avg_cost_tail"]) for x in items], dtype=float)
        cum = np.asarray([float(x["cum_cost_tail"]) for x in items], dtype=float)
        before = np.asarray([float(x["param_rmse_before"]) for x in items], dtype=float)
        after = np.asarray([float(x["param_rmse_after"]) for x in items], dtype=float)
        out.append(
            {
                "variant": variant,
                "avg_cost_tail_mean": float(np.mean(avg)),
                "avg_cost_tail_se": float(np.std(avg, ddof=1) / np.sqrt(avg.size)) if avg.size > 1 else 0.0,
                "cum_cost_tail_mean": float(np.mean(cum)),
                "cum_cost_tail_se": float(np.std(cum, ddof=1) / np.sqrt(cum.size)) if cum.size > 1 else 0.0,
                "param_rmse_before_mean": float(np.mean(before)),
                "param_rmse_after_mean": float(np.mean(after)),
            }
        )
    order = {
        "matched_no_em": 0,
        "misspecified_no_em": 1,
        "misspecified_plus_em": 2,
    }
    out.sort(key=lambda row: order.get(str(row["variant"]), 999))
    return out


def _write_summary(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "avg_cost_tail_mean",
        "avg_cost_tail_se",
        "cum_cost_tail_mean",
        "cum_cost_tail_se",
        "param_rmse_before_mean",
        "param_rmse_after_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config)
    _maybe_override_em_cfg(cfg, args)
    env = _build_env(cfg)
    truth = _true_params_from_cfg(cfg, env)

    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        em_data_cache: dict[tuple[int, bool], tuple] = {}
        seed_rows = _evaluate_case(
            cfg=cfg,
            env=env,
            seed=int(seed),
            truth=truth,
            em_data_cache=em_data_cache,
            misspec_profile=str(args.misspec_profile),
        )
        all_rows.extend(seed_rows)
        if args.print_per_seed:
            for row in seed_rows:
                print(
                    f"seed={row['seed']} variant={row['variant']} "
                    f"avg_tail={row['avg_cost_tail']:.4f} cum_tail={row['cum_cost_tail']:.2f} "
                    f"rmse_before={row['param_rmse_before']:.4f} rmse_after={row['param_rmse_after']:.4f}"
                )

    summary = _aggregate(all_rows)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(out_dir / "synthetic_em_ablation_rows.csv", all_rows)
    _write_summary(out_dir / "synthetic_em_ablation_summary.csv", summary)

    print("\n=== Synthetic EM ablation summary (post-EM horizon only) ===")
    for row in summary:
        print(
            f"{row['variant']}: "
            f"avg={row['avg_cost_tail_mean']:.4f} +/- {row['avg_cost_tail_se']:.4f}, "
            f"cum={row['cum_cost_tail_mean']:.2f} +/- {row['cum_cost_tail_se']:.2f}, "
            f"param_rmse {row['param_rmse_before_mean']:.4f} -> {row['param_rmse_after_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
