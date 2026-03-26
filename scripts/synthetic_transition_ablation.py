import argparse
import copy
import csv
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.synthetic_env import SyntheticTimeSeriesEnv
from models.factorized_slds import FactorizedSLDS
from models.router_model import feature_phi
from slds_imm_router import _clone_factorized_em_data, _collect_factorized_em_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic transition ablation on a context-gated switching environment. "
            "Compares fixed uniform transitions against learned attention transitions "
            "using tail routing cost and direct transition-quality metrics."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=ROOT / "out" / "synthetic_transition_ablation",
    )
    parser.add_argument("--T", type=int, default=3500)
    parser.add_argument("--num-experts", type=int, default=5)
    parser.add_argument("--num-regimes", type=int, default=2)
    parser.add_argument("--shared-dim", type=int, default=1)
    parser.add_argument("--idiosyncratic-dim", type=int, default=2)
    parser.add_argument("--delta-max", type=int, default=250)
    parser.add_argument("--noise-scale", type=float, default=0.12)
    parser.add_argument("--context-ar", type=float, default=0.85)
    parser.add_argument("--context-drift", type=float, default=0.0)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--em-tk", type=int, default=2000)
    parser.add_argument("--em-iters", type=int, default=8)
    parser.add_argument("--em-samples", type=int, default=10)
    parser.add_argument("--em-burn-in", type=int, default=20)
    parser.add_argument("--em-theta-lr", type=float, default=0.02)
    parser.add_argument("--em-theta-steps", type=int, default=30)
    parser.add_argument("--em-use-validation", action="store_true")
    parser.add_argument("--em-val-fraction", type=float, default=0.2)
    parser.add_argument("--em-val-strategy", type=str, default="rolling", choices=["tail", "rolling"])
    parser.add_argument("--em-val-roll-len", type=int, default=250)
    parser.add_argument("--em-val-roll-splits", type=int, default=3)
    parser.add_argument("--transition-mode", type=str, default="attention", choices=["attention", "linear"])
    parser.add_argument("--transition-init", type=str, default="uniform", choices=["uniform", "random"])
    parser.add_argument("--exploration", type=str, default="g_z", choices=["g", "g_z", "ucb", "sampling"])
    parser.add_argument("--mc-samples", type=int, default=12)
    parser.add_argument("--print-per-seed", action="store_true")
    return parser.parse_args()


def _build_env(args: argparse.Namespace) -> SyntheticTimeSeriesEnv:
    return SyntheticTimeSeriesEnv(
        num_experts=int(args.num_experts),
        num_regimes=int(args.num_regimes),
        T=int(args.T),
        seed=0,
        data_seed=42,
        unavailable_expert_idx=None,
        unavailable_start_t=None,
        unavailable_intervals=None,
        arrival_expert_idx=None,
        arrival_intervals=None,
        setting="context_gate_corr",
        noise_scale=float(args.noise_scale),
        tri_cycle_cfg={
            "context_ar": float(args.context_ar),
            "context_drift": float(args.context_drift),
            "gate_threshold": float(args.gate_threshold),
            "Pi_xneg": [[0.97, 0.03], [0.25, 0.75]],
            "Pi_xpos": [[0.75, 0.25], [0.03, 0.97]],
            "shared_noise_scale": 0.10,
            "indiv_noise_scale": 0.04,
            "base_slope": 0.8,
            "biases_by_regime": [
                [0.0, 0.1, 2.2, 2.3, 1.1],
                [2.2, 2.3, 0.0, 0.1, 1.1],
            ],
        },
    )


def _build_router(args: argparse.Namespace, env: SyntheticTimeSeriesEnv, seed: int) -> FactorizedSLDS:
    M = int(args.num_regimes)
    d_g = int(args.shared_dim)
    d_phi = int(args.idiosyncratic_dim)
    A_g = np.stack([np.eye(d_g, dtype=float) * 0.95 for _ in range(M)], axis=0)
    Q_g = np.stack([np.eye(d_g, dtype=float) * 0.05 for _ in range(M)], axis=0)
    A_u = np.stack([np.eye(d_phi, dtype=float) * 0.95 for _ in range(M)], axis=0)
    Q_u = np.stack([np.eye(d_phi, dtype=float) * 0.04 for _ in range(M)], axis=0)

    return FactorizedSLDS(
        M=M,
        d_g=d_g,
        d_phi=d_phi,
        feature_fn=feature_phi,
        B_dict=None,
        beta=np.zeros(env.num_experts, dtype=float),
        Delta_max=int(args.delta_max),
        R=float(0.01),
        R_mode="scalar",
        num_experts=int(env.num_experts),
        B_intercept_load=1.0,
        attn_dim=max(1, d_phi),
        A_g=A_g,
        A_u=A_u,
        Q_g=Q_g,
        Q_u=Q_u,
        eps=1e-8,
        exploration=str(args.exploration),
        exploration_mc_samples=int(args.mc_samples),
        exploration_ucb_samples=200,
        exploration_sampling_deterministic=False,
        exploration_diag_enabled=False,
        observation_mode="residual",
        transition_hidden_dims=None,
        transition_activation="relu",
        transition_device="cpu",
        transition_init=str(args.transition_init),
        transition_mode=str(args.transition_mode),
        feedback_mode="partial",
        seed=int(seed),
    )


def _tail_mask(t: int, t_start: int) -> bool:
    return int(t) >= int(t_start)


def _transition_fields(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "W_q": None if params.get("W_q") is None else params["W_q"].copy(),
        "W_k": None if params.get("W_k") is None else params["W_k"].copy(),
        "b_attn": None if params.get("b_attn") is None else params["b_attn"].copy(),
        "W_lin": None if params.get("W_lin") is None else params["W_lin"].copy(),
        "b_lin": None if params.get("b_lin") is None else params["b_lin"].copy(),
        "transition_model_state": copy.deepcopy(params.get("transition_model_state")),
        "transition_input_dim": params.get("transition_input_dim"),
        "transition_mode": params.get("transition_mode"),
        "context_dim": params.get("context_dim"),
    }


def _inject_transition_fields(base_params: dict[str, Any], learned_params: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_params)
    learned_transition = _transition_fields(learned_params)
    for key, value in learned_transition.items():
        merged[key] = value
    return merged


def _safe_prob(p: float, eps: float = 1e-12) -> float:
    return float(max(float(p), eps))


def _evaluate_router(
    router: FactorizedSLDS,
    env: SyntheticTimeSeriesEnv,
    tail_start: int,
) -> dict[str, Any]:
    router.reset_beliefs()

    cost_tail: list[float] = []
    transition_nll_tail: list[float] = []
    transition_acc_tail: list[float] = []
    transition_frob_tail: list[float] = []
    regime_acc_tail: list[float] = []

    nll_neg_tail: list[float] = []
    nll_pos_tail: list[float] = []
    acc_neg_tail: list[float] = []
    acc_pos_tail: list[float] = []
    frob_neg_tail: list[float] = []
    frob_pos_tail: list[float] = []

    pi_hat_neg_sum = np.zeros((env.num_regimes, env.num_regimes), dtype=float)
    pi_hat_pos_sum = np.zeros((env.num_regimes, env.num_regimes), dtype=float)
    pi_true_neg_sum = np.zeros((env.num_regimes, env.num_regimes), dtype=float)
    pi_true_pos_sum = np.zeros((env.num_regimes, env.num_regimes), dtype=float)
    n_neg = 0
    n_pos = 0

    for t in range(1, env.T):
        x_t = env.get_context(t)
        available = np.asarray(env.get_available_experts(t), dtype=int)
        r_t, cache = router.select_expert(x_t, available)
        preds = np.asarray(env.all_expert_predictions(x_t), dtype=float)
        y_t = float(np.asarray(env.y[t], dtype=float))
        residuals = preds - y_t
        loss_r = float(residuals[int(r_t)] ** 2)

        pi_hat = np.asarray(router._context_transition(x_t), dtype=float)
        pi_true = np.asarray(env.true_transition_matrix(x_t), dtype=float)
        z_prev = int(env.z[t - 1])
        z_cur = int(env.z[t])
        trans_prob = _safe_prob(pi_hat[z_prev, z_cur])
        trans_nll = -float(np.log(trans_prob))
        trans_acc = float(int(np.argmax(pi_hat[z_prev]) == z_cur))
        trans_frob = float(np.linalg.norm(pi_hat - pi_true))

        router.update_beliefs(
            r_t=int(r_t),
            loss_obs=float(residuals[int(r_t)]),
            losses_full=None,
            available_experts=available,
            cache=cache,
        )
        regime_acc = float(int(np.argmax(router.w) == z_cur))

        if _tail_mask(t, tail_start):
            cost_tail.append(loss_r)
            transition_nll_tail.append(trans_nll)
            transition_acc_tail.append(trans_acc)
            transition_frob_tail.append(trans_frob)
            regime_acc_tail.append(regime_acc)

            x_scalar = float(np.asarray(x_t, dtype=float).reshape(-1)[0])
            if x_scalar < float(getattr(env, "_context_gate_threshold", 0.0)):
                n_neg += 1
                nll_neg_tail.append(trans_nll)
                acc_neg_tail.append(trans_acc)
                frob_neg_tail.append(trans_frob)
                pi_hat_neg_sum += pi_hat
                pi_true_neg_sum += pi_true
            else:
                n_pos += 1
                nll_pos_tail.append(trans_nll)
                acc_pos_tail.append(trans_acc)
                frob_pos_tail.append(trans_frob)
                pi_hat_pos_sum += pi_hat
                pi_true_pos_sum += pi_true

    pi_hat_neg_mean = pi_hat_neg_sum / max(n_neg, 1)
    pi_hat_pos_mean = pi_hat_pos_sum / max(n_pos, 1)
    pi_true_neg_mean = pi_true_neg_sum / max(n_neg, 1)
    pi_true_pos_mean = pi_true_pos_sum / max(n_pos, 1)

    context_gap_hat = float(np.linalg.norm(pi_hat_pos_mean - pi_hat_neg_mean))
    context_gap_true = float(np.linalg.norm(pi_true_pos_mean - pi_true_neg_mean))

    return {
        "avg_cost_tail": float(np.mean(cost_tail)),
        "cum_cost_tail": float(np.sum(cost_tail)),
        "transition_nll_tail": float(np.mean(transition_nll_tail)),
        "transition_acc_tail": float(np.mean(transition_acc_tail)),
        "transition_frob_tail": float(np.mean(transition_frob_tail)),
        "regime_acc_tail": float(np.mean(regime_acc_tail)),
        "transition_nll_neg_tail": float(np.mean(nll_neg_tail)) if n_neg > 0 else float("nan"),
        "transition_nll_pos_tail": float(np.mean(nll_pos_tail)) if n_pos > 0 else float("nan"),
        "transition_acc_neg_tail": float(np.mean(acc_neg_tail)) if n_neg > 0 else float("nan"),
        "transition_acc_pos_tail": float(np.mean(acc_pos_tail)) if n_pos > 0 else float("nan"),
        "transition_frob_neg_tail": float(np.mean(frob_neg_tail)) if n_neg > 0 else float("nan"),
        "transition_frob_pos_tail": float(np.mean(frob_pos_tail)) if n_pos > 0 else float("nan"),
        "context_gap_hat": context_gap_hat,
        "context_gap_true": context_gap_true,
        "context_gap_recovery": float(context_gap_hat / max(context_gap_true, 1e-12)),
        "tail_steps": int(len(cost_tail)),
        "neg_steps_tail": int(n_neg),
        "pos_steps_tail": int(n_pos),
        "pi_hat_neg_mean": pi_hat_neg_mean.tolist(),
        "pi_hat_pos_mean": pi_hat_pos_mean.tolist(),
        "pi_true_neg_mean": pi_true_neg_mean.tolist(),
        "pi_true_pos_mean": pi_true_pos_mean.tolist(),
    }


def _fit_transition_only(
    router: FactorizedSLDS,
    env: SyntheticTimeSeriesEnv,
    args: argparse.Namespace,
) -> None:
    base_params = router._snapshot_params()
    warmup_data = _collect_factorized_em_data(
        router,
        env,
        t_end=int(args.em_tk),
        force_full_feedback=True,
    )
    ctx, avail, actions, residuals, residuals_full = _clone_factorized_em_data(warmup_data)
    router.fit_em(
        contexts=ctx,
        available_sets=avail,
        actions=actions,
        residuals=residuals,
        residuals_full=residuals_full,
        n_em=int(args.em_iters),
        n_samples=int(args.em_samples),
        burn_in=int(args.em_burn_in),
        val_fraction=float(args.em_val_fraction),
        val_strategy=str(args.em_val_strategy),
        val_roll_len=int(args.em_val_roll_len),
        val_roll_splits=int(args.em_val_roll_splits),
        theta_lr=float(args.em_theta_lr),
        theta_steps=int(args.em_theta_steps),
        seed=0,
        print_val_loss=False,
        epsilon_N=1e-3,
        use_validation=bool(args.em_use_validation),
        transition_log_stage="post_em_transition_ablation",
    )
    learned_params = router._snapshot_params()
    transition_only = _inject_transition_fields(base_params, learned_params)
    router._restore_params(transition_only)
    router.reset_beliefs()


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "avg_cost_tail",
        "cum_cost_tail",
        "transition_nll_tail",
        "transition_acc_tail",
        "transition_frob_tail",
        "regime_acc_tail",
        "transition_nll_neg_tail",
        "transition_nll_pos_tail",
        "transition_acc_neg_tail",
        "transition_acc_pos_tail",
        "transition_frob_neg_tail",
        "transition_frob_pos_tail",
        "context_gap_hat",
        "context_gap_true",
        "context_gap_recovery",
    ]
    variants = sorted({str(row["variant"]) for row in rows})
    out: list[dict[str, Any]] = []
    for variant in variants:
        subset = [row for row in rows if str(row["variant"]) == variant]
        agg: dict[str, Any] = {"variant": variant, "num_seeds": len(subset)}
        for key in metrics:
            vals = np.asarray([float(row[key]) for row in subset], dtype=float)
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    env = _build_env(args)
    tail_start = int(args.em_tk) + 1

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        router_uniform = _build_router(args, env, int(seed))
        uniform_metrics = _evaluate_router(router_uniform, env, tail_start=tail_start)
        uniform_row = {
            "seed": int(seed),
            "variant": "uniform_fixed",
            **uniform_metrics,
        }
        rows.append(uniform_row)
        if args.print_per_seed:
            print(json.dumps(uniform_row, indent=2))

        router_learned = _build_router(args, env, int(seed))
        _fit_transition_only(router_learned, env, args)
        learned_metrics = _evaluate_router(router_learned, env, tail_start=tail_start)
        learned_row = {
            "seed": int(seed),
            "variant": "attention_learned",
            **learned_metrics,
        }
        rows.append(learned_row)
        if args.print_per_seed:
            print(json.dumps(learned_row, indent=2))

    agg_rows = _aggregate(rows)
    out_dir = args.out_dir
    _write_csv(out_dir / "synthetic_transition_ablation_per_seed.csv", rows)
    _write_csv(out_dir / "synthetic_transition_ablation_summary.csv", agg_rows)

    for row in agg_rows:
        print(
            f"{row['variant']}: "
            f"avg_tail={row['avg_cost_tail_mean']:.4f}, "
            f"nll={row['transition_nll_tail_mean']:.4f}, "
            f"trans_acc={row['transition_acc_tail_mean']:.4f}, "
            f"regime_acc={row['regime_acc_tail_mean']:.4f}, "
            f"gap_recovery={row['context_gap_recovery_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
