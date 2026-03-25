import argparse
import copy
import csv
import pathlib
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factorized_slds import FactorizedSLDS
from models.router_model import feature_phi
from router_eval import run_factored_router_on_env
from scripts.measure_melbourne_runtime import (
    _apply_factorized_init,
    _build_env,
    _diag_stack,
    _load_cfg,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Melbourne L2D-SLDS cost/runtime tradeoffs."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOT / "config" / "config_melbourne_review.yaml",
        help="Base Melbourne config.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional horizon override for faster checks.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Timing repeats per candidate.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional subset of candidate names to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "out" / "melbourne_tradeoff_sweep.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def _make_A_g(num_regimes: int, d_g: int) -> list[list[list[float]]] | None:
    if d_g <= 0:
        return None
    base = [0.98, 0.97, 0.96, 0.95, 0.94]
    vals = base[:num_regimes]
    return [(np.eye(d_g, dtype=float) * vals[m]).tolist() for m in range(num_regimes)]


def _make_Q_g_scales(num_regimes: int) -> list[float] | None:
    base = [0.08, 0.10, 0.12, 0.14, 0.16]
    return base[:num_regimes]


def _candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "name": "base_r4_g2_gz",
            "num_regimes": 4,
            "shared_dim": 2,
            "exploration": ["g_z"],
            "exploration_mc_samples": 25,
        },
        {
            "name": "r4_g2_gz_mc12",
            "num_regimes": 4,
            "shared_dim": 2,
            "exploration": ["g_z"],
            "exploration_mc_samples": 12,
        },
        {
            "name": "r4_g2_gz_mc6",
            "num_regimes": 4,
            "shared_dim": 2,
            "exploration": ["g_z"],
            "exploration_mc_samples": 6,
        },
        {
            "name": "r3_g2_gz",
            "num_regimes": 3,
            "shared_dim": 2,
            "exploration": ["g_z"],
            "exploration_mc_samples": 25,
        },
        {
            "name": "r3_g2_gz_mc12",
            "num_regimes": 3,
            "shared_dim": 2,
            "exploration": ["g_z"],
            "exploration_mc_samples": 12,
        },
        {
            "name": "r2_g2_gz",
            "num_regimes": 2,
            "shared_dim": 2,
            "exploration": ["g_z"],
            "exploration_mc_samples": 25,
        },
        {
            "name": "r2_g2_gz_mc12",
            "num_regimes": 2,
            "shared_dim": 2,
            "exploration": ["g_z"],
            "exploration_mc_samples": 12,
        },
        {
            "name": "r4_g1_gz",
            "num_regimes": 4,
            "shared_dim": 1,
            "exploration": ["g_z"],
            "exploration_mc_samples": 25,
        },
        {
            "name": "r4_g1_gz_mc12",
            "num_regimes": 4,
            "shared_dim": 1,
            "exploration": ["g_z"],
            "exploration_mc_samples": 12,
        },
        {
            "name": "r3_g1_gz",
            "num_regimes": 3,
            "shared_dim": 1,
            "exploration": ["g_z"],
            "exploration_mc_samples": 25,
        },
        {
            "name": "r3_g1_gz_mc12",
            "num_regimes": 3,
            "shared_dim": 1,
            "exploration": ["g_z"],
            "exploration_mc_samples": 12,
        },
        {
            "name": "r2_g1_gz",
            "num_regimes": 2,
            "shared_dim": 1,
            "exploration": ["g_z"],
            "exploration_mc_samples": 25,
        },
        {
            "name": "r2_g1_gz_mc12",
            "num_regimes": 2,
            "shared_dim": 1,
            "exploration": ["g_z"],
            "exploration_mc_samples": 12,
        },
    ]


def _configure_candidate(
    base_cfg: dict[str, Any],
    cand: dict[str, Any],
    horizon: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    if horizon is not None:
        cfg["environment"]["T"] = int(horizon)
    cfg["environment"]["num_regimes"] = int(cand["num_regimes"])
    factor = cfg["routers"]["factorized_slds"]
    factor["num_regimes"] = int(cand["num_regimes"])
    factor["shared_dim"] = int(cand["shared_dim"])
    factor["A_u_scale"] = list(factor["A_u_scale"][: int(cand["num_regimes"])])
    factor["Q_u_scales"] = list(factor["Q_u_scales"][: int(cand["num_regimes"])])
    factor["A_g"] = _make_A_g(int(cand["num_regimes"]), int(cand["shared_dim"]))
    factor["Q_g_scales"] = _make_Q_g_scales(int(cand["num_regimes"]))
    factor["exploration"] = list(cand["exploration"])
    factor["exploration_mc_samples"] = int(cand["exploration_mc_samples"])
    factor["exploration_diag_enabled"] = False
    factor["transition_init"] = "uniform"
    return cfg


def _build_router(cfg: dict[str, Any], env) -> FactorizedSLDS:
    env_cfg = cfg["environment"]
    factor_cfg = cfg["routers"]["factorized_slds"]
    N = int(env_cfg["num_experts"])
    M = int(factor_cfg["num_regimes"])
    d_phi = int(factor_cfg["idiosyncratic_dim"])
    d_g = int(factor_cfg["shared_dim"])
    beta = np.zeros(N, dtype=float)
    A_g = None if d_g <= 0 else np.asarray(factor_cfg["A_g"], dtype=float)
    Q_g = None if d_g <= 0 else _diag_stack(factor_cfg["Q_g_scales"], d_g)
    A_u = _diag_stack(factor_cfg["A_u_scale"], d_phi)
    Q_u = _diag_stack(factor_cfg["Q_u_scales"], d_phi)
    router = FactorizedSLDS(
        M=M,
        d_g=d_g,
        d_phi=d_phi,
        A_g=A_g,
        Q_g=Q_g,
        A_u=A_u,
        Q_u=Q_u,
        feature_fn=feature_phi,
        beta=beta,
        Delta_max=int(factor_cfg["delta_max"]),
        R=float(factor_cfg.get("R_scalar", 1.0)),
        R_mode="scalar",
        num_experts=N,
        B_intercept_load=float(factor_cfg.get("B_intercept_load", 1.0)),
        attn_dim=int(factor_cfg.get("attn_dim", d_phi)),
        eps=float(factor_cfg.get("eps", 1e-8)),
        exploration=str(factor_cfg.get("exploration", ["g_z"])[0]),
        exploration_mc_samples=int(factor_cfg.get("exploration_mc_samples", 25)),
        exploration_ucb_samples=int(factor_cfg.get("exploration_ucb_samples", 200)),
        exploration_ucb_alpha=factor_cfg.get("exploration_ucb_alpha", None),
        exploration_ucb_schedule=str(
            factor_cfg.get("exploration_ucb_schedule", "inverse_t")
        ),
        exploration_sampling_deterministic=bool(
            factor_cfg.get("exploration_sampling_deterministic", False)
        ),
        exploration_diag_enabled=False,
        exploration_diag_stride=int(factor_cfg.get("exploration_diag_stride", 365)),
        exploration_diag_samples=int(factor_cfg.get("exploration_diag_samples", 80)),
        exploration_diag_print=False,
        exploration_diag_max_records=int(
            factor_cfg.get("exploration_diag_max_records", 2000)
        ),
        observation_mode="residual",
        transition_init=str(factor_cfg.get("transition_init", "uniform")),
        transition_mode=str(factor_cfg.get("transition_mode", "attention")),
        feedback_mode="partial",
        seed=int(factor_cfg.get("seed", 11)),
    )
    _apply_factorized_init(env, [router], factor_cfg)
    return router


def _measure_candidate(cfg: dict[str, Any], repeats: int) -> dict[str, Any]:
    costs_ref = None
    elapsed = []
    for _ in range(repeats):
        env = _build_env(cfg)
        router = _build_router(cfg, env)
        t0 = time.perf_counter()
        costs, _ = run_factored_router_on_env(router, env)
        t1 = time.perf_counter()
        elapsed.append(t1 - t0)
        if costs_ref is None:
            costs_ref = np.asarray(costs, dtype=float)
    avg_cost = float(np.nanmean(costs_ref))
    cumulative_cost = float(np.nansum(costs_ref))
    ms_per_step = 1000.0 * float(np.mean(elapsed)) / float(costs_ref.shape[0])
    return {
        "avg_cost": avg_cost,
        "cumulative_cost": cumulative_cost,
        "ms_per_step": ms_per_step,
        "steps": int(costs_ref.shape[0]),
    }


def _pareto_front(rows: list[dict[str, Any]]) -> list[str]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other["name"] == row["name"]:
                continue
            no_worse = (
                other["avg_cost"] <= row["avg_cost"]
                and other["ms_per_step"] <= row["ms_per_step"]
            )
            strictly_better = (
                other["avg_cost"] < row["avg_cost"]
                or other["ms_per_step"] < row["ms_per_step"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row["name"])
    return frontier


def main() -> None:
    args = _parse_args()
    base_cfg = _load_cfg(args.config)
    candidates = _candidate_grid()
    if args.names:
        wanted = set(args.names)
        candidates = [cand for cand in candidates if cand["name"] in wanted]
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        cfg = _configure_candidate(base_cfg, cand, args.horizon)
        result = _measure_candidate(cfg, args.repeats)
        row = {
            "name": cand["name"],
            "num_regimes": int(cand["num_regimes"]),
            "shared_dim": int(cand["shared_dim"]),
            "exploration": str(cand["exploration"][0]),
            "exploration_mc_samples": int(cand["exploration_mc_samples"]),
            **result,
        }
        rows.append(row)
        print(
            f"{row['name']}: avg={row['avg_cost']:.4f}, "
            f"cum={row['cumulative_cost']:.2f}, "
            f"time={row['ms_per_step']:.3f} ms/step"
        )

    frontier = set(_pareto_front(rows))
    for row in rows:
        row["pareto"] = row["name"] in frontier

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "num_regimes",
                "shared_dim",
                "exploration",
                "exploration_mc_samples",
                "avg_cost",
                "cumulative_cost",
                "ms_per_step",
                "steps",
                "pareto",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nPareto frontier:")
    for row in sorted(rows, key=lambda r: (r["avg_cost"], r["ms_per_step"])):
        if row["pareto"]:
            print(
                f"  {row['name']}: avg={row['avg_cost']:.4f}, "
                f"cum={row['cumulative_cost']:.2f}, "
                f"time={row['ms_per_step']:.3f} ms/step"
            )
    print(f"\nCSV saved to {args.output}")


if __name__ == "__main__":
    main()
