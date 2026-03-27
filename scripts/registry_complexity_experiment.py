import argparse
import csv
import copy
import pathlib
import sys
import time
from typing import Any

import numpy as np
import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional plotting dependency
    plt = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.synthetic_env import SyntheticTimeSeriesEnv
from models.factorized_slds import FactorizedSLDS
from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB
from models.router_model import feature_phi
from models.shared_linear_bandits import (
    LinearEnsembleSampling,
    LinearThompsonSampling,
    SharedLinUCB,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic registry/complexity experiment with bounded active set and "
            "growing cumulative expert catalog."
        )
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOT / "config" / "exp_synthetic_1.yaml",
        help="Base synthetic config from the paper.",
    )
    parser.add_argument(
        "--catalog-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Catalog sizes to evaluate.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=3000,
        help="Horizon for the churn experiment.",
    )
    parser.add_argument(
        "--active-size",
        type=int,
        default=4,
        help="Number of simultaneously available experts.",
    )
    parser.add_argument(
        "--block-len",
        type=int,
        default=30,
        help="Rounds between expert-identity churn events.",
    )
    parser.add_argument(
        "--replace-per-block",
        type=int,
        default=1,
        help="How many active experts to replace at each churn block.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="How many repeated timing passes to average per method.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=ROOT / "out" / "complexity_registry",
        help="Directory for CSV summaries and plots.",
    )
    parser.add_argument(
        "--include-neural",
        action="store_true",
        help="Also evaluate NeuralUCB in the full comparison table.",
    )
    return parser.parse_args()


def _load_cfg(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _diag_stack(scales: list[float], d: int) -> np.ndarray:
    arr = np.asarray(scales, dtype=float).reshape(-1)
    out = np.zeros((arr.shape[0], d, d), dtype=float)
    for m, val in enumerate(arr):
        out[m] = float(val) * np.eye(d, dtype=float)
    return out


def _build_family_tri_cycle_cfg(base_env_cfg: dict[str, Any], catalog_size: int) -> dict[str, Any]:
    tri_cycle = copy.deepcopy(base_env_cfg.get("tri_cycle", {}))
    d_g = int(tri_cycle.get("shared_dim", 2))
    if d_g != 2:
        raise ValueError("This experiment assumes shared_dim = 2 in the paper synthetic.")
    if catalog_size % 2 != 0:
        raise ValueError("catalog_size must be even so the two expert families stay balanced.")

    half = catalog_size // 2
    loadings = np.zeros((catalog_size, d_g), dtype=float)
    loadings[:half, 0] = 1.0
    loadings[half:, 1] = 1.0

    # Reuse the paper's two-family logic:
    # family A is good in regime 0 and worse in regime 1; family B is the reverse.
    offset_cycle = np.array([0.00, 0.06, -0.04, 0.10, -0.08, 0.14], dtype=float)
    reg0 = np.zeros(catalog_size, dtype=float)
    reg1 = np.zeros(catalog_size, dtype=float)
    for j in range(catalog_size):
        offs = float(offset_cycle[j % offset_cycle.size])
        if j < half:
            reg0[j] = 0.20 + offs
            reg1[j] = 1.60 + offs
        else:
            reg0[j] = 1.60 + offs
            reg1[j] = 0.80 + offs

    tri_cycle["shared_loadings"] = loadings.tolist()
    tri_cycle["biases_by_regime"] = [reg0.tolist(), reg1.tolist()]
    tri_cycle["regime_pattern"] = [0, 1]
    return tri_cycle


def _make_churn_schedule(
    T: int,
    catalog_size: int,
    active_size: int,
    block_len: int,
    replace_per_block: int,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    if active_size != 4:
        raise ValueError("This experiment currently assumes active_size = 4 for family-balanced slots.")
    if catalog_size < active_size:
        raise ValueError("catalog_size must be at least active_size.")
    if replace_per_block <= 0 or replace_per_block > active_size:
        raise ValueError("replace_per_block must be in {1, ..., active_size}.")
    if catalog_size % 2 != 0:
        raise ValueError("catalog_size must be even.")

    half = catalog_size // 2
    family_pools = {
        0: list(range(0, half)),
        1: list(range(half, catalog_size)),
    }
    slot_families = [0, 0, 1, 1]
    next_ptr = {0: 2, 1: 2}
    active_slots = [
        family_pools[0][0],
        family_pools[0][1],
        family_pools[1][0],
        family_pools[1][1],
    ]

    availability = np.zeros((T, catalog_size), dtype=int)
    cumulative_seen = np.zeros(T, dtype=int)
    seen: set[int] = set(active_slots)
    block_id = 0

    for t in range(T):
        if t > 0 and (t % block_len) == 0:
            for rep in range(replace_per_block):
                slot = (block_id * replace_per_block + rep) % active_size
                family = slot_families[slot]
                if next_ptr[family] < len(family_pools[family]):
                    active_slots[slot] = family_pools[family][next_ptr[family]]
                    seen.add(int(active_slots[slot]))
                    next_ptr[family] += 1
            block_id += 1
        availability[t, active_slots] = 1
        cumulative_seen[t] = len(seen)

    return availability, slot_families, cumulative_seen


def _build_env(base_cfg: dict[str, Any], catalog_size: int, args: argparse.Namespace) -> tuple[SyntheticTimeSeriesEnv, np.ndarray]:
    env_cfg = base_cfg["environment"]
    tri_cycle_cfg = _build_family_tri_cycle_cfg(env_cfg, catalog_size)
    availability, _slot_families, cumulative_seen = _make_churn_schedule(
        T=int(args.T),
        catalog_size=int(catalog_size),
        active_size=int(args.active_size),
        block_len=int(args.block_len),
        replace_per_block=int(args.replace_per_block),
    )
    env = SyntheticTimeSeriesEnv(
        num_experts=int(catalog_size),
        num_regimes=int(env_cfg.get("num_regimes", 2)),
        T=int(args.T),
        seed=int(env_cfg.get("seed", 11)),
        data_seed=int(env_cfg.get("data_seed", 42)),
        unavailable_expert_idx=None,
        unavailable_intervals=None,
        arrival_expert_idx=None,
        arrival_intervals=None,
        setting=str(env_cfg.get("setting", "tri_cycle_corr")),
        noise_scale=env_cfg.get("noise_scale", None),
        tri_cycle_cfg=tri_cycle_cfg,
        availability_schedule=availability,
    )
    return env, cumulative_seen


def _build_factorized_router(base_cfg: dict[str, Any], env: SyntheticTimeSeriesEnv) -> FactorizedSLDS:
    factor_cfg = base_cfg["routers"]["factorized_slds"]
    env_cfg = base_cfg["environment"]
    tri_cycle_cfg = _build_family_tri_cycle_cfg(env_cfg, env.num_experts)
    loadings = np.asarray(tri_cycle_cfg["shared_loadings"], dtype=float)
    M = int(factor_cfg["num_regimes"])
    d_g = int(factor_cfg["shared_dim"])
    d_phi = int(factor_cfg["idiosyncratic_dim"])
    if d_phi != 1:
        raise ValueError("This experiment expects idiosyncratic_dim = 1.")
    if loadings.shape != (env.num_experts, d_g):
        raise ValueError("Unexpected shared loading shape in complexity experiment.")
    B_dict = {
        int(k): loadings[int(k)].reshape(d_phi, d_g)
        for k in range(env.num_experts)
    }
    beta = np.zeros(env.num_experts, dtype=float)
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
        num_experts=env.num_experts,
        B_intercept_load=float(factor_cfg.get("B_intercept_load", 1.0)),
        A_g=np.asarray(factor_cfg["A_g"], dtype=float),
        A_u=np.asarray(factor_cfg["A_u"], dtype=float),
        Q_g=np.asarray(tri_cycle_cfg["g_covs"], dtype=float),
        Q_u=_diag_stack(factor_cfg["Q_u_scales"], d_phi),
        attn_dim=int(factor_cfg.get("attn_dim", d_phi)),
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
        transition_init="uniform",
        transition_mode=str(factor_cfg.get("transition_mode", "attention")),
        feedback_mode=str(factor_cfg.get("feedback_mode", "partial")),
        seed=int(factor_cfg.get("seed", 42)),
    )
    router.reset_beliefs()
    return router


def _build_linucb(base_cfg: dict[str, Any], env: SyntheticTimeSeriesEnv) -> LinUCB:
    cfg = base_cfg["baselines"]["linucb"]
    return LinUCB(
        num_experts=env.num_experts,
        feature_fn=feature_phi,
        alpha_ucb=float(cfg.get("alpha_ucb", 5.0)),
        lambda_reg=float(cfg.get("l2_reg", cfg.get("lambda_reg", 1.0))),
        beta=np.zeros(env.num_experts, dtype=float),
        feedback_mode=str(cfg.get("feedback_mode", "partial")),
        context_dim=1,
    )


def _build_shared_linucb(base_cfg: dict[str, Any], env: SyntheticTimeSeriesEnv) -> SharedLinUCB:
    cfg = base_cfg["baselines"].get("shared_linucb", base_cfg["baselines"]["linucb"])
    return SharedLinUCB(
        num_experts=env.num_experts,
        feature_fn=feature_phi,
        alpha_ucb=float(cfg.get("alpha_ucb", 5.0)),
        lambda_reg=float(cfg.get("l2_reg", cfg.get("lambda_reg", 1.0))),
        beta=np.zeros(env.num_experts, dtype=float),
        feedback_mode=str(cfg.get("feedback_mode", "partial")),
        context_dim=1,
        seed=int(cfg.get("seed", 0)),
    )


def _build_neuralucb(base_cfg: dict[str, Any], env: SyntheticTimeSeriesEnv) -> NeuralUCB:
    cfg = base_cfg["baselines"]["neural_ucb"]
    return NeuralUCB(
        num_experts=env.num_experts,
        feature_fn=feature_phi,
        alpha_ucb=float(cfg.get("alpha_ucb", 5.0)),
        lambda_reg=float(cfg.get("lambda_reg", 1.0)),
        beta=np.zeros(env.num_experts, dtype=float),
        hidden_dim=int(cfg.get("hidden_dim", 16)),
        nn_learning_rate=float(cfg.get("nn_learning_rate", 1e-3)),
        feedback_mode=str(cfg.get("feedback_mode", "partial")),
        seed=int(cfg.get("seed", 0)),
        context_dim=1,
    )


def _build_lints(base_cfg: dict[str, Any], env: SyntheticTimeSeriesEnv) -> LinearThompsonSampling:
    cfg = base_cfg["baselines"].get("lin_ts", base_cfg["baselines"].get("shared_linucb", base_cfg["baselines"]["linucb"]))
    return LinearThompsonSampling(
        num_experts=env.num_experts,
        feature_fn=feature_phi,
        lambda_reg=float(cfg.get("l2_reg", cfg.get("lambda_reg", 1.0))),
        beta=np.zeros(env.num_experts, dtype=float),
        feedback_mode=str(cfg.get("feedback_mode", "partial")),
        context_dim=1,
        posterior_scale=float(cfg.get("posterior_scale", 1.0)),
        seed=int(cfg.get("seed", 0)),
    )


def _build_ensemble(base_cfg: dict[str, Any], env: SyntheticTimeSeriesEnv) -> LinearEnsembleSampling:
    cfg = base_cfg["baselines"].get("ensemble_sampling", base_cfg["baselines"].get("shared_linucb", base_cfg["baselines"]["linucb"]))
    return LinearEnsembleSampling(
        num_experts=env.num_experts,
        feature_fn=feature_phi,
        ensemble_size=int(cfg.get("ensemble_size", 16)),
        lambda_reg=float(cfg.get("l2_reg", cfg.get("lambda_reg", 1.0))),
        obs_noise_std=float(cfg.get("obs_noise_std", 1.0)),
        beta=np.zeros(env.num_experts, dtype=float),
        feedback_mode=str(cfg.get("feedback_mode", "partial")),
        context_dim=1,
        seed=int(cfg.get("seed", 0)),
    )


def _precompute_episode(env: SyntheticTimeSeriesEnv) -> dict[str, Any]:
    steps = env.T - 1
    contexts = [env.get_context(t) for t in range(1, env.T)]
    available = [env.get_available_experts(t) for t in range(1, env.T)]
    losses = [env.losses(t) for t in range(1, env.T)]
    preds = np.asarray(env._preds_cache[1:], dtype=float)
    y = np.asarray(env.y[1:], dtype=float).reshape(steps, 1)
    residuals = preds - y
    return {
        "contexts": contexts,
        "available": available,
        "losses": losses,
        "residuals": residuals,
        "steps": steps,
    }


def _factorized_state_entries(router: FactorizedSLDS) -> int:
    total = int(router.w.size + router.mu_g.size + router.Sigma_g.size)
    total += int(sum(v.size for v in router.mu_u.values()))
    total += int(sum(v.size for v in router.Sigma_u.values()))
    return total


def _linucb_state_entries(model: LinUCB) -> int:
    return int(model.A.size + model.b.size)


def _shared_linucb_state_entries(model: SharedLinUCB) -> int:
    return int(model.A.size + model.b.size)


def _neuralucb_state_entries(model: NeuralUCB) -> int:
    return int(model.A.size + model.b_lin.size + model.W1.size + model.b1.size)


def _lints_state_entries(model: LinearThompsonSampling) -> int:
    return int(model.A.size + model.b.size)


def _ensemble_state_entries(model: LinearEnsembleSampling) -> int:
    return int(
        model.A.size
        + model.b.size
        + model.theta0_ensemble.size
        + model.ensemble_rhs.size
        + model.ensemble_thetas.size
    )


def _run_factorized(
    router: FactorizedSLDS,
    episode: dict[str, Any],
    timing_repeats: int,
) -> dict[str, Any]:
    contexts = episode["contexts"]
    available = episode["available"]
    losses = episode["losses"]
    residuals = episode["residuals"]
    steps = int(episode["steps"])

    costs = []
    registry_sizes = []
    state_entries = []
    elapsed_runs = []

    for rep in range(timing_repeats):
        router.reset_beliefs()
        router.current_step = 0
        run_costs = []
        run_registry = []
        run_state = []
        t0 = time.perf_counter()
        for idx in range(steps):
            x_t = contexts[idx]
            avail_t = available[idx]
            r_t, cache = router.select_expert(x_t, avail_t)
            router.update_beliefs(
                r_t=int(r_t),
                loss_obs=float(residuals[idx, int(r_t)]),
                losses_full=None,
                available_experts=avail_t,
                cache=cache,
            )
            run_costs.append(float(losses[idx][int(r_t)]))
            run_registry.append(len(router.registry))
            run_state.append(_factorized_state_entries(router))
        t1 = time.perf_counter()
        elapsed_runs.append(t1 - t0)
        if rep == 0:
            costs = run_costs
            registry_sizes = run_registry
            state_entries = run_state

    return {
        "avg_cost": float(np.mean(costs)),
        "cumulative_cost": float(np.sum(costs)),
        "ms_per_step": 1000.0 * float(np.mean(elapsed_runs)) / float(steps),
        "avg_registry": float(np.mean(registry_sizes)),
        "max_registry": int(np.max(registry_sizes)),
        "avg_state_entries": float(np.mean(state_entries)),
        "max_state_entries": int(np.max(state_entries)),
        "registry_sizes": np.asarray(registry_sizes, dtype=float),
    }


def _run_simple(
    model: Any,
    episode: dict[str, Any],
    timing_repeats: int,
    state_entry_fn,
) -> dict[str, Any]:
    contexts = episode["contexts"]
    available = episode["available"]
    losses = episode["losses"]
    steps = int(episode["steps"])

    costs = []
    elapsed_runs = []

    for rep in range(timing_repeats):
        if hasattr(model, "reset_state"):
            model.reset_state()
        t0 = time.perf_counter()
        run_costs = []
        for idx in range(steps):
            x_t = contexts[idx]
            avail_t = available[idx]
            r_t = int(model.select_expert(x_t, avail_t))
            model.update(
                x_t,
                losses[idx],
                avail_t,
                selected_expert=r_t,
            )
            run_costs.append(float(losses[idx][r_t]))
        t1 = time.perf_counter()
        elapsed_runs.append(t1 - t0)
        if rep == 0:
            costs = run_costs

    state_entries = int(state_entry_fn(model))
    return {
        "avg_cost": float(np.mean(costs)),
        "cumulative_cost": float(np.sum(costs)),
        "ms_per_step": 1000.0 * float(np.mean(elapsed_runs)) / float(steps),
        "avg_registry": float("nan"),
        "max_registry": -1,
        "avg_state_entries": float(state_entries),
        "max_state_entries": int(state_entries),
    }


def _plot_registry_trace(
    out_dir: pathlib.Path,
    largest_catalog: int,
    cumulative_seen: np.ndarray,
    registry_sizes: np.ndarray,
    block_len: int,
) -> None:
    if plt is None:
        return
    t = np.arange(1, cumulative_seen.shape[0] + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(t, cumulative_seen, label="Cumulative experts seen", linewidth=2.0, color="#1f77b4")
    ax.plot(t, registry_sizes, label="Maintained registry $|\\mathcal K_t|$", linewidth=2.0, color="#d62728")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Bounded Active Set, Growing Catalog (R={largest_catalog}, block={block_len})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "registry_trace.pdf")
    fig.savefig(out_dir / "registry_trace.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    base_cfg = _load_cfg(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    largest_trace = None
    largest_seen = None
    largest_catalog = None

    for catalog_size in args.catalog_sizes:
        env, cumulative_seen = _build_env(base_cfg, int(catalog_size), args)
        episode = _precompute_episode(env)

        ours = _build_factorized_router(base_cfg, env)
        ours_row = _run_factorized(ours, episode, timing_repeats=int(args.timing_repeats))
        ours_row.update(
            {
                "catalog_size": int(catalog_size),
                "active_size": int(args.active_size),
                "method": "L2D-SLDS",
            }
        )
        rows.append(ours_row)

        shared = _build_shared_linucb(base_cfg, env)
        shared_row = _run_simple(
            shared,
            episode,
            timing_repeats=int(args.timing_repeats),
            state_entry_fn=_shared_linucb_state_entries,
        )
        shared_row.update(
            {
                "catalog_size": int(catalog_size),
                "active_size": int(args.active_size),
                "method": "SharedLinUCB",
            }
        )
        rows.append(shared_row)

        lints = _build_lints(base_cfg, env)
        lints_row = _run_simple(
            lints,
            episode,
            timing_repeats=int(args.timing_repeats),
            state_entry_fn=_lints_state_entries,
        )
        lints_row.update(
            {
                "catalog_size": int(catalog_size),
                "active_size": int(args.active_size),
                "method": "LinTS",
            }
        )
        rows.append(lints_row)

        ensemble = _build_ensemble(base_cfg, env)
        ensemble_row = _run_simple(
            ensemble,
            episode,
            timing_repeats=int(args.timing_repeats),
            state_entry_fn=_ensemble_state_entries,
        )
        ensemble_row.update(
            {
                "catalog_size": int(catalog_size),
                "active_size": int(args.active_size),
                "method": "EnsembleSampling",
            }
        )
        rows.append(ensemble_row)

        linucb = _build_linucb(base_cfg, env)
        lin_row = _run_simple(
            linucb,
            episode,
            timing_repeats=int(args.timing_repeats),
            state_entry_fn=_linucb_state_entries,
        )
        lin_row.update(
            {
                "catalog_size": int(catalog_size),
                "active_size": int(args.active_size),
                "method": "LinUCB",
            }
        )
        rows.append(lin_row)

        if args.include_neural:
            neural = _build_neuralucb(base_cfg, env)
            neural_row = _run_simple(
                neural,
                episode,
                timing_repeats=int(args.timing_repeats),
                state_entry_fn=_neuralucb_state_entries,
            )
            neural_row.update(
                {
                    "catalog_size": int(catalog_size),
                    "active_size": int(args.active_size),
                    "method": "NeuralUCB",
                }
            )
            rows.append(neural_row)

        if largest_catalog is None or int(catalog_size) > int(largest_catalog):
            largest_catalog = int(catalog_size)
            largest_trace = np.asarray(ours_row["registry_sizes"], dtype=float)
            largest_seen = cumulative_seen[1:].astype(float)

        print(
            f"R={catalog_size:>3d} | ours avg={ours_row['avg_cost']:.4f}, "
            f"time={ours_row['ms_per_step']:.3f} ms/step, "
            f"avg|K|={ours_row['avg_registry']:.2f}, "
            f"SharedLinUCB avg={shared_row['avg_cost']:.4f}, "
            f"time={shared_row['ms_per_step']:.3f} ms/step"
        )

    csv_path = out_dir / "registry_complexity_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "catalog_size",
                "active_size",
                "method",
                "avg_cost",
                "cumulative_cost",
                "ms_per_step",
                "avg_registry",
                "max_registry",
                "avg_state_entries",
                "max_state_entries",
            ],
        )
        writer.writeheader()
        for row in rows:
            out_row = {k: row.get(k) for k in writer.fieldnames}
            writer.writerow(out_row)

    if largest_trace is not None and largest_seen is not None and largest_catalog is not None:
        _plot_registry_trace(
            out_dir=out_dir,
            largest_catalog=int(largest_catalog),
            cumulative_seen=np.asarray(largest_seen, dtype=float),
            registry_sizes=np.asarray(largest_trace, dtype=float),
            block_len=int(args.block_len),
        )

    print(f"\nSaved summary to {csv_path}")


if __name__ == "__main__":
    main()
