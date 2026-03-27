import copy
import csv
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "exp_FRED_rebuttal.yaml"
RUNNER = ROOT / "slds_imm_router.py"

EXPECTED_STATIC = {
    "expert0": 0.004411,
    "expert1": 0.004567,
    "expert2": 0.004505,
    "expert3": 0.004329,
    "oracle": 0.001754,
}
STATIC_TOL = 5e-5


def _deep_set(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def _trim_regime_lists(cfg: dict[str, Any], M: int) -> None:
    cfg["environment"]["num_regimes"] = M
    factor = cfg["routers"]["factorized_slds"]
    factor["num_regimes"] = M
    factor["A_g"] = factor["A_g"][:M]
    if isinstance(factor.get("A_u_scale"), list):
        factor["A_u_scale"] = factor["A_u_scale"][:M]
    if isinstance(factor.get("Q_u_scales"), list):
        factor["Q_u_scales"] = factor["Q_u_scales"][:M]


def _collapse_shared_dim_to_one(cfg: dict[str, Any]) -> None:
    factor = cfg["routers"]["factorized_slds"]
    M = int(factor["num_regimes"])
    factor["shared_dim"] = 1
    factor["A_g"] = [[[0.95]] for _ in range(M)]
    factor["Q_g_scales"] = 1.5
    b_dict_new: dict[int, list[list[float]]] = {}
    for k, rows in factor["B_dict"].items():
        rows_new = [[float(r[0])] for r in rows]
        b_dict_new[int(k)] = rows_new
    factor["B_dict"] = b_dict_new


def _candidate_specs() -> list[tuple[str, dict[str, Any], list[str]]]:
    return [
        ("paper_base", {}, []),
        ("tk3000", {"routers.factorized_slds.em_tk": 3000}, []),
        ("tk4000", {"routers.factorized_slds.em_tk": 4000}, []),
        ("em_n25", {"routers.factorized_slds.em_n": 25}, []),
        ("samples80", {"routers.factorized_slds.em_samples": 80}, []),
        (
            "tk3000_n25",
            {"routers.factorized_slds.em_tk": 3000, "routers.factorized_slds.em_n": 25},
            [],
        ),
        (
            "tk3000_s80",
            {"routers.factorized_slds.em_tk": 3000, "routers.factorized_slds.em_samples": 80},
            [],
        ),
        (
            "tk3000_n25_s80",
            {
                "routers.factorized_slds.em_tk": 3000,
                "routers.factorized_slds.em_n": 25,
                "routers.factorized_slds.em_samples": 80,
            },
            [],
        ),
        (
            "tk3000_n25_s80_b10",
            {
                "routers.factorized_slds.em_tk": 3000,
                "routers.factorized_slds.em_n": 25,
                "routers.factorized_slds.em_samples": 80,
                "routers.factorized_slds.em_burn_in": 10,
            },
            [],
        ),
        (
            "m2_tk3000_n25",
            {"routers.factorized_slds.em_tk": 3000, "routers.factorized_slds.em_n": 25},
            ["trim_M2"],
        ),
        (
            "dg1_tk3000_n25",
            {"routers.factorized_slds.em_tk": 3000, "routers.factorized_slds.em_n": 25},
            ["dg1"],
        ),
        (
            "m2_dg1_tk3000_n25",
            {"routers.factorized_slds.em_tk": 3000, "routers.factorized_slds.em_n": 25},
            ["trim_M2", "dg1"],
        ),
    ]


def _parse_metrics(stdout: str) -> dict[str, float]:
    wanted = {
        "L2D-SLDS w/ $g_t$ attention": "ours",
        "L2D SLDS w/ $g_t$ attention": "ours",
        "NeuralUCB (partial feedback)": "neural",
        "Always using expert 0": "expert0",
        "Always using expert 1": "expert1",
        "Always using expert 2": "expert2",
        "Always using expert 3": "expert3",
        "Oracle baseline": "oracle",
    }
    out: dict[str, float] = {}
    in_average_block = False
    ordered_prefixes = sorted(wanted.items(), key=lambda kv: len(kv[0]), reverse=True)
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line == "=== Average costs ===":
            in_average_block = True
            continue
        if line.startswith("=== ") and in_average_block:
            break
        if not in_average_block or ":" not in line:
            continue
        label, _, value_str = line.partition(":")
        label = label.strip()
        value_str = value_str.strip()
        for prefix, key in ordered_prefixes:
            if key in out:
                continue
            if label.startswith(prefix):
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str)
                if match:
                    out[key] = float(match.group(0))
                break
    return out


def _prepare_cfg(name: str, overrides: dict[str, Any], transforms: list[str]) -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    # Keep only the baseline needed for this sweep.
    neural_cfg = copy.deepcopy(cfg["baselines"]["neural_ucb"])
    cfg["baselines"] = {"neural_ucb": neural_cfg}

    # Turn off diagnostics for speed; does not affect routing costs.
    if "analysis" in cfg:
        cfg["analysis"]["pred_target_corr"] = {"enabled": False}
        cfg["analysis"]["tri_cycle_corr"] = {"enabled": False}
        cfg["analysis"]["pruning"] = {"enabled": False}
    cfg["transition_log"]["enabled"] = False

    for transform in transforms:
        if transform == "trim_M2":
            _trim_regime_lists(cfg, 2)
        elif transform == "dg1":
            _collapse_shared_dim_to_one(cfg)
        else:
            raise ValueError(f"Unknown transform {transform}")

    for path, value in overrides.items():
        _deep_set(cfg, path, value)

    # Fixed single-seed tuning.
    cfg["environment"]["seed"] = 13
    cfg["routers"]["factorized_slds"]["seed"] = 13
    return cfg


def run_candidate(index: int, out_dir: pathlib.Path) -> None:
    specs = _candidate_specs()
    name, overrides, transforms = specs[index]
    cfg = _prepare_cfg(name, overrides, transforms)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"fred_gap_{name}_",
        delete=False,
        dir=str(ROOT / "config"),
        encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = tmp.name

    env = os.environ.copy()
    env["FACTOR_DISABLE_PLOT_SHOW"] = "1"
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--config", tmp_path],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    (out_dir / f"{name}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{name}.stderr.txt").write_text(proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with code {proc.returncode}")

    metrics = _parse_metrics(proc.stdout)
    missing = {"ours", "neural", "expert0", "expert1", "expert2", "expert3", "oracle"} - set(metrics)
    if missing:
        raise RuntimeError(f"{name} missing metrics: {sorted(missing)}")

    static_ok = True
    max_static_diff = 0.0
    for key, expected in EXPECTED_STATIC.items():
        diff = abs(metrics[key] - expected)
        max_static_diff = max(max_static_diff, diff)
        if diff > STATIC_TOL:
            static_ok = False
    gap = metrics["neural"] - metrics["ours"]

    row = {
        "index": index,
        "name": name,
        "ours": metrics["ours"],
        "neural": metrics["neural"],
        "gap_vs_neural": gap,
        "expert0": metrics["expert0"],
        "expert1": metrics["expert1"],
        "expert2": metrics["expert2"],
        "expert3": metrics["expert3"],
        "oracle": metrics["oracle"],
        "static_ok": int(static_ok),
        "max_static_diff": max_static_diff,
    }
    with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(
        f"{name}: ours={metrics['ours']:.6f}, neural={metrics['neural']:.6f}, "
        f"gap={gap:.6f}, static_ok={static_ok}, max_static_diff={max_static_diff:.6g}"
    )


def main() -> None:
    parser = ArgumentParser(description="Single-seed FRED tuning sweep against NeuralUCB.")
    parser.add_argument("--index", type=int, required=True, help="Candidate index in [0, 11].")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=ROOT / "out" / "fred_gap_sweep",
        help="Directory for per-candidate outputs.",
    )
    args = parser.parse_args()
    run_candidate(args.index, args.out_dir)


if __name__ == "__main__":
    main()
