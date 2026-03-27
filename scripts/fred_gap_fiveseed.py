import copy
import csv
import os
import pathlib
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from math import sqrt

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tune_fred_gap import EXPECTED_STATIC, STATIC_TOL, _parse_metrics, _prepare_cfg

RUNNER = ROOT / "slds_imm_router.py"
SEEDS = [11, 12, 13, 14, 15]


def _candidate_specs():
    return [
        ("paper_base", {}, []),
        (
            "tk3000_n25",
            {"routers.factorized_slds.em_tk": 3000, "routers.factorized_slds.em_n": 25},
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
            "tk3000_n25_s80_noval",
            {
                "routers.factorized_slds.em_tk": 3000,
                "routers.factorized_slds.em_n": 25,
                "routers.factorized_slds.em_samples": 80,
                "routers.factorized_slds.em_use_validation": False,
            },
            [],
        ),
    ]


def _stderr(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / sqrt(arr.size))


def run_candidate(index: int, out_dir: pathlib.Path) -> None:
    specs = _candidate_specs()
    name, overrides, transforms = specs[index]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ours_vals = []
    neural_vals = []
    gaps = []
    max_static_diff = 0.0
    static_ok_all = True

    for seed in SEEDS:
        cfg = _prepare_cfg(name, overrides, transforms)
        cfg["environment"]["seed"] = int(seed)
        cfg["routers"]["factorized_slds"]["seed"] = int(seed)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"fred_gap_{name}_s{seed}_",
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

        (out_dir / f"{name}.seed{seed}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (out_dir / f"{name}.seed{seed}.stderr.txt").write_text(proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            raise RuntimeError(f"{name} seed {seed} failed with code {proc.returncode}")

        metrics = _parse_metrics(proc.stdout)
        missing = {"ours", "neural", "expert0", "expert1", "expert2", "expert3", "oracle"} - set(metrics)
        if missing:
            raise RuntimeError(f"{name} seed {seed} missing metrics: {sorted(missing)}")

        static_ok = True
        seed_max_diff = 0.0
        for key, expected in EXPECTED_STATIC.items():
            diff = abs(metrics[key] - expected)
            seed_max_diff = max(seed_max_diff, diff)
            if diff > STATIC_TOL:
                static_ok = False
        max_static_diff = max(max_static_diff, seed_max_diff)
        static_ok_all = static_ok_all and static_ok

        ours = float(metrics["ours"])
        neural = float(metrics["neural"])
        gap = float(neural - ours)

        ours_vals.append(ours)
        neural_vals.append(neural)
        gaps.append(gap)
        rows.append(
            {
                "seed": seed,
                "ours": ours,
                "neural": neural,
                "gap_vs_neural": gap,
                "expert0": float(metrics["expert0"]),
                "expert1": float(metrics["expert1"]),
                "expert2": float(metrics["expert2"]),
                "expert3": float(metrics["expert3"]),
                "oracle": float(metrics["oracle"]),
                "static_ok": int(static_ok),
                "max_static_diff": seed_max_diff,
            }
        )

    summary = {
        "index": index,
        "name": name,
        "ours_mean": float(np.mean(ours_vals)),
        "ours_se": _stderr(ours_vals),
        "neural_mean": float(np.mean(neural_vals)),
        "neural_se": _stderr(neural_vals),
        "gap_mean": float(np.mean(gaps)),
        "gap_se": _stderr(gaps),
        "static_ok_all": int(static_ok_all),
        "max_static_diff": max_static_diff,
    }

    with (out_dir / f"{name}.summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with (out_dir / f"{name}.seeds.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{name}: ours={summary['ours_mean']:.6f}±{summary['ours_se']:.6f}, "
        f"neural={summary['neural_mean']:.6f}±{summary['neural_se']:.6f}, "
        f"gap={summary['gap_mean']:.6f}±{summary['gap_se']:.6f}, "
        f"static_ok_all={bool(summary['static_ok_all'])}, "
        f"max_static_diff={summary['max_static_diff']:.6g}"
    )


def main() -> None:
    parser = ArgumentParser(description="Five-seed full-horizon FRED gap search against NeuralUCB.")
    parser.add_argument("--index", type=int, required=True, help="Candidate index in [0, 4].")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=ROOT / "out" / "fred_gap_fiveseed",
        help="Directory for per-candidate outputs.",
    )
    args = parser.parse_args()
    run_candidate(args.index, args.out_dir)


if __name__ == "__main__":
    main()
