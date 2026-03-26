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
RUNNER = ROOT / "slds_imm_router.py"


def _parse_args():
    parser = ArgumentParser(
        description="Run multi-seed regime-count ablations and summarize mean +/- SE."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Config path for one ablation setting.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 12, 13, 14, 15],
        help="Seeds to evaluate.",
    )
    parser.add_argument(
        "--out-csv",
        type=pathlib.Path,
        default=None,
        help="Optional CSV path for the summary row.",
    )
    return parser.parse_args()


def _load_cfg(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_average_cost(stdout: str, label_prefix: str) -> float:
    in_average_block = False
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
        if not label.startswith(label_prefix):
            continue
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str.strip())
        if match is None:
            continue
        return float(match.group(0))
    raise KeyError(f"Could not parse average cost for label prefix '{label_prefix}'.")


def _mean_se(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    mean = sum(vals) / n
    if n <= 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    se = (var ** 0.5) / (n ** 0.5)
    return mean, se


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config.resolve())
    factor_cfg = cfg["routers"]["factorized_slds"]
    env_cfg = cfg["environment"]

    label_prefix = str(factor_cfg.get("label_with_g", "L2D-SLDS"))
    horizon_raw = env_cfg.get("T", None)
    horizon = None if horizon_raw is None else int(horizon_raw)
    results: list[float] = []

    for seed in args.seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["environment"]["seed"] = int(seed)
        run_cfg["routers"]["factorized_slds"]["seed"] = int(seed)
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".yaml",
            prefix="regime_ablation_",
            delete=False,
            dir=str(ROOT / "config"),
            encoding="utf-8",
        ) as tmp:
            yaml.safe_dump(run_cfg, tmp, sort_keys=False)
            tmp_path = pathlib.Path(tmp.name)

        env = os.environ.copy()
        env["FACTOR_DISABLE_PLOT_SHOW"] = "1"
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.run(
            [sys.executable, str(RUNNER), "--config", str(tmp_path)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(proc.returncode)
        value = _extract_average_cost(proc.stdout, label_prefix)
        print(f"seed {seed}: {value:.4f}")
        results.append(value)

    mean, se = _mean_se(results)
    cum = None if horizon is None else mean * horizon
    cum_se = None if horizon is None else se * horizon
    summary = {
        "config": args.config.name,
        "label": label_prefix,
        "num_regimes": int(factor_cfg["num_regimes"]),
        "shared_dim": int(factor_cfg["shared_dim"]),
        "exploration": str(factor_cfg.get("exploration", ["g"])[0]),
        "avg": mean,
        "se": se,
        "cum": cum,
        "cum_se": cum_se,
    }
    print("SUMMARY", summary)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)


if __name__ == "__main__":
    main()
