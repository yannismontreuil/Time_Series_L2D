import copy
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from argparse import ArgumentParser

import yaml

from environment.etth1_env import ETTh1TimeSeriesEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "config" / "config_melbourne_review.yaml"
RUNNER = ROOT / "slds_imm_router.py"


def _parse_args():
    parser = ArgumentParser(description="Run multi-seed Melbourne evaluation and summarize mean +/- SE.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=BASE,
        help="Config path to evaluate. Defaults to config_melbourne_review.yaml.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 12, 13, 14, 15],
        help="Seeds to evaluate. Defaults to 11 12 13 14 15.",
    )
    return parser.parse_args()


def _extract_metrics(stdout: str, prefixes: dict[str, str]) -> dict[str, float]:
    found: dict[str, float] = {}
    ordered_prefixes = sorted(prefixes.items(), key=lambda item: len(item[0]), reverse=True)
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        label, _, value_str = line.partition(":")
        label = label.strip()
        value_str = value_str.strip()
        for prefix, key in ordered_prefixes:
            if label.startswith(prefix):
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str)
                if match is None:
                    continue
                found[key] = float(match.group(0))
                break
    return found


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    env = ETTh1TimeSeriesEnv(**base_cfg["environment"])
    horizon = int(env.T)

    prefixes = {
        "L2D-SLDS w/": "ours",
        "L2D-SLDS w/o": "ablation",
        "LinUCB (partial feedback)": "linucb",
        "SharedLinUCB (partial fb)": "shared",
        "NeuralUCB (partial feedback)": "neural",
        "LinTS (partial feedback)": "lints",
        "EnsembleSampling (partial fb)": "ensemble",
        "Random baseline": "random",
        "Oracle baseline": "oracle",
    }
    all_metrics = {k: [] for k in prefixes.values()}
    seeds = args.seeds

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["environment"]["seed"] = seed
        cfg["routers"]["factorized_slds"]["seed"] = seed
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".yaml",
            prefix="melbourne_seed_",
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
        os.remove(tmp_path)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(proc.returncode)

        found = _extract_metrics(proc.stdout, prefixes)
        print("seed", seed, found)
        missing = [key for key in all_metrics if key not in found]
        if missing:
            print(proc.stdout)
            raise KeyError(
                f"Missing parsed metrics for seed {seed}: {missing}. "
                "Check metric labels in the evaluation output."
            )
        for key in all_metrics:
            all_metrics[key].append(found[key])

    def mean_se(vals: list[float]) -> tuple[float, float]:
        n = len(vals)
        mean = sum(vals) / n
        if n <= 1:
            return mean, 0.0
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        se = (var ** 0.5) / (n ** 0.5)
        return mean, se

    print("SUMMARY")
    for key, vals in all_metrics.items():
        mean, se = mean_se(vals)
        mean_cum = mean * horizon
        se_cum = se * horizon
        print(key, f"avg={mean:.4f}", f"se={se:.4f}", f"cum={mean_cum:.2f}", f"cum_se={se_cum:.2f}")


if __name__ == "__main__":
    main()
