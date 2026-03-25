import copy
import os
import pathlib
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "config" / "config_jena_tuned.yaml"
RUNNER = ROOT / "slds_imm_router.py"


def main() -> None:
    with BASE.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

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
    seeds = [11, 12, 13, 14, 15]

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["environment"]["seed"] = seed
        cfg["routers"]["factorized_slds"]["seed"] = seed
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".yaml",
            prefix="jena_seed_",
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

        found = {}
        for line in proc.stdout.splitlines():
            for prefix, key in prefixes.items():
                if line.startswith(prefix + ":"):
                    found[key] = float(line.split(":")[-1].strip())
                    break
        print("seed", seed, found)
        for key in all_metrics:
            all_metrics[key].append(found[key])

    def mean_se(vals):
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
        print(key, f"{mean:.4f}", f"{se:.4f}")


if __name__ == "__main__":
    main()
