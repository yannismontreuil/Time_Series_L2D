import copy
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for tune_jena_gap.py") from exc


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "config_jena_smoke.yaml"
RUNNER = ROOT / "slds_imm_router.py"


def _deep_set(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def _parse_metrics(stdout: str) -> dict[str, float]:
    wanted = {
        "L2D-SLDS w/ $g_t$ (partial fb)": "ours",
        "L2D-SLDS w/t $g_t$ (partial fb)": "ablation",
        "LinUCB (partial feedback)": "linucb",
        "SharedLinUCB (partial fb)": "shared_linucb",
        "NeuralUCB (partial feedback)": "neuralucb",
        "LinTS (partial feedback)": "lints",
        "EnsembleSampling (partial fb)": "ensemble",
        "Oracle baseline": "oracle",
    }
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        for prefix, key in wanted.items():
            if line.startswith(prefix + ":"):
                val = float(line.split(":")[-1].strip())
                out[key] = val
    return out


def _score(metrics: dict[str, float]) -> tuple[float, float]:
    ours = metrics["ours"]
    baseline_keys = ["linucb", "shared_linucb", "neuralucb", "lints", "ensemble"]
    best_baseline = min(metrics[k] for k in baseline_keys if k in metrics)
    gap = best_baseline - ours
    return gap, best_baseline


def run_candidate(name: str, overrides: dict[str, Any]) -> tuple[dict[str, float], str]:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for path, value in overrides.items():
        _deep_set(cfg, path, value)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="tune_jena_",
        delete=False,
        dir=str(ROOT / "config"),
        encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = tmp.name

    env = os.environ.copy()
    env["FACTOR_DISABLE_PLOT_SHOW"] = "1"
    env["MPLBACKEND"] = "Agg"
    cmd = [sys.executable, str(RUNNER), "--config", tmp_path]
    proc = subprocess.run(
        cmd,
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

    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    metrics = _parse_metrics(proc.stdout)
    missing = {"ours", "shared_linucb", "linucb", "neuralucb", "lints", "ensemble"} - set(metrics)
    if missing:
        raise RuntimeError(f"{name} missing metrics {sorted(missing)}\n{proc.stdout}")
    return metrics, proc.stdout


def main() -> None:
    A_g_1 = [[[0.98]], [[0.97]], [[0.96]], [[0.95]]]
    Q_g_1 = [0.03, 0.05, 0.07, 0.09]
    A_g_3 = [
        [[0.985, 0.0, 0.0], [0.0, 0.985, 0.0], [0.0, 0.0, 0.985]],
        [[0.975, 0.0, 0.0], [0.0, 0.975, 0.0], [0.0, 0.0, 0.975]],
        [[0.965, 0.0, 0.0], [0.0, 0.965, 0.0], [0.0, 0.0, 0.965]],
        [[0.955, 0.0, 0.0], [0.0, 0.955, 0.0], [0.0, 0.0, 0.955]],
    ]
    Q_g_3 = [0.04, 0.06, 0.08, 0.10]
    candidates = [
        (
            "baseline_smoke",
            {},
        ),
        (
            "uniform_transition",
            {
                "routers.factorized_slds.transition_init": "uniform",
            },
        ),
        (
            "linear_transition_uniform",
            {
                "routers.factorized_slds.transition_init": "uniform",
                "routers.factorized_slds.transition_mode": "linear",
            },
        ),
        (
            "shared_dim1",
            {
                "routers.factorized_slds.transition_init": "uniform",
                "routers.factorized_slds.shared_dim": 1,
                "routers.factorized_slds.A_g": A_g_1,
                "routers.factorized_slds.Q_g_scales": Q_g_1,
            },
        ),
        (
            "shared_dim3",
            {
                "routers.factorized_slds.transition_init": "uniform",
                "routers.factorized_slds.shared_dim": 3,
                "routers.factorized_slds.A_g": A_g_3,
                "routers.factorized_slds.Q_g_scales": Q_g_3,
            },
        ),
        (
            "faster_u_dynamics",
            {
                "routers.factorized_slds.transition_init": "uniform",
                "routers.factorized_slds.A_u_scale": [0.97, 0.96, 0.95, 0.94],
                "routers.factorized_slds.Q_u_scales": [0.05, 0.06, 0.07, 0.08],
            },
        ),
        (
            "slower_u_dynamics",
            {
                "routers.factorized_slds.transition_init": "uniform",
                "routers.factorized_slds.A_u_scale": [0.992, 0.985, 0.978, 0.97],
                "routers.factorized_slds.Q_u_scales": [0.015, 0.02, 0.025, 0.03],
            },
        ),
        (
            "larger_g_init",
            {
                "routers.factorized_slds.transition_init": "uniform",
                "routers.factorized_slds.init_state_g_scale": 0.7,
                "routers.factorized_slds.B_intercept_load": 1.5,
            },
        ),
        (
            "smaller_g_init",
            {
                "routers.factorized_slds.transition_init": "uniform",
                "routers.factorized_slds.init_state_g_scale": 0.15,
                "routers.factorized_slds.B_intercept_load": 0.5,
            },
        ),
        (
            "lambda_zero",
            {
                "routers.lambda_risk": 0.0,
                "routers.factorized_slds.transition_init": "uniform",
            },
        ),
        (
            "lambda_more_negative",
            {
                "routers.lambda_risk": -0.25,
                "routers.factorized_slds.transition_init": "uniform",
            },
        ),
        (
            "full_feedback_shared_baselines",
            {
                "baselines.shared_linucb.feedback_mode": "full",
                "baselines.lin_ts.feedback_mode": "full",
                "baselines.ensemble_sampling.feedback_mode": "full",
            },
        ),
    ]

    results = []
    for name, overrides in candidates:
        try:
            metrics, _ = run_candidate(name, overrides)
            gap, best_baseline = _score(metrics)
            results.append((name, overrides, metrics, gap, best_baseline))
            print(
                f"{name}: ours={metrics['ours']:.4f}, "
                f"best_baseline={best_baseline:.4f}, gap={gap:.4f}, "
                f"shared_linucb={metrics['shared_linucb']:.4f}, "
                f"ensemble={metrics['ensemble']:.4f}, lints={metrics['lints']:.4f}"
            )
        except Exception as exc:
            print(f"{name}: FAILED: {exc}")

    results.sort(key=lambda item: item[3], reverse=True)
    best = results[0]
    print("\nBEST")
    print(best[0])
    print(best[1])
    print(best[2])
    print(f"gap={best[3]:.4f}")


if __name__ == "__main__":
    main()
