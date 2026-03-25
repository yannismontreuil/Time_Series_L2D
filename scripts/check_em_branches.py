import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factorized_slds import FactorizedSLDS


def feature_fn_factory(d: int):
    def feature_fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(d, 1)

    return feature_fn


def min_psd_eig(cov: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(cov)))


def main() -> None:
    rng = np.random.default_rng(1)
    T = 80
    N = 3
    d = 3
    contexts = [rng.normal(size=d) for _ in range(T)]
    available_sets = [list(range(N)) for _ in range(T)]
    actions = rng.integers(0, N, size=T)
    residuals = rng.normal(size=T)
    residuals_full = [rng.normal(size=N) for _ in range(T)]
    feature_fn = feature_fn_factory(d)

    cases = [
        ("partial-linear", dict(feedback_mode="partial", transition_mode="linear")),
        ("partial-attention", dict(feedback_mode="partial", transition_mode="attention")),
        ("full-linear", dict(feedback_mode="full", transition_mode="linear")),
        ("full-attention", dict(feedback_mode="full", transition_mode="attention")),
    ]

    for name, kwargs in cases:
        model = FactorizedSLDS(
            M=2,
            d_g=2,
            d_phi=d,
            feature_fn=feature_fn,
            num_experts=N,
            context_dim=d,
            **kwargs,
        )
        fit = model.fit_em(
            contexts,
            available_sets,
            actions,
            residuals,
            residuals_full=residuals_full if kwargs["feedback_mode"] == "full" else None,
            n_em=2,
            n_samples=3,
            burn_in=1,
            theta_steps=5,
            seed=0,
            print_val_loss=False,
        )
        best_score = float(fit["best_nll"])
        if not np.isfinite(best_score):
            raise AssertionError(f"{name} produced non-finite best_score={best_score}")
        min_qg = min(min_psd_eig(model.Q_g[m]) for m in range(model.M))
        min_qu = min(min_psd_eig(model.Q_u[m]) for m in range(model.M))
        if model.R_mode == "full":
            min_r = min(min_psd_eig(model.R[m, k]) for m in range(model.M) for k in range(N))
        else:
            min_r = float(np.min(model.R))
        print(
            f"{name} best_score={best_score:.6f} "
            f"min_qg={min_qg:.6e} min_qu={min_qu:.6e} min_r={min_r:.6e}"
        )
        if min_qg <= 0.0:
            raise AssertionError(f"{name} Q_g lost positive definiteness: {min_qg}")
        if min_qu <= 0.0:
            raise AssertionError(f"{name} Q_u lost positive definiteness: {min_qu}")
        if min_r <= 0.0:
            raise AssertionError(f"{name} R lost positivity: {min_r}")

    print("PASS")


if __name__ == "__main__":
    main()
