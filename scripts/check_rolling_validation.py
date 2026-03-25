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


def main() -> None:
    rng = np.random.default_rng(2)
    T = 50
    N = 2
    d = 2
    contexts = [rng.normal(size=d) for _ in range(T)]
    available_sets = [list(range(N)) for _ in range(T)]
    actions = rng.integers(0, N, size=T)
    residuals = rng.normal(size=T)
    feature_fn = feature_fn_factory(d)

    model = FactorizedSLDS(
        M=2,
        d_g=1,
        d_phi=d,
        feature_fn=feature_fn,
        num_experts=N,
        feedback_mode="partial",
        transition_mode="linear",
        context_dim=d,
    )

    calls = []
    original = model._evaluate_nll_with_prefix

    def wrapped(*args, **kwargs):
        value = float(original(*args, **kwargs))
        calls.append(value)
        return value

    model._evaluate_nll_with_prefix = wrapped
    model.fit_em(
        contexts,
        available_sets,
        actions,
        residuals,
        n_em=3,
        n_samples=3,
        burn_in=1,
        theta_steps=3,
        seed=0,
        print_val_loss=False,
        val_strategy="rolling",
        val_roll_splits=3,
        val_roll_len=10,
    )
    print(f"rolling_calls={len(calls)} values={calls}")
    if len(calls) == 0:
        raise AssertionError("Rolling validation did not call _evaluate_nll_with_prefix.")
    if not all(np.isfinite(v) for v in calls):
        raise AssertionError("Rolling validation produced a non-finite NLL.")
    print("PASS")


if __name__ == "__main__":
    main()
