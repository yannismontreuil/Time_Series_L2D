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


def transition_objective_linear(model: FactorizedSLDS, x_arr: np.ndarray, xi: np.ndarray) -> float:
    scores = np.einsum("mjd,td->tmj", model.W_lin, x_arr) + model.b_lin
    log_probs = scores - np.logaddexp.reduce(scores, axis=-1, keepdims=True)
    return float(-(xi * log_probs).sum() / max(float(xi.sum()), model.eps))


def transition_objective_attention(
    model: FactorizedSLDS, x_arr: np.ndarray, xi: np.ndarray
) -> float:
    q = np.einsum("mad,td->tma", model.W_q, x_arr)
    k = np.einsum("mad,td->tma", model.W_k, x_arr)
    scores = np.einsum("tma,tna->tmn", q, k) / (model.W_q.shape[1] ** 0.5)
    if model.b_attn is not None:
        scores = scores + model.b_attn
    log_probs = scores - np.logaddexp.reduce(scores, axis=-1, keepdims=True)
    return float(-(xi * log_probs).sum() / max(float(xi.sum()), model.eps))


def main() -> None:
    rng = np.random.default_rng(0)
    T = 50
    d = 4
    M = 3
    N = 2
    contexts = [rng.normal(size=d) for _ in range(T)]
    x_arr = np.stack([np.asarray(contexts[t], dtype=float).reshape(-1) for t in range(1, T)])
    xi = rng.random((T - 1, M, M))
    xi /= xi.sum(axis=(1, 2), keepdims=True)
    feature_fn = feature_fn_factory(d)

    for mode in ("linear", "attention"):
        model = FactorizedSLDS(
            M=M,
            d_g=1,
            d_phi=d,
            feature_fn=feature_fn,
            num_experts=N,
            transition_mode=mode,
            context_dim=d,
        )
        if mode == "linear":
            before = transition_objective_linear(model, x_arr, xi)
            model._train_transition_linear_torch(
                contexts, xi, lr=1e-2, steps=50, weight_decay=0.0, seed=0
            )
            after = transition_objective_linear(model, x_arr, xi)
        else:
            before = transition_objective_attention(model, x_arr, xi)
            model._train_transition_attention_torch(
                contexts, xi, lr=1e-2, steps=50, weight_decay=0.0, seed=0
            )
            after = transition_objective_attention(model, x_arr, xi)
        print(f"{mode} before={before:.6f} after={after:.6f}")
        if not np.isfinite(after):
            raise AssertionError(f"{mode} transition objective became non-finite.")
        if after > before + 1e-8:
            raise AssertionError(
                f"{mode} transition objective increased: before={before} after={after}"
            )

    print("PASS")


if __name__ == "__main__":
    main()
