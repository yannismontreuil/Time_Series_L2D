import numpy as np
from typing import Callable, Optional, Sequence


class L2D:
    """
    Unified Learning-to-Defer baseline with configurable architecture
    ("mlp" or "rnn") and optional sliding-window context.

    It uses the usual learning-to-defer surrogate with weights
        μ_j(x, z) = α_j * RMSE(m_j(x), y) + β_j
    and a log-softmax-based policy over available experts.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        learning_rate: float = 1e-2,
        arch: str = "mlp",
        hidden_dim: int = 8,
        window_size: int = 1,
        seed: Optional[int] = 0,
    ):
        """
        Parameters
        ----------
        num_experts : int
            Number of experts.
        feature_fn : callable
            Maps context x to base feature vector φ(x).
        alpha, beta : np.ndarray, optional
            Coefficients α_j and consultation costs β_j in μ_j.
        learning_rate : float
            Step size for gradient descent on the surrogate loss.
        arch : {"mlp", "rnn"}
            Policy architecture.
        hidden_dim : int
            Hidden dimension for MLP or RNN.
        window_size : int
            Sliding-window length (in feature steps). Use 1 for the
            usual single-step L2D; use >1 for L2D_SW.
        """
        self.N = int(num_experts)
        self.feature_fn = feature_fn

        if alpha is None:
            alpha = np.ones(self.N, dtype=float)
        else:
            alpha = np.asarray(alpha, dtype=float)
            assert alpha.shape == (self.N,)
        if beta is None:
            beta = np.zeros(self.N, dtype=float)
        else:
            beta = np.asarray(beta, dtype=float)
            assert beta.shape == (self.N,)

        self.alpha = alpha
        self.beta = beta
        self.learning_rate = float(learning_rate)

        arch = str(arch).lower()
        assert arch in ("mlp", "rnn"), "arch must be 'mlp' or 'rnn'."
        self.arch = arch
        self.hidden_dim = int(hidden_dim)

        self.window_size = max(1, int(window_size))

        dummy_x = np.zeros(1, dtype=float)
        base_phi = np.asarray(self.feature_fn(dummy_x), dtype=float).reshape(-1)
        self.base_d = int(base_phi.shape[0])
        self.d = self.base_d * self.window_size

        # Sliding-window buffer over base features
        self._phi_window = np.zeros((self.window_size, self.base_d), dtype=float)
        self._last_phi: np.ndarray | None = None

        # Initialize parameters depending on architecture.
        rng = np.random.default_rng(None if seed is None else int(seed))
        scale = 0.1

        if self.arch == "mlp":
            H = self.hidden_dim
            self.W1 = rng.normal(scale=scale, size=(H, self.d))
            self.b1 = np.zeros(H, dtype=float)
            self.W2 = rng.normal(scale=scale, size=(self.N, H))
            self.b2 = np.zeros(self.N, dtype=float)
            # RNN-specific fields are unused.
            self.W_xh = None
            self.W_hh = None
            self.b_h = None
            self.W_hy = None
            self.b_y = None
            self.h = None
            self._cache = None
        else:  # arch == "rnn"
            H = self.hidden_dim
            self.W_xh = rng.normal(scale=scale, size=(H, self.d))
            self.W_hh = rng.normal(scale=scale, size=(H, H))
            self.b_h = np.zeros(H, dtype=float)
            self.W_hy = rng.normal(scale=scale, size=(self.N, H))
            self.b_y = np.zeros(self.N, dtype=float)
            self.h = np.zeros(H, dtype=float)
            self._cache: dict | None = None
            # MLP-specific fields are unused.
            self.W1 = None
            self.b1 = None
            self.W2 = None
            self.b2 = None

    # ------------------------------------------------------------------
    # Feature handling (with optional sliding window)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """
        Reset internal state (RNN hidden state and sliding window).
        """
        self._phi_window[:] = 0.0
        self._last_phi = None
        if self.arch == "rnn":
            assert self.h is not None
            self.h[:] = 0.0
            self._cache = None

    def _advance_and_get_phi(self, x: np.ndarray) -> np.ndarray:
        """
        Update the sliding window with the new context and return the
        aggregated feature vector used for scoring.
        """
        base_phi = np.asarray(self.feature_fn(x), dtype=float).reshape(self.base_d)
        if self.window_size == 1:
            phi = base_phi
        else:
            # Shift window and append new base features.
            self._phi_window[:-1] = self._phi_window[1:]
            self._phi_window[-1] = base_phi
            phi = self._phi_window.reshape(self.d)
        self._last_phi = phi
        return phi

    def _get_phi_for_update(self, x: np.ndarray) -> np.ndarray:
        """
        Feature vector to use for the gradient step. In normal usage,
        this reuses the cached φ_t from the preceding select_expert
        call at the same time step.
        """
        if self._last_phi is not None:
            return self._last_phi
        # Fallback (should rarely be needed): single-step features
        base_phi = np.asarray(self.feature_fn(x), dtype=float).reshape(self.base_d)
        if self.window_size == 1:
            return base_phi
        return np.tile(base_phi, self.window_size)

    # ------------------------------------------------------------------
    # Forward pass (scores) for MLP / RNN
    # ------------------------------------------------------------------

    def _scores(self, phi_x: np.ndarray) -> np.ndarray:
        if self.arch == "mlp":
            assert self.W1 is not None and self.b1 is not None
            assert self.W2 is not None and self.b2 is not None
            z = self.W1 @ phi_x + self.b1
            h = np.tanh(z)
            # Cache for gradient
            self._cache = {"phi_x": phi_x, "z": z, "h": h}
            scores = self.W2 @ h + self.b2
            return scores

        # RNN architecture
        assert self.W_xh is not None and self.W_hh is not None
        assert self.b_h is not None and self.W_hy is not None and self.b_y is not None
        assert self.h is not None

        h_prev = self.h.copy()
        z = self.W_xh @ phi_x + self.W_hh @ h_prev + self.b_h
        h_t = np.tanh(z)
        scores = self.W_hy @ h_t + self.b_y

        self.h = h_t
        self._cache = {
            "phi_x": phi_x,
            "h_prev": h_prev,
            "z": z,
            "h_t": h_t,
            "scores": scores,
        }
        return scores

    # ------------------------------------------------------------------
    # API: select_expert / update
    # ------------------------------------------------------------------

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        phi_x = self._advance_and_get_phi(x)
        scores = self._scores(phi_x)
        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            raise ValueError("L2D: no available experts in select_expert.")
        avail_scores = scores[available_experts]
        idx = int(np.argmax(avail_scores))
        return int(available_experts[idx])

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
        selected_expert: int | None = None,
    ) -> None:
        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            raise ValueError("L2D: no available experts in update.")

        rmse = np.sqrt(np.maximum(losses_all, 0.0))
        mu = self.alpha * rmse + self.beta

        mu_avail = mu[available_experts]
        mu_sum_avail = float(mu_avail.sum())
        weights_avail = mu_sum_avail - mu_avail

        # Use cached features / activations if available, otherwise fall
        # back to recomputing from x (single-step).
        phi_x = self._get_phi_for_update(x)

        if self.arch == "mlp":
            assert self._cache is not None
            h = self._cache["h"]
            scores = self.W2 @ h + self.b2  # type: ignore[arg-type]
        else:
            assert self._cache is not None
            scores = self._cache["scores"]

        scores_avail = scores[available_experts]
        s_max = float(np.max(scores_avail))
        exp_s = np.exp(scores_avail - s_max)
        Z = float(exp_s.sum())
        softmax_avail = exp_s / Z

        weights_sum_avail = float(weights_avail.sum())
        grad_scores = np.zeros(self.N, dtype=float)
        grad_scores_avail = weights_sum_avail * softmax_avail - weights_avail
        grad_scores[available_experts] = grad_scores_avail

        lr = self.learning_rate

        if self.arch == "mlp":
            # Backprop through MLP
            assert self.W1 is not None and self.b1 is not None
            assert self.W2 is not None and self.b2 is not None
            assert self._cache is not None
            h = self._cache["h"]
            z = self._cache["z"]

            grad_W2 = np.outer(grad_scores, h)
            grad_b2 = grad_scores

            grad_h = self.W2.T @ grad_scores
            grad_z = grad_h * (1.0 - np.tanh(z) ** 2)

            grad_W1 = np.outer(grad_z, phi_x)
            grad_b1 = grad_z

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1
        else:
            # Backprop through RNN (single-step, no BPTT).
            assert self.W_xh is not None and self.W_hh is not None
            assert self.b_h is not None and self.W_hy is not None
            assert self.b_y is not None and self.h is not None
            assert self._cache is not None

            h_t = self._cache["h_t"]
            h_prev = self._cache["h_prev"]
            z = self._cache["z"]

            grad_W_hy = np.outer(grad_scores, h_t)
            grad_b_y = grad_scores

            grad_h_t = self.W_hy.T @ grad_scores
            grad_z = grad_h_t * (1.0 - np.tanh(z) ** 2)

            grad_W_xh = np.outer(grad_z, phi_x)
            grad_W_hh = np.outer(grad_z, h_prev)
            grad_b_h = grad_z

            self.W_hy -= lr * grad_W_hy
            self.b_y -= lr * grad_b_y
            self.W_xh -= lr * grad_W_xh
            self.W_hh -= lr * grad_W_hh
            self.b_h -= lr * grad_b_h


class LearningToDeferBaseline(L2D):
    """
    Backwards-compatible alias for the original L2D baseline, implemented
    as an MLP with window_size = 1.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        learning_rate: float = 1e-2,
    ):
        super().__init__(
            num_experts=num_experts,
            feature_fn=feature_fn,
            alpha=alpha,
            beta=beta,
            learning_rate=learning_rate,
            arch="mlp",
            hidden_dim=8,
            window_size=1,
        )


class L2D_RNN(L2D):
    """
    Learning-to-defer baseline with a simple RNN policy π implemented in NumPy.

    The RNN maintains a hidden state h_t and produces scores
        s_t = W_hy h_t + b_y,
    with
        h_t = tanh(W_xh φ(x_t) + W_hh h_{t-1} + b_h).

    As in LearningToDeferBaseline, training uses the Φ_def,k^u surrogate and
    a single-step (no backpropagation through time).
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        learning_rate: float = 1e-3,
        hidden_dim: int = 8,
        seed: Optional[int] = 0,
    ):
        super().__init__(
            num_experts=num_experts,
            feature_fn=feature_fn,
            alpha=alpha,
            beta=beta,
            learning_rate=learning_rate,
            arch="rnn",
            hidden_dim=hidden_dim,
            window_size=1,
            seed=seed,
        )


class L2D_SW(L2D):
    """
    L2D baseline with a sliding-window context of length `window_size`.

    The architecture (MLP or RNN) is controlled by `arch`.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        learning_rate: float = 1e-2,
        arch: str = "mlp",
        hidden_dim: int = 8,
        window_size: int = 5,
        seed: Optional[int] = 0,
    ):
        super().__init__(
            num_experts=num_experts,
            feature_fn=feature_fn,
            alpha=alpha,
            beta=beta,
            learning_rate=learning_rate,
            arch=arch,
            hidden_dim=hidden_dim,
            window_size=window_size,
            seed=seed,
        )
