import numpy as np
from typing import Callable, Sequence


class LearningToDeferBaseline:
    """
    Usual learning-to-defer baseline using the surrogate

        Φ_def,k^u(π, x, z) = sum_j (sum_{i≠j} μ_i(x, z)) Φ_01^u(π, x, j),

    where Φ_01^u is the log-softmax surrogate and

        μ_j(x, z) = α_j * RMSE(m_j(x), y) + β_j.

    In this implementation we work with a single step (k=1) and interpret
    RMSE(m_j(x), y) for a single sample as sqrt(squared_error), using the
    per-expert squared losses provided by the environment.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        learning_rate: float = 1e-2,
    ):
        """
        Parameters
        ----------
        num_experts : int
            Number of experts (size of action set).
        feature_fn : callable
            Maps context x to feature vector φ(x) used by the decision rule π.
        alpha : np.ndarray, optional
            Coefficients α_j in μ_j, shape (N,). Defaults to ones.
        beta : np.ndarray, optional
            Consultation costs β_j in μ_j, shape (N,). Defaults to zeros.
        learning_rate : float
            Step size for gradient descent on the surrogate loss.
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

        dummy_x = np.zeros(1, dtype=float)
        phi = self.feature_fn(dummy_x)
        d = int(np.asarray(phi, dtype=float).shape[0])
        self.d = d

        self.W = np.zeros((self.N, self.d), dtype=float)

    def _scores(self, x: np.ndarray) -> np.ndarray:
        phi_x = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
        return self.W @ phi_x

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        scores = self._scores(x)
        available_experts = np.asarray(list(available_experts), dtype=int)
        avail_scores = scores[available_experts]
        idx = int(np.argmax(avail_scores))
        return int(available_experts[idx])

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
    ) -> None:
        phi_x = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
        scores = self.W @ phi_x

        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        available_experts = np.asarray(list(available_experts), dtype=int)

        rmse = np.sqrt(np.maximum(losses_all, 0.0))
        mu = self.alpha * rmse + self.beta

        mu_avail = mu[available_experts]
        mu_sum_avail = float(mu_avail.sum())
        weights_avail = mu_sum_avail - mu_avail

        scores_avail = scores[available_experts]
        s_max = float(np.max(scores_avail))
        exp_s = np.exp(scores_avail - s_max)
        Z = float(exp_s.sum())
        softmax_avail = exp_s / Z

        weights_sum_avail = float(weights_avail.sum())
        grad_scores = np.zeros(self.N, dtype=float)
        grad_scores_avail = weights_sum_avail * softmax_avail - weights_avail
        grad_scores[available_experts] = grad_scores_avail

        grad_W = np.outer(grad_scores, phi_x)

        self.W -= self.learning_rate * grad_W


class L2D_RNN:
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
    ):
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

        dummy_x = np.zeros(1, dtype=float)
        phi = self.feature_fn(dummy_x)
        d = int(np.asarray(phi, dtype=float).shape[0])
        self.d = d

        self.hidden_dim = int(hidden_dim)

        rng = np.random.default_rng(0)
        scale = 0.1
        self.W_xh = rng.normal(scale=scale, size=(self.hidden_dim, self.d))
        self.W_hh = rng.normal(scale=scale, size=(self.hidden_dim, self.hidden_dim))
        self.b_h = np.zeros(self.hidden_dim, dtype=float)
        self.W_hy = rng.normal(scale=scale, size=(self.N, self.hidden_dim))
        self.b_y = np.zeros(self.N, dtype=float)

        self.h = np.zeros(self.hidden_dim, dtype=float)
        self._cache: dict | None = None

    def reset_state(self) -> None:
        self.h = np.zeros(self.hidden_dim, dtype=float)
        self._cache = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        phi_x = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
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

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        scores = self._forward(x)
        available_experts = np.asarray(list(available_experts), dtype=int)
        avail_scores = scores[available_experts]
        idx = int(np.argmax(avail_scores))
        return int(available_experts[idx])

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
    ) -> None:
        if self._cache is None:
            scores = self._forward(x)
            phi_x = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
            h_prev = self.h.copy()
            z = np.zeros_like(self.h)
            h_t = self.h.copy()
        else:
            phi_x = self._cache["phi_x"]
            h_prev = self._cache["h_prev"]
            z = self._cache["z"]
            h_t = self._cache["h_t"]
            scores = self._cache["scores"]

        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        available_experts = np.asarray(list(available_experts), dtype=int)

        rmse = np.sqrt(np.maximum(losses_all, 0.0))
        mu = self.alpha * rmse + self.beta

        mu_avail = mu[available_experts]
        mu_sum_avail = float(mu_avail.sum())
        weights_avail = mu_sum_avail - mu_avail

        scores_avail = scores[available_experts]
        s_max = float(np.max(scores_avail))
        exp_s = np.exp(scores_avail - s_max)
        Z = float(exp_s.sum())
        softmax_avail = exp_s / Z

        weights_sum_avail = float(weights_avail.sum())
        grad_scores = np.zeros(self.N, dtype=float)
        grad_scores_avail = weights_sum_avail * softmax_avail - weights_avail
        grad_scores[available_experts] = grad_scores_avail

        grad_W_hy = np.outer(grad_scores, h_t)
        grad_b_y = grad_scores

        grad_h_t = self.W_hy.T @ grad_scores
        grad_z = grad_h_t * (1.0 - np.tanh(z) ** 2)

        grad_W_xh = np.outer(grad_z, phi_x)
        grad_W_hh = np.outer(grad_z, h_prev)
        grad_b_h = grad_z

        lr = self.learning_rate
        self.W_hy -= lr * grad_W_hy
        self.b_y -= lr * grad_b_y
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
