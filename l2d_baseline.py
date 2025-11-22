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

        # Infer feature dimension from a dummy input (1D context by default).
        dummy_x = np.zeros(1, dtype=float)
        phi = self.feature_fn(dummy_x)
        d = int(np.asarray(phi, dtype=float).shape[0])
        self.d = d

        # Linear decision rule π_j(x) = w_j^T φ(x)
        self.W = np.zeros((self.N, self.d), dtype=float)

    # --------------------------------------------------------
    # Core helpers
    # --------------------------------------------------------

    def _scores(self, x: np.ndarray) -> np.ndarray:
        phi_x = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
        return self.W @ phi_x

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        """
        Select an expert using the learned decision rule π.

        At inference we simply pick argmax_j π_j(x) among available experts.
        """
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
        """
        Single gradient step on the surrogate Φ_def,k^u for one (x, y) pair.

        Parameters
        ----------
        x : np.ndarray
            Current context.
        losses_all : np.ndarray
            Squared losses ℓ_{j} = (m_j(x) - y)^2 for all experts j
            at this time step (provided by the environment).
        available_experts : sequence of int
            Indices of experts available at this time step.
        """
        phi_x = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
        scores = self.W @ phi_x  # shape (N,)

        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        available_experts = np.asarray(list(available_experts), dtype=int)

        # RMSE per expert from squared error
        rmse = np.sqrt(np.maximum(losses_all, 0.0))
        mu = self.alpha * rmse + self.beta  # shape (N,)

        # Restrict surrogate to available experts only.
        mu_avail = mu[available_experts]
        mu_sum_avail = float(mu_avail.sum())
        weights_avail = mu_sum_avail - mu_avail  # shape (|A|,)

        # Log-softmax surrogate Φ_01^u over available experts only
        scores_avail = scores[available_experts]
        s_max = float(np.max(scores_avail))
        exp_s = np.exp(scores_avail - s_max)
        Z = float(exp_s.sum())
        softmax_avail = exp_s / Z

        weights_sum_avail = float(weights_avail.sum())
        # Gradient w.r.t. scores for available experts:
        # ∂L/∂s_k = (sum_j w_j) * softmax_k - w_k, k ∈ A
        grad_scores = np.zeros(self.N, dtype=float)
        grad_scores_avail = weights_sum_avail * softmax_avail - weights_avail
        grad_scores[available_experts] = grad_scores_avail

        # Chain rule: scores = W @ φ(x)
        grad_W = np.outer(grad_scores, phi_x)  # shape (N, d)

        self.W -= self.learning_rate * grad_W
