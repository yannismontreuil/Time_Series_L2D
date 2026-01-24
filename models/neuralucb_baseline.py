import numpy as np
from typing import Callable, Sequence


class NeuralUCB:
    """
    Simple NeuralUCB-style baseline:

      - Shared nonlinear embedding h(x) learned online via SGD.
      - For each expert j, a linear ridge model on top of h(x) with
        LinUCB-style confidence bonus.

    We treat losses as the quantity to *minimize* and define a score
    for each expert j as:

        score_j(x) = μ_j(x) - alpha_ucb * σ_j(x) + β_j,

    where μ_j(x) is the predicted loss and σ_j(x) is an uncertainty
    term derived from the per-expert ridge covariance. The policy
    selects the expert with minimal score among available experts.

    This is a "NeuralLinear" approximation to NeuralUCB: the embedding
    is updated with SGD while the linear heads are updated in a
    closed-form ridge style.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha_ucb: float = 1.0,
        lambda_reg: float = 1.0,
        beta: np.ndarray | None = None,
        hidden_dim: int = 16,
        nn_learning_rate: float = 1e-3,
        feedback_mode: str = "partial",
        seed: int | None = 0,
        context_dim: int | None = None,
    ):
        self.N = int(num_experts)
        self.feature_fn = feature_fn

        alpha_ucb = float(alpha_ucb)
        assert alpha_ucb >= 0.0
        self.alpha_ucb = alpha_ucb

        lambda_reg = float(lambda_reg)
        assert lambda_reg > 0.0
        self.lambda_reg = lambda_reg

        if beta is None:
            beta = np.zeros(self.N, dtype=float)
        else:
            beta = np.asarray(beta, dtype=float)
            assert beta.shape == (self.N,)
        self.beta = beta

        feedback_mode = str(feedback_mode)
        assert feedback_mode in ("partial", "full")
        self.feedback_mode = feedback_mode

        # Base feature dimension
        if context_dim is None:
            dummy_x = np.zeros(1, dtype=float)
        else:
            dummy_x = np.zeros(int(context_dim), dtype=float)
        phi = np.asarray(self.feature_fn(dummy_x), dtype=float).reshape(-1)
        self.d = int(phi.shape[0])

        # Neural embedding: 1-hidden-layer MLP with ReLU.
        self.hidden_dim = int(hidden_dim)
        self.nn_learning_rate = float(nn_learning_rate)

        rng = np.random.default_rng(None if seed is None else int(seed))
        scale = 0.1
        self.W1 = rng.normal(scale=scale, size=(self.hidden_dim, self.d))
        self.b1 = np.zeros(self.hidden_dim, dtype=float)

        # Per-expert ridge matrices in embedding space (NeuralLinear).
        self.A = np.zeros((self.N, self.hidden_dim, self.hidden_dim), dtype=float)
        self.b_lin = np.zeros((self.N, self.hidden_dim), dtype=float)
        for j in range(self.N):
            self.A[j] = self.lambda_reg * np.eye(self.hidden_dim, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phi(self, x: np.ndarray) -> np.ndarray:
        phi = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
        return phi

    def _embed(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the embedding network.

        Returns (h, z), where h = ReLU(z).
        """
        z = self.W1 @ phi + self.b1
        h = np.maximum(z, 0.0)
        return h, z

    def _theta_and_sigma(self, j: int, h: np.ndarray) -> tuple[float, float]:
        """
        Return (mu_j, sigma_j) for expert j at the given embedding h.
        """
        A_j = self.A[j]
        b_j = self.b_lin[j]
        A_inv = np.linalg.inv(A_j)
        theta_j = A_inv @ b_j
        mu = float(np.squeeze(theta_j @ h))
        sigma_sq = float(np.squeeze(h @ (A_inv @ h)))
        sigma = float(np.sqrt(max(sigma_sq, 0.0)))
        return mu, sigma

    # ------------------------------------------------------------------
    # Public API: selection and update
    # ------------------------------------------------------------------

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        phi = self._phi(x)
        h, _ = self._embed(phi)

        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            raise ValueError("NeuralUCB: no available experts in select_expert.")

        scores = np.zeros(available_experts.size, dtype=float)
        for idx, j in enumerate(available_experts):
            mu_j, sigma_j = self._theta_and_sigma(int(j), h)
            scores[idx] = mu_j - self.alpha_ucb * sigma_j + self.beta[j]

        best_idx = int(np.argmin(scores))
        return int(available_experts[best_idx])

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
        selected_expert: int | None = None,
    ) -> None:
        phi = self._phi(x)
        h, z = self._embed(phi)

        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            raise ValueError("NeuralUCB: no available experts in update.")

        # Determine which experts to update (partial vs full feedback).
        if self.feedback_mode == "partial":
            if selected_expert is None:
                j_sel = self.select_expert(x, available_experts)
            else:
                j_sel = int(selected_expert)
                if j_sel not in available_experts:
                    raise ValueError(
                        "NeuralUCB: selected_expert must be in available_experts."
                    )
            update_indices = [int(j_sel)]
        else:
            update_indices = [int(j) for j in available_experts]

        # Update linear heads (ridge matrices) for chosen experts.
        for j in update_indices:
            ell = float(losses_all[j])
            self.A[j] += np.outer(h, h)
            self.b_lin[j] += h * ell

        # One-step SGD on embedding using squared-error loss for the
        # updated experts (average if more than one).
        grads_W1 = np.zeros_like(self.W1)
        grads_b1 = np.zeros_like(self.b1)
        for j in update_indices:
            A_j = self.A[j]
            b_j = self.b_lin[j]
            A_inv = np.linalg.inv(A_j)
            theta_j = A_inv @ b_j
            ell = float(losses_all[j])
            mu = float(theta_j @ h)
            err = mu - ell  # derivative of 0.5 * (mu - ell)^2 wrt mu
            # Gradients wrt h and z for ReLU
            grad_h = err * theta_j
            grad_z = grad_h * (z > 0.0)
            grads_W1 += np.outer(grad_z, phi)
            grads_b1 += grad_z

        if update_indices:
            lr = self.nn_learning_rate
            self.W1 -= lr * grads_W1
            self.b1 -= lr * grads_b1
