import numpy as np
from typing import Callable, Sequence


class LinUCB:
    """
    Linear UCB baseline for the routing problem with contextual experts.

    We model each expert j's loss as a linear function of features φ(x_t):
        ℓ_{j,t} ≈ θ_j^T φ(x_t),
    and maintain a separate ridge-regression model per expert.

    For each available expert j at time t, LinUCB computes:
        μ_j,t  = θ_j^T φ(x_t)
        σ_j,t  = sqrt( φ(x_t)^T A_j^{-1} φ(x_t) )
    and uses an optimistic score on *loss*:
        score_j = μ_j,t - alpha_ucb * σ_j,t + β_j,
    selecting the expert with minimal score among those available.

    Feedback modes:
      - "partial": update only the chosen expert using its observed loss.
      - "full":    update all available experts using their observed losses.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha_ucb: float = 1.0,
        lambda_reg: float = 1.0,
        beta: np.ndarray | None = None,
        feedback_mode: str = "partial",
    ):
        self.N = int(num_experts)
        self.feature_fn = feature_fn

        alpha_ucb = float(alpha_ucb)
        assert alpha_ucb >= 0.0, "alpha_ucb must be non-negative."
        self.alpha_ucb = alpha_ucb

        lambda_reg = float(lambda_reg)
        assert lambda_reg > 0.0, "lambda_reg must be positive."
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

        # Determine feature dimension from a dummy context.
        dummy_x = np.zeros(1, dtype=float)
        phi = np.asarray(self.feature_fn(dummy_x), dtype=float).reshape(-1)
        self.d = int(phi.shape[0])

        # Per-expert ridge-regression matrices:
        #   A_j = λ I + Σ φ φ^T
        #   b_j = Σ φ ℓ_{j}
        self.A = np.zeros((self.N, self.d, self.d), dtype=float)
        self.b = np.zeros((self.N, self.d), dtype=float)
        for j in range(self.N):
            self.A[j] = lambda_reg * np.eye(self.d, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_phi(self, x: np.ndarray) -> np.ndarray:
        phi = np.asarray(self.feature_fn(x), dtype=float).reshape(self.d)
        return phi

    def _theta_and_sigma(self, j: int, phi: np.ndarray) -> tuple[float, float]:
        """
        Return (mu_j, sigma_j) for expert j at the given feature vector φ.
        """
        A_j = self.A[j]
        b_j = self.b[j]
        # For small d, a direct inverse is fine.
        A_inv = np.linalg.inv(A_j)
        theta_j = A_inv @ b_j
        mu = float(theta_j @ phi)
        sigma_sq = float(phi @ (A_inv @ phi))
        sigma = float(np.sqrt(max(sigma_sq, 0.0)))
        return mu, sigma

    # ------------------------------------------------------------------
    # Public API: selection and update
    # ------------------------------------------------------------------

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        """
        One decision step for LinUCB: compute optimistic scores over the
        available experts and return the index of the selected expert.
        """
        phi = self._get_phi(x)
        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            # Fallback: if availability is empty, allow all experts.
            available_experts = np.arange(self.N, dtype=int)

        scores = np.zeros(available_experts.size, dtype=float)
        for idx, j in enumerate(available_experts):
            mu_j, sigma_j = self._theta_and_sigma(int(j), phi)
            # Optimistic in face of uncertainty on *loss*:
            #   lower confidence bound on loss: μ_j - α σ_j.
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
        """
        Update the per-expert linear models using the current feature
        vector and observed losses, according to the configured feedback
        mode ("partial" or "full").
        """
        phi = self._get_phi(x)
        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            available_experts = np.arange(self.N, dtype=int)

        if self.feedback_mode == "partial":
            # Use the provided selection when available to avoid updating
            # a different (counterfactual) expert.
            if selected_expert is None:
                j_sel = self.select_expert(x, available_experts)
            else:
                j_sel = int(selected_expert)
                if j_sel not in available_experts:
                    j_sel = self.select_expert(x, available_experts)
            ell = float(losses_all[j_sel])
            j_idx = int(j_sel)
            self.A[j_idx] += np.outer(phi, phi)
            self.b[j_idx] += phi * ell
        else:
            # Full-feedback variant: update all available experts using
            # their per-step losses.
            for j in available_experts:
                ell = float(losses_all[j])
                j_idx = int(j)
                self.A[j_idx] += np.outer(phi, phi)
                self.b[j_idx] += phi * ell
