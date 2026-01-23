import numpy as np
from typing import Callable, Optional, Sequence, Tuple, List


def feature_phi(x: np.ndarray) -> np.ndarray:
    """
    Feature map φ(x) = x (identity).
    """
    return np.asarray(x, dtype=float).reshape(-1)


class SLDSIMMRouter_Corr:
    """
    SLDS + IMM router with correlated experts via a shared latent factor,
    implemented as an exact joint Gaussian filter per regime (no star
    approximation).

    Latent structure (per regime k):
        g_t     ∈ R^{d_g}       : shared factor
        u_{j,t} ∈ R^{d_u}       : idiosyncratic state for expert j
        α_{j,t} = B_j g_t + u_{j,t}
        ℓ_{j,t} = φ(x_t)^T α_{j,t} + v_{j,t}

    Joint state per regime:
        x_t = [g_t; u_{1,t}; ...; u_{N,t}] ∈ R^{d_state},
        d_state = d_g + N * d_u.

    Dynamics (per regime k):
        g_{t+1}   = A_gk g_t   + w^g_k,
        u_{j,t+1} = A_uk u_{j,t} + w^u_k,
      with block-diagonal (A_k, Q_k) over [g_t; u_{1,t}; ...; u_{N,t}].

    This class:
      - runs an IMM filter over regimes with joint Gaussian states,
      - supports partial (bandit) and full feedback,
      - uses a myopic risk-adjusted selection rule,
      - provides horizon-H planning analogous to SLDSIMMRouter.
    """

    def __init__(
        self,
        num_experts: int,
        num_regimes: int,
        shared_dim: int,
        idiosyncratic_dim: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        A_g: np.ndarray,
        Q_g: np.ndarray,
        A_u: np.ndarray,
        Q_u: np.ndarray,
        B: np.ndarray,
        R: np.ndarray,
        Pi: np.ndarray,
        beta: Optional[np.ndarray] = None,
        lambda_risk: float | np.ndarray = 0.0,
        staleness_threshold: Optional[int] = None,
        exploration_mode: str = "greedy",
        feature_mode: str = "fixed",
        feature_learning_rate: float = 0.0,
        feature_freeze_after: Optional[int] = None,
        feature_log_interval: Optional[int] = None,
        feedback_mode: str = "partial",
        eps: float = 1e-8,
        g_mean0: Optional[np.ndarray] = None,
        g_cov0: Optional[np.ndarray] = None,
        u_mean0: Optional[np.ndarray] = None,
        u_cov0: Optional[np.ndarray] = None,
        feature_arch: str = "linear",
        feature_hidden_dim: Optional[int] = None,
        feature_activation: str = "tanh",
        seed: Optional[int] = None,
        context_dim: Optional[int] = None,
    ):
        self.N = int(num_experts)
        self.M = int(num_regimes)
        self.dg = int(shared_dim)
        self.du = int(idiosyncratic_dim)
        self.d_state = self.dg + self.N * self.du

        # Feature handling: base feature map (fixed) and optional
        # learnable linear transform on top.
        self.base_feature_fn = feature_fn
        feature_mode = str(feature_mode)
        assert feature_mode in ("fixed", "learnable")
        self.feature_mode = feature_mode
        self.feature_learning_rate = float(feature_learning_rate)
        self.feature_freeze_after: Optional[int] = (
            int(feature_freeze_after) if feature_freeze_after is not None else None
        )
        self.feature_log_interval: Optional[int] = (
            int(feature_log_interval)
            if feature_log_interval is not None and int(feature_log_interval) > 0
            else None
        )
        # History of feature map norms (time index, Frobenius norm)
        self.feature_W_norm_history: List[Tuple[int, float]] = []

        # Architecture of the learnable feature map:
        #   - "linear": φ(x) = W φ_base(x)
        #   - "mlp":    φ(x) = W2 σ(W1 φ_base(x) + b1) + b2
        feature_arch = str(feature_arch)
        assert feature_arch in ("linear", "mlp")
        self.feature_arch = feature_arch
        if feature_hidden_dim is None:
            feature_hidden_dim = self.du
        self.feature_hidden_dim = int(feature_hidden_dim)

        feature_activation = str(feature_activation)
        assert feature_activation in ("tanh", "relu")
        self.feature_activation = feature_activation

        # Determine base feature dimension and initialize learnable
        # projection if needed.
        try:
            if context_dim is None:
                dummy_x = np.zeros((1,), dtype=float)
            else:
                dummy_x = np.zeros((int(context_dim),), dtype=float)
            base_phi = np.asarray(self.base_feature_fn(dummy_x), dtype=float).reshape(-1)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("feature_fn must accept a 1D context and return a 1D array") from exc
        self.d_feat = int(base_phi.shape[0])
        # When using fixed features, ensure base feature dimension
        # matches the idiosyncratic dimension used in the SLDS.
        if self.feature_mode == "fixed":
            assert (
                self.d_feat == self.du
            ), f"Base feature dimension {self.d_feat} must match idiosyncratic_dim {self.du} in fixed feature mode."

        # Initialize learnable parameters depending on architecture.
        if self.feature_arch == "linear":
            # Learnable linear map W: φ(x) = W φ_base(x). Start from
            # identity so that 'learnable' mode initially reproduces the
            # fixed feature behavior.
            self.feature_W = np.eye(self.du, self.d_feat, dtype=float)
            self.feature_W1 = None
            self.feature_b1 = None
            self.feature_W2 = None
            self.feature_b2 = None
        else:
            # Two-layer MLP: φ(x) = W2 σ(W1 φ_base(x) + b1) + b2.
            H = self.feature_hidden_dim
            rng = np.random.default_rng(None if seed is None else int(seed))
            scale = 0.01
            self.feature_W1 = scale * rng.standard_normal((H, self.d_feat))
            self.feature_b1 = np.zeros(H, dtype=float)
            self.feature_W2 = scale * rng.standard_normal((self.du, H))
            self.feature_b2 = np.zeros(self.du, dtype=float)
            self.feature_W = None

        A_g = np.asarray(A_g, dtype=float)
        Q_g = np.asarray(Q_g, dtype=float)
        A_u = np.asarray(A_u, dtype=float)
        Q_u = np.asarray(Q_u, dtype=float)
        B = np.asarray(B, dtype=float)
        R = np.asarray(R, dtype=float)
        Pi = np.asarray(Pi, dtype=float)

        assert A_g.shape == (self.M, self.dg, self.dg)
        assert Q_g.shape == (self.M, self.dg, self.dg)
        assert A_u.shape == (self.M, self.du, self.du)
        assert Q_u.shape == (self.M, self.du, self.du)
        assert B.shape == (self.N, self.du, self.dg)
        assert R.shape == (self.M, self.N)
        assert Pi.shape == (self.M, self.M)
        assert np.allclose(Pi.sum(axis=1), 1.0, atol=1e-6)

        self.A_g = A_g
        self.Q_g = Q_g
        self.A_u = A_u
        self.Q_u = Q_u
        self.B = B
        self.R = R
        self.Pi = Pi

        if beta is None:
            beta = np.zeros(self.N, dtype=float)
        else:
            beta = np.asarray(beta, dtype=float)
            assert beta.shape == (self.N,)
        self.beta = beta

        # Exploration / selection mode:
        #   - "greedy": risk-adjusted myopic mean-variance rule
        #   - "ids": Information-Directed Sampling (IDS) on C_{j,t} = ℓ_{j,t}+β_j
        assert exploration_mode in ("greedy", "ids")
        self.exploration_mode = exploration_mode

        # Risk aversion parameter:
        #   - if scalar: regime-independent λ,
        #   - if array of shape (M,): regime-specific λ^{(k)}, combined
        #     at time t as \bar λ_t = sum_k b_t(k) λ^{(k)}.
        lambda_arr = np.asarray(lambda_risk, dtype=float)
        if lambda_arr.ndim == 0:
            self.lambda_risk = float(lambda_arr)
            self.lambda_risk_vec: Optional[np.ndarray] = None
        else:
            assert lambda_arr.shape == (self.M,)
            self.lambda_risk_vec = lambda_arr
            # Scalar field kept for backward-compatibility; not used when vec present.
            self.lambda_risk = 0.0
        self.eps = float(eps)

        if g_mean0 is None:
            g_mean0 = np.zeros(self.dg, dtype=float)
        else:
            g_mean0 = np.asarray(g_mean0, dtype=float)
            assert g_mean0.shape == (self.dg,)
        if g_cov0 is None:
            g_cov0 = np.eye(self.dg, dtype=float)
        else:
            g_cov0 = np.asarray(g_cov0, dtype=float)
            assert g_cov0.shape == (self.dg, self.dg)

        if u_mean0 is None:
            u_mean0 = np.zeros(self.du, dtype=float)
        else:
            u_mean0 = np.asarray(u_mean0, dtype=float)
            assert u_mean0.shape == (self.du,)
        if u_cov0 is None:
            u_cov0 = np.eye(self.du, dtype=float)
        else:
            u_cov0 = np.asarray(u_cov0, dtype=float)
            assert u_cov0.shape == (self.du, self.du)

        self.g_mean0 = g_mean0
        self.g_cov0 = g_cov0
        self.u_mean0 = u_mean0
        self.u_cov0 = u_cov0

        assert feedback_mode in ("partial", "full")
        self.feedback_mode = feedback_mode

        # Track which experts have ever been available (for birth prior).
        self._has_joined = np.zeros(self.N, dtype=bool)

        # Registry pruning: staleness threshold Δ_max (in decision epochs).
        # If None, pruning is disabled. If an expert has not been available
        # for more than Δ_max steps, we "prune" its idiosyncratic state and
        # treat it as a new birth the next time it becomes available.
        self.staleness_threshold: Optional[int] = (
            int(staleness_threshold) if staleness_threshold is not None else None
        )
        # Internal time counter (decision epochs) and last-availability times.
        self._time = 0  # will be incremented in update_beliefs
        self._last_available = -np.ones(self.N, dtype=int)

        # Precompute joint dynamics matrices A_k, Q_k
        self.A_joint = np.zeros((self.M, self.d_state, self.d_state), dtype=float)
        self.Q_joint = np.zeros((self.M, self.d_state, self.d_state), dtype=float)
        for k in range(self.M):
            A_k = np.zeros((self.d_state, self.d_state), dtype=float)
            Q_k = np.zeros((self.d_state, self.d_state), dtype=float)
            # Shared factor block
            g_slice = slice(0, self.dg)
            A_k[g_slice, g_slice] = self.A_g[k]
            Q_k[g_slice, g_slice] = self.Q_g[k]
            # Expert-specific blocks
            for j in range(self.N):
                u_slice = self._u_slice(j)
                A_k[u_slice, u_slice] = self.A_u[k]
                Q_k[u_slice, u_slice] = self.Q_u[k]
            self.A_joint[k] = A_k
            self.Q_joint[k] = Q_k

        self.reset_beliefs()

    # --------------------------------------------------------
    # Feature computation (fixed vs learnable)
    # --------------------------------------------------------

    def _feature_activation(self, z: np.ndarray) -> np.ndarray:
        if self.feature_activation == "tanh":
            return np.tanh(z)
        # "relu"
        return np.maximum(z, 0.0)

    def _feature_activation_deriv(self, z: np.ndarray) -> np.ndarray:
        if self.feature_activation == "tanh":
            a = np.tanh(z)
            return 1.0 - a * a
        # "relu"
        return (z > 0).astype(float)

    def _compute_base_feature(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the base (fixed) feature vector φ_base(x).
        """
        phi = self.base_feature_fn(np.asarray(x, dtype=float))
        return np.asarray(phi, dtype=float).reshape(self.d_feat)

    def _compute_feature(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the feature vector used by the SLDS emission.

        Returns (phi_t, phi_base_t), where:
          - phi_base_t is the fixed base feature,
          - phi_t is either phi_base_t ("fixed" mode) or a learnable
            projection of phi_base_t ("learnable" mode), implemented
            either as a single linear layer or as a two-layer MLP.
        """
        phi_base = self._compute_base_feature(x)
        if self.feature_mode == "learnable":
            if self.feature_arch == "linear":
                assert self.feature_W is not None
                phi = self.feature_W @ phi_base
            elif self.feature_arch == "mlp":
                assert self.feature_W1 is not None and self.feature_b1 is not None
                assert self.feature_W2 is not None and self.feature_b2 is not None
                z1 = self.feature_W1 @ phi_base + self.feature_b1
                h1 = self._feature_activation(z1)
                phi = self.feature_W2 @ h1 + self.feature_b2
            else:
                raise ValueError(f"Unknown feature_arch: {self.feature_arch}")
        else:
            phi = phi_base
        return phi, phi_base

    # --------------------------------------------------------
    # Index helpers
    # --------------------------------------------------------

    def _g_slice(self) -> slice:
        return slice(0, self.dg)

    def _u_slice(self, j: int) -> slice:
        j = int(j)
        start = self.dg + j * self.du
        end = start + self.du
        return slice(start, end)

    def _build_obs_vector(self, j: int, phi_t: np.ndarray) -> np.ndarray:
        """
        Build observation vector h_j such that
            ℓ_{j,t} = h_j^T x_t + v_{j,t}.
        """
        j = int(j)
        phi_t = np.asarray(phi_t, dtype=float).reshape(self.du)
        B_j = self.B[j]  # shape (d_u, d_g)
        a_j = B_j.T @ phi_t  # shape (d_g,)
        b_j = phi_t          # shape (d_u,)

        h = np.zeros(self.d_state, dtype=float)
        g_slice = self._g_slice()
        u_slice = self._u_slice(j)
        h[g_slice] = a_j
        h[u_slice] = b_j
        return h

    # --------------------------------------------------------
    # Belief initialization
    # --------------------------------------------------------

    def reset_beliefs(self, b0: Optional[np.ndarray] = None) -> None:
        if b0 is None:
            self.b = np.ones(self.M, dtype=float) / self.M
        else:
            b0 = np.asarray(b0, dtype=float)
            assert b0.shape == (self.M,)
            s = float(b0.sum())
            assert s > 0
            self.b = b0 / s

        self._has_joined[:] = False
        self._time = 0
        self._last_available[:] = -1

        self.m = np.zeros((self.M, self.d_state), dtype=float)
        self.P = np.zeros((self.M, self.d_state, self.d_state), dtype=float)

        for k in range(self.M):
            m_k = np.zeros(self.d_state, dtype=float)
            P_k = np.zeros((self.d_state, self.d_state), dtype=float)

            g_slice = self._g_slice()
            m_k[g_slice] = self.g_mean0
            P_k[g_slice, g_slice] = self.g_cov0

            for j in range(self.N):
                u_slice = self._u_slice(j)
                m_k[u_slice] = self.u_mean0
                P_k[u_slice, u_slice] = self.u_cov0

            P_k = 0.5 * (P_k + P_k.T) + self.eps * np.eye(self.d_state)
            self.m[k] = m_k
            self.P[k] = P_k

    # --------------------------------------------------------
    # IMM interaction + time update
    # --------------------------------------------------------

    def _interaction_and_time_update(self):
        return self._interaction_and_time_update_state(self.b, self.m, self.P)

    def _interaction_and_time_update_state(
        self,
        b: np.ndarray,
        m: np.ndarray,
        P: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        IMM interaction + time update for joint state.
        """
        M, d = self.M, self.d_state

        b = np.asarray(b, dtype=float).reshape(M)
        m = np.asarray(m, dtype=float).reshape(M, d)
        P = np.asarray(P, dtype=float).reshape(M, d, d)

        # Predict regime probabilities: b_{t+1|t} = b_t @ Pi
        b_pred = b @ self.Pi
        s = float(b_pred.sum())
        if s <= 0:
            b_pred = np.ones(M, dtype=float) / M
        else:
            b_pred /= s

        # Mixing weights μ_{i|k} = P(z_t = i | z_{t+1} = k, I_t)
        mu = np.zeros((M, M), dtype=float)
        for k in range(M):
            denom = float(b_pred[k])
            if denom <= self.eps:
                mu[:, k] = 1.0 / M
            else:
                mu[:, k] = self.Pi[:, k] * b
                s_k = float(mu[:, k].sum())
                if s_k <= self.eps:
                    mu[:, k] = 1.0 / M
                else:
                    mu[:, k] /= s_k

        # Moment-matching for each target regime k
        m_mix = np.zeros_like(m)
        P_mix = np.zeros_like(P)
        for k in range(M):
            weights = mu[:, k]
            m0 = np.zeros(d, dtype=float)
            for i in range(M):
                m0 += weights[i] * m[i]
            m_mix[k] = m0

            P0 = np.zeros((d, d), dtype=float)
            for i in range(M):
                diff = (m[i] - m0).reshape(d, 1)
                P0 += weights[i] * (P[i] + diff @ diff.T)
            P0 = 0.5 * (P0 + P0.T) + self.eps * np.eye(d)
            P_mix[k] = P0

        # Time update under joint dynamics
        m_pred = np.zeros_like(m_mix)
        P_pred = np.zeros_like(P_mix)
        for k in range(M):
            A_k = self.A_joint[k]
            Q_k = self.Q_joint[k]
            m_pred[k] = A_k @ m_mix[k]
            P_k = A_k @ P_mix[k] @ A_k.T + Q_k
            P_pred[k] = 0.5 * (P_k + P_k.T) + self.eps * np.eye(self.d_state)

        return b_pred, m_pred, P_pred

    # --------------------------------------------------------
    # Dynamic expert availability: birth prior for u_{j,t}
    # --------------------------------------------------------

    def _apply_new_expert_prior(
        self,
        available_experts: np.ndarray,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        mark_joined: bool = True,
    ) -> None:
        """
        For experts that become available for the first time, reset their
        idiosyncratic state u_{j,t} to the birth prior (u_mean0, u_cov0)
        and zero out cross-covariances with other components.
        """
        available_experts = np.asarray(available_experts, dtype=int)
        if available_experts.size == 0:
            return
        mask_new = ~self._has_joined[available_experts]
        if not np.any(mask_new):
            return

        new_indices = available_experts[mask_new]
        for j in new_indices:
            u_slice = self._u_slice(j)
            for k in range(self.M):
                m_k = m_pred[k]
                P_k = P_pred[k]
                # Set mean
                m_k[u_slice] = self.u_mean0
                # Zero cross-covariances
                P_k[u_slice, :] = 0.0
                P_k[:, u_slice] = 0.0
                # Set idiosyncratic block covariance
                P_k[u_slice, u_slice] = self.u_cov0
                P_pred[k] = 0.5 * (P_k + P_k.T) + self.eps * np.eye(self.d_state)

        if mark_joined:
            self._has_joined[new_indices] = True

    # --------------------------------------------------------
    # Registry pruning for stale experts
    # --------------------------------------------------------

    def _apply_pruning(
        self,
        available_experts: np.ndarray,
        m_post: np.ndarray,
        P_post: np.ndarray,
    ) -> None:
        """
        Prune experts that have not been available for more than
        staleness_threshold epochs.

        For any such expert j, we:
          - reset its idiosyncratic state u_{j,t} to the birth prior
            (u_mean0, u_cov0),
          - zero out cross-covariances involving u_{j,t},
          - mark _has_joined[j] = False so that the next time j becomes
            available, it is treated as a fresh birth.

        This approximates marginalizing out stale experts from the joint
        Gaussian state while keeping the overall state dimension fixed.
        """
        if self.staleness_threshold is None:
            return

        available_experts = np.asarray(available_experts, dtype=int)
        M = self.M

        # Candidates for pruning: have previously joined, are currently
        # unavailable, and last_available older than Δ_max.
        for j in range(self.N):
            if not self._has_joined[j]:
                continue
            if j in available_experts:
                continue
            last_t = self._last_available[j]
            if last_t < 0:
                continue
            if (self._time - last_t) <= self.staleness_threshold:
                continue

            # Prune expert j by resetting its local block and clearing
            # cross-covariances.
            u_slice = self._u_slice(j)
            for k in range(M):
                m_k = m_post[k]
                P_k = P_post[k]
                m_k[u_slice] = self.u_mean0
                # Zero cross-covariances involving u_{j,t}
                P_k[u_slice, :] = 0.0
                P_k[:, u_slice] = 0.0
                # Set idiosyncratic block to birth covariance
                P_k[u_slice, u_slice] = self.u_cov0
                P_post[k] = 0.5 * (P_k + P_k.T) + self.eps * np.eye(self.d_state)

            # Mark as no longer in the active registry
            self._has_joined[j] = False

    # --------------------------------------------------------
    # Predictive loss distribution (per expert)
    # --------------------------------------------------------

    def _predict_loss_distribution(
        self,
        phi_t: np.ndarray,
        b_pred: np.ndarray,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
    ):
        """
        For each regime k and expert j, compute predictive mean/variance
        of ℓ_{j,t} and then the mixture mean/variance across regimes.
        """
        M, N = self.M, self.N
        phi_t = np.asarray(phi_t, dtype=float).reshape(self.du)

        mu_kj = np.zeros((M, N), dtype=float)
        S_kj = np.zeros((M, N), dtype=float)

        for k in range(M):
            m_k = m_pred[k]
            P_k = P_pred[k]
            for j in range(N):
                h_j = self._build_obs_vector(j, phi_t)
                mu = float(h_j @ m_k)
                S = float(h_j @ (P_k @ h_j) + self.R[k, j])
                S = max(S, self.eps)
                mu_kj[k, j] = mu
                S_kj[k, j] = S

        mean_ell = np.zeros(N, dtype=float)
        var_ell = np.zeros(N, dtype=float)
        for j in range(N):
            mu_mix = float(b_pred @ mu_kj[:, j])
            mean_ell[j] = mu_mix
            var_within = float(b_pred @ S_kj[:, j])
            diff = mu_kj[:, j] - mu_mix
            var_between = float(b_pred @ (diff**2))
            var_ell[j] = max(var_within + var_between, self.eps)

        return mean_ell, var_ell, mu_kj, S_kj

    # --------------------------------------------------------
    # Selection rule at time t
    # --------------------------------------------------------

    def select_expert(
        self,
        x_t: np.ndarray,
        available_experts: Sequence[int],
    ) -> Tuple[int, dict]:
        """
        One decision step: given context x_t and available experts E_t,
        select r_t ∈ E_t using the configured exploration mode
        ("greedy" or "ids").
        """
        available_experts = np.asarray(list(available_experts), dtype=int)
        # Compute features for the current context.
        phi_t, phi_base_t = self._compute_feature(x_t)

        # IMM prediction
        b_pred, m_pred, P_pred = self._interaction_and_time_update()

        # Dynamic availability: apply birth prior for newly available experts
        self._apply_new_expert_prior(
            available_experts=available_experts,
            m_pred=m_pred,
            P_pred=P_pred,
            mark_joined=True,
        )

        # Predictive distribution of losses
        mean_ell, var_ell, mu_kj, S_kj = self._predict_loss_distribution(
            phi_t, b_pred, m_pred, P_pred
        )
        # Keep explicit aliases for regime-conditional means/variances
        # to avoid shadowing the array names inside loops below.
        mu_kj_mat = mu_kj
        S_kj_mat = S_kj

        # Expert selection rule
        if self.exploration_mode == "ids":
            # IDS on one-step cost C_{j,t} = ℓ_{j,t} + β_j, using the
            # IMM predictive mean and information gain proxy from the
            # theory document (Section: IDS for active exploration).
            avail = available_experts

            # Mixture mean cost proxy \widehat m_t(j) ≈ E[ℓ_{j,t} | H_t] + β_j
            mean_cost = mean_ell + self.beta  # shape (N,)
            mean_cost_avail = mean_cost[avail]
            m_min = float(mean_cost_avail.min())

            # Expected regret gap Δ_t(j) = \widehat m_t(j) - min_{j'} \widehat m_t(j')
            delta = np.zeros(self.N, dtype=float)
            delta[avail] = mean_cost[avail] - m_min

            # Information gain I_t(j) = Σ_k \bar c_t(k) I_t^{(k)}(j),
            # with I_t^{(k)}(j) = 0.5 * log( S^{(k)}_{j} / R_{k,j} ),
            # using S^{(k)}_{j} from the predictive variance and
            # clipping for numerical robustness.
            M = self.M
            info_raw = np.zeros(self.N, dtype=float)
            for j in avail:
                info_j = 0.0
                for k in range(M):
                    S_kj_val = float(S_kj_mat[k, j])
                    R_kj = float(self.R[k, j])
                    R_clipped = max(R_kj, self.eps)
                    S_clipped = max(S_kj_val, R_clipped)
                    info_kj = 0.5 * np.log(S_clipped / R_clipped)
                    info_j += float(b_pred[k]) * info_kj
                info_raw[j] = info_j

            # If all actions are essentially non-informative, fall back
            # to greedy w.r.t. mean cost.
            if np.max(info_raw[avail]) <= self.eps:
                avail_mean = mean_cost[avail]
                idx = int(np.argmin(avail_mean))
                r_t = int(avail[idx])
            else:
                info_eff = np.maximum(info_raw, self.eps)
                psi = np.full(self.N, np.inf, dtype=float)
                for j in avail:
                    psi[j] = (delta[j] ** 2) / info_eff[j]
                avail_psi = psi[avail]
                idx = int(np.argmin(avail_psi))
                r_t = int(avail[idx])
        else:
            # Greedy (risk-adjusted mean-variance) rule from the main
            # selection policy: J_{j,t} = \hat ℓ_{j,t} + β_j + λ_eff sqrt(Var).
            # Effective risk aversion at decision time t. When regime-
            # specific λ^{(k)} are provided, we combine them using the
            # predicted regime weights b_{t+1|t}.
            if getattr(self, "lambda_risk_vec", None) is not None:
                lambda_eff = float(self.lambda_risk_vec @ b_pred)
            else:
                lambda_eff = self.lambda_risk

            scores = mean_ell + self.beta + lambda_eff * np.sqrt(var_ell)
            avail_scores = scores[available_experts]
            idx = int(np.argmin(avail_scores))
            r_t = int(available_experts[idx])

        cache = {
            "x_t": np.asarray(x_t, dtype=float),
            "phi_t": np.asarray(phi_t, dtype=float),
            "phi_base_t": np.asarray(phi_base_t, dtype=float),
            "b_pred": b_pred,
            "m_pred": m_pred,
            "P_pred": P_pred,
            "mu_kj": mu_kj,
            "S_kj": S_kj,
        }
        return r_t, cache

    # --------------------------------------------------------
    # Belief update after observing losses
    # --------------------------------------------------------

    def _gaussian_logpdf(self, x: float, mean: float, var: float) -> float:
        var = max(float(var), self.eps)
        return -0.5 * (np.log(2.0 * np.pi * var) + (float(x) - float(mean)) ** 2 / var)

    def update_beliefs(
        self,
        r_t: int,
        loss_obs: float,
        losses_full: Optional[np.ndarray],
        available_experts: Sequence[int],
        cache: dict,
    ) -> None:
        # Advance internal time counter: this corresponds to the end of
        # decision epoch t.
        self._time += 1

        b_pred = cache["b_pred"]
        m_pred = cache["m_pred"].copy()
        P_pred = cache["P_pred"].copy()
        mu_kj = cache["mu_kj"]
        S_kj = cache["S_kj"]
        phi_t = cache["phi_t"]
        phi_base_t = cache.get("phi_base_t", None)

        M, N = self.M, self.N
        available_experts = np.asarray(list(available_experts), dtype=int)

        # Update last-availability timestamps for currently available experts.
        if available_experts.size > 0:
            self._last_available[available_experts] = self._time

        # Safety: ensure birth prior is applied if select_expert was bypassed
        self._apply_new_expert_prior(
            available_experts=available_experts,
            m_pred=m_pred,
            P_pred=P_pred,
            mark_joined=False,
        )

        if self.feedback_mode == "partial":
            # Single observed loss (bandit feedback) in original loss space.
            j_sel = int(r_t)
            h_sel = self._build_obs_vector(j_sel, phi_t)

            for k in range(M):
                S_k = S_kj[k, j_sel]
                mu_sel = mu_kj[k, j_sel]
                if S_k <= self.eps or not np.isfinite(S_k):
                    continue
                innov = float(loss_obs - mu_sel)
                P_k = P_pred[k]
                K_k = P_k @ h_sel / S_k
                m_pred[k] = m_pred[k] + K_k * innov
                P_update = P_k - np.outer(K_k, K_k) * S_k
                P_pred[k] = (
                    0.5 * (P_update + P_update.T) + self.eps * np.eye(self.d_state)
                )

            # Regime update via Bayes' rule
            log_like = np.zeros(M, dtype=float)
            for k in range(M):
                S_k = S_kj[k, j_sel]
                mu_sel = mu_kj[k, j_sel]
                log_like[k] = self._gaussian_logpdf(loss_obs, mu_sel, S_k)
        else:
            # Full feedback: observe losses for all available experts
            assert losses_full is not None
            losses_full = np.asarray(losses_full, dtype=float)
            assert losses_full.shape == (N,)

            # For regime update, we use the joint Gaussian panel
            # likelihood under each regime k, based on the prior
            # predictive state (m_pred[k], P_pred[k]) and the linear
            # observation model with rows built from _build_obs_vector.
            log_like = np.zeros(M, dtype=float)
            for k in range(M):
                avail = available_experts
                if avail.size == 0:
                    # No observations for regime update on this step.
                    continue

                # Assemble observation matrix H and vector y for the
                # currently available experts.
                H_rows = []
                y_vals = []
                for j in avail:
                    H_rows.append(self._build_obs_vector(j, phi_t))
                    y_vals.append(losses_full[j])
                H = np.vstack(H_rows)  # shape (L, d_state)
                y = np.asarray(y_vals, dtype=float).reshape(-1)

                m_k = m_pred[k]
                P_k = P_pred[k]

                R_diag = np.array([self.R[k, j] for j in avail], dtype=float)
                R_mat = np.diag(R_diag)

                S = H @ P_k @ H.T + R_mat
                S = 0.5 * (S + S.T) + self.eps * np.eye(S.shape[0])

                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)

                y_pred = H @ m_k
                innov = y - y_pred

                # Multivariate Gaussian log-likelihood
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    # Fallback: treat determinant magnitude robustly
                    det_val = max(float(np.linalg.det(S)), self.eps)
                    logdet = float(np.log(det_val))
                quad = float(innov.T @ (S_inv @ innov))
                L = float(S.shape[0])
                log_like[k] = -0.5 * (L * np.log(2.0 * np.pi) + logdet + quad)

                # Measurement assimilation with all available experts
                # jointly (Kalman batch update).
                K = P_k @ H.T @ S_inv
                m_k_new = m_k + K @ innov
                P_k_new = P_k - K @ S @ K.T
                P_k_new = (
                    0.5 * (P_k_new + P_k_new.T) + self.eps * np.eye(self.d_state)
                )

                m_pred[k] = m_k_new
                P_pred[k] = P_k_new

            # If there were no available experts, log_like remains zero
            # and we skip the batch assimilation above.
            # (The regime update below then reduces to using the prior b_pred.)

        # Regime probability update
        log_post = log_like + np.log(np.maximum(b_pred, self.eps))
        m_max = float(np.max(log_post))
        post_unnorm = np.exp(log_post - m_max)
        s = float(post_unnorm.sum())
        if s <= 0:
            b_post = np.ones(M, dtype=float) / M
        else:
            b_post = post_unnorm / s

        # Apply registry pruning for stale, currently unavailable experts.
        self._apply_pruning(
            available_experts=available_experts,
            m_post=m_pred,
            P_post=P_pred,
        )

        self.b = b_post
        self.m = m_pred
        self.P = P_pred

        # Optional online feature-step for learnable features
        if (
            self.feature_mode == "learnable"
            and self.feature_learning_rate > 0.0
            and phi_base_t is not None
        ):
            self._feature_step_learnable(
                phi_t=np.asarray(phi_t, dtype=float),
                phi_base_t=np.asarray(phi_base_t, dtype=float),
                r_t=int(r_t),
                loss_obs=float(loss_obs),
                losses_full=losses_full,
                available_experts=available_experts,
            )

    # --------------------------------------------------------
    # Horizon-H open-loop prediction
    # --------------------------------------------------------

    def precompute_horizon_states(
        self,
        H: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Starting from current posterior (b_t, m_t, P_t),
        apply H times the IMM prediction to obtain
        (b_{t+h|t}, m_{t+h|t}, P_{t+h|t}) for h=1,...,H.
        """
        b_curr = self.b.copy()
        m_curr = self.m.copy()
        P_curr = self.P.copy()

        b_list: List[np.ndarray] = []
        m_list: List[np.ndarray] = []
        P_list: List[np.ndarray] = []

        for _ in range(H):
            b_curr, m_curr, P_curr = self._interaction_and_time_update_state(
                b_curr, m_curr, P_curr
            )
            b_list.append(b_curr.copy())
            m_list.append(m_curr.copy())
            P_list.append(P_curr.copy())

        return b_list, m_list, P_list

    def plan_horizon_schedule(
        self,
        x_t: np.ndarray,
        H: int,
        experts_predict: Sequence[Callable[[np.ndarray], float]],
        context_update: Callable[[np.ndarray, float], np.ndarray],
        available_experts_per_h: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[np.ndarray], List[float]]:
        """
        Horizon-H planning: given current context x_t and belief state,
        compute a greedy open-loop schedule of expert indices
        (r_{t+1},...,r_{t+H}), assuming that at each future step we
        update the context according to the chosen expert's forecast.

        This does NOT mutate the internal belief state.
        """
        N = self.N
        if available_experts_per_h is None:
            available_experts_per_h = [list(range(N)) for _ in range(H)]
        assert len(available_experts_per_h) == H

        # Precompute open-loop latent states (no observation updates).
        b_list, m_list, P_list = self.precompute_horizon_states(H)

        # Local copy of which experts have joined so far. This allows
        # planning to apply the birth prior for experts that first
        # appear in future availability sets without mutating the
        # router's live registry.
        seen_future = self._has_joined.copy()

        x_curr = x_t
        schedule: List[int] = []
        contexts: List[np.ndarray] = []
        scores: List[float] = []

        for h in range(1, H + 1):
            b_h = b_list[h - 1]
            m_h = m_list[h - 1]
            P_h = P_list[h - 1]

            avail = np.asarray(list(available_experts_per_h[h - 1]), dtype=int)

            # Apply birth prior for experts that appear for the first time
            # in the planning horizon.
            if avail.size > 0:
                mask_new = ~seen_future[avail]
                if np.any(mask_new):
                    new_indices = avail[mask_new]
                    I_state = np.eye(self.d_state, dtype=float)
                    for j in new_indices:
                        u_slice = self._u_slice(j)
                        for k in range(self.M):
                            m_h[k, u_slice] = self.u_mean0
                            P_h[k, u_slice, :] = 0.0
                            P_h[k, :, u_slice] = 0.0
                            P_h[k, u_slice, u_slice] = self.u_cov0
                            P_h[k] = 0.5 * (P_h[k] + P_h[k].T) + self.eps * I_state
                    seen_future[new_indices] = True

            # Planning-time risk parameter: use risk-neutral planning
            # (λ_plan = 0) to minimize expected cost, independently of
            # the online λ used for bandit routing.
            lambda_eff_h = 0.0

            best_score: Optional[float] = None
            best_j: Optional[int] = None
            best_x_next: Optional[np.ndarray] = None

            for j in avail:
                # Hypothetical forecast and context propagation.
                y_hat_j = experts_predict[j](x_curr)
                x_next_j = context_update(x_curr, y_hat_j)
                phi_next_j, _ = self._compute_feature(x_next_j)
                phi_next_j = phi_next_j.reshape(self.du)

                # Predictive mean/variance for expert j at step t+h.
                h_j = self._build_obs_vector(j, phi_next_j)

                mu_k = np.zeros(self.M, dtype=float)
                S_k = np.zeros(self.M, dtype=float)
                for k in range(self.M):
                    mu_k[k] = float(h_j @ m_h[k])
                    S_val = float(h_j @ (P_h[k] @ h_j) + self.R[k, j])
                    S_k[k] = max(S_val, self.eps)

                mean_ell_j = float(b_h @ mu_k)
                var_within = float(b_h @ S_k)
                diff = mu_k - mean_ell_j
                var_between = float(b_h @ (diff**2))
                var_ell_j = max(var_within + var_between, self.eps)

                score_j = mean_ell_j + self.beta[j] + lambda_eff_h * np.sqrt(
                    var_ell_j
                )

                if (best_score is None) or (score_j < best_score):
                    best_score = score_j
                    best_j = int(j)
                    best_x_next = x_next_j

            assert best_j is not None
            schedule.append(best_j)
            contexts.append(best_x_next)
            scores.append(float(best_score))
            x_curr = best_x_next

        return schedule, contexts, scores

    # --------------------------------------------------------
    # Feature-step for learnable features (approximate EM-style)
    # --------------------------------------------------------

    def _feature_step_learnable(
        self,
        phi_t: np.ndarray,
        phi_base_t: np.ndarray,
        r_t: int,
        loss_obs: float,
        losses_full: Optional[np.ndarray],
        available_experts: np.ndarray,
    ) -> None:
        """
        One stochastic gradient-ascent step on the expected emission
        log-likelihood Q_em(θ), using the current filtered moments as
        a proxy for the star smoother statistics (no backprop through
        inference, as recommended in the paper's feature-step).

        We treat:
          - (g_t, u_{j,t}) moments from the current joint Gaussian
            posterior (self.m, self.P),
          - regime weights self.b as γ_t^{(k)},
          - and use the closed-form residual expansion from
            E_{j,t}^{(k)}(θ) to obtain ∂Q/∂φ_t, then map to θ via the
            linear relation φ_t = W φ_base_t.
        """
        phi_t = phi_t.reshape(self.du)
        phi_base_t = phi_base_t.reshape(self.d_feat)
        M = self.M

        # Observed expert set O_t
        if self.feedback_mode == "partial":
            obs_experts = np.array([int(r_t)], dtype=int)
        else:
            # Full feedback: use all currently available experts
            obs_experts = np.asarray(available_experts, dtype=int)
            if losses_full is None:
                return
            assert losses_full is not None
            losses_full = np.asarray(losses_full, dtype=float)

        grad_phi = np.zeros_like(phi_t)

        # For each observed expert j in O_t, accumulate gradient
        for j in obs_experts:
            j_int = int(j)
            # Observed loss for expert j
            if self.feedback_mode == "partial":
                ell_j = float(loss_obs)
            else:
                ell_j = float(losses_full[j_int])
                if not np.isfinite(ell_j):
                    continue

            B_j = self.B[j_int]  # (du, dg)

            # Regime-weighted contribution
            for k in range(M):
                gamma_k = float(self.b[k])
                if gamma_k <= 0.0:
                    continue

                R_kj = float(self.R[k, j_int])
                if R_kj <= 0.0:
                    continue

                m_k = self.m[k]
                P_k = self.P[k]

                g_slice = self._g_slice()
                u_slice = self._u_slice(j_int)

                m_g = m_k[g_slice]
                m_u = m_k[u_slice]
                P_g = P_k[g_slice, g_slice]
                P_u = P_k[u_slice, u_slice]
                rho = P_k[g_slice, u_slice]

                # a_{j,t}(θ) = B_j^T φ_t
                a_j = B_j.T @ phi_t

                # Residual r = ℓ_{j,t} - a_j^T m_g - φ_t^T m_u
                c_vec = B_j @ m_g + m_u  # dimension du
                residual = float(ell_j - (a_j @ m_g) - (phi_t @ m_u))

                # Quadratic terms in φ_t
                G_mat = B_j @ (P_g @ B_j.T)  # du x du
                G_sym = 0.5 * (G_mat + G_mat.T)
                M_mat = B_j @ rho  # du x du

                # ∂E_jt^{(k)}/∂φ_t based on
                #   E = r^2 + φ^T G φ + φ^T P_u φ + 2 φ^T M φ
                # with r = ℓ - φ^T c and M = B_j ρ.
                dE_dphi = (
                    -2.0 * residual * c_vec
                    + 2.0 * (G_sym @ phi_t)
                    + 2.0 * (P_u @ phi_t)
                    + 2.0 * ((M_mat + M_mat.T) @ phi_t)
                )

                # Q_em(θ) = -0.5 Σ_t Σ_k γ_t^{(k)} Σ_j E_jt^{(k)} / R_{k,j}
                # ⇒ ∂Q_em/∂φ_t contribution:
                grad_phi += -0.5 * gamma_k * (1.0 / R_kj) * dE_dphi

        # Map gradient wrt φ_t to feature parameters via chain rule.
        if self.feature_arch == "linear":
            # φ_t = W φ_base_t  ⇒  ∂Q/∂W = grad_phi ⊗ φ_base_t
            grad_W = np.outer(grad_phi, phi_base_t)
            assert self.feature_W is not None
            self.feature_W += self.feature_learning_rate * grad_W
            W_norm = float(np.linalg.norm(self.feature_W))
        else:
            # Two-layer MLP: φ = W2 σ(W1 φ_base + b1) + b2
            assert self.feature_W1 is not None and self.feature_b1 is not None
            assert self.feature_W2 is not None and self.feature_b2 is not None
            z1 = self.feature_W1 @ phi_base_t + self.feature_b1
            h1 = self._feature_activation(z1)
            act_deriv = self._feature_activation_deriv(z1)

            # Top layer gradients
            grad_W2 = np.outer(grad_phi, h1)
            grad_b2 = grad_phi

            # Backprop into hidden layer
            g1 = self.feature_W2.T @ grad_phi
            grad_z1 = g1 * act_deriv
            grad_W1 = np.outer(grad_z1, phi_base_t)
            grad_b1 = grad_z1

            lr = self.feature_learning_rate
            self.feature_W2 += lr * grad_W2
            self.feature_b2 += lr * grad_b2
            self.feature_W1 += lr * grad_W1
            self.feature_b1 += lr * grad_b1

            W_norm = float(np.linalg.norm(self.feature_W1) + np.linalg.norm(self.feature_W2))

        # Logging of feature-map norm for diagnostics / ablation.
        self.feature_W_norm_history.append((int(self._time), W_norm))
        if self.feature_log_interval is not None:
            if int(self._time) % self.feature_log_interval == 0:
                print(
                    f"[SLDSIMMRouter_Corr] t={self._time}, "
                    f"||feature_W||_F = {W_norm:.6f}"
                )


class RecurrentSLDSIMMRouter_Corr(SLDSIMMRouter_Corr):
    """
    Correlated SLDS-IMM router with a simple recurrent bias on the
    previously selected expert's idiosyncratic state u_{j,t}.

    Differences vs SLDSIMMRouter_Corr:
      - tracks r_prev (expert chosen at t-1), initialized to None
      - optional C_u[k, j, :] bias added to u_j dynamics when r_prev=j
      - reset_beliefs clears r_prev

    The bias is applied after the base IMM interaction+time update, by
    shifting the u_j slice of the joint mean for the previous expert.
    Covariances are left unchanged for simplicity (additive mean shift).
    """

    def __init__(
        self,
        *args,
        C_u: Optional[np.ndarray] = None,  # shape (M, N, d_u)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if C_u is None:
            self.C_u = np.zeros((self.M, self.N, self.du), dtype=float)
        else:
            C_u_arr = np.asarray(C_u, dtype=float)
            assert C_u_arr.shape == (self.M, self.N, self.du)
            self.C_u = C_u_arr

        self.r_prev: Optional[int] = None

    def reset_beliefs(self, b0: Optional[np.ndarray] = None) -> None:
        super().reset_beliefs(b0)
        self.r_prev = None

    def _interaction_and_time_update_state(
        self,
        b: np.ndarray,
        m: np.ndarray,
        P: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_pred, m_pred, P_pred = super()._interaction_and_time_update_state(b, m, P)

        # Apply recurrent bias to u_{r_prev} if available.
        if self.r_prev is not None and 0 <= self.r_prev < self.N:
            u_slice_prev = self._u_slice(self.r_prev)
            for k in range(self.M):
                m_pred[k, u_slice_prev] += self.C_u[k, self.r_prev]

        return b_pred, m_pred, P_pred

    def update_beliefs(
        self,
        r_t: int,
        loss_obs: float,
        losses_full: Optional[np.ndarray],
        available_experts: Sequence[int],
        cache: dict,
    ) -> None:
        super().update_beliefs(r_t, loss_obs, losses_full, available_experts, cache)
        self.r_prev = int(r_t)
