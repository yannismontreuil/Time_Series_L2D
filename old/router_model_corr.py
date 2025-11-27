import numpy as np
from typing import Callable, Optional, Sequence, Tuple, List


def feature_phi(x: np.ndarray) -> np.ndarray:
    """
    Example feature map φ(x) ∈ R^2:
        φ(x) = [1, x_0]^T
    Assumes x is 1D (shape (1,)).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.array([1.0, x[0]], dtype=float)


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
        lambda_risk: float = 0.0,
        feedback_mode: str = "partial",
        eps: float = 1e-8,
        g_mean0: Optional[np.ndarray] = None,
        g_cov0: Optional[np.ndarray] = None,
        u_mean0: Optional[np.ndarray] = None,
        u_cov0: Optional[np.ndarray] = None,
    ):
        self.N = int(num_experts)
        self.M = int(num_regimes)
        self.dg = int(shared_dim)
        self.du = int(idiosyncratic_dim)
        self.d_state = self.dg + self.N * self.du
        self.feature_fn = feature_fn

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

        self.lambda_risk = float(lambda_risk)
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
        select r_t ∈ E_t using the myopic risk-adjusted score.
        """
        available_experts = np.asarray(list(available_experts), dtype=int)
        phi_t = self.feature_fn(x_t)

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

        # Myopic risk-adjusted cost proxy
        scores = mean_ell + self.beta + self.lambda_risk * np.sqrt(var_ell)

        # Restrict to available experts and pick argmin
        avail_scores = scores[available_experts]
        idx = int(np.argmin(avail_scores))
        r_t = int(available_experts[idx])

        cache = {
            "phi_t": np.asarray(phi_t, dtype=float),
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
        b_pred = cache["b_pred"]
        m_pred = cache["m_pred"].copy()
        P_pred = cache["P_pred"].copy()
        mu_kj = cache["mu_kj"]
        S_kj = cache["S_kj"]
        phi_t = cache["phi_t"]

        M, N = self.M, self.N
        available_experts = np.asarray(list(available_experts), dtype=int)

        # Safety: ensure birth prior is applied if select_expert was bypassed
        self._apply_new_expert_prior(
            available_experts=available_experts,
            m_pred=m_pred,
            P_pred=P_pred,
            mark_joined=False,
        )

        if self.feedback_mode == "partial":
            # Single observed loss (bandit feedback)
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

            # For regime update, use prior predictive (mu_kj, S_kj)
            log_like = np.zeros(M, dtype=float)
            for k in range(M):
                acc = 0.0
                for j in available_experts:
                    ell_j = losses_full[j]
                    if not np.isfinite(ell_j):
                        continue
                    mu_jk = mu_kj[k, j]
                    S_jk = S_kj[k, j]
                    acc += self._gaussian_logpdf(ell_j, mu_jk, S_jk)
                log_like[k] = acc

            # Measurement assimilation with all available experts jointly
            for k in range(M):
                avail = available_experts
                if avail.size == 0:
                    continue

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

                K = P_k @ H.T @ S_inv
                y_pred = H @ m_k
                innov = y - y_pred

                m_k_new = m_k + K @ innov
                P_k_new = P_k - K @ S @ K.T
                P_k_new = (
                    0.5 * (P_k_new + P_k_new.T) + self.eps * np.eye(self.d_state)
                )

                m_pred[k] = m_k_new
                P_pred[k] = P_k_new

        # Regime probability update
        log_post = log_like + np.log(np.maximum(b_pred, self.eps))
        m_max = float(np.max(log_post))
        post_unnorm = np.exp(log_post - m_max)
        s = float(post_unnorm.sum())
        if s <= 0:
            b_post = np.ones(M, dtype=float) / M
        else:
            b_post = post_unnorm / s

        self.b = b_post
        self.m = m_pred
        self.P = P_pred

    # --------------------------------------------------------
    # Horizon-H open-loop prediction
    # --------------------------------------------------------

    def precompute_horizon_states(
        self,
        H: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Starting from current posterior (b_t, m_t, P_t),
        apply H times the IMM prediction to obtain:
            (b_{t+h|t}, m_{t+h|t}, P_{t+h|t}), h=1,...,H.
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
        (r_{t+1},...,r_{t+H}), assuming that at each future step we update
        the context according to the chosen expert's forecast.

        This does NOT mutate the internal belief state.
        """
        N = self.N
        if available_experts_per_h is None:
            available_experts_per_h = [list(range(N)) for _ in range(H)]
        assert len(available_experts_per_h) == H

        # Precompute open-loop latent states (no observation updates)
        b_list, m_list, P_list = self.precompute_horizon_states(H)

        x_curr = x_t
        schedule: List[int] = []
        contexts: List[np.ndarray] = []
        scores: List[float] = []

        for h in range(1, H + 1):
            b_h = b_list[h - 1]
            m_h = m_list[h - 1]
            P_h = P_list[h - 1]

            avail = np.asarray(list(available_experts_per_h[h - 1]), dtype=int)
            best_score: Optional[float] = None
            best_j: Optional[int] = None
            best_x_next: Optional[np.ndarray] = None

            for j in avail:
                # Hypothetical forecast and context propagation
                y_hat_j = experts_predict[j](x_curr)
                x_next_j = context_update(x_curr, y_hat_j)
                phi_next_j = self.feature_fn(x_next_j).reshape(self.du)

                # Predictive mean/variance for expert j at step t+h
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

                score_j = mean_ell_j + self.beta[j] + self.lambda_risk * np.sqrt(
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

