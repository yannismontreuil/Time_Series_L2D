import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple


def feature_phi(x: np.ndarray) -> np.ndarray:
    """
    Simple example feature map φ(x) ∈ R^2:
        φ(x) = [1, x_0]^T
    Assumes x is 1D (shape (1,)).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.array([1.0, x[0]], dtype=float)


class SLDSIMMRouter:
    """
    Switching Linear Dynamical System (SLDS) + Interacting Multiple Model (IMM)
    router for Learning-to-Defer in time series with multiple experts.

    Setup (per expert j and regime k):
        α_{j,t} ∈ R^d  : latent reliability state
        z_t ∈ {0,...,M-1}  : discrete regime
        φ_t = φ(x_t) ∈ R^d : feature vector
        ℓ_{j,t}            : observed loss

        α_{j,t+1} | (α_{j,t}, z_t = k) ~ N(A_k α_{j,t}, Q_k)
        ℓ_{j,t}   | (α_{j,t}, z_t = k) = φ_t^T α_{j,t} + v_{j,t},
                                            v_{j,t} ~ N(0, R_{k,j})

    IMM approximation:
        b_t(k) = P(z_t = k | I_t)
        α_{j,t} | {z_t = k, I_t} ~ N(m_{j,t|t}^{(k)}, P_{j,t|t}^{(k)})

    This class:
      - selects a single expert r_t ∈ E_t at each time t,
      - handles partial or full feedback on losses,
      - can perform horizon-H planning using expert-driven context updates.
    """

    def __init__(
        self,
        num_experts: int,
        num_regimes: int,
        state_dim: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        A: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Pi: np.ndarray,
        beta: Optional[np.ndarray] = None,
        lambda_risk: float | np.ndarray = 0.0,
        pop_mean: Optional[np.ndarray] = None,
        pop_cov: Optional[np.ndarray] = None,
        feedback_mode: str = "partial",
        eps: float = 1e-8,
    ):
        self.N = int(num_experts)
        self.M = int(num_regimes)
        self.d = int(state_dim)
        self.feature_fn = feature_fn

        A = np.asarray(A, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        Pi = np.asarray(Pi, dtype=float)

        assert A.shape == (self.M, self.d, self.d)
        assert Q.shape == (self.M, self.d, self.d)
        assert R.shape == (self.M, self.N)
        assert Pi.shape == (self.M, self.M)
        assert np.allclose(Pi.sum(axis=1), 1.0, atol=1e-6), "Rows of Pi must sum to 1."

        self.A = A
        self.Q = Q
        self.R = R
        self.Pi = Pi

        if beta is None:
            beta = np.zeros(self.N, dtype=float)
        else:
            beta = np.asarray(beta, dtype=float)
            assert beta.shape == (self.N,)
        self.beta = beta

        # Risk aversion parameter:
        #   - if scalar: regime-independent λ,
        #   - if array of shape (M,): regime-specific λ^{(k)}, combined as
        #     \bar λ_t = sum_k b_t(k) λ^{(k)}.
        lambda_arr = np.asarray(lambda_risk, dtype=float)
        if lambda_arr.ndim == 0:
            self.lambda_risk = float(lambda_arr)
            self.lambda_risk_vec: Optional[np.ndarray] = None
        else:
            assert lambda_arr.shape == (self.M,)
            self.lambda_risk_vec = lambda_arr
            # Scalar kept for backward-compatibility; not used when vec present.
            self.lambda_risk = 0.0

        if pop_mean is None:
            pop_mean = np.zeros(self.d, dtype=float)
        else:
            pop_mean = np.asarray(pop_mean, dtype=float)
            assert pop_mean.shape == (self.d,)
        if pop_cov is None:
            pop_cov = np.eye(self.d, dtype=float)
        else:
            pop_cov = np.asarray(pop_cov, dtype=float)
            assert pop_cov.shape == (self.d, self.d)

        self.pop_mean = pop_mean
        self.pop_cov = pop_cov

        assert feedback_mode in ("partial", "full")
        self.feedback_mode = feedback_mode

        self.eps = float(eps)

        # Track which experts have ever been available (dynamic addition).
        # An expert that appears for the first time in the availability set
        # is initialized from the population prior at that time.
        self._has_joined = np.zeros(self.N, dtype=bool)

        self.reset_beliefs()

    # --------------------------------------------------------
    # Belief initialization
    # --------------------------------------------------------

    def reset_beliefs(self, b0: Optional[np.ndarray] = None) -> None:
        """
        Initialize regime probabilities and expert reliability posteriors.
        """
        if b0 is None:
            self.b = np.ones(self.M, dtype=float) / self.M
        else:
            b0 = np.asarray(b0, dtype=float)
            assert b0.shape == (self.M,)
            s = b0.sum()
            assert s > 0
            self.b = b0 / s

        # When resetting beliefs, forget which experts have joined so far.
        self._has_joined[:] = False

        self.m = np.zeros((self.M, self.N, self.d), dtype=float)
        self.P = np.zeros((self.M, self.N, self.d, self.d), dtype=float)
        for k in range(self.M):
            for j in range(self.N):
                self.m[k, j] = self.pop_mean
                self.P[k, j] = self.pop_cov

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
    ):
        M, N, d = self.M, self.N, self.d

        b = np.asarray(b, dtype=float).reshape(M)
        m = np.asarray(m, dtype=float).reshape(M, N, d)
        P = np.asarray(P, dtype=float).reshape(M, N, d, d)

        # Predict regime distribution: b_{t+1|t}(k) = sum_i b_t(i) Pi[i,k]
        b_pred = b @ self.Pi
        s = b_pred.sum()
        if s <= 0:
            b_pred = np.ones(M, dtype=float) / M
        else:
            b_pred /= s

        # Mixing probabilities μ_{i|k} = P(z_t=i | z_{t+1}=k, I_t)
        mu = np.zeros((M, M), dtype=float)
        for k in range(M):
            denom = b_pred[k]
            if denom <= self.eps:
                mu[:, k] = 1.0 / M
            else:
                mu[:, k] = self.Pi[:, k] * b
                s_k = mu[:, k].sum()
                if s_k <= self.eps:
                    mu[:, k] = 1.0 / M
                else:
                    mu[:, k] /= s_k

        # Moment-matching for each target regime k
        m_mixed = np.zeros_like(m)
        P_mixed = np.zeros_like(P)
        for k in range(M):
            for j in range(N):
                # Mixed mean: sum_i μ_{i|k} m_{t|t}^{(i)}
                m0 = np.zeros(d, dtype=float)
                for i in range(M):
                    m0 += mu[i, k] * m[i, j]
                m_mixed[k, j] = m0

                # Mixed covariance: sum_i μ_{i|k}(P_i + (m_i-m0)(m_i-m0)^T)
                P0 = np.zeros((d, d), dtype=float)
                for i in range(M):
                    diff = (m[i, j] - m0).reshape(d, 1)
                    P0 += mu[i, k] * (P[i, j] + diff @ diff.T)
                P0 += self.eps * np.eye(d)
                P_mixed[k, j] = P0

        # Time update under A_k, Q_k
        m_pred = np.zeros_like(m_mixed)
        P_pred = np.zeros_like(P_mixed)
        for k in range(M):
            A_k = self.A[k]
            Q_k = self.Q[k]
            for j in range(N):
                m_pred[k, j] = A_k @ m_mixed[k, j]
                P_pred[k, j] = A_k @ P_mixed[k, j] @ A_k.T + Q_k
                P_pred[k, j] = (
                    0.5 * (P_pred[k, j] + P_pred[k, j].T) + self.eps * np.eye(d)
                )

        return b_pred, m_pred, P_pred

    # --------------------------------------------------------
    # Dynamic expert availability: prior for newly added experts
    # --------------------------------------------------------

    def _apply_new_expert_prior(
        self,
        available_experts: np.ndarray,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
        mark_joined: bool = True,
    ) -> None:
        """
        Apply the population prior to experts that become available for
        the first time at the current step.

        For any j with j in available_experts and _has_joined[j] == False,
        set for all regimes k:
            m_pred[k, j] = pop_mean
            P_pred[k, j] = pop_cov.

        This matches the "Expert Addition (New Expert Joins)" rule in the
        dynamic availability section of the theory document.
        """
        available_experts = np.asarray(available_experts, dtype=int)
        if available_experts.size == 0:
            return

        mask_new = ~self._has_joined[available_experts]
        if not np.any(mask_new):
            return

        new_indices = available_experts[mask_new]
        for j in new_indices:
            for k in range(self.M):
                m_pred[k, j] = self.pop_mean
                P_pred[k, j] = self.pop_cov

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
        M, N, d = self.M, self.N, self.d
        phi_t = np.asarray(phi_t, dtype=float).reshape(d)

        # Regime-conditional mean/variance for log-transformed loss
        #   \tilde ℓ_{j,t} = log(ℓ_{j,t} + obs_log_eps).
        mu_kj = np.zeros((M, N), dtype=float)  # mean of log-loss
        S_kj = np.zeros((M, N), dtype=float)   # var of log-loss
        for k in range(M):
            for j in range(N):
                m_kj = m_pred[k, j]
                P_kj = P_pred[k, j]
                mu = float(phi_t @ m_kj)
                S = float(phi_t @ (P_kj @ phi_t) + self.R[k, j])
                S_kj[k, j] = max(S, self.eps)
                mu_kj[k, j] = mu

        # Mixture mean/variance over regimes in original loss space.
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
    # Selection rule at time t (one-step decision)
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

        # IMM prediction (b_{t+1|t}, m_{t+1|t}, P_{t+1|t})
        b_pred, m_pred, P_pred = self._interaction_and_time_update()

        # Dynamic expert availability: if an expert becomes available for the
        # first time at this step, initialize its predicted state from the
        # population prior before computing predictive losses.
        self._apply_new_expert_prior(
            available_experts=available_experts,
            m_pred=m_pred,
            P_pred=P_pred,
            mark_joined=True,
        )

        # Predictive distribution of losses at time t+1
        mean_ell, var_ell, mu_kj, S_kj = self._predict_loss_distribution(
            phi_t, b_pred, m_pred, P_pred
        )

        # Effective risk aversion at decision time t
        if getattr(self, "lambda_risk_vec", None) is not None:
            lambda_eff = float(self.lambda_risk_vec @ self.b)
        else:
            lambda_eff = self.lambda_risk

        # Myopic risk-adjusted cost proxy
        scores = mean_ell + self.beta + lambda_eff * np.sqrt(var_ell)

        # Restrict to available experts and pick argmin
        avail_scores = scores[available_experts]
        idx = int(np.argmin(avail_scores))
        r_t = int(available_experts[idx])

        cache = {
            "phi_t": np.asarray(phi_t, dtype=float),
            "b_pred": b_pred,
            "m_pred": m_pred,
            "P_pred": P_pred,
            "mean_ell": mean_ell,
            "var_ell": var_ell,
            "mu_kj": mu_kj,
            "S_kj": S_kj,
        }
        return r_t, cache

    # --------------------------------------------------------
    # Belief update after observing losses
    # --------------------------------------------------------

    def _gaussian_logpdf(self, x: float, mean: float, var: float) -> float:
        var = max(float(var), self.eps)
        return -0.5 * (np.log(2.0 * np.pi * var) + (float(x) - float(mean))**2 / var)

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

        M, N, d = self.M, self.N, self.d
        phi_col = np.asarray(phi_t, dtype=float).reshape(d, 1)
        I_d = np.eye(d, dtype=float)

        available_experts = np.asarray(list(available_experts), dtype=int)

        # Safety: ensure that any expert that becomes available for the first
        # time (e.g., if update_beliefs is called in a custom loop) starts
        # from the population prior. In the standard online loop this is
        # already handled in select_expert, so we do not mark_joined here.
        self._apply_new_expert_prior(
            available_experts=available_experts,
            m_pred=m_pred,
            P_pred=P_pred,
            mark_joined=False,
        )

        # Expert-level state update (Kalman correction)
        if self.feedback_mode == "partial":
            j_sel = int(r_t)
            for k in range(M):
                S_k = S_kj[k, j_sel]
                mu_sel = mu_kj[k, j_sel]
                e_k = float(loss_obs - mu_sel)
                P_kj = P_pred[k, j_sel]
                K_k = (P_kj @ phi_col) / S_k
                m_pred[k, j_sel] = (m_pred[k, j_sel].reshape(d, 1) + K_k * e_k).reshape(
                    d
                )
                P_update = (I_d - K_k @ phi_col.T) @ P_kj
                P_pred[k, j_sel] = (
                    0.5 * (P_update + P_update.T) + self.eps * I_d
                )
        else:
            assert losses_full is not None
            losses_full = np.asarray(losses_full, dtype=float)
            assert losses_full.shape == (N,)
            for j in available_experts:
                ell_j = losses_full[j]
                if not np.isfinite(ell_j):
                    continue
                for k in range(M):
                    S_k = S_kj[k, j]
                    mu_pred = mu_kj[k, j]
                    e_k = float(ell_j - mu_pred)
                    P_kj = P_pred[k, j]
                    K_k = (P_kj @ phi_col) / S_k
                    m_pred[k, j] = (m_pred[k, j].reshape(d, 1) + K_k * e_k).reshape(d)
                    P_update = (I_d - K_k @ phi_col.T) @ P_kj
                    P_pred[k, j] = (
                        0.5 * (P_update + P_update.T) + self.eps * I_d
                    )

        # Regime update via Bayes' rule
        log_like = np.zeros(M, dtype=float)
        if self.feedback_mode == "partial":
            for k in range(M):
                S_k = S_kj[k, r_t]
                mu_sel = mu_kj[k, r_t]
                log_like[k] = self._gaussian_logpdf(loss_obs, mu_sel, S_k)
        else:
            for k in range(M):
                acc = 0.0
                for j in available_experts:
                    ell_j = losses_full[j]
                    if not np.isfinite(ell_j):
                        continue
                    S_val = S_kj[k, j]
                    mu_val = mu_kj[k, j]
                    acc += self._gaussian_logpdf(ell_j, mu_val, S_val)
                log_like[k] = acc

        log_post = log_like + np.log(np.maximum(b_pred, self.eps))
        m_max = float(np.max(log_post))
        post_unnorm = np.exp(log_post - m_max)
        s = post_unnorm.sum()
        if s <= 0:
            b_post = np.ones(M, dtype=float) / M
        else:
            b_post = post_unnorm / s

        self.b = b_post
        self.m = m_pred
        self.P = P_pred

    # --------------------------------------------------------
    # Horizon-H open-loop prediction of (b_{t+h|t}, m_{t+h|t}, P_{t+h|t})
    # --------------------------------------------------------

    def precompute_horizon_states(
        self,
        H: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Starting from current posterior (b_t, m_{t|t}, P_{t|t}),
        apply H times the IMM prediction to obtain:
            (b_{t+h|t}, m_{t+h|t}, P_{t+h|t})  for h=1,...,H.
        """
        b_curr = self.b.copy()
        m_curr = self.m.copy()
        P_curr = self.P.copy()

        b_list, m_list, P_list = [], [], []
        for _ in range(H):
            b_curr, m_curr, P_curr = self._interaction_and_time_update_state(
                b_curr, m_curr, P_curr
            )
            b_list.append(b_curr.copy())
            m_list.append(m_curr.copy())
            P_list.append(P_curr.copy())
        return b_list, m_list, P_list

    # --------------------------------------------------------
    # Horizon-H scheduling (planning with router-influenced contexts)
    # --------------------------------------------------------

    def plan_horizon_schedule(
        self,
        x_t: np.ndarray,
        H: int,
        experts_predict: Sequence[Callable[[np.ndarray], float]],
        context_update: Callable[[np.ndarray, float], np.ndarray],
        available_experts_per_h: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[np.ndarray], List[float]]:
        """
        Horizon-H planning: given current context x_t and belief (b_t, m_t, P_t),
        compute a greedy open-loop schedule of expert indices (r_{t+1},...,r_{t+H}),
        assuming that at each future step we update the context according to
        the chosen expert's forecast.

        This does NOT change the internal belief state; it is a planning tool.
        """
        N = self.N
        if available_experts_per_h is None:
            available_experts_per_h = [list(range(N)) for _ in range(H)]
        assert len(available_experts_per_h) == H

        # Precompute open-loop latent states (no observation updates)
        b_list, m_list, P_list = self.precompute_horizon_states(H)

        # Local copy of which experts have joined so far. This allows the
        # planner to apply the population prior to experts that first appear
        # in the provided availability sets at some future step.
        seen_future = self._has_joined.copy()

        x_curr = x_t
        schedule: List[int] = []
        contexts: List[np.ndarray] = []
        scores: List[float] = []

        for h in range(1, H + 1):
            b_h = b_list[h - 1]
            m_h = m_list[h - 1].copy()
            P_h = P_list[h - 1].copy()

            avail = np.asarray(list(available_experts_per_h[h - 1]), dtype=int)

            # Planning-time risk parameter: we use risk-neutral planning
            # (λ_plan = 0) to minimize expected cost, independently of
            # the online λ used for bandit routing.
            lambda_eff_h = 0.0

            # Apply population prior to experts that become available for the
            # first time on this future step (planning analogue of online
            # dynamic addition). We use a local mask so the router's live
            # state is not mutated by planning.
            if avail.size > 0:
                mask_new = ~seen_future[avail]
                if np.any(mask_new):
                    new_indices = avail[mask_new]
                    for j in new_indices:
                        for k in range(self.M):
                            m_h[k, j] = self.pop_mean
                            P_h[k, j] = self.pop_cov
                    seen_future[new_indices] = True
            best_score: Optional[float] = None
            best_j: Optional[int] = None
            best_x_next: Optional[np.ndarray] = None

            for j in avail:
                # Hypothetical forecast and context propagation
                y_hat_j = experts_predict[j](x_curr)
                x_next_j = context_update(x_curr, y_hat_j)
                phi_next_j = self.feature_fn(x_next_j).reshape(self.d)

                # Predictive loss distribution for expert j at step t+h
                mu_k = np.zeros(self.M, dtype=float)
                S_k = np.zeros(self.M, dtype=float)
                for k in range(self.M):
                    m_kj = m_h[k, j]
                    P_kj = P_h[k, j]
                    mu = float(phi_next_j @ m_kj)
                    S = float(phi_next_j @ (P_kj @ phi_next_j) + self.R[k, j])
                    mu_k[k] = mu
                    S_k[k] = max(S, self.eps)

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
