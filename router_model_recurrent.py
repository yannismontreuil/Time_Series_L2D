import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple

"""
Recurrent Switching Linear Dynamical System (r-SLDS) router for Learning-to-Defer
What is new:
- Time update using C[k, r_{t-1}]. Including expert choice r_prev shift the latent dynamic
- If stick_gamma and stick_kappa are provided, the transition matrix Π_t is rebuilt at each time step
  via a deterministic stick-breaking transformation on the belief-weighted mean latent state
  **Note**: No Pólya-Gamma augmentation is used here (though it is used in Linderman et al. 2017); 
            the stick-breaking is deterministic.
"""

def feature_phi(x: np.ndarray) -> np.ndarray:
    """
    Simple example feature map φ(x) ∈ R^2:
        φ(x) = [1, x_0]^T
    Assumes x is 1D (shape (1,)). Note x is the reliability modeled as a scalar.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.array([1.0, x[0]], dtype=float)


class RecurrentSLDSRouter:
    """
    Recurrent Switching Linear Dynamical System (r-SLDS) router for Learning-to-Defer
    in time series with multiple experts.

    This extends the standard SLDS by incorporating recurrent dependencies on the
    previously selected expert r_{t-1}, allowing the router to learn temporal patterns
    in expert performance and adapt more quickly to regime changes.

    Setup (per expert j and regime k):
        α_{j,t} ∈ R^d  : latent reliability state, reliability of expert j at time t
        z_t ∈ {0,...,M-1}  : discrete regime
        r_{t-1} ∈ {0,...,N-1} : previously selected expert (recurrence)
        φ_t = φ(x_t) ∈ R^d : feature vector
        ℓ_{j,t}            : observed loss

        α_{j,t+1} | (α_{j,t}, z_t = k, r_{t-1}) ~ N(A_k α_{j,t} + C_k[r_{t-1}], Q_k)
        ℓ_{j,t}   | (α_{j,t}, z_t = k) = φ_t^T α_{j,t} + v_{j,t},
                                            v_{j,t} ~ N(0, R_{k,j})

    where C_k[r_{t-1}] is a recurrent input that depends on the previous expert choice.

    This class:
      - maintains recurrent state based on past expert selections,
      - selects a single expert r_t ∈ E_t at each time t,
      - handles partial or full feedback on losses,
      - can perform horizon-H planning with recurrent context propagation.
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
        # Recurrent dynamics parameters
        C: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        lambda_risk: float | np.ndarray = 0.0,
        pop_mean: Optional[np.ndarray] = None,
        pop_cov: Optional[np.ndarray] = None,
        feedback_mode: str = "partial",
        eps: float = 1e-8,
        # Optional dynamic stick-breaking transitions (rSLDS; Linderman et al. 2017).
        # If provided, Pi becomes time-varying: π_t = stick_break(κ + Γ x̄_t),
        # where x̄_t is the belief-weighted mean latent state aggregated across experts.
        stick_gamma: Optional[np.ndarray] = None,  # shape (M, d)
        stick_kappa: Optional[np.ndarray] = None,  # shape (M,)
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

        # rSLDS stick-breaking gating params (optional). When set, they produce
        # a time-varying transition matrix Π_t via stick-breaking on the
        # aggregated continuous latent state; otherwise Π is static.
        # stick_gamma is R in paper Linderman et al. 2017.
        # stick_kappa is r in paper Linderman et al. 2017.
        if stick_gamma is None or stick_kappa is None:
            self.stick_gamma: Optional[np.ndarray] = None
            self.stick_kappa: Optional[np.ndarray] = None
        else:
            stick_gamma = np.asarray(stick_gamma, dtype=float)
            stick_kappa = np.asarray(stick_kappa, dtype=float)
            assert stick_gamma.shape == (self.M, self.d)
            assert stick_kappa.shape == (self.M,)
            self.stick_gamma = stick_gamma
            self.stick_kappa = stick_kappa

        # Recurrent dynamics: C[k, j, :] is the input bias when transitioning
        # from expert j under regime k. Shape: (M, N, d)
        if C is None: # default to zero recurrent input, then no recurrence
            C = np.zeros((self.M, self.N, self.d), dtype=float)
        else:
            C = np.asarray(C, dtype=float)
            assert C.shape == (self.M, self.N, self.d)
        self.C = C

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

        # Recurrent state: track the previously selected expert
        self.r_prev: Optional[int] = None

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

        self._has_joined[:] = False
        self.r_prev = None

        self.m = np.zeros((self.M, self.N, self.d), dtype=float)
        self.P = np.zeros((self.M, self.N, self.d, self.d), dtype=float)
        for k in range(self.M):
            for j in range(self.N):
                self.m[k, j] = self.pop_mean
                self.P[k, j] = self.pop_cov

    # --------------------------------------------------------
    # Dynamic stick-breaking transition (Linderman et al. 2017)
    # --------------------------------------------------------

    def _stick_breaking(self, nu: np.ndarray) -> np.ndarray:
        """
        Convert stick-breaking logits ν ∈ R^M to a probability vector π ∈ [0,1]^{M-1}.
        π^(k) (ν) = σ(ν_k) ∏_{j<k} (1-σ(ν_j)), with remainder on the last stick.
        σ is the logistic function e^x / (1+e^x).
        """
        M = self.M
        assert nu.shape == (M,) # the logits for M regimes
        sigma = 1.0 / (1.0 + np.exp(-nu))
        pi = np.zeros(M, dtype=float)
        remaining = 1.0
        for k in range(M):
            if k == M - 1:
                pi[k] = remaining
            else:
                pi[k] = remaining * sigma[k]
                remaining = remaining * (1.0 - sigma[k])
        s = pi.sum()
        if s <= 0:
            return np.ones(M, dtype=float) / M
        # In theory s should be 1.0 here, but numerical issues may arise.
        return pi / s

    def _compute_transition_matrix(self, b: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        If stick-breaking params are provided, build a time-varying Π_t that depends on
        the belief-weighted mean latent state x̄_t (aggregated across experts).
        Otherwise return the static Π.
        """
        if self.stick_gamma is None or self.stick_kappa is None:
            return self.Pi

        M, N, d = self.M, self.N, self.d
        b = np.asarray(b, dtype=float).reshape(M)
        m = np.asarray(m, dtype=float).reshape(M, N, d)

        # Aggregate latent state: belief-weighted across regimes, then average across experts.
        x_bar = (b.reshape(M, 1, 1) * m).sum(axis=0).mean(axis=0)  # shape (d,)
        # ν = κ + Γ x̄_t
        # same as the ν = r + R x̄_t in Linderman et al. 2017
        nu = self.stick_kappa + self.stick_gamma @ x_bar
        pi_vec = self._stick_breaking(nu)
        # Same row for all current regimes (standard rSLDS gating on x_t).
        Pi_dyn = np.tile(pi_vec.reshape(1, M), (M, 1))
        return Pi_dyn

    # --------------------------------------------------------
    # IMM interaction + recurrent time update
    # --------------------------------------------------------

    def _interaction_and_time_update(self):
        return self._interaction_and_time_update_state(
            self.b, self.m, self.P, self.r_prev
        )

    def _interaction_and_time_update_state(
        self,
        b: np.ndarray,
        m: np.ndarray,
        P: np.ndarray,
        r_prev: Optional[int] = None,
    ):
        M, N, d = self.M, self.N, self.d

        b = np.asarray(b, dtype=float).reshape(M)
        m = np.asarray(m, dtype=float).reshape(M, N, d)
        P = np.asarray(P, dtype=float).reshape(M, N, d, d)

        # Predict regime distribution using static Π or dynamic stick-breaking Π_t.
        Pi_t = self._compute_transition_matrix(b, m)
        b_pred = b @ Pi_t
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
                mu[:, k] = Pi_t[:, k] * b
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
                m0 = np.zeros(d, dtype=float)
                for i in range(M):
                    m0 += mu[i, k] * m[i, j]
                m_mixed[k, j] = m0

                P0 = np.zeros((d, d), dtype=float)
                for i in range(M):
                    diff = (m[i, j] - m0).reshape(d, 1)
                    P0 += mu[i, k] * (P[i, j] + diff @ diff.T)
                P0 += self.eps * np.eye(d)
                P_mixed[k, j] = P0

        # Time update with recurrent dynamics: α_{j,t+1|t} = A_k α_{j,t|t} + C_k[r_prev]
        m_pred = np.zeros_like(m_mixed)
        P_pred = np.zeros_like(P_mixed)
        for k in range(M):
            A_k = self.A[k]
            Q_k = self.Q[k]
            for j in range(N):
                # Recurrent input: if r_prev is set, use C_k[r_prev] as an additive bias
                c_input = np.zeros(d, dtype=float)
                if r_prev is not None and 0 <= r_prev < N:
                    c_input = self.C[k, r_prev]
                
                m_pred[k, j] = A_k @ m_mixed[k, j] + c_input
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

        mu_kj = np.zeros((M, N), dtype=float)
        S_kj = np.zeros((M, N), dtype=float)
        for k in range(M):
            for j in range(N):
                m_kj = m_pred[k, j]
                P_kj = P_pred[k, j]
                mu = float(phi_t @ m_kj)
                S = float(phi_t @ (P_kj @ phi_t) + self.R[k, j])
                S_kj[k, j] = max(S, self.eps)
                mu_kj[k, j] = mu

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
        select r_t ∈ E_t using the myopic risk-adjusted score with
        recurrent dynamics from r_{t-1}.
        """
        avail_arr = np.asarray(list(available_experts), dtype=int)
        phi_t = self.feature_fn(x_t)

        # IMM prediction with recurrent input from r_prev
        b_pred, m_pred, P_pred = self._interaction_and_time_update()

        # Dynamic expert availability
        self._apply_new_expert_prior(
            available_experts=avail_arr,
            m_pred=m_pred,
            P_pred=P_pred,
            mark_joined=True,
        )

        # Predictive distribution of losses
        mean_ell, var_ell, mu_kj, S_kj = self._predict_loss_distribution(
            phi_t, b_pred, m_pred, P_pred
        )

        # Effective risk aversion at decision time t
        lambda_vec = getattr(self, "lambda_risk_vec", None)
        lambda_eff = (
            float(lambda_vec @ self.b)
            if lambda_vec is not None
            else self.lambda_risk
        )

        # Myopic risk-adjusted cost proxy
        scores = mean_ell + self.beta + lambda_eff * np.sqrt(var_ell)

        # Restrict to available experts and pick argmin
        avail_scores = scores[avail_arr]
        idx = int(np.argmin(avail_scores))
        r_t = int(avail_arr[idx])

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

        avail_arr = np.asarray(list(available_experts), dtype=int)

        self._apply_new_expert_prior(
            available_experts=avail_arr,
            m_pred=m_pred,
            P_pred=P_pred,
            mark_joined=False,
        )

        # Expert-level state update (Kalman correction)
        losses_full_arr: Optional[np.ndarray] = None
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
            if losses_full is None:
                raise ValueError("losses_full must be provided in full feedback mode")
            losses_full_arr = np.asarray(losses_full, dtype=float)
            if losses_full_arr.shape != (N,):
                raise ValueError("losses_full must have shape (N,)")
            for j in avail_arr:
                ell_j = float(losses_full_arr[j])
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
            assert losses_full_arr is not None
            for k in range(M):
                acc = 0.0
                for j in avail_arr:
                    ell_j = float(losses_full_arr[j])
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
        
        # Update recurrent state: remember which expert we just selected
        self.r_prev = int(r_t)

    # --------------------------------------------------------
    # Horizon-H open-loop prediction with recurrent state tracking
    # --------------------------------------------------------

    def precompute_horizon_states(
        self,
        H: int,
        expert_sequence: Optional[List[int]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Starting from current posterior (b_t, m_{t|t}, P_{t|t}),
        apply H times the IMM prediction with recurrent dynamics, optionally
        conditioned on a provided expert sequence.

        If expert_sequence is None, use zeros for recurrent input (no expert influence).
        If expert_sequence is provided (length H), use r_{t+h-1} to influence α at step h.
        """
        b_curr = self.b.copy()
        m_curr = self.m.copy()
        P_curr = self.P.copy()

        b_list, m_list, P_list = [], [], []
        r_prev = self.r_prev  # Start with current recurrent state

        for h in range(H):
            b_curr, m_curr, P_curr = self._interaction_and_time_update_state(
                b_curr, m_curr, P_curr, r_prev
            )
            b_list.append(b_curr.copy())
            m_list.append(m_curr.copy())
            P_list.append(P_curr.copy())

            # Update recurrent state for next iteration
            if expert_sequence is not None and h < len(expert_sequence):
                r_prev = expert_sequence[h]
            else:
                r_prev = None

        return b_list, m_list, P_list

    # --------------------------------------------------------
    # Horizon-H scheduling with recurrent context propagation
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
        Horizon-H planning with recurrent dynamics: given current context x_t,
        belief (b_t, m_t, P_t), and previous expert r_{t-1}, compute a greedy
        open-loop schedule (r_{t+1},...,r_{t+H}) that accounts for recurrent
        state propagation.

        This does NOT change the internal belief state; it is a planning tool.
        """
        N = self.N
        if available_experts_per_h is None:
            available_experts_per_h = [list(range(N)) for _ in range(H)]
        assert len(available_experts_per_h) == H

        seen_future = self._has_joined.copy()

        x_curr = x_t
        schedule: List[int] = []
        contexts: List[np.ndarray] = []
        scores: List[float] = []

        # Local recurrent state for planning
        r_plan = self.r_prev

        for h in range(1, H + 1):
            # Time update with recurrent input from r_plan
            b_curr = self.b.copy()
            m_curr = self.m.copy()
            P_curr = self.P.copy()

            # Apply h-1 prediction steps to reach time t+h
            for _ in range(h - 1):
                b_curr, m_curr, P_curr = self._interaction_and_time_update_state(
                    b_curr, m_curr, P_curr, r_plan
                )

            b_h = b_curr
            m_h = m_curr.copy()
            P_h = P_curr.copy()

            avail = np.asarray(list(available_experts_per_h[h - 1]), dtype=int)
            if avail.size == 0:
                raise ValueError(f"No available experts for planning step {h}")

            # Planning-time risk parameter: risk-neutral
            lambda_eff_h = 0.0

            # Apply population prior to new experts
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

            assert best_j is not None and best_x_next is not None and best_score is not None
            schedule.append(best_j)
            contexts.append(best_x_next)
            scores.append(float(best_score))

            # Update planning state for next iteration
            x_curr = best_x_next
            r_plan = best_j

        return schedule, contexts, scores
