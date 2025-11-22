import numpy as np


class SLDSIMMRouter:
    """
    Switching Linear Dynamical System (SLDS) + Interacting Multiple Model (IMM)
    router for Learning-to-Defer in time series with multiple experts.

    Focus:
      - Single-expert selection at each time t (no top-k),
      - Partial or full feedback on expert losses,
      - Optional horizon-H planning using expert-driven context updates.

    Latent model (per expert j and regime k):
        alpha_{j,t} in R^d  : reliability state
        z_t in {0,...,M-1}  : discrete regime
        phi_t = phi(x_t)    : feature vector in R^d
        ell_{j,t}           : observed loss

        alpha_{j,t+1} | (alpha_{j,t}, z_t=k)
            ~ N(A_k alpha_{j,t}, Q_k)

        ell_{j,t} | (alpha_{j,t}, z_t=k)
            = phi_t^T alpha_{j,t} + v_{j,t},  v_{j,t} ~ N(0, R_{k,j})

    The IMM approximates P(z_t, alpha_{j,t} | I_t) by:
        b_t(k) = P(z_t=k | I_t)
        alpha_{j,t} | {z_t=k, I_t} ~ N(m_{j,t|t}^{(k)}, P_{j,t|t}^{(k)}).
    """

    def __init__(
        self,
        num_experts: int,
        num_regimes: int,
        state_dim: int,
        feature_fn,
        A: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Pi: np.ndarray,
        beta: np.ndarray | None = None,
        lambda_risk: float = 0.0,
        pop_mean: np.ndarray | None = None,
        pop_cov: np.ndarray | None = None,
        feedback_mode: str = "partial",
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        num_experts : int
            Number of experts N.
        num_regimes : int
            Number of regimes M.
        state_dim : int
            Dimension d of alpha_{j,t}. The feature map phi(x) must map to R^d.
        feature_fn : callable
            Feature map phi(x): context -> np.ndarray of shape (d,).
        A : np.ndarray
            State transition matrices A_k, shape (M, d, d).
        Q : np.ndarray
            Process noise covariances Q_k, shape (M, d, d).
        R : np.ndarray
            Observation noise variances R_{k,j}, shape (M, N).
        Pi : np.ndarray
            Regime transition matrix Pi[i, k] = P(z_t = k | z_{t-1} = i),
            shape (M, M). Each row must sum to 1.
        beta : np.ndarray, optional
            Consultation costs beta_j, shape (N,). Defaults to zero.
        lambda_risk : float
            Risk parameter lambda in the one-step score J_{j,t}.
        pop_mean : np.ndarray, optional
            Prior mean for alpha_{j,0}, shape (d,). Defaults to zero.
        pop_cov : np.ndarray, optional
            Prior covariance for alpha_{j,0}, shape (d,d). Defaults to I_d.
        feedback_mode : {"partial", "full"}
            "partial": only loss of selected expert is observed each step.
            "full"   : losses of all available experts are observed.
        eps : float
            Small positive constant for numerical stability.
        """
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

        self.lambda_risk = float(lambda_risk)

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

        self.reset_beliefs()

    # ------------------------------------------------------------------
    # Initialization of belief state
    # ------------------------------------------------------------------
    def reset_beliefs(self, b0: np.ndarray | None = None):
        """
        Reset belief to time t=0: b_0, m_{j,0|0}^{(k)}, P_{j,0|0}^{(k)}.

        Parameters
        ----------
        b0 : np.ndarray, optional
            Initial regime distribution, shape (M,).
            If None, set to uniform.
        """
        if b0 is None:
            self.b = np.ones(self.M, dtype=float) / self.M
        else:
            b0 = np.asarray(b0, dtype=float)
            assert b0.shape == (self.M,)
            s = b0.sum()
            assert s > 0
            self.b = b0 / s

        # m[k,j,:]  ≈ E[alpha_{j,0} | z_0=k, I_0]
        # P[k,j,:,:] ≈ Cov[alpha_{j,0} | z_0=k, I_0]
        self.m = np.zeros((self.M, self.N, self.d), dtype=float)
        self.P = np.zeros((self.M, self.N, self.d, self.d), dtype=float)
        for k in range(self.M):
            for j in range(self.N):
                self.m[k, j] = self.pop_mean
                self.P[k, j] = self.pop_cov

    # ------------------------------------------------------------------
    # IMM prediction: (b_t, m_t, P_t) -> (b_{t+1|t}, m_{t+1|t}, P_{t+1|t})
    # ------------------------------------------------------------------
    def _interaction_and_time_update(self):
        """Prediction step using the *current* belief (self.b, self.m, self.P)."""
        return self._interaction_and_time_update_state(self.b, self.m, self.P)

    def _interaction_and_time_update_state(self, b, m, P):
        """
        Pure (stateless) IMM interaction + time update:
            (b, m, P) -> (b_pred, m_pred, P_pred).

        Parameters
        ----------
        b : np.ndarray, shape (M,)
        m : np.ndarray, shape (M, N, d)
        P : np.ndarray, shape (M, N, d, d)

        Returns
        -------
        b_pred : np.ndarray, shape (M,)
        m_pred : np.ndarray, shape (M, N, d)
        P_pred : np.ndarray, shape (M, N, d, d)
        """
        M, N, d = self.M, self.N, self.d

        b = np.asarray(b, dtype=float).reshape(M)
        m = np.asarray(m, dtype=float).reshape(M, N, d)
        P = np.asarray(P, dtype=float).reshape(M, N, d, d)

        # Regime prediction: b_{t+1|t}(k) = sum_i b_t(i) Pi[i,k]
        b_pred = b @ self.Pi
        s = b_pred.sum()
        if s <= 0:
            b_pred = np.ones(M, dtype=float) / M
        else:
            b_pred /= s

        # Mixing probabilities mu[i,k] = P(z_t=i | z_{t+1}=k, I_t)
        mu = np.zeros((M, M), dtype=float)
        for k in range(M):
            denom = b_pred[k]
            if denom <= self.eps:
                mu[:, k] = 1.0 / M
            else:
                mu[:, k] = self.Pi[:, k] * b
                s = mu[:, k].sum()
                if s <= self.eps:
                    mu[:, k] = 1.0 / M
                else:
                    mu[:, k] /= s

        # Mixed means and covariances at time t
        m_mixed = np.zeros_like(m)
        P_mixed = np.zeros_like(P)
        for k in range(M):
            for j in range(N):
                # Mixed mean
                m0 = np.zeros(d, dtype=float)
                for i in range(M):
                    m0 += mu[i, k] * m[i, j]
                m_mixed[k, j] = m0

                # Mixed covariance
                P0 = np.zeros((d, d), dtype=float)
                for i in range(M):
                    diff = (m[i, j] - m0).reshape(d, 1)
                    P0 += mu[i, k] * (P[i, j] + diff @ diff.T)
                P0 += self.eps * np.eye(d)
                P_mixed[k, j] = P0

        # Time update: apply A_k, Q_k
        m_pred = np.zeros_like(m_mixed)
        P_pred = np.zeros_like(P_mixed)
        for k in range(M):
            A_k = self.A[k]
            Q_k = self.Q[k]
            for j in range(N):
                m_pred[k, j] = A_k @ m_mixed[k, j]
                P_pred[k, j] = A_k @ P_mixed[k, j] @ A_k.T + Q_k
                P_pred[k, j] = 0.5 * (P_pred[k, j] + P_pred[k, j].T) + self.eps * np.eye(d)

        return b_pred, m_pred, P_pred

    # ------------------------------------------------------------------
    # Predictive distribution of losses at time t (given I_{t-1})
    # ------------------------------------------------------------------
    def _predict_loss_distribution(self, phi_t, b_pred, m_pred, P_pred):
        """
        For each expert j, compute the mixture predictive mean and variance:
            ell_{j,t} | I_{t-1}  ~ mixture of Gaussians.

        Parameters
        ----------
        phi_t : np.ndarray, shape (d,)
        b_pred : np.ndarray, shape (M,)
        m_pred : np.ndarray, shape (M, N, d)
        P_pred : np.ndarray, shape (M, N, d, d)

        Returns
        -------
        mean_ell : np.ndarray, shape (N,)
            E[ell_{j,t} | I_{t-1}] for each expert j.
        var_ell : np.ndarray, shape (N,)
            Var(ell_{j,t} | I_{t-1}) for each expert j.
        mu_kj : np.ndarray, shape (M, N)
            Regime-conditional means mu_{j,t}^{(k)}.
        S_kj : np.ndarray, shape (M, N)
            Regime-conditional variances S_{j,t}^{(k)}.
        """
        M, N, d = self.M, self.N, self.d
        phi_t = np.asarray(phi_t, dtype=float).reshape(d)
        assert phi_t.shape == (d,)

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

    # ------------------------------------------------------------------
    # One-step selection: r_t in argmin_j J_{j,t}
    # ------------------------------------------------------------------
    def select_expert(self, x_t, available_experts):
        """
        Select an expert at time t given x_t and current belief.

        This computes the one-step, risk-adjusted scores
            J_{j,t} = E[ell_{j,t} | I_{t-1}] + beta_j
                      + lambda * sqrt(Var(ell_{j,t} | I_{t-1}))

        Parameters
        ----------
        x_t : any
            Context at time t. Passed to feature_fn to obtain phi_t.
        available_experts : sequence of int
            Subset of experts that are available at time t.

        Returns
        -------
        r_t : int
            Index of selected expert.
        cache : dict
            Predictive quantities needed for the measurement update,
            including (b_pred, m_pred, P_pred, mu_kj, S_kj, phi_t, ...).
        """
        available_experts = np.asarray(list(available_experts), dtype=int)
        phi_t = self.feature_fn(x_t)

        b_pred, m_pred, P_pred = self._interaction_and_time_update()
        mean_ell, var_ell, mu_kj, S_kj = self._predict_loss_distribution(
            phi_t, b_pred, m_pred, P_pred
        )

        scores = mean_ell + self.beta + self.lambda_risk * np.sqrt(var_ell)

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

    # ------------------------------------------------------------------
    # Measurement update + regime update at time t
    # ------------------------------------------------------------------
    def _gaussian_logpdf(self, x, mean, var):
        var = max(float(var), self.eps)
        return -0.5 * (np.log(2.0 * np.pi * var) + (float(x) - float(mean))**2 / var)

    def update_beliefs(self, r_t, loss_obs, losses_full, available_experts, cache):
        """
        Update the belief at time t using the observed loss(es).

        Parameters
        ----------
        r_t : int
            Selected expert at time t.
        loss_obs : float
            Observed loss ell_{r_t,t}. Used in partial-feedback mode.
        losses_full : np.ndarray or None
            In full-feedback mode, array of shape (N,) with all expert losses
            at time t. Entries for unobserved experts may be np.nan.
            Ignored in partial-feedback mode.
        available_experts : sequence of int
            Set of experts for which feedback is available at time t.
        cache : dict
            Output of select_expert(...) for the same time t.

        Side effect
        -----------
        Updates internal belief (self.b, self.m, self.P).
        """
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

        # Continuous-state update
        if self.feedback_mode == "partial":
            j_sel = int(r_t)
            for k in range(M):
                S_k = S_kj[k, j_sel]
                mu_sel = mu_kj[k, j_sel]
                e_k = float(loss_obs - mu_sel)
                P_kj = P_pred[k, j_sel]
                K_k = (P_kj @ phi_col) / S_k
                m_pred[k, j_sel] = (m_pred[k, j_sel].reshape(d, 1) + K_k * e_k).reshape(d)
                P_update = (I_d - K_k @ phi_col.T) @ P_kj
                P_pred[k, j_sel] = 0.5 * (P_update + P_update.T) + self.eps * I_d
        else:
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
                    P_pred[k, j] = 0.5 * (P_update + P_update.T) + self.eps * I_d

        # Regime probability update
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

    # ------------------------------------------------------------------
    # Horizon-H prediction of latent states (no future observations)
    # ------------------------------------------------------------------
    def precompute_horizon_states(self, H: int):
        """
        Compute (b_{t+h|t}, m_{t+h|t}, P_{t+h|t}) for h = 1,...,H by iterating
        the interaction + time update, starting from the current filtered
        belief at time t. Does NOT modify the internal state.
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

    # ------------------------------------------------------------------
    # Horizon-H scheduling forecast via expert-driven context updates
    # ------------------------------------------------------------------
    def plan_horizon_schedule(
        self,
        x_t,
        H: int,
        experts_predict,
        context_update,
        available_experts_per_h=None,
    ):
        """
        Horizon-H planning using expert-driven context updates.

        At planning time t we:
          - Start from current context x_t and belief (b_t, m_t, P_t).
          - Precompute (b_{t+h|t}, m_{t+h|t}, P_{t+h|t}) for h=1,...,H.
          - For h = 1,...,H (greedy in h):
              * For each candidate expert j in K_{t+h}, build
                    y_hat_j = experts_predict[j](x_curr)
                    x_next_j = context_update(x_curr, y_hat_j)
                and evaluate its risk-adjusted expected loss at horizon h.
              * Select r_{t+h} = argmin_j J_{j,t+h} and update x_curr := x_next_{r_{t+h}}.

        Parameters
        ----------
        x_t : any
            Current context at time t.
        H : int
            Planning horizon.
        experts_predict : sequence of callables
            experts_predict[j](x) -> y_hat_j (expert j's forecast).
        context_update : callable
            context_update(x, y_hat) -> x_next, the update rule Psi.
        available_experts_per_h : list or None
            If not None, list of length H; entry h-1 is the set of experts
            available at time t+h. If None, all experts are assumed available.

        Returns
        -------
        schedule : list of int
            Planned experts r_{t+1}, ..., r_{t+H}.
        contexts : list
            Planned surrogate contexts x_{t+1}, ..., x_{t+H}.
        scores : list of float
            Corresponding risk-adjusted scores J_{r_{t+h}, t+h}.
        """
        N = self.N
        if available_experts_per_h is None:
            available_experts_per_h = [list(range(N)) for _ in range(H)]
        assert len(available_experts_per_h) == H

        b_list, m_list, P_list = self.precompute_horizon_states(H)

        x_curr = x_t
        schedule = []
        contexts = []
        scores = []

        for h in range(1, H + 1):
            b_h = b_list[h - 1]
            m_h = m_list[h - 1]
            P_h = P_list[h - 1]

            avail = np.asarray(list(available_experts_per_h[h - 1]), dtype=int)
            best_score = None
            best_j = None
            best_x_next = None

            for j in avail:
                # Hypothetical context if we select j at this planning step
                y_hat_j = experts_predict[j](x_curr)
                x_next_j = context_update(x_curr, y_hat_j)
                phi_next_j = self.feature_fn(x_next_j).reshape(self.d)

                # Regime-conditional mean/variance of ell_{j,t+h}
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

                score_j = mean_ell_j + self.beta[j] + self.lambda_risk * np.sqrt(var_ell_j)

                if (best_score is None) or (score_j < best_score):
                    best_score = score_j
                    best_j = int(j)
                    best_x_next = x_next_j

            schedule.append(best_j)
            contexts.append(best_x_next)
            scores.append(float(best_score))
            x_curr = best_x_next

        return schedule, contexts, scores
