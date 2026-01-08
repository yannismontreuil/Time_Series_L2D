import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class FactorizedSLDS:
    """
    Factorized Switching Linear Dynamical System for bandit routing.

    State representation (mode-conditioned beliefs):
      - mu_g: (M, d_g) mean of global factor per mode
      - Sigma_g: (M, d_g, d_g) covariance of global factor per mode
      - mu_u: dict[k] -> (M, d_phi) idiosyncratic mean per mode for expert k
      - Sigma_u: dict[k] -> (M, d_phi, d_phi) idiosyncratic cov per mode
      - w: (M,) regime weights (probabilities)
      - registry: list of active expert IDs

    Observation model (scalar residual):
      y = phi(x)^T (B_k g + u_k) + eps,  eps ~ N(0, R)

    The class implements predict_step, select_action, update_step, manage_registry.
    Feature function feature_fn(x) must return a vector phi of length d_phi.
    """

    def __init__(
        self,
        M: int, # number of discrete latent states
        d_g: int,
        d_phi: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        B_dict: Optional[Dict[int, np.ndarray]] = None, # expert-specific loading matrices B_k
        beta: Optional[Dict[int, float]] = None,
        R: float = 1.0,
        A_g: Optional[np.ndarray] = None,
        A_u: Optional[np.ndarray] = None,
        Q_g: Optional[np.ndarray] = None,
        Q_u: Optional[np.ndarray] = None,
        Delta_max: int = 100,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.M = int(M)
        self.d_g = int(d_g)
        self.d_phi = int(d_phi)
        self.feature_fn = feature_fn

        # Mode-conditioned global factor beliefs
        self.mu_g = np.zeros((self.M, self.d_g), dtype=float)
        self.Sigma_g = np.stack([np.eye(self.d_g, dtype=float) for _ in range(self.M)])

        # Idiosyncratic per-expert beliefs stored as maps to mode-conditioned arrays
        self.mu_u: Dict[int, np.ndarray] = {}
        self.Sigma_u: Dict[int, np.ndarray] = {}

        # Transition / dynamics parameters per mode
        self.A_g = (
            np.stack([np.eye(self.d_g, dtype=float) for _ in range(self.M)])
            if A_g is None
            else np.asarray(A_g, dtype=float)
        )
        self.A_u = (
            np.stack([np.eye(self.d_phi, dtype=float) for _ in range(self.M)])
            if A_u is None
            else np.asarray(A_u, dtype=float)
        )
        # Process noise
        self.Q_g = (
            np.stack([np.eye(self.d_g, dtype=float) * 1e-3 for _ in range(self.M)])
            if Q_g is None
            else np.asarray(Q_g, dtype=float)
        )
        self.Q_u = (
            np.stack([np.eye(self.d_phi, dtype=float) * 1e-3 for _ in range(self.M)])
            if Q_u is None
            else np.asarray(Q_u, dtype=float)
        )

        # Observation noise
        self.R = float(R)

        # Expert-specific B matrices mapping g to feature-space (d_phi x d_g)
        self.B_dict = {} if B_dict is None else {k: np.asarray(v, dtype=float) for k, v in B_dict.items()}

        # Consultation cost per expert
        self.beta = {} if beta is None else {k: float(v) for k, v in beta.items()}

        # Regime weights
        self.w = np.ones(self.M, dtype=float) / float(self.M)

        # Registry and activity bookkeeping
        self.registry: List[int] = list(self.B_dict.keys())
        self.last_selected_step: Dict[int, int] = {k: 0 for k in self.registry}
        self.current_step = 0
        self.Delta_max = int(Delta_max)

        # Attention parameters for Pi_t (learnable linear projections)
        d_q = max(1, self.d_phi)
        self.W_q = rng.normal(scale=0.1, size=(self.M, d_q))
        self.W_k = rng.normal(scale=0.1, size=(self.M, d_q))

        # Population prior for new experts (used at birth)
        self.mu_u0 = np.zeros(self.d_phi, dtype=float)
        self.Sigma_u0 = np.eye(self.d_phi, dtype=float) * 0.1

    # ---------------------- Utilities ----------------------
    def _softmax_rows(self, X: np.ndarray) -> np.ndarray:
        X = X - np.max(X, axis=1, keepdims=True)
        expX = np.exp(X)
        return expX / np.sum(expX, axis=1, keepdims=True)

    # ---------------------- Predict (time update) ----------------------
    # Algorithm 4 IMM: Context-Dependent IMM Mixing (Moment Matching)
    def predict_step(self, context_x: np.ndarray) -> None:
        """
        Time update (IMM): compute Pi_t from context_x, mix mode-conditioned beliefs,
        then propagate each mixed belief forward under linear dynamics.
        """
        phi = np.asarray(self.feature_fn(context_x), dtype=float).reshape(self.d_phi)

        # 1) Compute mode transition matrix Pi (M x M) via softmax attention
        # q_i = W_q[i] . phi, k_j = W_k[j] . phi  => score_{ij} = q_i dot k_j
        q = (self.W_q @ phi).reshape(self.M, -1)  # (M, d_q)
        k = (self.W_k @ phi).reshape(self.M, -1)  # (M, d_q)
        scores = q @ k.T  # (M, M)
        Pi = self._softmax_rows(scores)

        # 2) IMM mixing: compute mixing weights mu^{i|j} and mix means/covariances
        w_prev = self.w.copy()
        # predictive regime weights w_pred[j] = sum_i w_prev[i] * Pi[i,j]
        w_pred = Pi.T @ w_prev  # (M,)
        # avoid division by zero
        w_pred = np.maximum(w_pred, 1e-12)

        mu_ij = (w_prev[:, None] * Pi) / w_pred[None, :]
        # mu_ij shape (M_i, M_j) where mu_ij[i,j] = prob(i at t-1 | j at t)

        # Mix global factor beliefs for each target mode j
        mixed_mu_g = np.zeros_like(self.mu_g)
        mixed_Sigma_g = np.zeros_like(self.Sigma_g)
        for j in range(self.M):
            # weighted means
            for i in range(self.M):
                w_ij = mu_ij[i, j]
                mixed_mu_g[j] += w_ij * self.mu_g[i]
            # weighted covariances (moment matching)
            for i in range(self.M):
                w_ij = mu_ij[i, j]
                diff = self.mu_g[i] - mixed_mu_g[j]
                mixed_Sigma_g[j] += w_ij * (self.Sigma_g[i] + np.outer(diff, diff))

        # Mix idiosyncratic states for each expert where present
        mixed_mu_u: Dict[int, np.ndarray] = {}
        mixed_Sigma_u: Dict[int, np.ndarray] = {}
        for k in self.registry:
            mu_modes = self.mu_u.get(k)
            Sigma_modes = self.Sigma_u.get(k)
            if mu_modes is None or Sigma_modes is None:
                # initialize if missing
                mu_modes = np.zeros((self.M, self.d_phi), dtype=float)
                Sigma_modes = np.stack([np.eye(self.d_phi, dtype=float) * 0.1 for _ in range(self.M)])
            mixed_mu_u[k] = np.zeros_like(mu_modes)
            mixed_Sigma_u[k] = np.zeros_like(Sigma_modes)
            for j in range(self.M):
                for i in range(self.M):
                    w_ij = mu_ij[i, j]
                    mixed_mu_u[k][j] += w_ij * mu_modes[i]
                for i in range(self.M):
                    w_ij = mu_ij[i, j]
                    diff = mu_modes[i] - mixed_mu_u[k][j]
                    mixed_Sigma_u[k][j] += w_ij * (Sigma_modes[i] + np.outer(diff, diff))

        # 3) Propagate mixed states forward under linear dynamics for each mode
        for m in range(self.M):
            # global
            self.mu_g[m] = self.A_g[m] @ mixed_mu_g[m]
            self.Sigma_g[m] = self.A_g[m] @ mixed_Sigma_g[m] @ self.A_g[m].T + self.Q_g[m]
            # idiosyncratic per expert
            for k in self.registry:
                self.mu_u[k][m] = self.A_u[m] @ mixed_mu_u[k][m]
                self.Sigma_u[k][m] = self.A_u[m] @ mixed_Sigma_u[k][m] @ self.A_u[m].T + self.Q_u[m]

        # update predictive regime weights
        self.w = w_pred
        self.current_step += 1
        # increment inactivity counters
        for k in list(self.last_selected_step.keys()):
            self.last_selected_step[k] += 1

    # ---------------------- Action selection (IDS) ----------------------
    # Algorithm 5 PREDICT AND SCORE: Mode-Wise Prediction and Myopic Costin
    def select_action(self, context_x: np.ndarray, active_experts: Optional[Sequence[int]] = None) -> int:
        """
        Select expert via Information-Directed Sampling (IDS) as described.
        Returns selected expert id.
        """
        if active_experts is None:
            active_experts = list(self.registry)
        phi = np.asarray(self.feature_fn(context_x), dtype=float).reshape(self.d_phi)

        best_k = None
        best_score = np.inf
        for k in active_experts:
            Bk = self.B_dict.get(k)
            if Bk is None:
                continue
            # compute predictive mean/variance marginalizing modes
            mu_y_modes = np.zeros(self.M, dtype=float)
            var_y_modes = np.zeros(self.M, dtype=float)
            for m in range(self.M):
                mu_g_m = self.mu_g[m]
                Sigma_g_m = self.Sigma_g[m]
                mu_u_m = self.mu_u[k][m]
                Sigma_u_m = self.Sigma_u[k][m]
                mu_y = phi @ (Bk @ mu_g_m + mu_u_m)
                # signal var from global factor
                sig_g = float(phi @ (Bk @ (Sigma_g_m @ (Bk.T @ phi))))
                # idiosyncratic var
                sig_u = float(phi @ (Sigma_u_m @ phi))
                var_y = sig_g + sig_u + self.R
                mu_y_modes[m] = mu_y
                var_y_modes[m] = var_y
            # marginal predictive mean/var across modes using weights w
            mu_y_marg = float(self.w @ mu_y_modes)
            var_y_marg = float(self.w @ (var_y_modes + mu_y_modes ** 2) - mu_y_marg ** 2)

            # Myopic expected loss: use variance + cost beta_k
            beta_k = self.beta.get(k, 0.0)
            regret = var_y_marg + beta_k

            # Information gain approximated using signal variance about g
            # compute expected signal variance across modes
            signal_var_modes = np.zeros(self.M, dtype=float)
            for m in range(self.M):
                Sigma_g_m = self.Sigma_g[m]
                sig_g = float(phi @ (Bk @ (Sigma_g_m @ (Bk.T @ phi))))
                signal_var_modes[m] = sig_g
            SignalVar = float(self.w @ signal_var_modes)
            NoiseVar = max(1e-12, float(self.w @ (var_y_modes - signal_var_modes)))
            IG = 0.5 * np.log(1.0 + SignalVar / NoiseVar)
            if IG <= 1e-12:
                score = np.inf
            else:
                score = (regret ** 2) / IG
            if score < best_score:
                best_score = score
                best_k = k
        return int(best_k) if best_k is not None else -1

    # ---------------------- Measurement update (partial feedback) ----------------------
    # Algorithm 6 CORRECT: Queried Kalman Update and Mode Posterior
    def update_step(self, selected_expert_id: int, observed_residual: float, context_x: Optional[np.ndarray] = None) -> None:
        """
        Perform measurement update only for the selected expert across modes.
        Update mode weights using observation likelihoods.
        observed_residual is scalar y_obs - prediction (or the observed residual directly).
        """
        k = int(selected_expert_id)
        if k not in self.registry:
            # unknown expert: birth it first
            self._birth_expert(k)
        phi = None
        if context_x is not None:
            phi = np.asarray(self.feature_fn(context_x), dtype=float).reshape(self.d_phi)

        like = np.zeros(self.M, dtype=float)
        for m in range(self.M):
            # joint state [g; u_k] mean and covariance
            mu_g = self.mu_g[m]
            Sigma_g = self.Sigma_g[m]
            mu_u = self.mu_u[k][m]
            Sigma_u = self.Sigma_u[k][m]
            # joint mean
            joint_mu = np.concatenate([mu_g, mu_u])
            # joint cov
            top = np.concatenate([Sigma_g, np.zeros((self.d_g, self.d_phi))], axis=1)
            bottom = np.concatenate([np.zeros((self.d_phi, self.d_g)), Sigma_u], axis=1)
            joint_Sigma = np.concatenate([top, bottom], axis=0)
            # observation matrix H: maps joint to scalar: phi^T [B_k, I]
            Bk = self.B_dict[k]
            if phi is None:
                raise ValueError("context_x required for measurement update to compute phi")
            H_g = (Bk.T @ phi).reshape(1, self.d_g)  # shape (1, d_g)
            H_u = phi.reshape(1, self.d_phi)  # shape (1, d_phi)
            H = np.concatenate([H_g, H_u], axis=1)  # shape (1, d_g + d_phi)
            # predictive observation mean and variance
            pred_mean = float(H @ joint_mu)
            pred_var = float(H @ joint_Sigma @ H.T) + self.R
            # Kalman gain
            S = pred_var
            K = (joint_Sigma @ H.T) / S  # shape (d_total, 1)
            innovation = observed_residual - pred_mean
            joint_mu_post = joint_mu + (K.flatten() * innovation)
            joint_Sigma_post = joint_Sigma - K @ H @ joint_Sigma
            # extract marginals
            self.mu_g[m] = joint_mu_post[: self.d_g]
            self.Sigma_g[m] = joint_Sigma_post[: self.d_g, : self.d_g]
            self.mu_u[k][m] = joint_mu_post[self.d_g :]
            self.Sigma_u[k][m] = joint_Sigma_post[self.d_g :, self.d_g :]

            # likelihood of observed residual under mode m
            like[m] = (1.0 / np.sqrt(2 * np.pi * pred_var)) * np.exp(-0.5 * innovation ** 2 / pred_var)

        # Update regime weights using Bayes rule: w_new ~ w_pred * like
        w_new = self.w * like
        s = float(np.sum(w_new))
        if s <= 0:
            # fallback to uniform
            self.w = np.ones_like(self.w) / float(self.M)
        else:
            self.w = w_new / s

        # mark selected expert activity
        self.last_selected_step[k] = 0

    # ---------------------- Registry management ----------------------
    # Algorithm 7 MANAGE REGISTRY: Entering/Stale Expert
    def manage_registry(self, current_experts: Sequence[int]) -> None:
        """
        Prune experts not selected for Delta_max steps and birth new experts in current_experts.
        """
        current_set = set(int(x) for x in current_experts)
        # Birth new experts
        for k in current_set:
            if k not in self.registry:
                self._birth_expert(k)
        # Prune old experts
        for k in list(self.registry):
            if self.last_selected_step.get(k, 0) > self.Delta_max and k not in current_set:
                # remove
                self.registry.remove(k)
                self.mu_u.pop(k, None)
                self.Sigma_u.pop(k, None)
                self.B_dict.pop(k, None)
                self.beta.pop(k, None)
                self.last_selected_step.pop(k, None)

    def _birth_expert(self, k: int) -> None:
        """
        Initialize idiosyncratic state for a newly observed expert using population prior.
        """
        self.registry.append(k)
        # initialize mode-conditioned copies of prior
        self.mu_u[k] = np.stack([self.mu_u0.copy() for _ in range(self.M)])
        self.Sigma_u[k] = np.stack([self.Sigma_u0.copy() for _ in range(self.M)])
        # default B_k matrix (small random mapping) if not provided
        if k not in self.B_dict:
            rng = np.random.default_rng(k)
            self.B_dict[k] = rng.normal(scale=0.01, size=(self.d_phi, self.d_g))
        if k not in self.beta:
            self.beta[k] = 0.0
        self.last_selected_step[k] = 0

