import numpy as np
from typing import Optional, Sequence

from models.router_model_corr import SLDSIMMRouter_Corr

class SLDSIMMRouter_Corr_EM(SLDSIMMRouter_Corr):
    """
    Correlated SLDS-IMM router with an approximate single-pass EM-style
    update of the linear-Gaussian dynamics and observation noise over an
    initial training window t = 1,...,t_k.

    This class extends SLDSIMMRouter_Corr by:
      - exposing a `training_mode` flag understood by router_eval.py,
      - accumulating approximate sufficient statistics for the dynamics
        and emission noise while training_mode is True and t <= t_k,
      - performing one EM-style M-step at t = t_k to update:
            A_g, Q_g  (shared-factor dynamics)
            A_u, Q_u  (idiosyncratic dynamics)
            R         (observation noise, per regime/expert),
        followed by a rebuild of the joint (A_joint, Q_joint) blocks.

    Notes
    -----
    - The E-step is approximate: we use filtered per-regime posteriors
      (b_t, m_t) and treat m_t[k] as a proxy for E[x_t | z_t=k, H_t].
      Cross-time covariances are ignored; the dynamics are fitted via
      weighted least squares on posterior means.
    - Feature-map learning (when feature_mode == "learnable") still uses
      the online approximate EM-style gradient in the base class.
    - EM-style learning is only implemented for full-feedback routers
      (feedback_mode == "full"). For partial feedback, the base router
      behaviour is used without EM parameter updates.
    """

    def __init__(
        self,
        num_experts: int,
        num_regimes: int,
        shared_dim: int,
        idiosyncratic_dim: int,
        feature_fn,
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
        feedback_mode: str = "full",
        eps: float = 1e-8,
        g_mean0: Optional[np.ndarray] = None,
        g_cov0: Optional[np.ndarray] = None,
        u_mean0: Optional[np.ndarray] = None,
        u_cov0: Optional[np.ndarray] = None,
        feature_arch: str = "linear",
        feature_hidden_dim: Optional[int] = None,
        feature_activation: str = "tanh",
        # EM-specific configuration
        em_tk: Optional[int] = None,
        em_min_weight: float = 1e-6,
        em_verbose: bool = False,
        # Feature-learning schedule across phases:
        #   Phase 0 (pretraining):      t < phase0_t_end       -> lr ≈ lr_phase0
        #   Phase 1 (offline SLDS/EM):  phase0_t_end <= t < em_tk -> lr ≈ lr_phase1
        #   Phase 2 (deployment):       t >= em_tk or training_mode=False -> lr ≈ lr_phase2
        phase0_t_end: Optional[int] = None,
        feature_lr_phase0: Optional[float] = None,
        feature_lr_phase1: Optional[float] = None,
        feature_lr_phase2: float = 0.0,
    ):
        super().__init__(
            num_experts=num_experts,
            num_regimes=num_regimes,
            shared_dim=shared_dim,
            idiosyncratic_dim=idiosyncratic_dim,
            feature_fn=feature_fn,
            A_g=A_g,
            Q_g=Q_g,
            A_u=A_u,
            Q_u=Q_u,
            B=B,
            R=R,
            Pi=Pi,
            beta=beta,
            lambda_risk=lambda_risk,
            staleness_threshold=staleness_threshold,
            exploration_mode=exploration_mode,
            feature_mode=feature_mode,
            feature_learning_rate=feature_learning_rate,
            feature_freeze_after=feature_freeze_after,
            feature_log_interval=feature_log_interval,
            feedback_mode=feedback_mode,
            eps=eps,
            g_mean0=g_mean0,
            g_cov0=g_cov0,
            u_mean0=u_mean0,
            u_cov0=u_cov0,
            feature_arch=feature_arch,
            feature_hidden_dim=feature_hidden_dim,
            feature_activation=feature_activation,
        )

        # Store copies of the initial linear-Gaussian parameters. These
        # serve as the "Corr (no EM)" baseline for comparison against
        # the EM-updated parameters after the M-step.
        self.A_g_init = np.array(self.A_g, copy=True)
        self.Q_g_init = np.array(self.Q_g, copy=True)
        self.A_u_init = np.array(self.A_u, copy=True)
        self.Q_u_init = np.array(self.Q_u, copy=True)
        self.R_init = np.array(self.R, copy=True)

        # Training configuration: EM-style learning will be active only
        # when training_mode is True and em_tk is not None.
        self.training_mode: bool = False
        self.em_tk: Optional[int] = int(em_tk) if em_tk is not None else None
        self.em_min_weight: float = float(em_min_weight)
        self.em_verbose: bool = bool(em_verbose)

        # Feature-learning schedule across the three phases. By default:
        #   - Phase 0 uses the base feature_learning_rate,
        #   - Phase 1 uses a 10x smaller learning rate,
        #   - Phase 2 freezes features (lr = 0) for safer deployment.
        lr_base = float(self.feature_learning_rate)
        self.feature_lr_phase0: float = float(
            lr_base if feature_lr_phase0 is None else feature_lr_phase0
        )
        self.feature_lr_phase1: float = float(
            (0.1 * lr_base) if feature_lr_phase1 is None else feature_lr_phase1
        )
        self.feature_lr_phase2: float = float(feature_lr_phase2)

        if self.em_tk is not None:
            if phase0_t_end is not None:
                self.phase0_t_end: Optional[int] = int(phase0_t_end)
            else:
                # Default: Phase 0 occupies roughly the first half of the
                # EM window, Phase 1 the second half.
                self.phase0_t_end = max(1, int(self.em_tk // 2))
        else:
            self.phase0_t_end = None

        # Internal accumulators for EM statistics.
        M = self.M
        self._em_initialized: bool = False
        self._em_done: bool = False

        # Previous-step posterior means for dynamics statistics.
        self._em_prev_m: Optional[np.ndarray] = None
        self._em_prev_b: Optional[np.ndarray] = None

        # The following fields are allocated lazily when EM is first used.
        self._em_Sx_g: Optional[np.ndarray] = None
        self._em_Sxy_g: Optional[np.ndarray] = None
        self._em_Syy_g: Optional[np.ndarray] = None
        self._em_weight_g: Optional[np.ndarray] = None

        self._em_Sx_u: Optional[np.ndarray] = None
        self._em_Sxy_u: Optional[np.ndarray] = None
        self._em_Syy_u: Optional[np.ndarray] = None
        self._em_weight_u: Optional[np.ndarray] = None

        # Residual-based statistics for R_{k,j}
        self._em_sum_resid2_R: Optional[np.ndarray] = None
        self._em_weight_R: Optional[np.ndarray] = None

    # --------------------------------------------------------
    # EM helpers
    # --------------------------------------------------------

    def _em_maybe_init_stats(self) -> None:
        if self._em_initialized:
            return
        if self.em_tk is None:
            return

        M, dg, du, N = self.M, self.dg, self.du, self.N

        self._em_Sx_g = np.zeros((M, dg, dg), dtype=float)
        self._em_Sxy_g = np.zeros((M, dg, dg), dtype=float)
        self._em_Syy_g = np.zeros((M, dg, dg), dtype=float)
        self._em_weight_g = np.zeros(M, dtype=float)

        self._em_Sx_u = np.zeros((M, du, du), dtype=float)
        self._em_Sxy_u = np.zeros((M, du, du), dtype=float)
        self._em_Syy_u = np.zeros((M, du, du), dtype=float)
        self._em_weight_u = np.zeros(M, dtype=float)

        self._em_sum_resid2_R = np.zeros((M, N), dtype=float)
        self._em_weight_R = np.zeros((M, N), dtype=float)

        self._em_initialized = True

    def _rebuild_joint_dynamics(self) -> None:
        """
        Rebuild the joint A_joint and Q_joint matrices from A_g, Q_g,
        A_u, Q_u and the current model dimensions.
        """
        M, d_state = self.M, self.d_state

        for k in range(M):
            A_k = np.zeros((d_state, d_state), dtype=float)
            Q_k = np.zeros((d_state, d_state), dtype=float)

            g_slice = self._g_slice()
            A_k[g_slice, g_slice] = self.A_g[k]
            Q_k[g_slice, g_slice] = self.Q_g[k]

            for j in range(self.N):
                u_slice = self._u_slice(j)
                A_k[u_slice, u_slice] = self.A_u[k]
                Q_k[u_slice, u_slice] = self.Q_u[k]

            self.A_joint[k] = A_k
            self.Q_joint[k] = Q_k

    def _em_accumulate_dynamics(
        self,
        prev_m: Optional[np.ndarray],
        prev_b: Optional[np.ndarray],
    ) -> None:
        """
        Accumulate approximate dynamics statistics using posterior means
        at times t-1 (prev_m, prev_b) and t (self.m, self.b).
        """
        if (
            not self.training_mode
            or self.em_tk is None
            or self._em_done
            or prev_m is None
            or prev_b is None
        ):
            return

        self._em_maybe_init_stats()
        assert self._em_Sx_g is not None
        assert self._em_Sxy_g is not None
        assert self._em_Syy_g is not None
        assert self._em_weight_g is not None
        assert self._em_Sx_u is not None
        assert self._em_Sxy_u is not None
        assert self._em_Syy_u is not None
        assert self._em_weight_u is not None

        M = self.M
        g_slice = self._g_slice()

        for k in range(M):
            w_k = float(self.b[k])
            if w_k <= 0.0:
                continue

            m_prev_k = prev_m[k]
            m_curr_k = self.m[k]

            # Shared factor dynamics: g_{t} -> g_{t+1}
            g_prev = m_prev_k[g_slice]
            g_curr = m_curr_k[g_slice]
            self._em_Sx_g[k] += w_k * np.outer(g_prev, g_prev)
            self._em_Sxy_g[k] += w_k * np.outer(g_curr, g_prev)
            self._em_Syy_g[k] += w_k * np.outer(g_curr, g_curr)
            self._em_weight_g[k] += w_k

            # Idiosyncratic dynamics: aggregate u_{j,t} -> u_{j,t+1}
            for j in range(self.N):
                u_slice = self._u_slice(j)
                u_prev = m_prev_k[u_slice]
                u_curr = m_curr_k[u_slice]

                self._em_Sx_u[k] += w_k * np.outer(u_prev, u_prev)
                self._em_Sxy_u[k] += w_k * np.outer(u_curr, u_prev)
                self._em_Syy_u[k] += w_k * np.outer(u_curr, u_curr)
                self._em_weight_u[k] += w_k

    def _em_accumulate_R(
        self,
        cache: dict,
        losses_full: Optional[np.ndarray],
        available_experts: Sequence[int],
    ) -> None:
        """
        Accumulate residual-based statistics for R_{k,j} using current
        posterior regime weights b_t and the IMM predictive means from
        the cache.

        Only used in full-feedback mode; for partial feedback we keep R
        fixed (since most losses are unobserved).
        """
        if (
            not self.training_mode
            or self.em_tk is None
            or self._em_done
            or losses_full is None
        ):
            return

        if self.feedback_mode != "full":
            return

        self._em_maybe_init_stats()
        assert self._em_sum_resid2_R is not None
        assert self._em_weight_R is not None

        mu_kj = cache.get("mu_kj", None)
        if mu_kj is None:
            return

        losses_full = np.asarray(losses_full, dtype=float)
        avail = np.asarray(list(available_experts), dtype=int)
        if avail.size == 0:
            return

        M = self.M

        for k in range(M):
            w_k = float(self.b[k])
            if w_k <= 0.0:
                continue
            for j in avail:
                ell_j = float(losses_full[j])
                if not np.isfinite(ell_j):
                    continue
                mu_kj_val = float(mu_kj[k, j])
                resid = ell_j - mu_kj_val
                self._em_sum_resid2_R[k, j] += w_k * resid * resid
                self._em_weight_R[k, j] += w_k

    def _em_run_m_step(self) -> None:
        """
        Perform a single EM-style M-step using the accumulated dynamics
        and residual statistics, and rebuild the joint dynamics.
        """
        if self.em_tk is None or self._em_done:
            return

        self._em_maybe_init_stats()
        assert self._em_Sx_g is not None
        assert self._em_Sxy_g is not None
        assert self._em_Syy_g is not None
        assert self._em_weight_g is not None
        assert self._em_Sx_u is not None
        assert self._em_Sxy_u is not None
        assert self._em_Syy_u is not None
        assert self._em_weight_u is not None
        assert self._em_sum_resid2_R is not None
        assert self._em_weight_R is not None

        M, dg, du, N = self.M, self.dg, self.du, self.N

        # Update A_g, Q_g and A_u, Q_u via weighted least squares on
        # posterior means.
        for k in range(M):
            # Shared factor dynamics
            w_g = float(self._em_weight_g[k])
            if w_g > self.em_min_weight:
                Sx = self._em_Sx_g[k]
                Sxy = self._em_Sxy_g[k]
                Syy = self._em_Syy_g[k]
                try:
                    Sx_inv = np.linalg.pinv(Sx)
                except np.linalg.LinAlgError:
                    Sx_inv = np.linalg.pinv(Sx + self.eps * np.eye(dg))
                A_g_new = Sxy @ Sx_inv
                # Q_g ≈ (Syy - A_g Sxy^T) / w_g
                Q_g_num = Syy - A_g_new @ Sxy.T
                Q_g_new = Q_g_num / max(w_g, self.em_min_weight)
                Q_g_new = 0.5 * (Q_g_new + Q_g_new.T) + self.eps * np.eye(dg)
                self.A_g[k] = A_g_new
                self.Q_g[k] = Q_g_new

            # Idiosyncratic dynamics (aggregated over experts)
            w_u = float(self._em_weight_u[k])
            if w_u > self.em_min_weight:
                Sx_u = self._em_Sx_u[k]
                Sxy_u = self._em_Sxy_u[k]
                Syy_u = self._em_Syy_u[k]
                try:
                    Sx_u_inv = np.linalg.pinv(Sx_u)
                except np.linalg.LinAlgError:
                    Sx_u_inv = np.linalg.pinv(Sx_u + self.eps * np.eye(du))
                A_u_new = Sxy_u @ Sx_u_inv
                Q_u_num = Syy_u - A_u_new @ Sxy_u.T
                Q_u_new = Q_u_num / max(w_u, self.em_min_weight)
                Q_u_new = 0.5 * (Q_u_new + Q_u_new.T) + self.eps * np.eye(du)
                self.A_u[k] = A_u_new
                self.Q_u[k] = Q_u_new

        # Update R_{k,j} from residual variances.
        for k in range(M):
            for j in range(N):
                w_R = float(self._em_weight_R[k, j])
                if w_R <= self.em_min_weight:
                    continue
                var_hat = float(self._em_sum_resid2_R[k, j] / max(w_R, self.em_min_weight))
                self.R[k, j] = max(var_hat, self.eps)

        # Rebuild joint dynamics with updated A_g, Q_g, A_u, Q_u.
        self._rebuild_joint_dynamics()

        self._em_done = True

        if self.em_verbose:
            mode = getattr(self, "feedback_mode", "unknown")
            print(
                f"[SLDSIMMRouter_Corr_EM] Completed EM M-step at t={self._time} "
                f"(feedback_mode={mode}). Learned parameters:"
            )

            # Print baseline (Corr, no EM) vs EM-updated parameters.
            print(f"A_g (Corr no EM): {self.A_g_init}")
            print(f"A_g (Corr EM):    {self.A_g}")
            print(f"Q_g (Corr no EM): {self.Q_g_init}")
            print(f"Q_g (Corr EM):    {self.Q_g}")

            print(f"A_u (Corr no EM): {self.A_u_init}")
            print(f"A_u (Corr EM):    {self.A_u}")
            print(f"Q_u (Corr no EM): {self.Q_u_init}")
            print(f"Q_u (Corr EM):    {self.Q_u}")

            print(f"R (Corr no EM):   {self.R_init}")
            print(f"R (Corr EM):      {self.R}")

    # --------------------------------------------------------
    # Belief update override with EM hooks
    # --------------------------------------------------------

    def update_beliefs(
        self,
        r_t: int,
        loss_obs: float,
        losses_full: Optional[np.ndarray],
        available_experts: Sequence[int],
        cache: dict,
    ) -> None:
        """
        Same filtering and regime update as SLDSIMMRouter_Corr, with
        additional accumulation of EM statistics while training_mode is
        True and t <= em_tk. At t = em_tk, a single EM-style M-step is
        performed to update dynamics (A_g, Q_g, A_u, Q_u) and noise R.
        """
        # Phase-controlled feature learning rate:
        #   - Phase 0 / 1 active only while training_mode is True,
        #   - Phase 2 (deployment) when t >= em_tk or training_mode is False.
        old_lr = self.feature_learning_rate
        eff_lr = old_lr
        t_prev = int(getattr(self, "_time", 0))

        if (
            self.feature_mode == "learnable"
            and old_lr > 0.0
        ):
            if not self.training_mode:
                # Phase 2: online deployment, freeze (or nearly freeze) features.
                eff_lr = self.feature_lr_phase2
            else:
                if self.em_tk is not None:
                    phase0_end = self.phase0_t_end
                    if phase0_end is not None and t_prev < phase0_end:
                        # Phase 0: feature pretraining (next-step style) on
                        # the initial segment of the trajectory.
                        eff_lr = self.feature_lr_phase0
                    elif t_prev < self.em_tk:
                        # Phase 1: offline SLDS/EM training with a lower
                        # feature learning rate to support switching.
                        eff_lr = self.feature_lr_phase1
                    else:
                        # Phase 2: post-EM deployment.
                        eff_lr = self.feature_lr_phase2
                else:
                    # No EM window specified: treat all training time as
                    # Phase 0-style pretraining.
                    eff_lr = self.feature_lr_phase0

        self.feature_learning_rate = eff_lr

        # Snapshot previous posterior before the base update; these
        # correspond to time t-1 when update_beliefs processes time t.
        prev_m = None
        prev_b = None
        if self.training_mode and self.em_tk is not None and not self._em_done:
            # Only meaningful after first step, when m/P/b have been set.
            if hasattr(self, "m") and hasattr(self, "b"):
                prev_m = None if self.m is None else np.asarray(self.m, dtype=float).copy()
                prev_b = None if self.b is None else np.asarray(self.b, dtype=float).copy()

        # Run the base-class filtering and (optional) feature-step update.
        super().update_beliefs(
            r_t=r_t,
            loss_obs=loss_obs,
            losses_full=losses_full,
            available_experts=available_experts,
            cache=cache,
        )

        # Restore original feature learning rate so that callers can
        # inspect self.feature_learning_rate without phase-specific
        # side effects.
        self.feature_learning_rate = old_lr

        # Current time index after the base update (decision epoch t).
        t_curr = int(self._time)

        if (
            not self.training_mode
            or self.em_tk is None
            or self._em_done
            or t_curr <= 0
        ):
            return

        # Only accumulate statistics while t <= em_tk.
        if t_curr <= self.em_tk:
            # Dynamics statistics use posterior at t-1 and t.
            self._em_accumulate_dynamics(prev_m, prev_b)
            # Residual statistics for R use posterior at t and predictive
            # means from the cache.
            self._em_accumulate_R(
                cache=cache,
                losses_full=losses_full,
                available_experts=available_experts,
            )

        # At t = em_tk, perform a single EM-style M-step.
        if t_curr == self.em_tk and not self._em_done:
            self._em_run_m_step()
