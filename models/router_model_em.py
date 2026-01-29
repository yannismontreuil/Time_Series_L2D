import numpy as np
from typing import Optional, Sequence

from models.router_model import SLDSIMMRouter


class SLDSIMMRouter_EM(SLDSIMMRouter):
    """
    Approximate EM-capable SLDS-IMM router for independent experts.

    This class augments SLDSIMMRouter with a single-pass EM-style update
    over an initial window t=1..em_tk when training_mode is True.

    Notes:
      - The E-step is approximate: uses filtered posteriors (IMM) and
        ignores cross-time covariances.
      - EM updates are only applied in full-feedback mode.
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
        beta: Optional[np.ndarray] = None,
        lambda_risk: float | np.ndarray = 0.0,
        pop_mean: Optional[np.ndarray] = None,
        pop_cov: Optional[np.ndarray] = None,
        feedback_mode: str = "full",
        eps: float = 1e-8,
        # EM configuration
        em_tk: Optional[int] = None,
        em_min_weight: float = 1e-6,
        em_verbose: bool = False,
        em_update_pi: bool = True,
        em_update_AQ: bool = True,
        em_update_R: bool = True,
        em_lambda_A: float = 1e-6,
        em_r_floor: float = 1e-8,
    ):
        super().__init__(
            num_experts=num_experts,
            num_regimes=num_regimes,
            state_dim=state_dim,
            feature_fn=feature_fn,
            A=A,
            Q=Q,
            R=R,
            Pi=Pi,
            beta=beta,
            lambda_risk=lambda_risk,
            pop_mean=pop_mean,
            pop_cov=pop_cov,
            feedback_mode=feedback_mode,
            eps=eps,
        )

        self.training_mode: bool = False
        self.em_tk: Optional[int] = int(em_tk) if em_tk is not None else None
        self.em_min_weight: float = float(em_min_weight)
        self.em_verbose: bool = bool(em_verbose)
        self.em_update_pi: bool = bool(em_update_pi)
        self.em_update_AQ: bool = bool(em_update_AQ)
        self.em_update_R: bool = bool(em_update_R)
        self.em_lambda_A: float = float(em_lambda_A)
        self.em_r_floor: float = float(em_r_floor)

        # Time counter for EM bookkeeping.
        self._time: int = 0

        # EM accumulators
        self._em_initialized: bool = False
        self._em_done: bool = False

        self._em_Sxx: Optional[np.ndarray] = None
        self._em_Sxprev: Optional[np.ndarray] = None
        self._em_Sx_xprev: Optional[np.ndarray] = None
        self._em_weight: Optional[np.ndarray] = None

        self._em_R_num: Optional[np.ndarray] = None
        self._em_R_den: Optional[np.ndarray] = None

        self._em_xi: Optional[np.ndarray] = None

    # --------------------------------------------------------
    # EM helpers
    # --------------------------------------------------------

    def _em_maybe_init_stats(self) -> None:
        if self._em_initialized or self.em_tk is None:
            return
        M, N, d = self.M, self.N, self.d
        self._em_Sxx = np.zeros((M, d, d), dtype=float)
        self._em_Sxprev = np.zeros((M, d, d), dtype=float)
        self._em_Sx_xprev = np.zeros((M, d, d), dtype=float)
        self._em_weight = np.zeros(M, dtype=float)

        self._em_R_num = np.zeros((M, N), dtype=float)
        self._em_R_den = np.zeros((M, N), dtype=float)

        self._em_xi = np.zeros((M, M), dtype=float)
        self._em_initialized = True

    def _em_accumulate(
        self,
        cache: dict,
        losses_full: Optional[np.ndarray],
        available_experts: Sequence[int],
        prev_m: Optional[np.ndarray],
        prev_P: Optional[np.ndarray],
        prev_b: Optional[np.ndarray],
    ) -> None:
        if (
            not self.training_mode
            or self.em_tk is None
            or self._em_done
            or self.feedback_mode != "full"
        ):
            return
        self._em_maybe_init_stats()
        if self._em_Sxx is None or self._em_weight is None:
            return

        t_curr = int(self._time)
        if t_curr <= 0:
            return

        phi_t = np.asarray(cache.get("phi_t"), dtype=float).reshape(self.d)
        b_curr = np.asarray(self.b, dtype=float)

        # Determine observed experts (available and finite loss).
        observed_idx = []
        if losses_full is not None:
            loss_arr = np.asarray(losses_full, dtype=float)
            for j in available_experts:
                j = int(j)
                if j < 0 or j >= loss_arr.shape[0]:
                    continue
                if np.isfinite(loss_arr[j]):
                    observed_idx.append(j)
        else:
            observed_idx = [int(j) for j in available_experts]

        # Transition counts for Pi (approximate xi)
        b_pred = cache.get("b_pred", None)
        if (
            self.em_update_pi
            and prev_b is not None
            and b_pred is not None
            and self._em_xi is not None
        ):
            b_pred_arr = np.asarray(b_pred, dtype=float)
            prev_b_arr = np.asarray(prev_b, dtype=float)
            for k in range(self.M):
                denom = max(float(b_pred_arr[k]), self.eps)
                for i in range(self.M):
                    xi = prev_b_arr[i] * float(self.Pi[i, k]) * (float(b_curr[k]) / denom)
                    self._em_xi[i, k] += xi

        # Dynamics statistics for A, Q
        if self.em_update_AQ and prev_m is not None and prev_P is not None:
            prev_m_arr = np.asarray(prev_m, dtype=float)
            prev_P_arr = np.asarray(prev_P, dtype=float)
            for k in range(self.M):
                w_k = max(float(b_curr[k]), self.em_min_weight)
                for j in observed_idx:
                    m_t = self.m[k, j]
                    P_t = self.P[k, j]
                    m_prev = prev_m_arr[k, j]
                    P_prev = prev_P_arr[k, j]
                    self._em_Sxx[k] += w_k * (P_t + np.outer(m_t, m_t))
                    self._em_Sxprev[k] += w_k * (P_prev + np.outer(m_prev, m_prev))
                    self._em_Sx_xprev[k] += w_k * np.outer(m_t, m_prev)
                    self._em_weight[k] += w_k

        # Observation noise R
        if self.em_update_R and losses_full is not None and self._em_R_num is not None:
            loss_arr = np.asarray(losses_full, dtype=float)
            for j in observed_idx:
                ell_j = float(loss_arr[j])
                for k in range(self.M):
                    w_k = max(float(b_curr[k]), self.em_min_weight)
                    m_t = self.m[k, j]
                    P_t = self.P[k, j]
                    resid = ell_j - float(phi_t @ m_t)
                    var = float(phi_t @ (P_t @ phi_t))
                    self._em_R_num[k, j] += w_k * (resid * resid + var)
                    self._em_R_den[k, j] += w_k

    def _em_run_m_step(self) -> None:
        if self._em_done or self.em_tk is None:
            return
        if self._em_Sxx is None or self._em_weight is None:
            return

        M, d = self.M, self.d
        if self.em_update_AQ:
            for k in range(M):
                weight = float(self._em_weight[k])
                if weight <= self.em_min_weight:
                    continue
                Sxprev = self._em_Sxprev[k]
                Sx_xprev = self._em_Sx_xprev[k]
                reg = self.em_lambda_A * np.eye(d, dtype=float)
                try:
                    A_k = np.linalg.solve(Sxprev + reg, Sx_xprev.T).T
                except np.linalg.LinAlgError:
                    A_k = np.linalg.solve(
                        Sxprev + (self.em_lambda_A + self.eps) * np.eye(d, dtype=float),
                        Sx_xprev.T,
                    ).T
                self.A[k] = A_k

                # Q update
                Sxx = self._em_Sxx[k]
                Q_k = (
                    Sxx
                    - A_k @ Sx_xprev.T
                    - Sx_xprev @ A_k.T
                    + A_k @ Sxprev @ A_k.T
                ) / max(weight, self.eps)
                Q_k = 0.5 * (Q_k + Q_k.T) + self.eps * np.eye(d, dtype=float)
                self.Q[k] = Q_k

        if self.em_update_R and self._em_R_num is not None and self._em_R_den is not None:
            for k in range(M):
                for j in range(self.N):
                    denom = float(self._em_R_den[k, j])
                    if denom <= self.em_min_weight:
                        continue
                    r_val = float(self._em_R_num[k, j]) / max(denom, self.eps)
                    self.R[k, j] = max(r_val, self.em_r_floor)

        if self.em_update_pi and self._em_xi is not None:
            for i in range(M):
                row_sum = float(np.sum(self._em_xi[i]))
                if row_sum <= self.em_min_weight:
                    continue
                self.Pi[i] = self._em_xi[i] / max(row_sum, self.eps)
            # Normalize rows to sum to 1.
            row_sums = np.maximum(self.Pi.sum(axis=1, keepdims=True), self.eps)
            self.Pi = self.Pi / row_sums

        if self.em_verbose:
            print("[SLDSIMMRouter_EM] EM M-step completed.")

        self._em_done = True

    # --------------------------------------------------------
    # Overrides
    # --------------------------------------------------------

    def reset_beliefs(self, b0: Optional[np.ndarray] = None) -> None:
        super().reset_beliefs(b0=b0)
        self._time = 0
        # Clear accumulators but do not reset _em_done so that EM is not rerun
        # after it has already been applied.
        self._em_initialized = False
        self._em_Sxx = None
        self._em_Sxprev = None
        self._em_Sx_xprev = None
        self._em_weight = None
        self._em_R_num = None
        self._em_R_den = None
        self._em_xi = None

    def update_beliefs(
        self,
        r_t: int,
        loss_obs: float,
        losses_full: Optional[np.ndarray],
        available_experts: Sequence[int],
        cache: dict,
    ) -> None:
        prev_m = None
        prev_P = None
        prev_b = None
        if self.training_mode and self.em_tk is not None and not self._em_done:
            prev_m = np.asarray(self.m, dtype=float).copy()
            prev_P = np.asarray(self.P, dtype=float).copy()
            prev_b = np.asarray(self.b, dtype=float).copy()

        super().update_beliefs(
            r_t=r_t,
            loss_obs=loss_obs,
            losses_full=losses_full,
            available_experts=available_experts,
            cache=cache,
        )

        self._time += 1
        t_curr = int(self._time)

        if (
            not self.training_mode
            or self.em_tk is None
            or self._em_done
            or t_curr <= 0
        ):
            return

        if t_curr <= self.em_tk:
            self._em_accumulate(
                cache=cache,
                losses_full=losses_full,
                available_experts=available_experts,
                prev_m=prev_m,
                prev_P=prev_P,
                prev_b=prev_b,
            )

        if t_curr == self.em_tk:
            self._em_run_m_step()
