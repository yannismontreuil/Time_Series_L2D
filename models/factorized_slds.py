import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from models.router_model import SLDSIMMRouter

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if torch is not None:
    class TransitionMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            num_modes: int,
            activation: str = "tanh",
        ) -> None:
            super().__init__()
            dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(num_modes) ** 2]
            layers: List[nn.Module] = []
            act = str(activation).lower()
            for idx in range(len(dims) - 1):
                layers.append(nn.Linear(dims[idx], dims[idx + 1]))
                if idx < len(dims) - 2:
                    if act == "tanh":
                        layers.append(nn.Tanh())
                    elif act == "relu":
                        layers.append(nn.ReLU())
                    elif act == "gelu":
                        layers.append(nn.GELU())
                    elif act == "elu":
                        layers.append(nn.ELU())
                    else:
                        raise ValueError("Unsupported transition_activation.")
            self.net = nn.Sequential(*layers)
            self.M = int(num_modes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.net(x)
            return out.view(-1, self.M, self.M)


class FactorizedSLDS(SLDSIMMRouter):
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
      e_{t,k} = phi(x_t)^T (B_k g_t + u_{t,k}) + eps,  eps ~ N(0, R_{z_t,k})

    The class exposes the same select/update interface as SLDSIMMRouter
    and also keeps the legacy predict/select/update API for backward
    compatibility.
    """

    def __init__(
        self,
        M: int,
        d_g: int,
        d_phi: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        B_dict: Optional[Dict[int, np.ndarray]] = None,
        beta: Optional[np.ndarray | Dict[int, float]] = None,
        R: float | np.ndarray = 1.0,
        A_g: Optional[np.ndarray] = None,
        A_u: Optional[np.ndarray] = None,
        Q_g: Optional[np.ndarray] = None,
        Q_u: Optional[np.ndarray] = None,
        Delta_max: int = 100,
        seed: int = 0,
        num_experts: Optional[int] = None,
        B_intercept_load: float = 1.0,
        attn_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        g_mean0: Optional[np.ndarray] = None,
        g_cov0: Optional[np.ndarray] = None,
        u_mean0: Optional[np.ndarray] = None,
        u_cov0: Optional[np.ndarray] = None,
        eps: float = 1e-8,
        feedback_mode: str = "partial",
        exploration: str = "g",
        exploration_mc_samples: int = 25,
        exploration_ucb_samples: int = 200,
        exploration_ucb_alpha: Optional[float] = None,
        exploration_ucb_schedule: str = "inverse_t",
        exploration_sampling_deterministic: bool = False,
        exploration_diag_enabled: bool = False,
        exploration_diag_stride: int = 100,
        exploration_diag_samples: int = 50,
        exploration_diag_print: bool = True,
        exploration_diag_max_records: int = 2000,
        state_clip: float = 1e6,
        cov_clip: float = 1e12,
        observation_mode: str = "residual",
        loss_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        loss_quad_points: int = 20,
        transition_model: Optional[object] = None,
        transition_hidden_dims: Optional[Sequence[int]] = None,
        transition_activation: str = "tanh",
        transition_device: Optional[str] = None,
        transition_mode: str = "attention",
    ) -> None:
        self.M = int(M)
        self.d_g = int(d_g)
        self.d_phi = int(d_phi)
        self.d = self.d_phi
        self.feature_fn = feature_fn

        self.N: Optional[int] = int(num_experts) if num_experts is not None else None
        if self.N is None and beta is not None and not isinstance(beta, dict):
            beta_arr = np.asarray(beta, dtype=float)
            if beta_arr.ndim == 1:
                self.N = int(beta_arr.shape[0])
        if self.N is None and isinstance(R, np.ndarray) and R.ndim == 2:
            self.N = int(R.shape[1])
        if self.N is None and B_dict:
            self.N = int(max(B_dict.keys()) + 1)

        self.beta = self._normalize_beta(beta)
        self.R = self._normalize_R(R)

        self.A_g = self._normalize_matrix(A_g, self.d_g)
        self.A_u = self._normalize_matrix(A_u, self.d_phi)
        self.Q_g = self._normalize_cov(Q_g, self.d_g)
        self.Q_u = self._normalize_cov(Q_u, self.d_phi)

        self.Delta_max = int(Delta_max)
        self.eps = float(eps)

        self.g_mean0 = self._normalize_mean_modes(g_mean0, self.d_g)
        self.g_cov0 = self._normalize_cov_modes(g_cov0, self.d_g, default_scale=1.0)
        self.u_mean0 = self._normalize_mean_modes(u_mean0, self.d_phi)
        self.u_cov0 = self._normalize_cov_modes(u_cov0, self.d_phi, default_scale=0.1)

        self.B_intercept_load = float(B_intercept_load)
        self.B_dict: Dict[int, np.ndarray] = {}
        if B_dict:
            for k, B_k in B_dict.items():
                B_arr = np.asarray(B_k, dtype=float)
                if B_arr.shape != (self.d_phi, self.d_g):
                    raise ValueError("B_k must have shape (d_phi, d_g).")
                self.B_dict[int(k)] = B_arr

        if attn_dim is None:
            attn_dim = max(1, self.d_phi)
        self.attn_dim = int(attn_dim)

        self._rng = np.random.default_rng(seed)
        self.context_dim: Optional[int] = (
            int(context_dim) if context_dim is not None else None
        )
        self.transition_model = transition_model
        if transition_hidden_dims is None:
            self.transition_hidden_dims = None
        else:
            self.transition_hidden_dims = [int(h) for h in transition_hidden_dims]
            if not self.transition_hidden_dims:
                raise ValueError("transition_hidden_dims must be non-empty.")
        self.transition_activation = str(transition_activation)
        self.transition_input_dim: Optional[int] = None
        self.transition_device: Optional["torch.device"] = None
        self._transition_device_str = transition_device
        self.transition_mode = str(transition_mode)
        if self.transition_mode not in ("attention", "linear"):
            raise ValueError("transition_mode must be 'attention' or 'linear'.")
        self.W_lin: Optional[np.ndarray] = None
        self.b_lin: Optional[np.ndarray] = None
        if self.transition_model is not None:
            if torch is None or nn is None:
                raise RuntimeError("transition_model requires PyTorch.")
            if not isinstance(self.transition_model, nn.Module):
                raise ValueError("transition_model must be a torch.nn.Module.")
            self.transition_device = torch.device(transition_device or "cpu")
            self.transition_model.to(self.transition_device)
            if self.context_dim is not None:
                self.transition_input_dim = self.context_dim
        self.W_q: Optional[np.ndarray] = None
        self.W_k: Optional[np.ndarray] = None
        self.b_attn: Optional[np.ndarray] = None
        if self.context_dim is not None:
            self._init_transition_params(self.context_dim)

        if feedback_mode not in ("partial", "full"):
            raise ValueError("feedback_mode must be 'partial' or 'full'.")
        self.feedback_mode = feedback_mode

        exploration = str(exploration).lower()
        if exploration not in ("g", "g_z", "ucb", "sampling"):
            raise ValueError(
                "exploration must be one of 'g', 'g_z', 'ucb', or 'sampling'."
            )
        self.exploration = exploration
        self.exploration_mc_samples = max(int(exploration_mc_samples), 0)
        self.exploration_ucb_samples = max(int(exploration_ucb_samples), 0)
        if exploration_ucb_alpha is None:
            self.exploration_ucb_alpha = None
        else:
            self.exploration_ucb_alpha = float(exploration_ucb_alpha)
            if not (0.0 < self.exploration_ucb_alpha < 1.0):
                raise ValueError("exploration_ucb_alpha must be in (0, 1).")
        exploration_ucb_schedule = str(exploration_ucb_schedule).lower()
        if exploration_ucb_schedule not in ("inverse_t", "inverse_t_log2", "fixed"):
            raise ValueError(
                "exploration_ucb_schedule must be 'inverse_t', "
                "'inverse_t_log2', or 'fixed'."
            )
        self.exploration_ucb_schedule = exploration_ucb_schedule
        self.exploration_sampling_deterministic = bool(exploration_sampling_deterministic)
        self.exploration_diag_enabled = bool(exploration_diag_enabled)
        self.exploration_diag_stride = max(int(exploration_diag_stride), 1)
        self.exploration_diag_samples = max(int(exploration_diag_samples), 0)
        self.exploration_diag_print = bool(exploration_diag_print)
        self.exploration_diag_max_records = max(int(exploration_diag_max_records), 0)
        self.exploration_diag_records: List[Tuple[int, float, float, float]] = []
        self.state_clip = float(state_clip)
        self.cov_clip = float(cov_clip)
        self.loss_fn = loss_fn
        self.loss_quad_points = max(int(loss_quad_points), 0)

        if observation_mode != "residual":
            raise ValueError(
                "FactorizedSLDS only supports residual observations "
                "(e_{t,k} = y_hat - y_t) per factorized_router.tex."
            )
        self.observation_mode = "residual"
        self.em_tk: Optional[int] = None
        self._online_em_cfg: Dict[str, Any] = {
            "enabled": False,
            "window": 0,
            "period": 1,
            "n_em": 1,
            "n_samples": 1,
            "burn_in": 0,
            "epsilon_N": 0.0,
            "theta_lr": 1e-2,
            "theta_steps": 1,
            "priors": None,
            "start_t": None,
            "seed": None,
            "print_val_loss": False,
            "use_state_priors": True,
        }
        self._online_em_suspended = False
        self._em_records: List[Dict[str, Any]] = []
        self._em_records_start: Optional[int] = None
        self.transition_log_cfg: Optional[dict] = None
        self.transition_log_label: Optional[str] = None

        # Debug diagnostics for tracking underperformance
        self._debug_diag_cfg: Dict[str, Any] = {
            "enabled": False,
            "print_stride": 100,  # Print every N steps
            "log_weights": True,  # Log regime weights
            "log_variances": True,  # Log predictive variances
            "log_info_gain": True,  # Log information gain values
            "log_em_params": True,  # Log EM parameter changes
            "weight_collapse_threshold": 0.99,  # Warn if max weight > this
            "variance_collapse_threshold": 1e-6,  # Warn if variance < this
        }
        self._debug_diag_history: List[Dict[str, Any]] = []
        self._em_params_before: Optional[Dict[str, Any]] = None

        self.registry: List[int] = []
        self.mu_u: Dict[int, np.ndarray] = {}
        self.Sigma_u: Dict[int, np.ndarray] = {}
        self.last_selected_time: Dict[int, int] = {}
        self.current_step = 0

        self.reset_beliefs()

    def _normalize_beta(self, beta: Optional[np.ndarray | Dict[int, float]]):
        if beta is None:
            if self.N is None:
                return {}
            return np.zeros(self.N, dtype=float)
        if isinstance(beta, dict):
            if self.N is None:
                return {int(k): float(v) for k, v in beta.items()}
            beta_arr = np.zeros(self.N, dtype=float)
            for k, v in beta.items():
                idx = int(k)
                if 0 <= idx < self.N:
                    beta_arr[idx] = float(v)
            return beta_arr
        beta_arr = np.asarray(beta, dtype=float)
        if beta_arr.ndim != 1:
            raise ValueError("beta must be a 1D array or dict.")
        if self.N is None:
            self.N = int(beta_arr.shape[0])
        elif beta_arr.shape != (self.N,):
            raise ValueError("beta must have shape (num_experts,)")
        return beta_arr

    def _normalize_R(self, R: float | np.ndarray):
        if np.ndim(R) == 0:
            r_scalar = float(R)
            if self.N is None:
                return r_scalar
            return np.full((self.M, self.N), r_scalar, dtype=float)
        R_arr = np.asarray(R, dtype=float)
        if R_arr.ndim == 1:
            if self.N is None:
                raise ValueError("R requires num_experts when given as a vector.")
            if R_arr.shape == (self.M,):
                return np.repeat(R_arr[:, None], self.N, axis=1)
            if R_arr.shape == (self.N,):
                return np.repeat(R_arr[None, :], self.M, axis=0)
        if R_arr.shape == (self.M, self.N):
            return R_arr
        raise ValueError("R must be scalar, (M,), (N,), or (M, N).")

    def _normalize_matrix(self, A: Optional[np.ndarray], d: int) -> np.ndarray:
        if A is None:
            return np.stack([np.eye(d, dtype=float) for _ in range(self.M)])
        A_arr = np.asarray(A, dtype=float)
        if A_arr.shape == (d, d):
            return np.stack([A_arr for _ in range(self.M)])
        if A_arr.shape != (self.M, d, d):
            print(A_arr.shape)
            raise ValueError("A must have shape (M, d, d).")
        return A_arr

    def _normalize_cov(
        self,
        Q: Optional[np.ndarray],
        d: int,
        default_scale: float = 1e-3,
    ) -> np.ndarray:
        if Q is None:
            return np.stack([np.eye(d, dtype=float) * default_scale for _ in range(self.M)])
        Q_arr = np.asarray(Q, dtype=float)
        if Q_arr.shape == (d, d):
            return np.stack([Q_arr for _ in range(self.M)])
        if Q_arr.shape != (self.M, d, d):
            raise ValueError("Covariance must have shape (M, d, d).")
        return Q_arr

    def _normalize_mean_modes(self, mean: Optional[np.ndarray], d: int) -> np.ndarray:
        if mean is None:
            return np.zeros((self.M, d), dtype=float)
        mean_arr = np.asarray(mean, dtype=float)
        if mean_arr.shape == (d,):
            return np.stack([mean_arr.copy() for _ in range(self.M)])
        if mean_arr.shape != (self.M, d):
            raise ValueError("Mean must have shape (d,) or (M, d).")
        return mean_arr

    def _normalize_cov_modes(
        self,
        cov: Optional[np.ndarray],
        d: int,
        default_scale: float,
    ) -> np.ndarray:
        if cov is None:
            return np.stack(
                [np.eye(d, dtype=float) * default_scale for _ in range(self.M)]
            )
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.shape == (d, d):
            return np.stack([cov_arr.copy() for _ in range(self.M)])
        if cov_arr.shape != (self.M, d, d):
            raise ValueError("Covariance must have shape (d, d) or (M, d, d).")
        return cov_arr

    def _get_beta(self, k: int) -> float:
        if isinstance(self.beta, dict):
            return float(self.beta.get(k, 0.0))
        if self.beta.size == 0:
            return 0.0
        if k < 0 or k >= self.beta.shape[0]:
            return 0.0
        return float(self.beta[k])

    def _get_R(self, m: int, k: int) -> float:
        if np.ndim(self.R) == 0:
            return float(self.R)
        return float(self.R[m, k])

    def _ensure_B(self, k: int) -> np.ndarray:
        if k in self.B_dict:
            return self.B_dict[k]
        B_k = np.zeros((self.d_phi, self.d_g), dtype=float)
        if self.d_phi > 0 and self.d_g > 0:
            B_k[0, 0] = self.B_intercept_load
        self.B_dict[k] = B_k
        return B_k

    def _compute_phi(self, x: np.ndarray) -> np.ndarray:
        phi = np.asarray(self.feature_fn(x), dtype=float).reshape(-1)
        if phi.shape[0] != self.d_phi:
            raise ValueError("feature_fn returned wrong dimension for d_phi.")
        return phi

    def _softmax_rows(self, X: np.ndarray) -> np.ndarray:
        X = X - np.max(X, axis=1, keepdims=True)
        expX = np.exp(X)
        denom = np.sum(expX, axis=1, keepdims=True)
        denom = np.maximum(denom, self.eps)
        return expX / denom

    def _sanitize_vector(self, vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=float)
        if not np.isfinite(vec).all():
            return np.asarray(fallback, dtype=float).copy()
        return np.clip(vec, -self.state_clip, self.state_clip)

    def _sanitize_cov(self, cov: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov, dtype=float)
        if not np.isfinite(cov).all():
            return np.asarray(fallback, dtype=float).copy()
        if cov.size == 0:
            return cov.copy()
        cov = np.clip(cov, -self.cov_clip, self.cov_clip)
        cov = 0.5 * (cov + cov.T)
        # Ensure the covariance is symmetric PSD for stable filtering.
        try:
            np.linalg.cholesky(cov + self.eps * np.eye(cov.shape[0]))
            return cov
        except np.linalg.LinAlgError:
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.clip(eigvals, self.eps, self.cov_clip)
                return (eigvecs * eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                return np.asarray(fallback, dtype=float).copy()

    def _transition_log_enabled(self) -> bool:
        if not isinstance(self.transition_log_cfg, dict):
            return False
        return bool(self.transition_log_cfg.get("enabled", False))

    def _transition_log_precision(self) -> int:
        cfg = self.transition_log_cfg or {}
        try:
            return int(cfg.get("precision", 4))
        except (TypeError, ValueError):
            return 4

    def _maybe_log_transition_params(self, stage: str) -> None:
        if not self._transition_log_enabled():
            return
        if not bool(self.transition_log_cfg.get("print_wlin_blin", False)):
            return
        if self.transition_mode != "linear":
            return
        if self.W_lin is None or self.b_lin is None:
            return
        label = self.transition_log_label or "FactorizedSLDS"
        precision = self._transition_log_precision()
        w_str = np.array2string(
            self.W_lin,
            precision=precision,
            floatmode="fixed",
            separator=", ",
        )
        b_str = np.array2string(
            self.b_lin,
            precision=precision,
            floatmode="fixed",
            separator=", ",
        )
        print(f"[transition-params] {label} {stage} W_lin={w_str} b_lin={b_str}")

    def _init_transition_params(self, context_dim: int) -> None:
        self.context_dim = int(context_dim)
        if self.transition_mode == "linear":
            self.W_lin = self._rng.normal(
                scale=0.1, size=(self.M, self.M, self.context_dim)
            )
            self.b_lin = np.zeros((self.M, self.M), dtype=float)
            self.W_q = None
            self.W_k = None
            self.b_attn = None
            self._maybe_log_transition_params("init")
        else:
            self.W_q = self._rng.normal(
                scale=0.1, size=(self.M, self.attn_dim, self.context_dim)
            )
            self.W_k = self._rng.normal(
                scale=0.1, size=(self.M, self.attn_dim, self.context_dim)
            )
            self.b_attn = np.zeros((self.M, self.M), dtype=float)
            self.W_lin = None
            self.b_lin = None

    def _ensure_transition_model(self, context_dim: int) -> None:
        if self.transition_model is not None:
            if self.transition_input_dim is None:
                self.transition_input_dim = int(context_dim)
            elif int(context_dim) != self.transition_input_dim:
                raise ValueError("Context dimension mismatch for transition model.")
            return
        if self.transition_hidden_dims is None:
            return
        if torch is None or nn is None:
            raise RuntimeError("Transition model requires PyTorch.")
        self.transition_input_dim = int(context_dim)
        self.transition_model = TransitionMLP(
            self.transition_input_dim,
            self.transition_hidden_dims,
            self.M,
            activation=self.transition_activation,
        )
        self.transition_device = torch.device(self._transition_device_str or "cpu")
        self.transition_model.to(self.transition_device)

    def _transition_logits_torch(self, x_t: "torch.Tensor") -> "torch.Tensor":
        if self.transition_model is None:
            raise RuntimeError("Transition model is not initialized.")
        return self.transition_model(x_t)

    def _transition_logits_numpy(self, x: np.ndarray) -> np.ndarray:
        if self.transition_model is None:
            raise RuntimeError("Transition model is not initialized.")
        if torch is None:
            raise RuntimeError("Transition model requires PyTorch.")
        if self.transition_device is None:
            self.transition_device = torch.device(self._transition_device_str or "cpu")
            self.transition_model.to(self.transition_device)
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.transition_device).reshape(
            1, -1
        )
        self.transition_model.eval()
        with torch.no_grad():
            logits = self.transition_model(x_t)
        return logits.squeeze(0).detach().cpu().numpy()

    def _context_transition(self, context_x: np.ndarray) -> np.ndarray:
        x = np.asarray(context_x, dtype=float).reshape(-1)
        if self.transition_model is not None or self.transition_hidden_dims is not None:
            self._ensure_transition_model(x.shape[0])
            logits = self._transition_logits_numpy(x)
            return self._softmax_rows(logits)
        if self.transition_mode == "linear":
            if self.W_lin is None or self.b_lin is None:
                self._init_transition_params(x.shape[0])
            if self.context_dim is None or x.shape[0] != self.context_dim:
                raise ValueError("Context dimension mismatch for transition parameters.")
            scores = np.einsum("mjd,d->mj", self.W_lin, x) + self.b_lin
            return self._softmax_rows(scores)
        if self.W_q is None or self.W_k is None:
            self._init_transition_params(x.shape[0])
        if self.context_dim is None or x.shape[0] != self.context_dim:
            raise ValueError("Context dimension mismatch for transition parameters.")
        q = np.einsum("mad,d->ma", self.W_q, x)
        k = np.einsum("mad,d->ma", self.W_k, x)
        scores = (q @ k.T) / np.sqrt(float(self.attn_dim))
        if self.b_attn is not None:
            scores = scores + self.b_attn
        return self._softmax_rows(scores)

    def reset_beliefs(self) -> None:
        self.w = np.ones(self.M, dtype=float) / float(self.M)
        self.mu_g = self.g_mean0.copy()
        self.Sigma_g = self.g_cov0.copy()
        self.mu_u = {}
        self.Sigma_u = {}
        self.registry = []
        self.last_selected_time = {}
        self.current_step = 0
        self._clear_em_history()

    def configure_online_em(
        self,
        enabled: bool,
        window: int,
        period: int,
        n_em: int,
        n_samples: int,
        burn_in: int,
        epsilon_N: float,
        theta_lr: float,
        theta_steps: int,
        priors: Optional[dict] = None,
        start_t: Optional[int] = None,
        seed: Optional[int] = None,
        print_val_loss: bool = False,
        use_state_priors: bool = True,
    ) -> None:
        self._online_em_cfg = {
            "enabled": bool(enabled),
            "window": int(window),
            "period": int(period),
            "n_em": int(n_em),
            "n_samples": int(n_samples),
            "burn_in": int(burn_in),
            "epsilon_N": float(epsilon_N),
            "theta_lr": float(theta_lr),
            "theta_steps": int(theta_steps),
            "priors": priors,
            "start_t": None if start_t is None else int(start_t),
            "seed": None if seed is None else int(seed),
            "print_val_loss": bool(print_val_loss),
            "use_state_priors": bool(use_state_priors),
        }

    def set_online_em_enabled(self, enabled: bool) -> None:
        self._online_em_cfg["enabled"] = bool(enabled)

    def suspend_online_em(self, suspend: bool = True) -> None:
        self._online_em_suspended = bool(suspend)

    def _clear_em_history(self) -> None:
        self._em_records = []
        self._em_records_start = None

    # -------------------------------------------------------------------------
    # Debug Diagnostics for Tracking Underperformance
    # -------------------------------------------------------------------------

    def configure_debug_diagnostics(
        self,
        enabled: bool = True,
        print_stride: int = 100,
        log_weights: bool = True,
        log_variances: bool = True,
        log_info_gain: bool = True,
        log_em_params: bool = True,
        weight_collapse_threshold: float = 0.99,
        variance_collapse_threshold: float = 1e-6,
    ) -> None:
        """Enable debug diagnostics to track regime weights, variances, and EM params."""
        self._debug_diag_cfg = {
            "enabled": bool(enabled),
            "print_stride": max(int(print_stride), 1),
            "log_weights": bool(log_weights),
            "log_variances": bool(log_variances),
            "log_info_gain": bool(log_info_gain),
            "log_em_params": bool(log_em_params),
            "weight_collapse_threshold": float(weight_collapse_threshold),
            "variance_collapse_threshold": float(variance_collapse_threshold),
        }
        self._debug_diag_history = []

    def _log_debug_diagnostics(
        self,
        t_now: int,
        w_post: np.ndarray,
        costs: Optional[Dict[int, float]] = None,
        info_gains: Optional[Dict[int, float]] = None,
        stats: Optional[Dict[int, dict]] = None,
    ) -> None:
        """Log diagnostic information about regime weights, variances, and info gains."""
        cfg = self._debug_diag_cfg
        if not cfg.get("enabled", False):
            return

        stride = cfg.get("print_stride", 100)
        should_print = (t_now % stride == 0)

        record: Dict[str, Any] = {"t": t_now}

        # Log regime weights
        if cfg.get("log_weights", True):
            w_arr = np.asarray(w_post, dtype=float)
            max_w = float(np.max(w_arr))
            entropy = -float(np.sum(w_arr * np.log(np.maximum(w_arr, self.eps))))
            record["w_max"] = max_w
            record["w_entropy"] = entropy
            record["w_argmax"] = int(np.argmax(w_arr))

            if should_print:
                w_str = np.array2string(w_arr, precision=4, separator=", ")
                print(f"[FactorizedSLDS DEBUG] t={t_now} w={w_str} entropy={entropy:.4f}")

            # Warn if weights are collapsing
            thresh = cfg.get("weight_collapse_threshold", 0.99)
            if max_w > thresh:
                print(
                    f"[FactorizedSLDS WARNING] t={t_now} regime weights collapsing: "
                    f"max_w={max_w:.6f} > {thresh}"
                )

        # Log predictive variances
        if cfg.get("log_variances", True) and stats is not None:
            var_thresh = cfg.get("variance_collapse_threshold", 1e-6)
            for k, stat in stats.items():
                var_modes = stat.get("var", None)
                if var_modes is not None:
                    min_var = float(np.min(var_modes))
                    record[f"var_min_k{k}"] = min_var
                    if should_print:
                        var_str = np.array2string(var_modes, precision=6, separator=", ")
                        print(f"[FactorizedSLDS DEBUG] t={t_now} k={k} var_modes={var_str}")
                    if min_var < var_thresh:
                        print(
                            f"[FactorizedSLDS WARNING] t={t_now} k={k} variance collapse: "
                            f"min_var={min_var:.2e} < {var_thresh:.2e}"
                        )

        # Log information gain
        if cfg.get("log_info_gain", True) and info_gains is not None:
            ig_vals = np.array(list(info_gains.values()), dtype=float)
            record["ig_mean"] = float(np.mean(ig_vals)) if ig_vals.size else 0.0
            record["ig_max"] = float(np.max(ig_vals)) if ig_vals.size else 0.0
            if should_print:
                ig_str = ", ".join(f"k{k}={v:.6f}" for k, v in info_gains.items())
                print(f"[FactorizedSLDS DEBUG] t={t_now} info_gains: {ig_str}")

        # Log costs
        if costs is not None:
            cost_vals = np.array(list(costs.values()), dtype=float)
            record["cost_min"] = float(np.min(cost_vals)) if cost_vals.size else 0.0
            if should_print:
                cost_str = ", ".join(f"k{k}={v:.4f}" for k, v in costs.items())
                print(f"[FactorizedSLDS DEBUG] t={t_now} costs: {cost_str}")

        self._debug_diag_history.append(record)

    def _check_dynamics_stability(self, warn: bool = True) -> Dict[str, Any]:
        """Check if dynamics matrices A_g and A_u have stable eigenvalues (< 1).

        Returns dict with stability info. If eigenvalues > 1, system is unstable
        and variances will explode over time.
        """
        result = {"stable": True, "A_g_max_eig": [], "A_u_max_eig": []}

        for m in range(self.M):
            # Check A_g stability
            if self.d_g > 0:
                eig_g = np.abs(np.linalg.eigvals(self.A_g[m]))
                max_eig_g = float(np.max(eig_g))
                result["A_g_max_eig"].append(max_eig_g)
                if max_eig_g > 1.0:
                    result["stable"] = False
                    if warn:
                        print(
                            f"[FactorizedSLDS WARNING] A_g[{m}] is UNSTABLE: "
                            f"max eigenvalue = {max_eig_g:.4f} > 1.0"
                        )

            # Check A_u stability
            if self.d_phi > 0:
                eig_u = np.abs(np.linalg.eigvals(self.A_u[m]))
                max_eig_u = float(np.max(eig_u))
                result["A_u_max_eig"].append(max_eig_u)
                if max_eig_u > 1.0:
                    result["stable"] = False
                    if warn:
                        print(
                            f"[FactorizedSLDS WARNING] A_u[{m}] is UNSTABLE: "
                            f"max eigenvalue = {max_eig_u:.4f} > 1.0"
                        )

        return result

    def _snapshot_em_params(self) -> Dict[str, Any]:
        """Snapshot current EM parameters for comparison."""
        return {
            "A_g": self.A_g.copy(),
            "Q_g": self.Q_g.copy(),
            "A_u": self.A_u.copy(),
            "Q_u": self.Q_u.copy(),
            "R": float(self.R) if np.ndim(self.R) == 0 else self.R.copy(),
            "B_dict": {k: v.copy() for k, v in self.B_dict.items()},
        }

    def _log_em_params_before(self) -> None:
        """Call before EM to snapshot parameters."""
        if not self._debug_diag_cfg.get("enabled", False):
            return
        if not self._debug_diag_cfg.get("log_em_params", True):
            return
        self._em_params_before = self._snapshot_em_params()

    def _log_em_params_after(self, stage: str = "EM") -> None:
        """Call after EM to compare and log parameter changes."""
        cfg = self._debug_diag_cfg
        if not cfg.get("enabled", False):
            return
        if not cfg.get("log_em_params", True):
            return
        if self._em_params_before is None:
            return

        before = self._em_params_before
        after = self._snapshot_em_params()

        def _param_change(name: str, before_val: np.ndarray, after_val: np.ndarray) -> str:
            before_arr = np.asarray(before_val, dtype=float)
            after_arr = np.asarray(after_val, dtype=float)
            diff = np.abs(after_arr - before_arr)
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            return f"{name}: max_delta={max_diff:.6f}, mean_delta={mean_diff:.6f}"

        print(f"[FactorizedSLDS DEBUG] {stage} parameter changes:")
        print(f"  {_param_change('A_g', before['A_g'], after['A_g'])}")
        print(f"  {_param_change('Q_g', before['Q_g'], after['Q_g'])}")
        print(f"  {_param_change('A_u', before['A_u'], after['A_u'])}")
        print(f"  {_param_change('Q_u', before['Q_u'], after['Q_u'])}")

        # Log R changes
        R_before = before['R']
        R_after = after['R']
        if np.ndim(R_before) == 0:
            print(f"  R: before={float(R_before):.6f}, after={float(R_after):.6f}")
        else:
            print(f"  {_param_change('R', R_before, R_after)}")

        # Log B_dict changes for shared experts
        for k in before['B_dict']:
            if k in after['B_dict']:
                print(f"  {_param_change(f'B[{k}]', before['B_dict'][k], after['B_dict'][k])}")

        # Check dynamics stability after EM
        stability = self._check_dynamics_stability(warn=True)
        if stability["stable"]:
            print(f"  Dynamics stability: OK (all eigenvalues < 1)")
        else:
            print(f"  Dynamics stability: UNSTABLE - variance explosion likely!")

        self._em_params_before = None

    def get_debug_diagnostics_history(self) -> List[Dict[str, Any]]:
        """Return the debug diagnostics history for analysis."""
        return self._debug_diag_history.copy()

    def _snapshot_em_state(self) -> Dict[str, Any]:
        return {
            "w": self.w.copy(),
            "mu_g": self.mu_g.copy(),
            "Sigma_g": self.Sigma_g.copy(),
            "mu_u": {k: v.copy() for k, v in self.mu_u.items()},
            "Sigma_u": {k: v.copy() for k, v in self.Sigma_u.items()},
            "registry": list(self.registry),
            "last_selected_time": dict(self.last_selected_time),
            "current_step": int(self.current_step),
        }

    def _record_em_observation(
        self,
        x_t: Optional[np.ndarray],
        available_experts: Sequence[int],
        action: int,
        residual: float,
        residuals_full: Optional[np.ndarray],
        state_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = self._online_em_cfg
        if not cfg.get("enabled", False) or self._online_em_suspended:
            return
        if x_t is None:
            return
        if not self._em_records:
            self._em_records_start = int(self.current_step)
        state_prior = None
        if cfg.get("use_state_priors", True):
            if state_override is None:
                state_prior = self._snapshot_em_state()
            else:
                state_prior = state_override
        state_post = None
        if cfg.get("use_state_priors", True):
            state_post = self._snapshot_em_state()
        record = {
            "t": int(self.current_step),
            "context": np.asarray(x_t, dtype=float).reshape(-1).copy(),
            "available": [int(k) for k in available_experts],
            "action": int(action),
            "residual": float(residual),
            "residual_full": None
            if residuals_full is None
            else np.asarray(residuals_full, dtype=float).copy(),
            "state_prior": state_prior,
            "state_post": state_post,
        }
        self._em_records.append(record)

    def _maybe_online_em_update(self) -> None:
        cfg = self._online_em_cfg
        if not cfg.get("enabled", False) or self._online_em_suspended:
            return
        window = int(cfg.get("window", 0))
        period = int(cfg.get("period", 1))
        if window <= 0:
            return
        t_now = int(self.current_step)
        start_t = cfg.get("start_t", None)
        if start_t is not None and t_now < int(start_t):
            return
        if t_now < window:
            return
        if period > 0 and (t_now % period) != 0:
            return
        if len(self._em_records) < window:
            return

        window_records = self._em_records[-window:]
        if cfg.get("print_val_loss", False):
            t_start = int(window_records[0]["t"])
            t_end = int(window_records[-1]["t"])
            print(
                f"[FactorizedSLDS EM] online update t={t_now} "
                f"window=[{t_start},{t_end}] n_em={cfg.get('n_em', 1)}"
            )
        contexts = [rec["context"] for rec in window_records]
        available_sets = [rec["available"] for rec in window_records]
        actions = [rec["action"] for rec in window_records]
        residuals = [rec["residual"] for rec in window_records]
        residuals_full = None
        if self.feedback_mode == "full":
            residuals_full = [rec["residual_full"] for rec in window_records]

        init_state = None
        if cfg.get("use_state_priors", True):
            init_state = window_records[0].get("state_prior", None)
            if init_state is None:
                init_state = window_records[0].get("state_post", None)

        seed_cfg = cfg.get("seed", None)
        seed = None if seed_cfg is None else int(seed_cfg) + int(t_now)
        self.fit_em(
            contexts=contexts,
            available_sets=available_sets,
            actions=actions,
            residuals=residuals,
            residuals_full=residuals_full,
            n_em=int(cfg.get("n_em", 1)),
            n_samples=int(cfg.get("n_samples", 1)),
            burn_in=int(cfg.get("burn_in", 0)),
            val_fraction=0.0,
            val_len=0,
            priors=cfg.get("priors", None),
            theta_lr=float(cfg.get("theta_lr", 1e-2)),
            theta_steps=int(cfg.get("theta_steps", 1)),
            seed=seed,
            print_val_loss=bool(cfg.get("print_val_loss", False)),
            epsilon_N=float(cfg.get("epsilon_N", 0.0)),
            init_state=init_state,
            use_validation=False,
            set_em_tk=False,
            transition_log_stage=f"post_online_em t={t_now}",
        )
        t_start = max(int(window_records[0]["t"]) - 1, 0)
        self._refresh_filter_state_from_window(
            contexts=contexts,
            available_sets=available_sets,
            actions=actions,
            residuals=residuals,
            residuals_full=residuals_full,
            init_state=init_state,
            t_start=t_start,
        )

    def _apply_em_init_state(
        self,
        init_state: Optional[Dict[str, Any]],
        t_start: int,
    ) -> None:
        if init_state is None:
            self.w = np.ones(self.M, dtype=float) / float(self.M)
            self.mu_g = self.g_mean0.copy()
            self.Sigma_g = self.g_cov0.copy()
            self.mu_u = {}
            self.Sigma_u = {}
            self.registry = []
            self.last_selected_time = {}
            self.current_step = int(t_start)
            return

        w_init = np.asarray(init_state.get("w", np.ones(self.M)), dtype=float).copy()
        if w_init.shape != (self.M,):
            raise ValueError("init_state['w'] must have shape (M,).")
        w_init = np.maximum(w_init, self.eps)
        w_init = w_init / np.sum(w_init)

        mu_g_init = self._normalize_mean_modes(init_state.get("mu_g", None), self.d_g)
        Sigma_g_init = self._normalize_cov_modes(
            init_state.get("Sigma_g", None), self.d_g, default_scale=1.0
        )

        mu_u_in = init_state.get("mu_u", {}) or {}
        Sigma_u_in = init_state.get("Sigma_u", {}) or {}
        mu_u_init = {
            int(k): self._normalize_mean_modes(v, self.d_phi) for k, v in mu_u_in.items()
        }
        Sigma_u_init = {
            int(k): self._normalize_cov_modes(v, self.d_phi, default_scale=1.0)
            for k, v in Sigma_u_in.items()
        }

        if init_state.get("registry", None) is None:
            registry = sorted(mu_u_init.keys())
        else:
            registry = [int(k) for k in init_state.get("registry", [])]
        for k in registry:
            if k not in mu_u_init:
                mu_u_init[k] = self.u_mean0.copy()
            if k not in Sigma_u_init:
                Sigma_u_init[k] = self.u_cov0.copy()

        last_selected = init_state.get("last_selected_time", None)
        if last_selected is None:
            last_selected_time = {int(k): int(t_start) for k in registry}
        else:
            last_selected_time = {int(k): int(v) for k, v in last_selected.items()}
            for k in registry:
                last_selected_time.setdefault(int(k), int(t_start))

        self.w = w_init
        self.mu_g = mu_g_init.copy()
        self.Sigma_g = Sigma_g_init.copy()
        self.mu_u = {k: v.copy() for k, v in mu_u_init.items()}
        self.Sigma_u = {k: v.copy() for k, v in Sigma_u_init.items()}
        self.registry = list(registry)
        self.last_selected_time = last_selected_time
        self.current_step = int(init_state.get("current_step", t_start))

    def _refresh_filter_state_from_window(
        self,
        contexts: Sequence[np.ndarray],
        available_sets: Sequence[Sequence[int]],
        actions: Sequence[int],
        residuals: Sequence[float],
        residuals_full: Optional[Sequence[np.ndarray]],
        init_state: Optional[Dict[str, Any]],
        t_start: int,
    ) -> None:
        # Align filtering beliefs with the updated parameters after online EM.
        state_before = self._snapshot_state()
        try:
            self._apply_em_init_state(init_state, t_start)
            self._filter_block_nll(
                contexts,
                available_sets,
                actions,
                residuals,
                residuals_full=residuals_full,
                accumulate=False,
            )
        except Exception:
            self._restore_state(state_before)
            raise

    def _interaction_and_time_update(
        self,
        context_x: np.ndarray,
        w_prev: np.ndarray,
        mu_g_prev: np.ndarray,
        Sigma_g_prev: np.ndarray,
        mu_u_prev: Dict[int, np.ndarray],
        Sigma_u_prev: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        Pi = self._context_transition(context_x)
        w_pred = Pi.T @ w_prev
        w_pred = np.maximum(w_pred, self.eps)
        w_pred = w_pred / np.sum(w_pred)

        mu_ij = np.zeros((self.M, self.M), dtype=float)
        for j in range(self.M):
            denom = w_pred[j]
            if denom <= self.eps:
                mu_ij[:, j] = 1.0 / self.M
            else:
                mu_ij[:, j] = Pi[:, j] * w_prev
                s = mu_ij[:, j].sum()
                if s <= self.eps:
                    mu_ij[:, j] = 1.0 / self.M
                else:
                    mu_ij[:, j] /= s

        if self.d_g == 0:
            mixed_mu_g = np.zeros((self.M, 0), dtype=float)
            mixed_Sigma_g = np.zeros((self.M, 0, 0), dtype=float)
        else:
            mu_g_prev = np.clip(mu_g_prev, -self.state_clip, self.state_clip)
            mixed_mu_g = np.zeros_like(mu_g_prev)
            mixed_Sigma_g = np.zeros_like(Sigma_g_prev)
            for j in range(self.M):
                for i in range(self.M):
                    w_ij = mu_ij[i, j]
                    mixed_mu_g[j] += w_ij * mu_g_prev[i]
                for i in range(self.M):
                    w_ij = mu_ij[i, j]
                    diff = mu_g_prev[i] - mixed_mu_g[j]
                    diff = np.clip(diff, -self.state_clip, self.state_clip)
                    mixed_Sigma_g[j] += w_ij * (
                        Sigma_g_prev[i] + np.outer(diff, diff)
                    )
                mixed_mu_g[j] = self._sanitize_vector(mixed_mu_g[j], self.g_mean0[j])
                mixed_Sigma_g[j] = self._sanitize_cov(mixed_Sigma_g[j], self.g_cov0[j])

        mixed_mu_u: Dict[int, np.ndarray] = {}
        mixed_Sigma_u: Dict[int, np.ndarray] = {}
        for k in self.registry:
            mu_modes = np.clip(mu_u_prev[k], -self.state_clip, self.state_clip)
            Sigma_modes = Sigma_u_prev[k]
            mixed_mu_u[k] = np.zeros_like(mu_modes)
            mixed_Sigma_u[k] = np.zeros_like(Sigma_modes)
            for j in range(self.M):
                for i in range(self.M):
                    w_ij = mu_ij[i, j]
                    mixed_mu_u[k][j] += w_ij * mu_modes[i]
                for i in range(self.M):
                    w_ij = mu_ij[i, j]
                    diff = mu_modes[i] - mixed_mu_u[k][j]
                    diff = np.clip(diff, -self.state_clip, self.state_clip)
                    mixed_Sigma_u[k][j] += w_ij * (
                        Sigma_modes[i] + np.outer(diff, diff)
                    )
                mixed_mu_u[k][j] = self._sanitize_vector(
                    mixed_mu_u[k][j], self.u_mean0[j]
                )
                mixed_Sigma_u[k][j] = self._sanitize_cov(
                    mixed_Sigma_u[k][j], self.u_cov0[j]
                )

        mu_g_pred = np.zeros_like(mixed_mu_g)
        Sigma_g_pred = np.zeros_like(mixed_Sigma_g)
        mu_u_pred: Dict[int, np.ndarray] = {}
        Sigma_u_pred: Dict[int, np.ndarray] = {}

        if self.d_g > 0:
            for m in range(self.M):
                mu_g_pred[m] = self.A_g[m] @ mixed_mu_g[m]
                Sigma_g_pred[m] = (
                    self.A_g[m] @ mixed_Sigma_g[m] @ self.A_g[m].T + self.Q_g[m]
                )
                Sigma_g_pred[m] = self._sanitize_cov(Sigma_g_pred[m], self.g_cov0[m])
                Sigma_g_pred[m] = 0.5 * (
                    Sigma_g_pred[m] + Sigma_g_pred[m].T
                ) + self.eps * np.eye(self.d_g, dtype=float)

        for k in self.registry:
            mu_u_pred[k] = np.zeros_like(mixed_mu_u[k])
            Sigma_u_pred[k] = np.zeros_like(mixed_Sigma_u[k])
            for m in range(self.M):
                mu_u_pred[k][m] = self.A_u[m] @ mixed_mu_u[k][m]
                Sigma_u_pred[k][m] = (
                    self.A_u[m] @ mixed_Sigma_u[k][m] @ self.A_u[m].T + self.Q_u[m]
                )
                Sigma_u_pred[k][m] = self._sanitize_cov(
                    Sigma_u_pred[k][m], self.u_cov0[m]
                )
                Sigma_u_pred[k][m] = 0.5 * (
                    Sigma_u_pred[k][m] + Sigma_u_pred[k][m].T
                ) + self.eps * np.eye(self.d_phi, dtype=float)

        return w_pred, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred

    def _compute_predictive_stats(
        self,
        phi: np.ndarray,
        w_pred: np.ndarray,
        mu_g_pred: np.ndarray,
        Sigma_g_pred: np.ndarray,
        mu_u_pred: Dict[int, np.ndarray],
        Sigma_u_pred: Dict[int, np.ndarray],
        available_experts: Sequence[int],
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, dict]]:
        costs: Dict[int, float] = {}
        info_gains: Dict[int, float] = {}
        stats: Dict[int, dict] = {}

        for k in available_experts:
            k = int(k)
            if k not in mu_u_pred or k not in Sigma_u_pred:
                continue
            if self.d_g > 0:
                Bk = self._ensure_B(k)
                h = Bk.T @ phi
            else:
                h = np.zeros(0, dtype=float)
            mean_modes = np.zeros(self.M, dtype=float)
            var_modes = np.zeros(self.M, dtype=float)
            loss_modes = np.zeros(self.M, dtype=float)
            signal_modes = np.zeros(self.M, dtype=float)
            noise_modes = np.zeros(self.M, dtype=float)
            b_modes = np.zeros(self.M, dtype=float)
            s_modes = np.zeros(self.M, dtype=float)
            ig_modes = np.zeros(self.M, dtype=float)
            for m in range(self.M):
                mu_u_m = mu_u_pred[k][m]
                Sigma_u_m = Sigma_u_pred[k][m]
                b_m = float(np.squeeze(phi @ mu_u_m))
                s_m = float(np.squeeze(phi @ (Sigma_u_m @ phi))) + self._get_R(m, k)
                s_m = max(s_m, self.eps)
                if self.d_g > 0:
                    mu_g_m = mu_g_pred[m]
                    Sigma_g_m = Sigma_g_pred[m]
                    mean_m = float(np.squeeze(h @ mu_g_m)) + b_m
                    signal = float(np.squeeze(h @ (Sigma_g_m @ h)))
                    signal = max(signal, 0.0)
                else:
                    mean_m = b_m
                    signal = 0.0
                mean_modes[m] = mean_m
                var_modes[m] = signal + s_m
                loss_modes[m] = self._expected_loss_gaussian(mean_modes[m], var_modes[m])
                signal_modes[m] = signal
                noise_modes[m] = s_m
                b_modes[m] = b_m
                s_modes[m] = s_m
                ig_modes[m] = 0.5 * np.log1p(signal / s_m)

            cost = float(w_pred @ loss_modes) + self._get_beta(k)
            ig = float(w_pred @ ig_modes)

            costs[k] = cost
            info_gains[k] = ig
            stats[k] = {
                "h": h,
                "mean": mean_modes,
                "var": var_modes,
                "signal": signal_modes,
                "noise": noise_modes,
                "b": b_modes,
                "s": s_modes,
            }

        return costs, info_gains, stats

    def _score_experts(
        self,
        phi: np.ndarray,
        w_pred: np.ndarray,
        mu_g_pred: np.ndarray,
        Sigma_g_pred: np.ndarray,
        mu_u_pred: Dict[int, np.ndarray],
        Sigma_u_pred: Dict[int, np.ndarray],
        available_experts: Sequence[int],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        costs, info_gains, _ = self._compute_predictive_stats(
            phi,
            w_pred,
            mu_g_pred,
            Sigma_g_pred,
            mu_u_pred,
            Sigma_u_pred,
            available_experts,
        )
        return costs, info_gains

    def _gaussian_logpdf_vec(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
    ) -> np.ndarray:
        var = np.maximum(var, self.eps)
        return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var)

    def _apply_loss_fn(self, x: np.ndarray) -> np.ndarray:
        if self.loss_fn is None:
            return np.asarray(x, dtype=float) ** 2
        try:
            return np.asarray(self.loss_fn(x), dtype=float)
        except Exception:
            flat = np.asarray(x, dtype=float).reshape(-1)
            vals = np.array([self.loss_fn(float(v)) for v in flat], dtype=float)
            return vals.reshape(np.asarray(x, dtype=float).shape)

    def _expected_loss_gaussian(self, mean: float, var: float) -> float:
        var = max(float(var), self.eps)
        if self.loss_fn is None:
            return var + float(mean) ** 2
        if self.loss_quad_points <= 0 or var <= self.eps:
            return float(self._apply_loss_fn(np.array([mean], dtype=float))[0])
        xs, ws = np.polynomial.hermite.hermgauss(self.loss_quad_points)
        samples = float(mean) + np.sqrt(2.0 * var) * xs
        vals = self._apply_loss_fn(samples)
        return float(np.sum(ws * vals) / np.sqrt(np.pi))

    def _logsumexp_vec(self, values: np.ndarray, axis: int = 0) -> np.ndarray:
        vmax = np.max(values, axis=axis, keepdims=True)
        return (vmax + np.log(np.sum(np.exp(values - vmax), axis=axis, keepdims=True))).squeeze(
            axis
        )

    def _estimate_iz(
        self,
        mean_modes: np.ndarray,
        var_modes: np.ndarray,
        w_pred: np.ndarray,
        n_samples: int,
    ) -> float:
        M = int(w_pred.shape[0])
        if M <= 1 or n_samples <= 0:
            return 0.0
        log_w = np.log(np.maximum(w_pred, self.eps))
        ig = 0.0
        for m in range(M):
            weight = float(w_pred[m])
            if weight <= 0.0:
                continue
            samples = self._rng.normal(
                loc=float(mean_modes[m]),
                scale=float(np.sqrt(var_modes[m])),
                size=int(n_samples),
            )
            log_p_m = self._gaussian_logpdf_vec(
                samples, float(mean_modes[m]), float(var_modes[m])
            )
            log_p_all = self._gaussian_logpdf_vec(
                samples[None, :], mean_modes[:, None], var_modes[:, None]
            )
            log_mix = self._logsumexp_vec(log_w[:, None] + log_p_all, axis=0)
            ig += weight * float(np.mean(log_p_m - log_mix))
        return max(float(ig), 0.0)

    def _compute_iz_scores(
        self,
        stats: Dict[int, dict],
        w_pred: np.ndarray,
        n_samples: int,
    ) -> Dict[int, float]:
        iz_scores: Dict[int, float] = {}
        if self.M <= 1 or n_samples <= 0:
            for k in stats:
                iz_scores[int(k)] = 0.0
            return iz_scores
        for k, stat in stats.items():
            iz_scores[int(k)] = self._estimate_iz(
                stat["mean"], stat["var"], w_pred, n_samples
            )
        return iz_scores

    def _ucb_alpha(self, t_now: int) -> float:
        t_val = max(int(t_now), 1)
        if self.exploration_ucb_schedule == "fixed":
            alpha = (
                float(self.exploration_ucb_alpha)
                if self.exploration_ucb_alpha is not None
                else 0.1
            )
        elif self.exploration_ucb_schedule == "inverse_t_log2":
            denom = t_val * (np.log(float(t_val) + 1.0) ** 2)
            alpha = 0.5 - 0.5 / max(denom, 1.0)
        else:
            alpha = 0.5 - 0.5 / t_val
        if self.exploration_ucb_alpha is not None and self.exploration_ucb_schedule != "fixed":
            alpha = min(alpha, float(self.exploration_ucb_alpha))
        if self.exploration_ucb_schedule == "fixed":
            return float(np.clip(alpha, self.eps, 1.0 - self.eps))
        return float(np.clip(alpha, self.eps, 0.5 - self.eps))

    def _sample_predictive_costs_from_stats(
        self,
        w_pred: np.ndarray,
        mu_g_pred: np.ndarray,
        Sigma_g_pred: np.ndarray,
        stats: Dict[int, dict],
        available_experts: Sequence[int],
        n_samples: int,
        deterministic: bool,
    ) -> Tuple[np.ndarray, List[int]]:
        avail = [int(k) for k in available_experts if int(k) in stats]
        if not avail or n_samples <= 0:
            return np.zeros((0, 0), dtype=float), avail
        H = np.stack([stats[k]["h"] for k in avail], axis=0)
        b_modes = np.stack([stats[k]["b"] for k in avail], axis=0)
        s_modes = np.stack([stats[k]["s"] for k in avail], axis=0)
        beta = np.array([self._get_beta(k) for k in avail], dtype=float)

        costs = np.zeros((int(n_samples), len(avail)), dtype=float)
        for n in range(int(n_samples)):
            z = int(self._rng.choice(self.M, p=w_pred))
            if self.d_g > 0:
                g = self._rng.multivariate_normal(mu_g_pred[z], Sigma_g_pred[z])
                mean = H @ g + b_modes[:, z]
            else:
                mean = b_modes[:, z]
            s = np.maximum(s_modes[:, z], self.eps)
            if deterministic:
                if self.loss_fn is None:
                    cost = s + mean**2
                else:
                    cost = np.array(
                        [
                            self._expected_loss_gaussian(float(mean[i]), float(s[i]))
                            for i in range(len(mean))
                        ],
                        dtype=float,
                    )
            else:
                e = mean + self._rng.normal(scale=np.sqrt(s), size=mean.shape)
                cost = self._apply_loss_fn(e)
            costs[n] = cost + beta
        return costs, avail

    def _select_ids(
        self,
        costs: Dict[int, float],
        info_gains: Dict[int, float],
    ) -> int:
        """Select expert using IDS rule: argmin_k Delta(k)^2 / IG(k).

        Numerical stability improvements:
        - If IG <= eps: score = 0 if Delta=0, else inf
        - If IG is very small (< 1e-4), cap the ratio to prevent explosion
        - The ratio is clipped to a maximum value to avoid numerical issues
        """
        min_cost = min(costs.values())
        scores: Dict[int, float] = {}

        # Maximum score threshold to prevent numerical instability
        max_score = 1e12

        for k, cost in costs.items():
            delta = cost - min_cost
            ig = info_gains.get(k, 0.0)

            if ig <= self.eps:
                # No information gain: prefer this expert only if it's already optimal
                scores[k] = 0.0 if abs(delta) <= self.eps else np.inf
            else:
                # Standard IDS ratio with numerical safeguards
                ratio = (delta ** 2) / ig

                # Cap extremely large ratios that could cause numerical issues
                # This happens when IG is very small but positive
                if ratio > max_score:
                    if self._debug_diag_cfg.get("enabled", False):
                        print(
                            f"[FactorizedSLDS WARNING] IDS ratio capped for k={k}: "
                            f"delta={delta:.6f}, ig={ig:.2e}, ratio={ratio:.2e} -> {max_score:.2e}"
                        )
                    ratio = max_score

                scores[k] = ratio

        if all(np.isinf(score) for score in scores.values()):
            # All experts have inf score - fall back to greedy selection
            return int(min(costs, key=costs.get))

        return int(min(scores, key=scores.get))

    def _select_expert_from_stats(
        self,
        w_pred: np.ndarray,
        mu_g_pred: np.ndarray,
        Sigma_g_pred: np.ndarray,
        costs: Dict[int, float],
        info_gains_g: Dict[int, float],
        stats: Dict[int, dict],
        available_experts: Sequence[int],
        t_now: int,
        iz_scores: Optional[Dict[int, float]] = None,
    ) -> int:
        if self.feedback_mode == "full":
            return int(min(costs, key=costs.get))

        if self.exploration == "sampling":
            cost_samples, avail = self._sample_predictive_costs_from_stats(
                w_pred,
                mu_g_pred,
                Sigma_g_pred,
                stats,
                available_experts,
                n_samples=1,
                deterministic=self.exploration_sampling_deterministic,
            )
            if cost_samples.size == 0:
                return int(min(costs, key=costs.get))
            idx = int(np.argmin(cost_samples[0]))
            return int(avail[idx])

        if self.exploration == "ucb":
            n_samples = int(self.exploration_ucb_samples)
            cost_samples, avail = self._sample_predictive_costs_from_stats(
                w_pred,
                mu_g_pred,
                Sigma_g_pred,
                stats,
                available_experts,
                n_samples=n_samples,
                deterministic=False,
            )
            if cost_samples.size == 0:
                return int(min(costs, key=costs.get))
            alpha_t = self._ucb_alpha(t_now)
            quantiles = np.quantile(cost_samples, alpha_t, axis=0)
            idx = int(np.argmin(quantiles))
            return int(avail[idx])

        if self.exploration == "g_z":
            info_gains: Dict[int, float] = {}
            if iz_scores is None:
                iz_scores = self._compute_iz_scores(
                    stats, w_pred, self.exploration_mc_samples
                )
            for k in stats:
                iz = iz_scores.get(k, 0.0)
                info_gains[k] = max(iz, 0.0) + info_gains_g.get(k, 0.0)
            return self._select_ids(costs, info_gains)

        return self._select_ids(costs, info_gains_g)

    def select_expert(
        self,
        x_t: np.ndarray,
        available_experts: Sequence[int],
    ) -> Tuple[int, dict]:
        avail_arr = np.asarray(list(available_experts), dtype=int)
        if avail_arr.size == 0:
            raise ValueError("No available experts to select from.")

        t_now = self.current_step + 1
        entering = self.manage_registry(avail_arr, t_now=t_now)
        phi = self._compute_phi(x_t)

        w_pred, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred = self._interaction_and_time_update(
            x_t,
            self.w,
            self.mu_g,
            self.Sigma_g,
            self.mu_u,
            self.Sigma_u,
        )
        for k in entering:
            mu_u_pred[int(k)] = self.u_mean0.copy()
            Sigma_u_pred[int(k)] = self.u_cov0.copy()

        costs, info_gains, stats = self._compute_predictive_stats(
            phi,
            w_pred,
            mu_g_pred,
            Sigma_g_pred,
            mu_u_pred,
            Sigma_u_pred,
            avail_arr,
        )
        if not costs:
            raise ValueError("No expert statistics available for selection.")
        iz_scores = None
        if self.exploration_diag_enabled:
            iz_scores = self._compute_iz_scores(
                stats, w_pred, self.exploration_diag_samples
            )

        r_t = self._select_expert_from_stats(
            w_pred,
            mu_g_pred,
            Sigma_g_pred,
            costs,
            info_gains,
            stats,
            avail_arr,
            t_now,
            iz_scores=iz_scores,
        )
        if self.exploration_diag_enabled:
            iz_vals = np.array(
                [iz_scores.get(int(k), 0.0) for k in avail_arr], dtype=float
            )
            iz_avg = float(np.mean(iz_vals)) if iz_vals.size else 0.0
            iz_max = float(np.max(iz_vals)) if iz_vals.size else 0.0
            iz_sel = float(iz_scores.get(int(r_t), 0.0))
            if self.exploration_diag_max_records > 0:
                if len(self.exploration_diag_records) < self.exploration_diag_max_records:
                    self.exploration_diag_records.append((t_now, iz_avg, iz_max, iz_sel))
            if self.exploration_diag_print and ((t_now - 1) % self.exploration_diag_stride == 0):
                label = getattr(self, "transition_log_label", None)
                if not label:
                    label = "FactorizedSLDS"
                print(
                    f"[Exploration diag] {label} t={t_now} "
                    f"iz_avg={iz_avg:.6f} iz_max={iz_max:.6f} iz_sel={iz_sel:.6f}"
                )

        # Debug diagnostics: log predictive stats before selection
        self._log_debug_diagnostics(
            t_now=t_now,
            w_post=w_pred,  # w_pred at selection time (before observation)
            costs=costs,
            info_gains=info_gains,
            stats=stats,
        )

        cache = {
            "x_t": np.asarray(x_t, dtype=float),
            "phi": phi,
            "w_pred": w_pred,
            "mu_g_pred": mu_g_pred,
            "Sigma_g_pred": Sigma_g_pred,
            "mu_u_pred": mu_u_pred,
            "Sigma_u_pred": Sigma_u_pred,
            "entering": entering,
            "t_now": t_now,
            "costs": costs,
            "info_gains": info_gains,
            "stats": stats,
        }
        return int(r_t), cache

    def _gaussian_logpdf(self, x: float, mean: float, var: float) -> float:
        var = max(float(var), self.eps)
        if not np.isfinite(var):
            return -np.inf
        diff = float(x) - float(mean)
        if not np.isfinite(diff):
            return -np.inf
        diff = float(np.clip(diff, -self.state_clip, self.state_clip))
        return -0.5 * (np.log(2.0 * np.pi * var) + (diff * diff) / var)

    def _gaussian_logpdf_mvn(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        d = int(x.shape[0])
        if d == 0:
            return 0.0
        mean = np.asarray(mean, dtype=float).reshape(-1)
        if mean.shape[0] != d:
            raise ValueError("Mean dimension mismatch in multivariate logpdf.")
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (d, d):
            raise ValueError("Covariance dimension mismatch in multivariate logpdf.")
        cov = 0.5 * (cov + cov.T)
        for i in range(5):
            jitter = (10.0**i) * self.eps
            cov_j = cov + jitter * np.eye(d, dtype=float)
            try:
                L = np.linalg.cholesky(cov_j)
                diff = x - mean
                solve = np.linalg.solve(L, diff)
                log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
                quad = float(np.dot(solve, solve))
                return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)
            except np.linalg.LinAlgError:
                continue
        return -np.inf

    def _update_from_predicted(
        self,
        selected_expert: int,
        observed_value: float,
        phi: np.ndarray,
        w_pred: np.ndarray,
        mu_g_pred: np.ndarray,
        Sigma_g_pred: np.ndarray,
        mu_u_pred: Dict[int, np.ndarray],
        Sigma_u_pred: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        k = int(selected_expert)
        if k not in mu_u_pred or k not in Sigma_u_pred:
            self._birth_expert(k)
            mu_u_pred[k] = self.mu_u[k]
            Sigma_u_pred[k] = self.Sigma_u[k]

        mu_g_post = mu_g_pred.copy()
        Sigma_g_post = Sigma_g_pred.copy()
        mu_u_post: Dict[int, np.ndarray] = {kk: vv.copy() for kk, vv in mu_u_pred.items()}
        Sigma_u_post: Dict[int, np.ndarray] = {kk: vv.copy() for kk, vv in Sigma_u_pred.items()}

        log_like = np.zeros(self.M, dtype=float)
        Bk = self._ensure_B(k)

        for m in range(self.M):
            mu_g = mu_g_pred[m]
            Sigma_g = Sigma_g_pred[m]
            mu_u = mu_u_pred[k][m]
            Sigma_u = Sigma_u_pred[k][m]

            joint_mu = np.concatenate([mu_g, mu_u])
            top = np.concatenate([Sigma_g, np.zeros((self.d_g, self.d_phi))], axis=1)
            bottom = np.concatenate(
                [np.zeros((self.d_phi, self.d_g)), Sigma_u], axis=1
            )
            joint_Sigma = np.concatenate([top, bottom], axis=0)

            H_g = (Bk.T @ phi).reshape(1, self.d_g)
            H_u = phi.reshape(1, self.d_phi)
            H = np.concatenate([H_g, H_u], axis=1)

            pred_mean = float(np.squeeze(H @ joint_mu))
            pred_var = float(np.squeeze(H @ joint_Sigma @ H.T)) + self._get_R(m, k)
            pred_var = max(pred_var, self.eps)

            K = (joint_Sigma @ H.T) / pred_var
            innovation = observed_value - pred_mean
            joint_mu_post = joint_mu + (K.flatten() * innovation)
            dim = joint_Sigma.shape[0]
            I = np.eye(dim, dtype=float)
            KH = K @ H
            tmp = I - KH
            R_val = self._get_R(m, k)
            joint_Sigma_post = tmp @ joint_Sigma @ tmp.T + (K * R_val) @ K.T

            mu_g_post[m] = joint_mu_post[: self.d_g]
            Sigma_g_post[m] = joint_Sigma_post[: self.d_g, : self.d_g]
            Sigma_g_post[m] = 0.5 * (
                Sigma_g_post[m] + Sigma_g_post[m].T
            ) + self.eps * np.eye(self.d_g, dtype=float)

            mu_u_post[k][m] = joint_mu_post[self.d_g :]
            Sigma_u_post[k][m] = joint_Sigma_post[self.d_g :, self.d_g :]
            Sigma_u_post[k][m] = 0.5 * (
                Sigma_u_post[k][m] + Sigma_u_post[k][m].T
            ) + self.eps * np.eye(self.d_phi, dtype=float)

            log_like[m] = self._gaussian_logpdf(observed_value, pred_mean, pred_var)

        log_post = log_like + np.log(np.maximum(w_pred, self.eps))
        log_post -= np.max(log_post)
        w_post = np.exp(log_post)
        w_post_sum = float(np.sum(w_post))
        if w_post_sum <= 0:
            w_post = np.ones(self.M, dtype=float) / float(self.M)
        else:
            w_post /= w_post_sum

        return w_post, mu_g_post, Sigma_g_post, mu_u_post, Sigma_u_post

    def _update_from_predicted_full(
        self,
        observed_residuals: np.ndarray,
        available_experts: Sequence[int],
        phi: np.ndarray,
        w_pred: np.ndarray,
        mu_g_pred: np.ndarray,
        Sigma_g_pred: np.ndarray,
        mu_u_pred: Dict[int, np.ndarray],
        Sigma_u_pred: Dict[int, np.ndarray],
        return_log_like: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        available = sorted(int(k) for k in available_experts)
        if len(available) == 0:
            log_like = np.zeros(self.M, dtype=float)
            if return_log_like:
                return (
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                    log_like,
                )
            return w_pred, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred

        obs_arr = np.asarray(observed_residuals, dtype=float)
        if obs_arr.ndim != 1:
            raise ValueError("losses_full must be a 1D array for full feedback.")

        mu_u_prior: Dict[int, np.ndarray] = {kk: vv.copy() for kk, vv in mu_u_pred.items()}
        Sigma_u_prior: Dict[int, np.ndarray] = {
            kk: vv.copy() for kk, vv in Sigma_u_pred.items()
        }
        for k in available:
            if k not in mu_u_prior or k not in Sigma_u_prior:
                self._birth_expert(k)
                mu_u_prior[k] = self.u_mean0.copy()
                Sigma_u_prior[k] = self.u_cov0.copy()

        mu_g_post = mu_g_pred.copy()
        Sigma_g_post = Sigma_g_pred.copy()
        mu_u_post: Dict[int, np.ndarray] = {kk: vv.copy() for kk, vv in mu_u_prior.items()}
        Sigma_u_post: Dict[int, np.ndarray] = {
            kk: vv.copy() for kk, vv in Sigma_u_prior.items()
        }

        for k in available:
            if k < 0 or k >= obs_arr.shape[0]:
                raise ValueError("losses_full must include all expert residuals.")
        y = np.array([float(obs_arr[k]) for k in available], dtype=float)
        n_avail = len(available)
        dim = int(self.d_g + n_avail * self.d_phi)

        H = np.zeros((n_avail, dim), dtype=float)
        for idx, k in enumerate(available):
            if self.d_g > 0:
                Bk = self._ensure_B(k)
                H[idx, : self.d_g] = (Bk.T @ phi).reshape(-1)
            if self.d_phi > 0:
                start = self.d_g + idx * self.d_phi
                H[idx, start : start + self.d_phi] = phi.reshape(-1)

        log_like = np.zeros(self.M, dtype=float)

        for m in range(self.M):
            if dim > 0:
                parts = []
                if self.d_g > 0:
                    parts.append(mu_g_pred[m])
                if self.d_phi > 0:
                    parts.extend(mu_u_prior[k][m] for k in available)
                joint_mu = np.concatenate(parts) if parts else np.zeros(0, dtype=float)

                joint_Sigma = np.zeros((dim, dim), dtype=float)
                offset = 0
                if self.d_g > 0:
                    joint_Sigma[: self.d_g, : self.d_g] = Sigma_g_pred[m]
                    offset = self.d_g
                if self.d_phi > 0:
                    for k in available:
                        joint_Sigma[
                            offset : offset + self.d_phi, offset : offset + self.d_phi
                        ] = Sigma_u_prior[k][m]
                        offset += self.d_phi
            else:
                joint_mu = np.zeros(0, dtype=float)
                joint_Sigma = np.zeros((0, 0), dtype=float)

            R_vec = np.array([self._get_R(m, k) for k in available], dtype=float)
            pred_mean = H @ joint_mu
            S = H @ joint_Sigma @ H.T + np.diag(R_vec)
            S = 0.5 * (S + S.T)
            log_like[m] = self._gaussian_logpdf_mvn(y, pred_mean, S)

            if dim == 0:
                continue

            PHt = joint_Sigma @ H.T
            try:
                L = np.linalg.cholesky(S + self.eps * np.eye(n_avail, dtype=float))
                tmp = np.linalg.solve(L, PHt.T)
                K = np.linalg.solve(L.T, tmp).T
            except np.linalg.LinAlgError:
                K = np.linalg.solve(S + self.eps * np.eye(n_avail, dtype=float), PHt.T).T

            innovation = y - pred_mean
            joint_mu_post = joint_mu + K @ innovation
            I = np.eye(dim, dtype=float)
            KH = K @ H
            tmp = I - KH
            joint_Sigma_post = tmp @ joint_Sigma @ tmp.T + (K * R_vec) @ K.T

            offset = 0
            if self.d_g > 0:
                mu_g_post[m] = joint_mu_post[: self.d_g]
                Sigma_g_post[m] = joint_Sigma_post[: self.d_g, : self.d_g]
                Sigma_g_post[m] = 0.5 * (
                    Sigma_g_post[m] + Sigma_g_post[m].T
                ) + self.eps * np.eye(self.d_g, dtype=float)
                offset = self.d_g
            if self.d_phi > 0:
                for k in available:
                    mu_u_post[k][m] = joint_mu_post[offset : offset + self.d_phi]
                    Sigma_u_post[k][m] = joint_Sigma_post[
                        offset : offset + self.d_phi, offset : offset + self.d_phi
                    ]
                    Sigma_u_post[k][m] = 0.5 * (
                        Sigma_u_post[k][m] + Sigma_u_post[k][m].T
                    ) + self.eps * np.eye(self.d_phi, dtype=float)
                    offset += self.d_phi

        log_post = log_like + np.log(np.maximum(w_pred, self.eps))
        log_post -= np.max(log_post)
        w_post = np.exp(log_post)
        w_post_sum = float(np.sum(w_post))
        if w_post_sum <= 0:
            w_post = np.ones(self.M, dtype=float) / float(self.M)
        else:
            w_post /= w_post_sum

        if return_log_like:
            return w_post, mu_g_post, Sigma_g_post, mu_u_post, Sigma_u_post, log_like
        return w_post, mu_g_post, Sigma_g_post, mu_u_post, Sigma_u_post

    def update_beliefs(
        self,
        r_t: int,
        loss_obs: float,
        losses_full: Optional[np.ndarray],
        available_experts: Sequence[int],
        cache: dict,
    ) -> None:
        phi = np.asarray(cache["phi"], dtype=float)
        w_pred = np.asarray(cache["w_pred"], dtype=float)
        mu_g_pred = np.asarray(cache["mu_g_pred"], dtype=float)
        Sigma_g_pred = np.asarray(cache["Sigma_g_pred"], dtype=float)
        mu_u_pred = dict(cache["mu_u_pred"])
        Sigma_u_pred = dict(cache["Sigma_u_pred"])

        state_prior = None
        if self._online_em_cfg.get("enabled", False) and self._online_em_cfg.get(
            "use_state_priors", True
        ):
            state_prior = {
                "w": w_pred.copy(),
                "mu_g": mu_g_pred.copy(),
                "Sigma_g": Sigma_g_pred.copy(),
                "mu_u": {k: v.copy() for k, v in mu_u_pred.items()},
                "Sigma_u": {k: v.copy() for k, v in Sigma_u_pred.items()},
                "registry": list(self.registry),
                "last_selected_time": dict(self.last_selected_time),
                "current_step": int(self.current_step),
            }

        if self.feedback_mode == "full":
            if losses_full is None:
                raise ValueError("losses_full must be provided in full feedback mode.")
            w_post, mu_g_post, Sigma_g_post, mu_u_post, Sigma_u_post = (
                self._update_from_predicted_full(
                    losses_full,
                    available_experts,
                    phi,
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                )
            )
        else:
            w_post, mu_g_post, Sigma_g_post, mu_u_post, Sigma_u_post = (
                self._update_from_predicted(
                    r_t,
                    loss_obs,
                    phi,
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                )
            )

        self.w = w_post
        self.mu_g = mu_g_post
        self.Sigma_g = Sigma_g_post
        self.mu_u = mu_u_post
        self.Sigma_u = Sigma_u_post

        self.current_step = int(cache.get("t_now", self.current_step + 1))
        if self.feedback_mode == "full":
            for k in available_experts:
                self.last_selected_time[int(k)] = self.current_step
        else:
            self.last_selected_time[int(r_t)] = self.current_step

        x_t = cache.get("x_t", None)
        residuals_full = losses_full if self.feedback_mode == "full" else None
        self._record_em_observation(
            x_t=x_t,
            available_experts=available_experts,
            action=int(r_t),
            residual=float(loss_obs),
            residuals_full=residuals_full,
            state_override=state_prior,
        )
        self._maybe_online_em_update()

    # Legacy API ----------------------------------------------------
    def predict_step(self, context_x: np.ndarray) -> None:
        w_pred, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred = self._interaction_and_time_update(
            context_x,
            self.w,
            self.mu_g,
            self.Sigma_g,
            self.mu_u,
            self.Sigma_u,
        )
        self.w = w_pred
        self.mu_g = mu_g_pred
        self.Sigma_g = Sigma_g_pred
        self.mu_u = mu_u_pred
        self.Sigma_u = Sigma_u_pred
        self.current_step += 1

    def select_action(
        self,
        context_x: np.ndarray,
        active_experts: Optional[Sequence[int]] = None,
    ) -> int:
        if active_experts is None:
            active_experts = list(self.registry)
        phi = self._compute_phi(context_x)
        costs, info_gains, stats = self._compute_predictive_stats(
            phi,
            self.w,
            self.mu_g,
            self.Sigma_g,
            self.mu_u,
            self.Sigma_u,
            active_experts,
        )
        if not costs:
            return -1
        return self._select_expert_from_stats(
            self.w,
            self.mu_g,
            self.Sigma_g,
            costs,
            info_gains,
            stats,
            active_experts,
            self.current_step + 1,
        )

    def update_step(
        self,
        selected_expert_id: int,
        observed_residual: float,
        context_x: Optional[np.ndarray] = None,
    ) -> None:
        if context_x is None:
            raise ValueError("context_x required for measurement update.")
        phi = self._compute_phi(context_x)
        w_post, mu_g_post, Sigma_g_post, mu_u_post, Sigma_u_post = self._update_from_predicted(
            selected_expert_id,
            observed_residual,
            phi,
            self.w,
            self.mu_g,
            self.Sigma_g,
            self.mu_u,
            self.Sigma_u,
        )
        self.w = w_post
        self.mu_g = mu_g_post
        self.Sigma_g = Sigma_g_post
        self.mu_u = mu_u_post
        self.Sigma_u = Sigma_u_post
        self.last_selected_time[int(selected_expert_id)] = self.current_step

    def manage_registry(self, current_experts: Sequence[int], t_now: Optional[int] = None) -> List[int]:
        if t_now is None:
            t_now = self.current_step + 1
        current_set = set(int(x) for x in current_experts)
        entering = sorted(current_set.difference(self.registry))

        for k in entering:
            self._birth_expert(k)

        for k in list(self.registry):
            last_time = int(self.last_selected_time.get(k, 0))
            if k not in current_set and (t_now - last_time) > self.Delta_max:
                self.registry.remove(k)
                self.mu_u.pop(k, None)
                self.Sigma_u.pop(k, None)
                self.last_selected_time.pop(k, None)
        return entering

    def _birth_expert(self, k: int) -> None:
        if k in self.registry:
            return
        self.registry.append(k)
        self._ensure_B(k)

        self.mu_u[k] = self.u_mean0.copy()
        self.Sigma_u[k] = self.u_cov0.copy()
        if isinstance(self.beta, dict) and k not in self.beta:
            self.beta[k] = 0.0
        self.last_selected_time[k] = 0

    # ---------------------- EM training ----------------------
    def _transition_state_dict(self) -> Optional[dict]:
        if self.transition_model is None:
            return None
        if torch is None:
            raise RuntimeError("Transition model requires PyTorch.")
        return {
            k: v.detach().cpu().clone()
            for k, v in self.transition_model.state_dict().items()
        }

    def _snapshot_params(self) -> dict:
        params = {
            "A_g": self.A_g.copy(),
            "Q_g": self.Q_g.copy(),
            "A_u": self.A_u.copy(),
            "Q_u": self.Q_u.copy(),
            "B_dict": {k: v.copy() for k, v in self.B_dict.items()},
            "R": float(self.R) if np.ndim(self.R) == 0 else self.R.copy(),
            "W_q": None if self.W_q is None else self.W_q.copy(),
            "W_k": None if self.W_k is None else self.W_k.copy(),
            "b_attn": None if self.b_attn is None else self.b_attn.copy(),
            "W_lin": None if self.W_lin is None else self.W_lin.copy(),
            "b_lin": None if self.b_lin is None else self.b_lin.copy(),
            "transition_mode": self.transition_mode,
            "context_dim": self.context_dim,
        }
        params["transition_model_state"] = self._transition_state_dict()
        params["transition_input_dim"] = self.transition_input_dim
        return params

    def _restore_params(self, params: dict) -> None:
        self.A_g = params["A_g"].copy()
        self.Q_g = params["Q_g"].copy()
        self.A_u = params["A_u"].copy()
        self.Q_u = params["Q_u"].copy()
        self.B_dict = {k: v.copy() for k, v in params["B_dict"].items()}
        self.R = float(params["R"]) if np.ndim(params["R"]) == 0 else params["R"].copy()
        self.transition_mode = params.get("transition_mode", self.transition_mode)
        self.W_q = None if params["W_q"] is None else params["W_q"].copy()
        self.W_k = None if params["W_k"] is None else params["W_k"].copy()
        self.b_attn = (
            None if params.get("b_attn", None) is None else params["b_attn"].copy()
        )
        self.W_lin = None if params.get("W_lin", None) is None else params["W_lin"].copy()
        self.b_lin = None if params.get("b_lin", None) is None else params["b_lin"].copy()
        self.context_dim = params.get("context_dim", self.context_dim)
        transition_state = params.get("transition_model_state", None)
        if transition_state is not None:
            input_dim = params.get("transition_input_dim", None)
            if input_dim is None:
                raise ValueError("Missing transition_input_dim for transition model.")
            self._ensure_transition_model(int(input_dim))
            if self.transition_model is None:
                raise RuntimeError("Transition model is not initialized.")
            self.transition_model.load_state_dict(transition_state)
            self.transition_input_dim = int(input_dim)

    def _snapshot_state(self) -> dict:
        return {
            "w": self.w.copy(),
            "mu_g": self.mu_g.copy(),
            "Sigma_g": self.Sigma_g.copy(),
            "mu_u": {k: v.copy() for k, v in self.mu_u.items()},
            "Sigma_u": {k: v.copy() for k, v in self.Sigma_u.items()},
            "registry": list(self.registry),
            "last_selected_time": dict(self.last_selected_time),
            "current_step": self.current_step,
        }

    def _restore_state(self, state: dict) -> None:
        self.w = state["w"].copy()
        self.mu_g = state["mu_g"].copy()
        self.Sigma_g = state["Sigma_g"].copy()
        self.mu_u = {k: v.copy() for k, v in state["mu_u"].items()}
        self.Sigma_u = {k: v.copy() for k, v in state["Sigma_u"].items()}
        self.registry = list(state["registry"])
        self.last_selected_time = dict(state["last_selected_time"])
        self.current_step = int(state["current_step"])

    def _logsumexp(self, values: np.ndarray) -> float:
        vmax = float(np.max(values))
        return vmax + float(np.log(np.sum(np.exp(values - vmax))))

    def _mode_residual_stats(
        self,
        phi: np.ndarray,
        mu_g_pred: np.ndarray,
        Sigma_g_pred: np.ndarray,
        mu_u_pred: Dict[int, np.ndarray],
        Sigma_u_pred: Dict[int, np.ndarray],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean_modes = np.zeros(self.M, dtype=float)
        var_modes = np.zeros(self.M, dtype=float)
        if self.d_g == 0:
            for m in range(self.M):
                mu_u_m = mu_u_pred[k][m]
                Sigma_u_m = Sigma_u_pred[k][m]
                mean_modes[m] = float(np.squeeze(phi @ mu_u_m))
                noise = float(np.squeeze(phi @ (Sigma_u_m @ phi))) + self._get_R(m, k)
                var_modes[m] = max(noise, self.eps)
            return mean_modes, var_modes

        Bk = self._ensure_B(k)
        for m in range(self.M):
            mu_g_m = mu_g_pred[m]
            Sigma_g_m = Sigma_g_pred[m]
            mu_u_m = mu_u_pred[k][m]
            Sigma_u_m = Sigma_u_pred[k][m]
            mean_modes[m] = float(np.squeeze(phi @ (Bk @ mu_g_m + mu_u_m)))
            signal = float(np.squeeze(phi @ (Bk @ (Sigma_g_m @ (Bk.T @ phi)))))
            noise = float(np.squeeze(phi @ (Sigma_u_m @ phi))) + self._get_R(m, k)
            var_modes[m] = max(signal + noise, self.eps)
        return mean_modes, var_modes

    def _filter_block_nll(
        self,
        contexts: Sequence[np.ndarray],
        available_sets: Sequence[Sequence[int]],
        actions: Sequence[int],
        residuals: Sequence[float],
        residuals_full: Optional[Sequence[np.ndarray]] = None,
        accumulate: bool = True,
    ) -> float:
        nll = 0.0
        T = len(contexts)
        full_feedback = self.feedback_mode == "full" and residuals_full is not None
        for t in range(T):
            x_t = contexts[t]
            available = available_sets[t]
            phi = self._compute_phi(x_t)

            t_now = self.current_step + 1
            entering = self.manage_registry(available, t_now=t_now)
            w_pred, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred = self._interaction_and_time_update(
                x_t,
                self.w,
                self.mu_g,
                self.Sigma_g,
                self.mu_u,
                self.Sigma_u,
            )
            for k in entering:
                mu_u_pred[int(k)] = self.u_mean0.copy()
                Sigma_u_pred[int(k)] = self.u_cov0.copy()

            if full_feedback:
                residuals_t = np.asarray(residuals_full[t], dtype=float)
                (
                    w_post,
                    mu_g_post,
                    Sigma_g_post,
                    mu_u_post,
                    Sigma_u_post,
                    log_like,
                ) = self._update_from_predicted_full(
                    residuals_t,
                    available,
                    phi,
                    w_pred,
                    mu_g_pred,
                    Sigma_g_pred,
                    mu_u_pred,
                    Sigma_u_pred,
                    return_log_like=True,
                )
                if accumulate:
                    nll += -self._logsumexp(
                        log_like + np.log(np.maximum(w_pred, self.eps))
                    )
            else:
                action = int(actions[t])
                residual = float(residuals[t])
                mean_modes, var_modes = self._mode_residual_stats(
                    phi, mu_g_pred, Sigma_g_pred, mu_u_pred, Sigma_u_pred, action
                )
                loglikes = np.log(np.maximum(w_pred, self.eps))
                for m in range(self.M):
                    loglikes[m] += self._gaussian_logpdf(
                        residual, mean_modes[m], var_modes[m]
                    )
                if accumulate:
                    nll += -self._logsumexp(loglikes)

                w_post, mu_g_post, Sigma_g_post, mu_u_post, Sigma_u_post = (
                    self._update_from_predicted(
                        action,
                        residual,
                        phi,
                        w_pred,
                        mu_g_pred,
                        Sigma_g_pred,
                        mu_u_pred,
                        Sigma_u_pred,
                    )
                )
            self.w = w_post
            self.mu_g = mu_g_post
            self.Sigma_g = Sigma_g_post
            self.mu_u = mu_u_post
            self.Sigma_u = Sigma_u_post
            self.current_step = t_now
            if full_feedback:
                for k in available:
                    self.last_selected_time[int(k)] = t_now
            else:
                self.last_selected_time[action] = t_now

        return nll

    def _evaluate_nll(
        self,
        contexts: Sequence[np.ndarray],
        available_sets: Sequence[Sequence[int]],
        actions: Sequence[int],
        residuals: Sequence[float],
        residuals_full: Optional[Sequence[np.ndarray]] = None,
    ) -> float:
        state = self._snapshot_state()
        em_records = list(self._em_records)
        em_start = self._em_records_start
        self.reset_beliefs()

        nll = self._filter_block_nll(
            contexts,
            available_sets,
            actions,
            residuals,
            residuals_full=residuals_full,
            accumulate=True,
        )

        self._restore_state(state)
        self._em_records = em_records
        self._em_records_start = em_start
        return nll / max(len(contexts), 1)

    def _evaluate_nll_with_prefix(
        self,
        prefix_contexts: Sequence[np.ndarray],
        prefix_available_sets: Sequence[Sequence[int]],
        prefix_actions: Sequence[int],
        prefix_residuals: Sequence[float],
        prefix_residuals_full: Optional[Sequence[np.ndarray]],
        contexts: Sequence[np.ndarray],
        available_sets: Sequence[Sequence[int]],
        actions: Sequence[int],
        residuals: Sequence[float],
        residuals_full: Optional[Sequence[np.ndarray]] = None,
    ) -> float:
        state = self._snapshot_state()
        em_records = list(self._em_records)
        em_start = self._em_records_start
        self.reset_beliefs()

        if prefix_contexts:
            self._filter_block_nll(
                prefix_contexts,
                prefix_available_sets,
                prefix_actions,
                prefix_residuals,
                residuals_full=prefix_residuals_full,
                accumulate=False,
            )
        nll = self._filter_block_nll(
            contexts,
            available_sets,
            actions,
            residuals,
            residuals_full=residuals_full,
            accumulate=True,
        )

        self._restore_state(state)
        self._em_records = em_records
        self._em_records_start = em_start
        return nll / max(len(contexts), 1)

    def _evaluate_nll_rolling(
        self,
        contexts: Sequence[np.ndarray],
        available_sets: Sequence[Sequence[int]],
        actions: Sequence[int],
        residuals: Sequence[float],
        residuals_full: Optional[Sequence[np.ndarray]],
        train_end: int,
        val_window_len: int,
        n_splits: int,
        stride: Optional[int] = None,
    ) -> Tuple[float, float, int, list[tuple[int, int]]]:
        T = len(contexts)
        train_end = max(1, min(int(train_end), T))
        val_window_len = max(1, int(val_window_len))
        val_tail_len = max(0, T - train_end)
        if val_tail_len <= 0:
            return float("inf"), 0.0, 0, []
        if val_window_len > val_tail_len:
            val_window_len = val_tail_len
        max_offset = max(0, val_tail_len - val_window_len)
        n_splits = max(1, int(n_splits))

        if stride is not None:
            stride = max(1, int(stride))
            offsets = list(range(0, max_offset + 1, stride))
        else:
            if max_offset == 0 or n_splits <= 1:
                offsets = [0]
            else:
                offsets = [
                    int(round(x))
                    for x in np.linspace(0, max_offset, num=n_splits)
                ]
        offsets = sorted(set(int(o) for o in offsets))
        if not offsets:
            offsets = [0]
        if len(offsets) > n_splits:
            idxs = np.linspace(0, len(offsets) - 1, num=n_splits)
            offsets = [offsets[int(round(i))] for i in idxs]
            offsets = sorted(set(offsets))

        scores = []
        split_sizes: list[tuple[int, int]] = []
        for offset in offsets:
            offset = int(offset)
            prefix_len = train_end + offset
            val_start = prefix_len
            val_end = min(T, val_start + val_window_len)
            if val_start >= val_end:
                continue
            split_sizes.append((int(prefix_len), int(val_end - val_start)))
            prefix_ctx = contexts[:prefix_len]
            prefix_avail = available_sets[:prefix_len]
            prefix_actions = actions[:prefix_len]
            prefix_residuals = residuals[:prefix_len]
            prefix_residuals_full = (
                None
                if residuals_full is None
                else residuals_full[:prefix_len]
            )
            val_ctx = contexts[val_start:val_end]
            val_avail = available_sets[val_start:val_end]
            val_actions = actions[val_start:val_end]
            val_residuals = residuals[val_start:val_end]
            val_residuals_full = (
                None
                if residuals_full is None
                else residuals_full[val_start:val_end]
            )
            score = self._evaluate_nll_with_prefix(
                prefix_ctx,
                prefix_avail,
                prefix_actions,
                prefix_residuals,
                prefix_residuals_full,
                val_ctx,
                val_avail,
                val_actions,
                val_residuals,
                residuals_full=val_residuals_full,
            )
            scores.append(float(score))
        if not scores:
            return float("inf"), 0.0, 0, []
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        return mean, std, len(scores), split_sizes

    def _train_transition_model(
        self,
        contexts: Sequence[np.ndarray],
        xi: np.ndarray,
        lr: float,
        steps: int,
        weight_decay: float,
        seed: Optional[int],
    ) -> None:
        if self.transition_model is None and self.transition_hidden_dims is None:
            return
        if torch is None or nn is None:
            raise RuntimeError("Transition model requires PyTorch.")
        if steps <= 0:
            return
        T_train = len(contexts)
        if T_train <= 1:
            return
        x_arr = np.stack(
            [np.asarray(contexts[t], dtype=float).reshape(-1) for t in range(1, T_train)]
        )
        self._ensure_transition_model(x_arr.shape[1])
        if self.transition_model is None:
            raise RuntimeError("Transition model is not initialized.")
        if xi.shape[0] != x_arr.shape[0]:
            raise ValueError("Transition count shape does not match contexts.")
        device = self.transition_device or torch.device(self._transition_device_str or "cpu")
        self.transition_model.to(device)
        if seed is not None:
            torch.manual_seed(int(seed))
        self.transition_model.train()
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=device)
        xi_tensor = torch.as_tensor(xi, dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(
            self.transition_model.parameters(),
            lr=float(lr)
        )
        best_state = None
        best_loss = float("inf")
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad()
            logits = self.transition_model(x_tensor)
            if logits.shape[-2:] != (self.M, self.M):
                raise ValueError("transition_model must output shape (batch, M, M).")
            log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            denom = max(float(xi_tensor.sum().item()), self.eps)
            loss = -(xi_tensor * log_probs).sum() / denom
            loss.backward()
            optimizer.step()
            loss_val = float(loss.detach().cpu())
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.transition_model.state_dict().items()
                }
        self.transition_model.eval()
        with torch.no_grad():
            logits = self.transition_model(x_tensor)
            log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            denom = max(float(xi_tensor.sum().item()), self.eps)
            final_loss = float((-(xi_tensor * log_probs).sum() / denom).detach().cpu())
        if final_loss < best_loss:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.transition_model.state_dict().items()
            }
        if best_state is not None:
            self.transition_model.load_state_dict(best_state)

    def _train_transition_linear_torch(
        self,
        contexts: Sequence[np.ndarray],
        xi: np.ndarray,
        lr: float,
        steps: int,
        weight_decay: float,
        seed: Optional[int],
    ) -> None:
        if torch is None:
            raise RuntimeError("Transition model requires PyTorch.")
        if steps <= 0:
            return
        T_train = len(contexts)
        if T_train <= 1:
            return
        x_arr = np.stack(
            [np.asarray(contexts[t], dtype=float).reshape(-1) for t in range(1, T_train)]
        )
        if self.W_lin is None or self.b_lin is None:
            self._init_transition_params(x_arr.shape[1])
        if xi.shape[0] != x_arr.shape[0]:
            raise ValueError("Transition count shape does not match contexts.")
        device = self.transition_device or torch.device(self._transition_device_str or "cpu")
        if seed is not None:
            torch.manual_seed(int(seed))
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=device)
        xi_tensor = torch.as_tensor(xi, dtype=torch.float32, device=device)
        W_lin = torch.as_tensor(self.W_lin, dtype=torch.float32, device=device).requires_grad_(True)
        b_lin = torch.as_tensor(self.b_lin, dtype=torch.float32, device=device).requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [W_lin, b_lin],
            lr=float(lr)
        )
        losses: list[float] = []
        step_dws: list[float] = []
        grad_norms: list[float] = []
        denom = max(float(xi_tensor.sum().detach().cpu()), float(self.eps))
        best_loss = float("inf")
        best_W = None
        best_b = None
        for _ in range(max(1, int(steps))):
            W_prev = W_lin.detach().clone()

            optimizer.zero_grad(set_to_none=True)

            scores = torch.einsum("mjd,td->tmj", W_lin, x_tensor) + b_lin
            log_probs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)

            loss = -(xi_tensor * log_probs).sum() / denom
            loss.backward()

            # Diagnostics (optional but useful)
            g = W_lin.grad
            grad_norms.append(float(g.detach().norm().cpu()) if g is not None else 0.0)

            optimizer.step()

            step_dws.append(float((W_lin.detach() - W_prev).norm().cpu()))
            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                best_W = W_lin.detach().cpu().clone()
                best_b = b_lin.detach().cpu().clone()

        with torch.no_grad():
            scores = torch.einsum("mjd,td->tmj", W_lin, x_tensor) + b_lin
            log_probs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
            final_loss = float((-(xi_tensor * log_probs).sum() / denom).detach().cpu())
        if final_loss < best_loss:
            best_W = W_lin.detach().cpu().clone()
            best_b = b_lin.detach().cpu().clone()

        # Save trained params
        if best_W is None or best_b is None:
            self.W_lin = W_lin.detach().cpu().numpy()
            self.b_lin = b_lin.detach().cpu().numpy()
        else:
            self.W_lin = best_W.numpy()
            self.b_lin = best_b.numpy()


    def _train_transition_attention_torch(
        self,
        contexts: Sequence[np.ndarray],
        xi: np.ndarray,
        lr: float,
        steps: int,
        weight_decay: float,
        seed: Optional[int],
    ) -> None:
        if torch is None:
            raise RuntimeError("Transition model requires PyTorch.")
        if steps <= 0:
            return
        T_train = len(contexts)
        if T_train <= 1:
            return
        x_arr = np.stack(
            [np.asarray(contexts[t], dtype=float).reshape(-1) for t in range(1, T_train)]
        )
        if self.W_q is None or self.W_k is None:
            self._init_transition_params(x_arr.shape[1])
        if xi.shape[0] != x_arr.shape[0]:
            raise ValueError("Transition count shape does not match contexts.")
        device = self.transition_device or torch.device(self._transition_device_str or "cpu")
        if seed is not None:
            torch.manual_seed(int(seed))
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=device)
        xi_tensor = torch.as_tensor(xi, dtype=torch.float32, device=device)
        W_q = torch.as_tensor(self.W_q, dtype=torch.float32, device=device).requires_grad_(True)
        W_k = torch.as_tensor(self.W_k, dtype=torch.float32, device=device).requires_grad_(True)
        b_attn = None
        params = [W_q, W_k]
        if self.b_attn is not None:
            b_attn = torch.as_tensor(self.b_attn, dtype=torch.float32, device=device).requires_grad_(True)
            params.append(b_attn)
        optimizer = torch.optim.AdamW(
            params,
            lr=float(lr)
        )
        scale = (W_q.shape[1] ** 0.5)
        denom = max(float(xi_tensor.sum().detach().cpu()), self.eps)
        best_loss = float("inf")
        best_Wq = None
        best_Wk = None
        best_b = None
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            q = torch.einsum("mad,td->tma", W_q, x_tensor)
            k = torch.einsum("mad,td->tma", W_k, x_tensor)
            scores = torch.einsum("tma,tna->tmn", q, k) / scale
            if b_attn is not None:
                scores = scores + b_attn  # ensure b_attn is (M,M)
            log_probs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
            loss = -(xi_tensor * log_probs).sum() / denom
            loss.backward()
            optimizer.step()
            loss_val = float(loss.detach().cpu())
            if loss_val < best_loss:
                best_loss = loss_val
                best_Wq = W_q.detach().cpu().clone()
                best_Wk = W_k.detach().cpu().clone()
                if b_attn is not None:
                    best_b = b_attn.detach().cpu().clone()
        with torch.no_grad():
            q = torch.einsum("mad,td->tma", W_q, x_tensor)
            k = torch.einsum("mad,td->tma", W_k, x_tensor)
            scores = torch.einsum("tma,tna->tmn", q, k) / scale
            if b_attn is not None:
                scores = scores + b_attn
            log_probs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
            final_loss = float((-(xi_tensor * log_probs).sum() / denom).detach().cpu())
        if final_loss < best_loss:
            best_Wq = W_q.detach().cpu().clone()
            best_Wk = W_k.detach().cpu().clone()
            if b_attn is not None:
                best_b = b_attn.detach().cpu().clone()
        if best_Wq is None or best_Wk is None:
            self.W_q = W_q.detach().cpu().numpy()
            self.W_k = W_k.detach().cpu().numpy()
            if b_attn is not None:
                self.b_attn = b_attn.detach().cpu().numpy()
        else:
            self.W_q = best_Wq.numpy()
            self.W_k = best_Wk.numpy()
            if b_attn is not None and best_b is not None:
                self.b_attn = best_b.numpy()

    def _kalman_sample(
        self,
        A_seq: np.ndarray,
        Q_seq: np.ndarray,
        H_seq: np.ndarray,
        R_seq: np.ndarray,
        y_seq: np.ndarray,
        obs_mask: np.ndarray,
        m0: np.ndarray,
        P0: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        T, d = y_seq.shape[0], m0.shape[0]
        if d == 0:
            return np.zeros((T, 0), dtype=float)
        m_pred = np.zeros((T, d), dtype=float)
        P_pred = np.zeros((T, d, d), dtype=float)
        m_filt = np.zeros((T, d), dtype=float)
        P_filt = np.zeros((T, d, d), dtype=float)

        for t in range(T):
            if t == 0:
                m_pred[t] = m0
                P_pred[t] = P0
            else:
                m_pred[t] = A_seq[t] @ m_filt[t - 1]
                P_pred[t] = A_seq[t] @ P_filt[t - 1] @ A_seq[t].T + Q_seq[t]
            P_pred[t] = 0.5 * (P_pred[t] + P_pred[t].T) + self.eps * np.eye(d)

            if obs_mask[t]:
                H_t = H_seq[t]
                R_t = float(R_seq[t])
                S = float(np.squeeze(H_t @ P_pred[t] @ H_t.T)) + R_t
                S = max(S, self.eps)
                K = (P_pred[t] @ H_t.T) / S
                innovation = float(np.squeeze(y_seq[t] - H_t @ m_pred[t]))
                m_filt[t] = m_pred[t] + (K.flatten() * innovation)
                P_filt[t] = P_pred[t] - K @ H_t @ P_pred[t]
                P_filt[t] = 0.5 * (P_filt[t] + P_filt[t].T) + self.eps * np.eye(d)
            else:
                m_filt[t] = m_pred[t]
                P_filt[t] = P_pred[t]

        x = np.zeros((T, d), dtype=float)
        x[T - 1] = rng.multivariate_normal(m_filt[T - 1], P_filt[T - 1])
        for t in range(T - 2, -1, -1):
            P_pred_next = P_pred[t + 1] + self.eps * np.eye(d, dtype=float)
            PHt = P_filt[t] @ A_seq[t + 1].T
            try:
                L = np.linalg.cholesky(P_pred_next)
                tmp = np.linalg.solve(L, PHt.T)
                J = np.linalg.solve(L.T, tmp).T
            except np.linalg.LinAlgError:
                J = np.linalg.solve(P_pred_next, PHt.T).T
            mean = m_filt[t] + J @ (x[t + 1] - m_pred[t + 1])
            cov = P_filt[t] - J @ P_pred_next @ J.T
            cov = 0.5 * (cov + cov.T) + self.eps * np.eye(d)
            x[t] = rng.multivariate_normal(mean, cov)

        return x

    def _kalman_sample_multi(
        self,
        A_seq: np.ndarray,
        Q_seq: np.ndarray,
        H_seq: Sequence[np.ndarray],
        R_seq: Sequence[np.ndarray],
        y_seq: Sequence[np.ndarray],
        m0: np.ndarray,
        P0: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        T, d = len(y_seq), m0.shape[0]
        if d == 0:
            return np.zeros((T, 0), dtype=float)
        m_pred = np.zeros((T, d), dtype=float)
        P_pred = np.zeros((T, d, d), dtype=float)
        m_filt = np.zeros((T, d), dtype=float)
        P_filt = np.zeros((T, d, d), dtype=float)

        for t in range(T):
            if t == 0:
                m_pred[t] = m0
                P_pred[t] = P0
            else:
                m_pred[t] = A_seq[t] @ m_filt[t - 1]
                P_pred[t] = A_seq[t] @ P_filt[t - 1] @ A_seq[t].T + Q_seq[t]
            P_pred[t] = 0.5 * (P_pred[t] + P_pred[t].T) + self.eps * np.eye(d)

            m_t = m_pred[t]
            P_t = P_pred[t]
            H_t = H_seq[t]
            y_t = y_seq[t]
            R_t = R_seq[t]
            if H_t is not None and len(y_t) > 0:
                for i in range(len(y_t)):
                    H_row = np.asarray(H_t[i], dtype=float).reshape(1, d)
                    R_val = float(R_t[i])
                    S = float(np.squeeze(H_row @ P_t @ H_row.T)) + R_val
                    S = max(S, self.eps)
                    K = (P_t @ H_row.T) / S
                    innovation = float(np.squeeze(y_t[i] - H_row @ m_t))
                    m_t = m_t + (K.flatten() * innovation)
                    P_t = P_t - K @ H_row @ P_t
                    P_t = 0.5 * (P_t + P_t.T) + self.eps * np.eye(d)
            m_filt[t] = m_t
            P_filt[t] = P_t

        x = np.zeros((T, d), dtype=float)
        x[T - 1] = rng.multivariate_normal(m_filt[T - 1], P_filt[T - 1])
        for t in range(T - 2, -1, -1):
            P_pred_next = P_pred[t + 1] + self.eps * np.eye(d, dtype=float)
            PHt = P_filt[t] @ A_seq[t + 1].T
            try:
                L = np.linalg.cholesky(P_pred_next)
                tmp = np.linalg.solve(L, PHt.T)
                J = np.linalg.solve(L.T, tmp).T
            except np.linalg.LinAlgError:
                J = np.linalg.solve(P_pred_next, PHt.T).T
            mean = m_filt[t] + J @ (x[t + 1] - m_pred[t + 1])
            cov = P_filt[t] - J @ P_pred_next @ J.T
            cov = 0.5 * (cov + cov.T) + self.eps * np.eye(d)
            x[t] = rng.multivariate_normal(mean, cov)

        return x

    def _sample_z_sequence(
        self,
        contexts: Sequence[np.ndarray],
        actions: np.ndarray,
        residuals: np.ndarray,
        g_prev: np.ndarray,
        u_prev: Dict[int, np.ndarray],
        w0: Optional[np.ndarray] = None,
        residuals_full: Optional[Sequence[np.ndarray]] = None,
        available_sets: Optional[Sequence[Sequence[int]]] = None,
        g_init_mean: Optional[np.ndarray] = None,
        g_init_cov: Optional[np.ndarray] = None,
        u_init_mean: Optional[Dict[int, np.ndarray]] = None,
        u_init_cov: Optional[Dict[int, np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        T = len(contexts)
        phi_seq = [self._compute_phi(x) for x in contexts]
        log_emission = np.zeros((T, self.M), dtype=float)
        full_feedback = (
            self.feedback_mode == "full"
            and residuals_full is not None
            and available_sets is not None
        )
        if g_init_mean is None:
            g_init_mean = self.g_mean0
        if g_init_cov is None:
            g_init_cov = self.g_cov0
        if u_init_mean is None:
            u_init_mean = {}
        if u_init_cov is None:
            u_init_cov = {}
        expert_ids = sorted(int(k) for k in u_prev.keys())

        for t in range(T):
            phi_t = phi_seq[t]
            if full_feedback:
                for k in available_sets[t]:
                    k = int(k)
                    Bk = self._ensure_B(k)
                    mean_shared = float(np.squeeze(phi_t @ (Bk @ g_prev[t] + u_prev[k][t])))
                    for m in range(self.M):
                        var = self._get_R(m, k)
                        log_emission[t, m] += self._gaussian_logpdf(
                            float(residuals_full[t][k]), mean_shared, var
                        )
            else:
                k = int(actions[t])
                Bk = self._ensure_B(k)
                mean_shared = float(np.squeeze(phi_t @ (Bk @ g_prev[t] + u_prev[k][t])))
                for m in range(self.M):
                    var = self._get_R(m, k)
                    log_emission[t, m] = self._gaussian_logpdf(
                        float(residuals[t]), mean_shared, var
                    )
            for m in range(self.M):
                if self.d_g > 0:
                    if t == 0:
                        log_emission[t, m] += self._gaussian_logpdf_mvn(
                            g_prev[t], g_init_mean[m], g_init_cov[m]
                        )
                    else:
                        mean_g = self.A_g[m] @ g_prev[t - 1]
                        log_emission[t, m] += self._gaussian_logpdf_mvn(
                            g_prev[t], mean_g, self.Q_g[m]
                        )
                if self.d_phi > 0 and expert_ids:
                    for k in expert_ids:
                        if k not in u_prev:
                            continue
                        if t == 0:
                            u0_mean = u_init_mean.get(k, self.u_mean0)[m]
                            u0_cov = u_init_cov.get(k, self.u_cov0)[m]
                            log_emission[t, m] += self._gaussian_logpdf_mvn(
                                u_prev[k][t], u0_mean, u0_cov
                            )
                        else:
                            mean_u = self.A_u[m] @ u_prev[k][t - 1]
                            log_emission[t, m] += self._gaussian_logpdf_mvn(
                                u_prev[k][t], mean_u, self.Q_u[m]
                            )

        Pi_seq = np.zeros((T, self.M, self.M), dtype=float)
        for t in range(T):
            Pi_seq[t] = self._context_transition(contexts[t])

        alpha = np.zeros((T, self.M), dtype=float)
        if w0 is None:
            w_init = np.ones(self.M, dtype=float)
        else:
            w_init = np.asarray(w0, dtype=float)
        if w_init.shape != (self.M,):
            raise ValueError("w0 must have shape (M,).")
        w_init = np.maximum(w_init, self.eps)
        w_init = w_init / np.sum(w_init)
        alpha[0] = np.log(w_init) + log_emission[0]
        alpha[0] -= self._logsumexp(alpha[0])
        for t in range(1, T):
            for j in range(self.M):
                prev = alpha[t - 1] + np.log(np.maximum(Pi_seq[t][..., j], self.eps))
                alpha[t, j] = log_emission[t, j] + self._logsumexp(prev)
            alpha[t] -= self._logsumexp(alpha[t])

        if rng is None:
            rng = self._rng
        z = np.zeros(T, dtype=int)
        probs = np.exp(alpha[T - 1] - np.max(alpha[T - 1]))
        probs = probs / probs.sum()
        z[T - 1] = int(rng.choice(self.M, p=probs))
        for t in range(T - 2, -1, -1):
            trans = np.log(np.maximum(Pi_seq[t + 1][:, z[t + 1]], self.eps))
            logp = alpha[t] + trans
            logp -= self._logsumexp(logp)
            p = np.exp(logp)
            z[t] = int(rng.choice(self.M, p=p))

        return z

    def fit_em(
        self,
        contexts: Sequence[np.ndarray],
        available_sets: Optional[Sequence[Sequence[int]]],
        actions: Sequence[int],
        residuals: Sequence[float],
        residuals_full: Optional[Sequence[np.ndarray]] = None,
        n_em: int = 5,
        n_samples: int = 10,
        burn_in: int = 5,
        val_fraction: float = 0.2,
        val_len: Optional[int] = None,
        val_strategy: str = "tail",
        val_roll_splits: Optional[int] = None,
        val_roll_len: Optional[int] = None,
        val_roll_stride: Optional[int] = None,
        priors: Optional[dict] = None,
        theta_lr: float = 1e-2,
        theta_steps: int = 1,
        seed: Optional[int] = 0,
        print_val_loss: bool = True,
        epsilon_N: float = 0.0,
        init_state: Optional[dict] = None,
        use_validation: bool = True,
        set_em_tk: bool = True,
        transition_log_stage: Optional[str] = None,
    ) -> dict:
        rng = np.random.default_rng(None if seed is None else int(seed))
        contexts = list(contexts)
        actions = np.asarray(actions, dtype=int)
        residuals = np.asarray(residuals, dtype=float)
        T = len(contexts)
        if set_em_tk:
            self.em_tk = T
        epsilon_N = float(epsilon_N)
        full_feedback = self.feedback_mode == "full"
        residuals_full_arr = None
        if full_feedback:
            if residuals_full is None:
                raise ValueError("residuals_full must be provided for full feedback EM.")
            residuals_full_arr = [np.asarray(r, dtype=float) for r in residuals_full]
            if len(residuals_full_arr) != T:
                raise ValueError("residuals_full length must match contexts.")
        if available_sets is None:
            if self.N is None:
                raise ValueError("available_sets required when num_experts is unknown.")
            available_sets = [list(range(self.N)) for _ in range(T)]
        available_sets = list(available_sets)

        if T == 0:
            raise ValueError("Empty training data.")

        val_strategy = str(val_strategy).lower()
        if val_strategy not in ("tail", "rolling"):
            raise ValueError("val_strategy must be 'tail' or 'rolling'.")
        split_idx = T
        do_validation = bool(use_validation and T > 1)
        val_window_len = None
        if do_validation and val_strategy == "rolling":
            if val_roll_len is not None:
                val_window_len = int(val_roll_len)
            elif val_len is not None:
                val_window_len = int(val_len)
            elif val_fraction > 0:
                val_window_len = max(1, int(val_fraction * T))
            else:
                do_validation = False
            if do_validation:
                if val_fraction > 0:
                    split_idx = max(1, int((1.0 - val_fraction) * T))
                else:
                    split_idx = max(1, T - int(val_window_len))
                if split_idx >= T:
                    do_validation = False
                    split_idx = T
            if do_validation:
                val_tail_len = T - split_idx
                if val_tail_len <= 0:
                    do_validation = False
                else:
                    if val_window_len is None:
                        val_window_len = val_tail_len
                    if val_window_len <= 0:
                        do_validation = False
                    elif val_window_len > val_tail_len:
                        val_window_len = val_tail_len
        elif do_validation:
            if val_len is not None:
                val_len_int = int(val_len)
                if val_len_int <= 0:
                    split_idx = T
                else:
                    split_idx = max(1, T - val_len_int)
            else:
                split_idx = max(1, int((1.0 - val_fraction) * T))
            if split_idx >= T:
                do_validation = False
                split_idx = T
        else:
            split_idx = T

        train_ctx = contexts[:split_idx]
        train_avail = available_sets[:split_idx]
        train_actions = actions[:split_idx]
        train_residuals = residuals[:split_idx]
        val_ctx = contexts[split_idx:] if do_validation else []
        val_avail = available_sets[split_idx:] if do_validation else []
        val_actions = actions[split_idx:] if do_validation else []
        val_residuals = residuals[split_idx:] if do_validation else []
        if full_feedback:
            train_residuals_full = residuals_full_arr[:split_idx]
            val_residuals_full = residuals_full_arr[split_idx:] if do_validation else None
        else:
            train_residuals_full = None
            val_residuals_full = None

        if priors is None:
            priors = {}
        lambda_A_g = float(priors.get("lambda_A_g", 1e-6))
        lambda_A_u = float(priors.get("lambda_A_u", 1e-6))
        M_A_g = np.asarray(priors.get("M_A_g", np.zeros((self.d_g, self.d_g))), dtype=float)
        M_A_u = np.asarray(priors.get("M_A_u", np.zeros((self.d_phi, self.d_phi))), dtype=float)
        Psi_g = np.asarray(priors.get("Psi_g", np.zeros((self.d_g, self.d_g))), dtype=float)
        Psi_u = np.asarray(priors.get("Psi_u", np.zeros((self.d_phi, self.d_phi))), dtype=float)
        nu_g = float(priors.get("nu_g", 0.0))
        nu_u = float(priors.get("nu_u", 0.0))
        M_B = np.asarray(
            priors.get("M_B", np.zeros((self.d_phi, self.d_g))), dtype=float
        )
        lambda_B = float(priors.get("lambda_B", 1e-6))
        a_R = float(priors.get("a_R", 1e-6))
        b_R = float(priors.get("b_R", 1e-6))
        lambda_theta = float(priors.get("lambda_theta", 0.0))

        w0 = None
        g_init_mean = self.g_mean0
        g_init_cov = self.g_cov0
        u_init_mean: Dict[int, np.ndarray] = {}
        u_init_cov: Dict[int, np.ndarray] = {}
        if init_state is not None:
            if "w" in init_state and init_state["w"] is not None:
                w0 = np.asarray(init_state["w"], dtype=float)
            if "mu_g" in init_state and init_state["mu_g"] is not None:
                g_init_mean = self._normalize_mean_modes(
                    np.asarray(init_state["mu_g"], dtype=float), self.d_g
                )
            if "Sigma_g" in init_state and init_state["Sigma_g"] is not None:
                g_init_cov = self._normalize_cov_modes(
                    np.asarray(init_state["Sigma_g"], dtype=float), self.d_g, default_scale=1.0
                )
            if "mu_u" in init_state and isinstance(init_state["mu_u"], dict):
                for k, arr in init_state["mu_u"].items():
                    u_init_mean[int(k)] = self._normalize_mean_modes(
                        np.asarray(arr, dtype=float), self.d_phi
                    )
            if "Sigma_u" in init_state and isinstance(init_state["Sigma_u"], dict):
                for k, arr in init_state["Sigma_u"].items():
                    u_init_cov[int(k)] = self._normalize_cov_modes(
                        np.asarray(arr, dtype=float), self.d_phi, default_scale=1.0
                    )

        best_params = self._snapshot_params()
        best_score = np.inf

        expert_ids = sorted({int(a) for a in train_actions})
        for avail in train_avail:
            expert_ids.extend(int(k) for k in avail)
        expert_ids = sorted(set(expert_ids))
        for k in expert_ids:
            self._ensure_B(k)

        g_prev = np.stack(
            [g_init_mean[rng.integers(self.M)] for _ in range(len(train_ctx))]
        )
        u_prev = {
            k: np.stack(
                [
                    u_init_mean.get(k, self.u_mean0)[rng.integers(self.M)]
                    for _ in range(len(train_ctx))
                ]
            )
            for k in expert_ids
        }

        # Debug diagnostics: snapshot EM parameters before training
        self._log_em_params_before()

        for em_idx in range(n_em):

            z_samples = []
            g_samples = []
            u_samples = {k: [] for k in expert_ids}

            phi_seq = [self._compute_phi(x) for x in train_ctx]
            total_samples = n_samples + burn_in
            for s in range(total_samples):
                z_seq = self._sample_z_sequence(
                    train_ctx,
                    train_actions,
                    train_residuals,
                    g_prev,
                    u_prev,
                    w0=w0,
                    residuals_full=train_residuals_full,
                    available_sets=train_avail,
                    g_init_mean=g_init_mean,
                    g_init_cov=g_init_cov,
                    u_init_mean=u_init_mean,
                    u_init_cov=u_init_cov,
                    rng=rng,
                )

                A_seq_g = self.A_g[z_seq]
                Q_seq_g = self.Q_g[z_seq]
                if full_feedback:
                    H_seq_g = []
                    R_seq_g = []
                    y_seq_g = []
                    for t in range(len(train_ctx)):
                        phi_t = phi_seq[t]
                        H_rows = []
                        R_vals = []
                        y_vals = []
                        for k in train_avail[t]:
                            k = int(k)
                            Bk = self._ensure_B(k)
                            H_rows.append((Bk.T @ phi_t).reshape(self.d_g))
                            R_vals.append(self._get_R(int(z_seq[t]), k))
                            y_vals.append(
                                float(train_residuals_full[t][k] - phi_t @ u_prev[k][t])
                            )
                        H_seq_g.append(np.asarray(H_rows, dtype=float))
                        R_seq_g.append(np.asarray(R_vals, dtype=float))
                        y_seq_g.append(np.asarray(y_vals, dtype=float))
                    g_seq = self._kalman_sample_multi(
                        A_seq_g,
                        Q_seq_g,
                        H_seq_g,
                        R_seq_g,
                        y_seq_g,
                        g_init_mean[z_seq[0]],
                        g_init_cov[z_seq[0]],
                        rng,
                    )
                else:
                    H_seq_g = np.zeros((len(train_ctx), 1, self.d_g), dtype=float)
                    R_seq_g = np.zeros(len(train_ctx), dtype=float)
                    y_seq_g = np.zeros(len(train_ctx), dtype=float)
                    obs_mask_g = np.ones(len(train_ctx), dtype=bool)
                    for t in range(len(train_ctx)):
                        k = int(train_actions[t])
                        phi_t = phi_seq[t]
                        Bk = self._ensure_B(k)
                        H_seq_g[t] = (Bk.T @ phi_t).reshape(1, self.d_g)
                        R_seq_g[t] = self._get_R(int(z_seq[t]), k)
                        y_seq_g[t] = float(train_residuals[t] - phi_t @ u_prev[k][t])
                    g_seq = self._kalman_sample(
                        A_seq_g,
                        Q_seq_g,
                        H_seq_g,
                        R_seq_g,
                        y_seq_g.reshape(-1, 1),
                        obs_mask_g,
                        g_init_mean[z_seq[0]],
                        g_init_cov[z_seq[0]],
                        rng,
                    )

                u_seq = {}
                for k in expert_ids:
                    A_seq_u = self.A_u[z_seq]
                    Q_seq_u = self.Q_u[z_seq]
                    H_seq_u = np.zeros((len(train_ctx), 1, self.d_phi), dtype=float)
                    R_seq_u = np.zeros(len(train_ctx), dtype=float)
                    y_seq_u = np.zeros(len(train_ctx), dtype=float)
                    obs_mask_u = np.zeros(len(train_ctx), dtype=bool)
                    Bk = self._ensure_B(k)
                    for t in range(len(train_ctx)):
                        if full_feedback:
                            if k not in train_avail[t]:
                                continue
                            resid_val = float(train_residuals_full[t][k])
                        else:
                            if int(train_actions[t]) != k:
                                continue
                            resid_val = float(train_residuals[t])
                        phi_t = phi_seq[t]
                        H_seq_u[t] = phi_t.reshape(1, self.d_phi)
                        R_seq_u[t] = self._get_R(int(z_seq[t]), k)
                        y_seq_u[t] = float(np.squeeze(resid_val - phi_t @ (Bk @ g_seq[t])))
                        obs_mask_u[t] = True
                    u_seq[k] = self._kalman_sample(
                        A_seq_u,
                        Q_seq_u,
                        H_seq_u,
                        R_seq_u,
                        y_seq_u.reshape(-1, 1),
                        obs_mask_u,
                        u_init_mean.get(k, self.u_mean0)[z_seq[0]],
                        u_init_cov.get(k, self.u_cov0)[z_seq[0]],
                        rng,
                    )

                if s >= burn_in:
                    z_samples.append(z_seq.copy())
                    g_samples.append(g_seq.copy())
                    for k in expert_ids:
                        u_samples[k].append(u_seq[k].copy())

                g_prev = g_seq
                u_prev = u_seq

            z_samples = np.asarray(z_samples, dtype=int)
            g_samples = np.asarray(g_samples, dtype=float)
            for k in expert_ids:
                u_samples[k] = np.asarray(u_samples[k], dtype=float)

            S = float(max(1, z_samples.shape[0]))
            T_train = len(train_ctx)
            gamma = np.zeros((T_train, self.M), dtype=float)
            xi = np.zeros((max(T_train - 1, 0), self.M, self.M), dtype=float)
            g_sum = (
                np.zeros((T_train, self.M, self.d_g), dtype=float)
                if self.d_g > 0
                else None
            )
            u_sum = {
                k: np.zeros((T_train, self.M, self.d_phi), dtype=float)
                for k in expert_ids
            }

            for s in range(z_samples.shape[0]):
                z_seq = z_samples[s]
                g_seq = g_samples[s] if self.d_g > 0 else None
                for t in range(T_train):
                    m = int(z_seq[t])
                    gamma[t, m] += 1.0
                    if g_sum is not None:
                        g_sum[t, m] += g_seq[t]
                for t in range(1, T_train):
                    xi[t - 1, z_seq[t - 1], z_seq[t]] += 1.0
                for k in expert_ids:
                    u_seq = u_samples[k][s]
                    for t in range(T_train):
                        m = int(z_seq[t])
                        u_sum[k][t, m] += u_seq[t]

            gamma /= S
            if xi.size:
                xi /= S

            counts = gamma * S
            denom = np.maximum(counts, 1.0)
            if g_sum is None:
                g_cond_mean = np.zeros((T_train, self.M, 0), dtype=float)
            else:
                g_cond_mean = g_sum / denom[..., None]
            u_cond_mean = {k: u_sum[k] / denom[..., None] for k in expert_ids}

            # M-step: A_g, Q_g
            for m in range(self.M):
                if self.d_g == 0:
                    continue
                sum_gg = np.zeros((self.d_g, self.d_g), dtype=float)
                sum_gprev = np.zeros((self.d_g, self.d_g), dtype=float)
                count = 0.0
                for s in range(z_samples.shape[0]):
                    z_seq = z_samples[s]
                    g_seq = g_samples[s]
                    for t in range(1, T_train):
                        if z_seq[t] != m:
                            continue
                        sum_gg += np.outer(g_seq[t], g_seq[t - 1])
                        sum_gprev += np.outer(g_seq[t - 1], g_seq[t - 1])
                        count += 1.0
                sum_gg /= S
                sum_gprev /= S
                count /= S
                if count <= epsilon_N:
                    continue
                reg = lambda_A_g * np.eye(self.d_g)
                # Use solve for numerical stability: A = X @ Y^{-1} => A^T = solve(Y, X^T)
                self.A_g[m] = np.linalg.solve(
                    sum_gprev + reg,
                    (sum_gg + lambda_A_g * M_A_g).T
                ).T
                sum_gres = np.zeros((self.d_g, self.d_g), dtype=float)
                for s in range(z_samples.shape[0]):
                    z_seq = z_samples[s]
                    g_seq = g_samples[s]
                    for t in range(1, T_train):
                        if z_seq[t] != m:
                            continue
                        diff = g_seq[t] - self.A_g[m] @ g_seq[t - 1]
                        sum_gres += np.outer(diff, diff)
                sum_gres /= S
                self.Q_g[m] = (Psi_g + sum_gres) / max(
                    nu_g + count + self.d_g + 1.0, self.eps
                )

            # M-step: A_u, Q_u
            for m in range(self.M):
                if self.d_phi == 0:
                    continue
                sum_uu = np.zeros((self.d_phi, self.d_phi), dtype=float)
                sum_uprev = np.zeros((self.d_phi, self.d_phi), dtype=float)
                count = 0.0
                for k in expert_ids:
                    u_samp = u_samples[k]
                    for s in range(z_samples.shape[0]):
                        z_seq = z_samples[s]
                        u_seq = u_samp[s]
                        for t in range(1, T_train):
                            if z_seq[t] != m:
                                continue
                            sum_uu += np.outer(u_seq[t], u_seq[t - 1])
                            sum_uprev += np.outer(u_seq[t - 1], u_seq[t - 1])
                            count += 1.0
                sum_uu /= S
                sum_uprev /= S
                count /= S
                if count <= epsilon_N:
                    continue
                reg = lambda_A_u * np.eye(self.d_phi)
                # Use solve for numerical stability: A = X @ Y^{-1} => A^T = solve(Y, X^T)
                self.A_u[m] = np.linalg.solve(
                    sum_uprev + reg,
                    (sum_uu + lambda_A_u * M_A_u).T
                ).T
                sum_ures = np.zeros((self.d_phi, self.d_phi), dtype=float)
                for k in expert_ids:
                    u_samp = u_samples[k]
                    for s in range(z_samples.shape[0]):
                        z_seq = z_samples[s]
                        u_seq = u_samp[s]
                        for t in range(1, T_train):
                            if z_seq[t] != m:
                                continue
                            diff = u_seq[t] - self.A_u[m] @ u_seq[t - 1]
                            sum_ures += np.outer(diff, diff)
                sum_ures /= S
                self.Q_u[m] = (Psi_u + sum_ures) / max(
                    nu_u + count + self.d_phi + 1.0, self.eps
                )

            # Update B_k via ridge regression (mode-conditional)
            if self.d_g > 0 and self.d_phi > 0:
                for k in expert_ids:
                    obs_count = 0
                    for t in range(T_train):
                        if full_feedback:
                            if k not in train_avail[t]:
                                continue
                        else:
                            if int(train_actions[t]) != k:
                                continue
                        obs_count += 1
                    if obs_count <= epsilon_N:
                        continue

                    XTX = np.zeros(
                        (self.d_phi * self.d_g, self.d_phi * self.d_g), dtype=float
                    )
                    XTy = np.zeros(self.d_phi * self.d_g, dtype=float)
                    for t in range(T_train):
                        if full_feedback:
                            if k not in train_avail[t]:
                                continue
                            resid_val = float(train_residuals_full[t][k])
                        else:
                            if int(train_actions[t]) != k:
                                continue
                            resid_val = float(train_residuals[t])
                        phi_t = phi_seq[t]
                        for s in range(z_samples.shape[0]):
                            m = int(z_samples[s, t])
                            g_t = g_samples[s, t]
                            u_t = u_samples[k][s, t]
                            x_vec = np.kron(g_t, phi_t)
                            y_val = float(resid_val - phi_t @ u_t)
                            w_tm = 1.0 / max(self._get_R(m, k), self.eps)
                            weight = w_tm / S
                            XTX += weight * np.outer(x_vec, x_vec)
                            XTy += weight * x_vec * y_val
                    reg = lambda_B * np.eye(self.d_phi * self.d_g)
                    vec_M = M_B.reshape(-1)
                    vec_B = np.linalg.solve(XTX + reg, XTy + lambda_B * vec_M)
                    self.B_dict[k] = vec_B.reshape(self.d_phi, self.d_g)

            # Update R_{m,k}
            if np.ndim(self.R) == 0:
                self.R = np.full(
                    (
                        self.M,
                        self.N if self.N is not None else max(expert_ids) + 1,
                    ),
                    float(self.R),
                )
            for m in range(self.M):
                for k in expert_ids:
                    num = 0.0
                    denom = 0.0
                    for t in range(T_train):
                        if full_feedback:
                            if k not in train_avail[t]:
                                continue
                            resid_val = float(train_residuals_full[t][k])
                        else:
                            if int(train_actions[t]) != k:
                                continue
                            resid_val = float(train_residuals[t])
                        phi_t = phi_seq[t]
                        Bk = self._ensure_B(k)
                        for s in range(z_samples.shape[0]):
                            if int(z_samples[s, t]) != m:
                                continue
                            g_t = g_samples[s, t]
                            u_t = u_samples[k][s, t]
                            resid = float(np.squeeze(resid_val - phi_t @ (Bk @ g_t + u_t)))
                            num += (resid ** 2) / S
                            denom += 1.0 / S
                    if denom <= epsilon_N:
                        continue
                    self.R[m, k] = (b_R + 0.5 * num) / (a_R + 0.5 * denom + 1.0)

            # Update transition parameters (torch model, attention, or linear).
            if self.transition_model is not None or self.transition_hidden_dims is not None:
                self._train_transition_model(
                    train_ctx,
                    xi,
                    lr=theta_lr,
                    steps=theta_steps,
                    weight_decay=lambda_theta,
                    seed=seed,
                )
            elif self.transition_mode == "linear":
                self._train_transition_linear_torch(
                    train_ctx,
                    xi,
                    lr=theta_lr,
                    steps=theta_steps,
                    weight_decay=lambda_theta,
                    seed=seed,
                )
            else:
                self._train_transition_attention_torch(
                    train_ctx,
                    xi,
                    lr=theta_lr,
                    steps=theta_steps,
                    weight_decay=lambda_theta,
                    seed=seed,
                )

            score_std = None
            score_n = None
            score_split_sizes = None
            if do_validation:
                if val_strategy == "rolling":
                    n_splits = int(val_roll_splits) if val_roll_splits is not None else 3
                    score, score_std, score_n, score_split_sizes = self._evaluate_nll_rolling(
                        contexts,
                        available_sets,
                        actions,
                        residuals,
                        residuals_full_arr if full_feedback else None,
                        train_end=split_idx,
                        val_window_len=int(val_window_len),
                        n_splits=n_splits,
                        stride=val_roll_stride,
                    )
                    metric_label = "val_roll_nll_mean"
                else:
                    score = self._evaluate_nll(
                        val_ctx,
                        val_avail,
                        val_actions,
                        val_residuals,
                        residuals_full=val_residuals_full if full_feedback else None,
                    )
                    metric_label = "val_nll"
            else:
                score = self._evaluate_nll(
                    train_ctx,
                    train_avail,
                    train_actions,
                    train_residuals,
                    residuals_full=train_residuals_full if full_feedback else None,
                )
                metric_label = "train_nll"
            if print_val_loss:
                if metric_label == "val_roll_nll_mean":
                    std_str = (
                        f" val_roll_nll_std={float(score_std):.6f}" if score_std is not None else ""
                    )
                    split_str = (
                        f" splits={int(score_n)}" if score_n is not None else ""
                    )
                    size_str = ""
                    if score_split_sizes:
                        train_sizes = ",".join(str(s[0]) for s in score_split_sizes)
                        val_sizes = ",".join(str(s[1]) for s in score_split_sizes)
                        size_str = f" train_n=[{train_sizes}] val_n=[{val_sizes}]"
                    print(
                        f"[FactorizedSLDS EM] iter {em_idx + 1}/{n_em} "
                        f"{metric_label}={score:.6f}{std_str}{split_str}{size_str}"
                    )
                    if score_n is not None and int(score_n) < 2:
                        print(
                            "[FactorizedSLDS EM] rolling validation collapsed to a single "
                            "split; reduce em_val_roll_len or increase em_val_fraction/em_tk "
                            "to enable multiple windows."
                        )
                else:
                    print(
                        f"[FactorizedSLDS EM] iter {em_idx + 1}/{n_em} "
                        f"{metric_label}={score:.6f}"
                    )
            if score < best_score:
                best_score = score
                best_params = self._snapshot_params()

        self._restore_params(best_params)
        stage = transition_log_stage or "post_em"
        self._maybe_log_transition_params(stage)

        # Debug diagnostics: log EM parameter changes
        self._log_em_params_after(stage=stage)

        if print_val_loss:
            if do_validation:
                if val_strategy == "rolling":
                    print(f"[FactorizedSLDS EM] best_val_roll_nll_mean={best_score:.6f}")
                else:
                    print(f"[FactorizedSLDS EM] best_val_nll={best_score:.6f}")
            else:
                print(f"[FactorizedSLDS EM] best_train_nll={best_score:.6f}")
        return {"best_nll": best_score}
