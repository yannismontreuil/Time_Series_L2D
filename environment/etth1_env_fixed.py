import os
from typing import List, Optional, Tuple

import numpy as np

try:  # optional dependency for PyTorch-based experts
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    print(f"PyTorch successfully imported in etth1_env_fixed.py: {torch.__version__}")
except ImportError as e:  # More specific exception handling
    print(f"PyTorch import failed in etth1_env_fixed.py: {e}")
    torch = None
except Exception as e:  # Catch other exceptions
    print(f"Unexpected error importing PyTorch in etth1_env_fixed.py: {e}")
    torch = None


def _select_torch_device():
    """Pick a torch device, preferring CUDA, then MPS, else CPU."""
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ETTh1TimeSeriesEnv:
    """
    Environment that wraps the ETTh1 dataset (electricity transformer
    temperature) and exposes the same interface as SyntheticTimeSeriesEnv.

    - We treat the oil temperature (column \"OT\") as the target y_t.
    - Context x_t is the previous oil temperature y_{t-1} (lag-1), with
      x_0 = 0 by default.
    - Experts are configurable predictors with different types:
      * AR(1) models with different variance
      * Neural networks trained on different data portions
      * AR(2) models trained on data segments
    - Loss is squared error between y_t and the selected expert's prediction.

    This class is designed so that existing router code (F-SLDS Router,
    UCB baselines, L2D baselines, plotting utilities) can be reused
    without modification.

    **NOTE**: x_t is used to represent lagged values of y_t, i.e. context
    at time t is x_t = y_{t-1}.
    """

    def __init__(
        self,
        csv_path: str = "data/ETTh1.csv",
        target_column: str = "OT",
        enabled_experts: Optional[List[str]] = None,  # New: list of expert types to enable
        T: Optional[int] = None,
        seed: int = 0,
        unavailable_expert_idx: Optional[List[int]] = None,
        unavailable_intervals: Optional[List[Tuple[int, int]]] = None,
        arrival_expert_idx: Optional[List[int]] = None,
        arrival_intervals: Optional[List[Tuple[int, int]]] = None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ETTh1 CSV not found at path: {csv_path}")

        # Load target column from CSV (simple parser; no external deps).
        y_all: List[float] = []
        with open(csv_path, "r", encoding="utf-8") as f: # read mode
            header_line = f.readline()
            if not header_line:
                raise ValueError("Empty ETTh1 CSV file.")
            header = header_line.strip().split(",")
            if target_column not in header:
                raise ValueError(
                    f"Target column '{target_column}' not found in header "
                    f"{header} of {csv_path}"
                )
            col_idx = header.index(target_column)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) <= col_idx:
                    continue
                try:
                    val = float(parts[col_idx])
                except ValueError:
                    continue
                y_all.append(val)

        if len(y_all) < 2:
            raise ValueError(
                "ETTh1 dataset must contain at least 2 observations for "
                "lagged context construction."
            )

        y_arr = np.asarray(y_all, dtype=float)

        # Truncate or use full series according to T
        if T is None:
            T_eff = y_arr.shape[0]
        else:
            T_eff = int(T)
            T_eff = max(2, min(T_eff, y_arr.shape[0]))
            y_arr = y_arr[:T_eff]

        self.T = T_eff
        self.target_column = str(target_column)

        # True target series
        self.y = y_arr.copy()

        # Context x_t := y_{t-1}, with x_0 = 0.0
        self.x = np.zeros(self.T, dtype=float)
        if self.T > 1:
            self.x[1:] = self.y[:-1]

        # Expert configuration: flexible enable/disable system
        # Available expert types:
        # - ar1_low_var: AR(1) with low variance
        # - ar1_high_var: AR(1) with higher variance
        # - nn_early: NN trained on early portion of data
        # - nn_late: NN trained on late portion of data
        # - ar2_segment1: AR(2) trained on first data segment
        # - ar2_segment2: AR(2) trained on second data segment

        self.available_expert_types = [
            "ar1_low_var", "ar1_high_var", "nn_early", "nn_late", "ar2_segment1", "ar2_segment2"
        ]


        if enabled_experts is None:
            raise ValueError(
                "enabled_experts must be provided as a list of expert types."
            )

        self.enabled_experts = enabled_experts
        self.num_experts = len(enabled_experts)

        # Validate enabled experts
        for expert_type in self.enabled_experts:
            if expert_type not in self.available_expert_types:
                raise ValueError(f"Unknown expert type: {expert_type}. Available: {self.available_expert_types}")

        # Initialize expert parameters
        # Here we use num_experts to ensure consistent indexing
        self.expert_weights = np.zeros(self.num_experts, dtype=float)
        self.expert_biases = np.zeros(self.num_experts, dtype=float)
        self.expert_types = {}  # Maps expert index to expert type
        self._nn_params = [None] * self.num_experts
        self._nn_expert_ids: List[int] = []
        self._ar2_params = {}  # Maps expert index to AR(2) parameters

        # Fit base AR(1) model for reference
        idx_lin = np.arange(1, self.T, dtype=int)
        if idx_lin.size >= 2:
            x_lin = self.x[idx_lin]
            y_lin = self.y[idx_lin]
            x_mean = float(x_lin.mean())
            y_mean = float(y_lin.mean())
            x_centered = x_lin - x_mean
            y_centered = y_lin - y_mean
            var_x = float(np.dot(x_centered, x_centered))
            if var_x <= 1e-12:
                w_lin = 0.0
            else:
                cov_xy = float(np.dot(x_centered, y_centered))
                w_lin = cov_xy / var_x
            b_lin = y_mean - w_lin * x_mean
        else:
            w_lin = 0.95
            b_lin = 0.0

        # Initialize each enabled expert
        rng = np.random.default_rng(seed)
        for i, expert_type in enumerate(self.enabled_experts):
            self.expert_types[i] = expert_type
            self._initialize_expert(i, expert_type, w_lin, b_lin, rng)

        # Expert availability over time: all experts available by default.
        self.availability = np.ones((self.T, self.num_experts), dtype=int)

        # Optional unavailability intervals for a single expert
        if (
            unavailable_expert_idx is not None
            and 0 <= unavailable_expert_idx[0] < self.num_experts
            and 0 <= unavailable_expert_idx[1] < self.num_experts
        ):
            self.availability[:, arrival_expert_idx[0]] = 0
            self.availability[:, arrival_expert_idx[1]] = 0
            if unavailable_intervals is not None and len(unavailable_intervals) >= 2:
                if unavailable_intervals[0] is not None:
                    for start, end in unavailable_intervals[0]:
                        t_start = max(int(start), 0)
                        t_end = min(int(end) + 1, self.T)
                        if t_start >= self.T or t_end <= 0:
                            continue
                        if t_start < t_end:
                            self.availability[t_start:t_end, unavailable_expert_idx[0]] = 0
                if unavailable_intervals[1] is not None:
                    for start, end in unavailable_intervals[1]:
                        t_start = max(int(start), 0)
                        t_end = min(int(end) + 1, self.T)
                        if t_start >= self.T or t_end <= 0:
                            continue
                        if t_start < t_end:
                            self.availability[t_start:t_end, unavailable_expert_idx[1]] = 0

        # Optional dynamic expert arrival intervals for 2 experts
        if (
            arrival_expert_idx is not None
            and len(arrival_expert_idx) >= 2
            and 0 <= arrival_expert_idx[0] < self.num_experts
            and 0 <= arrival_expert_idx[1] < self.num_experts
        ):
            self.availability[:, arrival_expert_idx[0]] = 0
            self.availability[:, arrival_expert_idx[1]] = 0
            if arrival_intervals is not None and len(arrival_intervals) >= 2:
                if arrival_intervals[0] is not None:
                    for start, end in arrival_intervals[0]:
                        t_start = max(int(start), 0)
                        t_end = min(int(end) + 1, self.T)
                        if t_start >= self.T or t_end <= 0:
                            continue
                        if t_start < t_end:
                            self.availability[t_start:t_end, arrival_expert_idx[0]] = 1
                if arrival_intervals[1] is not None:
                    for start, end in arrival_intervals[1]:
                        t_start = max(int(start), 0)
                        t_end = min(int(end) + 1, self.T)
                        if t_start >= self.T or t_end <= 0:
                            continue
                        if t_start < t_end:
                            self.availability[t_start:t_end, arrival_expert_idx[1]] = 1

        # There is no known discrete regime sequence for ETTh1; we set all
        # entries to 0 so that plotting utilities can still show a regime
        # track without breaking. But this should be ignored.
        self.z = np.zeros(self.T, dtype=int)

        # Track last time index queried via get_context so that
        # expert_predict can (optionally) depend only on the current
        # context x_t while still supporting interfaces shared with the
        # synthetic environment.
        self._last_t: Optional[int] = None

    def _initialize_expert(self, expert_idx: int, expert_type: str, w_base: float, b_base: float, rng: np.random.Generator):
        """Initialize a specific expert based on its type."""
        if expert_type == "ar1_low_var":
            # AR(1) with low variance (close to optimal)
            self.expert_weights[expert_idx] = w_base * 0.95
            self.expert_biases[expert_idx] = b_base + rng.normal(0, 0.05)

        elif expert_type == "ar1_high_var":
            # AR(1) with higher variance (more misspecified)
            self.expert_weights[expert_idx] = w_base * 1.20
            self.expert_biases[expert_idx] = b_base + rng.normal(0, 0.15)

        elif expert_type in ["nn_early", "nn_late"]:
            # Neural network experts
            self._initialize_nn_expert(expert_idx, expert_type, rng)
            self._nn_expert_ids.append(expert_idx)

        elif expert_type in ["ar2_segment1", "ar2_segment2"]:
            # AR(2) experts trained on different segments
            self._initialize_ar2_expert(expert_idx, expert_type, w_base, b_base, rng)
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    def _initialize_nn_expert(self, expert_idx: int, expert_type: str, rng: np.random.Generator):
        """Initialize neural network expert."""
        # Training data preparation
        idx_all = np.arange(1, self.T, dtype=int)
        x_all = self.x[idx_all]
        y_all = self.y[idx_all]
        n_all = len(x_all)

        if n_all < 10:
            # Fallback for short series
            self.expert_weights[expert_idx] = 0.9
            self.expert_biases[expert_idx] = 0.0
            return

        # Define training segments
        if expert_type == "nn_early":
            # Train on first 40% of data
            end_idx = max(1, int(0.4 * n_all))
            x_train = x_all[:end_idx]
            y_train = y_all[:end_idx]
        else:  # nn_late
            # Train on next 40% of data
            start_idx = max(1, int(0.4 * n_all))
            end_idx = max(1, int(0.8 * n_all))
            x_train = x_all[start_idx:end_idx]
            y_train = y_all[start_idx:end_idx]

        # Validation set: last 20% of full data
        n_val = max(1, int(0.2 * n_all))
        x_val = x_all[-n_val:]
        y_val = y_all[-n_val:]

        # Train neural network
        params = self._train_nn_expert([x_train], [y_train], x_val, y_val, 8, rng)
        self._nn_params[expert_idx] = params

    def _initialize_ar2_expert(self, expert_idx: int, expert_type: str, w_base: float, b_base: float, rng: np.random.Generator):
        """Initialize AR(2) expert trained on a data segment."""
        # AR(2) model: y_t = w1 * y_{t-1} + w2 * y_{t-2} + b + noise

        # Define training segments
        n_total = self.T
        if expert_type == "ar2_segment1":
            # Train on first 30% of data
            start_idx = 2  # Need at least 2 lags for AR(2)
            end_idx = max(start_idx + 1, int(0.3 * n_total))
        else:  # ar2_segment2
            # Train on middle 30% of data
            start_idx = max(2, int(0.35 * n_total))
            end_idx = max(start_idx + 1, int(0.65 * n_total))

        if end_idx <= start_idx + 1:
            # Fallback to simple AR(1) if not enough data
            self.expert_weights[expert_idx] = w_base * 0.9
            self.expert_biases[expert_idx] = b_base
            self._ar2_params[expert_idx] = None
            return

        # Prepare AR(2) regression data
        y_target = self.y[start_idx:end_idx]
        y_lag1 = self.y[start_idx-1:end_idx-1]
        y_lag2 = self.y[start_idx-2:end_idx-2]

        n_samples = len(y_target)
        if n_samples < 3:
            # Fallback
            self.expert_weights[expert_idx] = w_base * 0.9
            self.expert_biases[expert_idx] = b_base
            self._ar2_params[expert_idx] = None
            return

        # Build regression matrix [1, y_{t-1}, y_{t-2}]
        X = np.column_stack([np.ones(n_samples), y_lag1, y_lag2])

        # Ridge regression for stability
        lam = 1e-3
        XtX = X.T @ X + lam * np.eye(3)
        XtY = X.T @ y_target

        try:
            coeffs = np.linalg.solve(XtX, XtY)
            b_ar2, w1_ar2, w2_ar2 = coeffs

            # Store AR(2) parameters
            self._ar2_params[expert_idx] = (float(w1_ar2), float(w2_ar2), float(b_ar2))

            # For compatibility, store AR(1) approximation in main arrays
            self.expert_weights[expert_idx] = float(w1_ar2)
            self.expert_biases[expert_idx] = float(b_ar2)

        except np.linalg.LinAlgError:
            # Fallback on numerical issues
            self.expert_weights[expert_idx] = w_base * 0.9
            self.expert_biases[expert_idx] = b_base
            self._ar2_params[expert_idx] = None

    def _train_nn_expert(
        self,
        x_train_segments: List[np.ndarray],
        y_train_segments: List[np.ndarray],
        x_val_global: np.ndarray,
        y_val_global: np.ndarray,
        hidden_dim: int,
        rng_local: np.random.Generator,
        num_epochs: int = 100,  # Reduced for faster training
        learning_rate: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Train neural network expert."""
        # Concatenate all segments for training
        x_train_list = []
        y_train_list = []
        for x_seg, y_seg in zip(x_train_segments, y_train_segments):
            if x_seg.size > 0 and y_seg.size > 0:
                x_train_list.append(x_seg)
                y_train_list.append(y_seg)

        if not x_train_list:
            # Fallback: degenerate zero network
            W1 = np.zeros((hidden_dim, 1), dtype=float)
            b1 = np.zeros(hidden_dim, dtype=float)
            W2 = np.zeros((hidden_dim, 1), dtype=float)
            b2 = 0.0
            return W1, b1, W2, b2

        x_train = np.concatenate(x_train_list)
        y_train = np.concatenate(y_train_list)

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        N_total = x_train.shape[0]
        if N_total == 0:
            # Fallback: degenerate zero network
            W1 = np.zeros((hidden_dim, 1), dtype=float)
            b1 = np.zeros(hidden_dim, dtype=float)
            W2 = np.zeros((hidden_dim, 1), dtype=float)
            b2 = 0.0
            return W1, b1, W2, b2

        # Validation data
        if x_val_global.ndim == 1:
            x_val = x_val_global.reshape(-1, 1)
        else:
            x_val = x_val_global.reshape(-1, 1)
        if y_val_global.ndim == 1:
            y_val = y_val_global.reshape(-1, 1)
        else:
            y_val = y_val_global.reshape(-1, 1)

        # Simple NumPy-based training
        W1 = rng_local.normal(loc=0.0, scale=0.1, size=(hidden_dim, 1))
        b1 = np.zeros(hidden_dim, dtype=float)
        W2 = rng_local.normal(loc=0.0, scale=0.1, size=(hidden_dim, 1))
        b2 = 0.0
        best_W1 = W1.copy()
        best_b1 = b1.copy()
        best_W2 = W2.copy()
        best_b2 = b2
        best_val_rmse = np.inf

        for _ in range(num_epochs):
            # Forward pass
            z1 = x_train @ W1.T + b1  # (N_total, hidden_dim)
            h = np.tanh(z1)
            y_hat = h @ W2 + b2  # (N_total, 1)

            diff = y_hat - y_train
            d_yhat = (2.0 / max(N_total, 1)) * diff

            # Backprop
            d_W2 = h.T @ d_yhat  # (hidden_dim, 1)
            d_b2 = float(d_yhat.sum())

            d_h = d_yhat @ W2.T  # (N_total, hidden_dim)
            d_z1 = d_h * (1.0 - np.tanh(z1) ** 2)
            d_W1 = d_z1.T @ x_train  # (hidden_dim, 1)
            d_b1 = d_z1.sum(axis=0)  # (hidden_dim,)

            # Gradient step
            W1 -= learning_rate * d_W1
            W2 -= learning_rate * d_W2
            b1 -= learning_rate * d_b1
            b2 -= learning_rate * d_b2

            # Validation RMSE for checkpoint selection
            if x_val.shape[0] > 0:
                z1_val = x_val @ W1.T + b1
                h_val = np.tanh(z1_val)
                y_hat_val = h_val @ W2 + b2
                mse_val = float(np.mean((y_hat_val - y_val) ** 2))
                rmse_val = float(np.sqrt(mse_val))
                if rmse_val < best_val_rmse:
                    best_val_rmse = rmse_val
                    best_W1 = W1.copy()
                    best_b1 = b1.copy()
                    best_W2 = W2.copy()
                    best_b2 = b2

        return best_W1, best_b1, best_W2, best_b2

    # ------------------------------------------------------------------
    # Interface methods (matching SyntheticTimeSeriesEnv)
    # ------------------------------------------------------------------

    def get_context(self, t: int) -> np.ndarray:
        self._last_t = int(t)
        return np.array([self.x[t]], dtype=float)

    def expert_predict(self, j: int, x_t: np.ndarray) -> float:
        j_int = int(j)

        if j_int >= self.num_experts:
            raise IndexError(f"Expert index {j_int} out of range [0, {self.num_experts})")

        expert_type = self.expert_types.get(j_int, "unknown")

        # AR(2) experts
        if (expert_type in ["ar2_segment1", "ar2_segment2"] and
            j_int in self._ar2_params and
            self._ar2_params[j_int] is not None and
            self._last_t is not None):

            w1, w2, b = self._ar2_params[j_int]
            t_idx = int(self._last_t)

            # Get y_{t-1} and y_{t-2}
            y_lag1 = self.y[max(0, t_idx - 1)] if t_idx >= 1 else 0.0
            y_lag2 = self.y[max(0, t_idx - 2)] if t_idx >= 2 else 0.0

            return float(w1 * y_lag1 + w2 * y_lag2 + b)

        # Neural network experts
        if j_int in self._nn_expert_ids:
            params = self._nn_params[j_int]
            if params is None:
                # Fallback: linear prediction if NN params missing
                w = self.expert_weights[j_int]
                b = self.expert_biases[j_int]
                return float(w * float(x_t[0]) + b)

            W1, b1, W2, b2 = params
            # Single-sample forward pass
            x_val = float(x_t[0])
            x_arr = np.array([[x_val]], dtype=float)  # shape (1, 1)
            z1 = x_arr @ W1.T + b1  # shape (1, hidden_dim)
            h = np.tanh(z1)[0]      # shape (hidden_dim,)
            y_hat = float(h @ W2.reshape(-1) + b2)
            return y_hat

        # Default AR(1) experts (ar1_low_var, ar1_high_var)
        w = self.expert_weights[j_int]
        b = self.expert_biases[j_int]
        return float(w * float(x_t[0]) + b)

    def all_expert_predictions(self, x_t: np.ndarray) -> np.ndarray:
        return np.array(
            [self.expert_predict(j, x_t) for j in range(self.num_experts)],
            dtype=float,
        )

    def losses(self, t: int) -> np.ndarray:
        """
        Return squared losses ℓ_{j,t} for all experts at time t.
        """
        x_t = self.get_context(t)
        y_t = self.y[t]
        preds = self.all_expert_predictions(x_t)
        return (preds - y_t) ** 2

    def get_available_experts(self, t: int) -> np.ndarray:
        """
        Return indices of experts available at time t.
        """
        mask = self.availability[t].astype(bool)
        return np.where(mask)[0]
