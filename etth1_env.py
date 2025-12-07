import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # optional dependency for PyTorch-based experts
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


class ETTh1TimeSeriesEnv:
    """
    Environment that wraps the ETTh1 dataset (electricity transformer
    temperature) and exposes the same interface as SyntheticTimeSeriesEnv.

    - We treat the oil temperature (column \"OT\") as the target y_t.
    - Context x_t is the previous oil temperature y_{t-1} (lag-1), with
      x_0 = 0 by default.
    - Experts are simple linear predictors of y_t based on x_t:
          y_hat^{(j)}_t = w_j * x_t + b_j
    - Loss is squared error between y_t and the selected expert's
      prediction.

    This class is designed so that existing router code (SLDSIMMRouter,
    SLDSIMMRouter_Corr, L2D baselines, plotting utilities) can be reused
    without modification.
    """

    def __init__(
        self,
        csv_path: str = "Data/ETTh1.csv",
        target_column: str = "OT",
        num_experts: int = 3,
        num_regimes: int = 2,
        T: Optional[int] = None,
        seed: int = 0,
        unavailable_expert_idx: Optional[int] = None,
        unavailable_intervals: Optional[List[Tuple[int, int]]] = None,
        arrival_expert_idx: Optional[int] = None,
        arrival_intervals: Optional[List[Tuple[int, int]]] = None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ETTh1 CSV not found at path: {csv_path}")

        # Load target column from CSV (simple parser; no external deps).
        y_all: List[float] = []
        with open(csv_path, "r", encoding="utf-8") as f:
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
        self.num_experts = int(num_experts)
        self.num_regimes = int(num_regimes)
        self.target_column = str(target_column)

        # True target series
        self.y = y_arr.copy()

        # Context x_t := y_{t-1}, with x_0 = 0.0
        self.x = np.zeros(self.T, dtype=float)
        if self.T > 1:
            self.x[1:] = self.y[:-1]

        # Experts:
        #   - Experts 0 and 1: simple linear predictors y_hat = w_j x + b_j
        #     that are explicitly correlated (similar weights/biases).
        #   - Experts 2 and 3 (when present): small neural-network experts
        #     trained on different historical segments of the ETTh1
        #     series, simulating access to different databases.
        #   - Expert 4 (when present): a stronger global AR(1) baseline
        #     y_t ≈ w * x_t + b, fitted by least squares on the full
        #     ETTh1 history. This provides a more competitive constant
        #     expert for the ETTh1 experiment.
        #
        # This implements the requested structure for config_etth1:
        #   - 5 experts total.
        #   - Expert 0 and 1 correlated linear baselines. Expert 0 is
        #     fitted as a data-driven AR(1) baseline on the full
        #     history, while expert 1 is a slightly misspecified
        #     variant (heavily correlated but intentionally a bit
        #     worse on average).
        #   - Experts 2 and 3 are NN-based forecasters with different
        #     training histories; expert 4 is a stronger global
        #     multi-lag baseline fitted on all of ETTh1.

        # Fit a simple AR(1) baseline y_t ≈ w * x_t + b on the full
        # history to initialize the linear experts. Expert 0 will use
        # (w, b), and expert 1 will use a slightly perturbed slope so
        # that it remains strongly correlated but consistently a bit
        # worse than expert 0 on average.
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
            # Fallback for extremely short series.
            w_lin = 0.95
            b_lin = 0.0

        # Expert 0: slightly misspecified AR(1) baseline (2% slope
        # perturbation) so that it is close to, but not equal to, the
        # least-squares optimum.
        # Expert 1: more strongly perturbed slope (5% larger
        # magnitude), sharing the same intercept. Since (w_lin, b_lin)
        # is the least-squares optimum, these perturbations make
        # experts 0 and 1 progressively worse in terms of average MSE
        # while keeping them highly correlated.
        w0 = w_lin * 0.98
        w1 = w_lin * 1.05
        base_weights = np.array([w0, w1], dtype=float)
        base_biases = np.array([b_lin, b_lin], dtype=float)

        rng = np.random.default_rng(seed)

        # Default: initialize linear weights/biases for all experts;
        # NN-based experts will override these via learned parameters.
        self.expert_weights = np.zeros(self.num_experts, dtype=float)
        self.expert_biases = np.zeros(self.num_experts, dtype=float)
        # Linear, correlated experts 0 and 1.
        n_lin = min(2, self.num_experts)
        self.expert_weights[:n_lin] = base_weights[:n_lin]
        self.expert_biases[:n_lin] = base_biases[:n_lin]

        # Neural-network experts: indices >= 2 (if any).
        self._nn_params = [None] * self.num_experts  # type: ignore[var-annotated]
        self._nn_expert_ids: List[int] = []

        # Placeholder for expert-4 multi-lag baseline parameters:
        #   - _expert4_lags: array of positive integer lags (in steps),
        #   - _expert4_ar_params: (weights_for_lags, intercept).
        self._expert4_lags: Optional[np.ndarray] = None
        self._expert4_ar_params: Optional[Tuple[np.ndarray, float]] = None

        # If expert 4 is present, fit a strong multi-lag AR baseline on
        # the full history using several ETTh1-relevant lags (1, 24, 168
        # hours). This gives expert 4 access to richer temporal
        # structure than the other experts, making it a very strong
        # global baseline.
        if self.num_experts > 4:
            candidate_lags = np.array([1, 24, 168], dtype=int)
            max_lag = int(candidate_lags.max())
            if self.T > max_lag + 1:
                idx_all_lin = np.arange(max_lag, self.T, dtype=int)
                if idx_all_lin.size >= 2:
                    y_all_lin = self.y[idx_all_lin]
                    num_samples = idx_all_lin.shape[0]
                    num_features = 1 + candidate_lags.shape[0]
                    X = np.ones((num_samples, num_features), dtype=float)
                    for k, lag in enumerate(candidate_lags):
                        X[:, 1 + k] = self.y[idx_all_lin - int(lag)]
                    # Ridge-regularized least squares for numerical stability.
                    lam = 1e-3
                    XtX = X.T @ X
                    # Add λI to the diagonal.
                    diag_indices = np.arange(num_features)
                    XtX[diag_indices, diag_indices] += lam
                    XtY = X.T @ y_all_lin
                    coeffs = np.linalg.solve(XtX, XtY)
                    b_ols = float(coeffs[0])
                    w_lags = coeffs[1:].astype(float)
                    self._expert4_lags = candidate_lags.copy()
                    self._expert4_ar_params = (w_lags, b_ols)
                    # For backwards compatibility, keep a simple linear view
                    # on the lag-1 coefficient in expert_weights/biases.
                    if w_lags.size > 0:
                        self.expert_weights[4] = float(w_lags[0])
                        self.expert_biases[4] = b_ols

        if self.num_experts > 2:
            # Training data: pairs (x_t, y_t) for t = 1,...,T-1.
            idx_all = np.arange(1, self.T, dtype=int)
            x_all = self.x[idx_all]
            y_all = self.y[idx_all]
            n_all = idx_all.shape[0]
            if n_all >= 3:
                third = n_all // 3
            else:
                third = max(1, n_all // 2)
            two_third = min(2 * third, n_all)

            # Global validation set used for checkpoint selection:
            # we take the last 20% of the full history (at least one
            # sample) so that each NN expert is evaluated on the same
            # validation sequence, not only on a small subset of its
            # dedicated training history.
            if n_all >= 5:
                n_val_global = max(1, int(0.2 * n_all))
            else:
                n_val_global = 1
            if n_val_global >= n_all:
                n_val_global = 1
            idx_val_global = np.arange(n_all - n_val_global, n_all, dtype=int)
            x_val_global = x_all[idx_val_global]
            y_val_global = y_all[idx_val_global]

            # Segments simulating different historical databases. We
            # deliberately use overlapping segments so that NN experts
            # share some training data and become indirectly correlated:
            #   - Expert 2: early + middle history (0 .. two_third)
            #   - Expert 3: middle + late history (third .. end)
            seg_early = slice(0, two_third)
            seg_late = slice(third, n_all)

            def _train_nn_expert(
                x_train: np.ndarray,
                y_train: np.ndarray,
                x_val_global: np.ndarray,
                y_val_global: np.ndarray,
                hidden_dim: int,
                rng_local: np.random.Generator,
                num_epochs: int = 1000,
                learning_rate: float = 1e-2,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
                """
                Train a tiny 1-hidden-layer MLP for an expert:
                    h = tanh(W1 x + b1),  y_hat = W2^T h + b2

                For the ETTh1 experiment, experts 2 and 3 are trained via
                PyTorch when available, with a fallback to a pure NumPy
                implementation if torch is not installed. The returned
                parameters are always NumPy arrays so that inference can
                remain framework-agnostic.
                """
                if x_train.ndim == 1:
                    x_train = x_train.reshape(-1, 1)
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1)

                N_total = x_train.shape[0]
                if N_total == 0:
                    # Fallback: degenerate zero network.
                    W1 = np.zeros((hidden_dim, 1), dtype=float)
                    b1 = np.zeros(hidden_dim, dtype=float)
                    W2 = np.zeros((hidden_dim, 1), dtype=float)
                    b2 = 0.0
                    return W1, b1, W2, b2

                # Global validation set (shared across experts)
                if x_val_global.ndim == 1:
                    x_val = x_val_global.reshape(-1, 1)
                else:
                    x_val = x_val_global.reshape(-1, 1)
                if y_val_global.ndim == 1:
                    y_val = y_val_global.reshape(-1, 1)
                else:
                    y_val = y_val_global.reshape(-1, 1)

                # Prefer PyTorch training when available.
                if torch is not None:
                    # Construct a simple 1-hidden-layer MLP with tanh.
                    device = torch.device("cpu")

                    class _MLP(nn.Module):  # type: ignore[misc]
                        def __init__(self, in_dim: int, hid_dim: int):
                            super().__init__()
                            self.lin1 = nn.Linear(in_dim, hid_dim)
                            self.lin2 = nn.Linear(hid_dim, 1)

                        def forward(self, x_t: torch.Tensor) -> torch.Tensor:
                            h_t = torch.tanh(self.lin1(x_t))
                            return self.lin2(h_t)

                    # Seed PyTorch deterministically using rng_local.
                    torch_seed = int(rng_local.integers(0, 2**31 - 1))
                    torch.manual_seed(torch_seed)

                    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
                    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
                    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
                    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

                    model = _MLP(in_dim=1, hid_dim=hidden_dim).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    loss_fn = nn.MSELoss()

                    best_state = None
                    best_val_rmse = float("inf")

                    for _ in range(num_epochs):
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        pred = model(x_train_t)
                        loss = loss_fn(pred, y_train_t)
                        loss.backward()
                        optimizer.step()

                        if x_val_t.shape[0] > 0:
                            model.eval()
                            with torch.no_grad():
                                pred_val = model(x_val_t)
                                mse_val = float(loss_fn(pred_val, y_val_t).item())
                                rmse_val = float(np.sqrt(mse_val))
                            if rmse_val < best_val_rmse:
                                best_val_rmse = rmse_val
                                best_state = {
                                    "lin1_weight": model.lin1.weight.detach().cpu().numpy(),
                                    "lin1_bias": model.lin1.bias.detach().cpu().numpy(),
                                    "lin2_weight": model.lin2.weight.detach().cpu().numpy(),
                                    "lin2_bias": model.lin2.bias.detach().cpu().numpy(),
                                }

                    if best_state is None:
                        # Fallback: use current parameters if validation never improved.
                        best_state = {
                            "lin1_weight": model.lin1.weight.detach().cpu().numpy(),
                            "lin1_bias": model.lin1.bias.detach().cpu().numpy(),
                            "lin2_weight": model.lin2.weight.detach().cpu().numpy(),
                            "lin2_bias": model.lin2.bias.detach().cpu().numpy(),
                        }

                    W1_t = best_state["lin1_weight"]  # shape (hidden_dim, 1)
                    b1_t = best_state["lin1_bias"]    # shape (hidden_dim,)
                    W2_t = best_state["lin2_weight"].T  # convert (1, hidden_dim) -> (hidden_dim, 1)
                    b2_t = float(best_state["lin2_bias"].reshape(()))

                    W1 = np.asarray(W1_t, dtype=float)
                    b1 = np.asarray(b1_t, dtype=float)
                    W2 = np.asarray(W2_t, dtype=float)
                    b2 = float(b2_t)
                    return W1, b1, W2, b2

                # Fallback: original NumPy-based training if torch is not available.
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

                    # Validation RMSE for checkpoint selection.
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

            # Expert 2: NN trained on early + middle history (overlaps
            # with expert 3 on the middle part).
            if self.num_experts > 2:
                W1_2, b1_2, W2_2, b2_2 = _train_nn_expert(
                    x_all[seg_early],
                    y_all[seg_early],
                    x_val_global,
                    y_val_global,
                    hidden_dim=8,
                    rng_local=rng,
                )
                self._nn_params[2] = (W1_2, b1_2, W2_2, b2_2)
                self._nn_expert_ids.append(2)

            # Expert 3: NN trained on middle + late history (overlaps
            # with expert 2 on the middle part).
            if self.num_experts > 3:
                W1_3, b1_3, W2_3, b2_3 = _train_nn_expert(
                    x_all[seg_late],
                    y_all[seg_late],
                    x_val_global,
                    y_val_global,
                    hidden_dim=8,
                    rng_local=rng,
                )
                self._nn_params[3] = (W1_3, b1_3, W2_3, b2_3)
                self._nn_expert_ids.append(3)


        # Expert availability over time: all experts available by default.
        self.availability = np.ones((self.T, self.num_experts), dtype=int)

        # Optional unavailability intervals for a single expert
        if (
            unavailable_expert_idx is not None
            and 0 <= unavailable_expert_idx < self.num_experts
        ):
            if unavailable_intervals is not None:
                for start, end in unavailable_intervals:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, self.T)
                    if t_start >= self.T or t_end <= 0:
                        continue
                    if t_start < t_end:
                        self.availability[t_start:t_end, unavailable_expert_idx] = 0

        # Optional dynamic expert arrival intervals for a single expert
        if (
            arrival_expert_idx is not None
            and 0 <= arrival_expert_idx < self.num_experts
        ):
            self.availability[:, arrival_expert_idx] = 0
            if arrival_intervals is not None:
                for start, end in arrival_intervals:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, self.T)
                    if t_start >= self.T or t_end <= 0:
                        continue
                    if t_start < t_end:
                        self.availability[t_start:t_end, arrival_expert_idx] = 1

        # There is no known discrete regime sequence for ETTh1; we set all
        # entries to 0 so that plotting utilities can still show a regime
        # track without breaking.
        self.z = np.zeros(self.T, dtype=int)

        # Track last time index queried via get_context so that
        # expert_predict can (optionally) depend only on the current
        # context x_t while still supporting interfaces shared with the
        # synthetic environment.
        self._last_t: Optional[int] = None

    # ------------------------------------------------------------------
    # Interface methods (matching SyntheticTimeSeriesEnv)
    # ------------------------------------------------------------------

    def get_context(self, t: int) -> np.ndarray:
        self._last_t = int(t)
        return np.array([self.x[t]], dtype=float)

    def expert_predict(self, j: int, x_t: np.ndarray) -> float:
        j_int = int(j)

        # Strong multi-lag baseline for expert 4 (if configured):
        # y_hat ≈ b + Σ_m w_m * y_{t - lag_m}, where the lags typically
        # include 1, 24, and 168 hours of ETTh1 history. This expert can
        # leverage richer temporal structure than the others and is
        # intended to act as the strongest constant baseline.
        if (
            j_int == 4
            and self._expert4_ar_params is not None
            and self._expert4_lags is not None
            and self._last_t is not None
        ):
            w_lags, b_ar = self._expert4_ar_params
            t_idx = int(self._last_t)
            if t_idx >= int(self._expert4_lags.max()):
                # Build the multi-lag feature vector from past y-values.
                vals = []
                for lag in self._expert4_lags:
                    idx = t_idx - int(lag)
                    if idx < 0:
                        idx = 0
                    vals.append(self.y[idx])
                vals_arr = np.asarray(vals, dtype=float)
                return float(np.dot(w_lags, vals_arr) + b_ar)
            # For very early times where some lags are unavailable, fall
            # back to a simple linear-in-x prediction.
            x_val = float(x_t[0])
            w_simple = float(self.expert_weights[j_int])
            b_simple = float(self.expert_biases[j_int])
            return float(w_simple * x_val + b_simple)

        # Neural-network experts (indices >= 2 when configured).
        if hasattr(self, "_nn_expert_ids") and j_int in getattr(
            self, "_nn_expert_ids", []
        ):
            params = self._nn_params[j_int]
            if params is None:
                # Fallback: linear prediction if NN params missing.
                w = self.expert_weights[j_int]
                b = self.expert_biases[j_int]
                return float(w * float(x_t[0]) + b)
            W1, b1, W2, b2 = params
            # Single-sample forward pass consistent with the training
            # shapes used in _train_nn_expert, where W1 has shape
            # (hidden_dim, 1) and W2 has shape (hidden_dim, 1).
            x_val = float(x_t[0])
            x_arr = np.array([[x_val]], dtype=float)  # shape (1, 1)
            z1 = x_arr @ W1.T + b1  # shape (1, hidden_dim)
            h = np.tanh(z1)[0]      # shape (hidden_dim,)
            y_hat = float(h @ W2.reshape(-1) + b2)
            return y_hat

        # Default linear experts (0 and 1, or all experts if num_experts <= 2).
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
