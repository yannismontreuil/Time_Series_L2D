import os
from datetime import datetime
from typing import List, Optional, Tuple, Sequence

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
      * AR(2) models trained on data segments (banned for fairness)
    - Loss is squared error between y_t and the selected expert's prediction.

    Extensions:
    - Context x_t can include multiple covariates (e.g., other sensor
      channels), lagged target values, and optional time features.
      When no extra context is configured, it falls back to the legacy
      single-lag setup: x_t = y_{t-1}, with x_0 = 0.
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
        use_rich_context: bool = False,
        context_columns: Optional[Sequence[str]] = None,
        context_lags: Optional[Sequence[int]] = None,
        include_time_features: bool = False,
        time_features: Optional[Sequence[str]] = None,
        normalize_context: bool = False,
        normalization_window: Optional[int] = None,
        normalization_eps: float = 1e-6,
        normalization_mode: str = "zscore",
        feature_expansions: Optional[Sequence[str]] = None,
        lag_diff_pairs: Optional[Sequence[Sequence[int]]] = None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ETTh1 CSV not found at path: {csv_path}")

        # Load numeric columns from CSV (simple parser; no external deps).
        y_all: List[float] = []
        col_data: dict[str, List[float]] = {}
        date_vals: List[str] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                raise ValueError("Empty ETTh1 CSV file.")
            header = [h.strip() for h in header_line.strip().split(",")]
            if target_column not in header:
                raise ValueError(
                    f"Target column '{target_column}' not found in header "
                    f"{header} of {csv_path}"
                )
            col_idx = {name: i for i, name in enumerate(header)}
            date_idx = col_idx.get("date", None)
            numeric_columns = [name for name in header if name != "date"]
            if target_column not in numeric_columns:
                raise ValueError(
                    f"Target column '{target_column}' must be numeric in {csv_path}."
                )
            target_num_idx = numeric_columns.index(target_column)
            for name in numeric_columns:
                col_data[name] = []

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < len(header):
                    continue
                vals: List[float] = []
                ok = True
                for name in numeric_columns:
                    idx = col_idx[name]
                    try:
                        vals.append(float(parts[idx]))
                    except ValueError:
                        ok = False
                        break
                if not ok:
                    continue
                if date_idx is not None:
                    date_vals.append(parts[date_idx])
                for name, val in zip(numeric_columns, vals):
                    col_data[name].append(val)
                y_all.append(vals[target_num_idx])

        if len(y_all) < 2:
            raise ValueError(
                "ETTh1 dataset must contain at least 2 observations for "
                "lagged context construction."
            )

        y_arr = np.asarray(y_all, dtype=float)
        col_arrs = {name: np.asarray(vals, dtype=float) for name, vals in col_data.items()}

        # Truncate or use full series according to T
        if T is None:
            T_eff = y_arr.shape[0]
        else:
            T_eff = int(T)
            T_eff = max(2, min(T_eff, y_arr.shape[0]))
            y_arr = y_arr[:T_eff]
            col_arrs = {name: arr[:T_eff] for name, arr in col_arrs.items()}
            if date_vals:
                date_vals = date_vals[:T_eff]

        self.T = T_eff
        self.target_column = str(target_column)

        # True target series
        self.y = y_arr.copy()

        # Build context features (covariates, lags, and optional time features).
        # Only use advanced features if use_rich_context is True
        context_columns_local: List[str] = []
        context_lags_local: List[int] = []
        include_time_features_local = False
        time_features_local: List[str] = []

        if use_rich_context:
            context_columns_local = (
                [] if context_columns is None else [str(c) for c in context_columns]
            )
            if context_columns_local:
                missing = [c for c in context_columns_local if c not in col_arrs]
                if missing:
                    raise ValueError(
                        f"Context columns not found in CSV: {missing}. "
                        f"Available: {sorted(col_arrs.keys())}"
                    )

            if context_lags is not None:
                for lag in context_lags:
                    lag_int = int(lag)
                    if lag_int > 0:
                        context_lags_local.append(lag_int)
                context_lags_local = sorted(set(context_lags_local))

            include_time_features_local = bool(include_time_features)
            time_features_local = (
                ["hour", "dayofweek", "month"]
                if time_features is None
                else [str(f) for f in time_features]
            )

        feature_blocks: List[np.ndarray] = []
        feature_names: List[str] = []

        if context_columns_local:
            cols = [col_arrs[name] for name in context_columns_local]
            feature_blocks.append(np.column_stack(cols))
            feature_names.extend(context_columns_local)

        if context_lags_local:
            for lag in context_lags_local:
                lag_vals = np.zeros(self.T, dtype=float)
                if lag < self.T:
                    lag_vals[lag:] = self.y[:-lag]
                feature_blocks.append(lag_vals.reshape(-1, 1))
                feature_names.append(f"{target_column}_lag{lag}")

        if include_time_features_local:
            if not date_vals:
                raise ValueError(
                    "Time features requested but no 'date' column is available."
                )

            def _parse_dt(raw: str) -> datetime:
                try:
                    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return datetime.fromisoformat(raw)

            parsed = [_parse_dt(val) for val in date_vals]
            hours = np.array([dt.hour for dt in parsed], dtype=float)
            dows = np.array([dt.weekday() for dt in parsed], dtype=float)
            months = np.array([dt.month - 1 for dt in parsed], dtype=float)
            for name in time_features_local:
                if name == "hour":
                    angles = 2.0 * np.pi * hours / 24.0
                    feature_blocks.append(np.column_stack([np.sin(angles), np.cos(angles)]))
                    feature_names.extend(["hour_sin", "hour_cos"])
                elif name in ("dayofweek", "dow", "weekday"):
                    angles = 2.0 * np.pi * dows / 7.0
                    feature_blocks.append(np.column_stack([np.sin(angles), np.cos(angles)]))
                    feature_names.extend(["dow_sin", "dow_cos"])
                elif name in ("month", "monthofyear"):
                    angles = 2.0 * np.pi * months / 12.0
                    feature_blocks.append(np.column_stack([np.sin(angles), np.cos(angles)]))
                    feature_names.extend(["month_sin", "month_cos"])
                else:
                    raise ValueError(
                        f"Unsupported time feature '{name}'. "
                        "Use hour, dayofweek, or month."
                    )

        if feature_blocks:
            x_base = np.concatenate(feature_blocks, axis=1)
            base_names = feature_names
        else:
            # Legacy fallback: x_t := y_{t-1}, with x_0 = 0.0
            x_base = np.zeros((self.T, 1), dtype=float)
            if self.T > 1:
                x_base[1:, 0] = self.y[:-1]
            base_names = [f"{target_column}_lag1"]

        expansion_blocks: List[np.ndarray] = []
        expansion_names: List[str] = []
        expansions: List[str] = []
        if use_rich_context and feature_expansions is not None:
            expansions = [str(e).lower() for e in feature_expansions]
        for exp in expansions:
            if exp in ("squared", "square", "sq"):
                expansion_blocks.append(x_base ** 2)
                expansion_names.extend([f"{name}_sq" for name in base_names])
            elif exp in ("diff1", "delta1", "d1"):
                diff = np.zeros_like(x_base)
                if self.T > 1:
                    # First entry is zero; then differences
                    diff[1:] = x_base[1:] - x_base[:-1]
                expansion_blocks.append(diff)
                expansion_names.extend([f"{name}_d1" for name in base_names])
            elif exp in ("lag_diff", "lagdiff"):
                if lag_diff_pairs is None:
                    lag_diff_pairs = [(1, 24), (1, 168), (24, 168)]
                for pair in lag_diff_pairs:
                    if len(pair) != 2:
                        continue
                    lag_a, lag_b = int(pair[0]), int(pair[1])
                    name_a = f"{target_column}_lag{lag_a}"
                    name_b = f"{target_column}_lag{lag_b}"
                    if name_a not in base_names or name_b not in base_names:
                        continue
                    idx_a = base_names.index(name_a)
                    idx_b = base_names.index(name_b)
                    vals = (x_base[:, idx_a] - x_base[:, idx_b]).reshape(-1, 1)
                    expansion_blocks.append(vals)
                    expansion_names.append(f"{target_column}_lag{lag_a}_minus_lag{lag_b}")
            else:
                raise ValueError(
                    f"Unsupported feature expansion '{exp}'. "
                    "Use squared, diff1, or lag_diff."
                )

        if expansion_blocks:
            self.x = np.concatenate([x_base] + expansion_blocks, axis=1)
            self.context_feature_names = base_names + expansion_names
        else:
            self.x = x_base
            self.context_feature_names = base_names

        # Optional context normalization to handle mixed feature scales.
        if use_rich_context and normalize_context:
            mode = str(normalization_mode).lower()
            if mode not in ("zscore", "standard"):
                raise ValueError(
                    f"Unsupported normalization_mode '{normalization_mode}'. "
                    "Use 'zscore'/'standard'."
                )
            if normalization_window is None:
                norm_window = self.T
            else:
                norm_window = int(normalization_window)
                norm_window = max(2, min(norm_window, self.T))
            ref = self.x[:norm_window]
            mean = ref.mean(axis=0)
            std = ref.std(axis=0)
            std = np.where(std < float(normalization_eps), 1.0, std)
            self.x = (self.x - mean) / std
            self.context_norm_mean = mean
            self.context_norm_std = std
            self.context_norm_window = norm_window

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
        d_ctx = int(self.x.shape[1])
        self.expert_weights = np.zeros((self.num_experts, d_ctx), dtype=float)
        self.expert_biases = np.zeros(self.num_experts, dtype=float)
        self.expert_types = {}  # Maps expert index to expert type
        self._nn_params = [None] * self.num_experts
        self._nn_expert_ids: List[int] = []
        self._ar2_params = {}  # Maps expert index to AR(2) parameters

        # Fit base AR model for reference using ridge regression
        idx_lin = np.arange(1, self.T, dtype=int)
        if idx_lin.size >= 2:
            x_lin = self.x[idx_lin]
            y_lin = self.y[idx_lin]
            # Ridge-regularized linear regression with intercept.
            X = np.column_stack([np.ones(idx_lin.size, dtype=float), x_lin])
            lam = 1e-3
            XtX = X.T @ X
            diag_idx = np.arange(XtX.shape[0])
            XtX[diag_idx, diag_idx] += lam
            XtY = X.T @ y_lin
            coeffs = np.linalg.solve(XtX, XtY)
            b_lin = float(coeffs[0])
            w_lin = coeffs[1:].astype(float)
        else:
            # Fallback for extremely short series.
            w_lin = np.zeros(d_ctx, dtype=float)
            b_lin = float(self.y.mean()) if self.y.size > 0 else 0.0

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

    def _initialize_expert(self, expert_idx: int, expert_type: str, w_base: np.ndarray, b_base: float, rng: np.random.Generator):
        """Initialize a specific expert based on its type."""
        if expert_type == "ar1_low_var":
            # AR(1) with low variance
            self.expert_weights[expert_idx] = w_base * 0.8
            self.expert_biases[expert_idx] = b_base + rng.normal(0, 0.5)  # Increased variance from 0.5 to 1.5

        elif expert_type == "ar1_high_var":
            # AR(1) with higher variance (more misspecified)
            self.expert_weights[expert_idx] = w_base * 1.30
            self.expert_biases[expert_idx] = b_base + rng.normal(0, 1)

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

        if n_all < 20:  # Increased minimum data requirement
            # Fallback for short series
            self.expert_weights[expert_idx] = 0.9
            self.expert_biases[expert_idx] = 0.0
            return

        # Define training segments - use more data for each expert
        if expert_type == "nn_early":
            # Train on first 70% of data
            end_idx = max(1, int(0.7 * n_all))
            x_train = x_all[:end_idx]
            y_train = y_all[:end_idx]
        else:  # nn_late
            # Train on last 70% of data
            start_idx = max(1, int(0.4 * n_all))
            end_idx = n_all
            x_train = x_all[start_idx:end_idx]
            y_train = y_all[start_idx:end_idx]

        # Validation set: middle 20% of full data for better representation
        val_start = max(1, int(0.4 * n_all))
        val_end = max(val_start + 1, int(0.6 * n_all))
        x_val = x_all[val_start:val_end]
        y_val = y_all[val_start:val_end]

        # Train neural network with improved architecture
        params = self._train_nn_expert([x_train], [y_train], x_val, y_val, 32, rng)  # Increased hidden dim
        self._nn_params[expert_idx] = params

    def _initialize_ar2_expert(self, expert_idx: int, expert_type: str, w_base: np.ndarray, b_base: float, rng: np.random.Generator):
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
            # Fallback to simple linear model if not enough data
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

            # For compatibility, store a linear approximation using lag-1 from context
            # Find the lag1 feature in context
            lag1_name = f"{self.target_column}_lag1"
            if lag1_name in self.context_feature_names:
                idx = self.context_feature_names.index(lag1_name)
                self.expert_weights[expert_idx, idx] = float(w1_ar2)
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
        num_epochs: int = 300,  # Increased epochs
        learning_rate: float = 5e-3,  # Lower initial learning rate
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, float]:  # Added normalization params
        """Train neural network expert with improved architecture and training."""
        # Concatenate all segments for training
        x_train_list = []
        y_train_list = []
        for x_seg, y_seg in zip(x_train_segments, y_train_segments):
            if x_seg.size > 0 and y_seg.size > 0:
                x_train_list.append(x_seg)
                y_train_list.append(y_seg)

        if not x_train_list:
            # Fallback: degenerate zero network
            in_dim = int(self.x.shape[1])
            W1 = np.zeros((hidden_dim, in_dim), dtype=float)
            b1 = np.zeros(hidden_dim, dtype=float)
            W2 = np.zeros((hidden_dim, 1), dtype=float)
            b2 = 0.0
            return W1, b1, W2, b2, np.zeros(in_dim), np.ones(in_dim), 0.0, 1.0

        x_train = np.concatenate(x_train_list)
        y_train = np.concatenate(y_train_list)

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        in_dim = int(x_train.shape[1])
        N_total = x_train.shape[0]
        if N_total == 0:
            W1 = np.zeros((hidden_dim, in_dim), dtype=float)
            b1 = np.zeros(hidden_dim, dtype=float)
            W2 = np.zeros((hidden_dim, 1), dtype=float)
            b2 = 0.0
            return W1, b1, W2, b2, np.zeros(in_dim), np.ones(in_dim), 0.0, 1.0

        # Normalize inputs for better training stability
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        x_train_norm = (x_train - x_mean) / x_std

        # Normalize outputs
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if y_std < 1e-8:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std

        # Validation data normalization
        if x_val_global.ndim == 1:
            x_val = x_val_global.reshape(-1, 1)
        else:
            x_val = x_val_global
        if y_val_global.ndim == 1:
            y_val = y_val_global.reshape(-1, 1)
        else:
            y_val = y_val_global.reshape(-1, 1)

        x_val_norm = (x_val - x_mean) / x_std
        y_val_norm = (y_val - y_mean) / y_std

        # Initialize network with Xavier initialization
        scale = np.sqrt(2.0 / (in_dim + hidden_dim))
        W1 = rng_local.normal(loc=0.0, scale=scale, size=(hidden_dim, in_dim))
        b1 = np.zeros(hidden_dim, dtype=float)
        W2 = rng_local.normal(loc=0.0, scale=scale/hidden_dim, size=(hidden_dim, 1))
        b2 = 0.0

        # Best parameters tracking
        best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2
        best_val_rmse = np.inf
        patience = 50
        no_improve_count = 0

        # Learning rate schedule
        lr = learning_rate
        lr_decay = 0.95

        for epoch in range(num_epochs):
            # Forward pass
            z1 = x_train_norm @ W1.T + b1  # (N_total, hidden_dim)
            h = np.tanh(z1)
            y_hat = h @ W2 + b2  # (N_total, 1)

            # Loss with L2 regularization
            mse_loss = float(np.mean((y_hat - y_train_norm) ** 2))
            l2_reg = 1e-4 * (np.sum(W1**2) + np.sum(W2**2))
            total_loss = mse_loss + l2_reg

            # Gradients
            diff = y_hat - y_train_norm
            d_yhat = (2.0 / N_total) * diff

            # Backprop with regularization
            d_W2 = h.T @ d_yhat + 2e-4 * W2  # L2 regularization
            d_b2 = float(d_yhat.sum())

            d_h = d_yhat @ W2.T  # (N_total, hidden_dim)
            d_z1 = d_h * (1.0 - np.tanh(z1) ** 2)
            d_W1 = d_z1.T @ x_train_norm + 2e-4 * W1  # L2 regularization
            d_b1 = d_z1.sum(axis=0)  # (hidden_dim,)

            # Gradient step with clipping
            grad_clip = 1.0
            d_W1 = np.clip(d_W1, -grad_clip, grad_clip)
            d_W2 = np.clip(d_W2, -grad_clip, grad_clip)
            d_b1 = np.clip(d_b1, -grad_clip, grad_clip)
            d_b2 = np.clip(d_b2, -grad_clip, grad_clip)

            W1 -= lr * d_W1
            W2 -= lr * d_W2
            b1 -= lr * d_b1
            b2 -= lr * d_b2

            # Validation and early stopping
            if x_val.shape[0] > 0 and epoch % 10 == 0:
                z1_val = x_val_norm @ W1.T + b1
                h_val = np.tanh(z1_val)
                y_hat_val = h_val @ W2 + b2
                mse_val = float(np.mean((y_hat_val - y_val_norm) ** 2))
                rmse_val = float(np.sqrt(mse_val))

                if rmse_val < best_val_rmse:
                    best_val_rmse = rmse_val
                    best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    break

            # Learning rate decay
            if epoch > 0 and epoch % 50 == 0:
                lr *= lr_decay

        # Store normalization parameters with the model
        return best_W1, best_b1, best_W2, best_b2, x_mean, x_std, y_mean, y_std

    # ------------------------------------------------------------------
    # Interface methods (matching SyntheticTimeSeriesEnv)
    # ------------------------------------------------------------------

    def get_context(self, t: int) -> np.ndarray:
        self._last_t = int(t)
        return np.asarray(self.x[t], dtype=float)

    def expert_predict(self, j: int, x_t: np.ndarray) -> float:
        j_int = int(j)
        x_vec_full = np.asarray(x_t, dtype=float).reshape(-1)

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

        # Neural network experts with normalization
        if j_int in self._nn_expert_ids:
            params = self._nn_params[j_int]
            if params is None or len(params) < 8:
                # Fallback: linear prediction if NN params missing
                w = np.asarray(self.expert_weights[j_int], dtype=float).reshape(-1)
                b = float(self.expert_biases[j_int])
                return float(np.dot(w, x_vec_full) + b)

            W1, b1, W2, b2, x_mean, x_std, y_mean, y_std = params
            # Normalize input
            x_norm = (x_vec_full - x_mean) / np.maximum(x_std, 1e-8)

            # Forward pass
            x_arr = x_norm.reshape(1, -1)  # shape (1, d_ctx)
            z1 = x_arr @ W1.T + b1  # shape (1, hidden_dim)
            h = np.tanh(z1)[0]      # shape (hidden_dim,)
            y_hat_norm = float(h @ W2.reshape(-1) + b2)

            # Denormalize output
            y_hat = y_hat_norm * y_std + y_mean
            rng = np.random.default_rng()
            if expert_type == "nn_early":
                return y_hat + rng.normal(0, 6)
            else:  # nn_late
                return y_hat + rng.normal(0, 6)

        # Default linear experts (ar1_low_var, ar1_high_var)
        w = np.asarray(self.expert_weights[j_int], dtype=float).reshape(-1)
        b = float(self.expert_biases[j_int])
        return float(np.dot(w, x_vec_full) + b)

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
