import os
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Fixed seed for data/expert generation to ensure reproducibility of time
# series and expert behavior across experiments. The configurable seed in the
# config should only affect learning processes, not the underlying data or
# expert parameters. A separate data_seed can be supplied explicitly if needed.
DATA_GENERATION_SEED = 42

try:  # optional dependency for PyTorch-based experts
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
except Exception:  # pragma: no cover - optional dependency
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
    - Context x_t can include multiple covariates (e.g., other sensor
      channels), lagged target values, and optional time features.
      When no extra context is configured, it falls back to the legacy
      single-lag setup: x_t = y_{t-1}, with x_0 = 0.
    - Experts are simple linear predictors of y_t based on x_t:
          y_hat^{(j)}_t = w_j^T x_t + b_j
    - Loss is squared error between y_t and the selected expert's
      prediction.

    This class is designed so that existing router code (SLDSIMMRouter,
    SLDSIMMRouter_Corr, L2D baselines, plotting utilities) can be reused
    without modification.
    """

    def __init__(
        self,
        csv_path: str = "data/ETTh1.csv",
        target_column: str = "OT",
        num_experts: int = 3,
        num_regimes: int = 2,
        T: Optional[int] = None,
        seed: int = 0,
        data_seed: int | None = None,
        unavailable_expert_idx: Optional[int] = None,
        unavailable_intervals: Optional[List[Tuple[int, int]]] = None,
        arrival_expert_idx: Optional[List[int]] = None,
        arrival_intervals: Optional[List[Tuple[int, int]]] = None,
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
        expert_archs: Optional[Sequence[str]] = None,
        nn_expert_type: Optional[str] = None,
        rnn_hidden_dim: int = 8,
        rnn_spectral_radius: float = 0.9,
        rnn_ridge: float = 1e-3,
        rnn_washout: int = 5,
        rnn_input_scale: float = 0.5,
        rnn_share_reservoir: bool = True,
        rnn_hidden_dims: Optional[Sequence[int]] = None,
        rnn_spectral_radii: Optional[Sequence[float]] = None,
        rnn_ridges: Optional[Sequence[float]] = None,
        rnn_washouts: Optional[Sequence[int]] = None,
        rnn_input_scales: Optional[Sequence[float]] = None,
        expert_pred_noise_std: Optional[Sequence[float]] = None,
        expert_train_ranges: Optional[Sequence[Optional[Sequence[int]]]] = None,
        expert_train_date_ranges: Optional[Sequence[Optional[Sequence[str]]]] = None,
        arima_lags: Optional[Sequence[int]] = None,
        arima_diff_order: int = 1,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ETTh1 CSV not found at path: {csv_path}")

        # Use fixed seed for data/expert generation so that time series and
        # expert behavior are reproducible regardless of the configurable seed.
        # The `seed` argument is reserved for learning processes and does not
        # affect data generation here.
        data_seed = DATA_GENERATION_SEED if data_seed is None else int(data_seed)
        self.data_seed = int(data_seed)

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
        self.num_experts = int(num_experts)
        self.num_regimes = int(num_regimes)
        self.target_column = str(target_column)
        self._expert_pred_noise_std: Optional[np.ndarray] = None
        self._expert_pred_noise: Optional[np.ndarray] = None
        if expert_pred_noise_std is not None:
            std_arr = np.asarray(expert_pred_noise_std, dtype=float)
            if std_arr.shape == ():
                std_arr = np.full(self.num_experts, float(std_arr), dtype=float)
            if std_arr.shape != (self.num_experts,):
                raise ValueError(
                    "expert_pred_noise_std must be scalar or have length num_experts."
                )
            std_arr = np.clip(std_arr, 0.0, None)
            self._expert_pred_noise_std = std_arr
            if np.any(std_arr > 0.0):
                rng_noise = np.random.default_rng(int(self.data_seed) + 12345)
                noise = rng_noise.normal(
                    loc=0.0, scale=1.0, size=(self.T, self.num_experts)
                )
                noise *= std_arr.reshape(1, -1)
                self._expert_pred_noise = noise

        # True target series
        self.y = y_arr.copy()

        # Build context features (covariates, lags, and optional time features).
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

        context_lags_local: List[int] = []
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
        need_dates = bool(include_time_features_local or expert_train_date_ranges is not None)
        parsed_dates = None
        if need_dates:
            if not date_vals:
                raise ValueError(
                    "Date-based features or expert date ranges requested but no 'date' column is available."
                )

            def _parse_dt(raw: str) -> datetime:
                try:
                    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return datetime.fromisoformat(raw)

            parsed_dates = [_parse_dt(val) for val in date_vals]

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
            if parsed_dates is None:
                raise ValueError(
                    "Time features requested but parsed dates are unavailable."
                )
            hours = np.array([dt.hour for dt in parsed_dates], dtype=float)
            dows = np.array([dt.weekday() for dt in parsed_dates], dtype=float)
            months = np.array([dt.month - 1 for dt in parsed_dates], dtype=float)
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
        expansions = [] if feature_expansions is None else [str(e).lower() for e in feature_expansions]
        for exp in expansions:
            if exp in ("squared", "square", "sq"):
                expansion_blocks.append(x_base ** 2)
                expansion_names.extend([f"{name}_sq" for name in base_names])
            elif exp in ("diff1", "delta1", "d1"):
                diff = np.zeros_like(x_base)
                if self.T > 1:
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
        if normalize_context:
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
        d_ctx = int(self.x.shape[1])
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

        # Expert 0: slightly misspecified AR(1) baseline (2% slope
        # perturbation) so that it is close to, but not equal to, the
        # least-squares optimum.
        # Expert 1: more strongly perturbed slope (5% larger
        # magnitude), sharing the same intercept. Since (w_lin, b_lin)
        # is the least-squares optimum, these perturbations make
        # experts 0 and 1 progressively worse in terms of average MSE
        # while keeping them highly correlated.
        w0 = w_lin * 0.92
        w1 = w_lin * 1.05
        base_weights = np.vstack([w0, w1]).astype(float)
        base_biases = np.array([b_lin, b_lin], dtype=float)

        rng = np.random.default_rng(self.data_seed)

        # Default: initialize linear weights/biases for all experts;
        # NN-based experts will override these via learned parameters.
        self.expert_weights = np.zeros((self.num_experts, d_ctx), dtype=float)
        self.expert_biases = np.zeros(self.num_experts, dtype=float)
        # Linear, correlated experts 0 and 1.
        n_lin = min(2, self.num_experts)
        self.expert_weights[:n_lin] = base_weights[:n_lin]
        self.expert_biases[:n_lin] = base_biases[:n_lin]

        # Neural-network experts: indices >= 2 (if any).
        self._nn_params = [None] * self.num_experts  # type: ignore[var-annotated]
        self._nn_expert_ids: List[int] = []
        # RNN-style experts (Echo State Networks).
        self._rnn_params = [None] * self.num_experts  # type: ignore[var-annotated]
        self._rnn_preds = [None] * self.num_experts  # type: ignore[var-annotated]
        self._rnn_expert_ids: List[int] = []

        # Placeholder for expert-4 multi-lag baseline parameters:
        #   - _expert4_lags: array of positive integer lags (in steps),
        #   - _expert4_ar_params: (weights_for_lags, intercept).
        self._expert4_lags: Optional[np.ndarray] = None
        self._expert4_ar_params: Optional[Tuple[np.ndarray, float]] = None

        expert_archs_local: Optional[List[str]] = None
        if expert_archs is not None:
            expert_archs_local = [str(a).lower() for a in expert_archs]
            if len(expert_archs_local) != self.num_experts:
                raise ValueError(
                    f"expert_archs must have length {self.num_experts}, "
                    f"got {len(expert_archs_local)}."
                )

        nn_type_local = "mlp"
        if nn_expert_type is not None:
            nn_type_local = str(nn_expert_type).lower()

        def _arch_for_expert(j: int) -> str:
            if expert_archs_local is not None:
                return expert_archs_local[j]
            if j in (2, 3) and self.num_experts > 2:
                return nn_type_local
            return "ar"

        def _resolve_expert_param(
            value: Optional[Sequence],
            default_scalar,
            cast_fn,
            name: str,
        ) -> List:
            if value is None:
                return [cast_fn(default_scalar) for _ in range(self.num_experts)]
            arr = np.asarray(value)
            if arr.shape == ():
                return [cast_fn(arr.item()) for _ in range(self.num_experts)]
            if arr.shape != (self.num_experts,):
                raise ValueError(
                    f"{name} must be scalar or have length num_experts ({self.num_experts})."
                )
            return [cast_fn(v) for v in arr]

        rnn_hidden_dims_local = _resolve_expert_param(
            rnn_hidden_dims, rnn_hidden_dim, int, "rnn_hidden_dims"
        )
        rnn_spectral_radii_local = _resolve_expert_param(
            rnn_spectral_radii, rnn_spectral_radius, float, "rnn_spectral_radii"
        )
        rnn_ridges_local = _resolve_expert_param(
            rnn_ridges, rnn_ridge, float, "rnn_ridges"
        )
        rnn_washouts_local = _resolve_expert_param(
            rnn_washouts, rnn_washout, int, "rnn_washouts"
        )
        rnn_input_scales_local = _resolve_expert_param(
            rnn_input_scales, rnn_input_scale, float, "rnn_input_scales"
        )

        explicit_train_masks: Optional[list[Optional[np.ndarray]]] = None
        if expert_train_ranges is not None or expert_train_date_ranges is not None:
            if expert_train_ranges is not None and len(expert_train_ranges) != self.num_experts:
                raise ValueError(
                    "expert_train_ranges must have length num_experts when provided."
                )
            if expert_train_date_ranges is not None and len(expert_train_date_ranges) != self.num_experts:
                raise ValueError(
                    "expert_train_date_ranges must have length num_experts when provided."
                )
            if expert_train_date_ranges is not None and parsed_dates is None:
                raise ValueError(
                    "expert_train_date_ranges provided but dates are unavailable."
                )
            dates_arr = None if parsed_dates is None else np.asarray(parsed_dates)
            explicit_train_masks = []
            idx_all_full = np.arange(1, self.T, dtype=int)
            for j in range(self.num_experts):
                mask = None
                has_range = False
                if expert_train_date_ranges is not None:
                    dr = expert_train_date_ranges[j]
                    if dr is not None:
                        has_range = True
                        if len(dr) != 2:
                            raise ValueError(
                                "expert_train_date_ranges entries must be [start, end]."
                            )
                        start_raw, end_raw = dr
                        start_dt = None if start_raw in (None, "") else datetime.fromisoformat(str(start_raw))
                        end_dt = None if end_raw in (None, "") else datetime.fromisoformat(str(end_raw))
                        if dates_arr is None:
                            raise ValueError("Date ranges require parsed dates.")
                        if mask is None:
                            mask = np.ones(idx_all_full.shape[0], dtype=bool)
                        date_vals_local = dates_arr[idx_all_full]
                        if start_dt is not None:
                            mask &= date_vals_local >= start_dt
                        if end_dt is not None:
                            mask &= date_vals_local <= end_dt
                if expert_train_ranges is not None:
                    rr = expert_train_ranges[j]
                    if rr is not None:
                        has_range = True
                        if len(rr) != 2:
                            raise ValueError(
                                "expert_train_ranges entries must be [start, end]."
                            )
                        start_idx = int(rr[0]) if rr[0] is not None else 1
                        end_idx = int(rr[1]) if rr[1] is not None else (self.T - 1)
                        if mask is None:
                            mask = np.ones(idx_all_full.shape[0], dtype=bool)
                        mask &= idx_all_full >= start_idx
                        mask &= idx_all_full <= end_idx
                explicit_train_masks.append(mask if has_range else None)

        self._arima_params = [None] * self.num_experts  # type: ignore[var-annotated]
        self._arima_expert_ids: List[int] = []
        arima_lags_local: List[int] = []
        if arima_lags is None:
            arima_lags_local = [1, 5, 20]
        else:
            for lag in arima_lags:
                lag_int = int(lag)
                if lag_int > 0:
                    arima_lags_local.append(lag_int)
        arima_lags_local = sorted(set(arima_lags_local))
        arima_diff_order_local = int(arima_diff_order)
        if arima_diff_order_local not in (0, 1):
            raise ValueError("arima_diff_order must be 0 or 1.")

        # If expert 4 is present, fit a strong multi-lag AR baseline on
        # the full history using several ETTh1-relevant lags (1, 24, 168
        # hours). This gives expert 4 access to richer temporal
        # structure than the other experts, making it a very strong
        # global baseline.
        if self.num_experts > 4 and _arch_for_expert(4) in ("ar", "linear", "ar_multi"):
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
                    # on the lag-1 coefficient in expert_weights/biases when
                    # that lag is present in the context.
                    if w_lags.size > 0:
                        lag1_name = f"{self.target_column}_lag{int(candidate_lags[0])}"
                        if lag1_name in self.context_feature_names:
                            idx = self.context_feature_names.index(lag1_name)
                            self.expert_weights[4, idx] = float(w_lags[0])
                        self.expert_biases[4] = b_ols

        use_mlp = any(_arch_for_expert(j) == "mlp" for j in range(self.num_experts))
        use_rnn = any(_arch_for_expert(j) == "rnn" for j in range(self.num_experts))
        use_arima = any(_arch_for_expert(j) == "arima" for j in range(self.num_experts))

        if self.T > 1:
            idx_all = np.arange(1, self.T, dtype=int)
            x_all = self.x[idx_all]
            y_all = self.y[idx_all]
            n_all = idx_all.shape[0]
        else:
            idx_all = np.zeros(0, dtype=int)
            x_all = np.zeros((0, self.x.shape[1]), dtype=float)
            y_all = np.zeros(0, dtype=float)
            n_all = 0

        if n_all >= 3:
            third = n_all // 3
        else:
            third = max(1, n_all // 2)
        two_third = min(2 * third, n_all)

        seg_early = slice(0, two_third)
        seg_late = slice(third, n_all)

        def _training_mask_for_expert(j: int) -> np.ndarray:
            if explicit_train_masks is not None and explicit_train_masks[j] is not None:
                return explicit_train_masks[j]
            mask = np.zeros(n_all, dtype=bool)
            if j == 2:
                mask[seg_early] = True
                return mask
            if j == 3:
                mask[seg_late] = True
                return mask
            mask[:] = True
            return mask

        def _fit_linear_for_expert(train_mask: Optional[np.ndarray]) -> Optional[tuple[np.ndarray, float]]:
            if n_all <= 0:
                return None
            if train_mask is None or not np.any(train_mask):
                mask = np.ones(n_all, dtype=bool)
            else:
                mask = train_mask
            if np.sum(mask) < 2:
                return None
            X = np.column_stack([np.ones(int(np.sum(mask)), dtype=float), x_all[mask]])
            y_vec = y_all[mask]
            lam = 1e-3
            XtX = X.T @ X
            diag_idx = np.arange(XtX.shape[0])
            XtX[diag_idx, diag_idx] += lam
            XtY = X.T @ y_vec
            coeffs = np.linalg.solve(XtX, XtY)
            b = float(coeffs[0])
            w = coeffs[1:].astype(float)
            return w, b

        for j in range(self.num_experts):
            if _arch_for_expert(j) not in ("ar", "linear"):
                continue
            if explicit_train_masks is None and j < 2:
                continue
            params = _fit_linear_for_expert(_training_mask_for_expert(j))
            if params is None:
                continue
            w, b = params
            if w.shape == self.expert_weights[j].shape:
                self.expert_weights[j] = w
                self.expert_biases[j] = b

        if use_arima:
            def _fit_arima_for_expert(train_mask: np.ndarray) -> Optional[dict]:
                if n_all <= 0 or not arima_lags_local:
                    return None
                max_lag = max(arima_lags_local) + arima_diff_order_local
                rows = []
                targets = []
                y_mean = 0.0
                if train_mask is not None and train_mask.any():
                    y_mean = float(np.mean(self.y[idx_all[train_mask]]))
                elif self.y.size > 0:
                    y_mean = float(np.mean(self.y))
                for idx_pos, t in enumerate(idx_all):
                    if not train_mask[idx_pos]:
                        continue
                    if t <= max_lag:
                        continue
                    if arima_diff_order_local == 0:
                        feats = [self.y[t - lag] for lag in arima_lags_local]
                        target = self.y[t]
                    else:
                        feats = [
                            self.y[t - lag] - self.y[t - lag - 1]
                            for lag in arima_lags_local
                        ]
                        target = self.y[t] - self.y[t - 1]
                    rows.append(feats)
                    targets.append(float(target))
                if not rows:
                    return {"lags": arima_lags_local, "d": arima_diff_order_local, "w": None, "b": y_mean, "mean": y_mean}
                X = np.column_stack([np.ones(len(rows), dtype=float), np.asarray(rows, dtype=float)])
                y_vec = np.asarray(targets, dtype=float)
                lam = 1e-3
                XtX = X.T @ X
                diag_idx = np.arange(XtX.shape[0])
                XtX[diag_idx, diag_idx] += lam
                XtY = X.T @ y_vec
                coeffs = np.linalg.solve(XtX, XtY)
                b = float(coeffs[0])
                w = coeffs[1:].astype(float)
                return {"lags": arima_lags_local, "d": arima_diff_order_local, "w": w, "b": b, "mean": y_mean}

            for j in range(self.num_experts):
                if _arch_for_expert(j) != "arima":
                    continue
                train_mask = _training_mask_for_expert(j)
                params = _fit_arima_for_expert(train_mask)
                if params is not None:
                    self._arima_params[j] = params
                    self._arima_expert_ids.append(j)

        x_val_global = np.zeros((0, x_all.shape[1]), dtype=float)
        y_val_global = np.zeros((0,), dtype=float)
        use_global_val = self.num_experts > 2 and (use_mlp or use_rnn)
        if use_global_val and explicit_train_masks is None:
            # Global validation set used for checkpoint selection.
            if n_all >= 5:
                n_val_global = max(1, int(0.2 * n_all))
            else:
                n_val_global = 1
            if n_val_global >= n_all:
                n_val_global = 1
            idx_val_global = np.arange(n_all - n_val_global, n_all, dtype=int)
            x_val_global = x_all[idx_val_global]
            y_val_global = y_all[idx_val_global]

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
                in_dim = int(x_train.shape[1])
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1)

                N_total = x_train.shape[0]
                if N_total == 0:
                    # Fallback: degenerate zero network.
                    W1 = np.zeros((hidden_dim, in_dim), dtype=float)
                    b1 = np.zeros(hidden_dim, dtype=float)
                    W2 = np.zeros((hidden_dim, 1), dtype=float)
                    b2 = 0.0
                    return W1, b1, W2, b2

                # Global validation set (shared across experts)
                if x_val_global.ndim == 1:
                    x_val = x_val_global.reshape(-1, 1)
                else:
                    x_val = x_val_global.reshape(-1, in_dim)
                if y_val_global.ndim == 1:
                    y_val = y_val_global.reshape(-1, 1)
                else:
                    y_val = y_val_global.reshape(-1, 1)

                # Prefer PyTorch training when available.
                if torch is not None:
                    # Construct a simple 1-hidden-layer MLP with tanh.
                    device = _select_torch_device() or torch.device("cpu")

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

                    model = _MLP(in_dim=in_dim, hid_dim=hidden_dim).to(device)
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

                    W1_t = best_state["lin1_weight"]  # shape (hidden_dim, in_dim)
                    b1_t = best_state["lin1_bias"]    # shape (hidden_dim,)
                    W2_t = best_state["lin2_weight"].T  # convert (1, hidden_dim) -> (hidden_dim, 1)
                    b2_t = float(best_state["lin2_bias"].reshape(()))

                    W1 = np.asarray(W1_t, dtype=float)
                    b1 = np.asarray(b1_t, dtype=float)
                    W2 = np.asarray(W2_t, dtype=float)
                    b2 = float(b2_t)
                    return W1, b1, W2, b2

                # Fallback: original NumPy-based training if torch is not available.
                W1 = rng_local.normal(loc=0.0, scale=0.1, size=(hidden_dim, in_dim))
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

            def _train_esn_expert(
                x_train: np.ndarray,
                y_train: np.ndarray,
                x_full: np.ndarray,
                rng_local: np.random.Generator,
                hidden_dim: int,
                spectral_radius: float,
                ridge: float,
                washout: int,
                input_scale: float,
                shared_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
                if x_train.ndim == 1:
                    x_train = x_train.reshape(-1, 1)
                if x_full.ndim == 1:
                    x_full = x_full.reshape(-1, 1)
                in_dim = int(x_train.shape[1])
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1)
                if x_train.shape[0] == 0:
                    return None

                if shared_weights is None:
                    W_in = rng_local.normal(loc=0.0, scale=1.0, size=(hidden_dim, in_dim))
                    W_in *= float(input_scale)
                    W_h = rng_local.normal(loc=0.0, scale=1.0, size=(hidden_dim, hidden_dim))
                    if spectral_radius > 0:
                        eigs = np.linalg.eigvals(W_h)
                        rad = float(np.max(np.abs(eigs)))
                        if rad > 0:
                            W_h *= float(spectral_radius) / rad
                else:
                    W_in, W_h = shared_weights

                h = np.zeros(hidden_dim, dtype=float)
                states = []
                targets = []
                for t in range(x_train.shape[0]):
                    h = np.tanh(W_in @ x_train[t] + W_h @ h)
                    if t >= int(washout):
                        feats = np.concatenate([h, x_train[t], np.array([1.0])])
                        states.append(feats)
                        targets.append(float(y_train[t]))

                if not states:
                    return None
                H = np.vstack(states)
                y_vec = np.asarray(targets, dtype=float).reshape(-1, 1)
                XtX = H.T @ H
                XtX += float(ridge) * np.eye(XtX.shape[0])
                XtY = H.T @ y_vec
                W_out = np.linalg.solve(XtX, XtY).reshape(-1)

                preds = np.zeros(x_full.shape[0], dtype=float)
                h = np.zeros(hidden_dim, dtype=float)
                for t in range(x_full.shape[0]):
                    h = np.tanh(W_in @ x_full[t] + W_h @ h)
                    feats = np.concatenate([h, x_full[t], np.array([1.0])])
                    preds[t] = float(feats @ W_out)

                return W_in, W_h, W_out, preds

            shared_reservoir = None
            if use_rnn and bool(rnn_share_reservoir):
                rnn_ids = [j for j in range(self.num_experts) if _arch_for_expert(j) == "rnn"]
                if rnn_ids:
                    base_dim = rnn_hidden_dims_local[rnn_ids[0]]
                    base_scale = rnn_input_scales_local[rnn_ids[0]]
                    base_radius = rnn_spectral_radii_local[rnn_ids[0]]
                    for j in rnn_ids[1:]:
                        if (
                            rnn_hidden_dims_local[j] != base_dim
                            or rnn_input_scales_local[j] != base_scale
                            or rnn_spectral_radii_local[j] != base_radius
                        ):
                            raise ValueError(
                                "rnn_share_reservoir requires identical rnn_hidden_dims, "
                                "rnn_input_scales, and rnn_spectral_radii across RNN experts."
                            )
                    shared_reservoir = (
                        rng.normal(
                            loc=0.0,
                            scale=1.0,
                            size=(int(base_dim), int(self.x.shape[1])),
                        )
                        * float(base_scale),
                        rng.normal(
                            loc=0.0,
                            scale=1.0,
                            size=(int(base_dim), int(base_dim)),
                        ),
                    )
                    if base_radius > 0:
                        eigs = np.linalg.eigvals(shared_reservoir[1])
                        rad = float(np.max(np.abs(eigs)))
                        if rad > 0:
                            shared_reservoir = (
                                shared_reservoir[0],
                                shared_reservoir[1] * float(base_radius) / rad,
                            )

            for j in range(self.num_experts):
                arch = _arch_for_expert(j)
                if arch == "mlp":
                    mask = _training_mask_for_expert(j)
                    x_val_local = x_val_global
                    y_val_local = y_val_global
                    if explicit_train_masks is not None and mask is not None:
                        idx_masked = np.where(mask)[0]
                        if idx_masked.size >= 5:
                            n_val = max(1, int(0.2 * idx_masked.size))
                        else:
                            n_val = 0 if idx_masked.size <= 1 else 1
                        if n_val > 0:
                            idx_val = idx_masked[-n_val:]
                            x_val_local = x_all[idx_val]
                            y_val_local = y_all[idx_val]
                        else:
                            x_val_local = np.zeros((0, x_all.shape[1]), dtype=float)
                            y_val_local = np.zeros((0,), dtype=float)
                    W1_j, b1_j, W2_j, b2_j = _train_nn_expert(
                        x_all[mask],
                        y_all[mask],
                        x_val_local,
                        y_val_local,
                        hidden_dim=8,
                        rng_local=rng,
                    )
                    self._nn_params[j] = (W1_j, b1_j, W2_j, b2_j)
                    self._nn_expert_ids.append(j)
                elif arch == "rnn":
                    mask = _training_mask_for_expert(j)
                    result = _train_esn_expert(
                        x_all[mask],
                        y_all[mask],
                        x_all,
                        rng_local=rng,
                        hidden_dim=int(rnn_hidden_dims_local[j]),
                        spectral_radius=float(rnn_spectral_radii_local[j]),
                        ridge=float(rnn_ridges_local[j]),
                        washout=int(rnn_washouts_local[j]),
                        input_scale=float(rnn_input_scales_local[j]),
                        shared_weights=shared_reservoir,
                    )
                    if result is not None:
                        self._rnn_params[j] = result[:3]
                        self._rnn_preds[j] = result[3]
                        self._rnn_expert_ids.append(j)


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
        # if (
        #     arrival_expert_idx is not None
        # ):
        #     for i in range(len(arrival_expert_idx)):
        #         self.availability[:, arrival_expert_idx[i]] = 0
        #         if arrival_intervals[i] is not None:
        #             for start, end in arrival_intervals[i]:
        #                 t_start = max(int(start), 0)
        #                 t_end = min(int(end) + 1, self.T)
        #                 if t_start >= self.T or t_end <= 0:
        #                     continue
        #                 if t_start < t_end:
        #                     self.availability[t_start:t_end, arrival_expert_idx[i]] = 1

        if arrival_expert_idx is not None:
            if arrival_intervals is None:
                arrival_intervals_local = [None] * len(arrival_expert_idx)
            else:
                arrival_intervals_local = list(arrival_intervals)
                if len(arrival_intervals_local) != len(arrival_expert_idx):
                    raise ValueError(
                        "arrival_intervals must match arrival_expert_idx in length."
                    )

            for i, expert_idx in enumerate(arrival_expert_idx):
                self.availability[:, expert_idx] = 0
                raw_intervals = arrival_intervals_local[i]
                if raw_intervals is None:
                    continue
                if isinstance(raw_intervals, (tuple, list)) and len(raw_intervals) == 2 and not isinstance(
                        raw_intervals[0], (tuple, list)
                ):
                    intervals_iter = [raw_intervals]
                else:
                    intervals_iter = raw_intervals

                for start, end in intervals_iter:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, self.T)
                    if t_start < t_end:
                        self.availability[t_start:t_end, expert_idx] = 1

        # print(self.availability[5000])
        # print(self.availability[5001])

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
        return np.asarray(self.x[t], dtype=float)

    def expert_predict(self, j: int, x_t: np.ndarray) -> float:
        j_int = int(j)
        x_vec_full = np.asarray(x_t, dtype=float).reshape(-1)

        def _apply_pred_noise(y_hat: float) -> float:
            if self._expert_pred_noise is None or self._last_t is None:
                return float(y_hat)
            t_idx = int(self._last_t)
            t_idx = max(0, min(t_idx, self.T - 1))
            try:
                x_ref = np.asarray(self.x[t_idx], dtype=float).reshape(-1)
                if x_ref.shape == x_vec_full.shape and not np.allclose(x_ref, x_vec_full):
                    return float(y_hat)
            except Exception:
                return float(y_hat)
            return float(y_hat + self._expert_pred_noise[t_idx, j_int])

        if (
            hasattr(self, "_arima_expert_ids")
            and j_int in getattr(self, "_arima_expert_ids", [])
            and self._last_t is not None
        ):
            params = self._arima_params[j_int]
            if params is not None:
                lags = params.get("lags", [])
                diff_order = int(params.get("d", 0))
                w = params.get("w", None)
                b = float(params.get("b", 0.0))
                mean_fallback = float(params.get("mean", 0.0))
                t_idx = int(self._last_t)
                max_lag = max(lags) + diff_order if lags else diff_order
                if t_idx <= max_lag or w is None:
                    return _apply_pred_noise(mean_fallback)
                if diff_order == 0:
                    feats = np.array([self.y[t_idx - lag] for lag in lags], dtype=float)
                    pred = float(b + np.dot(w, feats))
                else:
                    feats = np.array(
                        [self.y[t_idx - lag] - self.y[t_idx - lag - 1] for lag in lags],
                        dtype=float,
                    )
                    diff_pred = float(b + np.dot(w, feats))
                    pred = float(self.y[t_idx - 1] + diff_pred)
                return _apply_pred_noise(pred)

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
                return _apply_pred_noise(float(np.dot(w_lags, vals_arr) + b_ar))
            # For very early times where some lags are unavailable, fall
            # back to a simple linear-in-x prediction.
            w_simple = np.asarray(self.expert_weights[j_int], dtype=float).reshape(-1)
            b_simple = float(self.expert_biases[j_int])
            return _apply_pred_noise(float(np.dot(w_simple, x_vec_full) + b_simple))

        # Neural-network experts (indices >= 2 when configured).
        if hasattr(self, "_nn_expert_ids") and j_int in getattr(
            self, "_nn_expert_ids", []
        ):
            params = self._nn_params[j_int]
            if params is None:
                # Fallback: linear prediction if NN params missing.
                w = np.asarray(self.expert_weights[j_int], dtype=float).reshape(-1)
                b = float(self.expert_biases[j_int])
                return _apply_pred_noise(float(np.dot(w, x_vec_full) + b))
            W1, b1, W2, b2 = params
            # Single-sample forward pass consistent with the training
            # shapes used in _train_nn_expert, where W1 has shape
            # (hidden_dim, d_ctx) and W2 has shape (hidden_dim, 1).
            x_vec = x_vec_full.reshape(1, -1)
            z1 = x_vec @ W1.T + b1  # shape (1, hidden_dim)
            h = np.tanh(z1)[0]      # shape (hidden_dim,)
            y_hat = float(h @ W2.reshape(-1) + b2)
            return _apply_pred_noise(y_hat)

        # RNN-style experts (precomputed predictions).
        if (
            hasattr(self, "_rnn_expert_ids")
            and j_int in getattr(self, "_rnn_expert_ids", [])
            and self._last_t is not None
        ):
            preds = self._rnn_preds[j_int]
            if preds is not None:
                t_idx = int(self._last_t)
                t_idx = max(0, min(t_idx, preds.shape[0] - 1))
                return _apply_pred_noise(float(preds[t_idx]))

        # Default linear experts (0 and 1, or all experts if num_experts <= 2).
        w = np.asarray(self.expert_weights[j_int], dtype=float).reshape(-1)
        b = float(self.expert_biases[j_int])
        return _apply_pred_noise(float(np.dot(w, x_vec_full) + b))

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
