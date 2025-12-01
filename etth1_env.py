import os
from typing import List, Optional, Sequence, Tuple

import numpy as np


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

        # Experts: simple linear predictors y_hat = w_j x + b_j.
        # We choose a small set of distinct behaviours:
        #   - Expert 0: persistence        (w=1.0, b=0.0)
        #   - Expert 1: slightly damped    (w=0.9, b=0.0)
        #   - Expert 2: slightly amplified (w=1.1, b=0.0)
        base_weights = np.array([1.0, 0.9, 1.1], dtype=float)
        base_biases = np.array([0.0, 0.0, 0.0], dtype=float)

        rng = np.random.default_rng(seed)
        if self.num_experts <= 3:
            self.expert_weights = base_weights[: self.num_experts].copy()
            self.expert_biases = base_biases[: self.num_experts].copy()
        else:
            extra = self.num_experts - 3
            extra_w = rng.normal(loc=1.0, scale=0.05, size=extra)
            extra_b = rng.normal(loc=0.0, scale=0.5, size=extra)
            self.expert_weights = np.concatenate([base_weights, extra_w])
            self.expert_biases = np.concatenate([base_biases, extra_b])

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

    # ------------------------------------------------------------------
    # Interface methods (matching SyntheticTimeSeriesEnv)
    # ------------------------------------------------------------------

    def get_context(self, t: int) -> np.ndarray:
        return np.array([self.x[t]], dtype=float)

    def expert_predict(self, j: int, x_t: np.ndarray) -> float:
        w = self.expert_weights[int(j)]
        b = self.expert_biases[int(j)]
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

