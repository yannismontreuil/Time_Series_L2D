import numpy as np


class SyntheticTimeSeriesEnv:
    """
    Simple synthetic environment:

      - A small number of regimes (typically 2, but can be >2 for
        experiments such as "noisy_forgetting") following a simple
        Markov chain / block structure.
      - True target y_t follows AR(1) with regime-dependent drift.
      - Context x_t is y_{t-1} (lag-1 context).
      - Experts are simple linear predictors of y_t:
            y_hat^{(j)}_t = w_j * x_t + b_j
      - Loss is squared error.

    This is deliberately simple: the router’s SLDS model does not need to
    match this true generative process exactly; it just infers from losses.
    """

    def __init__(
        self,
        num_experts: int = 3,
        num_regimes: int = 2,
        T: int = 200,
        seed: int = 0,
        unavailable_expert_idx: int | None = 1,
        unavailable_start_t: int | None = None,
        unavailable_intervals: list[tuple[int, int]] | None = None,
        arrival_expert_idx: int | None = None,
        arrival_intervals: list[tuple[int, int]] | None = None,
        setting: str = "easy_setting",
        noise_scale: float | None = None,
    ):
        rng = np.random.default_rng(seed)
        self.num_experts = num_experts
        self.num_regimes = num_regimes
        self.T = T
        self.setting = setting
        M = self.num_regimes

        # True regime transition matrix (for reference). For M=2 we keep
        # the original example; for M>2 we use a simple "sticky" chain.
        if M == 2:
            self.Pi_true = np.array(
                [[0.95, 0.05],
                 [0.10, 0.90]],
                dtype=float,
            )
        else:
            self.Pi_true = np.full((M, M), 0.0, dtype=float)
            for k in range(M):
                self.Pi_true[k, :] = 0.05 / max(M - 1, 1)
                self.Pi_true[k, k] = 0.95

        # Sample regime path z_t.
        #   - "easy_setting": first half regime 0, second half regime 1.
        #   - "noisy_forgetting":
        #         * if num_regimes <= 2: regime 0, then 1, then 0 again
        #           (three blocks), to expose potential catastrophic
        #           forgetting when the initial regime reappears;
        #         * if num_regimes >= 6: fixed multi-regime pattern
        #           0 → 1 → 2 → 1 → 3 → 4 → 5 → 2 in contiguous blocks;
        #         * otherwise (2 < num_regimes < 6): fallback multi-regime
        #           pattern 0 → 1 → 2 → ... → (M-1) → 0 in blocks.
        z = np.zeros(T, dtype=int)
        if T > 1:
            if setting == "noisy_forgetting":
                if M <= 2:
                    third = T // 3
                    two_third = 2 * T // 3
                    z[third:two_third] = 1
                    z[two_third:] = 0
                elif M >= 6:
                    # Multi-regime "forgetting" with explicit pattern
                    # 0 → 1 → 2 → 1 → 3 → 4 → 5 → 2.
                    pattern = [0, 1, 2, 1, 3, 4, 5, 2]
                    num_blocks = len(pattern)
                    block_len = max(1, T // num_blocks)
                    t = 0
                    for idx, k in enumerate(pattern):
                        # Ensure regime index fits within 0..M-1
                        k_eff = int(min(k, M - 1))
                        t_end = T if idx == num_blocks - 1 else min(T, t + block_len)
                        z[t:t_end] = k_eff
                        t = t_end
                    # Any leftover steps get the last regime.
                    if t < T:
                        z[t:] = z[t - 1]
                else:
                    # Fallback multi-regime "forgetting": 0,1,2,...,M-1,0 blocks.
                    num_blocks = M + 1  # 0,...,M-1,0
                    block_len = max(1, T // num_blocks)
                    t = 0
                    for b in range(num_blocks):
                        if b == 0 or b == num_blocks - 1:
                            k = 0
                        else:
                            k = min(b, M - 1)  # regimes 1,...,M-1
                        t_end = T if b == num_blocks - 1 else min(T, t + block_len)
                        z[t:t_end] = k
                        t = t_end
                    if t < T:
                        z[t:] = z[t - 1]
            else:
                change_point = T // 2
                z[change_point:] = 1
        self.z = z

        # AR(1) with regime-dependent drift / mean
        self.y = np.zeros(T, dtype=float)
        y = 0.0
        if noise_scale is None:
            if setting == "noisy_forgetting":
                noise = 0.6
            else:
                noise = 0.3
        else:
            noise = float(noise_scale)

        if setting == "noisy_forgetting" and M > 2:
            # For noisy_forgetting with more than 2 regimes, assign a
            # distinct drift level to each regime so that the time
            # series exhibits visibly different regime-dependent
            # behavior. Drift levels increase smoothly from 0 to 2.
            drift_levels = np.linspace(0.0, 2.0, M, dtype=float)
            for t in range(T):
                k = z[t]
                drift = drift_levels[int(k)]
                y = 0.8 * y + drift + rng.normal(scale=noise)
                self.y[t] = y
        else:
            # Original two-regime AR(1): drift 0 in regime 0, 1 in regime 1.
            for t in range(T):
                k = z[t]
                drift = 0.0 if k == 0 else 1.0
                y = 0.8 * y + drift + rng.normal(scale=noise)
                self.y[t] = y

        # Context x_t := y_{t-1}, with x_0 = 0
        self.x = np.zeros(T, dtype=float)
        self.x[0] = 0.0
        if T > 1:
            self.x[1:] = self.y[:-1]

        # Experts: different linear predictors y_hat = w_j x + b_j
        #   - Expert 0: tuned to regime 0  (drift ≈ 0)
        #   - Expert 1: tuned to regime 1  (drift ≈ 1)
        #   - Expert 2: average across regimes (drift ≈ 0.5)
        base_weights = np.array([0.8, 0.8, 0.8], dtype=float)
        base_biases = np.array([0.0, 1.0, 0.5], dtype=float)

        if num_experts <= 3:
            self.expert_weights = base_weights[:num_experts].copy()
            self.expert_biases = base_biases[:num_experts].copy()
        else:
            # For additional experts beyond the first three archetypes,
            # build correlated experts by perturbing one of the base
            # experts rather than sampling completely independently.
            extra = num_experts - 3
            extra_w = np.zeros(extra, dtype=float)
            extra_b = np.zeros(extra, dtype=float)
            for i in range(extra):
                base_idx = int(rng.integers(0, 3))
                extra_w[i] = base_weights[base_idx] + rng.normal(loc=0.0, scale=0.05)
                extra_b[i] = base_biases[base_idx] + rng.normal(loc=0.0, scale=0.2)
            self.expert_weights = np.concatenate([base_weights, extra_w])
            self.expert_biases = np.concatenate([base_biases, extra_b])

        # Expert availability over time: all experts available by default.
        # Optionally:
        #   - make one expert unavailable on one or more intervals
        #     via `unavailable_expert_idx` / `unavailable_intervals`, and/or
        #   - specify that one expert is only available on given intervals
        #     via `arrival_expert_idx` / `arrival_intervals` (dynamic addition).
        # For both types of intervals, [start, end] is interpreted with `end`
        # inclusive in the user-facing API.
        self.availability = np.ones((T, self.num_experts), dtype=int)
        if (
            unavailable_expert_idx is not None
            and 0 <= unavailable_expert_idx < self.num_experts
        ):
            # New: explicit list of unavailable intervals for this expert
            if unavailable_intervals is not None:
                for start, end in unavailable_intervals:
                    t_start = max(int(start), 0)
                    # `end` is inclusive in the user-facing API
                    t_end = min(int(end) + 1, T)
                    if t_start >= T or t_end <= 0:
                        continue
                    if t_start < t_end:
                        self.availability[t_start:t_end, unavailable_expert_idx] = 0
            # Backwards-compatible: single interval with heuristic end time.
            elif T > 2:
                if unavailable_start_t is None:
                    t_off = T // 4
                else:
                    t_off = int(unavailable_start_t)
                if 0 <= t_off < T - 1:
                    t_on = min(2 * t_off, T - 1)
                    if t_on > t_off:
                        self.availability[t_off:t_on, unavailable_expert_idx] = 0

        # Dynamic expert addition: restrict a single expert to be available
        # only on specified arrival intervals.
        if (
            arrival_expert_idx is not None
            and 0 <= arrival_expert_idx < self.num_experts
        ):
            # By default, the expert is unavailable everywhere, then we
            # turn it on only inside the specified arrival intervals.
            self.availability[:, arrival_expert_idx] = 0
            if arrival_intervals is not None:
                for start, end in arrival_intervals:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, T)  # inclusive end
                    if t_start >= T or t_end <= 0:
                        continue
                    if t_start < t_end:
                        self.availability[t_start:t_end, arrival_expert_idx] = 1

    def get_context(self, t: int) -> np.ndarray:
        return np.array([self.x[t]], dtype=float)

    def expert_predict(self, j: int, x_t: np.ndarray) -> float:
        w = self.expert_weights[j]
        b = self.expert_biases[j]
        return float(w * x_t[0] + b)

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
