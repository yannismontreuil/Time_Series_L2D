import numpy as np


class SyntheticTimeSeriesEnv:
    """
    Simple synthetic environment:

      - Two regimes (0,1) following a Markov chain.
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
    ):
        rng = np.random.default_rng(seed)
        self.num_experts = num_experts
        self.num_regimes = num_regimes
        self.T = T

        # True regime transition matrix
        self.Pi_true = np.array([[0.95, 0.05],
                                 [0.10, 0.90]], dtype=float)

        # Sample regime path z_t: first half regime 0, second half regime 1.
        z = np.zeros(T, dtype=int)
        if T > 1:
            change_point = T // 2
            z[change_point:] = 1
        self.z = z

        # AR(1) with regime-dependent drift
        self.y = np.zeros(T, dtype=float)
        y = 0.0
        for t in range(T):
            k = z[t]
            drift = 0.0 if k == 0 else 1.0
            y = 0.8 * y + drift + rng.normal(scale=0.3)
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
            extra = num_experts - 3
            extra_w = rng.normal(loc=0.8, scale=0.1, size=extra)
            extra_b = rng.normal(loc=0.5, scale=0.3, size=extra)
            self.expert_weights = np.concatenate([base_weights, extra_w])
            self.expert_biases = np.concatenate([base_biases, extra_b])

        # Expert availability over time: all experts available by default.
        # Optionally make one expert unavailable on an interval [t_off, t_on).
        self.availability = np.ones((T, self.num_experts), dtype=int)
        if (
            unavailable_expert_idx is not None
            and 0 <= unavailable_expert_idx < self.num_experts
            and T > 2
        ):
            if unavailable_start_t is None:
                t_off = T // 4
            else:
                t_off = int(unavailable_start_t)
            if 0 <= t_off < T - 1:
                t_on = min(2 * t_off, T - 1)
                if t_on > t_off:
                    self.availability[t_off:t_on, unavailable_expert_idx] = 0

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

