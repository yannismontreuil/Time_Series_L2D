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

    This is deliberately simple: the router's SLDS model does not need to
    match this true generative process exactsyly; it just infers from losses.
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
        """
        example: when M=3,
        Pi_true = [[0.95, 0.025, 0.025],
                   [0.025, 0.95, 0.025],
                   [0.025, 0.025, 0.95]]
        so that the system has a high probability of staying in the current regime 
        and a low, equally distributed probability of switching to any other regime.
        """

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
        #   - "sidekick_trap": Phase 1 (t=0..999) in regime 0 (calm),
        #                      Phase 2+3 (t≥1000) in regime 1 (storm),
        #                      for the "Correlated Sidekick Trap" experiment.
        #   - "cluster_transfer": first half regime 0, second half regime 1
        #                         for the "Sparse Cluster Transfer" experiment.
        #   - "theoretical_trap": Phase 1 (t=0..999) in regime 0 (good times),
        #                         Phase 2+3 (t≥1000) in regime 1 (bad times),
        #                         for the "Stale Prior" trap experiment.
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
            elif setting == "sidekick_trap":
                # Explicit two-regime pattern: first 1000 steps in regime 0,
                # remaining steps in regime 1 (storm).
                phase1_end = min(1000, T - 1)
                if phase1_end < 0:
                    phase1_end = 0
                if phase1_end + 1 < T:
                    z[phase1_end + 1 :] = 1
            elif setting == "theoretical_trap":
                # Same two-regime structure as sidekick_trap: first 1000
                # steps regime 0 (good), remaining steps regime 1 (bad).
                phase1_end = min(1000, T - 1)
                if phase1_end < 0:
                    phase1_end = 0
                if phase1_end + 1 < T:
                    z[phase1_end + 1 :] = 1
            elif setting == "cluster_transfer":
                # Two-regime pattern: first half regime 0, second half regime 1.
                change_point = T // 2
                z[change_point:] = 1
            elif setting == "easy_setting":
                change_point = T // 2
                z[change_point:] = 1
                # Default (including "easy_setting"): divide time into M blocks,
                # one block per regime 0, 1, 2, ..., M-1
            elif setting == "division_by_M":
                if M <= 2:
                    # For 2 regimes, keep original behavior: first half 0, second half 1
                    change_point = T // 2
                    z[change_point:] = 1
                else:
                    # For M > 2, divide T into M equal blocks
                    block_len = max(1, T // M)
                    t = 0
                    for k in range(M):
                        print(f"Regime {k}: time {t} to ", end="")
                        t_end = T if k == M - 1 else min(T, t + block_len)
                        print(t_end)
                        z[t:t_end] = k
                        t = t_end
            else:
                raise ValueError(f"Unknown setting: {setting}")
        self.z = z
        # Track the last time index used in get_context so that
        # expert_predict can implement regime-dependent behaviour in
        # special synthetic settings without changing its interface.
        self._last_t: int | None = None

        # Time series / shared factor dynamics.
        # - For "sidekick_trap", we construct an explicit "Day/Night"
        #   shared factor g_t := 10 * z_t so that g_t is 0 during Day
        #   (regime 0) and 10 during Night (regime 1), and set y_t := g_t
        #   for plotting / context.
        # - For all other settings (including "cluster_transfer"), we use
        #   the original AR(1) construction for y_t; cluster-specific
        #   factors g_A,t, g_B,t are defined separately and do not make
        #   the observable time series a step function.
        self.y = np.zeros(T, dtype=float)
        if setting == "sidekick_trap":
            """
            Regime Mapping: z_t = 0 => y_t = 0 (Day), z_t = 1 => y_t = 10 (Night)
            """
            g = 10.0 * z.astype(float)
            self.y[:] = g

            # Precompute high-contrast "Crossing Experts" losses with
            # noise as in the corrected trap design.
            penalty = 10.0
            # Base noise around means:
            loss_1 = rng.normal(loc=0.0, scale=0.05, size=T)
            loss_2 = rng.normal(loc=0.0, scale=0.05, size=T)
            loss_3 = rng.normal(loc=1.0, scale=0.05, size=T)

            # Regime 0 (Day): Expert 3 bad; 
            # Regime 1 (Night): Experts 1 & 2 bad.
            loss_3[z == 0] += penalty      # Expert 3 ~ 11.0 during Day
            loss_1[z == 1] += penalty      # Expert 1 ~ 10.0 during Night
            loss_2[z == 1] += penalty      # Expert 2 ~ 10.0 during Night

            loss_1 = np.maximum(loss_1, 0.0)
            loss_2 = np.maximum(loss_2, 0.0)
            loss_3 = np.maximum(loss_3, 0.0)

            base_losses = np.stack([loss_1, loss_2, loss_3], axis=1)
            if num_experts > 3:
                # For any extra experts, replicate Safe Harbor pattern.
                extra = num_experts - 3
                extra_mat = np.tile(loss_3.reshape(T, 1), (1, extra))
                self._trap_losses = np.concatenate(
                    [base_losses, extra_mat], axis=1
                )
            else:
                self._trap_losses = base_losses[:, :num_experts]
        else:
            y = 0.0
            if noise_scale is None:
                if setting == "noisy_forgetting":
                    noise = 0.6
                else:
                    noise = 0.3
            else:
                noise = float(noise_scale)

            if (setting == "noisy_forgetting" and M > 2) or (setting == "division_by_M" and M > 2):
                # For noisy_forgetting or division_by_M with more than 2 regimes, assign a
                # distinct drift level to each regime so that the time
                # series exhibits visibly different regime-dependent
                # behavior. Drift levels increase smoothly from 0 to 2.
                drift_levels = np.linspace(0.0, 2.0, M, dtype=float)
                for t in range(T):
                    k = z[t]
                    drift = drift_levels[int(k)]
                    y = 0.8 * y + drift + rng.normal(scale=noise)
                    self.y[t] = y
                    """
                    Assume M = 5, then drift_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
                    Thus, each regime k has a distinct drift level:
                    Regime 0: drift ~ 0.0, y_0 = 0.0 + noise
                    Regime 1: drift ~ 0.5, y_1 = 0.8*y_0 + 0.5 + noise
                    Regime 2: drift ~ 1.0, y_2 = 0.8*y_1 + 1.0 + noise
                    Regime 3: drift ~ 1.5, y_3 = 0.8*y_2 + 1.5 + noise
                    Regime 4: drift ~ 2.0, y_4 = 0.8*y_3 + 2.0 + noise
                    Assume noise = 0, then y_4 = 
                    """
            else:
                # Original two-regime AR(1): drift 0 in regime 0, 1 in regime 1.
                for t in range(T):
                    k = z[t]
                    drift = 0.0 if k == 0 else 1.0
                    y = 0.8 * y + drift + rng.normal(scale=noise)
                    self.y[t] = y

            # For the "cluster_transfer" experiment, define cluster
            # factors on top of the AR(1) time series so that the
            # observable series is not a step function.
            if setting == "cluster_transfer":
                # Two clusters A and B with shared factors g_A,t and g_B,t.
                # Regime 0: Cluster A good (~0.1), B bad (~5.0).
                # Regime 1: Cluster B good (~0.1), A bad (~5.0).
                g_A = np.where(z == 0, 0.1, 5.0) + rng.normal(
                    loc=0.0, scale=0.05, size=T
                )
                g_B = np.where(z == 0, 5.0, 0.1) + rng.normal(
                    loc=0.0, scale=0.05, size=T
                )
                self._g_A = g_A
                self._g_B = g_B
            if setting == "theoretical_trap":
                # Stale Prior trap experiment:
                # Regime 0 (good times): experts 0 and 1 ~ 0; expert 2 ~ 2.
                # Regime 1 (bad times):  experts 0 and 1 ~ 4 on average,
                # with enough variance that they are occasionally better
                # than expert 2, which remains ~2. This preserves
                # μ_hist < μ_safe < μ_latent in expectation, but makes
                # Regime 1 decisions non-trivial.
                loss_1 = rng.normal(loc=0.0, scale=0.05, size=T)
                loss_2 = rng.normal(loc=0.0, scale=0.05, size=T)
                loss_3 = rng.normal(loc=2.0, scale=0.05, size=T)

                bad_idx = np.where(z == 1)[0]
                if bad_idx.size > 0:
                    # In the bad regime, draw expert-0/1 losses from a
                    # broader distribution centered at 4.0 so that they
                    # sometimes fall below 2.0.
                    loss_1[bad_idx] = rng.normal(loc=4.0, scale=2.0, size=bad_idx.size)
                    loss_2[bad_idx] = rng.normal(loc=4.0, scale=2.0, size=bad_idx.size)

                loss_1 = np.maximum(loss_1, 0.0)
                loss_2 = np.maximum(loss_2, 0.0)
                loss_3 = np.maximum(loss_3, 0.0)

                base_losses2 = np.stack([loss_1, loss_2, loss_3], axis=1)
                if num_experts > 3:
                    extra = num_experts - 3
                    extra_mat = np.tile(loss_3.reshape(T, 1), (1, extra))
                    self._theoretical_losses = np.concatenate(
                        [base_losses2, extra_mat], axis=1
                    )
                else:
                    self._theoretical_losses = base_losses2[:, :num_experts]

        # Context x_t := y_{t-1}, with x_0 = 0
        self.x = np.zeros(T, dtype=float)
        self.x[0] = 0.0
        if T > 1:
            self.x[1:] = self.y[:-1]

        # Experts: different linear predictors y_hat = w_j x + b_j
        #   - Expert 0: slightly under-tuned to regime 0
        #   - Expert 1: tuned to the average AR(1) behaviour and designed
        #               to be best on average across regimes.
        #   - Expert 2: deliberately more biased.
        # We additionally enforce explicit correlation structure:
        #   - Experts 0 and 1 have similar linear forms, making them
        #     correlated in their predictions.
        #   - Experts 3 and 4 (when present) are constructed as
        #     perturbations of expert 1, making them strongly correlated
        #     with each other and with expert 1, but slightly worse on
        #     average due to biased intercepts.
        base_weights = np.array([0.6, 0.8, 0.6], dtype=float)
        base_biases = np.array([0.0, 0.5, 1.0], dtype=float)

        # For the theoretical trap experiment used in the synth_paper
        # setting, slightly improve expert 0's linear fit so that in the
        # second (bad) regime it occasionally competes more closely with
        # the safe-harbor expert. This increases oscillation in which
        # expert is momentarily best, while preserving the overall
        # qualitative structure of the experiment.
        if setting == "theoretical_trap" and num_experts > 0:
            # Slightly closer to the true AR(1) slope (0.8) so that
            # expert 0 is more competitive, especially in the higher-
            # variance second regime.
            base_weights[0] = 0.75

        if num_experts <= 3:
            self.expert_weights = base_weights[:num_experts].copy()
            self.expert_biases = base_biases[:num_experts].copy()
        else:
            # For additional experts beyond the first three archetypes,
            # build correlated experts by perturbing base experts. In
            # particular, experts 3 and 4 are constructed from expert 1
            # to enforce correlation between them, while being slightly
            # more biased (on average) so that expert 1 remains the best
            # single expert across runs. Any further experts are
            # generated by perturbing a random base expert.
            extra = num_experts - 3
            extra_w = np.zeros(extra, dtype=float)
            extra_b = np.zeros(extra, dtype=float)
            for i in range(extra):
                global_idx = 3 + i
                if global_idx in (3, 4):
                    # Experts 3 and 4: perturbations of expert 1 with a
                    # biased intercept to make them slightly worse on
                    # average than expert 1 while remaining correlated.
                    base_idx = 1
                    extra_w[i] = base_weights[base_idx] + rng.normal(
                        loc=0.0, scale=0.05
                    )
                    extra_b[i] = base_biases[base_idx] + rng.normal(
                        loc=0.3, scale=0.2
                    )
                else:
                    base_idx = int(rng.integers(0, 3))
                    extra_w[i] = base_weights[base_idx] + rng.normal(
                        loc=0.0, scale=0.05
                    )
                    extra_b[i] = base_biases[base_idx] + rng.normal(
                        loc=0.0, scale=0.2
                    )
            self.expert_weights = np.concatenate([base_weights, extra_w])
            self.expert_biases = np.concatenate([base_biases, extra_b])

        # Expert availability over time.
        # By default:
        #   - all experts available at all times, optionally modified by
        #     unavailable_*/arrival_* arguments.
        # For special settings like "cluster_transfer", we override this
        # with a handcrafted pattern.
        if setting == "cluster_transfer":
            # Sparse Cluster Transfer:
            #   - Cluster A: experts 0,1,2
            #   - Cluster B: experts 3,4,5
            # All experts have time-varying availability, with probes
            # available most of the time and targets appearing in blocks.
            self.availability = np.zeros((T, self.num_experts), dtype=int)

            # Expert 0 (A probe): available except on a few outages.
            if self.num_experts > 0:
                self.availability[:, 0] = 1
                for start, end in [(600, 700), (2200, 2300)]:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, T)
                    if t_start < t_end:
                        self.availability[t_start:t_end, 0] = 0

            # Expert 1 (A target 1): several blocks in Regime 0 and 1.
            if self.num_experts > 1:
                for start, end in [(200, 400), (800, 1000), (1800, 1900)]:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, T)
                    if t_start < t_end:
                        self.availability[t_start:t_end, 1] = 1

            # Expert 2 (A target 2): complementary blocks.
            if self.num_experts > 2:
                for start, end in [(100, 300), (1200, 1400), (2100, 2300)]:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, T)
                    if t_start < t_end:
                        self.availability[t_start:t_end, 2] = 1

            # Expert 3 (B probe): available except on a few outages.
            if self.num_experts > 3:
                self.availability[:, 3] = 1
                for start, end in [(900, 1000), (2500, 2600)]:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, T)
                    if t_start < t_end:
                        self.availability[t_start:t_end, 3] = 0

            # Expert 4 (B target 1): blocks mostly in Regime 1.
            if self.num_experts > 4:
                for start, end in [(1600, 1800), (2200, 2400)]:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, T)
                    if t_start < t_end:
                        self.availability[t_start:t_end, 4] = 1

            # Expert 5 (B target 2): complementary blocks.
            if self.num_experts > 5:
                for start, end in [(1700, 1900), (2300, 2500)]:
                    t_start = max(int(start), 0)
                    t_end = min(int(end) + 1, T)
                    if t_start < t_end:
                        self.availability[t_start:t_end, 5] = 1
        elif setting == "theoretical_trap":
            # Stale Prior trap:
            # Default: all experts available, except expert 1 (index 1)
            # goes offline during the start of the bad regime.
            self.availability = np.ones((T, self.num_experts), dtype=int)
            if self.num_experts > 1:
                t_start = max(1000, 0)
                t_end = min(2000, T)
                if t_start < t_end:
                    self.availability[t_start:t_end, 1] = 0
        else:
            # Default: all experts available, then apply optional masks.
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
        # Remember the time index of this context query so that
        # expert_predict can optionally condition on the current regime
        # in special synthetic settings (e.g., theoretical_trap).
        self._last_t = int(t)
        return np.array([self.x[t]], dtype=float)

    def expert_predict(self, j: int, x_t: np.ndarray) -> float:
        # Special design for the theoretical_trap experiment: we
        # construct regime-dependent expert behaviours that (i) make
        # experts 0 and 1 strongly correlated, (ii) make expert 1
        # excellent in the first regime but very bad in the second
        # while it is partly offline, and (iii) keep expert 2 as a
        # mediocre "safe harbour". This amplifies the stale-prior trap
        # while preserving the MSE+consultation evaluation.
        if getattr(self, "setting", None) == "theoretical_trap" and self._last_t is not None:
            t = int(self._last_t)
            if not (0 <= t < self.T):
                t = max(0, min(t, self.T - 1))
            reg = int(self.z[t])
            x_val = float(x_t[0])

            # Separate RNG stream for expert prediction noise.
            if not hasattr(self, "_rng_expert_noise"):
                self._rng_expert_noise = np.random.default_rng()

            # Regime 0 ("good times"):
            #   - Experts 0 and 1 track the AR(1) dynamics well:
            #         y_hat ≈ 0.8 * x_t  (small variance)
            #   - Expert 2 is a mediocre constant forecaster around 1.0.
            #
            # Regime 1 ("bad times"):
            #   - Expert 2 now tracks the AR(1) with drift 1.0:
            #         y_hat ≈ 0.8 * x_t + 1.0  (good)
            #   - Experts 0 and 1 become "disastrous" constant forecasters
            #     around 3.0, remaining highly correlated with each other.
            if reg == 0:
                if int(j) in (0, 1):
                    mean = 0.8 * x_val
                    noise = float(self._rng_expert_noise.normal(loc=0.0, scale=0.05))
                    return mean + noise
                else:
                    mean = 1.0
                    noise = float(self._rng_expert_noise.normal(loc=0.0, scale=0.05))
                    return mean + noise
            else:
                if int(j) >= 2:
                    mean = 0.8 * x_val + 1.0
                    noise = float(self._rng_expert_noise.normal(loc=0.0, scale=0.05))
                    return mean + noise
                else:
                    mean = 3.0
                    noise = float(self._rng_expert_noise.normal(loc=0.0, scale=0.1))
                    return mean + noise

        # Default linear expert for all other settings.
        w = self.expert_weights[j]
        b = self.expert_biases[j]
        return float(w * float(x_t[0]) + b)

    def all_expert_predictions(self, x_t: np.ndarray) -> np.ndarray:
        return np.array(
            [self.expert_predict(j, x_t) for j in range(self.num_experts)],
            dtype=float,
        )

    def losses(self, t: int) -> np.ndarray:
        """
        Return squared losses ℓ_{j,t} = (ŷ_{j,t} - y_t)^2 for all
        experts at time t. This quantity, plus the consultation cost
        β_j, is the evaluation cost of querying expert j and is used
        consistently for all routers, baselines, and the oracle.
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
