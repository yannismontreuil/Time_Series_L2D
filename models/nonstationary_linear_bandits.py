import math
from collections import deque
from typing import Callable, Sequence

import numpy as np

from models.linucb_baseline import LinUCB
from models.neuralucb_baseline import NeuralUCB
from models.shared_linear_bandits import _JointArmFeatureMap, _stable_cholesky


def _squash_nonnegative(x: float, scale: float) -> float:
    scale = max(float(scale), 1e-8)
    x = max(float(x), 0.0)
    return x / (x + scale)


def _bernoulli_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    q = np.clip(np.asarray(q, dtype=float), 1e-9, 1.0 - 1e-9)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


class DiscountedLinUCB:
    """
    D-LinUCB-style shared linear baseline for non-stationary environments.

    This is the direct paper-compatible baseline in this repo: one shared
    joint linear model with exponential forgetting.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        discount_gamma: float = 0.995,
        lambda_reg: float = 1.0,
        beta: np.ndarray | None = None,
        feedback_mode: str = "partial",
        context_dim: int | None = None,
        include_shared_context: bool = True,
        include_arm_bias: bool = True,
        include_arm_interactions: bool = True,
        param_norm_bound: float = 1.0,
        noise_std: float = 1.0,
        delta_confidence: float = 0.05,
        seed: int = 0,
    ):
        self.N = int(num_experts)
        self.feature_fn = feature_fn
        self.discount_gamma = float(discount_gamma)
        if not (0.0 < self.discount_gamma < 1.0):
            raise ValueError("discount_gamma must be in (0, 1).")
        self.lambda_reg = float(lambda_reg)
        if self.lambda_reg <= 0.0:
            raise ValueError("lambda_reg must be positive.")
        self.feedback_mode = str(feedback_mode).lower()
        if self.feedback_mode not in ("partial", "full"):
            raise ValueError("feedback_mode must be 'partial' or 'full'.")
        self.param_norm_bound = float(param_norm_bound)
        self.noise_std = float(noise_std)
        self.delta_confidence = float(delta_confidence)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        if beta is None:
            beta = np.zeros(self.N, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if beta.shape != (self.N,):
            raise ValueError("beta must have shape (num_experts,)")
        self.beta = beta

        self.feature_map = _JointArmFeatureMap(
            num_experts=self.N,
            feature_fn=self.feature_fn,
            context_dim=context_dim,
            include_shared_context=include_shared_context,
            include_arm_bias=include_arm_bias,
            include_arm_interactions=include_arm_interactions,
        )
        self.d = int(self.feature_map.total_dim)
        self.reset_state()

    def reset_state(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.V = self.lambda_reg * np.eye(self.d, dtype=float)
        self.V_tilde = self.lambda_reg * np.eye(self.d, dtype=float)
        self.b = np.zeros(self.d, dtype=float)

    def _context_features(self, x: np.ndarray) -> np.ndarray:
        return self.feature_map.context_features(x)

    def _joint_features(self, phi_x: np.ndarray, expert_idx: int) -> np.ndarray:
        return self.feature_map.joint_features_from_context(phi_x, expert_idx)

    def _iter_observed_losses(
        self,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
        selected_expert: int | None = None,
    ) -> list[tuple[int, float]]:
        losses = np.asarray(losses_all, dtype=float).reshape(self.N)
        available = np.asarray(list(available_experts), dtype=int)
        observations: list[tuple[int, float]] = []
        if self.feedback_mode == "partial":
            if selected_expert is None:
                raise ValueError("selected_expert must be provided in partial mode.")
            j = int(selected_expert)
            observations.append((j, float(losses[j])))
            return observations
        for j in available:
            ell = float(losses[int(j)])
            if np.isfinite(ell):
                observations.append((int(j), ell))
        return observations

    def _beta_t(self, max_feat_norm: float) -> float:
        t = max(self.step_count, 1)
        gamma = self.discount_gamma
        frac = (1.0 - gamma ** (2 * t)) / max(1.0 - gamma**2, 1e-12)
        inside = 1.0 + (max_feat_norm**2) * frac / max(self.lambda_reg * self.d, 1e-12)
        inside = max(inside, 1.0)
        return float(
            self.lambda_reg * self.param_norm_bound
            + self.noise_std
            * math.sqrt(max(2.0 * math.log(1.0 / max(self.delta_confidence, 1e-12)) + self.d * math.log(inside), 0.0))
        )

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        available = np.asarray(list(available_experts), dtype=int)
        if available.size == 0:
            raise ValueError("DiscountedLinUCB: no available experts.")
        phi_x = self._context_features(x)
        chol_v = _stable_cholesky(self.V)
        theta = np.linalg.solve(chol_v.T, np.linalg.solve(chol_v, self.b))
        feats = [self._joint_features(phi_x, int(j)) for j in available]
        max_feat_norm = max(float(np.linalg.norm(feat)) for feat in feats)
        beta_t = self._beta_t(max_feat_norm)

        best_score = None
        best_expert = None
        for j, feat in zip(available, feats):
            v_inv_feat = np.linalg.solve(chol_v.T, np.linalg.solve(chol_v, feat))
            sigma_sq = float(feat @ (self.V_tilde @ v_inv_feat))
            sigma = float(np.sqrt(max(sigma_sq, 0.0)))
            mu = float(theta @ feat)
            score = mu - beta_t * sigma + self.beta[int(j)]
            if best_score is None or score < best_score:
                best_score = score
                best_expert = int(j)
        return int(best_expert)

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
        selected_expert: int | None = None,
    ) -> None:
        phi_x = self._context_features(x)
        observations = self._iter_observed_losses(
            losses_all=losses_all,
            available_experts=available_experts,
            selected_expert=selected_expert,
        )
        if not observations:
            return
        gamma = self.discount_gamma
        self.V = gamma * self.V + (1.0 - gamma) * self.lambda_reg * np.eye(self.d)
        self.V_tilde = (
            (gamma**2) * self.V_tilde
            + (1.0 - gamma**2) * self.lambda_reg * np.eye(self.d)
        )
        self.b = gamma * self.b
        for j, ell in observations:
            feat = self._joint_features(phi_x, j)
            self.V += np.outer(feat, feat)
            self.V_tilde += np.outer(feat, feat)
            self.b += feat * float(ell)
        self.step_count += 1


class _CUSUMDetector:
    def __init__(self, warmup: int = 25, epsilon: float = 0.02, threshold: float = 0.25):
        self.warmup = int(warmup)
        self.epsilon = float(epsilon)
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self.samples: list[float] = []
        self.baseline_mean: float | None = None
        self.g_plus = 0.0
        self.g_minus = 0.0

    def update(self, z: float) -> bool:
        z = float(np.clip(z, 0.0, 1.0))
        self.samples.append(z)
        if self.baseline_mean is None:
            if len(self.samples) < self.warmup:
                return False
            self.baseline_mean = float(np.mean(self.samples[: self.warmup]))
            self.samples = self.samples[self.warmup :]
            return False
        s_plus = z - self.baseline_mean - self.epsilon
        s_minus = self.baseline_mean - z - self.epsilon
        self.g_plus = max(0.0, self.g_plus + s_plus)
        self.g_minus = max(0.0, self.g_minus + s_minus)
        return self.g_plus >= self.threshold or self.g_minus >= self.threshold


class CUSUMLinUCB(LinUCB):
    """
    Contextualized CUSUM-UCB analogue for routing.

    The original paper is non-contextual MAB. Here the change detector runs on
    bounded prediction-error magnitudes, while the action-selection backbone is
    the repo's contextual LinUCB.
    """

    def __init__(
        self,
        *args,
        detector_warmup: int = 25,
        detector_epsilon: float = 0.02,
        detector_threshold: float = 0.25,
        detector_scale: float = 1.0,
        random_explore_prob: float = 0.05,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detector_warmup = int(detector_warmup)
        self.detector_epsilon = float(detector_epsilon)
        self.detector_threshold = float(detector_threshold)
        self.detector_scale = float(detector_scale)
        self.random_explore_prob = float(random_explore_prob)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.detectors = [
            _CUSUMDetector(
                warmup=self.detector_warmup,
                epsilon=self.detector_epsilon,
                threshold=self.detector_threshold,
            )
            for _ in range(self.N)
        ]

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            raise ValueError("CUSUMLinUCB: no available experts.")
        if self.rng.random() < self.random_explore_prob:
            return int(self.rng.choice(available_experts))
        return super().select_expert(x, available_experts)

    def _reset_expert(self, j: int) -> None:
        self.A[int(j)] = self.lambda_reg * np.eye(self.d, dtype=float)
        self.b[int(j)] = 0.0
        self.detectors[int(j)].reset()

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
        selected_expert: int | None = None,
    ) -> None:
        phi = self._get_phi(x)
        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        observed: list[int]
        if self.feedback_mode == "partial":
            if selected_expert is None:
                raise ValueError("selected_expert must be provided in partial mode.")
            observed = [int(selected_expert)]
        else:
            observed = [int(j) for j in np.asarray(list(available_experts), dtype=int)]

        for j in observed:
            if not np.isfinite(losses_all[j]):
                continue
            mu_j, _ = self._theta_and_sigma(j, phi)
            err = abs(float(losses_all[j]) - mu_j)
            z = _squash_nonnegative(err, self.detector_scale)
            alarm = self.detectors[j].update(z)
            if alarm:
                self._reset_expert(j)
            self.A[j] += np.outer(phi, phi)
            self.b[j] += phi * float(losses_all[j])


class _GLRDetector:
    def __init__(self, delta: float = 0.05, min_window: int = 20):
        self.delta = float(delta)
        self.min_window = int(min_window)
        self.reset()

    def reset(self) -> None:
        self.samples: list[float] = []

    def _threshold(self, n: int) -> float:
        n = max(int(n), 2)
        return float(
            2.0 * math.log(max(3.0 * n * math.sqrt(float(n)) / max(self.delta, 1e-12), 1.0))
            + 6.0 * math.log1p(math.log(max(n, 2)))
        )

    def update(self, z: float) -> bool:
        z = float(np.clip(z, 0.0, 1.0))
        self.samples.append(z)
        n = len(self.samples)
        if n < 2 * self.min_window:
            return False
        arr = np.asarray(self.samples, dtype=float)
        prefix = np.cumsum(arr)
        total = float(prefix[-1])
        s_vals = np.arange(self.min_window, n - self.min_window + 1)
        left_sum = prefix[s_vals - 1]
        right_sum = total - left_sum
        mu_all = total / n
        mu_left = left_sum / s_vals
        mu_right = right_sum / (n - s_vals)
        stat = s_vals * _bernoulli_kl(mu_left, mu_all) + (n - s_vals) * _bernoulli_kl(mu_right, mu_all)
        return float(np.max(stat)) >= self._threshold(n)


class GLRLinUCB(LinUCB):
    """
    Contextualized GLR-CUCB analogue for routing.

    The original paper is a combinatorial semi-bandit with per-base-arm GLR
    detection and a global restart. Here the backbone is contextual LinUCB,
    while the detector monitors bounded prediction-error magnitudes.
    """

    def __init__(
        self,
        *args,
        detector_delta: float = 0.05,
        detector_min_window: int = 20,
        detector_scale: float = 1.0,
        forced_exploration_prob: float = 0.1,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detector_delta = float(detector_delta)
        self.detector_min_window = int(detector_min_window)
        self.detector_scale = float(detector_scale)
        self.forced_exploration_prob = float(forced_exploration_prob)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.detectors = [
            _GLRDetector(
                delta=self.detector_delta,
                min_window=self.detector_min_window,
            )
            for _ in range(self.N)
        ]
        self.restart_time = 0

    def reset_all(self) -> None:
        self.A[:] = 0.0
        self.b[:] = 0.0
        for j in range(self.N):
            self.A[j] = self.lambda_reg * np.eye(self.d, dtype=float)
            self.detectors[j].reset()
        self.restart_time = 0

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        available_experts = np.asarray(list(available_experts), dtype=int)
        if available_experts.size == 0:
            raise ValueError("GLRLinUCB: no available experts.")
        if self.rng.random() < self.forced_exploration_prob:
            target = int(self.restart_time % self.N)
            if target in set(int(j) for j in available_experts):
                return target
            return int(self.rng.choice(available_experts))
        return super().select_expert(x, available_experts)

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
        selected_expert: int | None = None,
    ) -> None:
        phi = self._get_phi(x)
        losses_all = np.asarray(losses_all, dtype=float).reshape(self.N)
        if self.feedback_mode == "partial":
            if selected_expert is None:
                raise ValueError("selected_expert must be provided in partial mode.")
            observed = [int(selected_expert)]
        else:
            observed = [int(j) for j in np.asarray(list(available_experts), dtype=int)]

        global_alarm = False
        for j in observed:
            if not np.isfinite(losses_all[j]):
                continue
            mu_j, _ = self._theta_and_sigma(j, phi)
            err = abs(float(losses_all[j]) - mu_j)
            z = _squash_nonnegative(err, self.detector_scale)
            if self.detectors[j].update(z):
                global_alarm = True
            self.A[j] += np.outer(phi, phi)
            self.b[j] += phi * float(losses_all[j])

        if global_alarm:
            self.reset_all()
        else:
            self.restart_time += 1


class SlidingWindowNeuralUCB(NeuralUCB):
    """
    Sliding-window NeuralUCB.

    Keeps the last `window_size` observed updates and refits the neural-linear
    model on that window after each round. This is the cleanest way to adapt the
    existing NeuralUCB approximation to a finite-memory non-stationary setting.
    """

    def __init__(self, *args, window_size: int = 90, seed: int | None = 0, **kwargs):
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        self._seed_value = 0 if seed is None else int(seed)
        super().__init__(*args, seed=seed, **kwargs)
        self._base_seed = self._seed_value
        self._history: deque[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = deque(
            maxlen=self.window_size
        )

    def reset_state(self) -> None:
        rng = np.random.default_rng(self._base_seed)
        scale = 0.1
        self.W1 = rng.normal(scale=scale, size=(self.hidden_dim, self.d))
        self.b1 = np.zeros(self.hidden_dim, dtype=float)
        self.A = np.zeros((self.N, self.hidden_dim, self.hidden_dim), dtype=float)
        self.b_lin = np.zeros((self.N, self.hidden_dim), dtype=float)
        for j in range(self.N):
            self.A[j] = self.lambda_reg * np.eye(self.hidden_dim, dtype=float)
        self._history = deque(maxlen=self.window_size)

    def _rebuild_from_window(self) -> None:
        history = list(self._history)
        rng = np.random.default_rng(self._base_seed)
        scale = 0.1
        self.W1 = rng.normal(scale=scale, size=(self.hidden_dim, self.d))
        self.b1 = np.zeros(self.hidden_dim, dtype=float)
        self.A = np.zeros((self.N, self.hidden_dim, self.hidden_dim), dtype=float)
        self.b_lin = np.zeros((self.N, self.hidden_dim), dtype=float)
        for j in range(self.N):
            self.A[j] = self.lambda_reg * np.eye(self.hidden_dim, dtype=float)

        for x, losses_all, available, selected in history:
            super().update(
                x=x,
                losses_all=losses_all,
                available_experts=available,
                selected_expert=selected,
            )

    def update(
        self,
        x: np.ndarray,
        losses_all: np.ndarray,
        available_experts: Sequence[int],
        selected_expert: int | None = None,
    ) -> None:
        losses = np.asarray(losses_all, dtype=float).reshape(self.N).copy()
        available = np.asarray(list(available_experts), dtype=int).copy()
        selected = -1 if selected_expert is None else int(selected_expert)
        self._history.append((np.asarray(x, dtype=float).copy(), losses, available, selected))
        self._rebuild_from_window()
