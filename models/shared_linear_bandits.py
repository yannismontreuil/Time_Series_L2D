import numpy as np
from typing import Callable, Sequence


def _stable_cholesky(mat: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    eye = np.eye(mat.shape[0], dtype=float)
    cur_jitter = float(jitter)
    for _ in range(8):
        try:
            return np.linalg.cholesky(mat + cur_jitter * eye)
        except np.linalg.LinAlgError:
            cur_jitter *= 10.0
    raise np.linalg.LinAlgError("Cholesky factorization failed even after jitter.")


def _solve_spd(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    chol = _stable_cholesky(mat)
    y = np.linalg.solve(chol, rhs)
    return np.linalg.solve(chol.T, y)


class _JointArmFeatureMap:
    """
    Joint linear features for (context, expert) pairs.

    Default feature layout:
      [shared context, expert bias one-hot, expert-context interactions]

    This gives one global/shared block while still allowing expert-dependent
    responses, which is the minimum sensible "shared across experts" linear
    baseline when no expert metadata are available.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        context_dim: int | None = None,
        include_shared_context: bool = True,
        include_arm_bias: bool = True,
        include_arm_interactions: bool = True,
    ):
        self.num_experts = int(num_experts)
        self.feature_fn = feature_fn
        self.include_shared_context = bool(include_shared_context)
        self.include_arm_bias = bool(include_arm_bias)
        self.include_arm_interactions = bool(include_arm_interactions)

        if context_dim is None:
            dummy_x = np.zeros(1, dtype=float)
        else:
            dummy_x = np.zeros(int(context_dim), dtype=float)
        phi = np.asarray(self.feature_fn(dummy_x), dtype=float).reshape(-1)
        self.context_feature_dim = int(phi.shape[0])

        dim = 0
        self.shared_slice = slice(dim, dim)
        if self.include_shared_context:
            self.shared_slice = slice(dim, dim + self.context_feature_dim)
            dim += self.context_feature_dim

        self.bias_slice = slice(dim, dim)
        if self.include_arm_bias:
            self.bias_slice = slice(dim, dim + self.num_experts)
            dim += self.num_experts

        self.interaction_slice = slice(dim, dim)
        if self.include_arm_interactions:
            self.interaction_slice = slice(
                dim, dim + self.num_experts * self.context_feature_dim
            )
            dim += self.num_experts * self.context_feature_dim

        self.total_dim = int(dim)
        if self.total_dim <= 0:
            raise ValueError("Joint arm feature map must have positive dimension.")

    def context_features(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.feature_fn(x), dtype=float).reshape(
            self.context_feature_dim
        )

    def joint_features(self, x: np.ndarray, expert_idx: int) -> np.ndarray:
        phi_x = self.context_features(x)
        return self.joint_features_from_context(phi_x, expert_idx)

    def joint_features_from_context(
        self,
        phi_x: np.ndarray,
        expert_idx: int,
    ) -> np.ndarray:
        j = int(expert_idx)
        if j < 0 or j >= self.num_experts:
            raise ValueError(f"expert_idx out of range: {j}")
        phi_x = np.asarray(phi_x, dtype=float).reshape(self.context_feature_dim)
        feat = np.zeros(self.total_dim, dtype=float)
        if self.include_shared_context:
            feat[self.shared_slice] = phi_x
        if self.include_arm_bias:
            feat[self.bias_slice.start + j] = 1.0
        if self.include_arm_interactions:
            start = self.interaction_slice.start + j * self.context_feature_dim
            feat[start : start + self.context_feature_dim] = phi_x
        return feat


class _SharedLinearBanditBase:
    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        beta: np.ndarray | None = None,
        feedback_mode: str = "partial",
        context_dim: int | None = None,
        lambda_reg: float = 1.0,
        include_shared_context: bool = True,
        include_arm_bias: bool = True,
        include_arm_interactions: bool = True,
        seed: int = 0,
    ):
        self.N = int(num_experts)
        self.feature_fn = feature_fn

        if beta is None:
            beta = np.zeros(self.N, dtype=float)
        else:
            beta = np.asarray(beta, dtype=float)
            if beta.shape != (self.N,):
                raise ValueError("beta must have shape (num_experts,)")
        self.beta = beta

        feedback_mode = str(feedback_mode).lower()
        if feedback_mode not in ("partial", "full"):
            raise ValueError("feedback_mode must be 'partial' or 'full'.")
        self.feedback_mode = feedback_mode

        lambda_reg = float(lambda_reg)
        if lambda_reg <= 0.0:
            raise ValueError("lambda_reg must be positive.")
        self.lambda_reg = lambda_reg

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
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
        self.A = self.lambda_reg * np.eye(self.d, dtype=float)
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
        if available.size == 0:
            raise ValueError("No available experts supplied to update().")

        observations: list[tuple[int, float]] = []
        if self.feedback_mode == "partial":
            if selected_expert is None:
                raise ValueError("selected_expert must be provided in partial mode.")
            j_sel = int(selected_expert)
            if j_sel not in available:
                raise ValueError("selected_expert must be in available_experts.")
            ell = float(losses[j_sel])
            if not np.isfinite(ell):
                raise ValueError("Observed partial-feedback loss must be finite.")
            observations.append((j_sel, ell))
            return observations

        for j in available:
            ell = float(losses[int(j)])
            if not np.isfinite(ell):
                continue
            observations.append((int(j), ell))
        return observations

    def _posterior_mean(self) -> np.ndarray:
        return _solve_spd(self.A, self.b)


class SharedLinUCB(_SharedLinearBanditBase):
    """
    Joint linear UCB baseline over expert-conditioned features.

    Predicted loss for expert j is approximated by θ^T φ(x, j), with one
    shared parameter vector θ fit across all experts.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        alpha_ucb: float = 1.0,
        lambda_reg: float = 1.0,
        beta: np.ndarray | None = None,
        feedback_mode: str = "partial",
        context_dim: int | None = None,
        include_shared_context: bool = True,
        include_arm_bias: bool = True,
        include_arm_interactions: bool = True,
        seed: int = 0,
    ):
        self.alpha_ucb = float(alpha_ucb)
        if self.alpha_ucb < 0.0:
            raise ValueError("alpha_ucb must be non-negative.")
        super().__init__(
            num_experts=num_experts,
            feature_fn=feature_fn,
            beta=beta,
            feedback_mode=feedback_mode,
            context_dim=context_dim,
            lambda_reg=lambda_reg,
            include_shared_context=include_shared_context,
            include_arm_bias=include_arm_bias,
            include_arm_interactions=include_arm_interactions,
            seed=seed,
        )

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        available = np.asarray(list(available_experts), dtype=int)
        if available.size == 0:
            raise ValueError("SharedLinUCB: no available experts in select_expert.")
        phi_x = self._context_features(x)
        chol = _stable_cholesky(self.A)
        theta = np.linalg.solve(chol.T, np.linalg.solve(chol, self.b))

        best_score = None
        best_expert = None
        for j in available:
            feat = self._joint_features(phi_x, int(j))
            proj = np.linalg.solve(chol, feat)
            sigma = float(np.sqrt(max(float(proj @ proj), 0.0)))
            mu = float(theta @ feat)
            score = mu - self.alpha_ucb * sigma + self.beta[int(j)]
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
        for j, ell in self._iter_observed_losses(
            losses_all=losses_all,
            available_experts=available_experts,
            selected_expert=selected_expert,
        ):
            feat = self._joint_features(phi_x, j)
            self.A += np.outer(feat, feat)
            self.b += feat * float(ell)


class LinearThompsonSampling(_SharedLinearBanditBase):
    """
    Gaussian linear Thompson sampling over joint expert-conditioned features.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        lambda_reg: float = 1.0,
        beta: np.ndarray | None = None,
        feedback_mode: str = "partial",
        context_dim: int | None = None,
        posterior_scale: float = 1.0,
        include_shared_context: bool = True,
        include_arm_bias: bool = True,
        include_arm_interactions: bool = True,
        seed: int = 0,
    ):
        self.posterior_scale = float(posterior_scale)
        if self.posterior_scale <= 0.0:
            raise ValueError("posterior_scale must be positive.")
        super().__init__(
            num_experts=num_experts,
            feature_fn=feature_fn,
            beta=beta,
            feedback_mode=feedback_mode,
            context_dim=context_dim,
            lambda_reg=lambda_reg,
            include_shared_context=include_shared_context,
            include_arm_bias=include_arm_bias,
            include_arm_interactions=include_arm_interactions,
            seed=seed,
        )

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        available = np.asarray(list(available_experts), dtype=int)
        if available.size == 0:
            raise ValueError(
                "LinearThompsonSampling: no available experts in select_expert."
            )
        phi_x = self._context_features(x)
        chol = _stable_cholesky(self.A)
        theta_mean = np.linalg.solve(chol.T, np.linalg.solve(chol, self.b))
        z = self.rng.normal(size=self.d)
        theta_sample = theta_mean + self.posterior_scale * np.linalg.solve(chol.T, z)

        best_score = None
        best_expert = None
        for j in available:
            feat = self._joint_features(phi_x, int(j))
            score = float(theta_sample @ feat) + self.beta[int(j)]
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
        for j, ell in self._iter_observed_losses(
            losses_all=losses_all,
            available_experts=available_experts,
            selected_expert=selected_expert,
        ):
            feat = self._joint_features(phi_x, j)
            self.A += np.outer(feat, feat)
            self.b += feat * float(ell)


class LinearEnsembleSampling(_SharedLinearBanditBase):
    """
    Linear ensemble sampling following Lu and Van Roy (2017).

    Each ensemble member starts from a prior sample and is updated using
    randomly perturbed observations while all members share the same
    posterior precision matrix.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        ensemble_size: int = 20,
        lambda_reg: float = 1.0,
        obs_noise_std: float = 1.0,
        beta: np.ndarray | None = None,
        feedback_mode: str = "partial",
        context_dim: int | None = None,
        include_shared_context: bool = True,
        include_arm_bias: bool = True,
        include_arm_interactions: bool = True,
        seed: int = 0,
    ):
        self.ensemble_size = int(ensemble_size)
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive.")
        self.obs_noise_std = float(obs_noise_std)
        if self.obs_noise_std <= 0.0:
            raise ValueError("obs_noise_std must be positive.")
        self.obs_noise_var = self.obs_noise_std ** 2
        super().__init__(
            num_experts=num_experts,
            feature_fn=feature_fn,
            beta=beta,
            feedback_mode=feedback_mode,
            context_dim=context_dim,
            lambda_reg=lambda_reg,
            include_shared_context=include_shared_context,
            include_arm_bias=include_arm_bias,
            include_arm_interactions=include_arm_interactions,
            seed=seed,
        )

    def reset_state(self) -> None:
        super().reset_state()
        prior_std = 1.0 / np.sqrt(self.lambda_reg)
        self.theta0_ensemble = self.rng.normal(
            loc=0.0,
            scale=prior_std,
            size=(self.ensemble_size, self.d),
        )
        self.ensemble_rhs = self.lambda_reg * np.array(self.theta0_ensemble, copy=True)
        self.ensemble_thetas = np.array(self.theta0_ensemble, copy=True)

    def _refresh_ensemble(self) -> None:
        chol = _stable_cholesky(self.A)
        y = np.linalg.solve(chol, self.ensemble_rhs.T)
        self.ensemble_thetas = np.linalg.solve(chol.T, y).T

    def select_expert(self, x: np.ndarray, available_experts: Sequence[int]) -> int:
        available = np.asarray(list(available_experts), dtype=int)
        if available.size == 0:
            raise ValueError(
                "LinearEnsembleSampling: no available experts in select_expert."
            )
        phi_x = self._context_features(x)
        member = int(self.rng.integers(self.ensemble_size))
        theta = self.ensemble_thetas[member]

        best_score = None
        best_expert = None
        for j in available:
            feat = self._joint_features(phi_x, int(j))
            score = float(theta @ feat) + self.beta[int(j)]
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
        obs_scale = 1.0 / self.obs_noise_var
        updated = False
        for j, ell in self._iter_observed_losses(
            losses_all=losses_all,
            available_experts=available_experts,
            selected_expert=selected_expert,
        ):
            feat = self._joint_features(phi_x, j)
            self.A += obs_scale * np.outer(feat, feat)
            noise = self.rng.normal(
                loc=0.0,
                scale=self.obs_noise_std,
                size=self.ensemble_size,
            )
            perturbed = float(ell) + noise
            self.ensemble_rhs += obs_scale * perturbed[:, None] * feat[None, :]
            updated = True
        if updated:
            self._refresh_ensemble()
