"""
neural_SSM.py

This file contains a PyTorch implementation of a neural, uncertainty-aware,
similarity-exploiting router inspired by NN_Time_series_overleaf.tex.

It is designed as a starting point for a faithful implementation of the paper's
model, but it is NOT wired into the existing NumPy router pipeline by default.
You should treat this as a standalone module to train and evaluate in a
PyTorch environment.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _resolve_device(device: str | None = None) -> torch.device:
    """Choose a torch device with auto fallback to CUDA → MPS → CPU."""
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = str(device).lower()
    if dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device(dev)


class SetEncoder(nn.Module):
    """
    DeepSets-style set encoder for available experts:
        s_t = rho( sum_j phi( e_j, v_j, stats_j ) )
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, per_expert: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        per_expert: (N, in_dim)
        mask: (N,) boolean or {0,1}, 1 = included in set.
        """
        if per_expert.numel() == 0:
            return torch.zeros(self.rho[-1].out_features, device=per_expert.device)
        h = self.phi(per_expert)  # (N, hidden_dim)
        mask = mask.float().unsqueeze(-1)  # (N, 1)
        h = h * mask
        h_sum = h.sum(dim=0)
        return self.rho(h_sum)


class GammaHead(nn.Module):
    """
    Mean head for positive Gamma likelihood:
        μ_{j,t} = softplus( w^T f_{j,t} ) + eps
    Shape parameter k is fixed (scalar) for now.
    """

    def __init__(self, in_dim: int, gamma_shape: float = 10.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.gamma_shape = float(gamma_shape)
        self.eps = 1e-6

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (..., in_dim)
        returns:  μ (...,)
        """
        logits = self.linear(features).squeeze(-1)
        mu = F.softplus(logits) + self.eps
        return mu

    def log_prob(self, ell: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Gamma(k, θ = μ/k) log-density at ell >= 0:
            log p(ℓ | μ) = (k-1) log ℓ - k ℓ / μ - k log(μ/k) - log Γ(k)
        """
        k = self.gamma_shape
        ell = torch.clamp(ell, min=self.eps)
        mu = torch.clamp(mu, min=self.eps)
        theta = mu / k
        log_p = (
            (k - 1.0) * torch.log(ell)
            - ell / theta
            - k * torch.log(theta)
            - torch.lgamma(torch.tensor(k, device=ell.device))
        )
        return log_p

    def gamma_score_log_mu(self, ell: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Score wrt log μ for Gamma(k, μ):
            δ = k (ℓ / μ - 1)
        """
        k = self.gamma_shape
        mu = torch.clamp(mu, min=self.eps)
        return k * (ell / mu - 1.0)


class NeuralSSMPaper(nn.Module):
    """
    Neural switching latent router (paper-style skeleton).

    This module implements:
      - deterministic belief state h_t via GRU,
      - discrete regimes z_t with prior p(z_t | h_{t-1}) and
        variational posterior q(z_t | h_t),
      - continuous factor g_t with Gaussian prior/posterior,
      - Gamma MSE likelihood for the chosen expert,
      - DeepSets-style set encoder for available experts.

    It computes a per-step masked ELBO term:
        E_q[ log p(ℓ_{r_t,t} | z_t, g_t, h_t) ]
        - KL( q(z_t, g_t | H_t) || p(z_t, g_t | h_{t-1}) )

    NOTE:
      - This is a best-effort, reasonably faithful skeleton, but you must
        integrate it with your own training loop and verify behavior.
      - It omits some details from the text (e.g., explicit expert
        memory broadcast), which you can add on top.
    """

    def __init__(
        self,
        num_experts: int,
        num_regimes: int,
        d_x: int = 1,
        d_phi: int = 32,
        d_h: int = 64,
        d_g: int = 16,
        d_e: int = 16,
        d_v: int = 8,
        gamma_shape: float = 10.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_regimes = num_regimes
        self.d_x = d_x
        self.d_phi = d_phi
        self.d_h = d_h
        self.d_g = d_g
        self.d_e = d_e
        self.d_v = d_v

        # Expert embeddings e_j
        self.expert_embed = nn.Embedding(num_experts, d_e)

        # Simple context encoder φ_η(x_t)
        self.phi_net = nn.Sequential(
            nn.Linear(d_x, d_phi),
            nn.ReLU(),
            nn.Linear(d_phi, d_phi),
            nn.ReLU(),
        )

        # Memory states v_{j,t} will be maintained externally (per sequence)

        # Set encoder for available experts: inputs [e_j, v_j, stats_j]
        set_in_dim = d_e + d_v + 2
        self.set_encoder = SetEncoder(
            in_dim=set_in_dim,
            hidden_dim=32,
            out_dim=32,
        )

        # Belief RNN f_ψ: input [φ_t, s_t, e_{r_{t-1}}, ℓ_{r_{t-1},t-1}]
        rnn_in_dim = d_phi + 32 + d_e + 1
        self.gru = nn.GRU(input_size=rnn_in_dim, hidden_size=d_h, batch_first=False)

        # Prior for z_t: p(z_t | h_{t-1})
        self.prior_z_net = nn.Linear(d_h, num_regimes)
        # Variational posterior q(z_t | h_t)
        self.post_z_net = nn.Linear(d_h, num_regimes)

        # Prior for g_t: p(g_t | z_t, h_{t-1})
        self.prior_g_net = nn.Linear(d_h + num_regimes, 2 * d_g)
        # Posterior q(g_t | z_t, h_t)
        self.post_g_net = nn.Linear(d_h + num_regimes, 2 * d_g)

        # Gamma mean head μ_θ(g_t, h_t, e_j, stats_j, v_j)
        gamma_in_dim = d_g + d_h + d_e + d_v + 2
        self.gamma_head = GammaHead(gamma_in_dim, gamma_shape=gamma_shape)

    def init_belief(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Initialize h_0 (shape [1, 1, d_h]) for GRU.
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, 1, self.d_h, device=device)

    @torch.no_grad()
    def predict_mu_all(
        self,
        x_t: torch.Tensor,
        avail_mask_t: torch.Tensor,
        r_prev: Optional[int],
        ell_prev: Optional[float],
        h_prev: torch.Tensor,
        v_prev: torch.Tensor,
        n_selected: torch.Tensor,
        last_seen: torch.Tensor,
        t_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Gamma means μ_{j,t} for all experts j using current belief
        state h_prev and context x_t, without consuming the current loss.

        Returns:
          mu_all_j : (N,) tensor of means for all experts
          h_t      : updated belief state after processing x_t
        """
        device = x_t.device
        N = self.num_experts

        # Context embedding
        x_t = x_t.view(1, -1)
        phi_t = self.phi_net(x_t).view(-1)  # (d_phi,)

        # Stats per expert
        t_scalar = float(max(t_index, 1))
        count_norm = n_selected.float() / t_scalar
        stale = torch.where(
            last_seen < 0,
            torch.full_like(last_seen, t_scalar),
            (t_scalar - last_seen.float()),
        )
        stale_norm = stale / t_scalar
        stats = torch.stack([count_norm, stale_norm], dim=-1)  # (N,2)

        # Set encoder over available experts
        e_all = self.expert_embed(torch.arange(N, device=device))
        per_expert_feats = torch.cat([e_all, v_prev, stats], dim=-1)
        s_t = self.set_encoder(per_expert_feats, avail_mask_t)  # (d_s,)

        # Previous chosen expert embedding and loss
        if r_prev is None:
            e_prev = torch.zeros(self.d_e, device=device)
            ell_prev_tensor = torch.zeros(1, device=device)
        else:
            e_prev = self.expert_embed(torch.tensor([r_prev], device=device)).view(-1)
            ell_prev_tensor = torch.tensor([ell_prev], device=device)

        # RNN input and update
        rnn_in = torch.cat([phi_t, s_t, e_prev, ell_prev_tensor], dim=-1)
        rnn_in = rnn_in.view(1, 1, -1)
        _, h_t = self.gru(rnn_in, h_prev)
        h_t_flat = h_t.view(-1)

        # Posterior over z_t based on h_t
        logit_q_z = self.post_z_net(h_t_flat)
        log_q_z = F.log_softmax(logit_q_z, dim=-1)
        q_z = log_q_z.exp()  # (M,)

        # Posterior over g_t for each regime; take E_q[g_t]
        eye_M = torch.eye(self.num_regimes, device=device)
        mean_q_list = []
        for k in range(self.num_regimes):
            onehot_k = eye_M[k]
            post_in = torch.cat([h_t_flat, onehot_k], dim=-1)
            pq = self.post_g_net(post_in)
            mean_q_k, _ = pq[: self.d_g], pq[self.d_g :]
            mean_q_list.append(mean_q_k)
        mean_q = torch.stack(mean_q_list, dim=0)  # (M,d_g)
        g_mean = (q_z.view(-1, 1) * mean_q).sum(dim=0)  # (d_g,)

        # Compute μ_j for all experts
        mu_all_j = torch.zeros(N, device=device)
        for j in range(N):
            e_j = e_all[j]
            v_j = v_prev[j]
            stats_j = stats[j]
            gamma_in = torch.cat([g_mean, h_t_flat, e_j, v_j, stats_j], dim=-1)
            mu_j = self.gamma_head(gamma_in.view(1, -1)).view(())
            mu_all_j[j] = mu_j

        return mu_all_j, h_t

    def _kl_categorical(
        self, log_q: torch.Tensor, log_p: torch.Tensor
    ) -> torch.Tensor:
        """
        KL( q || p ) for categorical distributions in log-space.
        log_q, log_p: (..., K)
        """
        q = log_q.exp()
        return torch.sum(q * (log_q - log_p), dim=-1)

    def _kl_gaussian(
        self,
        mean_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mean_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL between diagonal Gaussians N(mean_q, var_q) || N(mean_p, var_p).
        """
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        term = (
            logvar_p
            - logvar_q
            + (var_q + (mean_q - mean_p) ** 2) / var_p
            - 1.0
        )
        return 0.5 * torch.sum(term, dim=-1)

    def step(
        self,
        x_t: torch.Tensor,
        avail_mask_t: torch.Tensor,
        r_prev: Optional[int],
        ell_prev: Optional[float],
        h_prev: torch.Tensor,
        v_prev: torch.Tensor,
        n_selected: torch.Tensor,
        last_seen: torch.Tensor,
        t_index: int,
        r_t: int,
        ell_obs_t: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        One time-step update and ELBO contribution.

        Inputs:
          x_t           : (d_x,) context at time t
          avail_mask_t  : (N,) bool or {0,1} for availability
          r_prev, ell_prev : previous chosen expert and loss (or None)
          h_prev        : (1, 1, d_h) previous GRU state
          v_prev        : (N, d_v) expert memory states
          n_selected    : (N,) counts up to t-1
          last_seen     : (N,) last seen timestep indices
          t_index       : int time index (for staleness stats)
          r_t           : current chosen expert index
          ell_obs_t     : observed loss for r_t at time t

        Returns:
          elbo_t        : scalar ELBO contribution at time t
          aux           : dict with updated h_t, v_t, etc.
        """
        device = x_t.device
        N = self.num_experts

        # Context embedding
        x_t = x_t.view(1, -1)
        phi_t = self.phi_net(x_t).view(-1)  # (d_phi,)

        # Stats per expert: [norm count, norm staleness]
        t_scalar = float(max(t_index, 1))
        count_norm = n_selected.float() / t_scalar
        stale = torch.where(
            last_seen < 0,
            torch.full_like(last_seen, t_scalar),
            (t_scalar - last_seen.float()),
        )
        stale_norm = stale / t_scalar
        stats = torch.stack([count_norm, stale_norm], dim=-1)  # (N,2)

        # Set encoder over available experts
        e_all = self.expert_embed(torch.arange(N, device=device))
        per_expert_feats = torch.cat([e_all, v_prev, stats], dim=-1)
        s_t = self.set_encoder(per_expert_feats, avail_mask_t)  # (d_s,)

        # Previous chosen expert embedding and loss
        if r_prev is None:
            e_prev = torch.zeros(self.d_e, device=device)
            ell_prev_tensor = torch.zeros(1, device=device)
        else:
            e_prev = self.expert_embed(torch.tensor([r_prev], device=device)).view(-1)
            ell_prev_tensor = torch.tensor([ell_prev], device=device)

        # RNN input and update
        rnn_in = torch.cat([phi_t, s_t, e_prev, ell_prev_tensor], dim=-1)
        rnn_in = rnn_in.view(1, 1, -1)  # (seq=1, batch=1, rnn_in_dim)
        _, h_t = self.gru(rnn_in, h_prev)  # h_t: (1,1,d_h)
        h_t_flat = h_t.view(-1)  # (d_h,)

        # Prior for z_t: p(z_t | h_{t-1})
        logit_p_z = self.prior_z_net(h_prev.view(-1))
        log_p_z = F.log_softmax(logit_p_z, dim=-1)

        # Posterior q(z_t | h_t)
        logit_q_z = self.post_z_net(h_t_flat)
        log_q_z = F.log_softmax(logit_q_z, dim=-1)
        q_z = log_q_z.exp()

        # One-hot over regimes for each component
        eye_M = torch.eye(self.num_regimes, device=device)

        # KL for z_t
        kl_z = self._kl_categorical(log_q=log_q_z, log_p=log_p_z)

        # Gaussian prior/posterior for g_t for each regime k
        mean_q_list = []
        logvar_q_list = []
        mean_p_list = []
        logvar_p_list = []

        for k in range(self.num_regimes):
            onehot_k = eye_M[k]
            # Posterior q(g_t | z_t=k, h_t)
            post_in = torch.cat([h_t_flat, onehot_k], dim=-1)
            pq = self.post_g_net(post_in)
            mean_q_k, logvar_q_k = pq[: self.d_g], pq[self.d_g :]
            mean_q_list.append(mean_q_k)
            logvar_q_list.append(logvar_q_k)

            # Prior p(g_t | z_t=k, h_{t-1})
            prior_in = torch.cat([h_prev.view(-1), onehot_k], dim=-1)
            pp = self.prior_g_net(prior_in)
            mean_p_k, logvar_p_k = pp[: self.d_g], pp[self.d_g :]
            mean_p_list.append(mean_p_k)
            logvar_p_list.append(logvar_p_k)

        mean_q = torch.stack(mean_q_list, dim=0)  # (M,d_g)
        logvar_q = torch.stack(logvar_q_list, dim=0)
        mean_p = torch.stack(mean_p_list, dim=0)
        logvar_p = torch.stack(logvar_p_list, dim=0)

        # KL for g_t: E_{q(z_t)} KL(q(g_t|z_t) || p(g_t|z_t))
        kl_g_per_k = self._kl_gaussian(mean_q, logvar_q, mean_p, logvar_p)  # (M,)
        kl_g = torch.sum(q_z * kl_g_per_k)

        # Sample z_t and g_t for the likelihood
        with torch.no_grad():
            z_idx = torch.multinomial(q_z, num_samples=1).item()
        mean_q_sel = mean_q[z_idx]
        logvar_q_sel = logvar_q[z_idx]
        eps = torch.randn_like(mean_q_sel)
        g_t = mean_q_sel + torch.exp(0.5 * logvar_q_sel) * eps  # (d_g,)

        # Gamma mean μ_{r_t,t}
        r_t_tensor = torch.tensor([r_t], device=device)
        e_r = self.expert_embed(r_t_tensor).view(-1)
        stats_r = stats[r_t]
        v_r = v_prev[r_t]

        gamma_in = torch.cat([g_t, h_t_flat, e_r, v_r, stats_r], dim=-1)
        mu_r = self.gamma_head(gamma_in.view(1, -1)).view(())

        # Likelihood term (masked ELBO uses only ℓ_{r_t,t})
        ell_t_tensor = torch.tensor(ell_obs_t, device=device)
        log_like = self.gamma_head.log_prob(ell_t_tensor, mu_r)

        # ELBO contribution
        elbo_t = log_like - kl_z - kl_g

        aux: Dict = {
            "h_t": h_t,
            "g_t": g_t,
            "mu_r": mu_r,
            "log_like": log_like,
            "kl_z": kl_z,
            "kl_g": kl_g,
            "elbo_t": elbo_t,
        }
        return elbo_t, aux


class NeuralSSMRouter:
    """
    Thin wrapper around NeuralSSMPaper providing the same interface as
    the NumPy SLDS routers:

      - select_expert(x_t, available_experts) -> (r_t, cache)
      - update_beliefs(r_t, loss_obs, losses_full, available_experts, cache)

    This wrapper assumes the underlying NeuralSSMPaper is already
    initialized (and ideally trained). It does not perform parameter
    updates by default; it only maintains the belief state and expert
    statistics.
    """

    def __init__(
        self,
        num_experts: int,
        feature_fn,  # unused; kept for API compatibility
        num_regimes: int = 1,
        beta=None,
        lambda_risk: float = 0.0,
        feedback_mode: str = "partial",
        device: str = "auto",
        **kwargs,
    ):
        assert feedback_mode in ("partial", "full")
        self.N = int(num_experts)
        self.M = int(num_regimes)
        self.feedback_mode = feedback_mode
        self.lambda_risk = float(lambda_risk)

        if beta is None:
            import numpy as np

            beta = np.zeros(self.N, dtype=float)
        self.beta = beta

        self.device = _resolve_device(device)

        # Optional Gamma shape from kwargs (for compatibility with config)
        gamma_shape = float(kwargs.get("gamma_shape", 10.0))
        learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.model = NeuralSSMPaper(
            num_experts=self.N,
            num_regimes=self.M,
            d_x=1,
            gamma_shape=gamma_shape,
        ).to(self.device)
        self.model.train()

        # Simple online optimizer (per-step updates in update_beliefs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Belief and memory state
        self.h = self.model.init_belief(self.device)  # (1,1,d_h)
        self.v = torch.zeros(self.N, self.model.d_v, device=self.device)
        self.n_selected = torch.zeros(self.N, device=self.device)
        self.last_seen = -torch.ones(self.N, device=self.device)
        self.t = 0
        self.r_prev: Optional[int] = None
        self.ell_prev: Optional[float] = None

    def reset_beliefs(self):
        self.h = self.model.init_belief(self.device)
        self.v.zero_()
        self.n_selected.zero_()
        self.last_seen.fill_(-1)
        self.t = 0
        self.r_prev = None
        self.ell_prev = None

    def select_expert(self, x_t, available_experts):
        import numpy as np

        x_arr = np.asarray(x_t, dtype=float).reshape(-1)
        assert x_arr.size == 1, "NeuralSSMRouter currently assumes 1D context."
        x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=self.device)

        avail = np.asarray(list(available_experts), dtype=int)
        if avail.size == 0:
            avail = np.arange(self.N, dtype=int)
        avail_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        avail_mask[avail] = True

        # Save previous belief state for training step
        h_prev_for_step = self.h.clone()

        mu_all_j, h_t_new = self.model.predict_mu_all(
            x_t=x_tensor,
            avail_mask_t=avail_mask,
            r_prev=self.r_prev,
            ell_prev=self.ell_prev,
            h_prev=self.h,
            v_prev=self.v,
            n_selected=self.n_selected,
            last_seen=self.last_seen,
            t_index=self.t,
        )

        self.h = h_t_new

        # Risk-neutral score: μ_j + β_j
        scores = mu_all_j.detach().cpu().numpy() + self.beta
        scores_avail = scores[avail]
        idx = int(np.argmin(scores_avail))
        r_t = int(avail[idx])

        cache = {
            "x_tensor": x_tensor,
            "avail_mask": avail_mask,
            "h_prev": h_prev_for_step,
            "x_t": x_arr,
            "avail": avail,
            "mu_all_j": mu_all_j.detach().cpu().numpy(),
        }
        return r_t, cache

    def update_beliefs(
        self,
        r_t: int,
        loss_obs: float,
        losses_full,
        available_experts,
        cache,
    ):
        import numpy as np

        # Online training step using one-step ELBO if possible
        if self.optimizer is not None:
            x_tensor = cache.get("x_tensor", None)
            avail_mask = cache.get("avail_mask", None)
            h_prev = cache.get("h_prev", None)
            if (
                x_tensor is not None
                and avail_mask is not None
                and h_prev is not None
            ):
                x_tensor = x_tensor.to(self.device)
                avail_mask = avail_mask.to(self.device)
                h_prev = h_prev.to(self.device)
                ell_obs = float(loss_obs)
                elbo_t, _ = self.model.step(
                    x_t=x_tensor,
                    avail_mask_t=avail_mask,
                    r_prev=self.r_prev,
                    ell_prev=self.ell_prev,
                    h_prev=h_prev,
                    v_prev=self.v,
                    n_selected=self.n_selected,
                    last_seen=self.last_seen,
                    t_index=self.t,
                    r_t=int(r_t),
                    ell_obs_t=ell_obs,
                )
                loss = -elbo_t
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.t += 1
        avail = np.asarray(list(available_experts), dtype=int)
        if avail.size == 0:
            avail = np.arange(self.N, dtype=int)
        avail_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        avail_mask[avail] = True
        self.last_seen[avail_mask] = float(self.t)
        self.n_selected[r_t] += 1.0
        self.r_prev = int(r_t)
        self.ell_prev = float(loss_obs)
        # Memory v is not updated here; you can extend this wrapper to
        # use a graph-based broadcast scheme if desired.

    def plan_horizon_schedule(
        self,
        x_t,
        H: int,
        experts_predict: Sequence[Callable[[object], float]],
        context_update: Callable[[object, float], object],
        available_experts_per_h: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[object], List[float]]:
        """
        Horizon-H planning: mimic SLDS routers by producing a greedy
        open-loop schedule (r_{t+1},...,r_{t+H}) using the current
        belief state, without permanently mutating it.
        """
        import numpy as np

        if available_experts_per_h is None:
            available_experts_per_h = [list(range(self.N)) for _ in range(H)]
        assert len(available_experts_per_h) == H

        # Save current internal state
        h_saved = self.h.clone()
        v_saved = self.v.clone()
        n_saved = self.n_selected.clone()
        last_seen_saved = self.last_seen.clone()
        t_saved = self.t
        r_prev_saved = self.r_prev
        ell_prev_saved = self.ell_prev

        # Work on a local copy of context
        x_curr = np.asarray(x_t, dtype=float).reshape(-1)
        schedule: List[int] = []
        contexts: List[object] = []
        scores: List[float] = []

        for h_step in range(H):
            avail_list = list(available_experts_per_h[h_step])
            r_future, cache = self.select_expert(x_curr, avail_list)
            schedule.append(int(r_future))

            # Approximate score using predicted mean + beta
            mu_all_j = cache.get("mu_all_j", None)
            if mu_all_j is not None:
                score_val = float(mu_all_j[int(r_future)] + self.beta[int(r_future)])
            else:
                score_val = 0.0
            scores.append(score_val)

            # Propagate context using provided expert predictors
            y_hat = experts_predict[int(r_future)](x_curr)
            x_next = context_update(x_curr, float(y_hat))
            contexts.append(x_next)
            x_curr = np.asarray(x_next, dtype=float).reshape(-1)

        # Restore original state
        self.h = h_saved
        self.v = v_saved
        self.n_selected = n_saved
        self.last_seen = last_seen_saved
        self.t = t_saved
        self.r_prev = r_prev_saved
        self.ell_prev = ell_prev_saved

        return schedule, contexts, scores
