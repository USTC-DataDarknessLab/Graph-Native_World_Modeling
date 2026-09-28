
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LatentPredictor(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        distribution: str = "gaussian",
        min_logvar: float = -8.0,
        max_logvar: float = 5.0,
        normalization: str = "none",
        transition_mode: str = "absolute",
    ):
        super().__init__()
        if distribution not in {"gaussian", "deterministic"}:
            raise ValueError("distribution must be gaussian or deterministic")
        if normalization not in {"none", "layernorm"}:
            raise ValueError("normalization must be none or layernorm")
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar")
        if transition_mode not in {"absolute", "residual_hidden"}:
            raise ValueError("transition_mode must be absolute or residual_hidden")
        self.distribution = distribution
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.normalization = normalization
        self.transition_mode = transition_mode
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.residual_base = (
            None
            if transition_mode == "absolute"
            else (nn.Identity() if hidden_dim == latent_dim else nn.Linear(hidden_dim, latent_dim))
        )
        if transition_mode == "residual_hidden":


            nn.init.zeros_(self.mu_head.weight)
            nn.init.zeros_(self.mu_head.bias)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim) if distribution == "gaussian" else None

    def forward(
        self, h_next: torch.Tensor, *, sample: bool | None = None
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(h_next)
        delta_mu_raw = self.mu_head(features)
        if self.transition_mode == "residual_hidden":
            assert self.residual_base is not None
            raw_mu = self.residual_base(h_next) + delta_mu_raw
        else:
            raw_mu = delta_mu_raw
        mu = (
            F.layer_norm(raw_mu, (raw_mu.shape[-1],))
            if self.normalization == "layernorm"
            else raw_mu
        )
        if self.distribution == "deterministic":
            return {
                "raw_mu": raw_mu,
                "delta_mu_raw": delta_mu_raw,
                "mu": mu,
                "logvar": torch.zeros_like(mu),
                "sample": mu,
            }
        assert self.logvar_head is not None
        logvar = self.logvar_head(features).clamp(self.min_logvar, self.max_logvar)
        if sample is None:
            sample = self.training
        if sample:
            z_next = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z_next = mu
        return {
            "raw_mu": raw_mu,
            "delta_mu_raw": delta_mu_raw,
            "mu": mu,
            "logvar": logvar,
            "sample": z_next,
        }

    @staticmethod
    def gaussian_nll(
        target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        inverse_variance = torch.exp(-logvar)
        return 0.5 * (
            math.log(2.0 * math.pi) + logvar + (target - mu).square() * inverse_variance
        ).mean()
