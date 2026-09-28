
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def latent_transition_terms(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    cosine_weight: float = 0.0,
    variance_weight: float = 0.0,
    node_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if cosine_weight < 0 or variance_weight < 0:
        raise ValueError("Latent cosine/variance weights must be non-negative.")
    if target.shape != outputs["latent_mu"].shape:
        raise ValueError("Future latent target and predicted mean must have identical shapes.")
    if node_weight is not None:
        if node_weight.ndim != 1 or node_weight.shape[0] != target.shape[0]:
            raise ValueError(
                "node_weight must have shape [num_nodes] matching the latent target."
            )
        if bool((node_weight < 0).any()):
            raise ValueError("node_weight must be non-negative.")
        node_weight = node_weight.to(device=target.device, dtype=target.dtype)
        normalizer = node_weight.sum().clamp_min(1.0)

        def _node_mean(value: torch.Tensor) -> torch.Tensor:
            return (value.mean(dim=-1) * node_weight).sum() / normalizer

    else:

        def _node_mean(value: torch.Tensor) -> torch.Tensor:
            return value.mean()

    if outputs.get("latent_is_gaussian", False):
        logvar = outputs["latent_logvar"]
        pointwise = _node_mean(0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (target - outputs["latent_mu"]).square() * torch.exp(-logvar)
        ))
        variance = _node_mean(F.relu(logvar).square())
    else:
        pointwise = _node_mean((outputs["latent_mu"] - target).square())
        variance = pointwise.new_zeros(())

    cosine_per_node = 1.0 - F.cosine_similarity(
        outputs["latent_mu"], target, dim=-1, eps=1e-6
    )
    if node_weight is None:
        cosine = cosine_per_node.mean()
    else:
        cosine = (cosine_per_node * node_weight).sum() / normalizer
    total = pointwise + float(cosine_weight) * cosine + float(variance_weight) * variance
    return {
        "total": total,
        "pointwise": pointwise,
        "cosine": cosine,
        "variance": variance,
    }
