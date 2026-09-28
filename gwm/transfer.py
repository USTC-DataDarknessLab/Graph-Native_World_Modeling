
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GatedResidualTransferAdapter(nn.Module):

    def __init__(
        self,
        dimension: int,
        *,
        bottleneck: int = 16,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if dimension < 1 or bottleneck < 1:
            raise ValueError("Transfer-adapter dimensions must be positive.")
        if not 0.0 < float(initial_gate) < 1.0:
            raise ValueError("Transfer-adapter initial gate must lie in (0, 1).")
        self.normalization = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(initial_gate / (1.0 - initial_gate)))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.normalization(features))))
        return features + torch.sigmoid(self.gate_logit) * residual

    def gate_value(self) -> float:
        return float(torch.sigmoid(self.gate_logit.detach()).cpu())


class FrozenGraphTransferBranch(nn.Module):

    def __init__(
        self,
        input_adapter: nn.Module,
        graph_encoder: nn.Module,
        dimension: int,
        *,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("Transfer dimension must be positive.")
        if not 0.0 < float(initial_gate) < 1.0:
            raise ValueError("Transfer gate must lie in (0, 1).")
        self.input_adapter = input_adapter.requires_grad_(False)
        self.graph_encoder = graph_encoder.requires_grad_(False)
        self.normalization = nn.LayerNorm(dimension)
        self.projection = nn.Linear(dimension, dimension)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(initial_gate / (1.0 - initial_gate)))
        )
        self.input_adapter.eval()
        self.graph_encoder.eval()

    def train(self, mode: bool = True) -> "FrozenGraphTransferBranch":
        super().train(mode)


        self.input_adapter.eval()
        self.graph_encoder.eval()
        return self

    def forward(
        self,
        online: torch.Tensor,
        x_t: torch.Tensor,
        edge_index_t: torch.Tensor,
        edge_weight_t: torch.Tensor | None,
        *,
        topology_cache_key: object | None = None,
    ) -> torch.Tensor:
        encoder_kwargs: dict[str, object] = {
            "edge_weight": edge_weight_t,
            "apply_dropout": False,
        }
        if bool(getattr(self.graph_encoder, "supports_topology_cache", False)):
            encoder_kwargs["topology_cache_key"] = (
                "pretrained-transfer",
                topology_cache_key,
            )
        with torch.no_grad():
            source_input = self.input_adapter(x_t)
            source = self.graph_encoder(source_input, edge_index_t, **encoder_kwargs)
        residual = self.projection(self.normalization(source))
        return online + torch.sigmoid(self.gate_logit) * residual

    def gate_value(self) -> float:
        return float(torch.sigmoid(self.gate_logit.detach()).cpu())


__all__ = ["FrozenGraphTransferBranch", "GatedResidualTransferAdapter"]
