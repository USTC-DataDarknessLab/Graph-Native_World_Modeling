
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import SAGEConv as _PyGSAGEConv
except ImportError:
    _PyGSAGEConv = None


class _FallbackSAGEConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if edge_index.numel():
            source, destination = edge_index
            aggregate.index_add_(0, destination, x[source])
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=x.dtype).unsqueeze(1))
        aggregate = aggregate / degree.clamp_min(1.0)
        return self.lin_self(x) + self.lin_neigh(aggregate)


class GraphEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
        *,
        num_nodes: int | None = None,
        node_id_embedding_dim: int = 0,
    ):
        super().__init__()
        if node_id_embedding_dim < 0:
            raise ValueError("node_id_embedding_dim must be non-negative")
        if node_id_embedding_dim and (num_nodes is None or int(num_nodes) < 1):
            raise ValueError("num_nodes is required for a node-ID GraphEncoder feature.")
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.node_id_embedding_dim = int(node_id_embedding_dim)
        self.node_id_embedding = (
            nn.Embedding(self.num_nodes, self.node_id_embedding_dim)
            if self.node_id_embedding_dim
            else None
        )
        self.node_id_projection = (
            nn.Linear(self.node_id_embedding_dim, latent_dim, bias=False)
            if self.node_id_embedding_dim
            else None
        )
        if self.node_id_projection is not None:




            nn.init.zeros_(self.node_id_projection.weight)
        conv = _PyGSAGEConv if _PyGSAGEConv is not None else _FallbackSAGEConv





        self.conv1 = conv(input_dim, latent_dim)
        self.conv2 = conv(latent_dim, latent_dim)
        self.dropout = float(dropout)
        self.uses_pyg = _PyGSAGEConv is not None

    @staticmethod
    def _bidirectional(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        *,
        num_nodes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if edge_index.numel() == 0:
            return edge_index, edge_weight
        if edge_weight is None:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)


            return torch.unique(edge_index, dim=1), None
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must be a length-E tensor matching edge_index.")
        if num_nodes is None:
            raise ValueError("num_nodes is required to coalesce weighted edges.")
        source, destination = edge_index

        non_self = source != destination
        message_edges = torch.cat([edge_index, edge_index.flip(0)[:, non_self]], dim=1)
        message_weight = torch.cat([edge_weight, edge_weight[non_self]], dim=0)
        codes = message_edges[0] * int(num_nodes) + message_edges[1]
        unique_codes, inverse = torch.unique(codes, sorted=True, return_inverse=True)
        coalesced_weight = torch.zeros(
            unique_codes.shape[0], dtype=message_weight.dtype, device=message_weight.device
        )
        coalesced_weight.index_add_(0, inverse, message_weight)
        coalesced_edges = torch.stack(
            [unique_codes // int(num_nodes), unique_codes % int(num_nodes)], dim=0
        )
        return coalesced_edges, coalesced_weight

    @staticmethod
    def _weighted_neighbor_mean(
        x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(x)
        denominator = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if edge_index.numel() == 0:
            return aggregate
        source, destination = edge_index
        weight = edge_weight.to(dtype=x.dtype).unsqueeze(1)
        aggregate.index_add_(0, destination, x[source] * weight)
        denominator.index_add_(0, destination, weight)
        return aggregate / denominator.clamp_min(torch.finfo(x.dtype).eps)

    @staticmethod
    def _neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if edge_index.numel() == 0:
            return aggregate
        source, destination = edge_index
        aggregate.index_add_(0, destination, x[source])
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=x.dtype).unsqueeze(1))
        return aggregate / degree.clamp_min(1.0)

    @staticmethod
    def _sage_neighbor_transform_delta(
        conv: nn.Module, weighted_mean: torch.Tensor, binary_mean: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(conv, "lin_l"):
            transform = conv.lin_l
        else:
            transform = conv.lin_neigh

        return transform(weighted_mean) - transform(binary_mean)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        *,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        node_identity: torch.Tensor | None = None
        if self.node_id_embedding is not None:
            if x.shape[0] != self.num_nodes:
                raise ValueError(
                    "A node-ID GraphEncoder requires the fixed node universe used at construction."
                )
            assert self.node_id_projection is not None
            node_ids = torch.arange(x.shape[0], device=x.device)
            node_identity = self.node_id_projection(self.node_id_embedding(node_ids)).to(
                dtype=x.dtype
            )
        message_edge_index, message_edge_weight = self._bidirectional(
            edge_index, edge_weight, num_nodes=x.shape[0]
        )
        hidden = self.conv1(x, message_edge_index)
        if message_edge_weight is not None:
            hidden = hidden + self._sage_neighbor_transform_delta(
                self.conv1,
                self._weighted_neighbor_mean(x, message_edge_index, message_edge_weight),
                self._neighbor_mean(x, message_edge_index),
            )
        hidden = F.relu(hidden)
        if apply_dropout and self.training and self.dropout:
            hidden = F.dropout(hidden, p=self.dropout, training=True)
        output = self.conv2(hidden, message_edge_index)
        if message_edge_weight is not None:
            output = output + self._sage_neighbor_transform_delta(
                self.conv2,
                self._weighted_neighbor_mean(hidden, message_edge_index, message_edge_weight),
                self._neighbor_mean(hidden, message_edge_index),
            )
        if node_identity is not None:
            output = output + node_identity
        return output
