
from __future__ import annotations

from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F


def _autograd_safe_cache_tensor(value: torch.Tensor | None) -> torch.Tensor | None:

    if value is None:
        return None
    is_inference = getattr(value, "is_inference", None)
    if callable(is_inference) and bool(is_inference()):


        with torch.inference_mode(False):
            return value.detach().clone()
    return value


class _StrictMeanMessageLayer(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.self_linear = nn.Linear(dim, dim)
        self.neighbor_linear = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        if edge_index.numel():
            source, destination = edge_index
            if edge_weight is None:
                aggregate.index_add_(0, destination, x[source])
                degree.index_add_(
                    0, destination, torch.ones((destination.numel(), 1), device=x.device, dtype=x.dtype)
                )
            else:
                weight = edge_weight.to(device=x.device, dtype=x.dtype).clamp_min(0).unsqueeze(-1)
                aggregate.index_add_(0, destination, x[source] * weight)
                degree.index_add_(0, destination, weight)
        neighborhood_mean = aggregate / degree.clamp_min(1.0)
        return F.relu(self.self_linear(x) + self.neighbor_linear(neighborhood_mean))


class StrictStateAwareGraphTransformerEncoder(nn.Module):

    uses_pyg = False
    encoder_name = "strict_graph_level_sgt"
    supports_topology_cache = True

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        max_hops: int = 2,
        num_walks: int = 4,
        walk_length: int = 3,
        dropout: float = 0.1,
        use_edge_weights: bool = False,
        include_zero_hop: bool = False,
        isolated_zero_hop: bool = False,
        deterministic_walks: bool = False,
        topology_cache_max_bytes: int = 0,
        use_sgformer_global: bool = False,
    ):
        super().__init__()
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1.")
        if num_walks < 1:
            raise ValueError("num_walks must be at least 1 for strict SGT.")
        if walk_length < 1:
            raise ValueError("walk_length must be at least 1.")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.attention_scale = float(self.latent_dim) ** -0.5
        self.max_hops = int(max_hops)
        self.num_walks = int(num_walks)
        self.walk_length = int(walk_length)
        self.dropout = float(dropout)
        self.use_edge_weights = bool(use_edge_weights)
        self.include_zero_hop = bool(include_zero_hop)
        self.isolated_zero_hop = bool(isolated_zero_hop)
        self.deterministic_walks = bool(deterministic_walks)
        self.topology_cache_max_bytes = max(0, int(topology_cache_max_bytes))
        self.use_sgformer_global = bool(use_sgformer_global)
        self._topology_cache: OrderedDict[
            tuple[object, str, str],
            tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                int,
            ],
        ] = OrderedDict()
        self._topology_cache_bytes = 0
        self._topology_cache_hits = 0
        self._topology_cache_misses = 0
        self.input_projection = nn.Linear(self.input_dim, self.latent_dim)
        self.hop_layers = nn.ModuleList(
            [_StrictMeanMessageLayer(self.latent_dim) for _ in range(self.max_hops)]
        )

        self.path_projection = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        self.structure_projection = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

    def _sgformer_global_view(
        self, initial: torch.Tensor, graph_batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        query = self.structure_projection(initial).unsqueeze(1)
        key = self.path_projection(initial).unsqueeze(1)
        value = initial.unsqueeze(1)


        query = query / torch.linalg.vector_norm(query).clamp_min(1e-12)
        key = key / torch.linalg.vector_norm(key).clamp_min(1e-12)
        if graph_batch is None:
            key_value = torch.einsum("nhm,nhd->hmd", key, value)
            numerator = torch.einsum("nhm,hmd->nhd", query, key_value)
            node_count = initial.shape[0]
            numerator = numerator + node_count * value
            key_sum = key.sum(dim=0)
            denominator = torch.einsum("nhm,hm->nh", query, key_sum).unsqueeze(-1)
            denominator = denominator + node_count
        else:
            if graph_batch.ndim != 1 or graph_batch.shape[0] != initial.shape[0]:
                raise ValueError("graph_batch must contain one graph id per node")
            graph_batch = graph_batch.to(device=initial.device, dtype=torch.long)
            graph_count = int(graph_batch.max().item()) + 1 if graph_batch.numel() else 0
            key_sum = torch.zeros(
                (graph_count, key.shape[1], key.shape[2]), device=key.device, dtype=key.dtype
            )
            key_sum.index_add_(0, graph_batch, key)
            key_value = torch.zeros(
                (graph_count, key.shape[1], key.shape[2], value.shape[-1]),
                device=key.device,
                dtype=key.dtype,
            )
            key_value.index_add_(
                0, graph_batch, torch.einsum("nhm,nhd->nhmd", key, value)
            )
            numerator = torch.einsum("nhm,nhmd->nhd", query, key_value[graph_batch])
            node_count = torch.bincount(graph_batch, minlength=graph_count).to(value.dtype)
            numerator = numerator + node_count[graph_batch, None, None] * value
            denominator = torch.einsum(
                "nhm,nhm->nh", query, key_sum[graph_batch]
            ).unsqueeze(-1)
            denominator = denominator + node_count[graph_batch, None, None]
        global_view = numerator / denominator.clamp_min(1e-12)


        return 0.5 * global_view.squeeze(1) + 0.5 * initial

    @staticmethod
    def _undirected_message_edges(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        *,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if edge_index.numel() == 0:
            return edge_index, edge_weight
        if edge_weight is None:
            return torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1), None
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have one value per edge.")
        source, destination = edge_index
        non_self = source.ne(destination)
        edges = torch.cat([edge_index, edge_index.flip(0)[:, non_self]], dim=1)
        weights = torch.cat([edge_weight, edge_weight[non_self]], dim=0)
        codes = edges[0].long() * int(num_nodes) + edges[1].long()
        unique, inverse = torch.unique(codes, sorted=True, return_inverse=True)
        merged = torch.zeros(unique.numel(), device=weights.device, dtype=weights.dtype)
        merged.index_add_(0, inverse, weights)
        return torch.stack([unique // int(num_nodes), unique % int(num_nodes)]), merged

    @staticmethod
    def _csr_adjacency(
        edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return (
                torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device),
                torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device),
                edge_index.new_empty((0,)),
            )
        source, destination = edge_index
        order = torch.argsort(source)
        source, destination = source[order], destination[order]
        degree = torch.bincount(source, minlength=num_nodes)
        row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device)
        row_ptr[1:] = torch.cumsum(degree, dim=0)
        return row_ptr, degree, destination

    def _sample_walks(
        self, edge_index: torch.Tensor, num_nodes: int, *, deterministic: bool
    ) -> torch.Tensor:
        device = edge_index.device
        starts = torch.arange(num_nodes, device=device, dtype=torch.long)
        starts = starts[:, None].expand(num_nodes, self.num_walks).reshape(-1)
        row_ptr, degree, destinations = self._csr_adjacency(edge_index, num_nodes)
        current = starts
        walker_ids = torch.arange(current.numel(), device=device, dtype=torch.long)
        paths = [current]
        for step in range(self.walk_length):
            current_degree = degree[current]
            valid = current_degree > 0
            if deterministic:
                hashed = (
                    current * 1_103_515_245
                    + walker_ids * 12_345
                    + (step + 1) * 2_654_435_761
                )
                offset = torch.remainder(hashed, current_degree.clamp_min(1))
            else:
                offset = torch.floor(
                    torch.rand(current.shape, device=device) * current_degree.clamp_min(1)
                ).to(torch.long)
            next_node = current.clone()
            if bool(valid.any()):
                next_node[valid] = destinations[row_ptr[current[valid]] + offset[valid]]
            current = next_node
            paths.append(current)
        return torch.stack(paths, dim=-1).reshape(
            num_nodes, self.num_walks, self.walk_length + 1
        )

    def _forward_from_topology(
        self,
        x: torch.Tensor,
        message_edges: torch.Tensor,
        message_weight: torch.Tensor | None,
        walks: torch.Tensor,
        graph_batch: torch.Tensor | None = None,
        apply_dropout: bool = True,
    ) -> dict[str, torch.Tensor]:
        hidden = self.input_projection(x)
        zero_hop = hidden
        hop_views: list[torch.Tensor] = []
        for layer in self.hop_layers:
            hidden = layer(hidden, message_edges, message_weight)
            if apply_dropout and self.training and self.dropout:
                hidden = F.dropout(hidden, p=self.dropout, training=True)
            hop_views.append(hidden)



        base = hop_views[-1]
        path_nodes = base[walks]





        projected_nodes = self.path_projection(base)
        projected_path = projected_nodes[walks]
        projected_start = projected_nodes[:, None, None, :]
        path_scores = (projected_path * projected_start).sum(dim=-1) * self.attention_scale
        path_attention = torch.softmax(path_scores, dim=-1)
        path_views = (path_attention[..., None] * path_nodes).sum(dim=2)

        hop_candidates = torch.stack(hop_views, dim=1)
        if self.include_zero_hop:




            hop_candidates = torch.cat([zero_hop.unsqueeze(1), hop_candidates], dim=1)
        candidates = torch.cat([hop_candidates, path_views], dim=1)
        global_view = None
        if self.use_sgformer_global:



            global_view = self._sgformer_global_view(zero_hop, graph_batch=graph_batch)
            candidates = torch.cat([candidates, global_view.unsqueeze(1)], dim=1)
        projected_candidates = self.structure_projection(candidates)


        base_candidate_index = int(self.include_zero_hop) + self.max_hops - 1
        projected_base = projected_candidates[
            :, base_candidate_index : base_candidate_index + 1, :
        ]
        fusion_scores = (
            projected_candidates * projected_base
        ).sum(dim=-1) * self.attention_scale
        fusion_attention = torch.softmax(fusion_scores, dim=1)
        node_structural = (fusion_attention[..., None] * candidates).sum(dim=1)
        if self.isolated_zero_hop:






            active = torch.zeros(
                x.shape[0], dtype=torch.bool, device=x.device
            )
            if message_edges.numel():
                active.index_fill_(0, message_edges.reshape(-1).unique(), True)
            node_structural = torch.where(active.unsqueeze(-1), node_structural, zero_hop)

        graph_latent = node_structural.mean(dim=0)
        return {
            "node_structural": node_structural,
            "graph_latent": graph_latent,
            "hop_views": torch.stack(hop_views, dim=1),
            "path_views": path_views,
            "global_linear_view": (
                global_view
                if global_view is not None
                else node_structural.new_empty((0, node_structural.shape[-1]))
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        *,
        apply_dropout: bool = True,
        topology_cache_key: object | None = None,
        graph_batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:



        cache_key = None
        cached = None





        cache_walks = self.deterministic_walks or not (self.training and apply_dropout)
        if topology_cache_key is not None and self.topology_cache_max_bytes > 0:
            cache_key = (
                topology_cache_key,
                str(edge_index.device),
                "walks" if cache_walks else "topology",
            )
            cached = self._topology_cache.get(cache_key)
            if cached is not None:




                cached_edges, cached_weight, cached_walks, cached_bytes = cached
                safe_cached = (
                    _autograd_safe_cache_tensor(cached_edges),
                    _autograd_safe_cache_tensor(cached_weight),
                    _autograd_safe_cache_tensor(cached_walks),
                    cached_bytes,
                )
                if any(
                    old is not new
                    for old, new in zip(cached[:3], safe_cached[:3])
                ):
                    cached = safe_cached
                    self._topology_cache[cache_key] = cached
                self._topology_cache.move_to_end(cache_key)
                self._topology_cache_hits += 1
            else:
                self._topology_cache_misses += 1
        if cached is None:
            message_edges, message_weight = self._undirected_message_edges(
                edge_index,
                edge_weight if self.use_edge_weights else None,
                num_nodes=x.shape[0],
            )
            walks = self._sample_walks(
                message_edges,
                x.shape[0],
                deterministic=cache_walks,
            )
            if cache_key is not None:




                cached_message_edges = _autograd_safe_cache_tensor(message_edges)
                cached_message_weight = _autograd_safe_cache_tensor(message_weight)
                cached_walks = (
                    _autograd_safe_cache_tensor(walks) if cache_walks else None
                )
                tensors = [cached_message_edges]
                if cached_message_weight is not None:
                    tensors.append(cached_message_weight)
                if cache_walks:
                    tensors.append(cached_walks)
                entry_bytes = sum(
                    int(value.numel()) * int(value.element_size())
                    for value in tensors
                )
                if entry_bytes <= self.topology_cache_max_bytes:
                    while (
                        self._topology_cache
                        and self._topology_cache_bytes + entry_bytes
                        > self.topology_cache_max_bytes
                    ):
                        _old_key, old = self._topology_cache.popitem(last=False)
                        self._topology_cache_bytes -= int(old[3])
                    self._topology_cache[cache_key] = (
                        cached_message_edges,
                        cached_message_weight,
                        cached_walks,
                        entry_bytes,
                    )
                    self._topology_cache_bytes += entry_bytes
        else:
            message_edges, message_weight, cached_walks, _entry_bytes = cached
            walks = cached_walks
            if walks is None:


                walks = self._sample_walks(
                    message_edges,
                    x.shape[0],
                    deterministic=False,
                )
        return self._forward_from_topology(
            x, message_edges, message_weight, walks, graph_batch, apply_dropout
        )


class MentorNodewiseStateAwareGraphTransformerEncoder(StrictStateAwareGraphTransformerEncoder):

    uses_pyg = False
    encoder_name = "mentor_nodewise_sgt"

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        *,
        apply_dropout: bool = True,
        topology_cache_key: object | None = None,
        graph_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = super().forward(
            x,
            edge_index,
            edge_weight=edge_weight,
            apply_dropout=apply_dropout,
            topology_cache_key=topology_cache_key,
            graph_batch=graph_batch,
        )
        return outputs["node_structural"]


class MentorGWMStateAwareGraphTransformerEncoder(MentorNodewiseStateAwareGraphTransformerEncoder):

    encoder_name = "mentor_sgt_gwm"

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["use_edge_weights"] = True
        kwargs["use_sgformer_global"] = True
        super().__init__(*args, **kwargs)


class StrictHistoryAwareTransformerStateModel(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        max_relative_time: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if max_relative_time < 1:
            raise ValueError("max_relative_time must be at least 1.")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.max_relative_time = int(max_relative_time)
        self.dropout = float(dropout)

        self.z_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.query_projection = nn.Linear(self.latent_dim + self.hidden_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.relative_time_embedding = nn.Embedding(
            self.max_relative_time + 1, self.hidden_dim
        )
        self.relative_time_bias = nn.Embedding(
            self.max_relative_time + 1, self.num_heads
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self._past_latents: list[torch.Tensor] = []

    def reset_history(self) -> None:
        self._past_latents.clear()

    def _memory(self, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._past_latents:


            zero_distance = torch.zeros(1, dtype=torch.long, device=z_t.device)
            return (
                self.z_projection(z_t).unsqueeze(0)
                + self.relative_time_embedding(zero_distance),
                zero_distance,
            )
        past = torch.stack(self._past_latents, dim=0).to(device=z_t.device, dtype=z_t.dtype)
        count = past.shape[0]
        distances = torch.arange(count, 0, -1, device=z_t.device, dtype=torch.long)
        distances = distances.clamp_max(self.max_relative_time)
        return self.z_projection(past) + self.relative_time_embedding(distances), distances

    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor | None,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action is not None:
            raise ValueError("Strict passive SGT currently requires action=None.")
        if z_t.shape != (self.latent_dim,):
            raise ValueError(f"z_t must have shape {(self.latent_dim,)}, got {tuple(z_t.shape)}")
        if h_t is None:
            h_t = z_t.new_zeros((self.hidden_dim,))
        if h_t.shape != (self.hidden_dim,):
            raise ValueError(f"h_t must have shape {(self.hidden_dim,)}, got {tuple(h_t.shape)}")

        memory, distances = self._memory(z_t)
        query = self.query_projection(torch.cat([z_t, h_t], dim=-1)).reshape(
            self.num_heads, self.head_dim
        )
        keys = self.key_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        values = self.value_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        logits = torch.einsum("hd,thd->ht", query, keys) / math.sqrt(self.head_dim)
        logits = logits + self.relative_time_bias(distances).transpose(0, 1)
        attention = torch.softmax(logits, dim=-1)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout, training=True)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(attention.dtype).eps
            )
        aggregated = torch.einsum("ht,thd->hd", attention, values).reshape(self.hidden_dim)
        h_next = self.output_norm(h_t + self.ffn(self.output_projection(aggregated)))


        self._past_latents.append(z_t.detach())
        return h_next


class MentorNodewiseHistoryTransformerStateModel(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        *,
        history_window: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if history_window < 1:
            raise ValueError("history_window must be at least 1.")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_window = int(history_window)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)



        self.graph_projection = nn.Linear(self.latent_dim, self.hidden_dim)

        self.query_projection = nn.Linear(
            2 * self.latent_dim + self.hidden_dim, self.hidden_dim
        )
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)



        self.relative_time_embedding = nn.Embedding(
            self.history_window + 1, self.hidden_dim
        )
        self.relative_time_bias = nn.Embedding(
            self.history_window + 1, self.num_heads
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self._past_contexts: list[torch.Tensor] = []
    def reset_history(self) -> None:
        self._past_contexts.clear()

    def _memory(
        self, graph_context_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._past_contexts:






            distances = torch.zeros(1, dtype=torch.long, device=graph_context_t.device)
            base = graph_context_t.unsqueeze(0)
        else:
            base = torch.stack(self._past_contexts, dim=0).to(
                device=graph_context_t.device, dtype=graph_context_t.dtype
            )
            count = base.shape[0]
            distances = torch.arange(
                count, 0, -1, dtype=torch.long, device=graph_context_t.device
            ).clamp_max(self.history_window)
        memory = self.graph_projection(base) + self.relative_time_embedding(distances)
        return memory, distances

    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor | None,
        action: torch.Tensor | None = None,
        *,
        graph_context: torch.Tensor | None = None,
        commit_history: bool = True,
    ) -> torch.Tensor:
        if action is not None:
            raise ValueError("Mentor node-wise SGT currently implements passive action=None only.")
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        if h_t is None:
            h_t = z_t.new_zeros((z_t.shape[0], self.hidden_dim))
        if h_t.shape != (z_t.shape[0], self.hidden_dim):
            raise ValueError(
                f"h_t must have shape {(z_t.shape[0], self.hidden_dim)}, got {tuple(h_t.shape)}"
            )
        if graph_context is None:
            graph_context = z_t.mean(dim=0)
        if graph_context.shape != (self.latent_dim,):
            raise ValueError(
                f"graph_context must have shape {(self.latent_dim,)}, got {tuple(graph_context.shape)}"
            )

        memory, distances = self._memory(graph_context)
        query_input = torch.cat(
            [z_t, h_t, graph_context.unsqueeze(0).expand_as(z_t)], dim=-1
        )
        query = self.query_projection(query_input).reshape(
            z_t.shape[0], self.num_heads, self.head_dim
        )
        keys = self.key_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        values = self.value_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        logits = torch.einsum("nhd,thd->nht", query, keys) / math.sqrt(self.head_dim)
        logits = logits + self.relative_time_bias(distances).transpose(0, 1).unsqueeze(0)
        attention = torch.softmax(logits, dim=-1)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout, training=True)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(attention.dtype).eps
            )
        aggregated = torch.einsum("nht,thd->nhd", attention, values).reshape(
            z_t.shape[0], self.hidden_dim
        )
        h_next = self.output_norm(h_t + self.ffn(self.output_projection(aggregated)))




        if commit_history:
            self._past_contexts.append(graph_context.detach())
            if len(self._past_contexts) > self.history_window:
                self._past_contexts.pop(0)
        return h_next


class MentorActionNodewiseHistoryTransformerStateModel(
    MentorNodewiseHistoryTransformerStateModel
):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        *,
        action_dim: int = 0,
        separate_action_query: bool = False,
        bounded_action_residual: bool = False,
        zero_init_action_adapters: bool = False,
        action_adapter_initial_scale: float = 0.10,
        action_residual_max_scale: float = 0.25,
        action_adapter_squash: bool = False,
        action_adapter_temperature: float = 0.25,
        action_injection_mode: str = "full",
        history_window: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:



        super().__init__(
            latent_dim,
            hidden_dim,
            history_window=history_window,
            num_heads=num_heads,
            dropout=dropout,
        )
        if action_dim < 0:
            raise ValueError("action_dim must be non-negative.")
        self.action_dim = int(action_dim)
        self.separate_action_query = bool(separate_action_query)
        self.bounded_action_residual = bool(bounded_action_residual)
        self.zero_init_action_adapters = bool(zero_init_action_adapters)
        self.action_adapter_initial_scale = float(action_adapter_initial_scale)
        self.action_residual_max_scale = float(action_residual_max_scale)
        self.action_adapter_squash = bool(action_adapter_squash)
        self.action_adapter_temperature = float(action_adapter_temperature)
        if action_injection_mode not in {
            "full",
            "current_residual",
            "post_norm_residual",
        }:
            raise ValueError(
                "action_injection_mode must be 'full', 'current_residual', "
                "or 'post_norm_residual'."
            )
        self.action_injection_mode = str(action_injection_mode)
        if self.action_residual_max_scale <= 0.0:
            raise ValueError("action_residual_max_scale must be positive.")
        initial_scale_cap = (
            self.action_residual_max_scale
            if self.bounded_action_residual
            else 0.25
        )
        if not 0.0 <= self.action_adapter_initial_scale < initial_scale_cap:
            raise ValueError(
                "action_adapter_initial_scale must lie below the residual cap."
            )
        if self.action_adapter_temperature <= 0.0:
            raise ValueError("action_adapter_temperature must be positive.")
        self._past_node_actions: list[torch.Tensor] = []
        self.action_memory_projection: nn.Linear | None = None
        self.action_current_projection: nn.Linear | None = None
        self.action_query_projection: nn.Linear | None = None
        self.action_reinforcement: nn.Parameter | None = None
        self.action_residual_gate: nn.Parameter | None = None
        if self.action_dim:
            with torch.random.fork_rng(devices=[]):
                self.action_memory_projection = nn.Linear(
                    self.action_dim, self.hidden_dim, bias=False
                )
                self.action_current_projection = nn.Linear(
                    self.action_dim, self.hidden_dim, bias=False
                )
                self.action_query_projection = nn.Linear(
                    self.action_dim, self.hidden_dim, bias=False
                )
            self.action_reinforcement = nn.Parameter(torch.zeros(()))




            if self.bounded_action_residual:
                initial_gate = 0.0
                if self.zero_init_action_adapters:
                    assert self.action_memory_projection is not None
                    assert self.action_current_projection is not None
                    assert self.action_query_projection is not None
                    nn.init.zeros_(self.action_memory_projection.weight)
                    nn.init.zeros_(self.action_current_projection.weight)
                    nn.init.zeros_(self.action_query_projection.weight)
                    if self.action_adapter_initial_scale > 0.0:
                        initial_gate = math.atanh(
                            self.action_adapter_initial_scale
                            / self.action_residual_max_scale
                        )
                self.action_residual_gate = nn.Parameter(
                    torch.tensor(initial_gate, dtype=torch.float32)
                )
            else:
                self.action_residual_gate = nn.Parameter(torch.zeros(()))

    def action_residual_scale(self) -> torch.Tensor | float:
        if self.action_residual_gate is None:
            return 0.0
        if self.bounded_action_residual:
            return self.action_residual_max_scale * torch.tanh(
                self.action_residual_gate
            )
        return self.action_residual_gate

    def _project_action(
        self, projection: nn.Linear, action_nodes: torch.Tensor
    ) -> torch.Tensor:





        if action_nodes.ndim >= 2 and action_nodes.stride(-2) == 0:
            compact = action_nodes.select(-2, 0).unsqueeze(-2)
            value = projection(compact).expand(
                *action_nodes.shape[:-2], action_nodes.shape[-2], -1
            )
        else:
            value = projection(action_nodes)
        if self.action_adapter_squash:
            value = torch.tanh(value / self.action_adapter_temperature)
        return value

    def reset_history(self) -> None:
        super().reset_history()
        self._past_node_actions.clear()

    def _coerce_action(
        self, z_t: torch.Tensor, action: torch.Tensor | None
    ) -> torch.Tensor:
        if self.action_dim == 0:
            if action is not None:
                raise ValueError(
                    "This state model was initialized with action_dim=0; pass action=None."
                )
            return z_t.new_zeros((z_t.shape[0], 0))
        if action is None:
            return z_t.new_zeros((z_t.shape[0], self.action_dim))
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(z_t.shape[0], -1)
        if action.shape != (z_t.shape[0], self.action_dim):
            raise ValueError(
                "action must have shape [action_dim] or [num_nodes, action_dim]"
            )
        return action.to(device=z_t.device, dtype=z_t.dtype)

    def _action_memory(
        self, action_nodes_t: torch.Tensor, memory_length: int
    ) -> torch.Tensor:
        if self.action_dim == 0:
            return action_nodes_t.new_zeros((memory_length, 0))
        if self._past_contexts:
            if len(self._past_node_actions) != len(self._past_contexts):
                raise RuntimeError("Action and graph-history caches are out of sync.")
            actions = torch.stack(self._past_node_actions, dim=0).to(
                device=action_nodes_t.device, dtype=action_nodes_t.dtype
            )
        else:
            actions = action_nodes_t.unsqueeze(0)
        if actions.shape[0] != memory_length:
            raise RuntimeError("Action memory length does not match graph-memory length.")
        return actions.mean(dim=1)

    @torch.no_grad()
    def commit_observation(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        graph_context: torch.Tensor | None = None,
    ) -> None:
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        if graph_context is None:
            graph_context = z_t.mean(dim=0)
        action_nodes = self._coerce_action(z_t, action)



        self._past_contexts.append(graph_context.detach())
        self._past_node_actions.append(action_nodes.detach())
        if len(self._past_contexts) > self.history_window:
            self._past_contexts.pop(0)
            self._past_node_actions.pop(0)

    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor | None,
        action: torch.Tensor | None = None,
        *,
        graph_context: torch.Tensor | None = None,
        commit_history: bool = True,
    ) -> torch.Tensor:
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        if h_t is None:
            h_t = z_t.new_zeros((z_t.shape[0], self.hidden_dim))
        if h_t.shape != (z_t.shape[0], self.hidden_dim):
            raise ValueError(
                f"h_t must have shape {(z_t.shape[0], self.hidden_dim)}, got {tuple(h_t.shape)}"
            )
        if graph_context is None:
            graph_context = z_t.mean(dim=0)
        if graph_context.shape != (self.latent_dim,):
            raise ValueError(
                f"graph_context must have shape {(self.latent_dim,)}, got {tuple(graph_context.shape)}"
            )

        action_nodes = self._coerce_action(z_t, action)
        memory, distances = self._memory(graph_context)
        action_memory = self._action_memory(action_nodes, memory.shape[0])
        action_scale = self.action_residual_scale()
        if (
            self.action_memory_projection is not None
            and self.action_injection_mode == "full"
        ):
            memory = memory + action_scale * self._project_action(
                self.action_memory_projection, action_memory
            )

        query_input = torch.cat(
            [z_t, h_t, graph_context.unsqueeze(0).expand_as(z_t)], dim=-1
        )
        query = self.query_projection(query_input)
        if (
            self.action_query_projection is not None
            and self.action_injection_mode == "full"
        ):
            query = query + action_scale * self._project_action(
                self.action_query_projection, action_nodes
            )
        query = query.reshape(z_t.shape[0], self.num_heads, self.head_dim)
        keys = self.key_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        values = self.value_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        logits = torch.einsum("nhd,thd->nht", query, keys) / math.sqrt(self.head_dim)
        logits = logits + self.relative_time_bias(distances).transpose(0, 1).unsqueeze(0)
        if (
            self.action_reinforcement is not None
            and self.action_injection_mode == "full"
        ):
            current = F.normalize(action_nodes, dim=-1, eps=1e-8)
            past = F.normalize(
                self._action_memory(action_nodes, memory.shape[0]), dim=-1, eps=1e-8
            )
            overlap = torch.einsum("na,ta->nt", current, past)
            logits = logits + action_scale * self.action_reinforcement * overlap.unsqueeze(1)
        attention = torch.softmax(logits, dim=-1)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout, training=True)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(attention.dtype).eps
            )
        aggregated = torch.einsum("nht,thd->nhd", attention, values).reshape(
            z_t.shape[0], self.hidden_dim
        )
        action_update = 0.0
        if self.action_current_projection is not None:
            action_update = action_scale * self._project_action(
                self.action_current_projection, action_nodes
            )
        passive_update = h_t + self.ffn(self.output_projection(aggregated))
        if self.action_injection_mode == "post_norm_residual":




            h_next = self.output_norm(passive_update) + action_update
        else:
            h_next = self.output_norm(passive_update + action_update)
        if commit_history:
            self._past_contexts.append(graph_context.detach())
            self._past_node_actions.append(action_nodes.detach())
            if len(self._past_contexts) > self.history_window:
                self._past_contexts.pop(0)
                self._past_node_actions.pop(0)
        return h_next


class MentorGWMNodewiseHistoryTransformerStateModel(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        *,
        action_dim: int = 0,
        separate_action_query: bool = False,
        bounded_action_residual: bool = False,
        zero_init_action_adapters: bool = False,
        action_adapter_initial_scale: float = 0.10,
        action_residual_max_scale: float = 0.25,
        action_adapter_squash: bool = False,
        action_adapter_temperature: float = 0.25,
        action_injection_mode: str = "full",
        history_window: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if history_window < 1:
            raise ValueError("history_window must be at least one.")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.separate_action_query = bool(separate_action_query)
        self.bounded_action_residual = bool(bounded_action_residual)
        self.zero_init_action_adapters = bool(zero_init_action_adapters)
        self.action_adapter_initial_scale = float(action_adapter_initial_scale)
        self.action_residual_max_scale = float(action_residual_max_scale)
        self.action_adapter_squash = bool(action_adapter_squash)
        self.action_adapter_temperature = float(action_adapter_temperature)
        if action_injection_mode not in {
            "full",
            "current_residual",
            "post_norm_residual",
        }:
            raise ValueError(
                "action_injection_mode must be 'full', 'current_residual', "
                "or 'post_norm_residual'."
            )
        self.action_injection_mode = str(action_injection_mode)
        if self.action_dim < 0:
            raise ValueError("action_dim must be non-negative.")
        if self.action_residual_max_scale <= 0.0:
            raise ValueError("action_residual_max_scale must be positive.")
        initial_scale_cap = (
            self.action_residual_max_scale
            if self.bounded_action_residual
            else 0.25
        )
        if not 0.0 <= self.action_adapter_initial_scale < initial_scale_cap:
            raise ValueError(
                "action_adapter_initial_scale must lie below the residual cap."
            )
        if self.action_adapter_temperature <= 0.0:
            raise ValueError("action_adapter_temperature must be positive.")
        self.history_window = int(history_window)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)



        self.node_memory_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.graph_memory_projection = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
        self.current_projection = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)






        delay_action_modules = bool(self.action_dim and self.separate_action_query)
        if delay_action_modules:
            self.action_memory_projection = None
            self.action_current_projection = None
        else:
            self.action_memory_projection = (
                nn.Linear(self.action_dim, self.hidden_dim, bias=False)
                if self.action_dim
                else None
            )
            self.action_current_projection = (
                nn.Linear(self.action_dim, self.hidden_dim, bias=False)
                if self.action_dim
                else None
            )
        query_input_dim = 2 * self.latent_dim + self.hidden_dim
        self.query_projection = nn.Linear(
            query_input_dim
            + (0 if self.separate_action_query else self.action_dim),
            self.hidden_dim,
        )
        self.action_query_projection = None
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relative_time_embedding = nn.Embedding(self.history_window + 1, self.hidden_dim)
        self.relative_time_bias = nn.Embedding(self.history_window + 1, self.num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)





        self.action_reinforcement: nn.Parameter | None = (
            nn.Parameter(torch.zeros(())) if self.action_dim else None
        )
        if delay_action_modules:




            with torch.random.fork_rng(devices=[]):
                self.action_memory_projection = nn.Linear(
                    self.action_dim, self.hidden_dim, bias=False
                )
                self.action_current_projection = nn.Linear(
                    self.action_dim, self.hidden_dim, bias=False
                )
                self.action_query_projection = nn.Linear(
                    self.action_dim, self.hidden_dim, bias=False
                )
        self.action_residual_gate: nn.Parameter | None = None
        if self.action_dim and self.bounded_action_residual:





            initial_gate = 0.0
            if self.zero_init_action_adapters:







                assert self.action_memory_projection is not None
                assert self.action_current_projection is not None
                nn.init.zeros_(self.action_memory_projection.weight)
                nn.init.zeros_(self.action_current_projection.weight)
                if self.action_query_projection is not None:
                    nn.init.zeros_(self.action_query_projection.weight)
                else:
                    with torch.no_grad():
                        self.query_projection.weight[:, -self.action_dim :].zero_()
                if self.action_adapter_initial_scale > 0.0:
                    initial_gate = math.atanh(
                        self.action_adapter_initial_scale
                        / self.action_residual_max_scale
                    )
            self.action_residual_gate = nn.Parameter(
                torch.tensor(initial_gate, dtype=torch.float32)
            )
        elif self.action_dim:



            assert self.action_memory_projection is not None
            assert self.action_current_projection is not None
            nn.init.zeros_(self.action_memory_projection.weight)
            nn.init.zeros_(self.action_current_projection.weight)
            if self.action_query_projection is not None:
                nn.init.zeros_(self.action_query_projection.weight)
            else:
                with torch.no_grad():
                    self.query_projection.weight[:, -self.action_dim :].zero_()
        self._past_node_latents: list[torch.Tensor] = []
        self._past_graph_contexts: list[torch.Tensor] = []
        self._past_node_actions: list[torch.Tensor] = []
    def action_residual_scale(self) -> torch.Tensor | float:

        if self.action_residual_gate is None:
            return 1.0
        return self.action_residual_max_scale * torch.tanh(
            self.action_residual_gate
        )

    def _project_action(
        self, projection: nn.Linear, action_nodes: torch.Tensor
    ) -> torch.Tensor:
        value = projection(action_nodes)
        if self.action_adapter_squash:
            value = torch.tanh(value / self.action_adapter_temperature)
        return value

    def reset_history(self) -> None:
        self._past_node_latents.clear()
        self._past_graph_contexts.clear()
        self._past_node_actions.clear()

    def _coerce_action(
        self, z_t: torch.Tensor, action: torch.Tensor | None
    ) -> torch.Tensor:
        if self.action_dim == 0:
            if action is not None:
                raise ValueError(
                    "This state model was initialized with action_dim=0; pass action=None."
                )
            return z_t.new_zeros((z_t.shape[0], 0))
        if action is None:
            return z_t.new_zeros((z_t.shape[0], self.action_dim))
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(z_t.shape[0], -1)
        if action.shape != (z_t.shape[0], self.action_dim):
            raise ValueError(
                "action must have shape [action_dim] or [num_nodes, action_dim]"
            )
        return action.to(device=z_t.device, dtype=z_t.dtype)

    def _memory(
        self,
        z_t: torch.Tensor,
        graph_context_t: torch.Tensor,
        action_nodes_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._past_node_latents:
            node_latents = z_t.unsqueeze(0)
            graph_contexts = graph_context_t.unsqueeze(0)
            node_actions = action_nodes_t.unsqueeze(0)
            distances = torch.zeros(1, dtype=torch.long, device=z_t.device)
        else:
            node_latents = torch.stack(self._past_node_latents, dim=0).to(
                device=z_t.device, dtype=z_t.dtype
            )
            graph_contexts = torch.stack(self._past_graph_contexts, dim=0).to(
                device=z_t.device, dtype=z_t.dtype
            )
            if all(item.ndim == 1 for item in self._past_node_actions):
                compact_actions = torch.stack(
                    self._past_node_actions, dim=0
                ).to(device=z_t.device, dtype=z_t.dtype)
                node_actions = compact_actions.unsqueeze(1).expand(
                    -1, z_t.shape[0], -1
                )
            else:
                expanded_actions = [
                    item.unsqueeze(0).expand(z_t.shape[0], -1)
                    if item.ndim == 1
                    else item
                    for item in self._past_node_actions
                ]
                node_actions = torch.stack(expanded_actions, dim=0).to(
                    device=z_t.device, dtype=z_t.dtype
                )
            distances = torch.arange(
                node_latents.shape[0], 0, -1, dtype=torch.long, device=z_t.device
            ).clamp_max(self.history_window)
        memory = (
            self.node_memory_projection(node_latents)
            + self.graph_memory_projection(graph_contexts).unsqueeze(1)
            + self.relative_time_embedding(distances).unsqueeze(1)
        )
        if (
            self.action_memory_projection is not None
            and self.action_injection_mode == "full"
        ):
            memory = memory + self.action_residual_scale() * self._project_action(
                self.action_memory_projection, node_actions
            )
        return memory, distances, node_actions

    def _append_observation(
        self,
        z_t: torch.Tensor,
        graph_context_t: torch.Tensor,
        action_nodes_t: torch.Tensor,
    ) -> None:

        self._past_node_latents.append(z_t.detach())
        self._past_graph_contexts.append(graph_context_t.detach())


        if action_nodes_t.ndim == 2 and action_nodes_t.stride(0) == 0:
            self._past_node_actions.append(action_nodes_t[0].detach())
        else:
            self._past_node_actions.append(action_nodes_t.detach())
        if len(self._past_node_latents) > self.history_window:
            self._past_node_latents.pop(0)
            self._past_graph_contexts.pop(0)
            self._past_node_actions.pop(0)

    @torch.no_grad()
    def commit_observation(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        graph_context: torch.Tensor | None = None,
    ) -> None:

        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        if graph_context is None:
            graph_context = z_t.mean(dim=0)
        if graph_context.shape != (self.latent_dim,):
            raise ValueError(
                f"graph_context must have shape {(self.latent_dim,)}, got {tuple(graph_context.shape)}"
            )
        self._append_observation(
            z_t,
            graph_context,
            self._coerce_action(z_t, action),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor | None,
        action: torch.Tensor | None = None,
        *,
        graph_context: torch.Tensor | None = None,
        commit_history: bool = True,
    ) -> torch.Tensor:
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        if h_t is None:
            h_t = z_t.new_zeros((z_t.shape[0], self.hidden_dim))
        if h_t.shape != (z_t.shape[0], self.hidden_dim):
            raise ValueError(
                f"h_t must have shape {(z_t.shape[0], self.hidden_dim)}, got {tuple(h_t.shape)}"
            )
        if graph_context is None:
            graph_context = z_t.mean(dim=0)
        if graph_context.shape != (self.latent_dim,):
            raise ValueError(
                f"graph_context must have shape {(self.latent_dim,)}, got {tuple(graph_context.shape)}"
            )

        action_nodes = self._coerce_action(z_t, action)
        memory, distances, memory_actions = self._memory(
            z_t, graph_context, action_nodes
        )
        query_parts = [
            z_t,
            h_t,
            graph_context.unsqueeze(0).expand_as(z_t),
        ]
        if not self.separate_action_query:
            query_parts.append(action_nodes)
        query = self.query_projection(torch.cat(query_parts, dim=-1))
        if (
            self.action_query_projection is not None
            and self.action_injection_mode == "full"
        ):



            query = query + self.action_residual_scale() * self._project_action(
                self.action_query_projection, action_nodes
            )
        query = query.reshape(
            z_t.shape[0], self.num_heads, self.head_dim
        )
        keys = self.key_projection(memory).reshape(
            memory.shape[0], z_t.shape[0], self.num_heads, self.head_dim
        )
        values = self.value_projection(memory).reshape(
            memory.shape[0], z_t.shape[0], self.num_heads, self.head_dim
        )
        logits = torch.einsum("nhd,tnhd->nht", query, keys) / math.sqrt(self.head_dim)
        logits = logits + self.relative_time_bias(distances).transpose(0, 1).unsqueeze(0)
        if (
            self.action_dim
            and self.action_reinforcement is not None
            and self.action_injection_mode == "full"
        ):
            if action_nodes.stride(0) == 0 and memory_actions.stride(1) == 0:
                current_action = F.normalize(action_nodes[0], dim=-1, eps=1e-8)
                past_actions = F.normalize(
                    memory_actions[:, 0, :], dim=-1, eps=1e-8
                )
                action_overlap = torch.einsum(
                    "a,ta->t", current_action, past_actions
                ).unsqueeze(0).expand(z_t.shape[0], -1)
            else:
                current_action = F.normalize(action_nodes, dim=-1, eps=1e-8)
                past_actions = F.normalize(memory_actions, dim=-1, eps=1e-8)
                action_overlap = torch.einsum(
                    "na,tna->nt", current_action, past_actions
                )
            logits = logits + (
                self.action_residual_scale()
                * self.action_reinforcement
                * action_overlap.unsqueeze(1)
            )
        attention = torch.softmax(logits, dim=-1)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout, training=True)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(attention.dtype).eps
            )
        aggregated = torch.einsum("nht,tnhd->nhd", attention, values).reshape(
            z_t.shape[0], self.hidden_dim
        )
        current_update = self.current_projection(z_t)
        action_update = 0.0
        if self.action_current_projection is not None:
            action_update = (
                self.action_residual_scale()
                * self._project_action(self.action_current_projection, action_nodes)
            )
        passive_update = (
            h_t + current_update + self.ffn(self.output_projection(aggregated))
        )
        if self.action_injection_mode == "post_norm_residual":




            h_next = self.output_norm(passive_update) + action_update
        else:
            h_next = self.output_norm(passive_update + action_update)



        if commit_history:
            self._append_observation(z_t, graph_context, action_nodes)
        return h_next
