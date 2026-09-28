
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NodeStateModel(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

    def reset_history(self) -> None:
        return None

    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor | None,
        action: torch.Tensor | None = None,
        *,
        graph_context: torch.Tensor | None = None,
        commit_history: bool = True,
    ) -> torch.Tensor:



        del graph_context, commit_history
        if h_t is None:
            h_t = z_t.new_zeros((z_t.shape[0], self.hidden_dim))
        if h_t.shape != (z_t.shape[0], self.hidden_dim):
            raise ValueError(
                f"hidden state has shape {tuple(h_t.shape)}, expected {(z_t.shape[0], self.hidden_dim)}"
            )
        if self.action_dim == 0:
            if action is not None:
                raise ValueError("This passive GWM was initialized with action_dim=0.")
            state_input = z_t
        else:
            if action is None:
                action = z_t.new_zeros((z_t.shape[0], self.action_dim))
            if action.ndim == 1:
                action = action.unsqueeze(0).expand(z_t.shape[0], -1)
            if action.shape != (z_t.shape[0], self.action_dim):
                raise ValueError("action must have shape [action_dim] or [num_nodes, action_dim]")
            state_input = torch.cat([z_t, action], dim=-1)
        return self.gru(state_input, h_t)


class HistoryAwareTransformerStateModel(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int = 0,
        *,
        history_window: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if history_window < 1:
            raise ValueError("history_window must be at least 1.")
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by history_num_heads.")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.history_window = int(history_window)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)

        self.current_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.context_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.action_context_projection = (
            nn.Linear(self.action_dim, self.hidden_dim, bias=False)
            if self.action_dim
            else None
        )
        query_input_dim = self.latent_dim + self.hidden_dim + self.action_dim
        self.query_projection = nn.Linear(query_input_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relative_time_embedding = nn.Embedding(self.history_window, self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.norm_after_memory = nn.LayerNorm(self.hidden_dim)
        self.norm_after_ffn = nn.LayerNorm(self.hidden_dim)



        self.action_reinforcement = nn.Parameter(torch.zeros(()))
        self._history_contexts: list[torch.Tensor] = []
        self._history_actions: list[torch.Tensor] = []

    def reset_history(self) -> None:
        self._history_contexts.clear()
        self._history_actions.clear()

    def _coerce_action(
        self, z_t: torch.Tensor, action: torch.Tensor | None
    ) -> torch.Tensor:
        if self.action_dim == 0:
            if action is not None:
                raise ValueError("This passive GWM was initialized with action_dim=0.")
            return z_t.new_zeros((z_t.shape[0], 0))
        if action is None:
            return z_t.new_zeros((z_t.shape[0], self.action_dim))
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(z_t.shape[0], -1)
        if action.shape != (z_t.shape[0], self.action_dim):
            raise ValueError("action must have shape [action_dim] or [num_nodes, action_dim]")
        return action.to(device=z_t.device, dtype=z_t.dtype)

    def _memory_attention(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        action_nodes: torch.Tensor,
    ) -> torch.Tensor:

        q = self.query_projection(query).reshape(-1, self.num_heads, self.head_dim)
        k = self.key_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        v = self.value_projection(memory).reshape(-1, self.num_heads, self.head_dim)
        scores = torch.einsum("nhd,thd->nht", q, k) / (self.head_dim**0.5)
        if self.action_dim:
            previous_actions = torch.stack(self._history_actions, dim=0).to(
                device=query.device, dtype=query.dtype
            )
            current_norm = F.normalize(action_nodes, dim=-1, eps=1e-8)
            previous_norm = F.normalize(previous_actions, dim=-1, eps=1e-8)
            action_overlap = torch.einsum("na,ta->nt", current_norm, previous_norm)
            scores = scores + self.action_reinforcement * action_overlap[:, None, :]
        attention = torch.softmax(scores, dim=-1)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout, training=True)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(attention.dtype).eps
            )
        aggregated = torch.einsum("nht,thd->nhd", attention, v).reshape(
            query.shape[0], self.hidden_dim
        )
        return self.output_projection(aggregated)

    def forward(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor | None,
        action: torch.Tensor | None = None,
        *,
        graph_context: torch.Tensor | None = None,
        commit_history: bool = True,
    ) -> torch.Tensor:
        if h_t is None:
            h_t = z_t.new_zeros((z_t.shape[0], self.hidden_dim))
        if h_t.shape != (z_t.shape[0], self.hidden_dim):
            raise ValueError(
                f"hidden state has shape {tuple(h_t.shape)}, expected {(z_t.shape[0], self.hidden_dim)}"
            )
        action_nodes = self._coerce_action(z_t, action)
        if graph_context is None:
            graph_context = z_t.mean(dim=0)
        if graph_context.shape != (self.latent_dim,):
            raise ValueError(
                f"graph_context has shape {tuple(graph_context.shape)}, expected {(self.latent_dim,)}"
            )

        if self.action_dim:
            action_global = action_nodes.mean(dim=0)
            assert self.action_context_projection is not None
            action_context = self.action_context_projection(action_global)




            action_node_update = self.action_context_projection(action_nodes)
        else:
            action_global = z_t.new_zeros((0,))
            action_context = z_t.new_zeros((self.hidden_dim,))
            action_node_update = z_t.new_zeros((z_t.shape[0], self.hidden_dim))
        if self._history_contexts:
            contexts = torch.stack(self._history_contexts, dim=0).to(
                device=z_t.device, dtype=z_t.dtype
            )
            memory_length = contexts.shape[0]


            distances = torch.arange(
                memory_length, 0, -1, dtype=torch.long, device=z_t.device
            ).clamp_max(self.history_window) - 1
            memory = (
                self.context_projection(contexts)
                + torch.stack(
                    [
                        self.action_context_projection(history_action)
                        if self.action_context_projection is not None
                        else action_context.new_zeros((self.hidden_dim,))
                        for history_action in self._history_actions
                    ],
                    dim=0,
                ).to(device=z_t.device, dtype=z_t.dtype)
                + self.relative_time_embedding(distances)
            )
            query = torch.cat([z_t, h_t, action_nodes], dim=-1)
            memory_update = self._memory_attention(query, memory, action_nodes)
        else:
            memory_update = z_t.new_zeros((z_t.shape[0], self.hidden_dim))



        h_next = self.norm_after_memory(
            h_t + self.current_projection(z_t) + action_node_update + F.dropout(
                memory_update, p=self.dropout, training=self.training
            )
        )
        h_next = self.norm_after_ffn(h_next + self.ffn(h_next))





        if commit_history:
            self._history_contexts.append(graph_context.detach())
            self._history_actions.append(action_global.detach())
            if len(self._history_contexts) > self.history_window:
                self._history_contexts.pop(0)
                self._history_actions.pop(0)
        return h_next
