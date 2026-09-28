
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli, Categorical, Normal

from .actions import GraphEditAction, GraphEditSequence, GraphOperation


def calibrated_soft_node_action_probability(
    logits: torch.Tensor,
    *,
    historical_positive_count: torch.Tensor | float,
    historical_observation_count: torch.Tensor | float,
    mode: str = "none",
    smoothing: float = 1.0,
) -> torch.Tensor:
    if mode not in {"none", "past_global_prior", "past_global_prior_mass"}:
        raise ValueError(
            "mode must be 'none', 'past_global_prior', or "
            f"'past_global_prior_mass', got {mode!r}."
        )
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive.")
    if mode == "none":
        return torch.sigmoid(logits)

    dtype = logits.dtype
    device = logits.device
    positive = torch.as_tensor(
        historical_positive_count, dtype=dtype, device=device
    ).detach()
    observed = torch.as_tensor(
        historical_observation_count, dtype=dtype, device=device
    ).detach()


    rate = (positive + float(smoothing)) / (
        observed + 2.0 * float(smoothing)
    )
    rate = rate.clamp(min=torch.finfo(dtype).eps, max=1.0 - torch.finfo(dtype).eps)
    estimated_pos_weight = ((1.0 - rate) / rate).clamp(min=1.0, max=20.0)




    probability = torch.sigmoid(logits - torch.log(estimated_pos_weight))
    if mode == "past_global_prior_mass":




        current_mass = probability.mean().detach().clamp_min(
            torch.finfo(dtype).eps
        )
        mass_scale = (rate / current_mass).clamp(max=4.0).detach()
        probability = (probability * mass_scale).clamp(max=1.0)
    return probability


@dataclass(frozen=True)
class PolicySample:

    action: GraphEditAction
    target_index: int
    old_log_prob: torch.Tensor
    old_entropy: torch.Tensor
    includes_magnitude: bool = True


@dataclass(frozen=True)
class RewardedRollout:

    sample: PolicySample
    reward: float


@dataclass(frozen=True)
class SequenceDecision:

    action: GraphEditAction
    target_index: int
    includes_magnitude: bool
    stop_after: bool | None


@dataclass(frozen=True)
class PolicySequenceSample:

    action: GraphEditSequence
    group: GraphOperation
    decisions: tuple[SequenceDecision, ...]
    old_log_prob: torch.Tensor
    old_entropy: torch.Tensor


@dataclass(frozen=True)
class RewardedSequenceRollout:
    sample: PolicySequenceSample
    reward: float | torch.Tensor


class DynamicGroupSampler:

    def __init__(
        self,
        *,
        beta: float = 1.0,
        temperature: float = 1.0,
        smoothing: float = 1.0,
    ) -> None:
        if beta <= 0 or temperature <= 0 or smoothing <= 0:
            raise ValueError("beta, temperature, and smoothing must be positive.")
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.smoothing = float(smoothing)
        self.counts: Counter[GraphOperation] = Counter()

    def observe(self, actions: Iterable[GraphEditAction | GraphOperation]) -> None:
        for item in actions:
            operation = item.operation if isinstance(item, GraphEditAction) else item
            self.counts[GraphOperation(operation)] += 1

    def distribution(
        self, valid_operations: Sequence[GraphOperation]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        operations = [GraphOperation(operation) for operation in valid_operations]
        if not operations:
            raise ValueError("At least one valid operation group is required.")
        counts = torch.tensor(
            [self.counts[operation] for operation in operations], dtype=torch.float64
        )
        frequency = (counts + self.smoothing) / (
            counts.sum() + self.smoothing * len(operations)
        )
        rarity = frequency.pow(-self.beta)
        probabilities = torch.softmax(rarity / self.temperature, dim=0)
        return frequency, rarity, probabilities

    def allocate(
        self,
        valid_operations: Sequence[GraphOperation],
        *,
        total_budget: int,
        min_per_group: int = 2,
    ) -> dict[GraphOperation, int]:
        operations = [GraphOperation(operation) for operation in valid_operations]
        if min_per_group < 1:
            raise ValueError("min_per_group must be positive.")
        minimum = min_per_group * len(operations)
        if total_budget < minimum:
            raise ValueError(
                f"total_budget={total_budget} is smaller than the required minimum {minimum}."
            )
        _, _, probabilities = self.distribution(operations)
        remaining = total_budget - minimum
        raw = probabilities * remaining
        extra = torch.floor(raw).to(torch.long)
        leftover = int(remaining - int(extra.sum().item()))
        if leftover:
            order = torch.argsort(raw - extra, descending=True)
            extra[order[:leftover]] += 1
        return {
            operation: min_per_group + int(extra[index].item())
            for index, operation in enumerate(operations)
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "beta": self.beta,
            "temperature": self.temperature,
            "smoothing": self.smoothing,
            "counts": {operation.name: count for operation, count in self.counts.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.beta = float(state["beta"])
        self.temperature = float(state["temperature"])
        self.smoothing = float(state["smoothing"])
        raw_counts = state.get("counts", {})
        if not isinstance(raw_counts, Mapping):
            raise ValueError("DynamicGroupSampler counts must be a mapping.")
        self.counts = Counter(
            {GraphOperation[name]: int(count) for name, count in raw_counts.items()}
        )


class ActionAwareController(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        *,
        policy_dim: int = 64,
        node_state_dim: int | None = None,
        node_property_dim: int | None = None,
        node_aux_dim: int = 0,
        topology_count_context_dim: int = 0,
        min_logstd: float = -5.0,
        max_logstd: float = 1.0,
    ) -> None:
        super().__init__()
        if policy_dim < 1:
            raise ValueError("policy_dim must be positive.")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.policy_dim = int(policy_dim)
        self.node_state_dim = None if node_state_dim is None else int(node_state_dim)
        self.node_property_dim = (
            None if node_property_dim is None else int(node_property_dim)
        )
        self.node_aux_dim = int(node_aux_dim)
        if self.node_aux_dim < 0:
            raise ValueError("node_aux_dim must be non-negative.")
        self.topology_count_context_dim = int(topology_count_context_dim)
        if self.topology_count_context_dim < 0:
            raise ValueError("topology_count_context_dim must be non-negative.")
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.hidden_dim, self.policy_dim),
            nn.LayerNorm(self.policy_dim),
            nn.GELU(),
        )



        self.node_aux_encoder = (
            nn.Linear(self.node_aux_dim, self.policy_dim, bias=False)
            if self.node_aux_dim
            else None
        )
        self.operation_head = nn.Linear(self.policy_dim, len(GraphOperation))
        self.node_target_head = nn.Linear(self.policy_dim, 1)
        self.edge_target_head = nn.Sequential(
            nn.Linear(4 * self.policy_dim, self.policy_dim),
            nn.GELU(),
            nn.Linear(self.policy_dim, 1),
        )



        self.topology_count_head = nn.Sequential(
            nn.Linear(self.policy_dim, self.policy_dim),
            nn.GELU(),
            nn.Linear(self.policy_dim, 2),
        )




        self.topology_count_context = (
            nn.Sequential(
                nn.Linear(self.topology_count_context_dim, self.policy_dim),
                nn.LayerNorm(self.policy_dim),
                nn.GELU(),
            )
            if self.topology_count_context_dim
            else None
        )
        self.stop_head = nn.Sequential(
            nn.Linear(self.policy_dim + 1, self.policy_dim),
            nn.GELU(),
            nn.Linear(self.policy_dim, 1),
        )
        self.state_mu_head = (
            nn.Linear(self.policy_dim, self.node_state_dim)
            if self.node_state_dim is not None
            else None
        )
        self.state_logstd_head = (
            nn.Linear(self.policy_dim, self.node_state_dim)
            if self.node_state_dim is not None
            else None
        )
        self.property_mu_head = (
            nn.Linear(self.policy_dim, self.node_property_dim)
            if self.node_property_dim is not None
            else None
        )
        self.property_logstd_head = (
            nn.Linear(self.policy_dim, self.node_property_dim)
            if self.node_property_dim is not None
            else None
        )





        self.node_count_head = nn.Sequential(
            nn.Linear(self.policy_dim, self.policy_dim),
            nn.GELU(),
            nn.Linear(self.policy_dim, 1),
        )

    def node_state(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        node_aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_t.ndim != 2 or h_t.ndim != 2 or z_t.shape[0] != h_t.shape[0]:
            raise ValueError("z_t and h_t must be node-wise matrices with equal row counts.")
        if z_t.shape[1] != self.latent_dim or h_t.shape[1] != self.hidden_dim:
            raise ValueError("z_t or h_t has an incompatible feature dimension.")
        state = self.node_encoder(torch.cat([z_t, h_t], dim=-1))
        if self.node_aux_encoder is None:
            if node_aux is not None:
                raise ValueError("node_aux was supplied to a Controller without node_aux_dim.")
            return state
        if node_aux is None:
            raise ValueError("Controller with node_aux_dim requires node_aux.")
        if node_aux.ndim != 2 or node_aux.shape != (z_t.shape[0], self.node_aux_dim):
            raise ValueError(
                "node_aux must have shape "
                f"[{z_t.shape[0]}, {self.node_aux_dim}], got {tuple(node_aux.shape)}."
            )
        return state + self.node_aux_encoder(node_aux.to(device=z_t.device, dtype=z_t.dtype))

    def operation_logits(self, node_state: torch.Tensor) -> torch.Tensor:
        return self.operation_head(node_state.mean(dim=0))

    def topology_log_counts(
        self,
        node_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_state.ndim != 2 or node_state.shape[1] != self.policy_dim:
            raise ValueError(
                f"node_state must have shape [num_nodes, {self.policy_dim}]."
            )
        pooled = node_state.mean(dim=0)
        if self.topology_count_context is None:
            if context is not None:
                raise ValueError(
                    "context was supplied to a Controller without "
                    "topology_count_context_dim."
                )
        else:
            if context is None:
                raise ValueError(
                    "A topology count context is required by this Controller."
                )
            context = context.to(device=node_state.device, dtype=node_state.dtype)
            if context.ndim != 1 or context.shape[0] != self.topology_count_context_dim:
                raise ValueError(
                    "context must have shape "
                    f"[{self.topology_count_context_dim}], got {tuple(context.shape)}."
                )
            pooled = pooled + self.topology_count_context(context)
        return self.topology_count_head(pooled)

    def topology_edit_counts(
        self,
        node_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.expm1(
            self.topology_log_counts(node_state, context=context).clamp(0.0, 12.0)
        )

    def node_log_count(self, node_state: torch.Tensor) -> torch.Tensor:
        if node_state.ndim != 2 or node_state.shape[1] != self.policy_dim:
            raise ValueError(
                f"node_state must have shape [num_nodes, {self.policy_dim}]."
            )
        return self.node_count_head(node_state.mean(dim=0)).squeeze(-1)

    def node_edit_count(self, node_state: torch.Tensor) -> torch.Tensor:
        return torch.expm1(self.node_log_count(node_state).clamp(0.0, 12.0))

    def _operation_distribution(
        self,
        node_state: torch.Tensor,
        valid_operations: Sequence[GraphOperation] | None,
        *,
        precomputed_logits: torch.Tensor | None = None,
    ) -> Categorical:
        logits = (
            self.operation_logits(node_state)
            if precomputed_logits is None
            else precomputed_logits
        )
        if valid_operations is not None:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for operation in valid_operations:
                mask[int(GraphOperation(operation))] = True
            if not bool(mask.any()):
                raise ValueError("valid_operations cannot be empty.")
            logits = logits.masked_fill(~mask, -torch.inf)
        return Categorical(logits=logits)

    def _edge_logits(
        self, node_state: torch.Tensor, candidate_edges: torch.Tensor
    ) -> torch.Tensor:
        if candidate_edges.ndim != 2 or candidate_edges.shape[0] != 2:
            raise ValueError("candidate_edges must have shape [2, num_candidates].")
        if candidate_edges.numel() == 0:
            raise ValueError("An edge operation requires at least one valid candidate.")
        source, destination = candidate_edges
        source_state = node_state[source]
        destination_state = node_state[destination]
        pair = torch.cat(
            [
                source_state,
                destination_state,
                (source_state - destination_state).abs(),
                source_state * destination_state,
            ],
            dim=-1,
        )
        return self.edge_target_head(pair).squeeze(-1)

    def edge_logits_chunked(
        self,
        node_state: torch.Tensor,
        candidate_edges: torch.Tensor,
        *,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if candidate_edges.ndim != 2 or candidate_edges.shape[0] != 2:
            raise ValueError("candidate_edges must have shape [2, num_candidates].")
        if candidate_edges.shape[1] == 0:
            return node_state.new_empty((0,))
        return torch.cat(
            [
                self._edge_logits(node_state, candidate_edges[:, start:stop])
                for start in range(0, candidate_edges.shape[1], chunk_size)
                for stop in [min(start + chunk_size, candidate_edges.shape[1])]
            ]
        )

    def soft_edge_action_probabilities(
        self,
        node_state: torch.Tensor,
        candidate_edges: Mapping[GraphOperation, torch.Tensor],
        *,
        chunk_size: int = 65536,
        top_k_per_operation: int = 0,
        support_count_by_operation: Mapping[GraphOperation, int] | None = None,
        mode: str = "categorical",
    ) -> dict[GraphOperation, torch.Tensor]:
        if top_k_per_operation < 0:
            raise ValueError("top_k_per_operation must be non-negative.")
        if mode not in {"categorical", "multilabel"}:
            raise ValueError("mode must be categorical or multilabel.")
        available = [
            operation
            for operation in (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE)
            if operation in candidate_edges and candidate_edges[operation].shape[1] > 0
        ]
        if not available:
            return {}
        operation_probability = None
        if mode == "categorical":
            operation_probability = torch.softmax(
                self._operation_distribution(node_state, available).logits, dim=0
            )
        result: dict[GraphOperation, torch.Tensor] = {}
        for operation in available:
            candidates = candidate_edges[operation].to(
                device=node_state.device, dtype=torch.long
            )
            logits = self.edge_logits_chunked(
                node_state, candidates, chunk_size=chunk_size
            )
            if support_count_by_operation is not None:




                count = max(
                    0,
                    min(
                        int(support_count_by_operation.get(operation, 0)),
                        int(logits.numel()),
                    ),
                )
                probability = torch.zeros_like(logits)
                if count > 0:
                    support = torch.topk(logits, k=count).indices
                    probability[support] = (
                        torch.softmax(logits[support], dim=0)
                        if mode == "categorical"
                        else torch.sigmoid(logits[support])
                    )
                result[operation] = probability
            elif top_k_per_operation:




                count = min(int(top_k_per_operation), int(logits.numel()))
                support = torch.topk(logits, k=count).indices
                probability = torch.zeros_like(logits)
                probability[support] = (
                    torch.softmax(logits[support], dim=0)
                    if mode == "categorical"
                    else torch.sigmoid(logits[support])
                )
                result[operation] = probability
            elif mode == "multilabel":
                result[operation] = torch.sigmoid(logits)
            else:
                assert operation_probability is not None
                result[operation] = (
                    torch.softmax(logits, dim=0)
                    * operation_probability[int(operation)]
                )
        return result

    def _magnitude_distribution(
        self, node_state: torch.Tensor, operation: GraphOperation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if operation == GraphOperation.MODIFY_NODE_STATE:
            if self.state_mu_head is None or self.state_logstd_head is None:
                raise RuntimeError("Controller was created without node_state_dim.")
            mu = self.state_mu_head(node_state)
            logstd = self.state_logstd_head(node_state)
        elif operation == GraphOperation.MODIFY_NODE_PROPERTY:
            if self.property_mu_head is None or self.property_logstd_head is None:
                raise RuntimeError("Controller was created without node_property_dim.")
            mu = self.property_mu_head(node_state)
            logstd = self.property_logstd_head(node_state)
        elif operation in {GraphOperation.ADD_NODE, GraphOperation.REMOVE_NODE}:






            mu = node_state.new_zeros((node_state.shape[0], 1))
            logstd = node_state.new_full((node_state.shape[0], 1), -5.0)
        else:
            raise ValueError("Topology operations do not have a vector magnitude head.")
        return mu, logstd.clamp(self.min_logstd, self.max_logstd)

    def sample_action(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        operation: GraphOperation,
        *,
        candidate_edges: torch.Tensor | None = None,
        candidate_nodes: torch.Tensor | None = None,
        valid_operations: Sequence[GraphOperation] | None = None,
        sample_magnitude: bool = True,
    ) -> PolicySample:
        operation = GraphOperation(operation)
        state = self.node_state(z_t, h_t)
        operation_dist = self._operation_distribution(state, valid_operations)
        operation_index = torch.tensor(int(operation), device=z_t.device)
        operation_log_prob = operation_dist.log_prob(operation_index)
        operation_entropy = operation_dist.entropy()

        if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
            if candidate_edges is None:
                raise ValueError("candidate_edges are required for an edge action.")
            candidate_edges = candidate_edges.to(device=z_t.device, dtype=torch.long)
            target_dist = Categorical(logits=self._edge_logits(state, candidate_edges))
            target_index = target_dist.sample()
            edge = candidate_edges[:, target_index]
            action = GraphEditAction(
                operation, (int(edge[0].item()), int(edge[1].item()))
            )
            target_log_prob = target_dist.log_prob(target_index)
            entropy = operation_entropy + target_dist.entropy()
            return PolicySample(
                action,
                int(target_index.item()),
                (operation_log_prob + target_log_prob).detach(),
                entropy.detach(),
            )

        node_logits = self.node_target_head(state).squeeze(-1)
        if candidate_nodes is None:
            candidate_nodes = torch.arange(z_t.shape[0], device=z_t.device)
        else:
            candidate_nodes = candidate_nodes.to(device=z_t.device, dtype=torch.long)
        if candidate_nodes.numel() == 0:
            raise ValueError("A node operation requires at least one valid candidate.")
        target_dist = Categorical(logits=node_logits[candidate_nodes])
        candidate_index = target_dist.sample()
        target_index = candidate_nodes[candidate_index]
        mu, logstd = self._magnitude_distribution(state, operation)
        if sample_magnitude:
            value_dist = Normal(mu[target_index], logstd[target_index].exp())
            value = value_dist.sample().detach()
            value_log_prob = value_dist.log_prob(value).mean()
            value_entropy = value_dist.entropy().mean()
        else:





            value = torch.zeros_like(mu[target_index])
            value_log_prob = z_t.new_zeros(())
            value_entropy = z_t.new_zeros(())
        action = GraphEditAction(operation, (int(target_index.item()),), value)
        log_prob = (
            operation_log_prob
            + target_dist.log_prob(candidate_index)
            + value_log_prob
        )
        entropy = operation_entropy + target_dist.entropy() + value_entropy
        return PolicySample(
            action,
            int(candidate_index.item()),
            log_prob.detach(),
            entropy.detach(),
            includes_magnitude=sample_magnitude,
        )

    def evaluate_action(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        sample: PolicySample,
        *,
        candidate_edges: torch.Tensor | None = None,
        candidate_nodes: torch.Tensor | None = None,
        valid_operations: Sequence[GraphOperation] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.node_state(z_t, h_t)
        operation = sample.action.operation
        operation_dist = self._operation_distribution(state, valid_operations)
        operation_index = torch.tensor(int(operation), device=z_t.device)
        log_prob = operation_dist.log_prob(operation_index)
        entropy = operation_dist.entropy()
        target_index = torch.tensor(sample.target_index, device=z_t.device)

        if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
            if candidate_edges is None:
                raise ValueError("candidate_edges are required for an edge action.")
            target_dist = Categorical(
                logits=self._edge_logits(
                    state, candidate_edges.to(device=z_t.device, dtype=torch.long)
                )
            )
            return log_prob + target_dist.log_prob(target_index), entropy + target_dist.entropy()

        node_logits = self.node_target_head(state).squeeze(-1)
        if candidate_nodes is None:
            candidate_nodes = torch.arange(z_t.shape[0], device=z_t.device)
        else:
            candidate_nodes = candidate_nodes.to(device=z_t.device, dtype=torch.long)
        target_dist = Categorical(logits=node_logits[candidate_nodes])
        node_index = candidate_nodes[target_index]
        mu, logstd = self._magnitude_distribution(state, operation)
        value = sample.action.value
        assert value is not None
        value = value.to(device=z_t.device, dtype=z_t.dtype)
        log_prob = log_prob + target_dist.log_prob(target_index)
        entropy = entropy + target_dist.entropy()
        if sample.includes_magnitude:
            value_dist = Normal(mu[node_index], logstd[node_index].exp())
            log_prob = log_prob + value_dist.log_prob(value).mean()
            entropy = entropy + value_dist.entropy().mean()
        return log_prob, entropy

    def _stop_distribution(
        self,
        node_state: torch.Tensor,
        *,
        length: int,
        max_actions: int,
    ) -> Bernoulli:
        progress = node_state.new_tensor(
            [float(length) / float(max(max_actions, 1))]
        )
        context = torch.cat([node_state.mean(dim=0), progress], dim=0)
        return Bernoulli(logits=self.stop_head(context).squeeze(-1))

    @staticmethod
    def _candidate_mask(
        size: int, *, device: torch.device
    ) -> torch.Tensor:
        return torch.ones(size, dtype=torch.bool, device=device)

    def sample_action_sequence(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        group: GraphOperation,
        *,
        candidate_edges: Mapping[GraphOperation, torch.Tensor] | None = None,
        candidate_nodes: Mapping[GraphOperation, torch.Tensor] | None = None,
        valid_operations: Sequence[GraphOperation] | None = None,
        min_actions: int = 1,
        max_actions: int = 8,
        sample_magnitude: bool = True,
        node_state: torch.Tensor | None = None,
    ) -> PolicySequenceSample:
        if min_actions < 1 or max_actions < min_actions:
            raise ValueError("Require 1 <= min_actions <= max_actions.")
        state = self.node_state(z_t, h_t) if node_state is None else node_state
        operations = tuple(
            GraphOperation(operation)
            for operation in (valid_operations or tuple(GraphOperation))
        )
        if GraphOperation(group) not in operations:
            raise ValueError("The rollout group must be a valid operation.")
        operation_logits = self.operation_logits(state)
        node_target_logits = self.node_target_head(state).squeeze(-1)
        edge_target_logits: dict[GraphOperation, torch.Tensor] = {}
        magnitude_cache: dict[GraphOperation, tuple[torch.Tensor, torch.Tensor]] = {}
        edge_masks: dict[GraphOperation, torch.Tensor] = {}
        node_masks: dict[GraphOperation, torch.Tensor] = {}
        for operation in operations:
            if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
                candidates = None if candidate_edges is None else candidate_edges.get(operation)
                if candidates is None or candidates.shape[1] == 0:
                    continue
                edge_masks[operation] = self._candidate_mask(
                    candidates.shape[1], device=z_t.device
                )
            else:
                candidates = None if candidate_nodes is None else candidate_nodes.get(operation)
                count = z_t.shape[0] if candidates is None else candidates.numel()
                if count:
                    node_masks[operation] = self._candidate_mask(count, device=z_t.device)

        decisions: list[SequenceDecision] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for step in range(max_actions):
            available_operations = [
                operation
                for operation in operations
                if (
                    bool(edge_masks[operation].any())
                    if operation in edge_masks
                    else bool(node_masks.get(operation, torch.zeros((), dtype=torch.bool)).any())
                )
            ]
            if not available_operations:
                break
            operation_dist = self._operation_distribution(
                state,
                available_operations,
                precomputed_logits=operation_logits,
            )
            if step == 0:
                operation = GraphOperation(group)
                if operation not in available_operations:
                    raise ValueError("The requested rollout group has no valid target.")
            else:
                operation = GraphOperation(int(operation_dist.sample().item()))
            operation_index = torch.tensor(int(operation), device=z_t.device)
            log_prob = operation_dist.log_prob(operation_index)
            entropy = operation_dist.entropy()

            if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
                assert candidate_edges is not None
                candidates = candidate_edges[operation].to(
                    device=z_t.device, dtype=torch.long
                )
                mask = edge_masks[operation]
                if operation not in edge_target_logits:
                    edge_target_logits[operation] = self._edge_logits(state, candidates)
                logits = edge_target_logits[operation].masked_fill(~mask, -torch.inf)
                target_dist = Categorical(logits=logits)
                target_index = target_dist.sample()
                edge = candidates[:, target_index]
                edit = GraphEditAction(
                    operation, (int(edge[0].item()), int(edge[1].item()))
                )
                log_prob = log_prob + target_dist.log_prob(target_index)
                entropy = entropy + target_dist.entropy()
                edge_masks[operation][target_index] = False
                includes_magnitude = False
            else:
                raw_candidates = None if candidate_nodes is None else candidate_nodes.get(operation)
                candidates = (
                    torch.arange(z_t.shape[0], device=z_t.device)
                    if raw_candidates is None
                    else raw_candidates.to(device=z_t.device, dtype=torch.long)
                )
                mask = node_masks[operation]
                logits = node_target_logits[candidates]
                target_dist = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
                target_index = target_dist.sample()
                node_index = candidates[target_index]
                if operation not in magnitude_cache:
                    magnitude_cache[operation] = self._magnitude_distribution(state, operation)
                mu, logstd = magnitude_cache[operation]
                if sample_magnitude:
                    value_dist = Normal(mu[node_index], logstd[node_index].exp())
                    value = value_dist.sample().detach()
                    log_prob = log_prob + value_dist.log_prob(value).mean()
                    entropy = entropy + value_dist.entropy().mean()
                    includes_magnitude = True
                else:
                    value = torch.zeros_like(mu[node_index])
                    includes_magnitude = False
                log_prob = log_prob + target_dist.log_prob(target_index)
                entropy = entropy + target_dist.entropy()
                edit = GraphEditAction(operation, (int(node_index.item()),), value)
                node_masks[operation][target_index] = False

            stop_after: bool | None = None
            can_continue = step + 1 < max_actions and any(
                bool(mask.any()) for mask in [*edge_masks.values(), *node_masks.values()]
            )
            if step + 1 >= min_actions and can_continue:
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                stop_tensor = stop_dist.sample()
                stop_after = bool(stop_tensor.item())
                log_prob = log_prob + stop_dist.log_prob(stop_tensor)
                entropy = entropy + stop_dist.entropy()

            decisions.append(
                SequenceDecision(
                    edit,
                    int(target_index.item()),
                    includes_magnitude,
                    stop_after,
                )
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
            if stop_after:
                break

        if not decisions:
            raise RuntimeError("Controller could not sample a valid graph edit.")
        return PolicySequenceSample(
            GraphEditSequence(tuple(decision.action for decision in decisions)),
            GraphOperation(group),
            tuple(decisions),
            torch.stack(log_probs).sum().detach(),
            torch.stack(entropies).mean().detach(),
        )

    @torch.no_grad()
    def sample_node_action_sequences_batch(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        operation: GraphOperation,
        *,
        candidate_nodes: torch.Tensor | None = None,
        min_actions: int = 1,
        max_actions: int = 8,
        sample_magnitude: bool = True,
        node_state: torch.Tensor | None = None,
        batch_size: int,
    ) -> list[PolicySequenceSample]:
        operation = GraphOperation(operation)
        if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
            raise ValueError("Batch node sampling only supports node operations.")
        if batch_size < 1 or min_actions < 1 or max_actions < min_actions:
            raise ValueError("Invalid batch or action length.")
        state = self.node_state(z_t, h_t) if node_state is None else node_state
        candidates = (
            torch.arange(z_t.shape[0], device=z_t.device)
            if candidate_nodes is None
            else candidate_nodes.to(device=z_t.device, dtype=torch.long)
        )
        if candidates.numel() == 0:
            raise ValueError("A node operation requires at least one candidate.")
        operation_logits = self.operation_logits(state)
        operation_dist = self._operation_distribution(
            state, (operation,), precomputed_logits=operation_logits
        )
        operation_index = torch.tensor(int(operation), device=z_t.device)
        operation_log_prob = operation_dist.log_prob(operation_index)
        operation_entropy = operation_dist.entropy()
        node_logits = self.node_target_head(state).squeeze(-1)[candidates]
        mu, logstd = self._magnitude_distribution(state, operation)
        mu, logstd = mu[candidates], logstd[candidates]
        masks = torch.ones(
            (batch_size, candidates.numel()), dtype=torch.bool, device=z_t.device
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=z_t.device)
        decisions: list[list[SequenceDecision]] = [[] for _ in range(batch_size)]
        log_probs = [operation_log_prob.new_zeros(()) for _ in range(batch_size)]
        entropies = [operation_entropy.new_zeros(()) for _ in range(batch_size)]
        for step in range(max_actions):
            if not bool(active.any()):
                break
            active_ids = active.nonzero(as_tuple=False).flatten()
            target_dist = Categorical(
                logits=node_logits.unsqueeze(0)
                .expand(active_ids.numel(), -1)
                .masked_fill(~masks[active_ids], -torch.inf)
            )
            target_index = target_dist.sample()
            target_node = candidates[target_index]
            target_lp = target_dist.log_prob(target_index)
            target_entropy = target_dist.entropy()
            if sample_magnitude:
                selected_mu = mu[target_index]
                selected_logstd = logstd[target_index]
                value_dist = Normal(selected_mu, selected_logstd.exp())
                values = value_dist.sample()
                value_lp = value_dist.log_prob(values).mean(dim=-1)
                value_entropy = value_dist.entropy().mean(dim=-1)
                includes_magnitude = True
            else:
                values = torch.zeros(
                    (active_ids.numel(), mu.shape[-1]), device=z_t.device, dtype=z_t.dtype
                )
                value_lp = target_lp.new_zeros(target_lp.shape)
                value_entropy = target_lp.new_zeros(target_lp.shape)
                includes_magnitude = False
            masks[active_ids, target_index] = False
            can_continue = (step + 1 < max_actions) & masks[active_ids].any(dim=-1)
            stop_after = torch.zeros_like(active_ids, dtype=torch.bool)
            if step + 1 >= min_actions and bool(can_continue.any()):
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                sampled_stop = stop_dist.sample().expand(active_ids.numel())
                stop_after = sampled_stop.bool() & can_continue
                stop_lp = stop_dist.log_prob(sampled_stop) * can_continue.to(target_lp.dtype)
                stop_entropy = stop_dist.entropy().expand_as(stop_lp) * can_continue.to(target_lp.dtype)
            else:
                stop_lp = target_lp.new_zeros(target_lp.shape)
                stop_entropy = target_lp.new_zeros(target_lp.shape)
            for local, sample_id in enumerate(active_ids.tolist()):
                action = GraphEditAction(
                    operation,
                    (int(target_node[local].item()),),
                    values[local].detach(),
                )
                decisions[sample_id].append(
                    SequenceDecision(
                        action,
                        int(target_index[local].item()),
                        includes_magnitude,
                        bool(stop_after[local].item()) if bool(can_continue[local]) else None,
                    )
                )
                log_probs[sample_id] = log_probs[sample_id] + operation_log_prob + target_lp[local] + value_lp[local] + stop_lp[local]
                entropies[sample_id] = entropies[sample_id] + operation_entropy + target_entropy[local] + value_entropy[local] + stop_entropy[local]






            exhausted = ~can_continue
            active[active_ids[stop_after | exhausted]] = False
        return [
            PolicySequenceSample(
                GraphEditSequence(tuple(item.action for item in sample_decisions)),
                operation,
                tuple(sample_decisions),
                log_probs[index].detach(),
                (entropies[index] / max(len(sample_decisions), 1)).detach(),
            )
            for index, sample_decisions in enumerate(decisions)
        ]

    def evaluate_node_action_sequences_batch(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        samples: Sequence[PolicySequenceSample],
        *,
        candidate_nodes: torch.Tensor | None = None,
        max_actions: int,
        node_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("At least one sequence sample is required.")
        operation = samples[0].group
        if any(sample.group != operation for sample in samples):
            raise ValueError("All samples must share one operation group.")
        state = self.node_state(z_t, h_t) if node_state is None else node_state
        candidates = (
            torch.arange(z_t.shape[0], device=z_t.device)
            if candidate_nodes is None
            else candidate_nodes.to(device=z_t.device, dtype=torch.long)
        )
        operation_dist = self._operation_distribution(
            state, (operation,), precomputed_logits=self.operation_logits(state)
        )
        operation_index = torch.tensor(int(operation), device=z_t.device)
        operation_lp = operation_dist.log_prob(operation_index)
        operation_entropy = operation_dist.entropy()
        node_logits = self.node_target_head(state).squeeze(-1)[candidates]
        mu, logstd = self._magnitude_distribution(state, operation)
        mu, logstd = mu[candidates], logstd[candidates]
        batch_size = len(samples)
        masks = torch.ones(
            (batch_size, candidates.numel()), dtype=torch.bool, device=z_t.device
        )
        log_probs = torch.zeros(batch_size, device=z_t.device, dtype=z_t.dtype)
        entropy_sums = torch.zeros(batch_size, device=z_t.device, dtype=z_t.dtype)
        for step in range(max_actions):
            active_ids = [
                index for index, sample in enumerate(samples) if step < len(sample.decisions)
            ]
            if not active_ids:
                break
            ids = torch.tensor(active_ids, dtype=torch.long, device=z_t.device)
            decisions = [samples[index].decisions[step] for index in active_ids]
            log_probs[ids] = log_probs[ids] + operation_lp
            entropy_sums[ids] = entropy_sums[ids] + operation_entropy
            target_index = torch.tensor(
                [decision.target_index for decision in decisions],
                dtype=torch.long,
                device=z_t.device,
            )
            target_dist = Categorical(
                logits=node_logits.unsqueeze(0)
                .expand(len(active_ids), -1)
                .masked_fill(~masks[ids], -torch.inf)
            )
            target_lp = target_dist.log_prob(target_index)
            target_entropy = target_dist.entropy()
            for local, sample_id in enumerate(active_ids):
                masks[sample_id, target_index[local]] = False
            log_probs[ids] = log_probs[ids] + target_lp
            entropy_sums[ids] = entropy_sums[ids] + target_entropy
            magnitude_ids = [
                local for local, decision in enumerate(decisions) if decision.includes_magnitude
            ]
            if magnitude_ids:
                mid = torch.tensor(magnitude_ids, dtype=torch.long, device=z_t.device)
                selected_nodes = target_index[mid]
                value_dist = Normal(mu[selected_nodes], logstd[selected_nodes].exp())
                values = torch.stack(
                    [decisions[local].action.value.to(z_t.device, z_t.dtype) for local in magnitude_ids]
                )
                log_probs[ids[mid]] += value_dist.log_prob(values).mean(dim=-1)
                entropy_sums[ids[mid]] += value_dist.entropy().mean(dim=-1)
            stop_ids = [
                local for local, decision in enumerate(decisions) if decision.stop_after is not None
            ]
            if stop_ids:
                sid = torch.tensor(stop_ids, dtype=torch.long, device=z_t.device)
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                stop_values = torch.tensor(
                    [float(decisions[local].stop_after) for local in stop_ids],
                    device=z_t.device,
                )
                log_probs[ids[sid]] += stop_dist.log_prob(stop_values)
                entropy_sums[ids[sid]] += stop_dist.entropy()
        lengths = torch.tensor(
            [max(len(sample.decisions), 1) for sample in samples],
            dtype=z_t.dtype,
            device=z_t.device,
        )
        return log_probs, entropy_sums / lengths

    @torch.no_grad()
    def sample_edge_action_sequences_batch(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        operation: GraphOperation,
        *,
        candidate_edges: torch.Tensor,
        min_actions: int = 1,
        max_actions: int = 8,
        node_state: torch.Tensor | None = None,
        batch_size: int,
    ) -> list[PolicySequenceSample]:
        operation = GraphOperation(operation)
        if operation not in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
            raise ValueError("Batch edge sampling only supports edge operations.")
        if batch_size < 1 or min_actions < 1 or max_actions < min_actions:
            raise ValueError("Invalid batch or action length.")
        state = self.node_state(z_t, h_t) if node_state is None else node_state
        candidates = candidate_edges.to(device=z_t.device, dtype=torch.long)
        if candidates.ndim != 2 or candidates.shape[0] != 2 or not candidates.shape[1]:
            raise ValueError("An edge operation requires [2, E] candidate edges.")
        operation_dist = self._operation_distribution(
            state, (operation,), precomputed_logits=self.operation_logits(state)
        )
        operation_index = torch.tensor(int(operation), device=z_t.device)
        operation_lp = operation_dist.log_prob(operation_index)
        operation_entropy = operation_dist.entropy()
        edge_logits = self._edge_logits(state, candidates)
        masks = torch.ones(
            (batch_size, candidates.shape[1]), dtype=torch.bool, device=z_t.device
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=z_t.device)
        decisions: list[list[SequenceDecision]] = [[] for _ in range(batch_size)]
        log_probs = [operation_lp.new_zeros(()) for _ in range(batch_size)]
        entropies = [operation_entropy.new_zeros(()) for _ in range(batch_size)]
        for step in range(max_actions):
            if not bool(active.any()):
                break
            active_ids = active.nonzero(as_tuple=False).flatten()
            target_dist = Categorical(
                logits=edge_logits.unsqueeze(0)
                .expand(active_ids.numel(), -1)
                .masked_fill(~masks[active_ids], -torch.inf)
            )
            target_index = target_dist.sample()
            target_lp = target_dist.log_prob(target_index)
            target_entropy = target_dist.entropy()
            masks[active_ids, target_index] = False
            can_continue = (step + 1 < max_actions) & masks[active_ids].any(dim=-1)
            stop_after = torch.zeros_like(active_ids, dtype=torch.bool)
            if step + 1 >= min_actions and bool(can_continue.any()):
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                sampled_stop = stop_dist.sample((active_ids.numel(),)).reshape(-1)
                stop_after = sampled_stop.bool() & can_continue
                stop_lp = stop_dist.log_prob(sampled_stop) * can_continue.to(target_lp.dtype)
                stop_entropy = (
                    stop_dist.entropy().expand_as(stop_lp)
                    * can_continue.to(target_lp.dtype)
                )
            else:
                stop_lp = target_lp.new_zeros(target_lp.shape)
                stop_entropy = target_lp.new_zeros(target_lp.shape)
            selected_edges = candidates.index_select(1, target_index)
            for local, sample_id in enumerate(active_ids.tolist()):
                action = GraphEditAction(
                    operation,
                    (
                        int(selected_edges[0, local].item()),
                        int(selected_edges[1, local].item()),
                    ),
                )
                decisions[sample_id].append(
                    SequenceDecision(
                        action,
                        int(target_index[local].item()),
                        False,
                        bool(stop_after[local].item())
                        if bool(can_continue[local])
                        else None,
                    )
                )
                log_probs[sample_id] = (
                    log_probs[sample_id]
                    + operation_lp
                    + target_lp[local]
                    + stop_lp[local]
                )
                entropies[sample_id] = (
                    entropies[sample_id]
                    + operation_entropy
                    + target_entropy[local]
                    + stop_entropy[local]
                )
            exhausted = ~can_continue
            active[active_ids[stop_after | exhausted]] = False
        return [
            PolicySequenceSample(
                GraphEditSequence(tuple(item.action for item in sample_decisions)),
                operation,
                tuple(sample_decisions),
                log_probs[index].detach(),
                (entropies[index] / max(len(sample_decisions), 1)).detach(),
            )
            for index, sample_decisions in enumerate(decisions)
        ]

    def evaluate_edge_action_sequences_batch(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        samples: Sequence[PolicySequenceSample],
        *,
        candidate_edges: torch.Tensor,
        max_actions: int,
        node_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("At least one sequence sample is required.")
        operation = samples[0].group
        if operation not in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
            raise ValueError("Batch edge evaluation only supports edge operations.")
        if any(sample.group != operation for sample in samples):
            raise ValueError("All samples must share one operation group.")
        state = self.node_state(z_t, h_t) if node_state is None else node_state
        candidates = candidate_edges.to(device=z_t.device, dtype=torch.long)
        operation_dist = self._operation_distribution(
            state, (operation,), precomputed_logits=self.operation_logits(state)
        )
        operation_index = torch.tensor(int(operation), device=z_t.device)
        operation_lp = operation_dist.log_prob(operation_index)
        operation_entropy = operation_dist.entropy()
        edge_logits = self._edge_logits(state, candidates)
        batch_size = len(samples)
        masks = torch.ones(
            (batch_size, candidates.shape[1]), dtype=torch.bool, device=z_t.device
        )
        log_probs = torch.zeros(batch_size, device=z_t.device, dtype=z_t.dtype)
        entropy_sums = torch.zeros(batch_size, device=z_t.device, dtype=z_t.dtype)
        for step in range(max_actions):
            active_ids = [
                index for index, sample in enumerate(samples) if step < len(sample.decisions)
            ]
            if not active_ids:
                break
            ids = torch.tensor(active_ids, dtype=torch.long, device=z_t.device)
            decisions = [samples[index].decisions[step] for index in active_ids]
            target_index = torch.tensor(
                [decision.target_index for decision in decisions],
                dtype=torch.long,
                device=z_t.device,
            )
            target_dist = Categorical(
                logits=edge_logits.unsqueeze(0)
                .expand(len(active_ids), -1)
                .masked_fill(~masks[ids], -torch.inf)
            )
            log_probs[ids] += operation_lp + target_dist.log_prob(target_index)
            entropy_sums[ids] += operation_entropy + target_dist.entropy()
            masks[ids, target_index] = False
            stop_ids = [
                local for local, decision in enumerate(decisions) if decision.stop_after is not None
            ]
            if stop_ids:
                sid = torch.tensor(stop_ids, dtype=torch.long, device=z_t.device)
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                stop_values = torch.tensor(
                    [float(decisions[local].stop_after) for local in stop_ids],
                    device=z_t.device,
                )
                log_probs[ids[sid]] += stop_dist.log_prob(stop_values)
                entropy_sums[ids[sid]] += stop_dist.entropy()
        lengths = torch.tensor(
            [max(len(sample.decisions), 1) for sample in samples],
            dtype=z_t.dtype,
            device=z_t.device,
        )
        return log_probs, entropy_sums / lengths

    @torch.no_grad()
    def sample_mixed_edge_action_sequences_batch(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        groups: Sequence[GraphOperation],
        *,
        candidate_edges: Mapping[GraphOperation, torch.Tensor],
        valid_operations: Sequence[GraphOperation],
        min_actions: int = 1,
        max_actions: int = 8,
        node_state: torch.Tensor | None = None,
    ) -> list[PolicySequenceSample]:
        operations = tuple(GraphOperation(operation) for operation in valid_operations)
        edge_operations = {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
        if not groups or any(operation not in edge_operations for operation in operations):
            raise ValueError("Mixed edge batching requires non-empty edge-only operations.")
        normalized_groups = tuple(GraphOperation(group) for group in groups)
        if any(group not in operations for group in normalized_groups):
            raise ValueError("Every rollout group must be a valid edge operation.")
        if min_actions < 1 or max_actions < min_actions:
            raise ValueError("Require 1 <= min_actions <= max_actions.")

        state = self.node_state(z_t, h_t) if node_state is None else node_state
        operation_logits = self.operation_logits(state)
        candidates: dict[GraphOperation, torch.Tensor] = {}
        edge_logits: dict[GraphOperation, torch.Tensor] = {}
        masks: dict[GraphOperation, torch.Tensor] = {}
        batch_size = len(normalized_groups)
        for operation in operations:
            raw_candidates = candidate_edges.get(operation)
            if raw_candidates is None:
                raise ValueError(f"Missing candidates for {operation.name}.")
            current_candidates = raw_candidates.to(device=z_t.device, dtype=torch.long)
            if (
                current_candidates.ndim != 2
                or current_candidates.shape[0] != 2
                or current_candidates.shape[1] == 0
            ):
                raise ValueError(
                    f"{operation.name} requires non-empty [2, E] candidates."
                )
            candidates[operation] = current_candidates
            edge_logits[operation] = self._edge_logits(state, current_candidates)
            masks[operation] = torch.ones(
                (batch_size, current_candidates.shape[1]),
                dtype=torch.bool,
                device=z_t.device,
            )

        active = torch.ones(batch_size, dtype=torch.bool, device=z_t.device)
        decisions: list[list[SequenceDecision]] = [[] for _ in range(batch_size)]
        log_probs = torch.zeros(batch_size, dtype=z_t.dtype, device=z_t.device)
        entropy_sums = torch.zeros_like(log_probs)
        group_indices = torch.tensor(
            [int(group) for group in normalized_groups],
            dtype=torch.long,
            device=z_t.device,
        )

        for step in range(max_actions):
            if not bool(active.any()):
                break
            active_ids = active.nonzero(as_tuple=False).flatten()
            available = torch.zeros(
                (active_ids.numel(), operation_logits.numel()),
                dtype=torch.bool,
                device=z_t.device,
            )
            for operation in operations:
                available[:, int(operation)] = masks[operation][active_ids].any(dim=-1)
            operation_dist = Categorical(
                logits=operation_logits.unsqueeze(0)
                .expand(active_ids.numel(), -1)
                .masked_fill(~available, -torch.inf)
            )
            selected_operations = (
                group_indices[active_ids] if step == 0 else operation_dist.sample()
            )
            if not bool(
                available.gather(1, selected_operations.unsqueeze(1)).all()
            ):
                raise RuntimeError("A requested rollout group has no valid edge target.")
            step_log_prob = operation_dist.log_prob(selected_operations)
            step_entropy = operation_dist.entropy()
            selected_target_indices = torch.empty_like(selected_operations)
            selected_edges = torch.empty(
                (active_ids.numel(), 2), dtype=torch.long, device=z_t.device
            )

            for operation in operations:
                local_ids = (selected_operations == int(operation)).nonzero(
                    as_tuple=False
                ).flatten()
                if local_ids.numel() == 0:
                    continue
                sample_ids = active_ids[local_ids]
                target_dist = Categorical(
                    logits=edge_logits[operation]
                    .unsqueeze(0)
                    .expand(local_ids.numel(), -1)
                    .masked_fill(~masks[operation][sample_ids], -torch.inf)
                )
                target_indices = target_dist.sample()
                step_log_prob[local_ids] += target_dist.log_prob(target_indices)
                step_entropy[local_ids] += target_dist.entropy()
                masks[operation][sample_ids, target_indices] = False
                selected_target_indices[local_ids] = target_indices
                selected_edges[local_ids] = candidates[operation].index_select(
                    1, target_indices
                ).transpose(0, 1)

            can_continue = torch.zeros(
                active_ids.numel(), dtype=torch.bool, device=z_t.device
            )
            if step + 1 < max_actions:
                for operation in operations:
                    can_continue |= masks[operation][active_ids].any(dim=-1)
            stop_after = torch.zeros_like(can_continue)
            stop_recorded = torch.zeros_like(can_continue)
            if step + 1 >= min_actions and bool(can_continue.any()):
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                sampled_stop = stop_dist.sample((active_ids.numel(),)).reshape(-1)
                stop_recorded = can_continue
                stop_after = sampled_stop.bool() & can_continue
                step_log_prob += (
                    stop_dist.log_prob(sampled_stop)
                    * can_continue.to(step_log_prob.dtype)
                )
                step_entropy += (
                    stop_dist.entropy().expand_as(step_entropy)
                    * can_continue.to(step_entropy.dtype)
                )

            log_probs[active_ids] += step_log_prob
            entropy_sums[active_ids] += step_entropy
            for local, sample_id in enumerate(active_ids.tolist()):
                operation = GraphOperation(int(selected_operations[local].item()))
                edge = selected_edges[local]
                decisions[sample_id].append(
                    SequenceDecision(
                        GraphEditAction(
                            operation,
                            (int(edge[0].item()), int(edge[1].item())),
                        ),
                        int(selected_target_indices[local].item()),
                        False,
                        bool(stop_after[local].item())
                        if bool(stop_recorded[local])
                        else None,
                    )
                )
            active[active_ids[stop_after | ~can_continue]] = False

        return [
            PolicySequenceSample(
                GraphEditSequence(tuple(item.action for item in sample_decisions)),
                normalized_groups[index],
                tuple(sample_decisions),
                log_probs[index].detach(),
                (entropy_sums[index] / max(len(sample_decisions), 1)).detach(),
            )
            for index, sample_decisions in enumerate(decisions)
        ]

    def evaluate_mixed_edge_action_sequences_batch(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        samples: Sequence[PolicySequenceSample],
        *,
        candidate_edges: Mapping[GraphOperation, torch.Tensor],
        valid_operations: Sequence[GraphOperation],
        max_actions: int,
        node_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("At least one sequence sample is required.")
        operations = tuple(GraphOperation(operation) for operation in valid_operations)
        edge_operations = {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
        if any(operation not in edge_operations for operation in operations):
            raise ValueError("Mixed edge batching requires edge-only operations.")
        if any(
            decision.action.operation not in operations
            for sample in samples
            for decision in sample.decisions
        ):
            raise ValueError("A sampled decision is outside valid_operations.")

        state = self.node_state(z_t, h_t) if node_state is None else node_state
        operation_logits = self.operation_logits(state)
        batch_size = len(samples)
        edge_logits: dict[GraphOperation, torch.Tensor] = {}
        masks: dict[GraphOperation, torch.Tensor] = {}
        for operation in operations:
            current_candidates = candidate_edges[operation].to(
                device=z_t.device, dtype=torch.long
            )
            edge_logits[operation] = self._edge_logits(state, current_candidates)
            masks[operation] = torch.ones(
                (batch_size, current_candidates.shape[1]),
                dtype=torch.bool,
                device=z_t.device,
            )

        log_probs = torch.zeros(batch_size, dtype=z_t.dtype, device=z_t.device)
        entropy_sums = torch.zeros_like(log_probs)
        for step in range(max_actions):
            active_ids = [
                index for index, sample in enumerate(samples)
                if step < len(sample.decisions)
            ]
            if not active_ids:
                break
            ids = torch.tensor(active_ids, dtype=torch.long, device=z_t.device)
            current_decisions = [samples[index].decisions[step] for index in active_ids]
            available = torch.zeros(
                (ids.numel(), operation_logits.numel()),
                dtype=torch.bool,
                device=z_t.device,
            )
            for operation in operations:
                available[:, int(operation)] = masks[operation][ids].any(dim=-1)
            operation_dist = Categorical(
                logits=operation_logits.unsqueeze(0)
                .expand(ids.numel(), -1)
                .masked_fill(~available, -torch.inf)
            )
            selected_operations = torch.tensor(
                [int(decision.action.operation) for decision in current_decisions],
                dtype=torch.long,
                device=z_t.device,
            )
            log_probs[ids] += operation_dist.log_prob(selected_operations)
            entropy_sums[ids] += operation_dist.entropy()

            for operation in operations:
                local_ids = (selected_operations == int(operation)).nonzero(
                    as_tuple=False
                ).flatten()
                if local_ids.numel() == 0:
                    continue
                sample_ids = ids[local_ids]
                target_indices = torch.tensor(
                    [current_decisions[local].target_index for local in local_ids.tolist()],
                    dtype=torch.long,
                    device=z_t.device,
                )
                target_dist = Categorical(
                    logits=edge_logits[operation]
                    .unsqueeze(0)
                    .expand(local_ids.numel(), -1)
                    .masked_fill(~masks[operation][sample_ids], -torch.inf)
                )
                log_probs[sample_ids] += target_dist.log_prob(target_indices)
                entropy_sums[sample_ids] += target_dist.entropy()
                masks[operation][sample_ids, target_indices] = False

            stop_local_ids = [
                local
                for local, decision in enumerate(current_decisions)
                if decision.stop_after is not None
            ]
            if stop_local_ids:
                local_tensor = torch.tensor(
                    stop_local_ids, dtype=torch.long, device=z_t.device
                )
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                stop_values = torch.tensor(
                    [float(current_decisions[local].stop_after) for local in stop_local_ids],
                    dtype=z_t.dtype,
                    device=z_t.device,
                )
                log_probs[ids[local_tensor]] += stop_dist.log_prob(stop_values)
                entropy_sums[ids[local_tensor]] += stop_dist.entropy()

        lengths = torch.tensor(
            [max(len(sample.decisions), 1) for sample in samples],
            dtype=z_t.dtype,
            device=z_t.device,
        )
        return log_probs, entropy_sums / lengths

    @torch.no_grad()
    def greedy_action_sequence(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        *,
        candidate_edges: Mapping[GraphOperation, torch.Tensor] | None = None,
        candidate_nodes: Mapping[GraphOperation, torch.Tensor] | None = None,
        valid_operations: Sequence[GraphOperation] | None = None,
        min_actions: int = 1,
        max_actions: int = 8,
        predict_magnitude: bool = True,
        stop_threshold: float = 0.5,
        edge_chunk_size: int = 65536,
    ) -> PolicySequenceSample:
        if min_actions < 1 or max_actions < min_actions:
            raise ValueError("Require 1 <= min_actions <= max_actions.")
        if not 0.0 <= stop_threshold <= 1.0:
            raise ValueError("stop_threshold must lie in [0, 1].")
        if edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive.")
        state = self.node_state(z_t, h_t)
        operations = tuple(
            GraphOperation(operation)
            for operation in (valid_operations or tuple(GraphOperation))
        )
        if not operations:
            raise ValueError("At least one valid operation is required.")

        edge_masks: dict[GraphOperation, torch.Tensor] = {}
        node_masks: dict[GraphOperation, torch.Tensor] = {}
        for operation in operations:
            if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
                candidates = (
                    None if candidate_edges is None else candidate_edges.get(operation)
                )
                if candidates is not None and candidates.shape[1]:
                    edge_masks[operation] = self._candidate_mask(
                        candidates.shape[1], device=z_t.device
                    )
            else:
                candidates = (
                    None if candidate_nodes is None else candidate_nodes.get(operation)
                )
                count = z_t.shape[0] if candidates is None else candidates.numel()
                if count:
                    node_masks[operation] = self._candidate_mask(
                        count, device=z_t.device
                    )

        decisions: list[SequenceDecision] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for step in range(max_actions):
            available_operations = [
                operation
                for operation in operations
                if (
                    bool(edge_masks[operation].any())
                    if operation in edge_masks
                    else bool(
                        node_masks.get(
                            operation,
                            torch.zeros((), dtype=torch.bool, device=z_t.device),
                        ).any()
                    )
                )
            ]
            if not available_operations:
                break

            operation_dist = self._operation_distribution(
                state, available_operations
            )
            operation = GraphOperation(int(torch.argmax(operation_dist.logits).item()))
            operation_index = torch.tensor(int(operation), device=z_t.device)
            log_prob = operation_dist.log_prob(operation_index)
            entropy = operation_dist.entropy()

            if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
                assert candidate_edges is not None
                candidates = candidate_edges[operation].to(
                    device=z_t.device, dtype=torch.long
                )
                mask = edge_masks[operation]
                target_logits = torch.cat(
                    [
                        self._edge_logits(state, candidates[:, start:stop])
                        for start in range(0, candidates.shape[1], edge_chunk_size)
                        for stop in [min(start + edge_chunk_size, candidates.shape[1])]
                    ]
                ).masked_fill(~mask, -torch.inf)
                target_dist = Categorical(logits=target_logits)
                target_index = torch.argmax(target_logits)
                edge = candidates[:, target_index]
                edit = GraphEditAction(
                    operation, (int(edge[0].item()), int(edge[1].item()))
                )
                log_prob = log_prob + target_dist.log_prob(target_index)
                entropy = entropy + target_dist.entropy()
                mask[target_index] = False
                includes_magnitude = False
            else:
                raw_candidates = (
                    None if candidate_nodes is None else candidate_nodes.get(operation)
                )
                candidates = (
                    torch.arange(z_t.shape[0], device=z_t.device)
                    if raw_candidates is None
                    else raw_candidates.to(device=z_t.device, dtype=torch.long)
                )
                mask = node_masks[operation]
                target_logits = self.node_target_head(state).squeeze(-1)[
                    candidates
                ].masked_fill(~mask, -torch.inf)
                target_dist = Categorical(logits=target_logits)
                target_index = torch.argmax(target_logits)
                node_index = candidates[target_index]
                mu, _ = self._magnitude_distribution(state, operation)
                if predict_magnitude:
                    value = mu[node_index].detach()
                    includes_magnitude = True
                else:
                    value = torch.zeros_like(mu[node_index])
                    includes_magnitude = False
                edit = GraphEditAction(
                    operation, (int(node_index.item()),), value
                )
                log_prob = log_prob + target_dist.log_prob(target_index)
                entropy = entropy + target_dist.entropy()
                mask[target_index] = False

            stop_after: bool | None = None
            can_continue = step + 1 < max_actions and any(
                bool(mask.any())
                for mask in [*edge_masks.values(), *node_masks.values()]
            )
            if step + 1 >= min_actions and can_continue:
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                stop_after = bool(stop_dist.probs.item() >= stop_threshold)
                stop_value = z_t.new_tensor(float(stop_after))
                log_prob = log_prob + stop_dist.log_prob(stop_value)
                entropy = entropy + stop_dist.entropy()

            decisions.append(
                SequenceDecision(
                    edit,
                    int(target_index.item()),
                    includes_magnitude,
                    stop_after,
                )
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
            if stop_after:
                break

        if not decisions:
            raise RuntimeError("Controller could not decode a valid graph edit.")
        return PolicySequenceSample(
            GraphEditSequence(tuple(decision.action for decision in decisions)),
            decisions[0].action.operation,
            tuple(decisions),
            torch.stack(log_probs).sum().detach(),
            torch.stack(entropies).mean().detach(),
        )

    def evaluate_action_sequence(
        self,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        sample: PolicySequenceSample,
        *,
        candidate_edges: Mapping[GraphOperation, torch.Tensor] | None = None,
        candidate_nodes: Mapping[GraphOperation, torch.Tensor] | None = None,
        valid_operations: Sequence[GraphOperation] | None = None,
        max_actions: int,
        node_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.node_state(z_t, h_t) if node_state is None else node_state
        operations = tuple(
            GraphOperation(operation)
            for operation in (valid_operations or tuple(GraphOperation))
        )
        operation_logits = self.operation_logits(state)
        node_target_logits = self.node_target_head(state).squeeze(-1)
        edge_target_logits: dict[GraphOperation, torch.Tensor] = {}
        magnitude_cache: dict[GraphOperation, tuple[torch.Tensor, torch.Tensor]] = {}
        edge_masks: dict[GraphOperation, torch.Tensor] = {}
        node_masks: dict[GraphOperation, torch.Tensor] = {}
        for operation in operations:
            if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
                candidates = None if candidate_edges is None else candidate_edges.get(operation)
                if candidates is not None and candidates.shape[1]:
                    edge_masks[operation] = self._candidate_mask(
                        candidates.shape[1], device=z_t.device
                    )
            else:
                candidates = None if candidate_nodes is None else candidate_nodes.get(operation)
                count = z_t.shape[0] if candidates is None else candidates.numel()
                if count:
                    node_masks[operation] = self._candidate_mask(count, device=z_t.device)

        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for step, decision in enumerate(sample.decisions):
            available_operations = [
                operation
                for operation in operations
                if (
                    bool(edge_masks[operation].any())
                    if operation in edge_masks
                    else bool(node_masks.get(operation, torch.zeros((), dtype=torch.bool)).any())
                )
            ]
            operation = decision.action.operation
            operation_dist = self._operation_distribution(
                state,
                available_operations,
                precomputed_logits=operation_logits,
            )
            operation_index = torch.tensor(int(operation), device=z_t.device)
            log_prob = operation_dist.log_prob(operation_index)
            entropy = operation_dist.entropy()
            target_index = torch.tensor(decision.target_index, device=z_t.device)

            if operation in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
                assert candidate_edges is not None
                candidates = candidate_edges[operation].to(
                    device=z_t.device, dtype=torch.long
                )
                mask = edge_masks[operation]
                if operation not in edge_target_logits:
                    edge_target_logits[operation] = self._edge_logits(state, candidates)
                target_dist = Categorical(
                    logits=edge_target_logits[operation].masked_fill(~mask, -torch.inf)
                )
                log_prob = log_prob + target_dist.log_prob(target_index)
                entropy = entropy + target_dist.entropy()
                mask[target_index] = False
            else:
                raw_candidates = None if candidate_nodes is None else candidate_nodes.get(operation)
                candidates = (
                    torch.arange(z_t.shape[0], device=z_t.device)
                    if raw_candidates is None
                    else raw_candidates.to(device=z_t.device, dtype=torch.long)
                )
                mask = node_masks[operation]
                target_dist = Categorical(
                    logits=node_target_logits[candidates].masked_fill(~mask, -torch.inf)
                )
                node_index = candidates[target_index]
                log_prob = log_prob + target_dist.log_prob(target_index)
                entropy = entropy + target_dist.entropy()
                if decision.includes_magnitude:
                    if operation not in magnitude_cache:
                        magnitude_cache[operation] = self._magnitude_distribution(state, operation)
                    mu, logstd = magnitude_cache[operation]
                    value = decision.action.value
                    assert value is not None
                    value_dist = Normal(mu[node_index], logstd[node_index].exp())
                    value = value.to(device=z_t.device, dtype=z_t.dtype)
                    log_prob = log_prob + value_dist.log_prob(value).mean()
                    entropy = entropy + value_dist.entropy().mean()
                mask[target_index] = False

            if decision.stop_after is not None:
                stop_dist = self._stop_distribution(
                    state, length=step + 1, max_actions=max_actions
                )
                stop_value = z_t.new_tensor(float(decision.stop_after))
                log_prob = log_prob + stop_dist.log_prob(stop_value)
                entropy = entropy + stop_dist.entropy()
            log_probs.append(log_prob)
            entropies.append(entropy)
        return torch.stack(log_probs).sum(), torch.stack(entropies).mean()


def structure_node_weights(
    edge_index_t: torch.Tensor,
    *,
    num_nodes: int,
    historical_change_counts: torch.Tensor | None = None,
    history_steps: int = 0,
    degree_power: float = 1.0,
    volatility_power: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive.")
    if degree_power <= 0 or volatility_power <= 0:
        raise ValueError("degree_power and volatility_power must be positive.")
    if history_steps < 0:
        raise ValueError("history_steps must be non-negative.")
    device = edge_index_t.device
    degree = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    if edge_index_t.numel():
        ones = torch.ones(edge_index_t.shape[1], device=device)
        degree.index_add_(0, edge_index_t[0].long(), ones)
        degree.index_add_(0, edge_index_t[1].long(), ones)
    degree_score = (degree + eps).pow(float(degree_power))

    if historical_change_counts is None:
        change_counts = torch.zeros(num_nodes, device=device)
    else:
        if historical_change_counts.shape != (num_nodes,):
            raise ValueError("historical_change_counts must have shape [num_nodes].")
        change_counts = historical_change_counts.to(device=device, dtype=torch.float32)
    frequency = (change_counts + eps) / (float(history_steps) + eps)
    volatility_score = frequency.pow(float(volatility_power))
    combined = degree_score * volatility_score
    return combined / (combined.sum() + eps)


def _unique_codes(items: torch.Tensor, *, num_nodes: int, item_type: str) -> torch.Tensor:
    if item_type == "node":
        return torch.unique(items.to(torch.long).reshape(-1))
    if item_type != "edge":
        raise ValueError("item_type must be node or edge.")
    if items.ndim != 2 or items.shape[0] != 2:
        raise ValueError("Edge items must have shape [2, num_edges].")
    return torch.unique(items[0].long() * int(num_nodes) + items[1].long())


def structure_weighted_f1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    node_weights: torch.Tensor,
    item_type: str,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    num_nodes = int(node_weights.shape[0])
    predicted_codes = _unique_codes(predicted, num_nodes=num_nodes, item_type=item_type)
    target_codes = _unique_codes(target, num_nodes=num_nodes, item_type=item_type)
    if item_type == "node":
        predicted_weight = node_weights[predicted_codes]
        target_weight = node_weights[target_codes]
    else:
        predicted_weight = 0.5 * (
            node_weights[predicted_codes // num_nodes]
            + node_weights[predicted_codes % num_nodes]
        )
        target_weight = 0.5 * (
            node_weights[target_codes // num_nodes]
            + node_weights[target_codes % num_nodes]
        )
    if predicted_codes.numel() == 0 and target_codes.numel() == 0:
        one = node_weights.new_ones(())
        return {"precision": one, "recall": one, "f1": one}
    if predicted_codes.numel() and target_codes.numel():
        correct = torch.isin(predicted_codes, target_codes)
        true_positive = predicted_weight[correct].sum()
    else:
        true_positive = node_weights.new_zeros(())
    precision = true_positive / predicted_weight.sum().clamp_min(eps)
    recall = true_positive / target_weight.sum().clamp_min(eps)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(eps)
    return {"precision": precision, "recall": recall, "f1": f1}


def topology_evolution_reward(
    *,
    predicted_add: torch.Tensor,
    target_add: torch.Tensor,
    predicted_remove: torch.Tensor,
    target_remove: torch.Tensor,
    node_weights: torch.Tensor,
    addition_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= addition_weight <= 1.0:
        raise ValueError("addition_weight must lie in [0, 1].")
    add = structure_weighted_f1(
        predicted_add, target_add, node_weights=node_weights, item_type="edge"
    )
    remove = structure_weighted_f1(
        predicted_remove, target_remove, node_weights=node_weights, item_type="edge"
    )
    reward = float(addition_weight) * add["f1"] + (1.0 - float(addition_weight)) * remove["f1"]
    return {"reward": reward, "addition_f1": add["f1"], "deletion_f1": remove["f1"]}


def node_change_localization_reward(
    *,
    predicted_nodes: torch.Tensor,
    target_changed: torch.Tensor,
    node_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_nodes = torch.nonzero(target_changed.to(torch.bool), as_tuple=False).flatten()
    metrics = structure_weighted_f1(
        predicted_nodes, target_nodes, node_weights=node_weights, item_type="node"
    )
    return {"reward": metrics["f1"], **metrics}


def node_state_magnitude_reward(
    *,
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    target_changed: torch.Tensor,
    node_weights: torch.Tensor,
    selected_nodes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    changed = target_changed.to(torch.bool)
    if selected_nodes is not None:
        selected = torch.zeros_like(changed)
        selected[selected_nodes.to(device=changed.device, dtype=torch.long)] = True
        changed = changed & selected
        if not bool(changed.any()):
            zero = predicted_delta.new_zeros(())
            return {
                "reward": zero,
                "error": predicted_delta.new_full((), float("inf")),
            }
    if not bool(changed.any()):
        error = predicted_delta.abs().mean()
    else:



        row_error = (
            predicted_delta[changed] - target_delta[changed]
        ).abs().mean(dim=-1)
        weights = node_weights[changed]
        error = (row_error * weights).sum() / weights.sum().clamp_min(1e-8)
    return {"reward": torch.exp(-error), "error": error}


def representation_change_magnitude_reward(
    *,
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    item_weights: torch.Tensor,
    target_changed: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:

    if target_delta.ndim != 2 or predicted_delta.shape[-2:] != target_delta.shape:
        raise ValueError(
            "Predicted and target representation changes must share item and "
            "feature dimensions."
        )
    if predicted_delta.ndim < 2:
        raise ValueError("Representation changes require an item and feature axis.")
    if predicted_delta.shape[-2] != item_weights.numel():
        raise ValueError("One importance weight is required per representation row.")
    changed = target_changed.to(
        device=predicted_delta.device, dtype=torch.bool
    ).reshape(-1)
    if changed.numel() != predicted_delta.shape[-2]:
        raise ValueError("The changed-item mask must match representation rows.")
    if not bool(changed.any()):
        zero = predicted_delta.new_zeros(predicted_delta.shape[:-2])
        return {"reward": torch.exp(-zero), "error": zero}

    row_error = (predicted_delta - target_delta).abs().mean(dim=-1)
    weights = item_weights.to(
        device=predicted_delta.device, dtype=predicted_delta.dtype
    )[changed]
    error = (
        (row_error[..., changed] * weights).sum(dim=-1)
        / weights.sum().clamp_min(eps)
    )
    return {"reward": torch.exp(-error), "error": error}


def edge_representation_change_reward(
    *,
    predicted_node_next: torch.Tensor,
    current_node_embedding: torch.Tensor,
    target_node_next: torch.Tensor,
    current_edges: torch.Tensor,
    next_edges: torch.Tensor,
    node_weights: torch.Tensor,
    change_epsilon: float = 1e-6,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:

    if current_edges.ndim != 2 or current_edges.shape[0] != 2:
        raise ValueError("current_edges must have shape [2, num_edges].")
    if next_edges.ndim != 2 or next_edges.shape[0] != 2:
        raise ValueError("next_edges must have shape [2, num_edges].")
    if current_node_embedding.shape != target_node_next.shape:
        raise ValueError("Current and target node embeddings must match.")
    if predicted_node_next.shape[-2:] != current_node_embedding.shape:
        raise ValueError("Predicted next-node embeddings must match target nodes.")

    num_nodes = int(current_node_embedding.shape[0])
    current_codes = _unique_codes(
        current_edges, num_nodes=num_nodes, item_type="edge"
    )
    next_codes = _unique_codes(next_edges, num_nodes=num_nodes, item_type="edge")
    persistent_codes = current_codes[torch.isin(current_codes, next_codes)]
    batch_shape = predicted_node_next.shape[:-2]
    if persistent_codes.numel() == 0:
        one = predicted_node_next.new_ones(batch_shape)
        zero = predicted_node_next.new_zeros(batch_shape)
        return {
            "reward": one,
            "localization_f1": one,
            "magnitude_reward": one,
            "magnitude_error": zero,
            "changed_count": zero,
        }

    source = persistent_codes // num_nodes
    destination = persistent_codes % num_nodes
    current_edge_embedding = torch.cat(
        (current_node_embedding[source], current_node_embedding[destination]),
        dim=-1,
    )
    target_edge_embedding = torch.cat(
        (target_node_next[source], target_node_next[destination]), dim=-1
    )
    predicted_edge_embedding = torch.cat(
        (
            predicted_node_next[..., source, :],
            predicted_node_next[..., destination, :],
        ),
        dim=-1,
    )
    target_delta = target_edge_embedding - current_edge_embedding
    predicted_delta = predicted_edge_embedding - current_edge_embedding
    target_changed = target_delta.abs().amax(dim=-1).gt(float(change_epsilon))
    changed_count = int(target_changed.sum().item())

    edge_weights = 0.5 * (
        node_weights[source] + node_weights[destination]
    ).to(device=predicted_node_next.device, dtype=predicted_node_next.dtype)
    magnitude = representation_change_magnitude_reward(
        predicted_delta=predicted_delta,
        target_delta=target_delta,
        item_weights=edge_weights,
        target_changed=target_changed,
        eps=eps,
    )
    if changed_count == 0:
        localization = predicted_node_next.new_ones(batch_shape)
    else:
        predicted_score = predicted_delta.abs().mean(dim=-1)
        selected = torch.topk(
            predicted_score, k=changed_count, dim=-1
        ).indices
        selected_weight = edge_weights[selected]
        selected_correct = target_changed[selected]
        true_positive = (
            selected_weight * selected_correct.to(selected_weight.dtype)
        ).sum(dim=-1)
        predicted_mass = selected_weight.sum(dim=-1)
        target_mass = edge_weights[target_changed].sum()
        precision = true_positive / predicted_mass.clamp_min(eps)
        recall = true_positive / target_mass.clamp_min(eps)
        localization = (
            2.0 * precision * recall / (precision + recall).clamp_min(eps)
        )
    reward = 0.5 * (localization + magnitude["reward"])
    return {
        "reward": reward,
        "localization_f1": localization,
        "magnitude_reward": magnitude["reward"],
        "magnitude_error": magnitude["error"],
        "changed_count": predicted_node_next.new_full(
            batch_shape, float(changed_count)
        ),
    }


def _ndcg_at_k(prediction: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(prediction.shape[-1]))
    predicted_order = torch.topk(prediction, k=k, dim=-1).indices
    ideal_order = torch.topk(target, k=k, dim=-1).indices
    predicted_gain = torch.gather(target, -1, predicted_order)
    ideal_gain = torch.gather(target, -1, ideal_order)
    discount = 1.0 / torch.log2(
        torch.arange(2, k + 2, device=prediction.device, dtype=prediction.dtype)
    )
    dcg = (predicted_gain * discount).sum(dim=-1)
    idcg = (ideal_gain * discount).sum(dim=-1)
    return torch.where(idcg > 0, dcg / idcg.clamp_min(1e-8), torch.zeros_like(dcg))


def structure_weighted_ndcg(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    row_weights: torch.Tensor,
    k: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target property rows must match.")
    if prediction.shape[0] != row_weights.numel():
        raise ValueError("One structural weight is required per property row.")
    if prediction.shape[0] == 0:
        return prediction.new_zeros(())
    ndcg = _ndcg_at_k(prediction, target, k)
    weights = row_weights.to(device=prediction.device, dtype=prediction.dtype)
    return (ndcg * weights).sum() / weights.sum().clamp_min(eps)


def node_property_reward(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    labelled_nodes: torch.Tensor,
    node_weights: torch.Tensor,
    mode: str = "ranking",
    k: int = 10,
    selected_nodes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:



    if labelled_nodes.dtype == torch.bool:
        labelled = labelled_nodes.to(device=prediction.device)
        row_ids = torch.nonzero(labelled, as_tuple=False).flatten()
        prediction_rows = prediction.index_select(0, row_ids)
        target_rows = target.index_select(0, row_ids)
    else:
        row_ids = labelled_nodes.to(device=prediction.device, dtype=torch.long).reshape(-1)
        prediction_rows = prediction.index_select(0, row_ids)
        if target.shape[0] == row_ids.numel():
            target_rows = target
        else:
            target_rows = target.index_select(0, row_ids)
    if selected_nodes is not None and row_ids.numel():
        keep = torch.isin(
            row_ids,
            selected_nodes.to(device=row_ids.device, dtype=torch.long),
        )
        row_ids = row_ids[keep]
        prediction_rows = prediction_rows[keep]
        target_rows = target_rows[keep]
    if not row_ids.numel():
        zero = prediction.new_zeros(())
        return {"reward": zero, "error": zero}
    weights = node_weights.index_select(0, row_ids)
    if mode == "ranking":
        rows = _ndcg_at_k(prediction_rows, target_rows, k)
        reward = (rows * weights).sum() / weights.sum().clamp_min(1e-8)
        return {"reward": reward, "ndcg": reward}
    if mode != "regression":
        raise ValueError("Property reward mode must be ranking or regression.")
    row_error = F.smooth_l1_loss(
        prediction_rows, target_rows, reduction="none"
    ).mean(dim=-1)
    error = (row_error * weights).sum() / weights.sum().clamp_min(1e-8)
    return {"reward": torch.exp(-error), "error": error}


def grpo_loss(
    controller: ActionAwareController,
    rollouts: Sequence[RewardedRollout],
    *,
    z_t: torch.Tensor,
    h_t: torch.Tensor,
    candidate_edges: Mapping[GraphOperation, torch.Tensor] | None = None,
    candidate_nodes: Mapping[GraphOperation, torch.Tensor] | None = None,
    valid_operations: Sequence[GraphOperation] | None = None,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.01,
    entropy_coefficient: float = 0.0,
) -> dict[str, torch.Tensor]:
    if not rollouts:
        raise ValueError("At least one rollout is required.")
    if clip_epsilon <= 0 or kl_coefficient < 0 or entropy_coefficient < 0:
        raise ValueError("Invalid GRPO coefficient.")
    grouped: dict[GraphOperation, list[int]] = {}
    for index, rollout in enumerate(rollouts):
        grouped.setdefault(rollout.sample.action.operation, []).append(index)
    advantages = torch.zeros(len(rollouts), device=z_t.device, dtype=z_t.dtype)



    rewards = torch.stack(
        [
            torch.as_tensor(rollout.reward, device=z_t.device, dtype=z_t.dtype)
            .reshape(())
            .detach()
            for rollout in rollouts
        ]
    )
    for indices in grouped.values():
        index_tensor = torch.tensor(indices, device=z_t.device)
        group_reward = rewards[index_tensor]
        advantages[index_tensor] = (
            group_reward - group_reward.mean()
        ) / group_reward.std(unbiased=False).clamp_min(1e-8)

    objectives: list[torch.Tensor] = []
    kls: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    for index, rollout in enumerate(rollouts):
        operation = rollout.sample.action.operation
        current_log_prob, entropy = controller.evaluate_action(
            z_t,
            h_t,
            rollout.sample,
            candidate_edges=None if candidate_edges is None else candidate_edges.get(operation),
            candidate_nodes=None if candidate_nodes is None else candidate_nodes.get(operation),
            valid_operations=valid_operations,
        )
        old_log_prob = rollout.sample.old_log_prob.to(
            device=z_t.device, dtype=z_t.dtype
        )
        ratio = torch.exp(current_log_prob - old_log_prob)
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objectives.append(
            torch.minimum(ratio * advantages[index], clipped * advantages[index])
        )

        log_ratio = old_log_prob - current_log_prob
        kls.append(torch.exp(log_ratio) - log_ratio - 1.0)
        entropies.append(entropy)
        ratios.append(ratio)
    objective = torch.stack(objectives).mean()
    kl = torch.stack(kls).mean()
    entropy = torch.stack(entropies).mean()
    loss = -objective + float(kl_coefficient) * kl - float(entropy_coefficient) * entropy
    return {
        "loss": loss,
        "objective": objective,
        "kl": kl,
        "entropy": entropy,
        "mean_reward": rewards.mean(),
        "mean_ratio": torch.stack(ratios).mean(),
    }


def grpo_sequence_loss(
    controller: ActionAwareController,
    rollouts: Sequence[RewardedSequenceRollout],
    *,
    z_t: torch.Tensor,
    h_t: torch.Tensor,
    candidate_edges: Mapping[GraphOperation, torch.Tensor] | None = None,
    candidate_nodes: Mapping[GraphOperation, torch.Tensor] | None = None,
    valid_operations: Sequence[GraphOperation] | None = None,
    max_actions: int,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.01,
    entropy_coefficient: float = 0.0,
    node_state: torch.Tensor | None = None,
    batch_edge_sequences: bool = False,
    batch_mixed_edge_sequences: bool = False,
) -> dict[str, torch.Tensor]:
    if not rollouts:
        raise ValueError("At least one sequence rollout is required.")
    grouped: dict[GraphOperation, list[int]] = {}
    for index, rollout in enumerate(rollouts):
        grouped.setdefault(rollout.sample.group, []).append(index)
    rewards = torch.tensor(
        [rollout.reward for rollout in rollouts], device=z_t.device, dtype=z_t.dtype
    )
    advantages = torch.zeros_like(rewards)
    for indices in grouped.values():
        index_tensor = torch.tensor(indices, device=z_t.device)
        values = rewards[index_tensor]
        advantages[index_tensor] = (values - values.mean()) / values.std(
            unbiased=False
        ).clamp_min(1e-8)

    can_batch_mixed_edge_sequences = (
        batch_mixed_edge_sequences
        and candidate_edges is not None
        and valid_operations is not None
        and all(
            decision.action.operation
            in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
            for rollout in rollouts
            for decision in rollout.sample.decisions
        )
    )
    if can_batch_mixed_edge_sequences:
        current_log_prob, entropy = (
            controller.evaluate_mixed_edge_action_sequences_batch(
                z_t,
                h_t,
                [rollout.sample for rollout in rollouts],
                candidate_edges=candidate_edges,
                valid_operations=valid_operations,
                max_actions=max_actions,
                node_state=node_state,
            )
        )
        old_log_prob = torch.stack(
            [
                rollout.sample.old_log_prob.to(device=z_t.device, dtype=z_t.dtype)
                for rollout in rollouts
            ]
        )
        ratios = torch.exp(current_log_prob - old_log_prob)
        clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objective = torch.minimum(
            ratios * advantages, clipped * advantages
        ).mean()
        log_ratio = old_log_prob - current_log_prob
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        entropy_value = entropy.mean()
        loss = (
            -objective
            + float(kl_coefficient) * kl
            - float(entropy_coefficient) * entropy_value
        )
        return {
            "loss": loss,
            "objective": objective,
            "kl": kl,
            "entropy": entropy_value,
            "mean_reward": rewards.mean(),
            "mean_ratio": ratios.mean(),
        }

    can_batch_node_sequences = (
        len(rollouts) > 0
        and len(grouped) == 1
        and rollouts[0].sample.group
        not in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
    )
    if can_batch_node_sequences:
        current_log_prob, entropy = controller.evaluate_node_action_sequences_batch(
            z_t,
            h_t,
            [rollout.sample for rollout in rollouts],
            candidate_nodes=candidate_nodes.get(rollouts[0].sample.group)
            if candidate_nodes is not None
            else None,
            max_actions=max_actions,
            node_state=node_state,
        )
        old_log_prob = torch.stack(
            [rollout.sample.old_log_prob.to(device=z_t.device, dtype=z_t.dtype) for rollout in rollouts]
        )
        ratios = torch.exp(current_log_prob - old_log_prob)
        clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objective = torch.minimum(ratios * advantages, clipped * advantages).mean()
        log_ratio = old_log_prob - current_log_prob
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        entropy_value = entropy.mean()
        loss = -objective + float(kl_coefficient) * kl - float(entropy_coefficient) * entropy_value
        return {
            "loss": loss,
            "objective": objective,
            "kl": kl,
            "entropy": entropy_value,
            "mean_reward": rewards.mean(),
            "mean_ratio": ratios.mean(),
        }

    can_batch_edge_sequences = (
        batch_edge_sequences
        and len(rollouts) > 0
        and len(grouped) == 1
        and rollouts[0].sample.group
        in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
        and candidate_edges is not None
    )
    if can_batch_edge_sequences:
        operation = rollouts[0].sample.group
        current_log_prob, entropy = controller.evaluate_edge_action_sequences_batch(
            z_t,
            h_t,
            [rollout.sample for rollout in rollouts],
            candidate_edges=candidate_edges[operation],
            max_actions=max_actions,
            node_state=node_state,
        )
        old_log_prob = torch.stack(
            [
                rollout.sample.old_log_prob.to(device=z_t.device, dtype=z_t.dtype)
                for rollout in rollouts
            ]
        )
        ratios = torch.exp(current_log_prob - old_log_prob)
        clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objective = torch.minimum(ratios * advantages, clipped * advantages).mean()
        log_ratio = old_log_prob - current_log_prob
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        entropy_value = entropy.mean()
        loss = (
            -objective
            + float(kl_coefficient) * kl
            - float(entropy_coefficient) * entropy_value
        )
        return {
            "loss": loss,
            "objective": objective,
            "kl": kl,
            "entropy": entropy_value,
            "mean_reward": rewards.mean(),
            "mean_ratio": ratios.mean(),
        }

    objectives: list[torch.Tensor] = []
    kls: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    for index, rollout in enumerate(rollouts):
        current_log_prob, entropy = controller.evaluate_action_sequence(
            z_t,
            h_t,
            rollout.sample,
            candidate_edges=candidate_edges,
            candidate_nodes=candidate_nodes,
            valid_operations=valid_operations,
            max_actions=max_actions,
            node_state=node_state,
        )
        old_log_prob = rollout.sample.old_log_prob.to(
            device=z_t.device, dtype=z_t.dtype
        )
        ratio = torch.exp(current_log_prob - old_log_prob)
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objectives.append(
            torch.minimum(ratio * advantages[index], clipped * advantages[index])
        )
        log_ratio = old_log_prob - current_log_prob
        kls.append(torch.exp(log_ratio) - log_ratio - 1.0)
        entropies.append(entropy)
        ratios.append(ratio)
    objective = torch.stack(objectives).mean()
    kl = torch.stack(kls).mean()
    entropy = torch.stack(entropies).mean()
    loss = -objective + float(kl_coefficient) * kl - float(entropy_coefficient) * entropy
    return {
        "loss": loss,
        "objective": objective,
        "kl": kl,
        "entropy": entropy,
        "mean_reward": rewards.mean(),
        "mean_ratio": torch.stack(ratios).mean(),
    }


def edge_index_from_action(action: GraphEditAction, *, device: torch.device) -> torch.Tensor:
    if action.operation not in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(action.target, dtype=torch.long, device=device).reshape(2, 1)
