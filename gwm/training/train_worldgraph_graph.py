

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwm.action_rl import (
    ActionAwareController,
    DynamicGroupSampler,
    RewardedSequenceRollout,
    grpo_sequence_loss,
    node_change_localization_reward,
    node_state_magnitude_reward,
    structure_node_weights,
    structure_weighted_f1,
)
from gwm.actions import GraphActionEncoder, GraphOperation
from gwm.data.transition_dataset import GraphTransitionDataset
from gwm.model import GraphWorldModel
from gwm.pretraining import (
    load_worldgraph_pretrained_backbone,
    load_worldgraph_pretrained_t3_head,
)
from gwm.t3_structure import (
    DESCRIPTOR_NAMES,
    TargetCache,
    change_regression_metrics,
    classes,
    fit_statistics,
    macro_f1,
    regression_metrics,
    temporal_change_macro_f1,
)
from gwm.utils import resolve_device, seed_everything


class StructuralHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        decoupled_direction_head: bool = False,
        direction_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(DESCRIPTOR_NAMES)),
        )
        self.direction_network = (
            nn.Sequential(
                nn.Linear(direction_input_dim or input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3 * len(DESCRIPTOR_NAMES)),
            )
            if decoupled_direction_head
            else None
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)

    def direction_logits(self, value: torch.Tensor) -> torch.Tensor | None:
        if self.direction_network is None:
            return None
        return self.direction_network(value).view(
            value.shape[0], len(DESCRIPTOR_NAMES), 3
        )


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


T3_OPERATIONS = (
    GraphOperation.ADD_NODE,
    GraphOperation.REMOVE_NODE,
    GraphOperation.ADD_EDGE,
    GraphOperation.REMOVE_EDGE,
)
T3_NODE_OPERATIONS = frozenset(
    (GraphOperation.ADD_NODE, GraphOperation.REMOVE_NODE)
)
T3_EDGE_OPERATIONS = frozenset(
    (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE)
)


def _controller_key(operation: GraphOperation) -> str:
    return GraphOperation(operation).name.lower()


@contextmanager
def _preserve_world_rng(device: torch.device):
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


def _edge_weight(transition: dict[str, Any], suffix: str = "t") -> torch.Tensor | None:
    value = transition.get(f"edge_weight_{suffix}")
    return value if torch.is_tensor(value) and value.numel() else None


def _canonical_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    source = edge_index[0].long()
    destination = edge_index[1].long()
    keep = source.ne(destination)
    low = torch.minimum(source[keep], destination[keep])
    high = torch.maximum(source[keep], destination[keep])
    codes = torch.unique(low * int(num_nodes) + high)
    return torch.stack((codes // int(num_nodes), codes % int(num_nodes)))


def _active_nodes(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    active = torch.zeros(int(num_nodes), dtype=torch.bool, device=edge_index.device)
    if edge_index.numel():
        active.index_fill_(0, edge_index.reshape(-1).long(), True)
    return active


def _current_action_candidates(
    edge_index_t: torch.Tensor,
    *,
    num_nodes: int,
    historical_edge_codes: set[int],
    max_addition_candidates: int,
    generator: torch.Generator,
    active_override: torch.Tensor | None = None,
) -> tuple[
    dict[GraphOperation, torch.Tensor],
    dict[GraphOperation, torch.Tensor],
    torch.Tensor,
]:

    current = _canonical_edges(edge_index_t.detach().cpu(), num_nodes)
    active = (
        active_override.detach().cpu().to(dtype=torch.bool)
        if active_override is not None
        else _active_nodes(current, num_nodes)
    )
    if active.shape != (int(num_nodes),):
        raise ValueError("active_override must have shape [num_nodes].")
    current_codes = set(
        (current[0] * int(num_nodes) + current[1]).tolist()
    )
    node_candidates = {
        GraphOperation.ADD_NODE: torch.where(~active)[0],
        GraphOperation.REMOVE_NODE: torch.where(active)[0],
    }

    budget = max(int(max_addition_candidates), 1)
    selected_codes: set[int] = set()
    selected_pairs: list[tuple[int, int]] = []
    for code in sorted(historical_edge_codes):
        if code in current_codes:
            continue
        source, destination = divmod(int(code), int(num_nodes))
        if source == destination or not bool(active[source] and active[destination]):
            continue
        selected_codes.add(code)
        selected_pairs.append((source, destination))
        if len(selected_pairs) >= budget:
            break

    active_ids = torch.where(active)[0]
    legal_pairs = int(active_ids.numel()) * max(int(active_ids.numel()) - 1, 0) // 2
    available = max(0, legal_pairs - len(current_codes))
    target_count = min(budget, available)
    attempts = 0
    maximum_attempts = max(4096, 40 * max(target_count, 1))
    while len(selected_pairs) < target_count and attempts < maximum_attempts:
        remaining = target_count - len(selected_pairs)
        draw = min(262144, max(1024, 4 * remaining))
        left_index = torch.randint(
            max(int(active_ids.numel()), 1), (draw,), generator=generator
        )
        right_index = torch.randint(
            max(int(active_ids.numel()), 1), (draw,), generator=generator
        )
        if not active_ids.numel():
            break
        left = active_ids.index_select(0, left_index)
        right = active_ids.index_select(0, right_index)
        for source, destination in zip(left.tolist(), right.tolist()):
            if source == destination:
                continue
            if source > destination:
                source, destination = destination, source
            code = source * int(num_nodes) + destination
            if code in current_codes or code in selected_codes:
                continue
            selected_codes.add(code)
            selected_pairs.append((source, destination))
            if len(selected_pairs) >= target_count:
                break
        attempts += draw

    additions = (
        torch.tensor(selected_pairs, dtype=torch.long).t().contiguous()
        if selected_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return node_candidates, {
        GraphOperation.ADD_EDGE: additions,
        GraphOperation.REMOVE_EDGE: current,
    }, active


def _released_action_targets(
    transition: dict[str, Any],
    *,
    num_nodes: int,
    active_t: torch.Tensor,
) -> dict[GraphOperation, torch.Tensor]:

    next_edges = _canonical_edges(transition["edge_index_next"].detach().cpu(), num_nodes)
    active_next = _active_nodes(next_edges, num_nodes)
    added_nodes = torch.where((~active_t) & active_next)[0]
    removed_nodes = torch.where(active_t & (~active_next))[0]

    current_edges = _canonical_edges(transition["edge_index_t"].detach().cpu(), num_nodes)
    current_codes = current_edges[0] * int(num_nodes) + current_edges[1]
    next_codes = next_edges[0] * int(num_nodes) + next_edges[1]
    persistent = active_t & active_next
    added_edge_mask = (~torch.isin(next_codes, current_codes)) & persistent[
        next_edges[0]
    ] & persistent[next_edges[1]]
    removed_edge_mask = (~torch.isin(current_codes, next_codes)) & persistent[
        current_edges[0]
    ] & persistent[current_edges[1]]
    return {
        GraphOperation.ADD_NODE: added_nodes,
        GraphOperation.REMOVE_NODE: removed_nodes,
        GraphOperation.ADD_EDGE: next_edges[:, added_edge_mask],
        GraphOperation.REMOVE_EDGE: current_edges[:, removed_edge_mask],
    }


def _indices(dataset: GraphTransitionDataset, split: str) -> list[int]:
    return [i for i in range(len(dataset)) if str(dataset[i]["split"]) == split]


def _rotating_clip_indices(
    indices: list[int], steps: int, epoch: int, epochs: int
) -> list[int]:
    if not indices or steps <= 0 or steps >= len(indices):
        return list(indices)
    max_start = len(indices) - steps
    denominator = max(int(epochs) - 1, 1)
    start = round(max_start * max(int(epoch) - 1, 0) / denominator)
    return list(indices[start : start + steps])


def _evaluation_rollout_indices(
    dataset: GraphTransitionDataset,
    target_indices: list[int],
    history_window: int,
) -> tuple[list[int], set[int]]:
    if not target_indices:
        raise RuntimeError("No evaluation transitions are available.")
    start = max(0, int(target_indices[0]) - max(0, int(history_window) - 1))
    return list(range(start, int(target_indices[-1]) + 1)), set(target_indices)


def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    positive = labels.sum()
    negative = labels.numel() - positive
    if positive > 0 and negative > 0:
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=(negative / positive).clamp(max=50.0))
    return F.binary_cross_entropy_with_logits(logits, labels)


def _direction_classification_loss(
    prediction: torch.Tensor,
    target_classes: torch.Tensor,
    tolerance: torch.Tensor,
) -> torch.Tensor:
    scale = tolerance.to(device=prediction.device, dtype=prediction.dtype).clamp_min(1e-4)
    normalized = prediction / scale


    logits = torch.stack(
        (-normalized, 1.0 - normalized.abs(), normalized), dim=-1
    )
    return F.cross_entropy(logits.reshape(-1, 3), target_classes.reshape(-1))


def _balanced_direction_head_loss(
    logits: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    balance_power: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for descriptor in range(target_classes.shape[-1]):
        target = target_classes[:, descriptor]
        counts = torch.bincount(target, minlength=3).to(logits.dtype)
        present = counts > 0
        weights = torch.zeros_like(counts)
        if bool(present.any()):
            weights[present] = counts[present].clamp_min(1.0).pow(
                -float(balance_power)
            )
            weights[present] = weights[present] / weights[present].mean()
        losses.append(
            F.cross_entropy(logits[:, descriptor], target, weight=weights)
        )
    return torch.stack(losses).mean()


def _project_prediction_to_direction(
    prediction: torch.Tensor,
    direction_logits: torch.Tensor,
    tolerance: torch.Tensor,
    *,
    margin: float,
    confidence_threshold: float,
) -> torch.Tensor:
    probabilities = direction_logits.softmax(dim=-1)
    confidence, predicted_class = probabilities.max(dim=-1)
    boundary = tolerance.to(
        device=prediction.device, dtype=prediction.dtype
    ).view(1, -1)
    outer = boundary * (1.0 + max(float(margin), 1e-4))
    inner = boundary * (1.0 - min(max(float(margin), 1e-4), 0.99))
    decreased = torch.minimum(prediction, -outer)
    unchanged = prediction.clamp(min=-inner, max=inner)
    increased = torch.maximum(prediction, outer)
    projected = torch.where(
        predicted_class == 0,
        decreased,
        torch.where(predicted_class == 2, increased, unchanged),
    )
    return torch.where(
        confidence >= float(confidence_threshold), projected, prediction
    )


def _candidate_labels(
    candidates: torch.Tensor,
    target: torch.Tensor,
    *,
    num_nodes: int,
    item_type: str,
) -> torch.Tensor:
    if item_type == "node":
        return torch.isin(candidates.long(), target.long())
    if item_type != "edge":
        raise ValueError("item_type must be node or edge.")
    candidate_codes = candidates[0].long() * int(num_nodes) + candidates[1].long()
    target_codes = target[0].long() * int(num_nodes) + target[1].long()
    return torch.isin(candidate_codes, target_codes)


def _make_model(
    input_dim: int, args: argparse.Namespace
) -> tuple[GraphWorldModel, nn.ModuleDict, GraphActionEncoder, StructuralHead]:
    model = GraphWorldModel(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        action_dim=args.action_dim,


        input_adapter_type=(
            "residual_domain_invariant"
            if args.pretrained_backbone and args.pretrained_transfer_mode == "legacy"
            else "none"
        ),
        latent_distribution="gaussian",
        observable_latent_mode="mean",
        graph_encoder_type="mentor_sgt_gwm",
        state_model_type="mentor_sgt_gwm_transformer",
        sgt_num_hops=args.sgt_num_hops,
        sgt_num_walks=args.sgt_num_walks,
        sgt_walk_length=args.sgt_walk_length,
        sgt_attention_dropout=args.sgt_attention_dropout,
        sgt_topology_cache_bytes=args.sgt_topology_cache_bytes,
        sgt_deterministic_walks=True,
        history_window=args.history_window,
        history_num_heads=args.history_num_heads,
        latent_normalization="layernorm",
        target_encoder_momentum=args.target_encoder_momentum,
        separate_action_query=True,
        bounded_action_residual=True,
        zero_init_action_adapters=True,
        action_adapter_initial_scale=args.action_adapter_initial_scale,
        action_residual_max_scale=args.action_residual_max_scale,
        action_adapter_temperature=args.action_adapter_temperature,
    )
    controllers = nn.ModuleDict(
        {
            _controller_key(operation): ActionAwareController(
                args.latent_dim,
                args.hidden_dim,
                policy_dim=args.policy_dim,
                node_state_dim=input_dim,
            )
            for operation in T3_OPERATIONS
        }
    )
    action_encoder = GraphActionEncoder(
        args.latent_dim,
        args.action_dim,
        node_state_dim=input_dim,
    )



    head_input_dim = 2 * args.latent_dim + args.hidden_dim
    if getattr(args, "causal_structure_context", False):


        head_input_dim += 2 * len(DESCRIPTOR_NAMES) + 1
    if getattr(args, "causal_structure_history_context", False):



        head_input_dim += len(DESCRIPTOR_NAMES)
        head_input_dim += args.history_window * (len(DESCRIPTOR_NAMES) + 1)
    direction_input_dim = head_input_dim
    if getattr(args, "group_aware_decoder_context", False):
        group_context_dim = len(T3_OPERATIONS) * args.action_dim
        if getattr(args, "direction_group_aware_context_only", False):
            direction_input_dim += group_context_dim
        else:
            head_input_dim += group_context_dim
            direction_input_dim = head_input_dim
    head = StructuralHead(
        input_dim=head_input_dim,
        hidden_dim=args.latent_dim,
        decoupled_direction_head=getattr(args, "decoupled_direction_head", False),
        direction_input_dim=direction_input_dim,
    )
    if getattr(args, "pretrained_transfer_gate", False):


        with torch.random.fork_rng(devices=[]):
            model.graph_transfer_adapter = GatedResidualTransferAdapter(
                args.latent_dim,
                bottleneck=args.pretrained_transfer_gate_bottleneck,
                initial_gate=args.pretrained_transfer_gate_initial,
            )
            model.state_transfer_adapter = GatedResidualTransferAdapter(
                args.hidden_dim,
                bottleneck=args.pretrained_transfer_gate_bottleneck,
                initial_gate=args.pretrained_transfer_gate_initial,
            )
    return model, controllers, action_encoder, head


def _forward(
    model: GraphWorldModel,
    controllers: nn.ModuleDict,
    action_encoder: GraphActionEncoder,
    transition: dict[str, Any],
    hidden: torch.Tensor,
    candidate_nodes: dict[GraphOperation, torch.Tensor],
    candidate_edges: dict[GraphOperation, torch.Tensor],
    *,
    train: bool,
    max_actions: int,
    action_group_fusion: str = "mean",
    reuse_controller_policy_states: bool = False,
    direction_transfer_bypass: bool = False,
    direction_transfer_mix: float = 1.0,
    decode_node_state: bool = False,
    component_times: dict[str, float] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[GraphOperation, torch.Tensor],
]:

    def boundary(name: str, started_at: float) -> float:
        if component_times is None:
            return 0.0
        if transition["x_t"].device.type == "cuda":
            torch.cuda.synchronize(transition["x_t"].device)
        now = time.perf_counter()
        component_times[name] += now - started_at
        return now

    started_at = time.perf_counter() if component_times is not None else 0.0
    x_t = transition["x_t"]
    edge_t = transition["edge_index_t"]
    z_raw_base = model.encode_observed_graph(
        x_t,
        edge_t,
        _edge_weight(transition),
        apply_dropout=train,
        topology_cache_key=int(transition["transition_id"]),
    )
    z_raw = model.adapt_graph_output(z_raw_base)
    z_t = model._normalize_latent(z_raw)
    started_at = boundary("graph_encoder", started_at)
    encoded_actions: dict[GraphOperation, torch.Tensor] = {}
    training_policy_states: dict[GraphOperation, torch.Tensor] = {}

    def proposal_policy_state(operation: GraphOperation) -> torch.Tensor:
        controller = controllers[_controller_key(operation)]
        if train and reuse_controller_policy_states:




            state = controller.node_state(z_t.detach(), hidden.detach())
            training_policy_states[operation] = state
            return state.detach()
        with torch.no_grad():
            return controller.node_state(z_t.detach(), hidden.detach())



    for operation in T3_NODE_OPERATIONS:
        controller = controllers[_controller_key(operation)]
        candidates = candidate_nodes[operation].to(
            device=z_t.device, dtype=torch.long
        )
        if not candidates.numel():
            continue
        policy_state = proposal_policy_state(operation)
        with torch.no_grad():
            candidate_logits = controller.node_target_head(policy_state).squeeze(-1).index_select(
                0, candidates
            )
            probability = z_t.new_zeros(z_t.shape[0])
            probability.index_copy_(0, candidates, torch.sigmoid(candidate_logits))
        encoded_actions[operation] = action_encoder.encode_soft_node_action(
                z_t,
                probability,
                operation,
                localization_mode="local",
                support_count=min(max_actions, int(candidates.numel())),
            )["nodewise"]
    started_at = boundary("node_action_proposals", started_at)

    for operation in T3_EDGE_OPERATIONS:
        controller = controllers[_controller_key(operation)]
        candidates = candidate_edges[operation].to(
            device=z_t.device, dtype=torch.long
        )
        if candidates.shape[1] == 0:
            continue
        policy_state = proposal_policy_state(operation)
        with torch.no_grad():
            probability = torch.sigmoid(
                controller.edge_logits_chunked(policy_state, candidates)
            )
            support = min(max_actions, int(candidates.shape[1]))
            if probability.numel() > support:
                keep = probability.topk(support, sorted=False).indices
                sparse_probability = torch.zeros_like(probability)
                sparse_probability.index_copy_(0, keep, probability.index_select(0, keep))
                probability = sparse_probability
            count_index = 0 if operation == GraphOperation.ADD_EDGE else 1
            predicted_log_count = controller.topology_log_counts(policy_state)[count_index]
        encoded_actions[operation] = action_encoder.encode_soft_edge_actions(
                z_t,
                {operation: candidates},
                {operation: probability},
                operation_log_counts={operation: predicted_log_count},
            )["nodewise"]
    started_at = boundary("edge_action_proposals", started_at)
    if encoded_actions:
        stacked_actions = torch.stack(list(encoded_actions.values()), dim=0)
        if action_group_fusion == "mean":
            action = stacked_actions.mean(dim=0)
        elif action_group_fusion == "sqrt_sum":


            action = stacked_actions.sum(dim=0) / float(len(encoded_actions)) ** 0.5
        elif action_group_fusion == "sum":
            action = stacked_actions.sum(dim=0)
        else:
            raise ValueError(f"Unknown T3 action-group fusion: {action_group_fusion}")
    else:
        action = z_t.new_zeros((z_t.shape[0], action_encoder.action_dim))
    started_at = boundary("action_fusion", started_at)
    outputs = model(
        x_t,
        edge_t,
        hidden,
        action=action,
        edge_weight_t=_edge_weight(transition),
        commit_history=True,
        precomputed_z_t_raw=z_raw,
        precomputed_z_t_raw_is_adapted=True,
        decode_node_state=decode_node_state,
        decode_observables=decode_node_state,
        decode_node_features=decode_node_state,
    )
    if direction_transfer_bypass or direction_transfer_mix < 1.0:




        base_hidden = outputs["hidden_next_pre_transfer"]
        outputs["direction_base_z_t"] = model._normalize_latent(z_raw_base)
        outputs["direction_base_hidden_next"] = base_hidden
        outputs["direction_base_latent_next"] = model.latent_predictor(
            base_hidden, sample=False
        )["mu"]
    boundary("history_state_forward", started_at)
    outputs["t3_group_action_context"] = torch.cat(
        [
            encoded_actions.get(
                operation,
                z_t.new_zeros((z_t.shape[0], action_encoder.action_dim)),
            )
            for operation in T3_OPERATIONS
        ],
        dim=-1,
    )
    return outputs, z_raw, z_t, training_policy_states


def _controller_training_terms(
    *,
    model: GraphWorldModel,
    controllers: nn.ModuleDict,
    z_t: torch.Tensor,
    hidden: torch.Tensor,
    candidate_nodes: dict[GraphOperation, torch.Tensor],
    candidate_edges: dict[GraphOperation, torch.Tensor],
    targets: dict[GraphOperation, torch.Tensor],
    transition: dict[str, Any],
    group_sampler: DynamicGroupSampler,
    historical_change_counts: torch.Tensor,
    history_steps: int,
    rollout_budget: int,
    degree_power: float,
    volatility_power: float,
    max_actions: int,
    do_grpo: bool,
    grpo_clip: float,
    grpo_kl: float,
    grpo_weight: float,
    entropy_weight: float,
    controller_count_weight: float,
    magnitude_reward_weight: float,
    predicted_delta: torch.Tensor | None,
    prediction_centres: torch.Tensor | None,
    target_delta: torch.Tensor | None,
    target_changed: torch.Tensor | None,
    batch_edge_grpo_rollouts: bool = False,
    tensorized_grpo_rewards: bool = False,
    precomputed_policy_states: dict[GraphOperation, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = z_t.new_zeros(())
    supervised = zero
    count_loss = zero
    policy_states: dict[GraphOperation, torch.Tensor] = {}
    active_operations: list[GraphOperation] = []
    for operation in T3_OPERATIONS:
        controller = controllers[_controller_key(operation)]
        state = (
            precomputed_policy_states.get(operation)
            if precomputed_policy_states is not None
            else None
        )
        if state is None:
            state = controller.node_state(z_t.detach(), hidden.detach())
        policy_states[operation] = state
        target = targets[operation].to(device=z_t.device, dtype=torch.long)
        if operation in T3_NODE_OPERATIONS:
            candidates = candidate_nodes[operation].to(
                device=z_t.device, dtype=torch.long
            )
            if not candidates.numel():
                continue
            labels = _candidate_labels(
                candidates, target, num_nodes=z_t.shape[0], item_type="node"
            )
            logits = controller.node_target_head(state).squeeze(-1).index_select(
                0, candidates
            )
            supervised = supervised + _balanced_bce(logits, labels.float())
            count_loss = count_loss + F.smooth_l1_loss(
                controller.node_log_count(state),
                torch.log1p(target.new_tensor(float(target.numel())).to(z_t.dtype)),
            )
        else:
            candidates = candidate_edges[operation].to(
                device=z_t.device, dtype=torch.long
            )



            if operation == GraphOperation.ADD_EDGE and target.shape[1]:
                candidate_codes = candidates[0] * z_t.shape[0] + candidates[1]
                target_codes = target[0] * z_t.shape[0] + target[1]
                missing = ~torch.isin(target_codes, candidate_codes)
                if bool(missing.any()):
                    candidates = torch.cat((candidates, target[:, missing]), dim=1)
            if candidates.shape[1] == 0:
                continue
            labels = _candidate_labels(
                candidates, target, num_nodes=z_t.shape[0], item_type="edge"
            )
            positive = torch.where(labels)[0]
            negative = torch.where(~labels)[0]
            negative_budget = min(
                int(negative.numel()),
                max(256, min(8192, 16 * max(int(positive.numel()), 1))),
            )
            if negative.numel() > negative_budget:
                positions = torch.linspace(
                    0,
                    negative.numel() - 1,
                    steps=negative_budget,
                    device=negative.device,
                ).long()
                negative = negative.index_select(0, positions)
            selected = torch.cat((positive, negative))
            selected_edges = candidates.index_select(1, selected)
            logits = controller.edge_logits_chunked(state, selected_edges)
            supervised = supervised + _balanced_bce(
                logits, labels.index_select(0, selected).float()
            )
            count_index = 0 if operation == GraphOperation.ADD_EDGE else 1
            count_loss = count_loss + F.smooth_l1_loss(
                controller.topology_log_counts(state)[count_index],
                torch.log1p(target.new_tensor(float(target.shape[1])).to(z_t.dtype)),
            )
        active_operations.append(operation)
    normalizer = max(len(active_operations), 1)
    supervised = supervised / normalizer
    count_loss = count_loss / normalizer

    grpo = zero
    reward_values: list[float] = []
    use_magnitude_reward = False
    rollout_operations = tuple(
        operation
        for operation in T3_OPERATIONS
        if (
            candidate_nodes[operation].numel()
            if operation in T3_NODE_OPERATIONS
            else candidate_edges[operation].shape[1]
        )
    )
    if (
        do_grpo
        and rollout_operations
        and rollout_budget >= 2 * len(rollout_operations)
    ):
        allocation = group_sampler.allocate(
            rollout_operations,
            total_budget=rollout_budget,
            min_per_group=2,
        )
        node_weights = structure_node_weights(
            transition["edge_index_t"],
            num_nodes=z_t.shape[0],
            historical_change_counts=historical_change_counts,
            history_steps=history_steps,
            degree_power=degree_power,
            volatility_power=volatility_power,
        )
        group_rollouts: dict[GraphOperation, list[RewardedSequenceRollout]] = {}



        use_magnitude_reward = (
            float(magnitude_reward_weight) > 0.0
            and predicted_delta is not None
            and prediction_centres is not None
            and target_delta is not None
            and target_changed is not None
        )
        local_prediction_weights = (
            node_weights[prediction_centres.to(device=node_weights.device, dtype=torch.long)]
            if use_magnitude_reward
            else None
        )
        if use_magnitude_reward:
            assert predicted_delta is not None
            assert prediction_centres is not None
            assert target_delta is not None
            assert target_changed is not None
            if predicted_delta.shape != target_delta.shape:
                raise ValueError("T3 magnitude reward predictions and targets must match.")
            if predicted_delta.shape[0] != prediction_centres.numel():
                raise ValueError("T3 magnitude reward centres must match prediction rows.")
        with _preserve_world_rng(z_t.device), torch.no_grad():
            for operation in rollout_operations:
                controller = controllers[_controller_key(operation)]
                state = policy_states[operation]
                if operation in T3_NODE_OPERATIONS:
                    candidates = candidate_nodes[operation].to(z_t.device)
                    samples = controller.sample_node_action_sequences_batch(
                        z_t.detach(),
                        hidden.detach(),
                        operation,
                        candidate_nodes=candidates,
                        min_actions=1,
                        max_actions=min(max_actions, int(candidates.numel())),
                        sample_magnitude=False,
                        node_state=state.detach(),
                        batch_size=allocation[operation],
                    )
                elif batch_edge_grpo_rollouts:
                    candidates = candidate_edges[operation].to(z_t.device)
                    samples = controller.sample_edge_action_sequences_batch(
                        z_t.detach(),
                        hidden.detach(),
                        operation,
                        candidate_edges=candidates,
                        min_actions=1,
                        max_actions=min(max_actions, int(candidates.shape[1])),
                        node_state=state.detach(),
                        batch_size=allocation[operation],
                    )
                else:
                    candidates = candidate_edges[operation].to(z_t.device)
                    samples = [
                        controller.sample_action_sequence(
                            z_t.detach(),
                            hidden.detach(),
                            operation,
                            candidate_edges={operation: candidates},
                            valid_operations=(operation,),
                            min_actions=1,
                            max_actions=min(max_actions, int(candidates.shape[1])),
                            sample_magnitude=False,
                            node_state=state.detach(),
                        )
                        for _ in range(allocation[operation])
                    ]
                operation_rollouts: list[RewardedSequenceRollout] = []
                for sample in samples:
                    if operation in T3_NODE_OPERATIONS:
                        predicted = torch.tensor(
                            sample.action.target_nodes,
                            device=z_t.device,
                            dtype=torch.long,
                        )
                        item_type = "node"
                    else:
                        predicted = torch.tensor(
                            [edit.target for edit in sample.action],
                            device=z_t.device,
                            dtype=torch.long,
                        ).t().contiguous()
                        item_type = "edge"
                    if use_magnitude_reward:
                        assert local_prediction_weights is not None
                        assert prediction_centres is not None




                        selected_global = predicted.reshape(-1).to(
                            device=prediction_centres.device, dtype=torch.long
                        )
                        selected_local = torch.searchsorted(
                            prediction_centres, selected_global
                        )
                        valid = selected_local < prediction_centres.numel()
                        if bool(valid.any()):
                            valid_local = selected_local[valid]
                            valid_global = prediction_centres.index_select(0, valid_local)
                            valid = valid_global.eq(selected_global[valid])
                            selected_local = valid_local[valid].unique()
                        else:
                            selected_local = selected_local.new_empty((0,))
                        localization = node_change_localization_reward(
                            predicted_nodes=selected_local,
                            target_changed=target_changed,
                            node_weights=local_prediction_weights,
                        )
                        magnitude = node_state_magnitude_reward(
                            predicted_delta=predicted_delta,
                            target_delta=target_delta,
                            target_changed=target_changed,
                            node_weights=local_prediction_weights,
                            selected_nodes=selected_local,
                        )
                        localization_reward: float | torch.Tensor = (
                            localization["reward"].detach()
                            if tensorized_grpo_rewards
                            else float(localization["reward"].item())
                        )
                        magnitude_reward: float | torch.Tensor = (
                            magnitude["reward"].detach()
                            if tensorized_grpo_rewards
                            else float(magnitude["reward"].item())
                        )
                        mix = min(max(float(magnitude_reward_weight), 0.0), 1.0)
                        reward = (
                            (1.0 - mix) * localization_reward
                            + mix * magnitude_reward
                        )
                    else:
                        structure_reward = structure_weighted_f1(
                            predicted,
                            targets[operation].to(z_t.device),
                            node_weights=node_weights,
                            item_type=item_type,
                        )
                        reward = (
                            structure_reward["f1"].detach()
                            if tensorized_grpo_rewards
                            else float(structure_reward["f1"].item())
                        )
                    operation_rollouts.append(
                        RewardedSequenceRollout(
                            sample=sample, reward=reward
                        )
                    )
                group_rollouts[operation] = operation_rollouts

        for operation in rollout_operations:
            controller = controllers[_controller_key(operation)]
            state = policy_states[operation]
            operation_max_actions = min(
                max_actions,
                int(
                    candidate_nodes[operation].numel()
                    if operation in T3_NODE_OPERATIONS
                    else candidate_edges[operation].shape[1]
                ),
            )
            terms = grpo_sequence_loss(
                controller,
                group_rollouts[operation],
                z_t=z_t.detach(),
                h_t=hidden.detach(),
                candidate_nodes=(
                    {operation: candidate_nodes[operation].to(z_t.device)}
                    if operation in T3_NODE_OPERATIONS
                    else None
                ),
                candidate_edges=(
                    {operation: candidate_edges[operation].to(z_t.device)}
                    if operation in T3_EDGE_OPERATIONS
                    else None
                ),
                valid_operations=(operation,),
                max_actions=operation_max_actions,
                clip_epsilon=grpo_clip,
                kl_coefficient=grpo_kl,
                entropy_coefficient=entropy_weight,
                node_state=state,
                batch_edge_sequences=batch_edge_grpo_rollouts,
            )
            grpo = grpo + terms["loss"]
            reward_values.append(float(terms["mean_reward"].detach().item()))
        grpo = grpo / len(rollout_operations)
    return supervised + float(controller_count_weight) * count_loss + float(grpo_weight) * grpo, {
        "supervised": float(supervised.detach().item()),
        "count": float(count_loss.detach().item()),
        "grpo": float(grpo.detach().item()),
        "reward": sum(reward_values) / len(reward_values) if reward_values else 0.0,
        "magnitude_reward": float(magnitude_reward_weight) if use_magnitude_reward else 0.0,
    }


def _run_split(
    model: GraphWorldModel,
    controllers: nn.ModuleDict,
    action_encoder: GraphActionEncoder,
    head: StructuralHead,
    dataset: GraphTransitionDataset,
    cache: TargetCache,
    stats: dict[str, torch.Tensor],
    indices: list[int],
    *,
    device: torch.device,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    action_weight: float = 0.25,
    latent_weight: float = 0.1,
    controller_count_weight: float = 0.1,
    grpo_weight: float = 0.05,
    changed_center_loss_weight: float = 0.0,
    changed_descriptor_loss_weight: float = 0.0,
    changed_mae_loss_weight: float = 0.0,
    changed_rmse_loss_weight: float = 0.0,
    direction_loss_weight: float = 0.0,
    grpo_rollout_budget: int = 12,
    degree_power: float = 1.0,
    volatility_power: float = 1.0,
    grpo_interval: int = 8,
    grpo_warmup_epochs: int = 5,
    grpo_clip: float = 0.2,
    grpo_kl: float = 0.01,
    entropy_weight: float = 0.001,
    magnitude_reward_weight: float = 0.5,
    max_actions: int = 8,
    max_addition_candidates: int = 32768,
    action_group_fusion: str = "mean",
    direction_projection_margin: float = 0.05,
    direction_projection_confidence: float = 0.0,
    direction_class_balance_power: float = 0.5,
    causal_structure_context: bool = False,
    causal_structure_history_context: bool = False,
    group_aware_decoder_context: bool = False,
    direction_group_aware_context_only: bool = False,
    direction_backpropagate_context: bool = False,
    direction_transfer_bypass: bool = False,
    direction_transfer_mix: float = 1.0,
    proposal_seed: int = 1,
    epoch: int = 0,
    group_sampler: DynamicGroupSampler | None = None,
    rollout_indices: list[int] | None = None,
    target_indices: set[int] | None = None,
    action_candidate_cache: dict[
        tuple[int, int, tuple[int, ...]],
        dict[
            int,
            tuple[
                dict[GraphOperation, torch.Tensor],
                dict[GraphOperation, torch.Tensor],
                torch.Tensor,
            ],
        ],
    ] | None = None,
    batch_edge_grpo_rollouts: bool = False,
    tensorized_grpo_rewards: bool = False,
    reuse_controller_policy_states: bool = False,
    released_history_cache: dict[
        int,
        tuple[
            dict[GraphOperation, torch.Tensor],
            torch.Tensor,
            tuple[int, ...],
        ],
    ] | None = None,
) -> dict[str, float]:
    model.train(train)
    controllers.train(train)
    action_encoder.train(train)
    head.train(train)
    model.reset_history()
    hidden = model.initial_hidden(int(dataset.metadata["num_nodes"]), device)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_classes: list[torch.Tensor] = []
    predicted_classes: list[torch.Tensor] = []
    total = 0.0
    historical_change_counts = torch.zeros(
        int(dataset.metadata["num_nodes"]), device=device
    )
    history_steps = 0
    historical_edge_codes: set[int] = set()
    previous_descriptor_delta = torch.zeros(
        int(dataset.metadata["num_nodes"]), len(DESCRIPTOR_NAMES), device=device
    )
    descriptor_history_seen = torch.zeros(
        int(dataset.metadata["num_nodes"]), dtype=torch.bool, device=device
    )
    descriptor_delta_history = torch.zeros(
        int(dataset.metadata["num_nodes"]),
        max(int(model.history_window), 1),
        len(DESCRIPTOR_NAMES),
        device=device,
    )
    descriptor_delta_history_mask = torch.zeros(
        int(dataset.metadata["num_nodes"]),
        max(int(model.history_window), 1),
        dtype=torch.bool,
        device=device,
    )
    component_timing = os.environ.get("WORLDGRAPH_T3_COMPONENT_TIMING", "0") == "1"
    component_times: defaultdict[str, float] = defaultdict(float)
    component_steps = 0
    candidate_cache_hits = 0
    candidate_cache_misses = 0

    def boundary(name: str, started_at: float) -> float:
        if not component_timing:
            return 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        component_times[name] += now - started_at
        return now

    def advance_descriptor_history(
        centres: torch.Tensor, delta: torch.Tensor
    ) -> None:
        nonlocal descriptor_delta_history, descriptor_delta_history_mask
        descriptor_delta_history = torch.roll(
            descriptor_delta_history, shifts=1, dims=1
        )
        descriptor_delta_history_mask = torch.roll(
            descriptor_delta_history_mask, shifts=1, dims=1
        )
        descriptor_delta_history[:, 0].zero_()
        descriptor_delta_history_mask[:, 0].zero_()
        if centres.numel():
            descriptor_delta_history[centres, 0] = delta.detach()
            descriptor_delta_history_mask[centres, 0] = True
    if rollout_indices is None:
        rollout_indices = list(indices)
    if target_indices is None:
        target_indices = set(int(index) for index in indices)
    candidate_partition = None
    if action_candidate_cache is not None:




        candidate_partition = action_candidate_cache.setdefault(
            (
                int(proposal_seed),
                int(max_addition_candidates),
                tuple(int(index) for index in rollout_indices),
            ),
            {},
        )
    candidate_partition_complete = (
        candidate_partition is not None
        and all(int(index) in candidate_partition for index in rollout_indices)
    )
    for index in rollout_indices:
        started_at = time.perf_counter() if component_timing else 0.0
        transition_cpu = dataset[index]
        target_record = cache.get(transition_cpu)
        centres = target_record["centres"].to(device)
        num_nodes = int(dataset.metadata["num_nodes"])
        cached_candidates = (
            candidate_partition.get(int(index))
            if candidate_partition is not None
            else None
        )
        if cached_candidates is None:
            generator = torch.Generator().manual_seed(
                int(proposal_seed) * 1_000_003
                + int(transition_cpu["transition_id"])
            )
            candidate_nodes, candidate_edges, active_t = _current_action_candidates(
                transition_cpu["edge_index_t"],
                num_nodes=num_nodes,
                historical_edge_codes=historical_edge_codes,
                max_addition_candidates=max_addition_candidates,
                generator=generator,
            )
            candidate_cache_misses += 1
            if candidate_partition is not None:
                candidate_partition[int(index)] = (
                    candidate_nodes,
                    candidate_edges,
                    active_t,
                )
        else:
            candidate_nodes, candidate_edges, active_t = cached_candidates
            candidate_cache_hits += 1
        transition = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in transition_cpu.items()
        }
        started_at = boundary("data_target_candidate_prepare", started_at)
        hidden_before = hidden
        outputs, z_raw, z_t, proposal_policy_states = _forward(
            model,
            controllers,
            action_encoder,
            transition,
            hidden_before,
            candidate_nodes,
            candidate_edges,
            train=train,
            max_actions=max_actions,
            action_group_fusion=action_group_fusion,
            reuse_controller_policy_states=reuse_controller_policy_states,
            direction_transfer_bypass=direction_transfer_bypass,
            direction_transfer_mix=direction_transfer_mix,
            decode_node_state=False,
            component_times=component_times if component_timing else None,
        )
        hidden = outputs["hidden_next"].detach()
        started_at = time.perf_counter() if component_timing else 0.0




        released_history = (
            released_history_cache.get(int(index))
            if released_history_cache is not None
            else None
        )
        if released_history is None:
            released_targets = _released_action_targets(
                transition_cpu,
                num_nodes=num_nodes,
                active_t=active_t,
            )
            changed_node_indices = (
                torch.unique(
                    torch.cat(
                        [
                            target.reshape(-1)
                            for target in released_targets.values()
                            if target.numel()
                        ]
                    )
                )
                if any(target.numel() for target in released_targets.values())
                else torch.empty(0, dtype=torch.long)
            )
            current_edges = _canonical_edges(
                transition_cpu["edge_index_t"].detach().cpu(), num_nodes
            )
            current_edge_codes = tuple(
                (current_edges[0] * num_nodes + current_edges[1]).tolist()
            )
            if released_history_cache is not None:
                released_history_cache[int(index)] = (
                    released_targets,
                    changed_node_indices,
                    current_edge_codes,
                )
        else:
            released_targets, changed_node_indices, current_edge_codes = (
                released_history
            )
        changed_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if changed_node_indices.numel():
            changed_nodes.index_fill_(
                0, changed_node_indices.to(device=device, dtype=torch.long), True
            )


        if not (
            released_history_cache is not None and candidate_partition_complete
        ):
            historical_edge_codes.update(current_edge_codes)
        if centres.numel():
            target_delta = target_record["delta"].to(device)
        else:
            target_delta = torch.empty(
                (0, len(DESCRIPTOR_NAMES)), device=device
            )
        started_at = boundary("released_target_history", started_at)


        if int(index) not in target_indices:
            if centres.numel():
                previous_descriptor_delta.index_copy_(0, centres, target_delta)
                descriptor_history_seen.index_fill_(0, centres, True)
            advance_descriptor_history(centres, target_delta)
            historical_change_counts.add_(changed_nodes.float())
            history_steps += 1
            boundary("metrics_history_bookkeeping", started_at)
            component_steps += 1
            continue
        if not centres.numel():
            advance_descriptor_history(centres, target_delta)
            historical_change_counts.add_(changed_nodes.float())
            history_steps += 1
            boundary("metrics_history_bookkeeping", started_at)
            component_steps += 1
            continue
        target_norm = (target_delta - stats["mean"].to(device)) / stats["std"].to(device)
        decoder_parts = [
            outputs["latent_next"],
            z_t,
            outputs["hidden_next"],
        ]
        effective_direction_mix = (
            0.0 if direction_transfer_bypass else direction_transfer_mix
        )
        direction_decoder_parts = (
            [
                torch.lerp(
                    outputs[base_name], adapted_value,
                    effective_direction_mix,
                )
                for base_name, adapted_value in (
                    ("direction_base_latent_next", outputs["latent_next"]),
                    ("direction_base_z_t", z_t),
                    ("direction_base_hidden_next", outputs["hidden_next"]),
                )
            ]
            if effective_direction_mix < 1.0
            else decoder_parts
        )
        if causal_structure_context:
            current_mean = stats["current_mean"].to(device)
            current_std = stats["current_std"].to(device)
            current_context = (
                target_record["current"].to(device) - current_mean
            ) / current_std
            previous_context = (
                previous_descriptor_delta.index_select(0, centres)
                - stats["mean"].to(device)
            ) / stats["std"].to(device)
            seen_context = descriptor_history_seen.index_select(0, centres).to(
                dtype=z_t.dtype
            ).unsqueeze(-1)
            previous_context = torch.where(
                seen_context.bool(), previous_context, torch.zeros_like(previous_context)
            )
            structure_context = torch.cat(
                (current_context, previous_context, seen_context), dim=-1
            )
            full_structure_context = z_t.new_zeros(
                (num_nodes, structure_context.shape[-1])
            )
            full_structure_context.index_copy_(0, centres, structure_context)
            decoder_parts.append(full_structure_context)
            if direction_decoder_parts is not decoder_parts:
                direction_decoder_parts.append(full_structure_context)
        if causal_structure_history_context:
            current_context = (
                target_record["current"].to(device) - stats["current_mean"].to(device)
            ) / stats["current_std"].to(device)
            local_history = descriptor_delta_history.index_select(0, centres)
            local_history_mask = descriptor_delta_history_mask.index_select(
                0, centres
            )
            local_history = (
                local_history - stats["mean"].to(device).view(1, 1, -1)
            ) / stats["std"].to(device).view(1, 1, -1)
            local_history = torch.where(
                local_history_mask.unsqueeze(-1),
                local_history,
                torch.zeros_like(local_history),
            )
            structure_history_context = torch.cat(
                (
                    current_context,
                    local_history.flatten(start_dim=1),
                    local_history_mask.to(dtype=z_t.dtype),
                ),
                dim=-1,
            )
            full_history_context = z_t.new_zeros(
                (num_nodes, structure_history_context.shape[-1])
            )
            full_history_context.index_copy_(
                0, centres, structure_history_context
            )
            decoder_parts.append(full_history_context)
            if direction_decoder_parts is not decoder_parts:
                direction_decoder_parts.append(full_history_context)
        if group_aware_decoder_context and not direction_group_aware_context_only:
            decoder_parts.append(outputs["t3_group_action_context"])
            if direction_decoder_parts is not decoder_parts:
                direction_decoder_parts.append(outputs["t3_group_action_context"])
        decoder_input = torch.cat(decoder_parts, dim=-1)
        local_decoder_input = decoder_input.index_select(0, centres)
        prediction_norm = head(local_decoder_input)
        target_cls = classes(target_delta, stats["tolerance"].to(device))
        magnitude_prediction_raw = (
            prediction_norm * stats["std"].to(device) + stats["mean"].to(device)
        )




        direction_decoder_input = torch.cat(direction_decoder_parts, dim=-1).index_select(
            0, centres
        )
        if group_aware_decoder_context and direction_group_aware_context_only:
            direction_decoder_input = torch.cat(
                (
                    local_decoder_input,
                    outputs["t3_group_action_context"].index_select(0, centres),
                ),
                dim=-1,
            )
        direction_logits = head.direction_logits(
            direction_decoder_input
            if direction_backpropagate_context
            else direction_decoder_input.detach()
        )
        prediction_raw = (
            _project_prediction_to_direction(
                magnitude_prediction_raw,
                direction_logits,
                stats["tolerance"],
                margin=direction_projection_margin,
                confidence_threshold=direction_projection_confidence,
            )
            if direction_logits is not None
            else magnitude_prediction_raw
        )
        regression_terms = F.smooth_l1_loss(
            prediction_norm, target_norm, reduction="none"
        )
        if (
            float(changed_center_loss_weight) > 0.0
            or float(changed_descriptor_loss_weight) > 0.0
        ):




            changed_weight = 1.0 + float(changed_center_loss_weight) * (
                (target_cls != 1).any(dim=-1).to(regression_terms.dtype)
            )
            descriptor_weight = 1.0 + float(changed_descriptor_loss_weight) * (
                target_cls != 1
            ).to(regression_terms.dtype)
            combined_weight = changed_weight.unsqueeze(-1) * descriptor_weight
            regression = (regression_terms * combined_weight).sum() / (
                combined_weight.sum().clamp_min(1e-8)
            )
        else:
            regression = regression_terms.mean()
        normalized_error = prediction_norm - target_norm
        changed_rmse_terms: list[torch.Tensor] = []
        changed_mae_terms: list[torch.Tensor] = []
        for descriptor in range(normalized_error.shape[-1]):
            changed = target_cls[:, descriptor].ne(1)
            if bool(changed.any()):
                changed_mae_terms.append(
                    normalized_error[:, descriptor][changed].abs().mean()
                )
                changed_rmse_terms.append(
                    normalized_error[:, descriptor][changed]
                    .square()
                    .mean()
                    .add(1e-8)
                    .sqrt()
                )
        changed_rmse_loss = (
            torch.stack(changed_rmse_terms).mean()
            if changed_rmse_terms
            else normalized_error.new_zeros(())
        )
        changed_mae_loss = (
            torch.stack(changed_mae_terms).mean()
            if changed_mae_terms
            else normalized_error.new_zeros(())
        )
        direction_loss = (
            _balanced_direction_head_loss(
                direction_logits,
                target_cls,
                balance_power=direction_class_balance_power,
            )
            if direction_logits is not None
            else _direction_classification_loss(
                prediction_raw,
                target_cls,
                stats["tolerance"],
            )
        )
        started_at = boundary("decoder_and_task_loss", started_at)
        latent_target = model.encode_target(
            transition["x_next"],
            transition["edge_index_next"],
            edge_weight_next=_edge_weight(transition, "next"),
            topology_cache_key=int(transition["transition_id"]) + 1,
        )
        latent_loss = F.mse_loss(outputs["latent_mu"], latent_target)
        started_at = boundary("target_encoder", started_at)
        controller_loss = prediction_norm.new_zeros(())
        if train:
            if group_sampler is None:
                raise RuntimeError("Training T3 requires a DynamicGroupSampler.")
            controller_loss, _ = _controller_training_terms(
                model=model,
                controllers=controllers,
                z_t=z_t,
                hidden=hidden_before,
                candidate_nodes=candidate_nodes,
                candidate_edges=candidate_edges,
                targets=released_targets,
                transition=transition,
                group_sampler=group_sampler,
                historical_change_counts=historical_change_counts,
                history_steps=history_steps,
                rollout_budget=grpo_rollout_budget,
                degree_power=degree_power,
                volatility_power=volatility_power,
                max_actions=max_actions,
                do_grpo=(
                    epoch > grpo_warmup_epochs
                    and history_steps % max(grpo_interval, 1) == 0
                ),
                grpo_clip=grpo_clip,
                grpo_kl=grpo_kl,
                grpo_weight=grpo_weight,
                entropy_weight=entropy_weight,
                controller_count_weight=controller_count_weight,
                magnitude_reward_weight=magnitude_reward_weight,
                predicted_delta=(
                    prediction_norm * stats["std"].to(device) + stats["mean"].to(device)
                ),
                prediction_centres=centres,
                target_delta=target_delta,
                target_changed=(target_cls != 1).any(dim=-1),
                batch_edge_grpo_rollouts=batch_edge_grpo_rollouts,
                tensorized_grpo_rewards=tensorized_grpo_rewards,
                precomputed_policy_states=proposal_policy_states,
            )
        started_at = boundary("controller_supervised_grpo", started_at)
        loss = (
            regression
            + float(changed_mae_loss_weight) * changed_mae_loss
            + float(changed_rmse_loss_weight) * changed_rmse_loss
            + float(direction_loss_weight) * direction_loss
            + float(latent_weight) * latent_loss
            + float(action_weight) * controller_loss
        )
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *controllers.parameters(), *action_encoder.parameters(), *head.parameters()],
                5.0,
            )
            optimizer.step()
            model.update_target_encoder()
            assert group_sampler is not None
            for operation, target in released_targets.items():
                target_count = (
                    int(target.numel())
                    if operation in T3_NODE_OPERATIONS
                    else int(target.shape[1])
                )
                if target_count:
                    group_sampler.observe([operation] * target_count)
        started_at = boundary("backward_optimizer", started_at)
        historical_change_counts.add_(changed_nodes.float())
        previous_descriptor_delta.index_copy_(0, centres, target_delta.detach())
        descriptor_history_seen.index_fill_(0, centres, True)
        advance_descriptor_history(centres, target_delta)
        history_steps += 1
        total += float(loss.detach().cpu())
        predictions.append(prediction_raw.detach().cpu())
        targets.append(target_delta.detach().cpu())
        target_classes.append(target_cls.detach().cpu())
        predicted_classes.append(classes(predictions[-1], stats["tolerance"]).detach().cpu())
        boundary("metrics_history_bookkeeping", started_at)
        component_steps += 1
    if not predictions:
        raise RuntimeError("No non-empty T3 centres were evaluated.")
    finalize_started_at = time.perf_counter() if component_timing else 0.0
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    true_classes = torch.cat(target_classes)
    predicted_class_tensor = classes(prediction, stats["tolerance"])



    metrics = regression_metrics(
        prediction,
        target,
        mean=stats["mean"],
        std=stats["std"],
    )
    metrics.update(
        change_regression_metrics(
            prediction,
            target,
            true_classes,
            std=stats["std"],
        )
    )
    metrics["macro_f1"] = macro_f1(true_classes, predicted_class_tensor)
    metrics["change_macro_f1"] = temporal_change_macro_f1(target_classes, predicted_classes)
    metrics["loss"] = total / max(len(indices), 1)
    metrics["action_candidate_cache_hits"] = float(candidate_cache_hits)
    metrics["action_candidate_cache_misses"] = float(candidate_cache_misses)
    if component_timing:
        boundary("metrics_finalize", finalize_started_at)
        split_name = "train" if train else "eval"
        print(
            f"component_timing[t3/{split_name}] transitions={component_steps} "
            + ",".join(
                f"{name}:{component_times[name]:.3f}s"
                for name in (
                    "data_target_candidate_prepare",
                    "graph_encoder",
                    "node_action_proposals",
                    "edge_action_proposals",
                    "action_fusion",
                    "history_state_forward",
                    "released_target_history",
                    "decoder_and_task_loss",
                    "target_encoder",
                    "controller_supervised_grpo",
                    "backward_optimizer",
                    "metrics_history_bookkeeping",
                    "metrics_finalize",
                )
            )
            + f" candidate_cache_hits={candidate_cache_hits}"
            + f" candidate_cache_misses={candidate_cache_misses}",
            flush=True,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--target_cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--pretrained_backbone",
        default=None,
        help=(
            "Optional LOO-pretrained WorldGraph graph/state/latent backbone. "
            "Task-specific Action, Controller and decoder modules remain fresh."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_mode",
        choices=["structural", "legacy"],
        default="structural",
        help=(
            "structural keeps the target raw-feature input and transfers only "
            "feature-independent SGT weights; legacy uses the fixed-width adapter."
        ),
    )
    parser.add_argument(
        "--pretrained_backbone_blend",
        type=float,
        default=0.25,
        help="Fraction of compatible pretrained weights to load in structural mode.",
    )
    parser.add_argument(
        "--pretrained_transfer_gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Add identity-initialized gated residual adapters to the graph "
            "and history states of a pretrained downstream model. Defaults "
            "to enabled whenever a T3 pretrained backbone is supplied."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_gate_bottleneck",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--pretrained_transfer_gate_initial",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--pretrained_task_blend",
        type=float,
        default=0.5,
        help="Blend for the optional task-aligned T3 structural head.",
    )
    parser.add_argument(
        "--pretrained_magnitude_head_blend", type=float, default=None,
        help="Override the T3 magnitude-head transfer fraction only.",
    )
    parser.add_argument(
        "--pretrained_direction_head_blend", type=float, default=None,
        help="Override the T3 direction-head transfer fraction only.",
    )
    parser.add_argument(
        "--pretrained_transfer_gate_direction_bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use pre-transfer graph/history states only for the decoupled "
            "direction head, while retaining transfer gates for regression."
        ),
    )
    parser.add_argument(
        "--pretrained_direction_transfer_mix",
        type=float,
        default=1.0,
        help=(
            "Direction-head input blend between pre-transfer (0) and "
            "post-transfer (1) representations; regression is unchanged."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_state_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Transfer action-independent history-state layers. Structural mode "
            "skips source action projections and gates."
        ),
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--history_window", type=int, default=8)
    parser.add_argument(
        "--train_clip_transitions",
        type=int,
        default=0,
        help="Optional contiguous chronological training clip length; 0 uses the full train split.",
    )
    parser.add_argument(
        "--clip_history_burn_in",
        type=int,
        default=8,
        help="Observed prefix transitions replayed before a rotating training clip.",
    )
    parser.add_argument("--history_num_heads", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--action_dim", type=int, default=32)
    parser.add_argument("--policy_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sgt_num_hops", type=int, default=2)
    parser.add_argument("--sgt_num_walks", type=int, default=4)
    parser.add_argument("--sgt_walk_length", type=int, default=3)
    parser.add_argument(
        "--group_beta",
        type=float,
        default=1.0,
        help="Rarity exponent used by action-aware dynamic group sampling.",
    )
    parser.add_argument(
        "--degree_power",
        type=float,
        default=1.0,
        help="Exponent theta for degree-based structural importance.",
    )
    parser.add_argument(
        "--volatility_power",
        type=float,
        default=1.0,
        help="Exponent eta for historical change-frequency importance.",
    )
    parser.add_argument(
        "--sgt_topology_cache_bytes",
        type=int,
        default=0,
        help="Bounded cache for deterministic SGT topology and walk tensors.",
    )
    parser.add_argument("--sgt_attention_dropout", type=float, default=0.1)
    parser.add_argument("--target_encoder_momentum", type=float, default=0.99)
    parser.add_argument("--action_adapter_initial_scale", type=float, default=0.05)
    parser.add_argument("--action_residual_max_scale", type=float, default=0.15)
    parser.add_argument("--action_adapter_temperature", type=float, default=0.25)
    parser.add_argument("--action_weight", type=float, default=0.25)
    parser.add_argument("--latent_weight", type=float, default=0.1)
    parser.add_argument("--controller_count_weight", type=float, default=0.1)
    parser.add_argument(
        "--magnitude_reward_weight",
        type=float,
        default=0.5,
        help=(
            "Weight of the local-structure magnitude term in the T3 GRPO "
            "verifier; the remainder scores changed-centre localization."
        ),
    )
    parser.add_argument(
        "--changed_center_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Extra relative loss weight for centres with a non-zero T3 "
            "descriptor change; zero preserves the unweighted objective."
        ),
    )
    parser.add_argument(
        "--changed_descriptor_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Extra loss weight for individual descriptor cells whose change "
            "direction is non-zero under the train-only T3 tolerance."
        ),
    )
    parser.add_argument(
        "--changed_rmse_loss_weight",
        type=float,
        default=0.0,
        help="Weight of the descriptor-balanced changed-cell RMSE objective.",
    )
    parser.add_argument(
        "--changed_mae_loss_weight",
        type=float,
        default=0.0,
        help="Weight of the descriptor-balanced changed-cell MAE objective.",
    )
    parser.add_argument(
        "--direction_loss_weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for the three-way descriptor direction objective.",
    )
    parser.add_argument(
        "--decoupled_direction_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Predict T3 decrease/unchanged/increase separately from delta "
            "magnitude, then project the continuous output consistently."
        ),
    )
    parser.add_argument(
        "--direction_projection_margin",
        type=float,
        default=0.05,
        help="Relative margin around the train-only unchanged tolerance.",
    )
    parser.add_argument(
        "--direction_projection_confidence",
        type=float,
        default=0.0,
        help=(
            "Only enforce a decoupled direction prediction when its softmax "
            "confidence reaches this threshold."
        ),
    )
    parser.add_argument(
        "--direction_class_balance_power",
        type=float,
        default=0.5,
        help="Inverse-frequency exponent for the decoupled T3 direction head.",
    )
    parser.add_argument(
        "--selection_change_f1_weight",
        type=float,
        default=1.0,
        help="Validation-only Change Macro-F1 weight used for checkpoint selection.",
    )
    parser.add_argument(
        "--selection_change_rmse_weight",
        type=float,
        default=0.0,
        help="Validation-score weight on changed-cell RMSE (lower is better).",
    )
    parser.add_argument("--grpo_weight", type=float, default=0.05)
    parser.add_argument("--grpo_rollout_budget", type=int, default=12)
    parser.add_argument("--grpo_interval", type=int, default=8)
    parser.add_argument("--grpo_warmup_epochs", type=int, default=5)
    parser.add_argument("--grpo_clip", type=float, default=0.2)
    parser.add_argument("--grpo_kl", type=float, default=0.01)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument("--max_actions", type=int, default=8)
    parser.add_argument("--max_addition_candidates", type=int, default=32768)
    parser.add_argument(
        "--cache_action_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache deterministic T3 node/edge action candidate pools in memory "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--batch_edge_grpo_rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Batch same-group edge GRPO sampling and likelihood evaluation "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--tensorized_grpo_rewards",
        action="store_true",
        help=(
            "Experimental on-device aggregation of per-rollout verifier "
            "rewards. Disabled by default."
        ),
    )
    parser.add_argument(
        "--cache_released_history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache deterministic released graph edits and causal edge-history "
            "updates; enabled by default without changing model semantics."
        ),
    )
    parser.add_argument(
        "--reuse_controller_policy_states",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimentally reuse each Controller's differentiable policy "
            "state between causal action proposal and supervised/GRPO loss."
        ),
    )
    parser.add_argument(
        "--action_group_fusion",
        choices=["mean", "sqrt_sum", "sum"],
        default="mean",
        help="Causal fusion for the four T3 Controller action groups.",
    )
    parser.add_argument(
        "--causal_structure_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Give the T3 decoder current local descriptors and the last "
            "observed descriptor delta; both are causal at prediction time."
        ),
    )
    parser.add_argument(
        "--causal_structure_history_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Give the T3 decoder current local descriptors and a causal, "
            "time-aligned window of previously observed descriptor deltas."
        ),
    )
    parser.add_argument(
        "--group_aware_decoder_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expose the four operation-specific action embeddings separately.",
    )
    parser.add_argument(
        "--direction_group_aware_context_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Expose operation-specific action context to the decoupled "
            "direction head without perturbing the magnitude decoder."
        ),
    )
    parser.add_argument(
        "--direction_backpropagate_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow the auxiliary direction objective to refine the shared "
            "causal representation instead of training only its own head."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--controller_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.pretrained_transfer_gate is None:
        args.pretrained_transfer_gate = bool(args.pretrained_backbone)
    if args.grpo_rollout_budget < 2 * len(T3_OPERATIONS):
        parser.error(
            f"--grpo_rollout_budget must be at least {2 * len(T3_OPERATIONS)}."
        )
    if args.train_clip_transitions < 0 or args.clip_history_burn_in < 0:
        parser.error("Training clip and history burn-in must be non-negative.")
    if (
        args.grpo_interval < 1
        or args.max_actions < 1
        or args.max_addition_candidates < 1
    ):
        parser.error(
            "--grpo_interval, --max_actions, and --max_addition_candidates "
            "must be positive."
        )
    if not 0.0 <= args.magnitude_reward_weight <= 1.0:
        parser.error("--magnitude_reward_weight must lie in [0, 1].")
    if (
        args.changed_center_loss_weight < 0.0
        or args.changed_descriptor_loss_weight < 0.0
        or args.changed_mae_loss_weight < 0.0
        or args.changed_rmse_loss_weight < 0.0
    ):
        parser.error("T3 changed-target loss weights must be non-negative.")
    if args.direction_loss_weight < 0.0 or args.selection_change_f1_weight <= 0.0:
        parser.error("Direction loss must be non-negative and selection weight positive.")
    if not 0.0 < args.direction_projection_margin < 1.0:
        parser.error("--direction_projection_margin must lie in (0, 1).")
    if not 0.0 <= args.direction_projection_confidence <= 1.0:
        parser.error("--direction_projection_confidence must lie in [0, 1].")
    if not 0.0 <= args.direction_class_balance_power <= 1.0:
        parser.error("--direction_class_balance_power must lie in [0, 1].")
    if args.direction_group_aware_context_only and not (
        args.group_aware_decoder_context and args.decoupled_direction_head
    ):
        parser.error(
            "--direction_group_aware_context_only requires both "
            "--group_aware_decoder_context and --decoupled_direction_head."
        )
    if args.direction_backpropagate_context and not args.decoupled_direction_head:
        parser.error(
            "--direction_backpropagate_context requires "
            "--decoupled_direction_head."
        )
    if args.pretrained_transfer_gate and not args.pretrained_backbone:
        parser.error("--pretrained_transfer_gate requires --pretrained_backbone.")
    if args.pretrained_transfer_gate_direction_bypass and not (
        args.pretrained_transfer_gate and args.decoupled_direction_head
    ):
        parser.error(
            "--pretrained_transfer_gate_direction_bypass requires both "
            "--pretrained_transfer_gate and --decoupled_direction_head."
        )
    if not 0.0 <= args.pretrained_direction_transfer_mix <= 1.0:
        parser.error("--pretrained_direction_transfer_mix must lie in [0, 1].")
    if args.pretrained_direction_transfer_mix < 1.0 and not (
        args.pretrained_transfer_gate and args.decoupled_direction_head
    ):
        parser.error(
            "--pretrained_direction_transfer_mix below 1 requires "
            "--pretrained_transfer_gate and --decoupled_direction_head."
        )
    if (
        args.pretrained_transfer_gate_direction_bypass
        and args.pretrained_direction_transfer_mix != 1.0
    ):
        parser.error("Direction bypass and a non-default direction mix cannot be combined.")
    if args.pretrained_transfer_gate_bottleneck < 1:
        parser.error("--pretrained_transfer_gate_bottleneck must be positive.")
    if not 0.0 < args.pretrained_transfer_gate_initial < 1.0:
        parser.error("--pretrained_transfer_gate_initial must lie in (0, 1).")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset = GraphTransitionDataset(ROOT / args.processed)
    cache = TargetCache(ROOT / args.target_cache)
    train_indices = _indices(dataset, "train")
    val_indices = _indices(dataset, "val")
    test_indices = _indices(dataset, "test")
    stats = fit_statistics(dataset, train_indices, cache)
    cache.save()
    model, controllers, action_encoder, head = _make_model(int(dataset.metadata["node_feature_dim"]), args)
    model.to(device); controllers.to(device); action_encoder.to(device); head.to(device)
    if args.pretrained_backbone:
        summary = load_worldgraph_pretrained_backbone(
            model,
            args.pretrained_backbone,
            map_location="cpu",
            expected_task="T3",
            expected_dataset=args.dataset,
            transfer_mode=args.pretrained_transfer_mode,
            transfer_state_model=args.pretrained_transfer_state_model,
            backbone_blend=args.pretrained_backbone_blend,
        )
        print(
            "loaded_pretrained_backbone="
            + str(summary["path"])
            + " tensors="
            + str(summary["loaded_tensors"])
            + " by_component="
            + json.dumps(summary["loaded_by_component"], sort_keys=True),
            flush=True,
        )
        task_head_summary = load_worldgraph_pretrained_t3_head(
            head,
            args.pretrained_backbone,
            map_location="cpu",
            blend=args.pretrained_task_blend,
            magnitude_blend=args.pretrained_magnitude_head_blend,
            direction_blend=args.pretrained_direction_head_blend,
        )
        if task_head_summary is not None:
            print(
                "loaded_pretrained_task_structure_head="
                + json.dumps(task_head_summary, sort_keys=True),
                flush=True,
            )
    optimizer = torch.optim.AdamW(
        [
            {"params": [*model.parameters(), *action_encoder.parameters(), *head.parameters()], "lr": args.lr},
            {"params": controllers.parameters(), "lr": args.controller_lr},
        ],
        weight_decay=args.weight_decay,
    )
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    group_sampler = DynamicGroupSampler(beta=args.group_beta)
    action_candidate_cache: dict[Any, Any] | None = (
        {} if args.cache_action_candidates else None
    )
    released_history_cache: dict[Any, Any] | None = (
        {} if args.cache_released_history else None
    )
    print(
        "action_groups="
        + ",".join(operation.name for operation in T3_OPERATIONS)
        + f" rollout_budget={args.grpo_rollout_budget} min_per_group=2 "
        + "structure_reward=degree+historical_variability"
        + f" group_beta={args.group_beta:g}"
        + f" degree_power={args.degree_power:g}"
        + f" volatility_power={args.volatility_power:g}"
        + f" magnitude_reward_weight={args.magnitude_reward_weight:g}",
        flush=True,
    )
    print(
        f"graph_encoder={model.graph_encoder_type} "
        f"state_model={model.state_model_type} "
        f"pretrained_transfer_gate={args.pretrained_transfer_gate} "
        f"direction_transfer_bypass="
        f"{args.pretrained_transfer_gate_direction_bypass} "
        f"direction_transfer_mix={args.pretrained_direction_transfer_mix}",
        flush=True,
    )
    print(
        f"train_clip_transitions={args.train_clip_transitions} "
        f"clip_history_burn_in={args.clip_history_burn_in} "
        f"cache_action_candidates={args.cache_action_candidates} "
        f"batch_edge_grpo_rollouts={args.batch_edge_grpo_rollouts} "
        f"tensorized_grpo_rewards={args.tensorized_grpo_rewards} "
        f"cache_released_history={args.cache_released_history} "
        f"reuse_controller_policy_states={args.reuse_controller_policy_states}",
        flush=True,
    )
    checkpoint = ROOT / args.checkpoint
    for epoch in range(1, args.epochs + 1):
        train_targets = _rotating_clip_indices(
            train_indices, args.train_clip_transitions, epoch, args.epochs
        )
        if train_targets == train_indices:
            train_rollout = train_indices
        else:
            clip_start = int(train_targets[0])
            rollout_start = max(0, clip_start - int(args.clip_history_burn_in))
            train_rollout = list(range(rollout_start, int(train_targets[-1]) + 1))
        train = _run_split(
            model, controllers, action_encoder, head, dataset, cache, stats, train_targets,
            device=device, train=True, optimizer=optimizer, epoch=epoch,
            group_sampler=group_sampler,
            action_weight=args.action_weight,
            latent_weight=args.latent_weight,
            controller_count_weight=args.controller_count_weight,
            grpo_weight=args.grpo_weight,
            grpo_rollout_budget=args.grpo_rollout_budget,
            degree_power=args.degree_power,
            volatility_power=args.volatility_power,
            grpo_interval=args.grpo_interval,
            grpo_warmup_epochs=args.grpo_warmup_epochs,
            grpo_clip=args.grpo_clip,
            grpo_kl=args.grpo_kl,
            entropy_weight=args.entropy_weight,
            magnitude_reward_weight=args.magnitude_reward_weight,
            changed_center_loss_weight=args.changed_center_loss_weight,
            changed_descriptor_loss_weight=args.changed_descriptor_loss_weight,
            changed_mae_loss_weight=args.changed_mae_loss_weight,
            changed_rmse_loss_weight=args.changed_rmse_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
            max_actions=args.max_actions,
            max_addition_candidates=args.max_addition_candidates,
            action_group_fusion=args.action_group_fusion,
            direction_projection_margin=args.direction_projection_margin,
            direction_projection_confidence=args.direction_projection_confidence,
            direction_class_balance_power=args.direction_class_balance_power,
            causal_structure_context=args.causal_structure_context,
            causal_structure_history_context=args.causal_structure_history_context,
            group_aware_decoder_context=args.group_aware_decoder_context,
            direction_group_aware_context_only=args.direction_group_aware_context_only,
            direction_backpropagate_context=args.direction_backpropagate_context,
            direction_transfer_bypass=args.pretrained_transfer_gate_direction_bypass,
            direction_transfer_mix=args.pretrained_direction_transfer_mix,
            proposal_seed=args.seed,
            rollout_indices=train_rollout,
            target_indices=set(train_targets),
            action_candidate_cache=action_candidate_cache,
            batch_edge_grpo_rollouts=args.batch_edge_grpo_rollouts,
            tensorized_grpo_rewards=args.tensorized_grpo_rewards,
            reuse_controller_policy_states=args.reuse_controller_policy_states,
            released_history_cache=released_history_cache,
        )
        if epoch % args.val_every:
            continue
        with torch.inference_mode():
            validation_rollout, validation_targets = _evaluation_rollout_indices(
                dataset, val_indices, args.history_window
            )
            validation = _run_split(
                model, controllers, action_encoder, head, dataset, cache, stats, val_indices,
                device=device, train=False,
                rollout_indices=validation_rollout,
                target_indices=validation_targets,
                max_actions=args.max_actions,
                max_addition_candidates=args.max_addition_candidates,
                degree_power=args.degree_power,
                volatility_power=args.volatility_power,
                action_group_fusion=args.action_group_fusion,
                direction_projection_margin=args.direction_projection_margin,
                direction_projection_confidence=args.direction_projection_confidence,
                direction_class_balance_power=args.direction_class_balance_power,
                causal_structure_context=args.causal_structure_context,
                causal_structure_history_context=args.causal_structure_history_context,
                group_aware_decoder_context=args.group_aware_decoder_context,
                direction_group_aware_context_only=args.direction_group_aware_context_only,
                direction_backpropagate_context=args.direction_backpropagate_context,
                direction_transfer_bypass=args.pretrained_transfer_gate_direction_bypass,
                direction_transfer_mix=args.pretrained_direction_transfer_mix,
                proposal_seed=args.seed,
                action_candidate_cache=action_candidate_cache,
                released_history_cache=released_history_cache,
            )

        score = (
            float(args.selection_change_f1_weight) * validation["change_macro_f1"]
            - validation["change_mae"]
            - float(args.selection_change_rmse_weight) * validation["change_rmse"]
        )
        history.append(
            {
                "epoch": epoch,
                "train": train,
                "validation": validation,
                "score": score,
            }
        )
        print(
            f"epoch={epoch:03d} train_mae={train['mae']:.5f} train_f1={train['macro_f1']:.5f} "
            f"val_mae={validation['mae']:.5f} val_rmse={validation['rmse']:.5f} "
            f"val_change_mae={validation['change_mae']:.5f} "
            f"val_change_rmse={validation['change_rmse']:.5f} "
            f"val_change_f1={validation['change_macro_f1']:.5f}",
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "controllers": controllers.state_dict(),
                    "action_encoder": action_encoder.state_dict(),
                    "head": head.state_dict(),
                    "stats": stats,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "group_sampler": group_sampler.state_dict(),
                },
                checkpoint,
            )
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"]); controllers.load_state_dict(saved["controllers"])
    action_encoder.load_state_dict(saved["action_encoder"]); head.load_state_dict(saved["head"])
    with torch.no_grad():
        test_rollout, test_targets = _evaluation_rollout_indices(
            dataset, test_indices, args.history_window
        )
        test = _run_split(
            model, controllers, action_encoder, head, dataset, cache, stats, test_indices,
            device=device, train=False,
            rollout_indices=test_rollout,
            target_indices=test_targets,
            max_actions=args.max_actions,
            max_addition_candidates=args.max_addition_candidates,
            degree_power=args.degree_power,
            volatility_power=args.volatility_power,
            action_group_fusion=args.action_group_fusion,
            direction_projection_margin=args.direction_projection_margin,
            direction_projection_confidence=args.direction_projection_confidence,
            direction_class_balance_power=args.direction_class_balance_power,
            causal_structure_context=args.causal_structure_context,
            causal_structure_history_context=args.causal_structure_history_context,
            group_aware_decoder_context=args.group_aware_decoder_context,
            direction_group_aware_context_only=args.direction_group_aware_context_only,
            direction_backpropagate_context=args.direction_backpropagate_context,
            direction_transfer_bypass=args.pretrained_transfer_gate_direction_bypass,
            direction_transfer_mix=args.pretrained_direction_transfer_mix,
            proposal_seed=args.seed,
            action_candidate_cache=action_candidate_cache,
            released_history_cache=released_history_cache,
        )


    cache.save()
    transfer_gates = None
    if args.pretrained_transfer_gate:
        graph_adapter = model.graph_transfer_adapter
        state_adapter = model.state_transfer_adapter
        if not isinstance(graph_adapter, GatedResidualTransferAdapter) or not isinstance(
            state_adapter, GatedResidualTransferAdapter
        ):
            raise RuntimeError("Pretrained transfer gates were requested but not initialized.")
        transfer_gates = {
            "graph": graph_adapter.gate_value(),
            "state": state_adapter.gate_value(),
        }
        print(
            f"[TRANSFER-GATE] graph={transfer_gates['graph']:.4f} "
            f"state={transfer_gates['state']:.4f}",
            flush=True,
        )
    payload = {
        "task": "T3_graph", "model": "WorldGraph", "dataset": args.dataset, "seed": args.seed,
        "best_epoch": best_epoch, "best_validation_score": best_score,
        "test": test, "history": history, "checkpoint": str(checkpoint),
        "transfer_gates": transfer_gates,
        "direction_transfer_mix": args.pretrained_direction_transfer_mix,
        "descriptors": DESCRIPTOR_NAMES, "statistics": {key: value.tolist() for key, value in stats.items()},
    }
    result_path = ROOT / args.results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "[T3-metric] MAE/RMSE are train-statistics-standardized across the six descriptors",
        flush=True,
    )
    print(
        f"[TEST-MEAN/T3/{args.dataset}/n=1] Change MAE={test['change_mae']:.4f} | "
        f"Change RMSE={test['change_rmse']:.4f} | MAE={test['mae']:.4f} | RMSE={test['rmse']:.4f} | "
        f"Change Macro-F1={test['change_macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
