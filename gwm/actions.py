
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator

import torch
from torch import nn


class GraphOperation(IntEnum):

    ADD_EDGE = 0
    REMOVE_EDGE = 1
    MODIFY_NODE_STATE = 2
    MODIFY_NODE_PROPERTY = 3




    ADD_NODE = 4
    REMOVE_NODE = 5


EDGE_OPERATIONS = frozenset({GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE})
VALUE_NODE_OPERATIONS = frozenset(
    {GraphOperation.MODIFY_NODE_STATE, GraphOperation.MODIFY_NODE_PROPERTY}
)
MEMBERSHIP_NODE_OPERATIONS = frozenset(
    {GraphOperation.ADD_NODE, GraphOperation.REMOVE_NODE}
)


def operation_requires_value(operation: GraphOperation) -> bool:

    return GraphOperation(operation) in VALUE_NODE_OPERATIONS


@dataclass(frozen=True)
class GraphEditAction:

    operation: GraphOperation
    target: tuple[int, ...]
    value: torch.Tensor | None = None

    def __post_init__(self) -> None:
        operation = GraphOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        expected = 2 if operation in EDGE_OPERATIONS else 1
        if len(self.target) != expected:
            raise ValueError(
                f"{operation.name} expects {expected} target node IDs, got {self.target}."
            )
        if any(int(node) < 0 for node in self.target):
            raise ValueError("Action target node IDs must be non-negative.")
        object.__setattr__(self, "target", tuple(int(node) for node in self.target))
        if operation_requires_value(operation) and self.value is None:
            raise ValueError(f"{operation.name} requires a delta value.")

    @property
    def target_nodes(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(self.target))

    def to(self, device: torch.device | str) -> "GraphEditAction":
        return GraphEditAction(
            self.operation,
            self.target,
            None if self.value is None else self.value.to(device),
        )


@dataclass(frozen=True)
class GraphEditSequence:

    actions: tuple[GraphEditAction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        if not self.actions:
            raise ValueError("GraphEditSequence must contain at least one edit.")

    def __iter__(self) -> Iterator[GraphEditAction]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def target_nodes(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                node for action in self.actions for node in action.target_nodes
            )
        )

    def operation_count(self, operation: GraphOperation) -> int:
        operation = GraphOperation(operation)
        return sum(action.operation == operation for action in self.actions)

    def to(self, device: torch.device | str) -> "GraphEditSequence":
        return GraphEditSequence(tuple(action.to(device) for action in self.actions))


class GraphActionEncoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        node_state_dim: int | None = None,
        node_property_dim: int | None = None,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or action_dim < 1:
            raise ValueError("latent_dim and action_dim must be positive.")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.node_state_dim = (
            None if node_state_dim is None else int(node_state_dim)
        )
        self.node_property_dim = (
            None if node_property_dim is None else int(node_property_dim)
        )
        self.operation_embedding = nn.Embedding(len(GraphOperation), self.action_dim)
        self.output_norm = nn.LayerNorm(self.action_dim)

    def forward(
        self, z_t: torch.Tensor, action: GraphEditAction | GraphEditSequence
    ) -> dict[str, torch.Tensor]:
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        sequence = (
            action
            if isinstance(action, GraphEditSequence)
            else GraphEditSequence((action,))
        )
        if any(node >= z_t.shape[0] for node in sequence.target_nodes):
            raise ValueError("Action target lies outside the current node universe.")
        type_tokens = [
            self.operation_embedding(
                torch.tensor(
                    int(edit.operation), dtype=torch.long, device=z_t.device
                )
            )
            for edit in sequence
        ]
        pooled = torch.stack(type_tokens, dim=0).mean(dim=0)
        nodewise = self.output_norm(pooled).unsqueeze(0).expand(
            z_t.shape[0], -1
        ).clone()
        target_mask = torch.zeros(
            z_t.shape[0], device=z_t.device, dtype=torch.bool
        )
        if sequence.target_nodes:
            target_mask[
                torch.tensor(
                    sequence.target_nodes, dtype=torch.long, device=z_t.device
                )
            ] = True
        return {
            "nodewise": nodewise,
            "global": self.output_norm(pooled),
            "target_mask": target_mask,
        }

    def encode_soft_node_action(
        self,
        z_t: torch.Tensor,
        target_probability: torch.Tensor,
        operation: GraphOperation,
        *,
        node_value: torch.Tensor | None = None,
        confidence_gate: bool = False,
        localization_mode: str = "expected",
        change_gated_value: bool = False,
        top_k: int = 0,
        support_count: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        operation = GraphOperation(operation)
        if operation not in VALUE_NODE_OPERATIONS | MEMBERSHIP_NODE_OPERATIONS:
            raise ValueError("Soft node actions are defined only for unary node edits.")
        probability = target_probability.to(device=z_t.device, dtype=z_t.dtype)
        if probability.ndim == 2 and probability.shape[1] == 1:
            probability = probability.squeeze(1)
        if probability.shape != (z_t.shape[0],):
            raise ValueError(
                "target_probability must have shape [num_nodes], got "
                f"{tuple(probability.shape)}"
            )
        probability = probability.clamp(0.0, 1.0)
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if support_count is not None:
            count = max(0, min(int(support_count), int(probability.numel())))
            selection_mask = torch.zeros_like(probability)
            if count > 0:
                selected = probability.topk(count, sorted=False).indices
                selection_mask.scatter_(0, selected, 1.0)
            probability = probability * selection_mask
        elif 0 < top_k < probability.numel():



            selected = probability.topk(int(top_k), sorted=False).indices
            selection_mask = torch.zeros_like(probability)
            selection_mask.scatter_(0, selected, 1.0)
            probability = probability * selection_mask
        if localization_mode not in {"expected", "centered", "centered_local", "local"}:
            raise ValueError(
                "localization_mode must be 'expected', 'centered', "
                "'centered_local', or 'local'."
            )
        operation_index = torch.tensor(
            int(operation), dtype=torch.long, device=z_t.device
        )
        type_bias = self.operation_embedding(operation_index)



        action_mass = probability.mean()
        confidence = (
            (2.0 * probability - 1.0).abs().mean()
            if confidence_gate
            else probability.new_ones(())
        )
        pooled = action_mass * self.output_norm(type_bias)
        nodewise = confidence * pooled.unsqueeze(0).expand(
            z_t.shape[0], -1
        )
        return {
            "nodewise": nodewise,
            "global": confidence * pooled,
            "target_mask": probability.ge(0.5),
            "target_probability": probability,
            "localization_mode": localization_mode,
            "change_gated_value": bool(change_gated_value),
            "top_k": int(top_k),
            "support_count": None if support_count is None else int(support_count),
            "confidence": confidence,
            "action_mass": action_mass,
        }

    def encode_soft_edge_actions(
        self,
        z_t: torch.Tensor,
        candidate_edges: dict[GraphOperation, torch.Tensor],
        edge_probability: dict[GraphOperation, torch.Tensor],
        *,
        chunk_size: int = 65536,
        confidence_gate: bool = False,
        operation_log_counts: dict[GraphOperation, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if z_t.ndim != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_t must have shape [num_nodes, {self.latent_dim}], got {tuple(z_t.shape)}"
            )
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        num_nodes = int(z_t.shape[0])
        endpoint_mass = z_t.new_zeros((num_nodes,))
        endpoint_mass_by_operation = {
            operation: z_t.new_zeros((num_nodes,))
            for operation in (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE)
        }
        type_mass: list[torch.Tensor] = []
        for operation in (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE):
            edges = candidate_edges.get(operation)
            probability = edge_probability.get(operation)
            if edges is None or probability is None or edges.shape[1] == 0:
                type_mass.append(z_t.new_zeros(()))
                continue
            edges = edges.to(device=z_t.device, dtype=torch.long)
            probability = probability.to(device=z_t.device, dtype=z_t.dtype)
            if probability.shape != (edges.shape[1],):
                raise ValueError(
                    f"Probability for {operation.name} must have shape "
                    f"[{edges.shape[1]}], got {tuple(probability.shape)}."
                )
            probability = probability.clamp_min(0.0)
            for start in range(0, edges.shape[1], chunk_size):
                stop = min(start + chunk_size, edges.shape[1])
                source = edges[0, start:stop]
                destination = edges[1, start:stop]
                mass = probability[start:stop]
                endpoint_mass = endpoint_mass.index_add(0, source, mass)
                endpoint_mass = endpoint_mass.index_add(0, destination, mass)
                endpoint_mass_by_operation[operation] = endpoint_mass_by_operation[
                    operation
                ].index_add(0, source, mass)
                endpoint_mass_by_operation[operation] = endpoint_mass_by_operation[
                    operation
                ].index_add(0, destination, mass)
            type_mass.append(probability.mean())
        type_tokens = [
            self.operation_embedding(
                torch.tensor(int(operation), dtype=torch.long, device=z_t.device)
            )
            for operation in (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE)
        ]
        pooled = sum(
            (mass * token for mass, token in zip(type_mass, type_tokens)),
            z_t.new_zeros((self.action_dim,)),
        )
        action_mass = torch.stack(type_mass).sum()
        operation_count_values: list[torch.Tensor] = []
        for operation in (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE):
            if operation_log_counts is None or operation not in operation_log_counts:
                operation_count_values.append(z_t.new_zeros(()))
            else:
                operation_count_values.append(
                    operation_log_counts[operation].to(
                        device=z_t.device, dtype=z_t.dtype
                    ).reshape(())
                )
        nodewise = pooled.unsqueeze(0).expand(num_nodes, -1).clone()
        if confidence_gate:
            all_probability = torch.cat(
                [
                    edge_probability[operation].to(z_t).clamp_min(1e-12)
                    for operation in (
                        GraphOperation.ADD_EDGE,
                        GraphOperation.REMOVE_EDGE,
                    )
                    if operation in edge_probability
                    and edge_probability[operation].numel()
                ]
            )
            normalized_probability = all_probability / all_probability.sum().clamp_min(1e-12)
            entropy = -(normalized_probability * normalized_probability.log()).sum()
            maximum_entropy = entropy.new_tensor(
                float(max(int(normalized_probability.numel()), 1))
            ).log().clamp_min(1e-6)
            confidence = (1.0 - entropy / maximum_entropy).clamp(0.0, 1.0)
        else:
            confidence = action_mass.new_ones(())
        return {
            "nodewise": confidence * self.output_norm(nodewise),
            "global": confidence * self.output_norm(pooled),
            "target_probability": endpoint_mass.clamp_max(1.0),




            "target_probability_by_operation": torch.stack(
                [
                    endpoint_mass_by_operation[GraphOperation.ADD_EDGE].clamp_max(1.0),
                    endpoint_mass_by_operation[GraphOperation.REMOVE_EDGE].clamp_max(1.0),
                ],
                dim=-1,
            ),


            "operation_log_count": torch.stack(operation_count_values),
            "confidence": confidence,
            "action_mass": action_mass,
        }


def _edge_codes(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    return edge_index[0].to(torch.long) * int(num_nodes) + edge_index[1].to(torch.long)


def is_valid_graph_action(
    action: GraphEditAction,
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    node_active_mask: torch.Tensor | None = None,
) -> bool:
    if any(node >= num_nodes for node in action.target_nodes):
        return False
    if action.operation in EDGE_OPERATIONS:
        source, destination = action.target
        code = source * int(num_nodes) + destination
        present = bool((_edge_codes(edge_index, num_nodes) == code).any().item())
        return not present if action.operation == GraphOperation.ADD_EDGE else present
    if action.operation in MEMBERSHIP_NODE_OPERATIONS and node_active_mask is not None:
        active = node_active_mask.to(device=edge_index.device, dtype=torch.bool)
        if active.shape != (num_nodes,):
            raise ValueError("node_active_mask must have shape [num_nodes].")
        present = bool(active[action.target[0]].item())
        return not present if action.operation == GraphOperation.ADD_NODE else present
    return True


def apply_graph_edit(
    *,
    x_t: torch.Tensor,
    edge_index_t: torch.Tensor,
    action: GraphEditAction,
    edge_weight_t: torch.Tensor | None = None,
    property_slice: slice | None = None,
    property_update_mode: str = "additive",
    node_activity_index: int | None = None,
    node_active_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    num_nodes = int(x_t.shape[0])
    if not is_valid_graph_action(
        action,
        num_nodes=num_nodes,
        edge_index=edge_index_t,
        node_active_mask=node_active_mask,
    ):
        raise ValueError(f"Invalid {action.operation.name} action for current graph.")
    x_next = x_t.clone()
    edge_next = edge_index_t.clone()
    weight_next = None if edge_weight_t is None else edge_weight_t.clone()

    if action.operation == GraphOperation.ADD_EDGE:
        new_edge = edge_next.new_tensor(action.target).reshape(2, 1)
        edge_next = torch.cat([edge_next, new_edge], dim=1)
        if weight_next is not None:
            new_weight = 1.0
            if action.value is not None and action.value.numel():
                new_weight = float(action.value.reshape(-1)[0].item())
            weight_next = torch.cat(
                [weight_next, weight_next.new_tensor([new_weight])], dim=0
            )
    elif action.operation == GraphOperation.REMOVE_EDGE:
        source, destination = action.target
        keep = ~(
            edge_next[0].eq(int(source)) & edge_next[1].eq(int(destination))
        )
        edge_next = edge_next[:, keep]
        if weight_next is not None:
            weight_next = weight_next[keep]
    elif action.operation in MEMBERSHIP_NODE_OPERATIONS:
        if node_activity_index is None:
            raise ValueError(
                "node_activity_index is required for ADD_NODE/REMOVE_NODE actions."
            )
        activity_index = int(node_activity_index)
        if not 0 <= activity_index < x_next.shape[1]:
            raise ValueError("node_activity_index lies outside x_t feature dimensions.")
        node = action.target[0]
        if action.operation == GraphOperation.ADD_NODE:



            x_next[node, activity_index] = 1.0
        else:
            x_next[node] = 0.0
            x_next[node, activity_index] = 0.0
            keep = ~(edge_next[0].eq(int(node)) | edge_next[1].eq(int(node)))
            edge_next = edge_next[:, keep]
            if weight_next is not None:
                weight_next = weight_next[keep]
    else:
        node = action.target[0]
        feature_slice = property_slice if action.operation == GraphOperation.MODIFY_NODE_PROPERTY else None
        if feature_slice is None:
            start, stop = 0, x_next.shape[1]
        else:
            start = 0 if feature_slice.start is None else int(feature_slice.start)
            stop = x_next.shape[1] if feature_slice.stop is None else int(feature_slice.stop)
        if not 0 <= start < stop <= x_next.shape[1]:
            raise ValueError("property_slice lies outside x_t feature dimensions.")
        delta = action.value
        assert delta is not None
        delta = delta.to(device=x_next.device, dtype=x_next.dtype).reshape(-1)
        width = stop - start
        if delta.numel() == 1:
            delta = delta.expand(width)
        if delta.numel() != width:
            raise ValueError(
                f"Action delta has {delta.numel()} values, expected {width}."
            )
        if (
            action.operation == GraphOperation.MODIFY_NODE_PROPERTY
            and property_update_mode == "logit_shift"
        ):
            current_logit = x_next[node, start:stop].clamp_min(1e-8).log()
            x_next[node, start:stop] = torch.softmax(current_logit + delta, dim=0)
        elif property_update_mode == "additive":
            x_next[node, start:stop] = x_next[node, start:stop] + delta
        else:
            raise ValueError(
                "property_update_mode must be 'additive' or 'logit_shift'."
            )

    return {
        "x": x_next,
        "edge_index": edge_next,
        "edge_weight": weight_next,
    }


def apply_graph_edit_sequence(
    *,
    x_t: torch.Tensor,
    edge_index_t: torch.Tensor,
    action: GraphEditSequence,
    edge_weight_t: torch.Tensor | None = None,
    property_slice: slice | None = None,
    property_update_mode: str = "additive",
    undirected_edges: bool = False,
    node_activity_index: int | None = None,
    node_active_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    current: dict[str, torch.Tensor | None] = {
        "x": x_t,
        "edge_index": edge_index_t,
        "edge_weight": edge_weight_t,
    }
    active = None if node_active_mask is None else node_active_mask.clone()
    for edit in action:
        current = apply_graph_edit(
            x_t=current["x"],
            edge_index_t=current["edge_index"],
            edge_weight_t=current["edge_weight"],
            action=edit,
            property_slice=property_slice,
            property_update_mode=property_update_mode,
            node_activity_index=node_activity_index,
            node_active_mask=active,
        )
        if active is not None and edit.operation in MEMBERSHIP_NODE_OPERATIONS:
            active[edit.target[0]] = edit.operation == GraphOperation.ADD_NODE
        if (
            undirected_edges
            and edit.operation
            in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
            and edit.target[0] != edit.target[1]
        ):
            reverse_edit = GraphEditAction(
                operation=edit.operation,
                target=(edit.target[1], edit.target[0]),
                value=edit.value,
            )
            current = apply_graph_edit(
                x_t=current["x"],
                edge_index_t=current["edge_index"],
                edge_weight_t=current["edge_weight"],
                action=reverse_edit,
                property_slice=property_slice,
                property_update_mode=property_update_mode,
                node_activity_index=node_activity_index,
                node_active_mask=active,
            )
    return current


def bounded_graph_edit_sequence(
    action: GraphEditSequence,
    *,
    max_abs_delta: float,
) -> GraphEditSequence:
    if max_abs_delta <= 0.0:
        raise ValueError("max_abs_delta must be positive.")
    bounded: list[GraphEditAction] = []
    for edit in action:
        if edit.operation not in {
            GraphOperation.MODIFY_NODE_STATE,
            GraphOperation.MODIFY_NODE_PROPERTY,
        }:
            bounded.append(edit)
            continue
        assert edit.value is not None
        bounded.append(
            GraphEditAction(
                operation=edit.operation,
                target=edit.target,
                value=float(max_abs_delta) * torch.tanh(edit.value),
            )
        )
    return GraphEditSequence(tuple(bounded))


def actions_target_nodes(
    actions: Iterable[GraphEditAction] | GraphEditSequence,
) -> set[int]:
    return {node for action in actions for node in action.target_nodes}
