

from __future__ import annotations

import argparse
import copy
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwm.training.runtime_cpu import configure_cpu_environment, configure_torch_runtime

_CPU_THREADS = configure_cpu_environment(overwrite=True)

import torch
from torch.func import vmap
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

configure_torch_runtime(torch, _CPU_THREADS)

from gwm.action_rl import (
    ActionAwareController,
    DynamicGroupSampler,
    RewardedSequenceRollout,
    calibrated_soft_node_action_probability,
    grpo_sequence_loss,
    node_change_localization_reward,
    representation_change_magnitude_reward,
    structure_node_weights,
)
from gwm.actions import GraphActionEncoder, GraphOperation
from gwm.architecture import (
    add_world_model_architecture_args,
    architecture_label,
    world_model_architecture_kwargs,
)
from gwm.benchmark_protocol import assert_common_runtime
from gwm.data.node_property_transition import (
    NodePropertyTransitionDataset,
    fit_semantic_change_threshold,
    incident_degree,
    semantic_change_labels,
    support_topk_jaccard_distance,
    transition_statistics,
)
from gwm.data.transition_cache import TransitionRecordCache
from gwm.latent_objective import latent_transition_terms
from gwm.model import GraphWorldModel
from gwm.pretraining import (
    load_worldgraph_pretrained_activity_controller,
    load_worldgraph_pretrained_backbone,
    load_worldgraph_pretrained_task_module,
)
from gwm.property_transition import property_prediction_loss
from gwm.t2_semantic_training import (
    SemanticHistoryCache,
    OnlineSemanticChangeHistory,
    OnlineSemanticPropertyHistory,
    _support_jaccard_distance,
    cached_semantic_change_labels,
    _candidate_binary_loss,
    _candidate_focal_binary_loss,
    _candidate_pairwise_ranking_loss,
    _current_semantic_reconstruction_loss,
    _edge_weight_for_dataset,
    _semantic_prediction,
    apply_t2_thresholds,
    evaluate_t2_semantic_sequence,
    fit_t2_training_statistics,
    build_semantic_history_cache,
    select_t2_thresholds,
    t2_transition_to_device,
)


def _load_semantic_history_payload(path: Path) -> dict[str, Any]:

    try:
        return torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=False)
from gwm.utils import count_parameters, resolve_device, save_json, seed_everything


DEFAULT_PROCESSED = {
    "trade": "data/processed/trade_T1.pt",
    "genre": "data/processed/genre_T1.pt",
    "reddit": "data/processed/reddit_T1.pt",
}

NODE_OPERATIONS = (
    GraphOperation.ADD_NODE,
    GraphOperation.REMOVE_NODE,
    GraphOperation.MODIFY_NODE_PROPERTY,
)


def _count_support_operations(mode: str) -> frozenset[GraphOperation]:

    mapping = {
        "none": frozenset(),
        "add": frozenset((GraphOperation.ADD_NODE,)),
        "edits": frozenset((GraphOperation.ADD_NODE, GraphOperation.REMOVE_NODE)),
        "all": frozenset(NODE_OPERATIONS),
        "remove": frozenset((GraphOperation.REMOVE_NODE,)),
    }
    return mapping[mode]


def _resolved_count_operations(
    mode: str,
    *,
    fallback: frozenset[GraphOperation],
) -> frozenset[GraphOperation]:

    if mode == "same":
        return fallback
    return _count_support_operations(mode)


def _decoder_consistency_operations(mode: str) -> frozenset[GraphOperation]:
    mapping = {
        "all": frozenset(NODE_OPERATIONS),
        "property": frozenset((GraphOperation.MODIFY_NODE_PROPERTY,)),
        "activity": frozenset((GraphOperation.ADD_NODE, GraphOperation.REMOVE_NODE)),
        "removal": frozenset((GraphOperation.REMOVE_NODE,)),
    }
    return mapping[mode]


@contextmanager
def _preserve_world_rng(device: torch.device):

    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        yield


def _path(value: str | None, default: Path) -> Path:
    return ROOT / value if value else default


def _display_path(path: Path) -> str:

    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _fine_timing_start(device: torch.device | None = None) -> float:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _fine_timing_add(
    timings: dict[str, float] | None,
    name: str,
    started: float,
    device: torch.device | None = None,
) -> None:
    if timings is None:
        return
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def _chronological_train_transition_ids(
    dataset: NodePropertyTransitionDataset,
) -> list[int]:

    base = dataset.base
    target_offset = 1 if bool(getattr(base, "property_state_in_graph", True)) else 0
    return [
        transition_id
        for transition_id in range(int(base.transition_count))
        if str(base.snapshots[transition_id + target_offset]["split"]) == "train"
    ]


def _training_clip_ids(
    train_ids: list[int],
    *,
    epoch: int,
    max_train_transitions: int | None,
    train_clip_transitions: int | None,
) -> tuple[list[int], list[int]]:

    if not train_ids:
        raise ValueError("No chronological training transitions were found.")
    if train_clip_transitions is None:
        limit = len(train_ids) if max_train_transitions is None else min(
            int(max_train_transitions), len(train_ids)
        )
        return train_ids[:limit], []

    clip_size = min(int(train_clip_transitions), len(train_ids))
    last_start = len(train_ids) - clip_size



    chunks_per_cycle = (len(train_ids) + clip_size - 1) // clip_size
    chunk = (int(epoch) - 1) % chunks_per_cycle
    start = min(chunk * clip_size, last_start)
    return train_ids[start : start + clip_size], train_ids[:start]


class _PastOperationRates:

    def __init__(self, device: torch.device) -> None:
        self.positive = {op: torch.zeros((), device=device) for op in NODE_OPERATIONS}
        self.observed = {op: torch.zeros((), device=device) for op in NODE_OPERATIONS}

    def probability(self, operation: GraphOperation) -> tuple[torch.Tensor, torch.Tensor]:
        return self.positive[operation], self.observed[operation]

    def observe(self, operation: GraphOperation, labels: torch.Tensor, known: torch.Tensor) -> None:
        device = self.positive[operation].device
        labels = labels.to(device=device, dtype=torch.float32)
        known = known.to(device=device, dtype=torch.bool)
        if operation in {GraphOperation.ADD_NODE, GraphOperation.REMOVE_NODE}:

            self.positive[operation] += labels.sum().detach()
            self.observed[operation] += float(labels.numel())
        else:


            self.positive[operation] += labels[known].sum().detach()
            self.observed[operation] += known.to(dtype=torch.float32).sum().detach()

    def clone(self) -> "_PastOperationRates":

        device = next(iter(self.positive.values())).device
        result = _PastOperationRates(device)
        result.positive = {
            operation: value.clone() for operation, value in self.positive.items()
        }
        result.observed = {
            operation: value.clone() for operation, value in self.observed.items()
        }
        return result


class _RecentOperationCountHistory:

    def __init__(self, window: int) -> None:
        if int(window) < 1:
            raise ValueError("Recent operation-count history requires window >= 1.")
        self.window = int(window)
        self.values = {
            operation: deque(maxlen=self.window) for operation in NODE_OPERATIONS
        }

    def log_count(self, operation: GraphOperation, *, device: torch.device) -> torch.Tensor | None:
        values = self.values[operation]
        if not values:
            return None
        return torch.log1p(torch.tensor(float(sum(values) / len(values)), device=device))

    def observe(self, operation: GraphOperation, labels: torch.Tensor, known: torch.Tensor) -> None:
        if bool(known.any()):
            self.values[operation].append(
                float(labels[known].to(dtype=torch.float32).sum().detach().cpu().item())
            )


def _candidate_nodes(transition: dict[str, Any], operation: GraphOperation) -> torch.Tensor:

    cached_key = {
        GraphOperation.ADD_NODE: "node_addition_candidates",
        GraphOperation.REMOVE_NODE: "node_removal_candidates",
        GraphOperation.MODIFY_NODE_PROPERTY: "node_property_candidates",
    }.get(operation)
    if cached_key is not None and cached_key in transition:
        return transition[cached_key]

    eligible = transition.get("node_edit_eligible_mask")
    if eligible is None:
        eligible = torch.ones_like(transition["node_active_t"], dtype=torch.bool)
    else:
        eligible = eligible.to(torch.bool)
    if operation == GraphOperation.ADD_NODE:
        return ((~transition["node_active_t"]) & eligible).nonzero(as_tuple=False).flatten()
    if operation == GraphOperation.REMOVE_NODE:
        return (transition["node_active_t"] & eligible).nonzero(as_tuple=False).flatten()
    if operation == GraphOperation.MODIFY_NODE_PROPERTY:


        return torch.unique(transition["property_node_ids_t"].long())
    raise ValueError(f"Unsupported node-level operation {operation!r}")


class _StaticCandidateIndexCache:

    def __init__(self, max_bytes: int = 16 * 1024**2) -> None:
        self.records = TransitionRecordCache(max_bytes)
        self.hits = 0
        self.misses = 0

    def attach(self, transition: dict[str, Any]) -> dict[str, Any]:
        transition_id = int(transition.get("transition_id", -1))
        key = ("t1-static-candidates", transition_id)
        payload = self.records.get(key) if transition_id >= 0 else None
        if payload is None:
            self.misses += 1
            active = transition["node_active_t"].detach().cpu().to(torch.bool)
            eligible = transition.get("node_edit_eligible_mask")
            eligible = (
                torch.ones_like(active)
                if eligible is None
                else eligible.detach().cpu().to(torch.bool)
            )
            payload = {
                "node_addition_candidates": ((~active) & eligible)
                .nonzero(as_tuple=False)
                .flatten(),
                "node_removal_candidates": (active & eligible)
                .nonzero(as_tuple=False)
                .flatten(),
                "node_property_candidates": torch.unique(
                    transition["property_node_ids_t"].detach().cpu().long()
                ),
                "semantic_comparable_indices": transition[
                    "semantic_comparable_mask"
                ]
                .detach()
                .cpu()
                .to(torch.bool)
                .nonzero(as_tuple=False)
                .flatten(),
            }
            if transition_id >= 0:
                payload = self.records.put(key, payload)
        else:
            self.hits += 1
        prepared = dict(transition)
        prepared.update(payload)
        return prepared


class _CausalValidationPreparationCache:

    _PAYLOAD_KEYS = (
        "semantic_history_source_features",
        "semantic_history_target_features",
        "semantic_change_distance_cached",
    )

    def __init__(self, max_bytes: int = 64 * 1024**2) -> None:
        self.records = TransitionRecordCache(max_bytes)
        self.hits = 0
        self.misses = 0

    def prepare(
        self,
        transition: dict[str, Any],
        history: OnlineSemanticChangeHistory,
        *,
        enabled: bool,
    ) -> dict[str, Any]:


        if not enabled or not bool(transition.get("defer_dense_features", False)):
            return _with_causal_semantic_history_features(
                transition, history, enabled=enabled
            )
        transition_id = int(transition.get("transition_id", -1))
        view = "full" if "x_next" in transition else "forecast"
        key = ("t1-validation-preparation", view, transition_id)
        payload = self.records.get(key) if transition_id >= 0 else None
        if payload is None:
            self.misses += 1
            prepared = _with_causal_semantic_history_features(
                transition, history, enabled=True
            )
            payload = {
                name: prepared[name]
                for name in self._PAYLOAD_KEYS
                if name in prepared
            }
            if transition_id >= 0:
                payload = self.records.put(key, payload)
            return prepared
        self.hits += 1
        prepared = dict(transition)
        prepared.update(payload)
        return prepared


def _property_history_logit_bias(
    history: OnlineSemanticChangeHistory | None,
    candidates: torch.Tensor,
    *,
    weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:

    if history is None or weight <= 0.0 or candidates.numel() == 0:
        return None
    score = history.score(candidates).to(device=device, dtype=dtype)
    scale = score.std(unbiased=False).clamp_min(1e-4)
    return float(weight) * (score - score.mean()) / scale


def _operation_targets(
    transition: dict[str, Any],
    operation: GraphOperation,
    *,
    semantic_threshold: float,
    semantic_topk: int,
    semantic_history_cache: SemanticHistoryCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    candidates = _candidate_nodes(transition, operation)
    num_nodes = int(transition["node_active_t"].numel())
    full = torch.zeros(num_nodes, device=transition["x_t"].device, dtype=torch.bool)
    if operation == GraphOperation.ADD_NODE:
        labels = transition["node_active_next"].index_select(0, candidates)
        known = torch.ones_like(labels, dtype=torch.bool)
        full = transition["node_added"].to(torch.bool)
        return candidates, labels, known, full
    if operation == GraphOperation.REMOVE_NODE:
        labels = (~transition["node_active_next"]).index_select(0, candidates)
        known = torch.ones_like(labels, dtype=torch.bool)
        full = transition["node_removed"].to(torch.bool)
        return candidates, labels, known, full
    if operation != GraphOperation.MODIFY_NODE_PROPERTY:
        raise ValueError(f"Unsupported operation {operation!r}")

    labels = torch.zeros(candidates.numel(), device=full.device, dtype=torch.bool)
    known = torch.zeros_like(labels)
    comparable_indices = transition.get("semantic_comparable_indices")
    if comparable_indices is None:
        comparable_indices = transition["semantic_comparable_mask"].to(
            torch.bool
        ).nonzero(as_tuple=False).flatten()
    if comparable_indices.numel():
        ids = transition["semantic_node_ids"].index_select(
            0, comparable_indices
        ).long()
        cached_labels = None
        cached_distance = transition.get("semantic_change_distance_cached")
        if cached_distance is not None:
            cached_labels = (
                cached_distance.gt(float(semantic_threshold)), cached_distance
            )
        if semantic_history_cache is not None:
            if cached_labels is None:
                cached_labels = cached_semantic_change_labels(
                    semantic_history_cache,
                    transition_id=int(transition.get("transition_id", -1)),
                    node_ids=transition["semantic_node_ids"],
                    comparable=transition["semantic_comparable_mask"],
                    threshold=semantic_threshold,
                )
        if cached_labels is None:
            changed, _ = semantic_change_labels(
                transition["semantic_current"].index_select(0, comparable_indices),
                transition["semantic_next"].index_select(0, comparable_indices),
                threshold=semantic_threshold,
                topk=semantic_topk,
            )
        else:
            changed = cached_labels[0].to(device=full.device)
        full.index_copy_(0, ids, changed)
        candidate_lookup = torch.full(
            (num_nodes,), -1, device=full.device, dtype=torch.long
        )
        candidate_lookup[candidates] = torch.arange(candidates.numel(), device=full.device)
        positions = candidate_lookup.index_select(0, ids)
        labels.index_copy_(0, positions, changed)
        known.index_fill_(0, positions, True)
    return candidates, labels, known, full


def _controller_key(operation: GraphOperation) -> str:
    return operation.name.lower()


def _configured_action_modes(args: argparse.Namespace) -> dict[GraphOperation, str]:

    raw = {
        GraphOperation.ADD_NODE: args.add_action_mode,
        GraphOperation.REMOVE_NODE: args.remove_action_mode,
        GraphOperation.MODIFY_NODE_PROPERTY: args.property_action_mode,
    }
    return {operation: mode for operation, mode in raw.items() if mode is not None}


def _with_causal_semantic_history_features(
    transition: dict[str, Any],
    history: OnlineSemanticChangeHistory,
    *,
    enabled: bool,
) -> dict[str, Any]:

    if not enabled:
        return transition

    num_nodes = int(transition["x_t"].shape[0])
    all_nodes = torch.arange(num_nodes, dtype=torch.long)
    source_score = history.score(all_nodes).to(dtype=transition["x_t"].dtype)
    source_seen = history.distance_seen.to(dtype=transition["x_t"].dtype)

    target_score = source_score.clone()
    target_seen = source_seen.clone()
    target_ids = transition["property_node_ids"].detach().cpu().long()
    target_values = transition["property_target"].detach().cpu().float()
    if target_ids.numel():
        prior_seen = history.last_seen.index_select(0, target_ids)
        if bool(prior_seen.any()):
            seen_ids = target_ids[prior_seen]
            cached_target_ids = None
            cached_target_index = None
            cached_target_valid = None
            transition_id = int(transition.get("transition_id", -1))
            if history._cache is not None and transition_id >= 0:
                cached_target_ids = history._cache.target_node_ids.get(transition_id)
                cached_target_index = history._cache.target_index.get(transition_id)
                cached_target_valid = history._cache.target_valid.get(transition_id)
            if (
                cached_target_ids is not None
                and cached_target_index is not None
                and cached_target_valid is not None
                and torch.equal(cached_target_ids, target_ids)
            ):
                distance = _support_jaccard_distance(
                    history._last_support_index.index_select(0, seen_ids),
                    history._last_support_valid.index_select(0, seen_ids),
                    cached_target_index[prior_seen],
                    cached_target_valid[prior_seen],
                )
            else:
                distance = support_topk_jaccard_distance(
                    history.last_value.index_select(0, seen_ids),
                    target_values[prior_seen],
                    topk=history.topk,
                )
            distance = distance.to(dtype=target_score.dtype)
            target_score.index_copy_(0, seen_ids, distance)
            target_seen.index_fill_(0, seen_ids, 1.0)

    result = dict(transition)
    if history._cache is not None:
        cached_target = cached_semantic_change_labels(
            history._cache,
            transition_id=int(transition.get("transition_id", -1)),
            node_ids=transition["semantic_node_ids"],
            comparable=transition["semantic_comparable_mask"],


            threshold=0.0,
        )
        if cached_target is not None:
            result["semantic_change_distance_cached"] = cached_target[1]
    source_features = torch.stack([source_score, source_seen], dim=-1)
    if bool(transition.get("defer_dense_features", False)):




        result["semantic_history_source_features"] = source_features
    else:
        result["x_t"] = torch.cat([transition["x_t"], source_features], dim=-1)



    if "x_next" in transition:
        target_features = torch.stack(
            [
                target_score.to(dtype=transition["x_next"].dtype),
                target_seen.to(dtype=transition["x_next"].dtype),
            ],
            dim=-1,
        )
        if bool(transition.get("defer_dense_features", False)):
            result["semantic_history_target_features"] = target_features
        else:
            result["x_next"] = torch.cat(
                [transition["x_next"], target_features], dim=-1
            )
    return result


def _attach_causal_property_reference(
    outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    history: OnlineSemanticPropertyHistory,
    *,
    mode: str,
) -> None:
    if mode == "current":
        return
    if mode != "historical_mean":
        raise ValueError(f"Unsupported semantic reference mode: {mode}")
    reference, observed = history.reference(transition["semantic_node_ids"])
    device = outputs["latent_next"].device
    outputs["_semantic_property_reference"] = reference.to(device=device)
    outputs["_semantic_property_reference_observed_mask"] = observed.to(device=device)


def _new_controllers(
    *,
    latent_dim: int,
    hidden_dim: int,
    property_dim: int,
    policy_dim: int,
    controller_remove_observed_degree: bool,
    device: torch.device,
    controller_history_features: bool = False,
    controller_history_feature_dim: int = 0,
    controller_addition_source_visibility: bool = False,
) -> torch.nn.ModuleDict:



    if controller_history_features and controller_history_feature_dim < 1:
        raise ValueError(
            "controller_history_feature_dim must be positive when history features are enabled"
        )
    modules: dict[str, ActionAwareController] = {}
    for operation in NODE_OPERATIONS:
        use_observed_degree = bool(
            controller_remove_observed_degree
            and operation == GraphOperation.REMOVE_NODE
        )
        history_dim = (
            int(controller_history_feature_dim) if controller_history_features else 0
        )
        use_source_visibility = bool(
            controller_addition_source_visibility
            and operation == GraphOperation.ADD_NODE
        )
        controller = ActionAwareController(
            latent_dim,
            hidden_dim,
            policy_dim=policy_dim,
            node_property_dim=property_dim,
            node_aux_dim=int(use_observed_degree) + history_dim + int(use_source_visibility),
        )


        controller.use_observed_degree_aux = use_observed_degree
        controller.history_feature_dim = history_dim
        controller.use_source_visibility_aux = use_source_visibility
        modules[_controller_key(operation)] = controller
    return torch.nn.ModuleDict(modules).to(device)


def _controller_node_aux(
    transition: dict[str, Any], controller: ActionAwareController
) -> torch.Tensor | None:

    parts: list[torch.Tensor] = []
    if bool(getattr(controller, "use_observed_degree_aux", False)):
        degree = transition.get("node_observed_degree_t")
        if degree is None:
            degree = incident_degree(
                transition["edge_index_t"],
                int(transition["node_active_t"].numel()),
            )
        degree = torch.log1p(degree)
        parts.append((degree / degree.max().clamp_min(1.0)).unsqueeze(-1))
    history_dim = int(getattr(controller, "history_feature_dim", 0))
    if history_dim:
        history = transition.get("controller_history_features")
        if history is None:
            raise ValueError(
                "Controller history features were enabled but are missing from the transition"
            )
        expected = (int(transition["node_active_t"].numel()), history_dim)
        if history.ndim != 2 or tuple(history.shape) != expected:
            raise ValueError(
                "controller_history_features must have shape "
                f"{expected}, got {tuple(history.shape)}"
            )
        parts.append(history.to(dtype=transition["x_t"].dtype))
    if bool(getattr(controller, "use_source_visibility_aux", False)):
        source_visible = _addition_source_visibility(transition).to(
            dtype=transition["x_t"].dtype
        )
        parts.append(source_visible.unsqueeze(-1))
    if not parts:
        if controller.node_aux_dim != 0:
            raise ValueError("Controller auxiliary layout is inconsistent with node_aux_dim")
        return None
    result = torch.cat(parts, dim=-1)
    if result.shape[1] != controller.node_aux_dim:
        raise ValueError("Controller auxiliary feature dimension is inconsistent")
    return result


def _vectorized_controller_node_forwards(
    *,
    controllers: torch.nn.ModuleDict,
    transition: dict[str, Any],
    z_t: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[
    dict[GraphOperation, torch.Tensor],
    dict[GraphOperation, torch.Tensor],
]:

    ordered = [controllers[_controller_key(op)] for op in NODE_OPERATIONS]
    common_input = torch.cat([z_t, hidden], dim=-1)
    linear_weights = torch.stack(
        [controller.node_encoder[0].weight for controller in ordered]
    )
    linear_biases = torch.stack(
        [controller.node_encoder[0].bias for controller in ordered]
    )
    projected = torch.einsum("ni,opi->onp", common_input, linear_weights)
    projected = projected + linear_biases[:, None, :]

    layer_norms = [controller.node_encoder[1] for controller in ordered]
    eps = float(layer_norms[0].eps)
    if any(float(layer_norm.eps) != eps for layer_norm in layer_norms[1:]):
        raise ValueError("Controller LayerNorm eps values must match")
    mean = projected.mean(dim=-1, keepdim=True)
    variance = (projected - mean).square().mean(dim=-1, keepdim=True)
    normalized = (projected - mean) * torch.rsqrt(variance + eps)
    norm_weights = torch.stack([layer_norm.weight for layer_norm in layer_norms])
    norm_biases = torch.stack([layer_norm.bias for layer_norm in layer_norms])
    states_tensor = F.gelu(
        normalized * norm_weights[:, None, :] + norm_biases[:, None, :]
    )

    states: dict[GraphOperation, torch.Tensor] = {}
    for index, (operation, controller) in enumerate(zip(NODE_OPERATIONS, ordered)):
        state = states_tensor[index]
        if controller.node_aux_encoder is not None:
            auxiliary = _controller_node_aux(transition, controller)
            if auxiliary is None:
                raise ValueError("Controller auxiliary input is missing")
            state = state + F.linear(
                auxiliary.to(device=z_t.device, dtype=z_t.dtype),
                controller.node_aux_encoder.weight,
                bias=None,
            )
        states[operation] = state

    target_weights = torch.stack(
        [controller.node_target_head.weight.squeeze(0) for controller in ordered]
    )
    target_biases = torch.stack(
        [controller.node_target_head.bias.squeeze(0) for controller in ordered]
    )
    logits_tensor = torch.einsum("onp,op->on", states_tensor, target_weights)
    logits_tensor = logits_tensor + target_biases[:, None]


    logits = {
        operation: (
            controller.node_target_head(states[operation]).squeeze(-1)
            if controller.node_aux_encoder is not None
            else logits_tensor[index]
        )
        for index, (operation, controller) in enumerate(zip(NODE_OPERATIONS, ordered))
    }
    return states, logits


def _addition_source_visibility(transition: dict[str, Any]) -> torch.Tensor:

    cached = transition.get("node_source_visibility")
    if cached is not None:
        return cached
    x_t = transition["x_t"]
    if x_t.ndim != 2:
        raise ValueError("x_t must be a node-feature matrix")
    return x_t.detach().abs().sum(dim=-1).gt(1e-8)


def _node_activity_observed_context(
    transition: dict[str, Any],
    *,
    enabled: bool,
    include_source_visibility: bool = False,
    include_full_state: bool = False,
    include_history: bool = False,
) -> torch.Tensor | None:

    if not enabled:
        return None
    active = transition["node_active_t"].to(dtype=torch.float32).unsqueeze(-1)
    degree = transition.get("node_observed_degree_t")
    if degree is None:
        degree = incident_degree(
            transition["edge_index_t"], int(transition["node_active_t"].numel())
        )
    degree = torch.log1p(degree)
    degree = (degree / degree.max().clamp_min(1.0)).unsqueeze(-1)
    parts = [active, degree.to(dtype=active.dtype)]
    if include_source_visibility:




        parts.append(
            _addition_source_visibility(transition)
            .to(device=active.device, dtype=active.dtype)
            .unsqueeze(-1)
        )
    if include_full_state:




        parts.append(transition["x_t"].to(dtype=active.dtype))
    if include_history:





        history = transition.get("controller_history_features")
        if history is None:
            raise ValueError(
                "node activity history context requires controller history features"
            )
        if history.ndim != 2 or history.shape[0] != active.shape[0]:
            raise ValueError(
                "controller_history_features must align with node activity rows"
            )
        parts.append(history.to(device=active.device, dtype=active.dtype))
        trajectory = transition.get("activity_trajectory_features")
        if trajectory is not None:
            if trajectory.ndim != 2 or trajectory.shape[0] != active.shape[0]:
                raise ValueError(
                    "activity trajectory features must align with node activity rows"
                )
            parts.append(
                trajectory.to(device=active.device, dtype=active.dtype)
            )
    return torch.cat(parts, dim=-1)


def _node_activity_action_context(
    action_details: dict[str, Any],
    *,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> torch.Tensor | None:

    if not enabled:
        return None
    proposals = action_details.get("proposals", {})
    if not proposals:
        return torch.zeros((num_nodes, len(NODE_OPERATIONS)), device=device, dtype=dtype)
    columns: list[torch.Tensor] = []
    for operation in NODE_OPERATIONS:
        probability = proposals[operation.name]["target_probability"]
        if probability.ndim != 1 or probability.shape[0] != num_nodes:
            raise ValueError(
                "Each node-operation action probability must have shape [num_nodes]."
            )
        columns.append(probability.to(device=device, dtype=dtype))
    return torch.stack(columns, dim=-1)


def _node_activity_sequence_context(
    sequence: Any,
    *,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> torch.Tensor | None:

    if not enabled:
        return None
    context = torch.zeros((num_nodes, len(NODE_OPERATIONS)), device=device, dtype=dtype)
    column_by_operation = {operation: index for index, operation in enumerate(NODE_OPERATIONS)}
    for edit in sequence:
        column = column_by_operation.get(edit.operation)
        if column is None:
            continue
        targets = torch.tensor(edit.target_nodes, device=device, dtype=torch.long)
        context[targets, column] = 1.0
    return context


def _soft_action(
    *,
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    controllers: torch.nn.ModuleDict,
    transition: dict[str, Any],
    dataset_name: str,
    hidden: torch.Tensor,
    rates: _PastOperationRates,
    enabled: bool,
    action_mode: str,
    action_mode_overrides: dict[GraphOperation, str] | None,
    action_topk: int,
    count_support_operations: frozenset[GraphOperation],
    count_decision_operations: frozenset[GraphOperation],
    use_property_magnitude: bool,
    property_history: OnlineSemanticChangeHistory | None,
    property_history_action_weight: float,
    controller_gradient: bool = False,
    vectorized_controller_forward: bool = False,
    detach_action_encoder: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, Any]]:




    edge_weight = _edge_weight_for_dataset(transition, dataset_name)
    z_raw = model.encode_observed_graph(
        transition["x_t"],
        transition["edge_index_t"],
        edge_weight,
        topology_cache_key=int(transition["transition_id"]),
    )
    z_t = model._normalize_latent(z_raw)
    if not enabled:
        return z_raw, None, z_t, {
            "action_mass": z_t.new_zeros(()),
            "proposals": {},
        }

    tokens: list[torch.Tensor] = []
    proposals: dict[str, Any] = {}
    vectorized_states: dict[GraphOperation, torch.Tensor] = {}
    vectorized_logits: dict[GraphOperation, torch.Tensor] = {}
    if vectorized_controller_forward:
        with (nullcontext() if controller_gradient else torch.no_grad()):
            vectorized_states, vectorized_logits = (
                _vectorized_controller_node_forwards(
                    controllers=controllers,
                    transition=transition,
                    z_t=z_t.detach(),
                    hidden=hidden.detach(),
                )
            )
    for operation in NODE_OPERATIONS:
        controller = controllers[_controller_key(operation)]
        candidates = _candidate_nodes(transition, operation)



        with (nullcontext() if controller_gradient else torch.no_grad()):
            if vectorized_controller_forward:
                policy_state = vectorized_states[operation]
                full_logits = vectorized_logits[operation]
            else:
                policy_state = controller.node_state(
                    z_t.detach(),
                    hidden.detach(),
                    node_aux=_controller_node_aux(transition, controller),
                )
                full_logits = controller.node_target_head(policy_state).squeeze(-1)
            logits = full_logits
            if operation == GraphOperation.MODIFY_NODE_PROPERTY:
                history_bias = _property_history_logit_bias(
                    property_history,
                    transition.get("node_property_candidates_cpu", candidates),
                    weight=property_history_action_weight,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                if history_bias is not None:
                    logits = logits.clone()
                    logits.index_add_(0, candidates, history_bias)
            positive, observed = rates.probability(operation)
            probability = calibrated_soft_node_action_probability(
                logits,
                historical_positive_count=positive,
                historical_observation_count=observed,
                mode="past_global_prior",
            )
            legal = torch.zeros_like(probability)
            if candidates.numel():
                legal.index_fill_(0, candidates, 1.0)
            probability = probability * legal
            controller_probability = torch.sigmoid(logits) * legal
            support_count = None
            predicted_log_count = None
            if operation in (count_support_operations | count_decision_operations):





                predicted_log_count = torch.nan_to_num(
                    controller.node_log_count(policy_state).detach(),
                    nan=0.0,
                    posinf=12.0,
                    neginf=0.0,
                ).clamp(0.0, 12.0)
            if operation in count_support_operations and predicted_log_count is not None:
                if detach_action_encoder:



                    support_count = torch.expm1(
                        predicted_log_count.clamp(0.0, 12.0)
                    ).round().clamp(0, int(candidates.numel())).to(torch.long)
                    order = probability.argsort(descending=True)
                    keep_sorted = torch.arange(
                        probability.numel(), device=probability.device
                    ).lt(support_count)
                    selection_mask = torch.zeros_like(probability).scatter(
                        0, order, keep_sorted.to(dtype=probability.dtype)
                    )
                    probability = probability * selection_mask
                else:
                    support_count = min(
                        max(int(torch.expm1(predicted_log_count.clamp(0.0, 12.0)).round().item()), 0),
                        int(candidates.numel()),
                    )
            node_value = None
            if (
                operation == GraphOperation.MODIFY_NODE_PROPERTY
                and use_property_magnitude
                and not detach_action_encoder
            ):


                node_value = controller.property_mu_head(policy_state)



        action_z = z_t.detach() if detach_action_encoder else z_t
        action_probability = probability.detach() if detach_action_encoder else probability
        action_node_value = (
            node_value.detach() if detach_action_encoder and node_value is not None
            else node_value
        )


        with torch.no_grad() if detach_action_encoder else nullcontext():
            encoded = action_encoder.encode_soft_node_action(
                action_z,
                action_probability,
                operation,
                node_value=action_node_value,
                localization_mode=(action_mode_overrides or {}).get(operation, action_mode),
                top_k=action_topk,


                support_count=(
                    None if torch.is_tensor(support_count) else support_count
                ),
                change_gated_value=(
                    operation == GraphOperation.MODIFY_NODE_PROPERTY
                    and use_property_magnitude
                ),
            )





        tokens.append(
            encoded["global"] if detach_action_encoder else encoded["nodewise"]
        )
        proposal = {
            "candidate_count": int(candidates.numel()),
            "candidates": candidates,
            "action_mass": encoded["action_mass"].detach(),
            "predicted_count": probability.index_select(
                0, candidates
            ).sum().round().detach()
            if candidates.numel()
            else probability.new_zeros(()),
            "support_count": support_count,
            "predicted_log_count": predicted_log_count,




            "target_probability": encoded["target_probability"],
            "controller_probability": controller_probability,




            "_policy_state": policy_state,
            "_full_logits": full_logits,
        }
        proposals[operation.name] = proposal
    return z_raw, torch.stack(tokens).sum(dim=0), z_t, {
        "action_mass": torch.stack(
            [item["action_mass"] for item in proposals.values()]
        ).sum(),
        "proposals": proposals,
    }


@torch.no_grad()
def _prepare_training_clip_state(
    *,
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    controllers: torch.nn.ModuleDict,
    dataset: NodePropertyTransitionDataset,
    prefix_ids: list[int],
    device: torch.device,
    semantic_threshold: float,
    semantic_topk: int,
    action_enabled: bool,
    action_mode: str,
    action_mode_overrides: dict[GraphOperation, str] | None,
    action_topk: int,
    count_support_operations: frozenset[GraphOperation],
    count_decision_operations: frozenset[GraphOperation],
    use_property_magnitude: bool,
    property_history_action_weight: float,
    semantic_history_feature: bool,
    node_activity_current_state_context: bool,
    node_activity_source_visibility_context: bool,
    node_activity_full_observed_context: bool,
    node_activity_history_context: bool,
    node_activity_action_context: bool,
    history_burn_in: int,
    static_candidate_cache: _StaticCandidateIndexCache | None = None,
    vectorized_controller_forward: bool = False,
    prefix_statistics_cache: dict[
        int,
        tuple[
            tuple[int, ...],
            _PastOperationRates,
            OnlineSemanticChangeHistory,
            OnlineSemanticPropertyHistory,
            torch.Tensor,
            int,
        ],
    ] | None = None,
) -> tuple[
    torch.Tensor,
    _PastOperationRates,
    OnlineSemanticChangeHistory,
    OnlineSemanticPropertyHistory,
    torch.Tensor,
    int,
]:

    replay_from = max(0, len(prefix_ids) - int(history_burn_in))
    num_nodes = int(dataset.metadata["num_nodes"])
    property_dim = int(dataset.metadata["official_target_dim"])








    cached_length = 0
    cached = None
    if prefix_statistics_cache:
        for candidate_length in sorted(prefix_statistics_cache, reverse=True):
            if candidate_length > replay_from:
                continue
            candidate = prefix_statistics_cache[candidate_length]
            if candidate[0] == tuple(prefix_ids[:candidate_length]):
                cached_length = candidate_length
                cached = candidate
                break
    if cached is None:
        property_history = OnlineSemanticChangeHistory(
            num_nodes, property_dim, topk=semantic_topk
        )
        property_history.attach_cache(getattr(dataset, "semantic_history_cache", None))
        property_reference_history = OnlineSemanticPropertyHistory(
            num_nodes, property_dim
        )
        rates = _PastOperationRates(device)
        historical_change_counts = torch.zeros(num_nodes, dtype=torch.float32)
        history_steps = 0
    else:
        rates = cached[1].clone()
        property_history = cached[2].clone()
        property_reference_history = cached[3].clone()
        historical_change_counts = cached[4].clone()
        history_steps = int(cached[5])

    def observe_released_transition(transition_cpu: dict[str, Any]) -> None:
        nonlocal history_steps
        property_history.observe(
            transition_cpu["property_node_ids_t"],
            transition_cpu["property_observed_t"],
            transition_id=int(transition_cpu["transition_id"]),
        )
        property_reference_history.observe(
            transition_cpu["property_node_ids_t"],
            transition_cpu["property_observed_t"],
        )
        changed_nodes = torch.zeros(num_nodes, dtype=torch.bool)
        for operation in NODE_OPERATIONS:
            _candidates, labels, known, full = _operation_targets(
                transition_cpu,
                operation,
                semantic_threshold=semantic_threshold,
                semantic_topk=semantic_topk,
                semantic_history_cache=getattr(
                    dataset, "semantic_history_cache", None
                ),
            )
            rates.observe(operation, labels, known)
            changed_nodes |= full.detach().cpu().to(torch.bool)
        historical_change_counts.add_(changed_nodes.to(torch.float32))
        history_steps += 1

    for transition_id in prefix_ids[cached_length:replay_from]:
        observe_released_transition(dataset.transition(transition_id))

    hidden = model.initial_hidden(num_nodes, device)
    if not prefix_ids:
        if prefix_statistics_cache is not None:
            prefix_statistics_cache.clear()
            prefix_statistics_cache[0] = (
                (), rates.clone(), property_history.clone(),
                property_reference_history.clone(),
                historical_change_counts.clone(), history_steps,
            )
        return (
            hidden,
            rates,
            property_history,
            property_reference_history,
            historical_change_counts,
            history_steps,
        )

    was_model_training = model.training
    was_encoder_training = action_encoder.training
    was_controller_training = controllers.training
    model.eval(); action_encoder.eval(); controllers.eval()
    try:
        for transition_id in prefix_ids[replay_from:]:
            transition_cpu = dataset.transition(transition_id)
            if static_candidate_cache is not None:
                transition_cpu = static_candidate_cache.attach(transition_cpu)
            property_history.observe(
                transition_cpu["property_node_ids_t"],
                transition_cpu["property_observed_t"],
                transition_id=int(transition_cpu["transition_id"]),
            )
            property_reference_history.observe(
                transition_cpu["property_node_ids_t"],
                transition_cpu["property_observed_t"],
            )
            transition = t2_transition_to_device(
                _with_causal_semantic_history_features(
                    transition_cpu,
                    property_history,
                    enabled=semantic_history_feature,
                ),
                device,
            )
            z_raw, action, _z_t, _details = _soft_action(
                model=model,
                action_encoder=action_encoder,
                controllers=controllers,
                transition=transition,
                dataset_name=dataset.dataset_name,
                hidden=hidden,
                rates=rates,
                enabled=action_enabled,
                action_mode=action_mode,
                action_mode_overrides=action_mode_overrides,
                action_topk=action_topk,
                count_support_operations=count_support_operations,
                count_decision_operations=count_decision_operations,
                use_property_magnitude=use_property_magnitude,
                property_history=property_history,
                property_history_action_weight=property_history_action_weight,
                vectorized_controller_forward=vectorized_controller_forward,
            )
            hidden = model(
                transition["x_t"],
                transition["edge_index_t"],
                hidden,
                action=action,
                edge_weight_t=_edge_weight_for_dataset(
                    transition, dataset.dataset_name
                ),
                precomputed_z_t_raw=z_raw,
                decode_observables=False,
                node_activity_observed_context=_node_activity_observed_context(
                    transition,
                    enabled=node_activity_current_state_context,
                    include_source_visibility=(
                        node_activity_source_visibility_context
                    ),
                    include_full_state=node_activity_full_observed_context,
                    include_history=node_activity_history_context,
                ),
                node_activity_action_context=_node_activity_action_context(
                    _details,
                    num_nodes=int(transition["x_t"].shape[0]),
                    device=transition["x_t"].device,
                    dtype=transition["x_t"].dtype,
                    enabled=node_activity_action_context and action_enabled,
                ),
            )["hidden_next"].detach()




            changed_nodes = torch.zeros(num_nodes, dtype=torch.bool)
            for operation in NODE_OPERATIONS:
                _candidates, labels, known, full = _operation_targets(
                    transition_cpu,
                    operation,
                    semantic_threshold=semantic_threshold,
                    semantic_topk=semantic_topk,
                    semantic_history_cache=getattr(dataset, "semantic_history_cache", None),
                )
                rates.observe(operation, labels, known)
                changed_nodes |= full.detach().cpu().to(torch.bool)
            historical_change_counts.add_(changed_nodes.to(torch.float32))
            history_steps += 1
        if prefix_statistics_cache is not None:




            prefix_statistics_cache.clear()
            prefix_statistics_cache[len(prefix_ids)] = (
                tuple(prefix_ids),
                rates.clone(),
                property_history.clone(),
                property_reference_history.clone(),
                historical_change_counts.clone(),
                history_steps,
            )
    finally:
        model.train(was_model_training)
        action_encoder.train(was_encoder_training)
        controllers.train(was_controller_training)
    return (
        hidden,
        rates,
        property_history,
        property_reference_history,
        historical_change_counts,
        history_steps,
    )


def _action_effect_parameters(
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    *,
    include_state_model: bool = False,
    include_action_plan: bool = False,
    include_operation_count: bool = False,
    include_observable_decoders: bool = False,
    decoder_consistency_operations: frozenset[GraphOperation] = frozenset(),
) -> tuple[torch.nn.Parameter, ...]:

    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in action_encoder.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            parameters.append(parameter)
            seen.add(id(parameter))
    for name, parameter in model.state_model.named_parameters():




        if (
            (include_state_model or name.startswith("action_"))
            and parameter.requires_grad
            and id(parameter) not in seen
        ):
            parameters.append(parameter)
            seen.add(id(parameter))
    if include_action_plan:




        for module in (model.latent_predictor, model.action_plan_decoder):
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
    if include_operation_count:



        for module in (model.latent_predictor, model.node_operation_count_decoder):
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
    if include_observable_decoders:




        for module in (
            model.latent_predictor,
            model.node_activity_decoder,
            model.node_change_decoder,
            model.node_property_decoder,
            model.property_transition_gate_decoder,
            model.node_operation_count_decoder,
        ):
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
    if decoder_consistency_operations:



        modules: list[torch.nn.Module | None] = []
        if {
            GraphOperation.ADD_NODE,
            GraphOperation.REMOVE_NODE,
        } & decoder_consistency_operations:
            modules.append(model.node_activity_decoder)
        if GraphOperation.MODIFY_NODE_PROPERTY in decoder_consistency_operations:
            modules.append(model.node_change_decoder)
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
    return tuple(parameters)


def _action_plan_consistency_loss(
    outputs: dict[str, torch.Tensor],
    action_details: dict[str, Any],
    *,
    positive_weight_cap: float,
    target_mode: str,
) -> torch.Tensor:

    if "action_plan_logits" not in outputs:
        raise RuntimeError(
            "Action-plan consistency requires GraphWorldModel.action_plan_decoder."
        )
    target_key = (
        "target_probability"
        if target_mode == "action_probability"
        else "controller_probability"
    )
    proposal_target = torch.stack(
        [
            action_details["proposals"][operation.name][target_key]
            for operation in NODE_OPERATIONS
        ],
        dim=-1,
    ).detach().to(dtype=outputs["action_plan_logits"].dtype)
    logits = outputs["action_plan_logits"]
    if logits.shape != proposal_target.shape:
        raise RuntimeError(
            "Action-plan logits and Controller proposal must both have shape "
            f"[num_nodes, {len(NODE_OPERATIONS)}], got {tuple(logits.shape)} "
            f"and {tuple(proposal_target.shape)}."
        )
    if positive_weight_cap > 0.0:
        positive = proposal_target.sum(dim=0)
        negative = (1.0 - proposal_target).sum(dim=0)
        pos_weight = (negative / positive.clamp_min(1e-6)).clamp(
            min=1.0, max=float(positive_weight_cap)
        )
        return F.binary_cross_entropy_with_logits(
            logits, proposal_target, pos_weight=pos_weight
        )
    return F.binary_cross_entropy_with_logits(logits, proposal_target)


def _action_decoder_consistency_loss(
    outputs: dict[str, torch.Tensor],
    action_details: dict[str, Any],
    *,
    positive_weight_cap: float,
    target_mode: str,
    operations: frozenset[GraphOperation],
) -> torch.Tensor:

    decoder_logits = {
        GraphOperation.ADD_NODE: outputs["node_activation_logits"],
        GraphOperation.REMOVE_NODE: outputs["node_deactivation_logits"],
        GraphOperation.MODIFY_NODE_PROPERTY: outputs["node_change_logits"],
    }
    losses: list[torch.Tensor] = []
    target_key = (
        "target_probability"
        if target_mode == "action_probability"
        else "controller_probability"
    )
    for operation in NODE_OPERATIONS:
        if operation not in operations:
            continue
        proposal = action_details["proposals"][operation.name]
        candidates = proposal["candidates"]
        if candidates.numel() == 0:
            continue
        logits = decoder_logits[operation].index_select(0, candidates)
        target = proposal[target_key].index_select(0, candidates)
        target = target.detach().to(dtype=logits.dtype)
        if positive_weight_cap > 0.0:
            positive = target.sum()
            negative = (1.0 - target).sum()
            pos_weight = (negative / positive.clamp_min(1e-6)).clamp(
                min=1.0, max=float(positive_weight_cap)
            )
            losses.append(
                F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            )
        else:
            losses.append(F.binary_cross_entropy_with_logits(logits, target))
    if not losses:
        return outputs["latent_next"].new_zeros(())
    return torch.stack(losses).mean()


def _observable_transition_loss(
    terms: dict[str, torch.Tensor], *, node_operation_count_weight: float = 0.0
) -> torch.Tensor:

    return (
        terms["activity"]
        + terms["semantic_change"]
        + terms["property_transition_gate"]
        + terms["semantic_value"]
        + float(node_operation_count_weight) * terms["operation_count"]
    )


def _natural_addition_weighted_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
    natural_addition_mask: torch.Tensor,
    *,
    pos_weight: float,
    natural_positive_weight: float,
    candidate_indices: torch.Tensor | None = None,
) -> torch.Tensor:

    if natural_positive_weight < 1.0:
        raise ValueError("natural_positive_weight must be at least one")
    if candidate_indices is None:
        candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return logits.new_zeros(())
    selected_logits = logits.index_select(0, candidate_indices)
    selected_target = target.index_select(0, candidate_indices).to(
        dtype=logits.dtype
    )
    loss = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_target,
        pos_weight=selected_logits.new_tensor(float(pos_weight)),
        reduction="none",
    )
    if natural_positive_weight == 1.0:
        return loss.mean()
    natural_positive = (
        natural_addition_mask.to(device=logits.device, dtype=torch.bool).index_select(
            0, candidate_indices
        )
        & selected_target.gt(0.5)
    )
    weights = torch.ones_like(loss)
    weights = torch.where(
        natural_positive,
        weights.new_full((), float(natural_positive_weight)),
        weights,
    )
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _dynamic_balanced_hard_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    hard_fraction: float,
    focal_gamma: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:

    if not 0.0 < float(hard_fraction) <= 1.0:
        raise ValueError("hard_fraction must lie in (0, 1]")
    if float(focal_gamma) < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    logits = logits.reshape(-1)
    target = target.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if logits.numel() != target.numel():
        raise ValueError("dynamic hard-loss logits and target must align")
    if logits.numel() == 0:
        return logits.new_zeros(())
    entry_weight = (
        torch.ones_like(logits)
        if sample_weight is None
        else sample_weight.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    )
    if entry_weight.numel() != logits.numel():
        raise ValueError("dynamic hard-loss sample weights must align")






    signed_margin = torch.where(target.gt(0.5), logits, -logits).clamp(
        min=-30.0, max=30.0
    )
    per_entry = F.softplus(-signed_margin)
    if focal_gamma > 0.0:
        per_entry = per_entry * torch.sigmoid(-signed_margin).pow(
            float(focal_gamma)
        )

    class_losses: list[torch.Tensor] = []
    for positive_class in (False, True):
        mask = target.gt(0.5) if positive_class else target.le(0.5)
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        count = max(
            1,
            int(torch.ceil(logits.new_tensor(
                float(hard_fraction) * int(indices.numel())
            )).item()),
        )
        class_entry = per_entry.index_select(0, indices)


        hardest = class_entry.detach().topk(min(count, int(indices.numel()))).indices
        selected_indices = indices.index_select(0, hardest)
        selected_loss = per_entry.index_select(0, selected_indices)
        selected_weight = entry_weight.index_select(0, selected_indices)
        class_losses.append(
            (selected_loss * selected_weight).sum()
            / selected_weight.sum().clamp_min(1e-6)
        )
    if not class_losses:
        return logits.new_zeros(())
    return torch.stack(class_losses).mean()


def _causal_trajectory_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    trajectory: torch.Tensor | None,
    *,
    channel: int,
    direction: float,
    pos_weight: float,
    coefficient: float,
) -> torch.Tensor | None:
    if coefficient <= 0.0 or trajectory is None or trajectory.ndim != 2:
        return None
    if trajectory.shape[1] <= int(channel) or candidates.numel() == 0:
        return None
    logit_candidates = candidates.to(device=logits.device, dtype=torch.long)
    trajectory_candidates = candidates.to(
        device=trajectory.device, dtype=torch.long
    )
    selected_logits = logits.index_select(0, logit_candidates)
    selected_target = target.index_select(0, logit_candidates).to(
        dtype=selected_logits.dtype
    )
    trend = trajectory[:, int(channel)].index_select(0, trajectory_candidates).to(
        device=selected_logits.device, dtype=selected_logits.dtype
    )
    trend = (trend - trend.mean()) / trend.std(unbiased=False).clamp_min(1e-4)
    prior = torch.sigmoid(float(direction) * trend)
    disagreement = torch.where(
        selected_target.gt(0.5), 1.0 - prior, prior
    )
    weights = 1.0 + float(coefficient) * disagreement.detach()
    per_entry = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_target,
        pos_weight=selected_logits.new_tensor(float(pos_weight)),
        reduction="none",
    )
    return (per_entry * weights).sum() / weights.sum().clamp_min(1e-6)


def _semantic_hard_pair_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:

    if float(margin) <= 0.0:
        return logits.new_zeros(())
    logits = logits.reshape(-1)
    target = target.reshape(-1).to(device=logits.device, dtype=torch.bool)
    positives = logits[target]
    negatives = logits[~target]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.new_zeros(())
    width = min(64, int(positives.numel()), int(negatives.numel()))
    hard_positive = positives.topk(width, largest=False).values
    hard_negative = negatives.topk(width, largest=True).values
    return F.softplus(
        hard_negative.detach().mean()
        - hard_positive.mean()
        + float(margin)
    )


def _world_loss(
    *,
    model: GraphWorldModel,
    outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    transition_cpu: dict[str, Any],
    dataset_name: str,
    statistics: dict[str, float | int],
    semantic_threshold: float,
    semantic_topk: int,
    semantic_history_cache: SemanticHistoryCache | None,
    lambda_activity: float,
    lambda_semantic_change: float,
    lambda_semantic_distance: float,
    lambda_semantic_value: float,
    changed_semantic_value_weight: float,
    lambda_current_semantic: float,
    lambda_latent: float,
    lambda_latent_cosine: float,
    lambda_latent_variance: float,
    changed_node_latent_weight: float,
    semantic_property_mode: str,
    semantic_pos_weight_scale: float,
    lambda_property_transition_gate: float,
    lambda_node_operation_count: float,
    lambda_activation_ranking: float,
    lambda_deactivation: float,
    deactivation_focal_gamma: float,
    lambda_deactivation_ranking: float,
    natural_addition_loss_weight: float,
    dynamic_change_hard_fraction: float,
    dynamic_change_focal_gamma: float,
    semantic_hard_pair_weight: float,
    semantic_hard_pair_margin: float,
    activity_trajectory_supervision_bias: float,
    removal_history_supervision_bias: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active_t = transition["node_active_t"]
    active_next = transition["node_active_next"]
    eligible = transition.get("node_edit_eligible_mask")
    if eligible is None:
        eligible = torch.ones_like(active_t, dtype=torch.bool)
    else:
        eligible = eligible.to(torch.bool)
    activation_candidates = (~active_t) & eligible
    deactivation_candidates = active_t & eligible
    activation_candidate_indices = _candidate_nodes(
        transition, GraphOperation.ADD_NODE
    )
    deactivation_candidate_indices = _candidate_nodes(
        transition, GraphOperation.REMOVE_NODE
    )
    if dynamic_change_hard_fraction > 0.0:
        activation_logits = outputs["node_activation_logits"].index_select(
            0, activation_candidate_indices
        )
        activation_target = active_next.index_select(
            0, activation_candidate_indices
        )
        activation_weight = torch.ones_like(activation_logits)
        if natural_addition_loss_weight > 1.0:
            natural_positive = transition["node_added_natural"].index_select(
                0, activation_candidate_indices
            ) & activation_target.to(torch.bool)
            activation_weight = torch.where(
                natural_positive,
                activation_weight.new_full(
                    (), float(natural_addition_loss_weight)
                ),
                activation_weight,
            )
        activation_loss = _dynamic_balanced_hard_focal_loss(
            activation_logits,
            activation_target,
            hard_fraction=dynamic_change_hard_fraction,
            focal_gamma=dynamic_change_focal_gamma,
            sample_weight=activation_weight,
        )
    else:
        activation_loss = _natural_addition_weighted_loss(
            outputs["node_activation_logits"],
            active_next,
            activation_candidates,
            transition["node_added_natural"],
            pos_weight=float(statistics["activation_pos_weight"]),
            natural_positive_weight=float(natural_addition_loss_weight),
            candidate_indices=activation_candidate_indices,
        )
    trajectory = transition.get("activity_trajectory_features")
    trajectory_activation_loss = _causal_trajectory_boundary_loss(
        outputs["node_activation_logits"],
        active_next,
        activation_candidate_indices,
        trajectory,
        channel=1,
        direction=1.0,
        pos_weight=float(statistics["activation_pos_weight"]),
        coefficient=float(activity_trajectory_supervision_bias),
    )
    if trajectory_activation_loss is not None:
        activation_loss = trajectory_activation_loss
    activation_ranking_loss = (
        _candidate_pairwise_ranking_loss(
            outputs["node_activation_logits"],
            active_next,
            activation_candidates,
        )
        if lambda_activation_ranking > 0.0
        else activation_loss.new_zeros(())
    )
    if dynamic_change_hard_fraction > 0.0:
        deactivation_loss = _dynamic_balanced_hard_focal_loss(
            outputs["node_deactivation_logits"].index_select(
                0, deactivation_candidate_indices
            ),
            (~active_next).index_select(0, deactivation_candidate_indices),
            hard_fraction=dynamic_change_hard_fraction,
            focal_gamma=dynamic_change_focal_gamma,
        )
    else:
        deactivation_loss = _candidate_focal_binary_loss(
            outputs["node_deactivation_logits"],
            ~active_next,
            deactivation_candidates,
            pos_weight=float(statistics["deactivation_pos_weight"]),
            gamma=float(deactivation_focal_gamma),
            candidate_indices=deactivation_candidate_indices,
        )
    trajectory_deactivation_loss = _causal_trajectory_boundary_loss(
        outputs["node_deactivation_logits"],
        (~active_next),
        deactivation_candidate_indices,
        trajectory,
        channel=3,
        direction=-1.0,
        pos_weight=float(statistics["deactivation_pos_weight"]),
        coefficient=float(activity_trajectory_supervision_bias),
    )
    if float(removal_history_supervision_bias) > 0.0:
        trajectory_deactivation_loss = _causal_trajectory_boundary_loss(
            outputs["node_deactivation_logits"],
            (~active_next),
            deactivation_candidate_indices,
            transition.get("controller_history_features"),
            channel=2,
            direction=-1.0,
            pos_weight=float(statistics["deactivation_pos_weight"]),
            coefficient=float(removal_history_supervision_bias),
        )
    if trajectory_deactivation_loss is not None:
        deactivation_loss = trajectory_deactivation_loss
    deactivation_ranking_loss = (
        _candidate_pairwise_ranking_loss(
            outputs["node_deactivation_logits"],
            ~active_next,
            deactivation_candidates,
        )
        if lambda_deactivation_ranking > 0.0
        else deactivation_loss.new_zeros(())
    )




    activity_loss = 0.5 * (
        activation_loss + float(lambda_deactivation) * deactivation_loss
    )
    comparable = transition["semantic_comparable_mask"]
    comparable_indices = transition.get("semantic_comparable_indices")
    if comparable_indices is None:
        comparable_indices = comparable.to(torch.bool).nonzero(
            as_tuple=False
        ).flatten()
    semantic_loss = activity_loss.new_zeros(())
    semantic_distance_loss = activity_loss.new_zeros(())
    semantic_hard_pair_loss = activity_loss.new_zeros(())
    property_transition_gate_loss = activity_loss.new_zeros(())
    semantic_count = activity_loss.new_zeros(())
    semantic_changed_node_mask = torch.zeros_like(active_t, dtype=torch.bool)
    changed_semantic_rows = torch.zeros(
        transition["semantic_next"].shape[0],
        device=active_t.device,
        dtype=torch.bool,
    )
    if comparable_indices.numel():
        distance = transition.get("semantic_change_distance_cached")
        if distance is None:
            labels, distance = semantic_change_labels(
                transition["semantic_current"].index_select(
                    0, comparable_indices
                ),
                transition["semantic_next"].index_select(0, comparable_indices),
                threshold=semantic_threshold,
                topk=semantic_topk,
            )
        else:
            labels = distance.gt(float(semantic_threshold))
        comparable_node_ids = transition["semantic_node_ids"].index_select(
            0, comparable_indices
        )
        logits = outputs["node_change_logits"].index_select(
            0, comparable_node_ids
        )
        if dynamic_change_hard_fraction > 0.0:
            semantic_loss = _dynamic_balanced_hard_focal_loss(
                logits,
                labels,
                hard_fraction=dynamic_change_hard_fraction,
                focal_gamma=dynamic_change_focal_gamma,
            )
        else:
            semantic_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels.to(dtype=logits.dtype),
                pos_weight=logits.new_tensor(
                    float(statistics["semantic_pos_weight"])
                    * float(semantic_pos_weight_scale)
                ),
            )





        if lambda_semantic_distance > 0.0:
            semantic_distance_loss = F.smooth_l1_loss(
                torch.sigmoid(logits), distance.to(dtype=logits.dtype)
            )
        semantic_hard_pair_loss = _semantic_hard_pair_margin_loss(
            logits,
            labels,
            margin=semantic_hard_pair_margin,
        )
        if (
            lambda_property_transition_gate > 0.0
            and "node_property_transition_gate_logits" in outputs
        ):
            gate_logits = outputs[
                "node_property_transition_gate_logits"
            ].index_select(0, comparable_node_ids)
            property_transition_gate_loss = float(
                lambda_property_transition_gate
            ) * F.binary_cross_entropy_with_logits(
                gate_logits,
                labels.to(dtype=gate_logits.dtype),
                pos_weight=gate_logits.new_tensor(
                    float(statistics["semantic_pos_weight"])
                    * float(semantic_pos_weight_scale)
                ),
            )
        semantic_count = labels.to(dtype=activity_loss.dtype).sum()
        if changed_semantic_value_weight > 1.0 or changed_node_latent_weight > 1.0:
            changed_positions = labels.to(torch.bool).nonzero(
                as_tuple=False
            ).flatten()
            if changed_positions.numel():
                changed_rows = comparable_indices.index_select(
                    0, changed_positions
                )
                changed_semantic_rows.index_fill_(0, changed_rows, True)
                if changed_node_latent_weight > 1.0:
                    changed_property_ids = comparable_node_ids.index_select(
                        0, changed_positions
                    )
                    semantic_changed_node_mask.index_fill_(
                        0, changed_property_ids, True
                    )
    operation_count_loss = activity_loss.new_zeros(())
    if (
        lambda_node_operation_count > 0.0
        and "node_operation_count_log1p" in outputs
    ):
        target_counts = torch.stack(
            [
                active_next[activation_candidates].to(dtype=activity_loss.dtype).sum(),
                (~active_next[deactivation_candidates]).to(dtype=activity_loss.dtype).sum(),
                semantic_count,
            ]
        )
        predicted_counts = outputs["node_operation_count_log1p"].reshape(-1)
        if predicted_counts.numel() != target_counts.numel():
            raise ValueError(
                "node_operation_count_log1p must contain one value for each "
                "Task-1 node operation."
            )
        operation_count_loss = F.smooth_l1_loss(
            predicted_counts,
            torch.log1p(target_counts),
        )
    semantic_prediction, _, decoded = _semantic_prediction(
        outputs, transition, property_mode=semantic_property_mode
    )
    property_reference = outputs.get(
        "_semantic_property_reference", transition["semantic_current"]
    )
    semantic_value_loss = property_prediction_loss(
        decoded,
        semantic_prediction,
        transition["semantic_next"],
        property_reference,
        property_mode=semantic_property_mode,
    )
    if changed_semantic_value_weight > 1.0 and bool(changed_semantic_rows.any()):





        changed_count = int(changed_semantic_rows.sum().item())
        stable_rows = ~changed_semantic_rows
        changed_loss = property_prediction_loss(
            decoded[changed_semantic_rows],
            semantic_prediction[changed_semantic_rows],
            transition["semantic_next"][changed_semantic_rows],
            property_reference[changed_semantic_rows],
            property_mode=semantic_property_mode,
        )
        if bool(stable_rows.any()):
            stable_count = int(stable_rows.sum().item())
            stable_loss = property_prediction_loss(
                decoded[stable_rows],
                semantic_prediction[stable_rows],
                transition["semantic_next"][stable_rows],
                property_reference[stable_rows],
                property_mode=semantic_property_mode,
            )
            semantic_value_loss = (
                stable_loss * float(stable_count)
                + changed_loss * float(changed_semantic_value_weight * changed_count)
            ) / float(stable_count + changed_semantic_value_weight * changed_count)
        else:
            semantic_value_loss = changed_loss
    current_semantic_loss = (
        _current_semantic_reconstruction_loss(outputs, transition)
        if lambda_current_semantic > 0.0
        else activity_loss.new_zeros(())
    )
    latent_loss = activity_loss.new_zeros(())
    if bool(transition_cpu["latent_train_allowed"]):
        target_z = model.encode_target(
            transition["x_next"],
            transition["edge_index_next"],
            edge_weight_next=_edge_weight_for_dataset(
                transition, dataset_name, suffix="next"
            ),
            topology_cache_key=int(transition_cpu["transition_id"]) + 1,
        )
        latent_node_weight = None
        if changed_node_latent_weight > 1.0:
            changed_nodes = (
                transition["node_added"].to(torch.bool)
                | transition["node_removed"].to(torch.bool)
                | semantic_changed_node_mask
            )
            latent_node_weight = torch.ones_like(
                active_t, dtype=target_z.dtype
            )
            latent_node_weight[changed_nodes] = float(changed_node_latent_weight)
        latent_loss = latent_transition_terms(
            outputs,
            target_z,
            cosine_weight=lambda_latent_cosine,
            variance_weight=lambda_latent_variance,
            node_weight=latent_node_weight,
        )["total"]
    total = (
        lambda_activity * activity_loss
        + lambda_semantic_change * semantic_loss
        + lambda_semantic_distance * semantic_distance_loss
        + float(semantic_hard_pair_weight) * semantic_hard_pair_loss
        + property_transition_gate_loss
        + lambda_semantic_value * semantic_value_loss
        + lambda_current_semantic * current_semantic_loss
        + lambda_node_operation_count * operation_count_loss
        + lambda_activation_ranking * activation_ranking_loss
        + lambda_deactivation_ranking * deactivation_ranking_loss
        + lambda_latent * latent_loss
    )
    return total, {
        "activity": activity_loss,
        "activation": activation_loss,
        "activation_ranking": activation_ranking_loss,
        "deactivation": deactivation_loss,
        "semantic_change": semantic_loss,
        "semantic_distance": semantic_distance_loss,
        "semantic_hard_pair": semantic_hard_pair_loss,
        "property_transition_gate": property_transition_gate_loss,
        "semantic_value": semantic_value_loss,
        "operation_count": operation_count_loss,
        "deactivation_ranking": deactivation_ranking_loss,
        "current_semantic": current_semantic_loss,
        "latent": latent_loss,
    }


def _rollout_reward(
    *,
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    z_raw: torch.Tensor,
    z_t: torch.Tensor,
    hidden: torch.Tensor,
    transition: dict[str, Any],
    sequence: Any,
    operation: GraphOperation,
    candidates: torch.Tensor,
    full_target: torch.Tensor,
    dataset_name: str,
    node_weights: torch.Tensor,
    node_activity_current_state_context: bool = False,
    node_activity_source_visibility_context: bool = False,
    node_activity_full_observed_context: bool = False,
    node_activity_history_context: bool = False,
    node_activity_action_context: bool = False,
    composite_task_reward: bool = False,
    candidates_by_operation: dict[GraphOperation, torch.Tensor] | None = None,
    targets_by_operation: dict[GraphOperation, torch.Tensor] | None = None,
    semantic_property_mode: str = "logit_residual_mixture",
    semantic_topk: int = 10,
    fine_timing: dict[str, float] | None = None,
) -> torch.Tensor:
    rollout_started = _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
    encoded = action_encoder(z_t, sequence.action)["nodewise"]
    outputs = model(
        transition["x_t"],
        transition["edge_index_t"],
        hidden,
        action=encoded,
        edge_weight_t=_edge_weight_for_dataset(transition, dataset_name),
        commit_history=False,
        precomputed_z_t_raw=z_raw,
        decode_node_features=False,
        decode_node_property=composite_task_reward,
        decode_current_property=False,
        decode_property_transition_gate=False,
        node_activity_observed_context=_node_activity_observed_context(
            transition,
            enabled=node_activity_current_state_context,
            include_source_visibility=node_activity_source_visibility_context,
            include_full_state=node_activity_full_observed_context,
            include_history=node_activity_history_context,
        ),
        node_activity_action_context=_node_activity_sequence_context(
            sequence.action,
            num_nodes=int(transition["x_t"].shape[0]),
            device=transition["x_t"].device,
            dtype=transition["x_t"].dtype,
            enabled=node_activity_action_context,
        ),
    )
    _fine_timing_add(fine_timing, "grpo_rollout", rollout_started, z_t.device)
    reward_started = _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
    if composite_task_reward:
        if candidates_by_operation is None or targets_by_operation is None:
            raise ValueError(
                "Composite T1 reward requires candidates and targets for all operations."
            )
        component_rewards: list[torch.Tensor] = []
        output_keys = {
            GraphOperation.ADD_NODE: "node_activation_logits",
            GraphOperation.REMOVE_NODE: "node_deactivation_logits",
            GraphOperation.MODIFY_NODE_PROPERTY: "node_change_logits",
        }
        for component_operation in NODE_OPERATIONS:
            component_candidates = candidates_by_operation[component_operation]
            component_target = targets_by_operation[component_operation]
            component_score = torch.sigmoid(outputs[output_keys[component_operation]])
            target_count = int(component_target.to(torch.bool).sum().item())
            count = min(target_count, int(component_candidates.numel()))
            if count:
                selected = component_candidates.index_select(
                    0,
                    component_score.index_select(0, component_candidates)
                    .topk(count)
                    .indices,
                )
            else:
                selected = component_candidates.new_empty((0,), dtype=torch.long)
            component_rewards.append(
                node_change_localization_reward(
                    predicted_nodes=selected,
                    target_changed=component_target,
                    node_weights=node_weights,
                )["reward"]
            )

        semantic_prediction, _, _ = _semantic_prediction(
            outputs,
            transition,
            property_mode=semantic_property_mode,
        )
        comparable_indices = transition.get("semantic_comparable_indices")
        if comparable_indices is None:
            comparable_indices = transition["semantic_comparable_mask"].to(
                torch.bool
            ).nonzero(as_tuple=False).flatten()
        semantic_ids = transition["semantic_node_ids"].index_select(
            0, comparable_indices
        ).long()
        changed_rows = targets_by_operation[
            GraphOperation.MODIFY_NODE_PROPERTY
        ].index_select(0, semantic_ids)
        current_rows = transition["semantic_current"].index_select(
            0, comparable_indices
        )
        target_rows = transition["semantic_next"].index_select(
            0, comparable_indices
        )
        magnitude_reward = representation_change_magnitude_reward(
            predicted_delta=(
                semantic_prediction.index_select(0, comparable_indices)
                - current_rows
            ),
            target_delta=target_rows - current_rows,
            item_weights=node_weights.index_select(0, semantic_ids),
            target_changed=changed_rows,
        )["reward"]
        component_rewards.append(magnitude_reward)
        reward = torch.stack(component_rewards).mean()
        _fine_timing_add(fine_timing, "structure_aware_reward", reward_started, z_t.device)
        return reward

    if operation == GraphOperation.ADD_NODE:
        score = torch.sigmoid(outputs["node_activation_logits"])
    elif operation == GraphOperation.REMOVE_NODE:
        score = torch.sigmoid(outputs["node_deactivation_logits"])
    else:
        score = torch.sigmoid(outputs["node_change_logits"])
    if candidates.numel() == 0:
        _fine_timing_add(fine_timing, "structure_aware_reward", reward_started, z_t.device)
        return score.new_zeros(())
    count = min(max(len(sequence.action), 1), int(candidates.numel()))
    selected = candidates.index_select(0, score.index_select(0, candidates).topk(count).indices)
    reward = node_change_localization_reward(
        predicted_nodes=selected,
        target_changed=full_target,
        node_weights=node_weights,
    )["reward"]
    _fine_timing_add(fine_timing, "structure_aware_reward", reward_started, z_t.device)
    return reward


def _batched_rollout_rewards(
    *,
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    z_raw: torch.Tensor,
    z_t: torch.Tensor,
    hidden: torch.Tensor,
    transition: dict[str, Any],
    sequences: list[Any],
    operation: GraphOperation,
    candidates: torch.Tensor,
    full_target: torch.Tensor,
    dataset_name: str,
    node_weights: torch.Tensor,
    node_activity_current_state_context: bool = False,
    node_activity_source_visibility_context: bool = False,
    node_activity_full_observed_context: bool = False,
    node_activity_history_context: bool = False,
    node_activity_action_context: bool = False,
    fine_timing: dict[str, float] | None = None,
) -> list[float]:

    if not sequences:
        return []
    rollout_started = _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
    encoded_actions = torch.stack(
        [action_encoder(z_t, sequence.action)["nodewise"] for sequence in sequences]
    )
    observed_context = _node_activity_observed_context(
        transition,
        enabled=node_activity_current_state_context,
        include_source_visibility=node_activity_source_visibility_context,
        include_full_state=node_activity_full_observed_context,
        include_history=node_activity_history_context,
    )
    action_contexts = (
        torch.stack(
            [
                _node_activity_sequence_context(
                    sequence.action,
                    num_nodes=int(transition["x_t"].shape[0]),
                    device=transition["x_t"].device,
                    dtype=transition["x_t"].dtype,
                    enabled=True,
                )
                for sequence in sequences
            ]
        )
        if node_activity_action_context
        else None
    )
    edge_weight_t = _edge_weight_for_dataset(transition, dataset_name)

    def decode_logits(
        encoded_action: torch.Tensor,
        action_context: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs = model(
            transition["x_t"],
            transition["edge_index_t"],
            hidden,
            action=encoded_action,
            edge_weight_t=edge_weight_t,
            commit_history=False,
            precomputed_z_t_raw=z_raw,
            decode_node_features=False,
            decode_node_property=False,
            decode_current_property=False,
            decode_property_transition_gate=False,
            node_activity_observed_context=observed_context,
            node_activity_action_context=action_context,
        )
        if operation == GraphOperation.ADD_NODE:
            return outputs["node_activation_logits"]
        if operation == GraphOperation.REMOVE_NODE:
            return outputs["node_deactivation_logits"]
        return outputs["node_change_logits"]

    if action_contexts is None:
        logits = vmap(
            lambda encoded_action: decode_logits(encoded_action, None),
            randomness="different",
        )(encoded_actions)
    else:
        logits = vmap(decode_logits, randomness="different")(
            encoded_actions, action_contexts
        )
    _fine_timing_add(fine_timing, "grpo_rollout", rollout_started, z_t.device)
    reward_started = _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
    scores = torch.sigmoid(logits)
    rewards: list[float] = []
    for index, sequence in enumerate(sequences):
        score = scores[index]
        if candidates.numel() == 0:
            reward = score.new_zeros(())
        else:
            count = min(max(len(sequence.action), 1), int(candidates.numel()))
            selected = candidates.index_select(
                0, score.index_select(0, candidates).topk(count).indices
            )
            reward = node_change_localization_reward(
                predicted_nodes=selected,
                target_changed=full_target,
                node_weights=node_weights,
            )["reward"]
        rewards.append(float(reward.item()))
    _fine_timing_add(fine_timing, "structure_aware_reward", reward_started, z_t.device)
    return rewards


def _controller_loss(
    *,
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    controllers: torch.nn.ModuleDict,
    group_sampler: DynamicGroupSampler,
    historical_change_counts: torch.Tensor,
    history_steps: int,
    z_raw: torch.Tensor,
    z_t: torch.Tensor,
    hidden: torch.Tensor,
    transition: dict[str, Any],
    dataset_name: str,
    semantic_threshold: float,
    semantic_topk: int,
    semantic_history_cache: SemanticHistoryCache | None,
    statistics: dict[str, float | int],
    rollout_budget: int,
    max_actions: int,
    supervised_weight: float,
    grpo_weight: float,
    grpo_clip: float,
    grpo_kl: float,
    entropy_weight: float,
    do_grpo: bool,
    use_property_magnitude: bool,
    property_magnitude_weight: float,
    node_count_weight: float,
    activation_ranking_weight: float,
    deactivation_ranking_weight: float,
    count_supervision_operations: frozenset[GraphOperation],
    semantic_pos_weight_scale: float,
    node_activity_current_state_context: bool,
    node_activity_source_visibility_context: bool,
    node_activity_full_observed_context: bool,
    node_activity_history_context: bool,
    node_activity_action_context: bool,
    natural_addition_loss_weight: float,
    controller_addition_source_visibility: bool,
    controller_addition_source_visibility_balance: str,
    batch_grpo_rollout_forward: bool = False,
    vectorized_controller_forward: bool = False,
    precomputed_proposals: dict[str, Any] | None = None,
    composite_task_reward: bool = False,
    semantic_property_mode: str = "logit_residual_mixture",
    fine_timing: dict[str, float] | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[GraphOperation, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:

    zero = z_t.new_zeros(())
    supervised_started = (
        _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
    )
    target_info: dict[
        GraphOperation, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    supervised_total = zero
    property_magnitude_total = zero
    node_count_total = zero
    activation_ranking_total = zero
    deactivation_ranking_total = zero
    policy_states: dict[GraphOperation, torch.Tensor] = {}
    candidates_by_op: dict[GraphOperation, torch.Tensor] = {}
    full_by_op: dict[GraphOperation, torch.Tensor] = {}
    vectorized_states: dict[GraphOperation, torch.Tensor] = {}
    vectorized_logits: dict[GraphOperation, torch.Tensor] = {}
    if vectorized_controller_forward:
        vectorized_states, vectorized_logits = _vectorized_controller_node_forwards(
            controllers=controllers,
            transition=transition,
            z_t=z_t.detach(),
            hidden=hidden.detach(),
        )
    for operation in NODE_OPERATIONS:
        candidates, labels, known, full = _operation_targets(
            transition,
            operation,
            semantic_threshold=semantic_threshold,
            semantic_topk=semantic_topk,
            semantic_history_cache=semantic_history_cache,
        )
        target_info[operation] = (labels, known, full)
        candidates_by_op[operation] = candidates
        full_by_op[operation] = full
        known_is_full = operation in {
            GraphOperation.ADD_NODE,
            GraphOperation.REMOVE_NODE,
        }
        known_indices = None
        if not known_is_full:
            known_indices = known.nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0 or (
            known_indices is not None and known_indices.numel() == 0
        ):
            continue
        controller = controllers[_controller_key(operation)]
        proposal = (
            None
            if precomputed_proposals is None
            else precomputed_proposals.get(operation.name)
        )
        policy_state = None if proposal is None else proposal.get("_policy_state")
        full_logits = None if proposal is None else proposal.get("_full_logits")
        if policy_state is None or full_logits is None:
            if vectorized_controller_forward:
                policy_state = vectorized_states[operation]
                full_logits = vectorized_logits[operation]
            else:
                policy_state = controller.node_state(
                    z_t.detach(),
                    hidden.detach(),
                    node_aux=_controller_node_aux(transition, controller),
                )
                full_logits = controller.node_target_head(policy_state).squeeze(-1)
        policy_states[operation] = policy_state
        logits = full_logits.index_select(0, candidates)
        selected_logits = (
            logits
            if known_indices is None
            else logits.index_select(0, known_indices)
        )
        selected_labels = (
            labels
            if known_indices is None
            else labels.index_select(0, known_indices)
        ).to(dtype=selected_logits.dtype)
        if operation == GraphOperation.ADD_NODE:
            pos_weight = float(statistics["activation_pos_weight"])
        elif operation == GraphOperation.REMOVE_NODE:
            pos_weight = float(statistics["deactivation_pos_weight"])
        else:
            pos_weight = (
                float(statistics["semantic_pos_weight"])
                * float(semantic_pos_weight_scale)
            )
        if (
            operation == GraphOperation.ADD_NODE
            and controller_addition_source_visibility
            and controller_addition_source_visibility_balance == "train_global"
        ):
            source_visible = _addition_source_visibility(transition).index_select(
                0, candidates
            )
            if known_indices is not None:
                source_visible = source_visible.index_select(0, known_indices)




            visible_weight = selected_logits.new_tensor(
                float(statistics["activation_visible_pos_weight"])
            )
            masked_weight = selected_logits.new_tensor(
                float(statistics["activation_masked_pos_weight"])
            )
            per_positive_weight = torch.where(
                source_visible, visible_weight, masked_weight
            )
            per_entry = F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_labels,
                reduction="none",
            )
            supervised_loss = torch.where(
                selected_labels.gt(0.5), per_positive_weight, torch.ones_like(per_entry)
            ) * per_entry
        elif (
            operation == GraphOperation.ADD_NODE
            and controller_addition_source_visibility
        ):
            source_visible = _addition_source_visibility(transition).index_select(
                0, candidates
            )
            if known_indices is not None:
                source_visible = source_visible.index_select(0, known_indices)
            per_entry = F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_labels,
                reduction="none",
            )
            grouped_loss_sum = selected_logits.new_zeros(())
            grouped_count = selected_logits.new_zeros(())




            for group in (source_visible, ~source_visible):
                group_weight_mask = group.to(dtype=selected_logits.dtype)
                group_size = group_weight_mask.sum()
                positive = (selected_labels * group_weight_mask).sum()
                negative = group_size - positive
                group_weight = (
                    (negative / positive.clamp_min(1.0)).clamp_min(1.0)
                )
                group_weight = torch.where(
                    positive.gt(0), group_weight, torch.ones_like(group_weight)
                )
                entry_weight = torch.where(
                    selected_labels.gt(0.5),
                    group_weight,
                    torch.ones_like(per_entry),
                )
                present = group_size.gt(0).to(dtype=selected_logits.dtype)
                grouped_loss_sum = grouped_loss_sum + present * (
                    (per_entry * entry_weight * group_weight_mask).sum()
                    / group_size.clamp_min(1.0)
                )
                grouped_count = grouped_count + present
            supervised_loss = grouped_loss_sum / grouped_count.clamp_min(1.0)
        else:
            supervised_loss = F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_labels,
                pos_weight=selected_logits.new_tensor(pos_weight),
                reduction="none",
            )
        if (
            operation == GraphOperation.ADD_NODE
            and natural_addition_loss_weight > 1.0
            and not controller_addition_source_visibility
        ):
            natural_positive = (
                transition["node_added_natural"].to(
                    device=selected_logits.device, dtype=torch.bool
                ).index_select(0, candidates)
                & selected_labels.to(torch.bool)
            )
            supervised_weights = torch.where(
                natural_positive,
                supervised_loss.new_full((), float(natural_addition_loss_weight)),
                torch.ones_like(supervised_loss),
            )
            supervised_loss = (
                supervised_loss * supervised_weights
            ).sum() / supervised_weights.sum().clamp_min(1.0)
        else:
            supervised_loss = supervised_loss.mean()
        supervised_total = supervised_total + supervised_loss

        if (
            operation == GraphOperation.ADD_NODE
            and activation_ranking_weight > 0.0
        ) or (
            operation == GraphOperation.REMOVE_NODE
            and deactivation_ranking_weight > 0.0
        ):
            candidate_mask = torch.zeros_like(full, dtype=torch.bool)
            candidate_mask.index_fill_(
                0,
                candidates
                if known_indices is None
                else candidates.index_select(0, known_indices),
                True,
            )
            if (
                operation == GraphOperation.ADD_NODE
                and activation_ranking_weight > 0.0
            ):
                activation_ranking_total = activation_ranking_total + (
                    _candidate_pairwise_ranking_loss(
                        full_logits, full, candidate_mask
                    )
                )
            if (
                operation == GraphOperation.REMOVE_NODE
                and deactivation_ranking_weight > 0.0
            ):
                deactivation_ranking_total = deactivation_ranking_total + (
                    _candidate_pairwise_ranking_loss(
                        full_logits, full, candidate_mask
                    )
                )

        if node_count_weight > 0.0 and operation in count_supervision_operations:




            target_log_count = torch.log1p(
                selected_labels.sum()
            )
            predicted_log_count = torch.nan_to_num(
                controller.node_log_count(policy_state),
                nan=0.0,
                posinf=12.0,
                neginf=0.0,
            ).clamp(0.0, 12.0)
            node_count_total = node_count_total + F.smooth_l1_loss(
                predicted_log_count, target_log_count
            )









        if (
            operation == GraphOperation.MODIFY_NODE_PROPERTY
            and use_property_magnitude
            and property_magnitude_weight > 0.0
        ):
            comparable_indices = transition.get("semantic_comparable_indices")
            if comparable_indices is None:
                comparable_indices = transition[
                    "semantic_comparable_mask"
                ].to(torch.bool).nonzero(as_tuple=False).flatten()
            if comparable_indices.numel():
                semantic_ids = transition["semantic_node_ids"].index_select(
                    0, comparable_indices
                ).long()
                cached_distance = transition.get(
                    "semantic_change_distance_cached"
                )
                if cached_distance is None:
                    changed, _ = semantic_change_labels(
                        transition["semantic_current"].index_select(
                            0, comparable_indices
                        ),
                        transition["semantic_next"].index_select(
                            0, comparable_indices
                        ),
                        threshold=semantic_threshold,
                        topk=semantic_topk,
                    )
                else:
                    changed = cached_distance.gt(float(semantic_threshold))
                changed_positions = changed.to(torch.bool).nonzero(
                    as_tuple=False
                ).flatten()
                if changed_positions.numel():
                    target_delta = (
                        transition["semantic_next"].index_select(
                            0, comparable_indices
                        ).index_select(0, changed_positions)
                        - transition["semantic_current"].index_select(
                            0, comparable_indices
                        ).index_select(0, changed_positions)
                    )
                    selected_ids = semantic_ids.index_select(
                        0, changed_positions
                    )
                    magnitude_mu, magnitude_logstd = controller._magnitude_distribution(
                        policy_state, operation
                    )
                    prediction_mu = magnitude_mu.index_select(0, selected_ids)
                    prediction_var = (
                        2.0 * magnitude_logstd.index_select(0, selected_ids)
                    ).exp().clamp_min(1e-8)
                    property_magnitude_total = property_magnitude_total + F.gaussian_nll_loss(
                        prediction_mu,
                        target_delta,
                        prediction_var,
                        full=True,
                        reduction="mean",
                    )

    _fine_timing_add(
        fine_timing, "controller_supervised_update", supervised_started, z_t.device
    )
    grpo_total = zero
    reward_values: list[torch.Tensor] = []
    if do_grpo and rollout_budget >= 2:
        group_started = (
            _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
        )
        allocation = group_sampler.allocate(
            NODE_OPERATIONS, total_budget=rollout_budget, min_per_group=2
        )
        _fine_timing_add(
            fine_timing, "dynamic_group_sampling", group_started, z_t.device
        )
        reward_setup_started = (
            _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
        )
        node_weights = structure_node_weights(
            transition["edge_index_t"],
            num_nodes=int(transition["node_active_t"].numel()),
            historical_change_counts=historical_change_counts,
            history_steps=history_steps,
        )
        _fine_timing_add(
            fine_timing, "structure_aware_reward", reward_setup_started, z_t.device
        )
        was_model_training, was_encoder_training = model.training, action_encoder.training
        with _preserve_world_rng(z_t.device), torch.no_grad():
            model.eval()
            action_encoder.eval()
            rollout_groups: list[
                tuple[
                    GraphOperation,
                    ActionAwareController,
                    torch.Tensor,
                    torch.Tensor,
                    list[Any],
                ]
            ] = []
            rollout_sampling_started = (
                _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
            )
            for operation in NODE_OPERATIONS:
                candidates = candidates_by_op[operation]
                if candidates.numel() == 0:
                    continue
                controller = controllers[_controller_key(operation)]
                state = policy_states.get(operation)
                if state is None:
                    state = controller.node_state(
                        z_t.detach(),
                        hidden.detach(),
                        node_aux=_controller_node_aux(transition, controller),
                    )
                    policy_states[operation] = state
                samples = controller.sample_node_action_sequences_batch(
                    z_t.detach(),
                    hidden.detach(),
                    operation,
                    candidate_nodes=candidates,
                    min_actions=1,
                    max_actions=max_actions,
                    sample_magnitude=(
                        operation == GraphOperation.MODIFY_NODE_PROPERTY
                        and use_property_magnitude
                    ),
                    node_state=state.detach(),
                    batch_size=allocation[operation],
                )
                rollout_groups.append(
                    (operation, controller, state, candidates, samples)
                )
            _fine_timing_add(
                fine_timing, "grpo_rollout", rollout_sampling_started, z_t.device
            )
            for operation, controller, state, candidates, samples in rollout_groups:
                if batch_grpo_rollout_forward and not composite_task_reward:
                    rewards = _batched_rollout_rewards(
                        model=model,
                        action_encoder=action_encoder,
                        z_raw=z_raw.detach(),
                        z_t=z_t.detach(),
                        hidden=hidden.detach(),
                        transition=transition,
                        sequences=samples,
                        operation=operation,
                        candidates=candidates,
                        full_target=full_by_op[operation],
                        dataset_name=dataset_name,
                        node_weights=node_weights,
                        node_activity_current_state_context=(
                            node_activity_current_state_context
                        ),
                        node_activity_source_visibility_context=(
                            node_activity_source_visibility_context
                        ),
                        node_activity_full_observed_context=(
                            node_activity_full_observed_context
                        ),
                        node_activity_history_context=(
                            node_activity_history_context
                        ),
                        node_activity_action_context=(
                            node_activity_action_context
                        ),
                        fine_timing=fine_timing,
                    )
                else:
                    rewards = [
                        float(
                            _rollout_reward(
                                model=model,
                                action_encoder=action_encoder,
                                z_raw=z_raw.detach(),
                                z_t=z_t.detach(),
                                hidden=hidden.detach(),
                                transition=transition,
                                sequence=sample,
                                operation=operation,
                                candidates=candidates,
                                full_target=full_by_op[operation],
                                dataset_name=dataset_name,
                                node_weights=node_weights,
                                node_activity_current_state_context=(
                                    node_activity_current_state_context
                                ),
                                node_activity_source_visibility_context=(
                                    node_activity_source_visibility_context
                                ),
                                node_activity_full_observed_context=(
                                    node_activity_full_observed_context
                                ),
                                node_activity_history_context=(
                                    node_activity_history_context
                                ),
                                node_activity_action_context=(
                                    node_activity_action_context
                                ),
                                composite_task_reward=composite_task_reward,
                                candidates_by_operation=candidates_by_op,
                                targets_by_operation=full_by_op,
                                semantic_property_mode=semantic_property_mode,
                                semantic_topk=semantic_topk,
                                fine_timing=fine_timing,
                            ).item()
                        )
                        for sample in samples
                    ]
                rollouts = [
                    RewardedSequenceRollout(sample=sample, reward=float(reward))
                    for sample, reward in zip(samples, rewards)
                ]
                grpo_loss_started = (
                    _fine_timing_start(z_t.device) if fine_timing is not None else 0.0
                )
                terms = grpo_sequence_loss(
                    controller,
                    rollouts,
                    z_t=z_t.detach(),
                    h_t=hidden.detach(),
                    candidate_nodes={operation: candidates},
                    valid_operations=(operation,),
                    max_actions=max_actions,
                    clip_epsilon=grpo_clip,
                    kl_coefficient=grpo_kl,
                    entropy_coefficient=entropy_weight,
                    node_state=state,
                )
                _fine_timing_add(
                    fine_timing, "grpo_rollout", grpo_loss_started, z_t.device
                )
                grpo_total = grpo_total + terms["loss"]
                reward_values.append(terms["mean_reward"].detach())
        model.train(was_model_training)
        action_encoder.train(was_encoder_training)
    total = (
        supervised_weight * supervised_total
        + property_magnitude_weight * property_magnitude_total
        + node_count_weight * node_count_total
        + activation_ranking_weight * activation_ranking_total
        + deactivation_ranking_weight * deactivation_ranking_total
        + grpo_weight * grpo_total
    )
    return total, {
        "supervised": supervised_total.detach(),
        "property_magnitude": property_magnitude_total.detach(),
        "node_count": node_count_total.detach(),
        "activation_ranking": activation_ranking_total.detach(),
        "deactivation_ranking": deactivation_ranking_total.detach(),
        "grpo": grpo_total.detach(),
        "reward": torch.stack(reward_values).mean() if reward_values else zero,
    }, target_info


def _action_evaluator(
    *,
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    controllers: torch.nn.ModuleDict,
    dataset_name: str,
    semantic_threshold: float,
    semantic_topk: int,
    enabled: bool,
    action_mode: str,
    action_mode_overrides: dict[GraphOperation, str] | None,
    action_topk: int,
    count_support_operations: frozenset[GraphOperation],
    count_decision_operations: frozenset[GraphOperation],
    use_property_magnitude: bool,
    property_history_action_weight: float,
    semantic_history_feature: bool,
    semantic_history_cache: SemanticHistoryCache | None,
    semantic_reference_mode: str,
    semantic_property_mode: str,
    semantic_gate_bias: float,
    semantic_property_change_bias: float,
    node_activity_current_state_context: bool,
    node_activity_source_visibility_context: bool,
    node_activity_full_observed_context: bool,
    node_activity_history_context: bool,
    node_activity_action_context: bool,
    node_operation_count_history_window: int = 0,
    node_operation_count_source: str = "action_history",
    node_activity_score_source: str = "future_latent",
    node_addition_score_source: str = "same",
    node_removal_score_source: str = "same",
    node_removal_history_bias: float = 0.0,
    node_removal_group_calibration_mode: str = "none",
    node_removal_group_calibration_weight: float = 0.0,
    semantic_history_bias: float = 0.0,
    semantic_transition_gate_bias: float = 0.0,
    semantic_volatility_residual_bias: float = 0.0,
    static_candidate_cache: _StaticCandidateIndexCache | None = None,
    validation_preparation_cache: _CausalValidationPreparationCache | None = None,
    vectorized_controller_forward: bool = False,
    decode_node_property: bool = True,
    device: torch.device,
    scoring_split: str | None = None,
):
    if node_activity_score_source not in {"future_latent", "controller"}:
        raise ValueError(
            "node_activity_score_source must be 'future_latent' or 'controller'."
        )
    if node_addition_score_source not in {"same", "future_latent", "controller"}:
        raise ValueError(
            "node_addition_score_source must be 'same', 'future_latent', or "
            "'controller'."
        )
    if node_removal_score_source not in {"same", "future_latent", "controller"}:
        raise ValueError(
            "node_removal_score_source must be 'same', 'future_latent', or "
            "'controller'."
        )
    if node_removal_group_calibration_mode not in {
        "none",
        "zscore",
        "percentile",
        "stratified_percentile",
    }:
        raise ValueError(
            "node_removal_group_calibration_mode must be none, zscore, "
            "percentile, or stratified_percentile."
        )
    if not 0.0 <= float(node_removal_group_calibration_weight) <= 1.0:
        raise ValueError(
            "node_removal_group_calibration_weight must lie in [0, 1]."
        )
    operation_score_source = {
        GraphOperation.ADD_NODE: (
            node_activity_score_source
            if node_addition_score_source == "same"
            else node_addition_score_source
        ),
        GraphOperation.REMOVE_NODE: (
            node_activity_score_source
            if node_removal_score_source == "same"
            else node_removal_score_source
        ),
    }
    if node_operation_count_source not in {
        "action_history",
        "controller",
        "future_latent",
    }:
        raise ValueError(
            "node_operation_count_source must be action_history, controller, "
            "or future_latent."
        )
    rates = _PastOperationRates(device)
    recent_count_history = (
        _RecentOperationCountHistory(node_operation_count_history_window)
        if node_operation_count_history_window > 0
        else None
    )
    property_history: OnlineSemanticChangeHistory | None = None
    property_reference_history: OnlineSemanticPropertyHistory | None = None

    @torch.no_grad()
    def prepare_transition(transition_cpu: dict[str, Any]) -> dict[str, Any]:

        nonlocal property_history, property_reference_history
        if property_history is None:
            property_history = OnlineSemanticChangeHistory(
                num_nodes=int(transition_cpu["node_active_t"].numel()),
                semantic_dim=int(transition_cpu["property_observed_t"].shape[1]),
                topk=semantic_topk,
            )
            property_history.attach_cache(semantic_history_cache)
            property_reference_history = OnlineSemanticPropertyHistory(
                num_nodes=int(transition_cpu["node_active_t"].numel()),
                semantic_dim=int(transition_cpu["property_observed_t"].shape[1]),
            )
        assert property_reference_history is not None
        property_history.observe(
            transition_cpu["property_node_ids_t"],
            transition_cpu["property_observed_t"],
            transition_id=int(transition_cpu["transition_id"]),
        )
        property_reference_history.observe(
            transition_cpu["property_node_ids_t"],
            transition_cpu["property_observed_t"],
        )
        prepared = (
            static_candidate_cache.attach(transition_cpu)
            if static_candidate_cache is not None
            else transition_cpu
        )
        if validation_preparation_cache is not None:
            return validation_preparation_cache.prepare(
                prepared,
                property_history,
                enabled=semantic_history_feature,
            )
        return _with_causal_semantic_history_features(
            prepared, property_history, enabled=semantic_history_feature
        )

    @torch.no_grad()
    def forward_step(
        _transition_cpu: dict[str, Any], transition: dict[str, Any], hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        nonlocal property_history, property_reference_history
        if property_history is None or property_reference_history is None:
            raise RuntimeError("Action evaluator transition preparation was skipped.")
        assert property_reference_history is not None




        proposal_enabled = enabled or any(
            source == "controller" for source in operation_score_source.values()
        )
        z_raw, proposed_action, _z_t, details = _soft_action(
            model=model,
            action_encoder=action_encoder,
            controllers=controllers,
            transition=transition,
            dataset_name=dataset_name,
            hidden=hidden,
            rates=rates,
            enabled=proposal_enabled,
            action_mode=action_mode,
            action_mode_overrides=action_mode_overrides,
            action_topk=action_topk,
            count_support_operations=count_support_operations,
            count_decision_operations=count_decision_operations,
            use_property_magnitude=use_property_magnitude,
            property_history=property_history,
            property_history_action_weight=property_history_action_weight,
            vectorized_controller_forward=vectorized_controller_forward,
        )
        action = proposed_action if enabled else None
        outputs = model(
            transition["x_t"],
            transition["edge_index_t"],
            hidden,
            action=action,
            edge_weight_t=_edge_weight_for_dataset(transition, dataset_name),
            decode_observables=(
                scoring_split is None
                or _transition_cpu["split"] == scoring_split
            ),
            decode_node_features=False,
            decode_node_property=decode_node_property,
            decode_current_property=False,
            node_property_ids=transition["semantic_node_ids"],
            precomputed_z_t_raw=z_raw,
            node_activity_observed_context=_node_activity_observed_context(
                transition,
                enabled=node_activity_current_state_context,
                include_source_visibility=node_activity_source_visibility_context,
                include_full_state=node_activity_full_observed_context,
                include_history=node_activity_history_context,
            ),
            node_activity_action_context=_node_activity_action_context(
                details,
                num_nodes=int(transition["x_t"].shape[0]),
                device=transition["x_t"].device,
                dtype=transition["x_t"].dtype,
                enabled=node_activity_action_context and enabled,
            ),
        )
        for operation, output_key in (
            (GraphOperation.ADD_NODE, "node_activation_logits"),
            (GraphOperation.REMOVE_NODE, "node_deactivation_logits"),
        ):
            if operation_score_source[operation] != "controller":
                continue






            probability = details["proposals"][operation.name][
                "controller_probability"
            ].to(device=outputs["latent_next"].device, dtype=outputs["latent_next"].dtype)
            controller_logits = torch.logit(
                probability.clamp(1e-6, 1.0 - 1e-6)
            )
            outputs[output_key] = controller_logits
            if operation == GraphOperation.REMOVE_NODE and node_removal_history_bias != 0.0:





                history_features = transition.get("controller_history_features")
                if history_features is not None and history_features.ndim == 2:
                    candidates = _candidate_nodes(transition, operation)
                    if candidates.numel() > 0 and history_features.shape[1] >= 3:
                        trend = -history_features[:, 2].to(
                            device=outputs[output_key].device,
                            dtype=outputs[output_key].dtype,
                        ).index_select(0, candidates)
                        trend = (trend - trend.mean()) / trend.std(
                            unbiased=False
                        ).clamp_min(1e-4)
                        outputs[output_key] = outputs[output_key].clone()
                        outputs[output_key].index_add_(
                            0,
                            candidates,
                            float(node_removal_history_bias) * trend,
                        )
        if (
            node_removal_group_calibration_mode != "none"
            and float(node_removal_group_calibration_weight) > 0.0
            and "node_deactivation_logits" in outputs
        ):








            output_key = "node_deactivation_logits"
            candidates = _candidate_nodes(transition, GraphOperation.REMOVE_NODE)
            if candidates.numel() > 1:
                logits = outputs[output_key]
                values = logits.index_select(0, candidates)

                def percentile_logits(group_values: torch.Tensor) -> torch.Tensor:
                    order = torch.argsort(group_values, stable=True)
                    ranks = torch.empty_like(order)
                    ranks.index_copy_(
                        0,
                        order,
                        torch.arange(order.numel(), device=order.device),
                    )
                    probability = (
                        ranks.to(dtype=group_values.dtype) + 0.5
                    ) / float(order.numel())
                    return torch.logit(probability.clamp(1e-4, 1.0 - 1e-4))

                if node_removal_group_calibration_mode == "zscore":
                    relative = (values - values.mean()) / values.std(
                        unbiased=False
                    ).clamp_min(1e-4)
                elif node_removal_group_calibration_mode == "percentile":
                    relative = percentile_logits(values)
                else:
                    trajectory = transition.get("activity_trajectory_features")
                    if trajectory is None or trajectory.ndim != 2:
                        raise ValueError(
                            "stratified removal calibration requires causal "
                            "activity trajectory features"
                        )
                    activity = trajectory[:, 2].to(
                        device=values.device, dtype=values.dtype
                    ).index_select(0, candidates)
                    relative = torch.empty_like(values)
                    activity_order = torch.argsort(activity, stable=True)
                    for stratum in torch.tensor_split(
                        activity_order, min(3, int(activity_order.numel()))
                    ):
                        if stratum.numel() == 0:
                            continue
                        relative.index_copy_(
                            0,
                            stratum,
                            percentile_logits(values.index_select(0, stratum)),
                        )
                weight = float(node_removal_group_calibration_weight)
                calibrated = (1.0 - weight) * values + weight * relative
                outputs[output_key] = logits.clone()
                outputs[output_key].index_copy_(0, candidates, calibrated)
        if (
            float(semantic_history_bias) != 0.0
            and "node_change_logits" in outputs
            and transition["x_t"].shape[1] >= 2
        ):
            semantic_history = transition["x_t"][..., -2].to(
                device=outputs["node_change_logits"].device,
                dtype=outputs["node_change_logits"].dtype,
            )
            comparable_nodes = transition["semantic_node_ids"].to(
                device=semantic_history.device
            )
            if comparable_nodes.numel() > 0:
                signal = semantic_history.index_select(0, comparable_nodes)
                signal = (signal - signal.mean()) / signal.std(
                    unbiased=False
                ).clamp_min(1e-4)
                outputs["node_change_logits"] = outputs[
                    "node_change_logits"
                ].clone()
                outputs["node_change_logits"].index_add_(
                    0,
                    comparable_nodes,
                    float(semantic_history_bias) * signal,
                )
        _attach_causal_property_reference(
            outputs,
            transition,
            property_reference_history,
            mode=semantic_reference_mode,
        )
        if (
            float(semantic_property_change_bias) != 0.0
            and "node_change_logits" in outputs
        ):






            predicted_property, _delta, _decoded = _semantic_prediction(
                outputs,
                transition,
                property_mode=semantic_property_mode,
                property_gate_bias=float(semantic_gate_bias),
            )
            comparable = transition["semantic_comparable_mask"].to(
                device=predicted_property.device, dtype=torch.bool
            )
            if bool(comparable.any()):
                implied_distance = support_topk_jaccard_distance(
                    transition["semantic_current"][comparable],
                    predicted_property[comparable],
                    topk=semantic_topk,
                ).to(dtype=outputs["node_change_logits"].dtype)
                implied_distance = (
                    implied_distance - implied_distance.mean()
                ) / implied_distance.std(unbiased=False).clamp_min(1e-4)
                comparable_nodes = transition["semantic_node_ids"].index_select(
                    0, comparable.nonzero(as_tuple=False).flatten()
                )
                outputs["node_change_logits"] = outputs[
                    "node_change_logits"
                ].clone()
                outputs["node_change_logits"].index_add_(
                    0,
                    comparable_nodes,
                    float(semantic_property_change_bias) * implied_distance,
                )
        if (
            float(semantic_transition_gate_bias) != 0.0
            and "node_change_logits" in outputs
            and "node_property_transition_gate_logits" in outputs
        ):



            gate_logits = outputs["node_property_transition_gate_logits"]
            comparable = transition["semantic_comparable_mask"].to(
                device=gate_logits.device, dtype=torch.bool
            )
            if bool(comparable.any()):
                comparable_rows = comparable.nonzero(as_tuple=False).flatten()
                comparable_nodes = transition["semantic_node_ids"].index_select(
                    0, comparable_rows
                )
                gate_signal = torch.sigmoid(
                    gate_logits.index_select(0, comparable_nodes)
                ).to(dtype=outputs["node_change_logits"].dtype)
                gate_signal = (gate_signal - gate_signal.mean()) / gate_signal.std(
                    unbiased=False
                ).clamp_min(1e-4)
                outputs["node_change_logits"] = outputs[
                    "node_change_logits"
                ].clone()
                outputs["node_change_logits"].index_add_(
                    0,
                    comparable_nodes,
                    float(semantic_transition_gate_bias) * gate_signal,
                )
        if (
            float(semantic_volatility_residual_bias) != 0.0
            and "node_change_logits" in outputs
        ):




            comparable = transition["semantic_comparable_mask"].to(torch.bool)
            if bool(comparable.any()):
                comparable_rows = comparable.nonzero(as_tuple=False).flatten()
                comparable_nodes = transition["semantic_node_ids"].index_select(
                    0, comparable_rows
                )
                volatility = property_history.score(comparable_nodes).to(
                    device=outputs["node_change_logits"].device,
                    dtype=outputs["node_change_logits"].dtype,
                )
                volatility = (volatility - volatility.mean()) / volatility.std(
                    unbiased=False
                ).clamp_min(1e-4)
                outputs["node_change_logits"] = outputs[
                    "node_change_logits"
                ].clone()
                outputs["node_change_logits"].index_add_(
                    0,
                    comparable_nodes,
                    float(semantic_volatility_residual_bias) * volatility,
                )
        target_records: list[tuple[GraphOperation, torch.Tensor, torch.Tensor]] = []
        for operation in NODE_OPERATIONS:
            _candidates, labels, known, _full = _operation_targets(
                transition,
                operation,
                semantic_threshold=semantic_threshold,
                semantic_topk=semantic_topk,
                semantic_history_cache=semantic_history_cache,
            )
            target_records.append((operation, labels, known))
        if details["proposals"]:
            decision_counts: list[torch.Tensor | None] = []
            future_counts = outputs.get("node_operation_count_log1p")
            score_transition = (
                scoring_split is None or _transition_cpu["split"] == scoring_split
            )
            for operation_index, operation in enumerate(NODE_OPERATIONS):
                if operation not in count_decision_operations:
                    decision_counts.append(None)
                    continue
                if node_operation_count_source == "future_latent":
                    if not score_transition:
                        decision_counts.append(None)
                        continue
                    if future_counts is None:
                        raise RuntimeError(
                            "node_operation_count_source='future_latent' requires "
                            "a node_operation_count_decoder."
                        )
                    value = future_counts.reshape(-1)[operation_index]
                elif node_operation_count_source == "controller":
                    value = details["proposals"][operation.name]["predicted_log_count"]
                else:
                    value = (
                        None
                        if recent_count_history is None
                        else recent_count_history.log_count(
                            operation,
                            device=outputs["latent_next"].device,
                        )
                    )
                    if value is None:
                        value = details["proposals"][operation.name][
                            "predicted_log_count"
                        ]
                decision_counts.append(value)
            if any(value is not None for value in decision_counts):
                outputs["_decision_count_log1p"] = torch.stack(
                    [
                        value
                        if value is not None
                        else outputs["latent_next"].new_full((), float("nan"))
                        for value in decision_counts
                    ]
                )
        for operation, labels, known in target_records:
            rates.observe(operation, labels, known)
            if recent_count_history is not None:
                recent_count_history.observe(operation, labels, known)



        if "x_next" in transition:
            outputs["_target_x_next"] = transition["x_next"]
        return outputs




    forward_step.prepare_transition = prepare_transition
    return forward_step


def _selection_score(
    validation: dict[str, Any], *, property_weight: float = 0.0, metric: str = "auprc"
) -> float:

    if property_weight < 0.0:
        raise ValueError("property_weight must be non-negative")
    if metric not in {"auprc", "f1", "min_f1"}:
        raise ValueError("metric must be 'auprc', 'f1', or 'min_f1'")
    localization_metric = "f1" if metric == "min_f1" else metric
    values = [
        validation["node_addition"][localization_metric],
        validation["node_removal"][localization_metric],
        validation["semantic_change"][localization_metric],
    ]
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return float("-inf")
    if metric == "min_f1":



        return float(min(usable))
    property_ndcg = validation["semantic_prediction"].get("official_ndcg")
    if property_weight > 0.0 and property_ndcg is not None:
        return float(
            (sum(usable) + float(property_weight) * float(property_ndcg))
            / (len(usable) + float(property_weight))
        )
    return float(sum(usable) / len(usable))


def _shape_sanity(
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    controllers: torch.nn.ModuleDict,
    dataset: NodePropertyTransitionDataset,
    *,
    device: torch.device,
    semantic_threshold: float,
    semantic_topk: int,
    action_mode: str,
    action_mode_overrides: dict[GraphOperation, str] | None,
    action_topk: int,
    count_support_operations: frozenset[GraphOperation],
    count_decision_operations: frozenset[GraphOperation],
    use_property_magnitude: bool,
    property_history_action_weight: float,
    semantic_history_feature: bool,
    node_activity_current_state_context: bool,
    node_activity_source_visibility_context: bool,
    node_activity_full_observed_context: bool,
    node_activity_history_context: bool,
    node_activity_action_context: bool,
) -> dict[str, list[int]]:
    transition_cpu = dataset[0]
    history = OnlineSemanticChangeHistory(
        int(dataset.metadata["num_nodes"]),
        int(dataset.metadata["official_target_dim"]),
        topk=semantic_topk,
    )
    history.attach_cache(getattr(dataset, "semantic_history_cache", None))
    history.observe(
        transition_cpu["property_node_ids_t"],
        transition_cpu["property_observed_t"],
        transition_id=int(transition_cpu["transition_id"]),
    )
    transition = t2_transition_to_device(
        _with_causal_semantic_history_features(
            transition_cpu, history, enabled=semantic_history_feature
        ),
        device,
    )
    hidden = model.initial_hidden(int(dataset.metadata["num_nodes"]), device)
    with torch.no_grad():
        z_raw, action, _z_t, _details = _soft_action(
            model=model,
            action_encoder=action_encoder,
            controllers=controllers,
            transition=transition,
            dataset_name=dataset.dataset_name,
            hidden=hidden,
            rates=_PastOperationRates(device),
            enabled=True,
            action_mode=action_mode,
            action_mode_overrides=action_mode_overrides,
            action_topk=action_topk,
            count_support_operations=count_support_operations,
            count_decision_operations=count_decision_operations,
            use_property_magnitude=use_property_magnitude,
            property_history=None,
            property_history_action_weight=property_history_action_weight,
        )
        outputs = model(
            transition["x_t"],
            transition["edge_index_t"],
            hidden,
            action=action,
            edge_weight_t=_edge_weight_for_dataset(transition, dataset.dataset_name),
            precomputed_z_t_raw=z_raw,
            node_activity_observed_context=_node_activity_observed_context(
                transition,
                enabled=node_activity_current_state_context,
                include_source_visibility=node_activity_source_visibility_context,
                include_full_state=node_activity_full_observed_context,
                include_history=node_activity_history_context,
            ),
            node_activity_action_context=_node_activity_action_context(
                _details,
                num_nodes=int(transition["x_t"].shape[0]),
                device=transition["x_t"].device,
                dtype=transition["x_t"].dtype,
                enabled=node_activity_action_context,
            ),
        )
    return {
        "x_t": list(transition["x_t"].shape),
        "edge_index_t": list(transition["edge_index_t"].shape),
        "action": [] if action is None else list(action.shape),
        "z_t": list(outputs["z_t"].shape),
        "hidden_next": list(outputs["hidden_next"].shape),
        "latent_next": list(outputs["latent_next"].shape),
        "node_activation_logits": list(outputs["node_activation_logits"].shape),
        "node_deactivation_logits": list(outputs["node_deactivation_logits"].shape),
        "semantic_change_logits": list(outputs["node_change_logits"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DEFAULT_PROCESSED), required=True)
    parser.add_argument("--processed", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action_dim", type=int, default=32)
    parser.add_argument("--policy_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--controller_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--fused_adamw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PyTorch's FP32 fused AdamW implementation.",
    )
    parser.add_argument("--lambda_activity", type=float, default=1.0)
    parser.add_argument(
        "--natural_addition_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Relative training weight for naturally occurring ADD_NODE positives; "
            "one preserves the uniform node-addition objective."
        ),
    )
    parser.add_argument("--lambda_semantic_change", type=float, default=1.0)
    parser.add_argument(
        "--lambda_semantic_distance",
        type=float,
        default=0.0,
        help=(
            "Weight of the continuous Top-k-support distance auxiliary for "
            "semantic-change localization; zero preserves the binary-only loss."
        ),
    )
    parser.add_argument(
        "--semantic_pos_weight_scale",
        type=float,
        default=1.0,
        help="Scale for the train-split semantic-change positive-class weight.",
    )
    parser.add_argument(
        "--dynamic_change_hard_fraction",
        type=float,
        default=0.0,
        help=(
            "Enable per-transition, per-class dynamic hard-example supervision "
            "for addition, removal, and semantic change; zero preserves the "
            "ordinary objectives."
        ),
    )
    parser.add_argument(
        "--dynamic_change_focal_gamma",
        type=float,
        default=1.0,
        help="Focal exponent used by --dynamic_change_hard_fraction.",
    )
    parser.add_argument(
        "--semantic_hard_pair_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for causal hard-positive versus hard-negative semantic "
            "ranking; zero preserves the ordinary semantic objective."
        ),
    )
    parser.add_argument(
        "--semantic_hard_pair_margin",
        type=float,
        default=0.5,
        help="Logit margin used by --semantic_hard_pair_weight.",
    )
    parser.add_argument("--lambda_semantic_value", type=float, default=1.0)
    parser.add_argument(
        "--changed_semantic_value_weight",
        type=float,
        default=1.0,
        help=(
            "Relative property-loss weight for rows with a released positive "
            "semantic-change label; one preserves uniform row weighting."
        ),
    )
    parser.add_argument(
        "--lambda_property_transition_gate",
        type=float,
        default=0.0,
        help=(
            "Supervise the optional Zhat-derived property transition gate with "
            "the released semantic-change label."
        ),
    )
    parser.add_argument("--lambda_current_semantic", type=float, default=0.1)
    parser.add_argument(
        "--selection_property_weight",
        type=float,
        default=0.0,
        help=(
            "Validation-only weight of official property NDCG@10 when choosing "
            "the Task-1 checkpoint."
        ),
    )
    parser.add_argument(
        "--selection_metric",
        choices=["auprc", "f1", "min_f1"],
        default="auprc",
        help=(
            "Validation metric averaged across node addition, node removal, "
            "and semantic-change localization when selecting the checkpoint; "
            "min_f1 instead maximizes the weakest validation F1."
        ),
    )
    parser.add_argument(
        "--decision_calibration_tail_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of the latest validation transition groups used to fit "
            "binary decision thresholds; one preserves full-validation calibration."
        ),
    )
    parser.add_argument(
        "--node_operation_count_decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Decode next-step ADD_NODE, REMOVE_NODE, and "
            "MODIFY_NODE_PROPERTY cardinalities from predicted future latent."
        ),
    )
    parser.add_argument(
        "--lambda_node_operation_count",
        type=float,
        default=0.0,
        help="Weight of log(1 + next operation count) supervision for the optional count decoder.",
    )
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_latent_cosine", type=float, default=0.1)
    parser.add_argument("--lambda_latent_variance", type=float, default=1e-3)
    parser.add_argument(
        "--changed_node_latent_weight",
        type=float,
        default=1.0,
        help=(
            "Optional relative future-latent loss weight for nodes with a "
            "released addition, removal, or semantic-property change; one "
            "keeps uniform latent supervision."
        ),
    )
    parser.add_argument(
        "--lambda_deactivation",
        type=float,
        default=1.0,
        help=(
            "Relative REMOVE_NODE BCE/focal-loss weight inside the node-edit "
            "objective; one preserves equal ADD/REMOVE weighting."
        ),
    )
    parser.add_argument(
        "--action_lambda_deactivation",
        type=float,
        default=None,
        help=(
            "Optional REMOVE_NODE loss weight for the action-conditioned "
            "transition branch. When omitted, it equals --lambda_deactivation."
        ),
    )
    parser.add_argument(
        "--deactivation_focal_gamma",
        type=float,
        default=0.0,
        help=(
            "Focal exponent for sparse REMOVE_NODE supervision; zero is "
            "the legacy class-weighted BCE objective."
        ),
    )
    parser.add_argument(
        "--lambda_deactivation_ranking",
        type=float,
        default=0.0,
        help=(
            "Weight of the sparse REMOVE_NODE pairwise ranking auxiliary; "
            "zero preserves the original BCE-only objective."
        ),
    )
    parser.add_argument(
        "--lambda_activation_ranking",
        type=float,
        default=0.0,
        help=(
            "Weight of the ADD_NODE pairwise ranking auxiliary; zero keeps "
            "the original class-weighted BCE objective."
        ),
    )
    parser.add_argument(
        "--node_change_decoder_input",
        choices=[
            "future",
            "latent_delta",
            "latent_abs_delta",
            "future_concat_delta",
            "future_concat_abs_delta",
        ],
        default="future",
        help=(
            "Input to the semantic-change decoder: the predicted future "
            "latent, its signed/absolute difference from the observed latent, "
            "or a concatenation of the two."
        ),
    )
    parser.add_argument(
        "--node_activity_decoder_input",
        choices=[
            "future",
            "latent_delta",
            "latent_abs_delta",
            "future_concat_delta",
            "future_concat_abs_delta",
            "future_concat_observed",
            "future_concat_delta_observed",
            "future_concat_hidden_observed",
        ],
        default="future",
        help=(
            "Input to the future-latent node addition/removal decoder. "
            "Delta modes compare Zhat_(t+1) with the observed source Z_t; "
            "observed modes append configured current node descriptors; the "
            "hidden-observed mode also exposes the updated state-module hidden "
            "state to the activity readout."
        ),
    )
    parser.add_argument(
        "--node_activity_current_state_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append the observed current membership and degree descriptors to "
            "the Task-1 node activity decoder. Requires --observed_degree_feature."
        ),
    )
    parser.add_argument(
        "--node_activity_full_observed_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append the complete causal source node-feature matrix to the "
            "Task-1 activity decoder input."
        ),
    )
    parser.add_argument(
        "--node_activity_source_visibility_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append a causal source-feature visibility indicator to the Task-1 "
            "node activity decoder. Requires --node_activity_current_state_context."
        ),
    )
    parser.add_argument(
        "--node_addition_source_visibility_threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit separate validation-only ADD_NODE score thresholds for "
            "source-visible and source-masked candidates."
        ),
    )
    parser.add_argument(
        "--node_activity_history_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append the compact strictly-pre-source temporal summary to the "
            "change-aware ADD_NODE/REMOVE_NODE decoder."
        ),
    )
    parser.add_argument(
        "--node_activity_trajectory_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append past-only weighted-strength and unweighted-degree trajectory "
            "features to the activity change-aware decoder context."
        ),
    )
    parser.add_argument(
        "--node_activity_history_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized causal history-gated residual to the "
            "operation-specific activity decoder. Requires history context."
        ),
    )
    parser.add_argument(
        "--node_activity_history_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized direct linear prior from causal history "
            "to the ADD_NODE/REMOVE_NODE decoder. Requires history context."
        ),
    )
    parser.add_argument(
        "--node_activity_history_prior_remove_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route the direct activity history prior only to REMOVE_NODE, "
            "leaving the ADD_NODE logit on its future-latent path."
        ),
    )
    parser.add_argument(
        "--node_activity_action_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append the Controller's current-state-only ADD_NODE, REMOVE_NODE, "
            "and MODIFY_NODE_PROPERTY proposal probabilities to the Task-1 "
            "future-latent readout."
        ),
    )
    parser.add_argument(
        "--node_activity_addition_source_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized source-graph residual branch to the "
            "ADD_NODE activity readout."
        ),
    )
    parser.add_argument(
        "--node_change_source_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized source-graph residual branch to the "
            "semantic-change localization readout."
        ),
    )
    parser.add_argument(
        "--semantic_causal_expert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse a compact released-history semantic-change expert with the "
            "ordinary future-latent stable-node expert through a causal gate."
        ),
    )
    parser.add_argument(
        "--node_change_history_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized direct causal-history prior to semantic "
            "change localization. Requires semantic causal expert."
        ),
    )
    parser.add_argument(
        "--detach_node_change_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train the semantic-change decoder jointly while preventing its "
            "localization loss from updating the shared future-state path."
        ),
    )
    parser.add_argument(
        "--node_activity_separate_heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use independent future-latent MLP readouts for ADD_NODE and "
            "REMOVE_NODE."
        ),
    )
    parser.add_argument(
        "--node_activity_decoder_activation",
        choices=["relu", "gelu"],
        default="relu",
        help=(
            "Nonlinearity in the future-latent ADD_NODE/REMOVE_NODE decoder."
        ),
    )
    parser.add_argument(
        "--node_activity_score_source",
        choices=["future_latent", "controller"],
        default="future_latent",
        help=(
            "Rank Task-1 ADD_NODE and REMOVE_NODE candidates with the "
            "future-latent decoder or the past-only Controller action score."
        ),
    )
    parser.add_argument(
        "--node_addition_score_source",
        choices=["same", "future_latent", "controller"],
        default="same",
        help=(
            "Optional ADD_NODE scoring override; 'same' reuses "
            "--node_activity_score_source."
        ),
    )
    parser.add_argument(
        "--node_removal_score_source",
        choices=["same", "future_latent", "controller"],
        default="same",
        help=(
            "Optional REMOVE_NODE scoring override; 'same' reuses "
            "--node_activity_score_source."
        ),
    )
    parser.add_argument(
        "--node_removal_history_bias",
        type=float,
        default=0.0,
        help=(
            "Evaluation-time coefficient for the standardized negative "
            "past interaction-strength trend in REMOVE_NODE ranking."
        ),
    )
    parser.add_argument(
        "--node_removal_group_calibration_mode",
        choices=["none", "zscore", "percentile", "stratified_percentile"],
        default="none",
        help=(
            "Causal per-transition normalization for REMOVE_NODE logits; "
            "stratified_percentile normalizes inside prior-activity strata."
        ),
    )
    parser.add_argument(
        "--node_removal_group_calibration_weight",
        type=float,
        default=0.0,
        help=(
            "Blend weight between learned removal logits and their causal "
            "per-transition normalized score."
        ),
    )
    parser.add_argument(
        "--node_removal_group_calibration_weight_grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Validation-only candidates for causal per-transition removal "
            "score normalization."
        ),
    )
    parser.add_argument(
        "--semantic_history_bias",
        type=float,
        default=0.0,
        help="Causal standardized past semantic-volatility bias for localization.",
    )
    parser.add_argument(
        "--semantic_property_change_bias",
        type=float,
        default=0.0,
        help=(
            "Fuse the causal Top-k support change implied by the predicted "
            "future semantic property into semantic-change localization."
        ),
    )
    parser.add_argument(
        "--semantic_property_change_bias_grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Validation-only candidates for predicted-property change "
            "fusion; select by semantic-change F1 after checkpoint selection."
        ),
    )
    parser.add_argument(
        "--activity_trajectory_supervision_bias",
        type=float,
        default=0.0,
        help=(
            "Causal trajectory disagreement weight for ADD/REMOVE boundary "
            "supervision using prior source-side trajectory features."
        ),
    )
    parser.add_argument(
        "--removal_history_supervision_bias",
        type=float,
        default=0.0,
        help=(
            "Causal controller-history disagreement weight for REMOVE_NODE "
            "boundary supervision; ADD_NODE is unchanged."
        ),
    )
    parser.add_argument(
        "--node_removal_history_bias_grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional validation-only candidates for the REMOVE_NODE history "
            "coefficient. The final readout selects the candidate with the "
            "best validation removal F1 after checkpoint selection."
        ),
    )
    parser.add_argument(
        "--evaluate_latent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute target-encoder latent diagnostics at final test time. "
            "Disabling this leaves the trained transition and every Task-1 "
            "observable metric unchanged."
        ),
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help=(
            "Select and save the best validation checkpoint without reading "
            "the test split."
        ),
    )
    parser.add_argument("--semantic_topk", type=int, default=10)
    parser.add_argument("--semantic_threshold_quantile", type=float, default=0.60)
    parser.add_argument("--semantic_property_mode", choices=["logit_residual", "logit_residual_mixture"], default="logit_residual_mixture")
    parser.add_argument(
        "--semantic_gate_bias",
        type=float,
        default=0.0,
        help=(
            "Evaluation-time additive bias for the semantic property "
            "transition gate logit; zero preserves the uncalibrated decoder."
        ),
    )
    parser.add_argument(
        "--semantic_gate_bias_grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional validation-only candidates for the semantic transition "
            "gate bias. The final readout selects the candidate with the best "
            "validation Changed-NDCG@10 after checkpoint selection."
        ),
    )
    parser.add_argument(
        "--semantic_transition_gate_bias",
        type=float,
        default=0.0,
        help=(
            "Causal readout coefficient for the predicted semantic transition "
            "gate; it is fused over current-state comparable nodes only."
        ),
    )
    parser.add_argument(
        "--semantic_volatility_residual_bias",
        type=float,
        default=0.0,
        help=(
            "Causal semantic-history volatility residual fused into change "
            "ranking over comparable nodes."
        ),
    )
    parser.add_argument(
        "--semantic_reference_mode",
        choices=["current", "historical_mean"],
        default="current",
        help=(
            "Observable property reference for the future-latent readout: "
            "Y_t or the causal per-node mean through time t."
        ),
    )
    parser.add_argument("--semi_synthetic_node_edits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--observed_degree_feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a normalized degree computed from the current observed edge set.",
    )
    parser.add_argument(
        "--controller_remove_observed_degree",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Provide normalized current degree only to the REMOVE_NODE "
            "Controller proposal head."
        ),
    )
    parser.add_argument(
        "--controller_history_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Provide the Controller with a compact node summary computed from "
            "snapshots strictly before the current source graph."
        ),
    )
    parser.add_argument(
        "--controller_history_window",
        type=int,
        default=8,
        help="Number of preceding snapshots used by the Controller history summary.",
    )
    parser.add_argument(
        "--controller_addition_source_visibility",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Condition the ADD_NODE Controller on whether each candidate retains "
            "released source-state evidence, with source-regime-balanced supervision."
        ),
    )
    parser.add_argument(
        "--controller_addition_source_visibility_balance",
        choices=["per_transition", "train_global"],
        default="per_transition",
        help=(
            "Supervision balance for the observable source-visible and "
            "source-masked ADD_NODE regimes."
        ),
    )
    parser.add_argument("--construction_seed", type=int, default=1)
    parser.add_argument("--construction_scale", type=float, default=1.0)
    parser.add_argument(
        "--construction_rate_floor",
        type=float,
        default=0.0,
        help=(
            "Minimum train-defined fraction of shared nodes used for each "
            "semi-synthetic node-edit type; zero preserves natural-rate calibration."
        ),
    )
    parser.add_argument(
        "--construction_strategy",
        choices=[
            "past_observable", "temporal_trend", "temporal_stratified",
            "temporal_stratified_sampled", "long_horizon_stratified",
            "long_horizon_weighted_stratified", "long_horizon_hybrid_stratified",
            "random",
        ],
        default="past_observable",
    )
    parser.add_argument(
        "--construction_contrastive_addition_negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mask matched shared-node non-additions on both source and target "
            "snapshots using the causal complement of the addition priority."
        ),
    )
    parser.add_argument("--action_mode", choices=["expected", "centered", "centered_local", "local"], default="centered_local")
    parser.add_argument(
        "--add_action_mode",
        choices=["expected", "centered", "centered_local", "local"],
        default=None,
        help="Optional ADD_NODE localization override.",
    )
    parser.add_argument(
        "--remove_action_mode",
        choices=["expected", "centered", "centered_local", "local"],
        default=None,
        help="Optional REMOVE_NODE localization override.",
    )
    parser.add_argument(
        "--property_action_mode",
        choices=["expected", "centered", "centered_local", "local"],
        default=None,
        help="Optional MODIFY_NODE_PROPERTY localization override.",
    )
    parser.add_argument("--action_topk", type=int, default=0)
    parser.add_argument(
        "--action_count_support_mode",
        choices=["none", "add", "edits", "all", "remove"],
        default="none",
        help=(
            "Which Controller operation uses its current-state predicted "
            "node-edit count as a sparse action support."
        ),
    )
    parser.add_argument(
        "--controller_count_supervision_mode",
        choices=["same", "none", "add", "edits", "all", "remove"],
        default="same",
        help=(
            "Which Controller count heads receive released count supervision; "
            "'same' preserves --action_count_support_mode behavior."
        ),
    )
    parser.add_argument(
        "--controller_count_decision_mode",
        choices=["same", "none", "add", "edits", "all", "remove"],
        default="same",
        help=(
            "Which Controller count heads select ranked edit outputs at evaluation; "
            "'same' preserves --action_count_support_mode behavior."
        ),
    )
    parser.add_argument(
        "--node_operation_count_history_window",
        type=int,
        default=0,
        help=(
            "Use the mean cardinality of this many completed transitions as "
            "a causal count reference at evaluation; zero disables it."
        ),
    )
    parser.add_argument(
        "--node_operation_count_source",
        choices=["action_history", "controller", "future_latent"],
        default="action_history",
        help=(
            "Source of the ranked node-edit cardinality at evaluation: a "
            "causal history reference, the Controller count head, or the "
            "Zhat-derived operation-count decoder."
        ),
    )
    parser.add_argument(
        "--force_node_addition_count_decision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Execute ADD_NODE as a count-conditioned ranked action when "
            "a valid Controller count is available."
        ),
    )
    parser.add_argument(
        "--force_node_removal_count_decision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Execute REMOVE_NODE as a count-conditioned ranked action when "
            "a valid Controller count is available."
        ),
    )
    parser.add_argument("--action_warmup_epochs", type=int, default=5)
    parser.add_argument("--grpo_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--action_use_property_magnitude",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode a Controller-predicted property delta in the action token.",
    )
    parser.add_argument(
        "--action_injection_mode",
        choices=["full", "current_residual", "post_norm_residual"],
        default="full",
    )
    parser.add_argument("--controller_supervised_weight", type=float, default=1.0)
    parser.add_argument(
        "--controller_property_magnitude_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of conditional Gaussian supervision for the predicted "
            "MODIFY_NODE_PROPERTY magnitude; active only with "
            "--action_use_property_magnitude."
        ),
    )
    parser.add_argument(
        "--controller_node_count_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of released log node-edit-count supervision for each "
            "operation-specific Controller head."
        ),
    )
    parser.add_argument(
        "--controller_activation_ranking_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the Controller ADD_NODE pairwise ranking auxiliary; "
            "zero preserves the BCE-only Controller objective."
        ),
    )
    parser.add_argument(
        "--controller_deactivation_ranking_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the Controller REMOVE_NODE hard-negative pairwise "
            "ranking loss."
        ),
    )
    parser.add_argument(
        "--property_history_action_weight",
        type=float,
        default=0.0,
        help=(
            "Past-only semantic-volatility ranking weight for the "
            "MODIFY_NODE_PROPERTY Controller proposal."
        ),
    )
    parser.add_argument(
        "--semantic_history_feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append causal per-node semantic-change history and availability "
            "channels to the observed source graph state."
        ),
    )
    parser.add_argument(
        "--semantic_history_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Precompute causal semantic-history supports once and reuse them "
            "across epochs. This is the shared default for every dataset."
        ),
    )
    parser.add_argument(
        "--property_transition_gate_decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a separate future-latent gate for property residual decoding "
            "instead of sharing the semantic-change localization head."
        ),
    )
    parser.add_argument(
        "--action_world_weight",
        type=float,
        default=1.0,
        help="Weight of released next-state supervision on the action-conditioned branch.",
    )
    parser.add_argument(
        "--action_update_state_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also update the shared State Module with the action-conditioned "
            "next-state objective. The zero-action objective remains active "
            "and anchors the observational transition."
        ),
    )
    parser.add_argument(
        "--action_observable_decoder_update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also update Zhat's latent predictor and Task-1 observable "
            "decoders with released future supervision from the action branch."
        ),
    )
    parser.add_argument(
        "--action_advantage_weight",
        type=float,
        default=0.5,
        help="Weight of the action-vs-zero-action observable hinge.",
    )
    parser.add_argument(
        "--action_advantage_margin",
        type=float,
        default=0.0,
        help="Allowed action-branch observable loss gap relative to zero action.",
    )
    parser.add_argument(
        "--action_plan_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional Zhat consistency loss against the detached, past-only "
            "three-operation Controller plan."
        ),
    )
    parser.add_argument(
        "--action_plan_positive_weight_cap",
        type=float,
        default=0.0,
        help=(
            "Optional positive-class cap for the sparse action-plan loss; "
            "zero leaves it unweighted."
        ),
    )
    parser.add_argument(
        "--action_plan_target_mode",
        choices=["action_probability", "controller_probability"],
        default="action_probability",
        help=(
            "Past-only target used by the optional Zhat action-plan readout."
        ),
    )
    parser.add_argument(
        "--action_decoder_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional consistency loss from detached Controller proposals to "
            "the standard Task-1 decoders of Zhat."
        ),
    )
    parser.add_argument(
        "--action_decoder_positive_weight_cap",
        type=float,
        default=0.0,
        help=(
            "Optional positive-class cap for sparse decoder consistency; "
            "zero leaves it unweighted."
        ),
    )
    parser.add_argument(
        "--action_decoder_target_mode",
        choices=["action_probability", "controller_probability"],
        default="action_probability",
        help=(
            "Past-only Controller target used by the optional released-decoder "
            "consistency loss."
        ),
    )
    parser.add_argument(
        "--action_decoder_consistency_operations",
        choices=["all", "property", "activity", "removal"],
        default="all",
        help=(
            "Released Task-1 decoder heads receiving the optional "
            "Controller-plan consistency objective."
        ),
    )
    parser.add_argument("--grpo_weight", type=float, default=0.05)
    parser.add_argument(
        "--grpo_rollout_budget",
        type=int,
        default=12,
        help=(
            "Total GRPO rollouts per update. The default leaves budget beyond "
            "the two-rollout minimum for each of the three operation groups, "
            "so dynamic rarity-based group allocation is active."
        ),
    )
    parser.add_argument("--grpo_interval", type=int, default=4)
    parser.add_argument("--grpo_clip", type=float, default=0.2)
    parser.add_argument("--grpo_kl", type=float, default=0.01)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument(
        "--batch_grpo_rollout_forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Batch independent T1 GRPO world-model reward forwards within "
            "each action group."
        ),
    )
    parser.add_argument(
        "--t1_composite_structure_reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Score every T1 rollout by the equal-weight mean of node addition, "
            "node removal, representation-change localization, and "
            "representation-change magnitude rewards. Enabled by default."
        ),
    )
    parser.add_argument(
        "--cache_static_action_candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache only immutable one-dimensional T1 action-candidate index "
            "tensors in a 16 MiB process-local LRU."
        ),
    )
    parser.add_argument(
        "--vectorized_controller_forward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate the independent operation-specific Controller trunks "
            "in one batched kernel without sharing parameters."
        ),
    )
    parser.add_argument(
        "--reuse_controller_forward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse the differentiable Controller forward already computed "
            "for the current action proposal in the full-RL supervised "
            "Controller loss. This is an opt-in execution optimization; "
            "the default preserves the established two-forward path."
        ),
    )
    parser.add_argument(
        "--skip_controller_warmup_update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In an opt-in full-RL timing/training run, skip the Controller "
            "optimization pass while actions are disabled by the warm-up. "
            "Causal target/rate bookkeeping is retained; the default keeps "
            "the established warm-up update."
        ),
    )
    parser.add_argument(
        "--controller_update_interval",
        type=int,
        default=1,
        help=(
            "Opt-in interval (in chronological transitions) for the full-RL "
            "Controller optimizer update. A value greater than one scales "
            "the retained loss before stepping; the default 1 preserves the "
            "established per-transition update."
        ),
    )
    parser.add_argument(
        "--cache_validation_preparation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache only deterministic causal validation feature channels in "
            "a 64 MiB process-local LRU; hidden states and predictions are "
            "always recomputed."
        ),
    )
    parser.add_argument("--max_actions", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--max_train_transitions", type=int, default=None)
    parser.add_argument(
        "--cache_transitions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache deterministic augmented transitions with the shared bounded "
            "LRU policy. Enabled by default for every dataset."
        ),
    )
    parser.add_argument(
        "--defer_dense_masking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply deterministic semi-synthetic dense node masks after the "
            "source tensors reach the device, avoiding repeated CPU copies."
        ),
    )
    parser.add_argument(
        "--train_clip_transitions",
        type=int,
        default=None,
        help=(
            "Length of a rotating chronological training clip.  Use this for "
            "long timelines instead of repeatedly taking the earliest "
            "--max_train_transitions transitions."
        ),
    )
    parser.add_argument(
        "--clip_history_burn_in",
        type=int,
        default=32,
        help=(
            "Number of released prefix transitions replayed through the "
            "recurrent state before a rotating training clip."
        ),
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=1,
        help="Evaluate the complete chronological validation prefix every N epochs.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--pretrained_backbone",
        default=None,
        help=(
            "Optional LOO-pretrained WorldGraph graph/state/latent backbone. "
            "Task-specific Action, Controller and decoder modules remain fresh."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_state_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Transfer action-independent history-state layers. Action projection "
            "and gate tensors remain excluded in structural mode; use --no-... "
            "for a fresh state module."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_mode",
        choices=["structural", "legacy"],
        default="structural",
        help=(
            "structural keeps the target raw-feature input and transfers only "
            "feature-independent SGT weights; legacy uses the fixed-width "
            "domain-invariant adapter path."
        ),
    )
    parser.add_argument(
        "--pretrained_backbone_blend",
        type=float,
        default=0.25,
        help="Fraction of compatible pretrained weights to load in structural mode.",
    )
    parser.add_argument("--pretrained_graph_encoder_blend", type=float, default=None)
    parser.add_argument("--pretrained_state_model_blend", type=float, default=None)
    parser.add_argument("--pretrained_latent_predictor_blend", type=float, default=None)
    parser.add_argument(
        "--pretrained_controller_blend",
        type=float,
        default=0.0,
        help=(
            "Fraction of the optional source-only activity policy used to "
            "initialize the downstream ADD_NODE Controller (default: 0; "
            "the shared backbone remains the only transferred component)."
        ),
    )
    parser.add_argument(
        "--pretrained_task_blend",
        type=float,
        default=0.5,
        help="Blend for optional task-aligned semantic and node-policy transfer.",
    )
    parser.add_argument("--results", default=None)
    parser.add_argument("--latent_distribution", choices=["gaussian", "deterministic"], default="gaussian")
    parser.add_argument("--observable_latent_mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--target_encoder_momentum", type=float, default=0.99)
    parser.add_argument("--action_adapter_initial_scale", type=float, default=0.05)
    parser.add_argument("--action_residual_max_scale", type=float, default=0.15)
    parser.add_argument("--action_adapter_temperature", type=float, default=0.25)
    add_world_model_architecture_args(parser)
    parser.set_defaults(
        graph_encoder_type="mentor_sgt_gwm",
        state_model_type="mentor_sgt_gwm_transformer",
        sgt_deterministic_walks=True,
        latent_normalization="layernorm",
    )
    args = parser.parse_args()
    protocol_metadata: dict[str, Any] | None = None
    try:
        protocol_entry = assert_common_runtime(
            "T1",
            args.dataset,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            val_every=args.val_every,
        )
    except ValueError as error:
        parser.error(str(error))
    construction = protocol_entry["dataset_spec"]["node_edit_construction"]
    features = protocol_entry["task_spec"]["common_source_features"]
    expected = {
        "semi_synthetic_node_edits": bool(construction["semi_synthetic"]),
        "construction_seed": int(args.seed),
        "construction_scale": float(construction["scale"]),
        "construction_rate_floor": float(construction["rate_floor"]),
        "construction_strategy": str(construction["strategy"]),
        "construction_contrastive_addition_negatives": bool(
            construction["contrastive_addition_negatives"]
        ),
        "observed_degree_feature": bool(features["observed_degree"]),
        "semantic_topk": 10,
        "semantic_threshold_quantile": float(
            protocol_entry["task_spec"]["semantic_change"]["train_quantile"]
        ),
    }
    mismatches = [
        name
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if mismatches:
        parser.error(
            "Protocol shared setting mismatch for T1/"
            f"{args.dataset}: {', '.join(mismatches)}"
        )
    protocol_metadata = {
        "id": protocol_entry["protocol_id"],
        "digest": protocol_entry["protocol_digest"],
        "task": "T1",
        "dataset": args.dataset,
    }
    if args.action_dim < 1 or args.policy_dim < 1 or args.max_actions < 1:
        parser.error("action_dim, policy_dim, and max_actions must be positive")
    if args.grpo_rollout_budget < 2 * len(NODE_OPERATIONS):
        parser.error("grpo_rollout_budget must allocate at least two rollouts per operation")
    if args.controller_update_interval < 1:
        parser.error("controller_update_interval must be positive")
    if (
        args.grpo_interval < 1
        or args.val_every < 1
        or args.clip_history_burn_in < 0
        or args.epochs < 1
        or args.patience < 0
        or args.action_warmup_epochs < 0
        or args.grpo_warmup_epochs < 0
        or args.node_operation_count_history_window < 0
    ):
        parser.error("invalid epoch, patience, or GRPO interval")
    if args.controller_history_window < 1:
        parser.error("--controller_history_window must be positive")
    if args.max_train_transitions is not None and args.max_train_transitions < 1:
        parser.error("max_train_transitions must be positive when provided")
    if args.train_clip_transitions is not None and args.train_clip_transitions < 1:
        parser.error("train_clip_transitions must be positive when provided")
    if args.max_train_transitions is not None and args.train_clip_transitions is not None:
        parser.error(
            "Use either --max_train_transitions for a fixed smoke prefix or "
            "--train_clip_transitions for rotating chronological clips, not both."
        )
    if args.node_activity_current_state_context and not args.observed_degree_feature:
        parser.error(
            "--node_activity_current_state_context requires --observed_degree_feature"
        )
    if args.node_activity_full_observed_context and not args.node_activity_current_state_context:
        parser.error(
            "--node_activity_full_observed_context requires "
            "--node_activity_current_state_context"
        )
    if (
        args.node_activity_source_visibility_context
        and not args.node_activity_current_state_context
    ):
        parser.error(
            "--node_activity_source_visibility_context requires "
            "--node_activity_current_state_context"
        )
    if args.node_activity_history_context and not args.controller_history_features:
        parser.error(
            "--node_activity_history_context requires --controller_history_features"
        )
    if (
        args.node_activity_trajectory_features
        and not args.node_activity_history_context
    ):
        parser.error(
            "--node_activity_trajectory_features requires "
            "--node_activity_history_context"
        )
    if (
        args.node_removal_group_calibration_mode == "stratified_percentile"
        and not args.node_activity_trajectory_features
    ):
        parser.error(
            "stratified removal group calibration requires "
            "--node_activity_trajectory_features"
        )
    if args.node_activity_history_gate and not args.node_activity_history_context:
        parser.error(
            "--node_activity_history_gate requires --node_activity_history_context"
        )
    if args.node_activity_history_prior and not args.node_activity_history_context:
        parser.error(
            "--node_activity_history_prior requires --node_activity_history_context"
        )
    if (
        args.node_activity_history_prior_remove_only
        and not args.node_activity_history_prior
    ):
        parser.error(
            "--node_activity_history_prior_remove_only requires "
            "--node_activity_history_prior"
        )
    if args.semantic_causal_expert and not args.semantic_history_feature:
        parser.error(
            "--semantic_causal_expert requires --semantic_history_feature"
        )
    if args.node_change_history_prior and not args.semantic_causal_expert:
        parser.error(
            "--node_change_history_prior requires --semantic_causal_expert"
        )
    if (
        args.node_activity_decoder_input
        in {"future_concat_observed", "future_concat_delta_observed"}
        and not args.node_activity_current_state_context
    ):
        parser.error(
            "observed node-activity decoder inputs require "
            "--node_activity_current_state_context"
        )
    if (
        args.natural_addition_loss_weight < 1.0
        or args.controller_property_magnitude_weight < 0.0
        or args.controller_node_count_weight < 0.0
        or args.controller_activation_ranking_weight < 0.0
        or args.controller_deactivation_ranking_weight < 0.0
        or args.property_history_action_weight < 0.0
        or args.action_world_weight < 0.0
        or args.action_advantage_weight < 0.0
        or args.action_advantage_margin < 0.0
        or args.action_plan_consistency_weight < 0.0
        or args.action_plan_positive_weight_cap < 0.0
        or args.action_decoder_consistency_weight < 0.0
        or args.action_decoder_positive_weight_cap < 0.0
        or args.semantic_pos_weight_scale <= 0.0
        or args.lambda_semantic_distance < 0.0
        or args.changed_semantic_value_weight < 1.0
        or args.changed_node_latent_weight < 1.0
        or args.lambda_property_transition_gate < 0.0
        or args.selection_property_weight < 0.0
        or args.lambda_node_operation_count < 0.0
        or args.lambda_deactivation < 0.0
        or (
            args.action_lambda_deactivation is not None
            and args.action_lambda_deactivation < 0.0
        )
        or args.deactivation_focal_gamma < 0.0
        or args.lambda_deactivation_ranking < 0.0
        or not 0.0 <= args.node_removal_group_calibration_weight <= 1.0
        or args.lambda_activation_ranking < 0.0
        or not 0.0 <= args.dynamic_change_hard_fraction <= 1.0
        or args.dynamic_change_focal_gamma < 0.0
        or args.semantic_hard_pair_weight < 0.0
        or args.semantic_hard_pair_margin < 0.0
        or not torch.isfinite(torch.tensor(args.semantic_history_bias))
        or not torch.isfinite(torch.tensor(args.semantic_property_change_bias))
        or not torch.isfinite(torch.tensor(args.semantic_transition_gate_bias))
        or not torch.isfinite(torch.tensor(args.semantic_volatility_residual_bias))
        or not torch.isfinite(torch.tensor(args.activity_trajectory_supervision_bias))
        or not torch.isfinite(torch.tensor(args.removal_history_supervision_bias))
    ):
        parser.error("action predictive and advantage weights/margin must be non-negative")
    if not 0.0 < args.decision_calibration_tail_fraction <= 1.0:
        parser.error("--decision_calibration_tail_fraction must lie in (0, 1]")
    if not 0.0 <= args.target_encoder_momentum < 1.0:
        parser.error("target_encoder_momentum must lie in [0, 1)")
    if args.action_lambda_deactivation is None:
        args.action_lambda_deactivation = args.lambda_deactivation

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(
        "runtime_cpu="
        f"intraop_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()}",
        flush=True,
    )
    action_mode_overrides = _configured_action_modes(args)
    count_support_operations = _count_support_operations(args.action_count_support_mode)
    count_supervision_operations = _resolved_count_operations(
        args.controller_count_supervision_mode,
        fallback=count_support_operations,
    )
    count_decision_operations = _resolved_count_operations(
        args.controller_count_decision_mode,
        fallback=count_support_operations,
    )
    decoder_consistency_operations = _decoder_consistency_operations(
        args.action_decoder_consistency_operations
    )
    processed = ROOT / (args.processed or DEFAULT_PROCESSED[args.dataset])
    dataset = NodePropertyTransitionDataset(
        processed,
        dataset=args.dataset,
        include_property_observation_mask=True,
        include_observed_degree=args.observed_degree_feature,
        include_controller_history_features=args.controller_history_features,
        include_activity_trajectory_features=args.node_activity_trajectory_features,
        controller_history_window=args.controller_history_window,
        semi_synthetic_node_edits=args.semi_synthetic_node_edits,
        construction_seed=args.construction_seed,
        construction_scale=args.construction_scale,
        construction_rate_floor=args.construction_rate_floor,
        construction_strategy=args.construction_strategy,
        construction_contrastive_addition_negatives=(
            args.construction_contrastive_addition_negatives
        ),
        cache_transitions=args.cache_transitions,
        defer_dense_masking=args.defer_dense_masking,
    )
    if args.semantic_history_cache:
        cache_root = ROOT / "benchmark_artifacts" / "semantic_history_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = (
            f"supportv3_{args.dataset}_s{args.construction_seed}_"
            f"{args.construction_strategy}_scale{args.construction_scale:g}_"
            f"floor{args.construction_rate_floor:g}_"
            f"semi{int(args.semi_synthetic_node_edits)}_"
            f"contrastive{int(args.construction_contrastive_addition_negatives)}_"
            f"topk{args.semantic_topk}"
        )
        cache_path = cache_root / f"{cache_key}.pt"
        lock_path = cache_root / f"{cache_key}.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if cache_path.exists():
                payload = _load_semantic_history_payload(cache_path)
                cache = SemanticHistoryCache()
                cache.__dict__.update(payload)
                required = {
                    "source_node_ids", "source_index", "source_valid",
                    "target_node_ids", "target_index", "target_valid",
                }
                if not required.issubset(cache.__dict__):
                    raise ValueError(f"Incompatible semantic cache: {cache_path}")
                print(f"loaded semantic history cache={cache_path}", flush=True)
            else:
                print(
                    "building semantic history cache (CPU top-k semantics preserved)",
                    flush=True,
                )
                cache = build_semantic_history_cache(
                    dataset, topk=args.semantic_topk
                )
                temporary_path = cache_path.with_suffix(
                    f".tmp.{os.getpid()}.pt"
                )
                torch.save(cache.__dict__, temporary_path)
                os.replace(temporary_path, cache_path)



                del cache
                gc.collect()
                payload = _load_semantic_history_payload(cache_path)
                cache = SemanticHistoryCache()
                cache.__dict__.update(payload)
                print(f"saved semantic history cache={cache_path}", flush=True)
            del payload
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        dataset.semantic_history_cache = cache
    else:
        dataset.semantic_history_cache = None
    semantic_info = fit_semantic_change_threshold(
        dataset, topk=args.semantic_topk, quantile=args.semantic_threshold_quantile
    )
    semantic_threshold = float(semantic_info["threshold"])
    statistics = fit_t2_training_statistics(
        dataset, semantic_threshold=semantic_threshold, semantic_topk=args.semantic_topk
    )
    split_stats = transition_statistics(
        dataset, semantic_threshold=semantic_threshold, topk=args.semantic_topk
    )
    model_kwargs = {
        "input_dim": int(dataset.metadata["model_input_dim"])
        + (2 if args.semantic_history_feature else 0),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "action_dim": int(args.action_dim),
        "latent_distribution": args.latent_distribution,
        "observable_latent_mode": args.observable_latent_mode,
        "node_property_dim": int(dataset.metadata["official_target_dim"]),
        "node_activity_decoder": True,
        "property_decoder_zero_init": True,
        "target_encoder_momentum": float(args.target_encoder_momentum),
        "separate_action_query": True,
        "bounded_action_residual": True,
        "zero_init_action_adapters": True,
        "action_adapter_initial_scale": float(args.action_adapter_initial_scale),
        "action_residual_max_scale": float(args.action_residual_max_scale),
        "action_adapter_squash": True,
        "action_adapter_temperature": float(args.action_adapter_temperature),
        "action_injection_mode": args.action_injection_mode,


        "input_adapter_type": (
            "residual_domain_invariant"
            if args.pretrained_backbone and args.pretrained_transfer_mode == "legacy"
            else "none"
        ),
        "action_plan_decoder": args.action_plan_consistency_weight > 0.0,
        "action_plan_decoder_dim": len(NODE_OPERATIONS),
        "node_operation_count_decoder": args.node_operation_count_decoder,
        "property_transition_gate_decoder": args.property_transition_gate_decoder,
        "node_change_decoder_input": args.node_change_decoder_input,
        "node_activity_decoder_input": args.node_activity_decoder_input,
        "node_activity_observed_dim": (
            2
            + int(args.node_activity_source_visibility_context)
            + (
                int(dataset.metadata["controller_history_feature_dim"])
                if args.node_activity_history_context
                else 0
            )
            + (
                int(dataset.metadata["activity_trajectory_feature_dim"])
                if args.node_activity_trajectory_features
                else 0
            )
            + (
                int(dataset.metadata["model_input_dim"])
                + (2 if args.semantic_history_feature else 0)
                if args.node_activity_full_observed_context
                else 0
            )
            if args.node_activity_current_state_context
            else 0
        ),
        "node_activity_action_dim": (
            len(NODE_OPERATIONS) if args.node_activity_action_context else 0
        ),
        "node_activity_history_dim": (
            int(dataset.metadata["controller_history_feature_dim"])
            + int(dataset.metadata["activity_trajectory_feature_dim"])
            if (args.node_activity_history_gate or args.node_activity_history_prior)
            else 0
        ),
        "node_activity_history_prior": args.node_activity_history_prior,
        "node_activity_history_prior_remove_only": (
            args.node_activity_history_prior_remove_only
        ),
        "node_activity_addition_source_dim": (
            int(dataset.metadata["model_input_dim"])
            + (2 if args.semantic_history_feature else 0)
            if args.node_activity_addition_source_context
            else 0
        ),
        "node_change_source_dim": (
            int(dataset.metadata["model_input_dim"])
            + (2 if args.semantic_history_feature else 0)
            if args.node_change_source_context
            else 0
        ),
        "node_change_causal_dim": (2 if args.semantic_causal_expert else 0),
        "node_change_history_prior": args.node_change_history_prior,
        "detach_node_change_input": args.detach_node_change_input,
        "node_activity_separate_heads": args.node_activity_separate_heads,
        "node_activity_decoder_activation": args.node_activity_decoder_activation,
        **world_model_architecture_kwargs(args),
    }
    model = GraphWorldModel(**model_kwargs).to(device)
    if args.pretrained_backbone:
        summary = load_worldgraph_pretrained_backbone(
            model,
            args.pretrained_backbone,
            map_location="cpu",
            expected_task="T1",
            expected_dataset=args.dataset,
            transfer_state_model=args.pretrained_transfer_state_model,
            backbone_blend=args.pretrained_backbone_blend,
            graph_encoder_blend=args.pretrained_graph_encoder_blend,
            state_model_blend=args.pretrained_state_model_blend,
            latent_predictor_blend=args.pretrained_latent_predictor_blend,
            transfer_mode=args.pretrained_transfer_mode,
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
        semantic_summary = load_worldgraph_pretrained_task_module(
            model.node_change_decoder,
            args.pretrained_backbone,
            source_prefix="semantic_change_decoder",
            blend=args.pretrained_task_blend,
        )
        if semantic_summary is not None:
            print(
                "loaded_pretrained_task_semantic="
                + json.dumps(semantic_summary, sort_keys=True),
                flush=True,
            )


    with torch.random.fork_rng(devices=[]):
        action_encoder = GraphActionEncoder(
            args.latent_dim,
            args.action_dim,
            node_property_dim=int(dataset.metadata["official_target_dim"]),
        ).to(device)
        controllers = _new_controllers(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            property_dim=int(dataset.metadata["official_target_dim"]),
            policy_dim=args.policy_dim,
            controller_remove_observed_degree=args.controller_remove_observed_degree,
            device=device,
            controller_history_features=args.controller_history_features,
            controller_history_feature_dim=int(
                dataset.metadata["controller_history_feature_dim"]
            ),
            controller_addition_source_visibility=(
                args.controller_addition_source_visibility
            ),
        )
        if args.pretrained_backbone:
            for operation, prefix in (
                (GraphOperation.ADD_NODE, "node_add_controller"),
                (GraphOperation.REMOVE_NODE, "node_remove_controller"),
            ):
                task_summary = load_worldgraph_pretrained_task_module(
                    controllers[operation.name.lower()],
                    args.pretrained_backbone,
                    source_prefix=prefix,
                    blend=args.pretrained_task_blend,
                )
                if task_summary is not None:
                    print(
                        f"loaded_pretrained_task_{operation.name.lower()}="
                        + json.dumps(task_summary, sort_keys=True),
                        flush=True,
                    )



        if args.pretrained_backbone:
            activity_summary = load_worldgraph_pretrained_activity_controller(
                controllers[GraphOperation.ADD_NODE.name.lower()],
                args.pretrained_backbone,
                map_location="cpu",
                blend=args.pretrained_controller_blend,
            )
            if activity_summary is not None:
                print(
                    "loaded_pretrained_activity_controller="
                    + json.dumps(activity_summary, sort_keys=True),
                    flush=True,
                )
    world_optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": action_encoder.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        fused=args.fused_adamw,
    )
    controller_optimizer = torch.optim.AdamW(
        controllers.parameters(),
        lr=args.controller_lr,
        weight_decay=args.weight_decay,
        fused=args.fused_adamw,
    )
    action_effect_parameters = _action_effect_parameters(
        model,
        action_encoder,
        include_state_model=args.action_update_state_model,
        include_action_plan=args.action_plan_consistency_weight > 0.0,
        include_operation_count=(
            args.node_operation_count_decoder
            and args.lambda_node_operation_count > 0.0
        ),
        include_observable_decoders=args.action_observable_decoder_update,
        decoder_consistency_operations=(
            decoder_consistency_operations
            if args.action_decoder_consistency_weight > 0.0
            else frozenset()
        ),
    )
    stem = f"t1_node_action_rl_{args.dataset}_{args.graph_encoder_type}_{args.state_model_type}_seed{args.seed}"
    checkpoint_path = _path(args.checkpoint, ROOT / "checkpoints" / "gwm_zero_t1_action_rl" / f"{stem}.pt")
    result_path = _path(args.results, ROOT / "results" / "gwm_zero_t1_action_rl" / f"{stem}.json")
    print(
        f"device={device} dataset={args.dataset} architecture={architecture_label(args)} "
        f"nodes={dataset.metadata['num_nodes']} world_parameters={count_parameters(model)} "
        f"action_encoder_parameters={count_parameters(action_encoder)} controller_parameters={count_parameters(controllers)}",
        flush=True,
    )
    print("semantic_threshold_train=", semantic_info, flush=True)
    print(
        "tensor_shapes=",
        _shape_sanity(
            model, action_encoder, controllers, dataset, device=device,
            semantic_threshold=semantic_threshold, semantic_topk=args.semantic_topk,
            action_mode=args.action_mode, action_mode_overrides=action_mode_overrides,
            action_topk=args.action_topk,
            count_support_operations=count_support_operations,
            count_decision_operations=count_decision_operations,
            use_property_magnitude=args.action_use_property_magnitude,
            property_history_action_weight=args.property_history_action_weight,
            semantic_history_feature=args.semantic_history_feature,
            node_activity_current_state_context=(
                args.node_activity_current_state_context
            ),
            node_activity_source_visibility_context=(
                args.node_activity_source_visibility_context
            ),
            node_activity_full_observed_context=(
                args.node_activity_full_observed_context
            ),
            node_activity_history_context=args.node_activity_history_context,
            node_activity_action_context=args.node_activity_action_context,
        ),
        flush=True,
    )


    model.reset_history()

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    group_sampler = DynamicGroupSampler()
    pending_group_counts = {
        operation: torch.zeros((), device=device) for operation in NODE_OPERATIONS
    }
    train_transition_ids = _chronological_train_transition_ids(dataset)
    prefix_statistics_cache: dict[
        int,
        tuple[
            tuple[int, ...],
            _PastOperationRates,
            OnlineSemanticChangeHistory,
            OnlineSemanticPropertyHistory,
            torch.Tensor,
            int,
        ],
    ] = {}
    static_candidate_cache = (
        _StaticCandidateIndexCache()
        if args.cache_static_action_candidates
        else None
    )
    validation_preparation_cache = (
        _CausalValidationPreparationCache()
        if args.cache_validation_preparation
        else None
    )
    profile_steps = max(0, int(os.environ.get("WORLDGRAPH_PROFILE_STEPS", "0")))
    profile_epoch = int(os.environ.get("WORLDGRAPH_PROFILE_EPOCH", "1"))
    component_timing = os.environ.get("WORLDGRAPH_COMPONENT_TIMING", "0") == "1"
    rl_fine_timing = os.environ.get("WORLDGRAPH_RL_FINE_TIMING", "0") == "1"
    component_stop_epoch = max(
        0, int(os.environ.get("WORLDGRAPH_COMPONENT_STOP_EPOCH", "0"))
    )
    for epoch in range(1, args.epochs + 1):
        component_times: defaultdict[str, float] = defaultdict(float)
        rl_fine_times: dict[str, float] | None = {} if rl_fine_timing else None
        model.train(); action_encoder.train(); controllers.train()
        model.reset_history()
        selected_train_ids, prefix_ids = _training_clip_ids(
            train_transition_ids,
            epoch=epoch,
            max_train_transitions=args.max_train_transitions,
            train_clip_transitions=args.train_clip_transitions,
        )
        action_enabled = bool(epoch > args.action_warmup_epochs)
        (
            hidden,
            rates,
            property_history,
            property_reference_history,
            historical_change_counts,
            history_steps,
        ) = _prepare_training_clip_state(
            model=model,
            action_encoder=action_encoder,
            controllers=controllers,
            dataset=dataset,
            prefix_ids=prefix_ids,
            device=device,
            semantic_threshold=semantic_threshold,
            semantic_topk=args.semantic_topk,
            action_enabled=action_enabled,
            action_mode=args.action_mode,
            action_mode_overrides=action_mode_overrides,
            action_topk=args.action_topk,
            count_support_operations=count_support_operations,
            count_decision_operations=count_decision_operations,
            use_property_magnitude=args.action_use_property_magnitude,
            property_history_action_weight=args.property_history_action_weight,
            semantic_history_feature=args.semantic_history_feature,
            node_activity_current_state_context=(
                args.node_activity_current_state_context
            ),
            node_activity_source_visibility_context=(
                args.node_activity_source_visibility_context
            ),
            node_activity_full_observed_context=(
                args.node_activity_full_observed_context
            ),
            node_activity_history_context=args.node_activity_history_context,
            node_activity_action_context=args.node_activity_action_context,
            history_burn_in=args.clip_history_burn_in,
            static_candidate_cache=static_candidate_cache,
            vectorized_controller_forward=args.vectorized_controller_forward,
            prefix_statistics_cache=prefix_statistics_cache,
        )




        historical_change_counts = historical_change_counts.to(device)
        sums: defaultdict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros((), device=device)
        )
        steps = 0
        profiler = None
        if profile_steps and epoch == profile_epoch:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            )
            profiler.__enter__()
        for transition_id in selected_train_ids:
            component_started_at = time.perf_counter()
            transition_cpu = dataset.transition(transition_id)
            if static_candidate_cache is not None:
                transition_cpu = static_candidate_cache.attach(transition_cpu)
            property_history.observe(
                transition_cpu["property_node_ids_t"],
                transition_cpu["property_observed_t"],
                transition_id=int(transition_cpu["transition_id"]),
            )
            property_reference_history.observe(
                transition_cpu["property_node_ids_t"],
                transition_cpu["property_observed_t"],
            )
            prepared_transition = _with_causal_semantic_history_features(
                transition_cpu,
                property_history,
                enabled=args.semantic_history_feature,
            )
            transition = t2_transition_to_device(prepared_transition, device)
            if component_timing:
                torch.cuda.synchronize(device)
                component_times["data_prepare"] += time.perf_counter() - component_started_at
            hidden_before = hidden
            reuse_controller_forward = bool(
                action_enabled
                and args.reuse_controller_forward
            )
            component_started_at = time.perf_counter()
            z_raw, action, z_t, action_details = _soft_action(
                model=model,
                action_encoder=action_encoder,
                controllers=controllers,
                transition=transition,
                dataset_name=dataset.dataset_name,
                hidden=hidden_before,
                rates=rates,
                enabled=action_enabled,
                action_mode=args.action_mode,
                action_mode_overrides=action_mode_overrides,
                action_topk=args.action_topk,
                count_support_operations=count_support_operations,
                count_decision_operations=count_decision_operations,
                use_property_magnitude=args.action_use_property_magnitude,
                property_history=property_history,
                property_history_action_weight=args.property_history_action_weight,
                controller_gradient=bool(
                    action_enabled and reuse_controller_forward
                ),
                vectorized_controller_forward=args.vectorized_controller_forward,
                detach_action_encoder=bool(reuse_controller_forward),
            )
            if component_timing:
                torch.cuda.synchronize(device)
                component_times["action_proposal"] += time.perf_counter() - component_started_at
            component_started_at = time.perf_counter()
            outputs = model(
                transition["x_t"], transition["edge_index_t"], hidden_before,
                action=(
                    action.detach()
                    if reuse_controller_forward and action is not None
                    else action
                ),
                edge_weight_t=_edge_weight_for_dataset(transition, dataset.dataset_name),
                precomputed_z_t_raw=z_raw,
                decode_node_features=False,
                decode_current_property=(args.lambda_current_semantic > 0.0),
                node_property_ids=transition["semantic_node_ids"],
                current_property_node_ids=transition["property_node_ids_t"],
                node_activity_observed_context=_node_activity_observed_context(
                    transition,
                    enabled=args.node_activity_current_state_context,
                    include_source_visibility=(
                        args.node_activity_source_visibility_context
                    ),
                    include_full_state=args.node_activity_full_observed_context,
                    include_history=args.node_activity_history_context,
                ),
                node_activity_action_context=_node_activity_action_context(
                    action_details,
                    num_nodes=int(transition["x_t"].shape[0]),
                    device=transition["x_t"].device,
                    dtype=transition["x_t"].dtype,
                    enabled=(
                        args.node_activity_action_context and action_enabled
                    ),
                ),
            )
            _attach_causal_property_reference(
                outputs,
                transition,
                property_reference_history,
                mode=args.semantic_reference_mode,
            )
            if component_timing:
                torch.cuda.synchronize(device)
                component_times["action_forward"] += time.perf_counter() - component_started_at
            component_started_at = time.perf_counter()
            world_loss, terms = _world_loss(
                model=model, outputs=outputs, transition=transition,
                transition_cpu=transition_cpu, dataset_name=dataset.dataset_name,
                statistics=statistics, semantic_threshold=semantic_threshold,
                semantic_topk=args.semantic_topk,
                semantic_history_cache=getattr(dataset, "semantic_history_cache", None),
                lambda_activity=args.lambda_activity,
                lambda_semantic_change=args.lambda_semantic_change,
                lambda_semantic_distance=args.lambda_semantic_distance,
                lambda_semantic_value=args.lambda_semantic_value,
                changed_semantic_value_weight=args.changed_semantic_value_weight,
                lambda_current_semantic=args.lambda_current_semantic,
                lambda_latent=args.lambda_latent,
                lambda_latent_cosine=args.lambda_latent_cosine,
                lambda_latent_variance=args.lambda_latent_variance,
                changed_node_latent_weight=args.changed_node_latent_weight,
                semantic_property_mode=args.semantic_property_mode,
                semantic_pos_weight_scale=args.semantic_pos_weight_scale,
                lambda_property_transition_gate=args.lambda_property_transition_gate,
                lambda_node_operation_count=args.lambda_node_operation_count,
                lambda_activation_ranking=args.lambda_activation_ranking,
                lambda_deactivation=args.action_lambda_deactivation,
                deactivation_focal_gamma=args.deactivation_focal_gamma,
                lambda_deactivation_ranking=args.lambda_deactivation_ranking,
                natural_addition_loss_weight=args.natural_addition_loss_weight,
                dynamic_change_hard_fraction=args.dynamic_change_hard_fraction,
                dynamic_change_focal_gamma=args.dynamic_change_focal_gamma,
                semantic_hard_pair_weight=args.semantic_hard_pair_weight,
                semantic_hard_pair_margin=args.semantic_hard_pair_margin,
                activity_trajectory_supervision_bias=args.activity_trajectory_supervision_bias,
                removal_history_supervision_bias=args.removal_history_supervision_bias,
            )
            zero_world_loss = world_loss
            zero_terms = terms
            action_effect_grads: tuple[torch.Tensor | None, ...] | None = None
            action_advantage = world_loss.new_zeros(())
            action_plan_loss = world_loss.new_zeros(())
            action_decoder_loss = world_loss.new_zeros(())
            action_observable = _observable_transition_loss(
                terms,
                node_operation_count_weight=args.lambda_node_operation_count,
            )
            if component_timing:
                torch.cuda.synchronize(device)
                component_times["world_loss"] += time.perf_counter() - component_started_at
            component_started_at = time.perf_counter()
            world_optimizer.zero_grad(set_to_none=True)
            zero_world_loss.backward()
            if action_effect_grads is not None:
                for parameter, gradient in zip(action_effect_parameters, action_effect_grads):
                    if gradient is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = gradient.detach().clone()
                    else:
                        parameter.grad.add_(gradient.detach())
            clip_grad_norm_([*model.parameters(), *action_encoder.parameters()], args.grad_clip)
            world_optimizer.step()
            model.update_target_encoder()
            if component_timing:
                torch.cuda.synchronize(device)
                component_times["backward_optimizer"] += time.perf_counter() - component_started_at

            controller_update_enabled = bool(
                (action_enabled or not args.skip_controller_warmup_update)
                and steps % args.controller_update_interval == 0
            )
            if controller_update_enabled:
                component_started_at = (
                    _fine_timing_start(device)
                    if rl_fine_times is not None else time.perf_counter()
                )
                do_grpo = bool(
                    action_enabled
                    and epoch > args.grpo_warmup_epochs
                    and steps % args.grpo_interval == 0
                )
                if do_grpo:




                    group_count_started = (
                        _fine_timing_start(device)
                        if rl_fine_times is not None else 0.0
                    )
                    pending_values = torch.stack(
                        [pending_group_counts[op] for op in NODE_OPERATIONS]
                    ).detach().cpu().tolist()
                    for operation, count in zip(NODE_OPERATIONS, pending_values):
                        group_sampler.counts[operation] += int(count)
                        pending_group_counts[operation].zero_()
                    _fine_timing_add(
                        rl_fine_times,
                        "dynamic_group_sampling",
                        group_count_started,
                        device,
                    )
                controller_loss, controller_terms, targets = _controller_loss(
                    model=model, action_encoder=action_encoder, controllers=controllers,
                    group_sampler=group_sampler,
                    historical_change_counts=historical_change_counts,
                    history_steps=history_steps,
                    z_raw=z_raw.detach(), z_t=z_t.detach(),
                    hidden=hidden_before.detach(), transition=transition,
                    dataset_name=dataset.dataset_name, semantic_threshold=semantic_threshold,
                    semantic_topk=args.semantic_topk,
                    semantic_history_cache=getattr(dataset, "semantic_history_cache", None),
                    statistics=statistics,
                    rollout_budget=args.grpo_rollout_budget, max_actions=args.max_actions,
                    supervised_weight=args.controller_supervised_weight,
                    grpo_weight=args.grpo_weight, grpo_clip=args.grpo_clip,
                    grpo_kl=args.grpo_kl, entropy_weight=args.entropy_weight,
                    do_grpo=do_grpo,
                    use_property_magnitude=args.action_use_property_magnitude,
                    property_magnitude_weight=args.controller_property_magnitude_weight,
                    node_count_weight=args.controller_node_count_weight,
                    activation_ranking_weight=(
                        args.controller_activation_ranking_weight
                    ),
                    deactivation_ranking_weight=(
                        args.controller_deactivation_ranking_weight
                    ),
                    count_supervision_operations=count_supervision_operations,
                    semantic_pos_weight_scale=args.semantic_pos_weight_scale,
                    node_activity_current_state_context=(
                        args.node_activity_current_state_context
                    ),
                    node_activity_source_visibility_context=(
                        args.node_activity_source_visibility_context
                    ),
                    node_activity_full_observed_context=(
                        args.node_activity_full_observed_context
                    ),
                    node_activity_history_context=(
                        args.node_activity_history_context
                    ),
                    node_activity_action_context=args.node_activity_action_context,
                    natural_addition_loss_weight=args.natural_addition_loss_weight,
                    controller_addition_source_visibility=(
                        args.controller_addition_source_visibility
                    ),
                    controller_addition_source_visibility_balance=(
                        args.controller_addition_source_visibility_balance
                    ),
                    batch_grpo_rollout_forward=args.batch_grpo_rollout_forward,
                    vectorized_controller_forward=(
                        args.vectorized_controller_forward
                    ),
                    precomputed_proposals=(
                        action_details.get("proposals")
                        if reuse_controller_forward
                        else None
                    ),
                    composite_task_reward=args.t1_composite_structure_reward,
                    semantic_property_mode=args.semantic_property_mode,
                    fine_timing=rl_fine_times,
                )
                if args.controller_update_interval > 1:
                    controller_loss = controller_loss * float(
                        args.controller_update_interval
                    )
                rl_backward_started = (
                    _fine_timing_start(device) if rl_fine_times is not None else 0.0
                )
                controller_optimizer.zero_grad(set_to_none=True)
                controller_loss.backward()
                clip_grad_norm_(controllers.parameters(), args.grad_clip)
                controller_optimizer.step()
                _fine_timing_add(
                    rl_fine_times, "rl_backward", rl_backward_started, device
                )
                _fine_timing_add(
                    rl_fine_times, "controller_step_total", component_started_at, device
                )
                if component_timing:
                    torch.cuda.synchronize(device)
                    component_times["controller_grpo"] += time.perf_counter() - component_started_at
            else:


                controller_loss = z_t.new_zeros(())
                controller_terms = {
                    "supervised": z_t.new_zeros(()),
                    "property_magnitude": z_t.new_zeros(()),
                    "node_count": z_t.new_zeros(()),
                    "activation_ranking": z_t.new_zeros(()),
                    "deactivation_ranking": z_t.new_zeros(()),
                    "grpo": z_t.new_zeros(()),
                    "reward": z_t.new_zeros(()),
                }
                targets = {}
                for operation in NODE_OPERATIONS:
                    _candidates, labels, known, full = _operation_targets(
                        transition,
                        operation,
                        semantic_threshold=semantic_threshold,
                        semantic_topk=args.semantic_topk,
                        semantic_history_cache=getattr(
                            dataset, "semantic_history_cache", None
                        ),
                    )
                    targets[operation] = (labels, known, full)
            changed_nodes = torch.zeros_like(historical_change_counts, dtype=torch.bool)
            for operation, (labels, known, full) in targets.items():
                rates.observe(operation, labels, known)
                observed_labels = (
                    labels
                    if operation
                    in {GraphOperation.ADD_NODE, GraphOperation.REMOVE_NODE}
                    else labels[known]
                )
                pending_group_counts[operation].add_(
                    observed_labels.to(dtype=torch.float32).sum().detach()
                )
                changed_nodes |= full.detach().to(
                    device=changed_nodes.device, dtype=torch.bool
                )
            if targets:
                historical_change_counts.add_(changed_nodes.to(torch.float32))
                history_steps += 1
            hidden = outputs["hidden_next"].detach()
            sums["total"] += zero_world_loss.detach()
            sums["action_world"] += world_loss.detach()
            sums["action_advantage"] += action_advantage.detach()
            sums["action_plan"] += action_plan_loss.detach()
            sums["action_decoder"] += action_decoder_loss.detach()
            sums["activity"] += terms["activity"].detach()
            sums["semantic_change"] += terms["semantic_change"].detach()
            sums["semantic_distance"] += terms["semantic_distance"].detach()
            sums["property_transition_gate"] += terms[
                "property_transition_gate"
            ].detach()
            sums["semantic_value"] += terms["semantic_value"].detach()
            sums["operation_count"] += terms["operation_count"].detach()
            sums["activation_ranking"] += terms["activation_ranking"].detach()
            sums["deactivation_ranking"] += terms[
                "deactivation_ranking"
            ].detach()
            sums["latent"] += terms["latent"].detach()
            sums["controller"] += controller_loss.detach()
            sums["controller_supervised"] += controller_terms["supervised"]
            sums["controller_property_magnitude"] += controller_terms["property_magnitude"]
            sums["controller_node_count"] += controller_terms["node_count"]
            sums["controller_activation_ranking"] += controller_terms[
                "activation_ranking"
            ]
            sums["controller_deactivation_ranking"] += controller_terms[
                "deactivation_ranking"
            ]
            sums["controller_grpo"] += controller_terms["grpo"]
            sums["reward"] += controller_terms["reward"]
            sums["action_mass"] += torch.as_tensor(
                action_details["action_mass"], device=device
            ).detach()
            steps += 1
            if profiler is not None:
                profiler.step()
                if steps >= profile_steps:
                    profiler.__exit__(None, None, None)
                    print(profiler.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=30
                    ), flush=True)
                    print(profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=30
                    ), flush=True)
                    profiler = None

        if profiler is not None:
            profiler.__exit__(None, None, None)
            print(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=30
                ),
                flush=True,
            )
            nonzero_stacks: dict[tuple[str, ...], int] = defaultdict(int)
            for event in profiler.events():
                if event.name == "aten::nonzero":
                    nonzero_stacks[tuple(event.stack[-8:])] += 1
            for stack, count in sorted(
                nonzero_stacks.items(), key=lambda item: item[1], reverse=True
            )[:20]:
                print(f"PROFILE_NONZERO count={count}", flush=True)
                print("\n".join(stack), flush=True)
        divisor = max(steps, 1)
        sum_keys = list(sums)
        sum_values = (
            torch.stack([sums[key] for key in sum_keys])
            .div(float(divisor))
            .detach()
            .cpu()
            .tolist()
        )
        train = dict(zip(sum_keys, sum_values))
        train["transitions"] = steps
        train["action_enabled"] = action_enabled
        train["clip_first_transition"] = int(selected_train_ids[0])
        train["clip_last_transition"] = int(selected_train_ids[-1])
        train["prefix_transitions"] = len(prefix_ids)
        if rl_fine_times is not None:
            phase_names = (
                "controller_supervised_update",
                "grpo_rollout",
                "dynamic_group_sampling",
                "structure_aware_reward",
                "rl_backward",
            )
            for name in phase_names:
                rl_fine_times.setdefault(name, 0.0)
            rl_fine_times["controller_step_unattributed"] = (
                rl_fine_times.get("controller_step_total", 0.0)
                - sum(rl_fine_times[name] for name in phase_names)
            )
            train["rl_fine_timing"] = dict(rl_fine_times)
            print(
                f"epoch={epoch:03d} rl_fine_timing="
                + ",".join(
                    f"{name}:{seconds:.3f}s"
                    for name, seconds in rl_fine_times.items()
                ),
                flush=True,
            )
        graph_encoder = model.graph_encoder
        target_graph_encoder = model.target_graph_encoder
        train["topology_cache_hits"] = int(
            getattr(graph_encoder, "_topology_cache_hits", 0)
        ) + int(getattr(target_graph_encoder, "_topology_cache_hits", 0))
        train["topology_cache_misses"] = int(
            getattr(graph_encoder, "_topology_cache_misses", 0)
        ) + int(getattr(target_graph_encoder, "_topology_cache_misses", 0))
        should_validate = epoch % args.val_every == 0 or epoch == args.epochs
        if not should_validate:
            print(
                f"epoch={epoch:03d} total={train.get('total', 0.0):.5f} "
                f"activity={train.get('activity', 0.0):.5f} semantic={train.get('semantic_change', 0.0):.5f} "
                f"semantic_distance={train.get('semantic_distance', 0.0):.5f} "
                f"controller={train.get('controller', 0.0):.5f} "
                f"clip={train['clip_first_transition']}:{train['clip_last_transition']} "
                + "(validation deferred)",
                flush=True,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train": train,
                    "validation": None,
                    "selection_score": None,
                }
            )
            continue



        validation_started_at = time.perf_counter()
        with _preserve_world_rng(device):
            model.eval(); action_encoder.eval(); controllers.eval()


            model.reset_history()
            evaluator = _action_evaluator(
                model=model, action_encoder=action_encoder, controllers=controllers,
                dataset_name=dataset.dataset_name, semantic_threshold=semantic_threshold,
                semantic_topk=args.semantic_topk, enabled=action_enabled,
                action_mode=args.action_mode, action_mode_overrides=action_mode_overrides,
                action_topk=args.action_topk, device=device,
                count_support_operations=count_support_operations,
                count_decision_operations=count_decision_operations,
                use_property_magnitude=args.action_use_property_magnitude,
                property_history_action_weight=args.property_history_action_weight,
                semantic_history_feature=args.semantic_history_feature,
                semantic_history_cache=getattr(dataset, "semantic_history_cache", None),
                semantic_reference_mode=args.semantic_reference_mode,
                semantic_property_mode=args.semantic_property_mode,
                semantic_gate_bias=args.semantic_gate_bias,
                semantic_property_change_bias=(
                    args.semantic_property_change_bias
                ),
                node_activity_current_state_context=(
                    args.node_activity_current_state_context
                ),
                node_activity_source_visibility_context=(
                    args.node_activity_source_visibility_context
                ),
                node_activity_full_observed_context=(
                    args.node_activity_full_observed_context
                ),
                node_activity_history_context=args.node_activity_history_context,
                node_activity_action_context=args.node_activity_action_context,
                node_operation_count_history_window=(
                    args.node_operation_count_history_window
                ),
                node_operation_count_source=args.node_operation_count_source,
                node_activity_score_source=args.node_activity_score_source,
                node_addition_score_source=args.node_addition_score_source,
                node_removal_score_source=args.node_removal_score_source,
                node_removal_history_bias=args.node_removal_history_bias,
                node_removal_group_calibration_mode=(
                    args.node_removal_group_calibration_mode
                ),
                node_removal_group_calibration_weight=(
                    args.node_removal_group_calibration_weight
                ),
                semantic_history_bias=args.semantic_history_bias,
                semantic_transition_gate_bias=args.semantic_transition_gate_bias,
                semantic_volatility_residual_bias=args.semantic_volatility_residual_bias,
                static_candidate_cache=static_candidate_cache,
                validation_preparation_cache=validation_preparation_cache,
                vectorized_controller_forward=args.vectorized_controller_forward,
                decode_node_property=True,
                scoring_split="val",
            )
            with torch.inference_mode():
                validation_raw = evaluate_t2_semantic_sequence(
                    model, dataset, split="val", device=device,
                    semantic_threshold=semantic_threshold, semantic_topk=args.semantic_topk,
                    semantic_property_mode=args.semantic_property_mode,
                    semantic_gate_bias=args.semantic_gate_bias, compute_latent=False,
                    minimal_metrics=False,
                    compute_baseline_diagnostics=args.validation_only,
                    forward_step=evaluator,
                )
        validation_thresholds = select_t2_thresholds(
            validation_raw,
            force_count_names=(
                tuple(
                    name
                    for name, enabled in (
                        ("activation", args.force_node_addition_count_decision),
                        ("deactivation", args.force_node_removal_count_decision),
                    )
                    if enabled
                )
            ),
            source_visibility_threshold_names=(
                ("activation",)
                if args.node_addition_source_visibility_threshold
                else ()
            ),
            calibration_group_tail_fraction=(
                args.decision_calibration_tail_fraction
            ),
        )
        validation = apply_t2_thresholds(validation_raw, validation_thresholds)
        validation_seconds = time.perf_counter() - validation_started_at
        if component_timing:
            component_times["validation"] = validation_seconds
        score = (
            _selection_score(
                validation,
                property_weight=args.selection_property_weight,
                metric=args.selection_metric,
            )
            if action_enabled else None
        )
        print(
            f"epoch={epoch:03d} total={train.get('total', 0.0):.5f} "
            f"activity={train.get('activity', 0.0):.5f} semantic={train.get('semantic_change', 0.0):.5f} "
            f"semantic_distance={train.get('semantic_distance', 0.0):.5f} "
            f"count={train.get('operation_count', 0.0):.5f} "
            f"add_rank={train.get('activation_ranking', 0.0):.5f} "
            f"remove_rank={train.get('deactivation_ranking', 0.0):.5f} "
            f"controller={train.get('controller', 0.0):.5f} reward={train.get('reward', 0.0):.5f} "
            f"plan={train.get('action_plan', 0.0):.5f} decoder={train.get('action_decoder', 0.0):.5f} "
            f"action_mass={train.get('action_mass', 0.0):.5f} "
            f"| "
            + f"cache_hit={train['topology_cache_hits']} "
            f"cache_miss={train['topology_cache_misses']} | "
            f"val_add_ap={validation['node_addition']['auprc']!s} "
            f"val_remove_ap={validation['node_removal']['auprc']!s} "
            f"val_semantic_ap={validation['semantic_change']['auprc']!s}",
            flush=True,
        )
        if component_timing:
            print(
                f"epoch={epoch:03d} component_timing="
                + ",".join(
                    f"{name}:{component_times[name]:.3f}s"
                    for name in (
                        "data_prepare", "action_proposal", "zero_action_forward",
                        "action_forward", "world_loss", "backward_optimizer",
                        "controller_grpo", "validation",
                    )
                ),
                flush=True,
            )
        history.append(
            {
                "epoch": epoch,
                "train": train,
                "validation": validation,
                "selection_score": score,
            }
        )
        if score is not None and score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "action_encoder_state": action_encoder.state_dict(),
                    "controllers_state": controllers.state_dict(),
                    "model_kwargs": model_kwargs,
                    "training_args": vars(args),
                    "semantic_threshold": semantic_info,
                    "statistics": statistics,
                    "validation_thresholds": validation_thresholds,
                    "best_epoch": best_epoch,
                    "best_validation_score": best_score,
                },
                checkpoint_path,
            )
        elif score is not None:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
                break
        if component_stop_epoch and epoch >= component_stop_epoch:
            print(f"component_timing_stop epoch={epoch}", flush=True)
            break

    if component_stop_epoch:
        print(
            f"component_timing_complete epochs={len(history)}",
            flush=True,
        )
        return

    if best_epoch == 0:
        raise RuntimeError("No action-conditioned validation checkpoint was selected.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    action_encoder.load_state_dict(checkpoint["action_encoder_state"])
    controllers.load_state_dict(checkpoint["controllers_state"])
    model.eval(); action_encoder.eval(); controllers.eval()
    best_action_enabled = bool(
        best_epoch > args.action_warmup_epochs
    )
    if args.validation_only:
        best_validation = next(
            record["validation"]
            for record in history
            if record["epoch"] == best_epoch and record["validation"] is not None
        )
        result = {
            "model": "WorldGraph",
            "task": "node_addition_removal_and_semantic_feature_change",
            "dataset": args.dataset,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "validation_only": True,
            "semantic_threshold": semantic_info,
            "training_statistics": statistics,
            "transition_statistics": split_stats,
            "history": history,
            "validation": best_validation,
            "benchmark_protocol": protocol_metadata,
        }
        save_json(result, result_path)
        print(f"saved_checkpoint={_display_path(checkpoint_path)}", flush=True)
        print(f"saved_results={_display_path(result_path)}", flush=True)
        print(
            "validation="
            f" add_f1={best_validation['node_addition']['f1']:.4f}"
            f" remove_f1={best_validation['node_removal']['f1']:.4f}"
            f" semantic_f1={best_validation['semantic_change']['f1']:.4f}",
            flush=True,
        )
        return






    evaluation_removal_history_bias = float(args.node_removal_history_bias)
    evaluation_removal_group_calibration_weight = float(
        args.node_removal_group_calibration_weight
    )
    evaluation_semantic_gate_bias = float(args.semantic_gate_bias)
    evaluation_semantic_property_change_bias = float(
        args.semantic_property_change_bias
    )

    def _deduplicated_candidates(
        candidates: list[float] | None, default: float
    ) -> list[float]:
        values = [float(default)]
        if candidates is not None:
            values.extend(float(value) for value in candidates)
        unique: list[float] = []
        for value in values:
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError("Readout-calibration candidates must be finite.")
            if value not in unique:
                unique.append(value)
        return unique

    def _evaluate_final_readout(
        *,
        removal_history_bias: float,
        removal_group_calibration_weight: float,
        semantic_gate_bias: float,
        semantic_property_change_bias: float,
    ) -> tuple[dict[str, Any], dict[str, dict[str, float | None]]]:


        with _preserve_world_rng(device):
            seed_everything(int(args.seed) + 30_000)
            model.reset_history()
            calibration_evaluator = _action_evaluator(
                model=model, action_encoder=action_encoder, controllers=controllers,
                dataset_name=dataset.dataset_name, semantic_threshold=semantic_threshold,
                semantic_topk=args.semantic_topk, enabled=best_action_enabled,
                action_mode=args.action_mode, action_mode_overrides=action_mode_overrides,
                action_topk=args.action_topk, device=device,
                count_support_operations=count_support_operations,
                count_decision_operations=count_decision_operations,
                use_property_magnitude=args.action_use_property_magnitude,
                property_history_action_weight=args.property_history_action_weight,
                semantic_history_feature=args.semantic_history_feature,
                semantic_history_cache=getattr(dataset, "semantic_history_cache", None),
                semantic_reference_mode=args.semantic_reference_mode,
                semantic_property_mode=args.semantic_property_mode,
                semantic_gate_bias=semantic_gate_bias,
                semantic_property_change_bias=semantic_property_change_bias,
                node_activity_current_state_context=(
                    args.node_activity_current_state_context
                ),
                node_activity_source_visibility_context=(
                    args.node_activity_source_visibility_context
                ),
                node_activity_full_observed_context=(
                    args.node_activity_full_observed_context
                ),
                node_activity_history_context=args.node_activity_history_context,
                node_activity_action_context=args.node_activity_action_context,
                node_operation_count_history_window=(
                    args.node_operation_count_history_window
                ),
                node_operation_count_source=args.node_operation_count_source,
                node_activity_score_source=args.node_activity_score_source,
                node_addition_score_source=args.node_addition_score_source,
                node_removal_score_source=args.node_removal_score_source,
                node_removal_history_bias=removal_history_bias,
                node_removal_group_calibration_mode=(
                    args.node_removal_group_calibration_mode
                ),
                node_removal_group_calibration_weight=(
                    removal_group_calibration_weight
                ),
                semantic_history_bias=args.semantic_history_bias,
                semantic_transition_gate_bias=args.semantic_transition_gate_bias,
                semantic_volatility_residual_bias=args.semantic_volatility_residual_bias,
                static_candidate_cache=static_candidate_cache,
                validation_preparation_cache=validation_preparation_cache,
                vectorized_controller_forward=args.vectorized_controller_forward,
                scoring_split="val",
            )
            raw = evaluate_t2_semantic_sequence(
                model, dataset, split="val", device=device,
                semantic_threshold=semantic_threshold, semantic_topk=args.semantic_topk,
                semantic_property_mode=args.semantic_property_mode,
                semantic_gate_bias=semantic_gate_bias,
                compute_latent=False, forward_step=calibration_evaluator,
            )
        thresholds = select_t2_thresholds(
            raw,
            force_count_names=(
                tuple(
                    name
                    for name, enabled in (
                        ("activation", args.force_node_addition_count_decision),
                        ("deactivation", args.force_node_removal_count_decision),
                    )
                    if enabled
                )
            ),
            source_visibility_threshold_names=(
                ("activation",)
                if args.node_addition_source_visibility_threshold
                else ()
            ),
            calibration_group_tail_fraction=(
                args.decision_calibration_tail_fraction
            ),
        )
        return apply_t2_thresholds(raw, thresholds), thresholds

    readout_calibration: dict[str, Any] = {
        "split": "validation",
        "node_removal_history_bias": {
            "base": float(args.node_removal_history_bias),
            "selected": evaluation_removal_history_bias,
            "selection": "fixed",
        },
        "node_removal_group_calibration": {
            "mode": args.node_removal_group_calibration_mode,
            "base": float(args.node_removal_group_calibration_weight),
            "selected": evaluation_removal_group_calibration_weight,
            "selection": "fixed",
        },
        "semantic_gate_bias": {
            "base": float(args.semantic_gate_bias),
            "selected": evaluation_semantic_gate_bias,
            "selection": "fixed",
        },
        "semantic_property_change_bias": {
            "base": float(args.semantic_property_change_bias),
            "selected": evaluation_semantic_property_change_bias,
            "selection": "fixed",
        },
    }
    uses_readout_grid = bool(
        args.semantic_gate_bias_grid
        or args.semantic_property_change_bias_grid
        or args.node_removal_history_bias_grid
        or args.node_removal_group_calibration_weight_grid
    )
    if args.semantic_gate_bias_grid:
        gate_candidates = _deduplicated_candidates(
            args.semantic_gate_bias_grid, evaluation_semantic_gate_bias
        )
        best_gate_key: tuple[float, float, float] | None = None
        for candidate in gate_candidates:
            candidate_validation, _unused = _evaluate_final_readout(
                removal_history_bias=evaluation_removal_history_bias,
                removal_group_calibration_weight=(
                    evaluation_removal_group_calibration_weight
                ),
                semantic_gate_bias=candidate,
                semantic_property_change_bias=(
                    evaluation_semantic_property_change_bias
                ),
            )
            prediction = candidate_validation["semantic_prediction"]
            changed_ndcg = prediction.get("changed_ndcg_at_10")
            ordinary_ndcg = prediction.get("official_ndcg")
            key = (
                float("-inf") if changed_ndcg is None else float(changed_ndcg),
                float("-inf") if ordinary_ndcg is None else float(ordinary_ndcg),
                -abs(float(candidate) - float(args.semantic_gate_bias)),
            )
            if best_gate_key is None or key > best_gate_key:
                best_gate_key = key
                evaluation_semantic_gate_bias = float(candidate)
        readout_calibration["semantic_gate_bias"] = {
            "base": float(args.semantic_gate_bias),
            "candidates": gate_candidates,
            "selected": evaluation_semantic_gate_bias,
            "selection": "maximize validation Changed-NDCG@10, then NDCG@10",
        }
    if args.semantic_property_change_bias_grid:
        property_change_candidates = _deduplicated_candidates(
            args.semantic_property_change_bias_grid,
            evaluation_semantic_property_change_bias,
        )
        best_property_change_key: tuple[float, float, float] | None = None
        for candidate in property_change_candidates:
            candidate_validation, _unused = _evaluate_final_readout(
                removal_history_bias=evaluation_removal_history_bias,
                removal_group_calibration_weight=(
                    evaluation_removal_group_calibration_weight
                ),
                semantic_gate_bias=evaluation_semantic_gate_bias,
                semantic_property_change_bias=candidate,
            )
            semantic = candidate_validation["semantic_change"]
            f1 = semantic.get("f1")
            auprc = semantic.get("auprc")
            key = (
                float("-inf") if f1 is None else float(f1),
                float("-inf") if auprc is None else float(auprc),
                -abs(
                    float(candidate)
                    - float(args.semantic_property_change_bias)
                ),
            )
            if (
                best_property_change_key is None
                or key > best_property_change_key
            ):
                best_property_change_key = key
                evaluation_semantic_property_change_bias = float(candidate)
        readout_calibration["semantic_property_change_bias"] = {
            "base": float(args.semantic_property_change_bias),
            "candidates": property_change_candidates,
            "selected": evaluation_semantic_property_change_bias,
            "selection": "maximize validation semantic-change F1, then AUPRC",
        }
    if args.node_removal_history_bias_grid:
        removal_candidates = _deduplicated_candidates(
            args.node_removal_history_bias_grid, evaluation_removal_history_bias
        )
        best_removal_key: tuple[float, float, float] | None = None
        for candidate in removal_candidates:
            candidate_validation, _unused = _evaluate_final_readout(
                removal_history_bias=candidate,
                removal_group_calibration_weight=(
                    evaluation_removal_group_calibration_weight
                ),
                semantic_gate_bias=evaluation_semantic_gate_bias,
                semantic_property_change_bias=(
                    evaluation_semantic_property_change_bias
                ),
            )
            removal = candidate_validation["node_removal"]
            f1 = removal.get("f1")
            auprc = removal.get("auprc")
            key = (
                float("-inf") if f1 is None else float(f1),
                float("-inf") if auprc is None else float(auprc),
                -abs(float(candidate) - float(args.node_removal_history_bias)),
            )
            if best_removal_key is None or key > best_removal_key:
                best_removal_key = key
                evaluation_removal_history_bias = float(candidate)
        readout_calibration["node_removal_history_bias"] = {
            "base": float(args.node_removal_history_bias),
            "candidates": removal_candidates,
            "selected": evaluation_removal_history_bias,
            "selection": "maximize validation removal F1, then AUPRC",
        }
    if args.node_removal_group_calibration_weight_grid:
        group_calibration_candidates = _deduplicated_candidates(
            args.node_removal_group_calibration_weight_grid,
            evaluation_removal_group_calibration_weight,
        )
        if any(
            not 0.0 <= candidate <= 1.0
            for candidate in group_calibration_candidates
        ):
            raise ValueError(
                "Removal group-calibration candidates must lie in [0, 1]."
            )
        best_group_calibration_key: tuple[float, float, float] | None = None
        for candidate in group_calibration_candidates:
            candidate_validation, _unused = _evaluate_final_readout(
                removal_history_bias=evaluation_removal_history_bias,
                removal_group_calibration_weight=candidate,
                semantic_gate_bias=evaluation_semantic_gate_bias,
                semantic_property_change_bias=(
                    evaluation_semantic_property_change_bias
                ),
            )
            removal = candidate_validation["node_removal"]
            f1 = removal.get("f1")
            auprc = removal.get("auprc")
            key = (
                float("-inf") if f1 is None else float(f1),
                float("-inf") if auprc is None else float(auprc),
                -abs(
                    float(candidate)
                    - float(args.node_removal_group_calibration_weight)
                ),
            )
            if (
                best_group_calibration_key is None
                or key > best_group_calibration_key
            ):
                best_group_calibration_key = key
                evaluation_removal_group_calibration_weight = float(candidate)
        readout_calibration["node_removal_group_calibration"] = {
            "mode": args.node_removal_group_calibration_mode,
            "base": float(args.node_removal_group_calibration_weight),
            "candidates": group_calibration_candidates,
            "selected": evaluation_removal_group_calibration_weight,
            "selection": "maximize validation removal F1, then AUPRC",
        }
    final_validation: dict[str, Any] | None = None



    final_validation_thresholds = copy.deepcopy(checkpoint["validation_thresholds"])
    if uses_readout_grid:
        final_validation, calibrated_thresholds = _evaluate_final_readout(
            removal_history_bias=evaluation_removal_history_bias,
            removal_group_calibration_weight=(
                evaluation_removal_group_calibration_weight
            ),
            semantic_gate_bias=evaluation_semantic_gate_bias,
            semantic_property_change_bias=(
                evaluation_semantic_property_change_bias
            ),
        )
        if args.node_removal_history_bias_grid:
            final_validation_thresholds["deactivation"] = copy.deepcopy(
                calibrated_thresholds["deactivation"]
            )
        if args.node_removal_group_calibration_weight_grid:
            final_validation_thresholds["deactivation"] = copy.deepcopy(
                calibrated_thresholds["deactivation"]
            )
        if args.semantic_property_change_bias_grid:
            for name in ("semantic", "semantic_stable_activity"):
                final_validation_thresholds[name] = copy.deepcopy(
                    calibrated_thresholds[name]
                )
        print(
            "readout_calibration="
            f" removal_history_bias={evaluation_removal_history_bias:.4g}"
            f" removal_group_calibration="
            f"{args.node_removal_group_calibration_mode}:"
            f"{evaluation_removal_group_calibration_weight:.4g}"
            f" semantic_gate_bias={evaluation_semantic_gate_bias:.4g}"
            f" semantic_property_change_bias="
            f"{evaluation_semantic_property_change_bias:.4g}"
            f" val_remove_f1={float(final_validation['node_removal']['f1']):.4f}"
            f" val_semantic_f1={float(final_validation['semantic_change']['f1']):.4f}"
            f" val_changed_ndcg={float(final_validation['semantic_prediction']['changed_ndcg_at_10']):.4f}",
            flush=True,
        )
    evaluator = _action_evaluator(
        model=model, action_encoder=action_encoder, controllers=controllers,
        dataset_name=dataset.dataset_name, semantic_threshold=semantic_threshold,
        semantic_topk=args.semantic_topk, enabled=best_action_enabled,
        action_mode=args.action_mode, action_mode_overrides=action_mode_overrides,
        action_topk=args.action_topk, device=device,
        count_support_operations=count_support_operations,
        count_decision_operations=count_decision_operations,
        use_property_magnitude=args.action_use_property_magnitude,
        property_history_action_weight=args.property_history_action_weight,
        semantic_history_feature=args.semantic_history_feature,
        semantic_history_cache=getattr(dataset, "semantic_history_cache", None),
        semantic_reference_mode=args.semantic_reference_mode,
        semantic_property_mode=args.semantic_property_mode,
        semantic_gate_bias=evaluation_semantic_gate_bias,
        semantic_property_change_bias=(
            evaluation_semantic_property_change_bias
        ),
        node_activity_current_state_context=(
            args.node_activity_current_state_context
        ),
                node_activity_source_visibility_context=(
                    args.node_activity_source_visibility_context
                ),
                node_activity_full_observed_context=(
                    args.node_activity_full_observed_context
                ),
                node_activity_history_context=args.node_activity_history_context,
                node_activity_action_context=args.node_activity_action_context,
        node_operation_count_history_window=(
            args.node_operation_count_history_window
        ),
        node_operation_count_source=args.node_operation_count_source,
        node_activity_score_source=args.node_activity_score_source,
        node_addition_score_source=args.node_addition_score_source,
        node_removal_score_source=args.node_removal_score_source,
        node_removal_history_bias=evaluation_removal_history_bias,
        node_removal_group_calibration_mode=(
            args.node_removal_group_calibration_mode
        ),
        node_removal_group_calibration_weight=(
            evaluation_removal_group_calibration_weight
        ),
        semantic_history_bias=args.semantic_history_bias,
        semantic_transition_gate_bias=args.semantic_transition_gate_bias,
        semantic_volatility_residual_bias=args.semantic_volatility_residual_bias,
        static_candidate_cache=static_candidate_cache,
        validation_preparation_cache=validation_preparation_cache,
        vectorized_controller_forward=args.vectorized_controller_forward,
        scoring_split="test",
    )
    model.reset_history()
    test_raw = evaluate_t2_semantic_sequence(
        model, dataset, split="test", device=device,
        semantic_threshold=semantic_threshold, semantic_topk=args.semantic_topk,
        semantic_property_mode=args.semantic_property_mode,
        semantic_gate_bias=evaluation_semantic_gate_bias,
        decision_thresholds=None, compute_latent=args.evaluate_latent, forward_step=evaluator,
    )
    test = apply_t2_thresholds(test_raw, final_validation_thresholds)
    result = {
        "model": "WorldGraph",
        "task": "node_addition_removal_and_semantic_feature_change",
        "dataset": args.dataset,
        "architecture": architecture_label(args),
        "observed_state": "g_t=(V_t,E_t,X_t,Y_t)",
        "action": {
            "enabled": True,
            "operations": [operation.name for operation in NODE_OPERATIONS],
            "proposal": "three past-only operation-specific Controller heads",
            "state_module_input": "summed node-wise GraphActionEncoder action tokens",
            "action_updates_shared_state_model": bool(args.action_update_state_model),
            "count_roles": {
                "action_support": [
                    op.name for op in NODE_OPERATIONS if op in count_support_operations
                ],
                "supervision": [
                    op.name for op in NODE_OPERATIONS if op in count_supervision_operations
                ],
                "evaluation_decision": [
                    op.name for op in NODE_OPERATIONS if op in count_decision_operations
                ],
            },
            "future_target_in_action_input": False,
            "controller_supervision": "released target labels after the action proposal",
            "grpo": "group-relative rewards from action-conditioned future-latent decoders",
            "composite_task_reward": bool(args.t1_composite_structure_reward),
            "best_checkpoint_action_enabled": best_action_enabled,
            "node_activity_score_source": args.node_activity_score_source,
            "node_addition_score_source": args.node_addition_score_source,
            "node_removal_score_source": args.node_removal_score_source,
            "node_removal_history_bias": evaluation_removal_history_bias,
            "node_removal_group_calibration_mode": (
                args.node_removal_group_calibration_mode
            ),
            "node_removal_group_calibration_weight": (
                evaluation_removal_group_calibration_weight
            ),
            "node_activity_action_context": bool(args.node_activity_action_context),
            "count_history_window": int(args.node_operation_count_history_window),
            "count_source": args.node_operation_count_source,
            "force_node_removal_count_decision": bool(
                args.force_node_removal_count_decision
            ),
            "force_node_addition_count_decision": bool(
                args.force_node_addition_count_decision
            ),
        },
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "latent_diagnostics": bool(args.evaluate_latent),
        "semantic_threshold": semantic_info,
        "semantic_property_readout": {
            "mode": args.semantic_property_mode,
            "gate_bias": evaluation_semantic_gate_bias,
            "property_change_bias": evaluation_semantic_property_change_bias,
            "reference": args.semantic_reference_mode,
            "reference_information": (
                "released Y_(v,t)"
                if args.semantic_reference_mode == "current"
                else "per-node mean of released Y_(v,<=t)"
            ),
        },
        "training_statistics": statistics,
        "transition_statistics": split_stats,
        "parameter_count": {
            "world": count_parameters(model),
            "action_encoder": count_parameters(action_encoder),
            "controller": count_parameters(controllers),
        },
        "history": history,
        "readout_calibration": readout_calibration,
        "final_readout_validation": final_validation,
        "test": test,
        "benchmark_protocol": protocol_metadata,
    }
    save_json(result, result_path)
    print(f"saved_checkpoint={_display_path(checkpoint_path)}", flush=True)
    print(f"saved_results={_display_path(result_path)}", flush=True)



    print(
        f"[TEST-MEAN/node/{args.dataset}/n=1] "
        f"Add. F1={float(test['node_addition']['f1']):.4f} "
        f"| Rem. F1={float(test['node_removal']['f1']):.4f} "
        f"| Sem. F1={float(test['semantic_change']['f1']):.4f} "
        f"| NDCG@10={float(test['semantic_prediction']['official_ndcg']):.4f} "
        f"| Changed-NDCG@10={float(test['semantic_prediction']['changed_ndcg_at_10']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
