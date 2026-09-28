
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .tgbn_genre_dataset import TGBNGenreTransitionDataset
from .tgbn_reddit_dataset import TGBNRedditTransitionDataset
from .tgbn_trade_dataset import TGBNTradeTransitionDataset
from .transition_cache import (
    DEFAULT_TRANSITION_CACHE_BYTES,
    TransitionRecordCache,
)


NodePropertyDatasetName = Literal["trade", "genre", "reddit"]


def incident_activity_mask(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    active = torch.zeros(int(num_nodes), dtype=torch.bool, device=edge_index.device)
    if edge_index.numel():
        nodes = edge_index.reshape(-1)
        if int(nodes.min()) < 0 or int(nodes.max()) >= int(num_nodes):
            raise ValueError("edge_index contains an ID outside the node universe")



        active[nodes] = True
    return active


def support_topk_jaccard_distance(
    current: torch.Tensor,
    following: torch.Tensor,
    *,
    topk: int = 10,
) -> torch.Tensor:
    if current.ndim != 2 or following.ndim != 2 or current.shape != following.shape:
        raise ValueError("current and following must be equal [rows, dimensions] matrices")
    if topk < 1:
        raise ValueError("topk must be positive")
    if current.shape[0] == 0:
        return current.new_empty((0,))
    k = min(int(topk), int(current.shape[1]))
    current_value, current_index = current.topk(k, dim=1)
    following_value, following_index = following.topk(k, dim=1)
    current_valid = current_value.gt(0)
    following_valid = following_value.gt(0)
    intersection = (
        (current_index.unsqueeze(2) == following_index.unsqueeze(1))
        & current_valid.unsqueeze(2)
        & following_valid.unsqueeze(1)
    ).any(dim=2).sum(dim=1)
    union = current_valid.sum(dim=1) + following_valid.sum(dim=1) - intersection
    return torch.where(
        union.gt(0),
        1.0 - intersection.to(torch.float32) / union.to(torch.float32),
        torch.zeros_like(union, dtype=torch.float32),
    )


def semantic_change_labels(
    current: torch.Tensor,
    following: torch.Tensor,
    *,
    threshold: float,
    topk: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = support_topk_jaccard_distance(current, following, topk=topk)
    return distance.gt(float(threshold)), distance


def mask_graph_nodes(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    node_mask = node_mask.to(device=edge_index.device, dtype=torch.bool)
    if node_mask.ndim != 1:
        raise ValueError("node_mask must be one-dimensional")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    keep = ~(node_mask.index_select(0, edge_index[0]) | node_mask.index_select(0, edge_index[1]))
    masked_weight = None if edge_weight is None else edge_weight[keep]
    return edge_index[:, keep], masked_weight


def select_constructed_node_edits(
    shared_mask: torch.Tensor,
    *,
    addition_count: int,
    removal_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shared_mask.ndim != 1:
        raise ValueError("shared_mask must be one-dimensional")
    if addition_count < 0 or removal_count < 0:
        raise ValueError("constructed edit counts must be non-negative")
    candidates = shared_mask.detach().cpu().to(torch.bool).nonzero(as_tuple=False).flatten()
    total = min(int(addition_count) + int(removal_count), int(candidates.numel()))
    add_count = min(int(addition_count), total)
    remove_count = min(int(removal_count), total - add_count)
    addition = torch.zeros_like(shared_mask, dtype=torch.bool, device="cpu")
    removal = torch.zeros_like(shared_mask, dtype=torch.bool, device="cpu")
    if total:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        selected = candidates[torch.randperm(candidates.numel(), generator=generator)[:total]]
        addition[selected[:add_count]] = True
        removal[selected[add_count : add_count + remove_count]] = True
    return addition.to(shared_mask.device), removal.to(shared_mask.device)


def select_ranked_constructed_node_edits(
    shared_mask: torch.Tensor,
    addition_priority: torch.Tensor,
    removal_priority: torch.Tensor,
    *,
    addition_count: int,
    removal_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if addition_priority.shape != shared_mask.shape or removal_priority.shape != shared_mask.shape:
        raise ValueError("priority vectors must have the same shape as shared_mask")
    candidates = shared_mask.detach().cpu().to(torch.bool).nonzero(as_tuple=False).flatten()
    addition = torch.zeros_like(shared_mask, dtype=torch.bool, device="cpu")
    removal = torch.zeros_like(shared_mask, dtype=torch.bool, device="cpu")
    if candidates.numel() == 0:
        return addition.to(shared_mask.device), removal.to(shared_mask.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    jitter = torch.rand(candidates.numel(), generator=generator) * 1e-4
    add_count = min(max(int(addition_count), 0), int(candidates.numel()))
    add_order = torch.argsort(
        addition_priority.detach().cpu().float().index_select(0, candidates) + jitter
    )
    added_nodes = candidates[add_order[:add_count]]
    addition[added_nodes] = True

    remaining = candidates[~addition.index_select(0, candidates)]
    remove_count = min(max(int(removal_count), 0), int(remaining.numel()))
    if remove_count:
        removal_jitter = torch.rand(remaining.numel(), generator=generator) * 1e-4
        remove_order = torch.argsort(
            removal_priority.detach().cpu().float().index_select(0, remaining)
            + removal_jitter
        )
        removal[remaining[remove_order[:remove_count]]] = True
    return addition.to(shared_mask.device), removal.to(shared_mask.device)


def select_contrastive_addition_non_edits(
    shared_mask: torch.Tensor,
    addition_mask: torch.Tensor,
    removal_mask: torch.Tensor,
    addition_priority: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    if not (
        shared_mask.shape
        == addition_mask.shape
        == removal_mask.shape
        == addition_priority.shape
    ):
        raise ValueError("contrastive construction tensors must have equal shapes")
    available = (
        shared_mask.detach().cpu().to(torch.bool)
        & ~addition_mask.detach().cpu().to(torch.bool)
        & ~removal_mask.detach().cpu().to(torch.bool)
    ).nonzero(as_tuple=False).flatten()
    selected_mask = torch.zeros_like(shared_mask, dtype=torch.bool, device="cpu")
    count = min(max(int(count), 0), int(available.numel()))
    if count == 0:
        return selected_mask.to(shared_mask.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 73_919)
    values = addition_priority.detach().cpu().float().index_select(0, available)


    values = values + torch.rand(available.numel(), generator=generator) * 1e-4
    selected = available[torch.argsort(values, descending=True)[:count]]
    selected_mask[selected] = True
    return selected_mask.to(shared_mask.device)


def select_temporal_stratified_constructed_node_edits(
    shared_mask: torch.Tensor,
    addition_priority: torch.Tensor,
    removal_priority: torch.Tensor,
    history_activity: torch.Tensor,
    *,
    addition_count: int,
    removal_count: int,
    seed: int,
    num_strata: int = 3,
    sample_pool_fraction: float | None = None,
    removal_history_activity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_values = [addition_priority, removal_priority, history_activity]
    if removal_history_activity is not None:
        all_values.append(removal_history_activity)
    if any(value.shape != shared_mask.shape for value in all_values):
        raise ValueError("all priority vectors must have the same shape as shared_mask")
    if num_strata < 1:
        raise ValueError("num_strata must be positive")
    if sample_pool_fraction is not None and not 0.0 < sample_pool_fraction <= 1.0:
        raise ValueError("sample_pool_fraction must lie in (0, 1]")

    candidates = shared_mask.detach().cpu().to(torch.bool).nonzero(as_tuple=False).flatten()
    addition = torch.zeros_like(shared_mask, dtype=torch.bool, device="cpu")
    removal = torch.zeros_like(shared_mask, dtype=torch.bool, device="cpu")
    if candidates.numel() == 0:
        return addition.to(shared_mask.device), removal.to(shared_mask.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def _stratified_select(
        available: torch.Tensor,
        priority: torch.Tensor,
        count: int,
        activity_source: torch.Tensor,
    ) -> torch.Tensor:
        count = min(max(int(count), 0), int(available.numel()))
        if count == 0 or available.numel() == 0:
            return available.new_empty((0,))




        activity = activity_source.detach().cpu().float().index_select(0, available)
        activity = activity + torch.rand(available.numel(), generator=generator) * 1e-4
        ordered = available[torch.argsort(activity)]
        parts = [part for part in torch.tensor_split(ordered, min(num_strata, int(ordered.numel()))) if part.numel()]




        strongest: list[tuple[float, int]] = []
        for position, part in enumerate(parts):
            values = priority.detach().cpu().float().index_select(0, part)
            strongest.append((float(values.min().item()), position))
        strongest.sort(key=lambda item: item[0])
        allocation = [0] * len(parts)
        for _, position in strongest[:count]:
            allocation[position] += 1
        remaining = count - sum(allocation)
        while remaining > 0:


            choices: list[tuple[float, int]] = []
            for position, part in enumerate(parts):
                if allocation[position] >= int(part.numel()):
                    continue
                values = priority.detach().cpu().float().index_select(0, part)
                choices.append((float(values.kthvalue(allocation[position] + 1).values.item()), position))
            if not choices:
                break
            _, position = min(choices, key=lambda item: item[0])
            allocation[position] += 1
            remaining -= 1

        selected: list[torch.Tensor] = []
        for part, take in zip(parts, allocation):
            if take <= 0:
                continue
            values = priority.detach().cpu().float().index_select(0, part)
            values = values + torch.rand(part.numel(), generator=generator) * 1e-4
            order = torch.argsort(values)
            if sample_pool_fraction is None:
                selected.append(part[order[:take]])
            else:
                pool_size = min(
                    int(part.numel()),
                    max(int(take), int(np.ceil(float(part.numel()) * sample_pool_fraction))),
                )
                pool = part[order[:pool_size]]
                selected.append(
                    pool[torch.randperm(pool.numel(), generator=generator)[:take]]
                )
        return torch.cat(selected) if selected else available.new_empty((0,))

    added_nodes = _stratified_select(
        candidates, addition_priority, addition_count, history_activity
    )
    if added_nodes.numel():
        addition[added_nodes] = True
    remaining = candidates[~addition.index_select(0, candidates)]
    removed_nodes = _stratified_select(
        remaining,
        removal_priority,
        removal_count,
        history_activity if removal_history_activity is None else removal_history_activity,
    )
    if removed_nodes.numel():
        removal[removed_nodes] = True
    return addition.to(shared_mask.device), removal.to(shared_mask.device)


def incident_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel():
        ones = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
        degree.index_add_(0, edge_index[0], ones)
        degree.index_add_(0, edge_index[1], ones)
    return degree


def incident_weight(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
) -> torch.Tensor:
    if edge_weight is None:
        return incident_degree(edge_index, num_nodes)
    if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.shape[1]:
        raise ValueError("edge_weight must align with edge_index columns")
    strength = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel():
        weights = edge_weight.to(device=edge_index.device, dtype=torch.float32).abs()
        strength.index_add_(0, edge_index[0], weights)
        strength.index_add_(0, edge_index[1], weights)
    return strength


def append_observed_membership_features(
    x: torch.Tensor,
    active: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    include_degree: bool,
    degree: torch.Tensor | None = None,
) -> torch.Tensor:
    features = [x, active.to(dtype=x.dtype).unsqueeze(-1)]
    if include_degree:
        if degree is None:
            degree = incident_degree(edge_index, int(x.shape[0]))
        degree = torch.log1p(degree)
        features.append((degree / degree.max().clamp_min(1.0)).to(x.dtype).unsqueeze(-1))
    return torch.cat(features, dim=-1)


def _zero_rows(value: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    if bool(node_mask.any()):
        result[node_mask.to(result.device)] = 0
    return result


def _filter_property_rows(
    transition: dict[str, Any], node_mask: torch.Tensor, *, source: bool
) -> None:
    ids_key = "property_node_ids_t" if source else "property_node_ids"
    values_key = "property_observed_t" if source else "property_target"
    ids = transition[ids_key].long()
    keep = ~node_mask.to(ids.device).index_select(0, ids)
    transition[ids_key] = ids[keep]
    transition[values_key] = transition[values_key][keep]
    if not source:
        for key in ("property_current_target", "property_current_observed_mask"):
            if key in transition:
                transition[key] = transition[key][keep]


class NodePropertyTransitionDataset(Sequence[dict[str, Any]]):

    _BASE_TYPES = {
        "trade": TGBNTradeTransitionDataset,
        "genre": TGBNGenreTransitionDataset,
        "reddit": TGBNRedditTransitionDataset,
    }
    _CONTROLLER_HISTORY_DIM = 4
    _ACTIVITY_TRAJECTORY_DIM = 4

    def __init__(
        self,
        source: str | Path | dict[str, Any],
        *,
        dataset: NodePropertyDatasetName,
        split: str | None = None,
        include_property_observation_mask: bool = True,
        include_observed_degree: bool = False,
        include_controller_history_features: bool = False,
        include_activity_trajectory_features: bool = False,
        controller_history_window: int = 8,
        semi_synthetic_node_edits: bool = False,
        construction_seed: int = 1,
        construction_scale: float = 1.0,
        construction_rate_floor: float = 0.0,
        construction_strategy: str = "past_observable",
        construction_contrastive_addition_negatives: bool = False,
        cache_transitions: bool = True,
        transition_cache_max_bytes: int = DEFAULT_TRANSITION_CACHE_BYTES,
        defer_dense_masking: bool = False,
    ) -> None:
        if dataset not in self._BASE_TYPES:
            raise ValueError(f"Unsupported T2 semantic dataset {dataset!r}")
        self.dataset_name: NodePropertyDatasetName = dataset
        self.base = self._BASE_TYPES[dataset](
            source,
            split=split,
            include_property_observation_mask=include_property_observation_mask,
        )
        if construction_scale < 0:
            raise ValueError("construction_scale must be non-negative")
        if not 0.0 <= construction_rate_floor <= 1.0:
            raise ValueError("construction_rate_floor must lie in [0, 1]")
        if controller_history_window < 1:
            raise ValueError("controller_history_window must be positive")
        if construction_strategy not in {
            "past_observable",
            "temporal_trend",
            "temporal_stratified",
            "temporal_stratified_sampled",
            "long_horizon_stratified",
            "long_horizon_weighted_stratified",
            "long_horizon_hybrid_stratified",
            "random",
        }:
            raise ValueError(
                "construction_strategy must be past_observable, temporal_trend, "
                "temporal_stratified, temporal_stratified_sampled, "
                "long_horizon_stratified, long_horizon_weighted_stratified, "
                "long_horizon_hybrid_stratified, or random"
            )
        self.semi_synthetic_node_edits = bool(semi_synthetic_node_edits)
        self.include_observed_degree = bool(include_observed_degree)
        self.include_controller_history_features = bool(
            include_controller_history_features
        )
        self.include_activity_trajectory_features = bool(
            include_activity_trajectory_features
        )
        self.controller_history_window = int(controller_history_window)
        self._controller_history_cache: dict[int, torch.Tensor] = {}
        self._activity_trajectory_cache: dict[int, torch.Tensor] = {}





        self._constructed_mask_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._activity_mask_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}



        self._observed_degree_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}



        self._masked_edge_cache: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor | None,
            ],
        ] = {}
        self.construction_seed = int(construction_seed)
        self.construction_scale = float(construction_scale)
        self.construction_rate_floor = float(construction_rate_floor)
        self.construction_strategy = str(construction_strategy)
        self.construction_contrastive_addition_negatives = bool(
            construction_contrastive_addition_negatives
        )




        self._node_edit_eligible = torch.ones(
            int(self.base.metadata["num_nodes"]), dtype=torch.bool
        )
        if self.dataset_name in {"genre", "reddit"}:
            coordinate_nodes = int(self.base.metadata["official_target_dim"])
            self._node_edit_eligible[:coordinate_nodes] = False





        self.defer_dense_masking = bool(defer_dense_masking)






        self.cache_transitions = bool(cache_transitions)
        self._transition_cache = (
            TransitionRecordCache(int(transition_cache_max_bytes))
            if self.cache_transitions
            else None
        )
        construction = self._fit_construction_rates()
        construction["construction_rate_floor"] = self.construction_rate_floor
        construction["effective_addition_rate"] = max(
            float(construction["addition_rate"]) * self.construction_scale,
            self.construction_rate_floor,
        )
        construction["effective_removal_rate"] = max(
            float(construction["removal_rate"]) * self.construction_scale,
            self.construction_rate_floor,
        )
        self.metadata = dict(self.base.metadata)
        self.metadata.update(
            {
                "node_level_task": "node_addition_removal_and_semantic_feature_change",
                "t2_task": "node_semantic_transition",
                "t2_node_membership_definition": (
                    "m_(v,t)=1 iff v is incident to an observed edge in E_t; "
                    "inactive graph membership is not inferred from a missing official property row"
                ),
                "t2_semantic_state_definition": (
                    "official current property Y_t for the same node, with semantic "
                    "supervision only on consecutive released rows"
                ),
                "t2_input_activity_mask": True,
                "t2_input_observed_degree": self.include_observed_degree,
                "controller_history_features": self.include_controller_history_features,
                "controller_history_window": self.controller_history_window,
                "controller_history_feature_dim": self._CONTROLLER_HISTORY_DIM
                if self.include_controller_history_features
                else 0,
                "activity_trajectory_features": (
                    self.include_activity_trajectory_features
                ),
                "activity_trajectory_feature_dim": self._ACTIVITY_TRAJECTORY_DIM
                if self.include_activity_trajectory_features
                else 0,
                "node_edit_protocol": (
                    "natural incident-edge membership changes plus deterministic one-sided "
                    "masking of shared nodes; construction rates fitted on train only"
                    if self.semi_synthetic_node_edits
                    else "natural incident-edge membership changes only"
                ),
                "semi_synthetic_node_edits": self.semi_synthetic_node_edits,
                "construction_seed": self.construction_seed,
                "construction_scale": self.construction_scale,
                "construction_rate_floor": self.construction_rate_floor,
                "construction_strategy": self.construction_strategy,
                "construction_contrastive_addition_negatives": (
                    self.construction_contrastive_addition_negatives
                ),
                "node_edit_candidate_scope": (
                    "dynamic entity nodes only"
                    if self.dataset_name in {"genre", "reddit"}
                    else "all graph nodes"
                ),
                "defer_dense_masking": self.defer_dense_masking,
                "construction_train_statistics": construction,
                "model_input_dim": int(self.base.metadata["model_input_dim"])
                + 1
                + int(self.include_observed_degree),
            }
        )

    def _fit_construction_rates(self) -> dict[str, float | int]:
        num_nodes = int(self.base.metadata["num_nodes"])
        additions = removals = rows = transitions = 0
        eligible = self._node_edit_eligible
        for transition_id in range(int(self.base.transition_count)):
            current = self.base.snapshots[transition_id]
            following = self.base.snapshots[transition_id + 1]
            target_split = str(following["split"])
            if target_split != "train":
                if transitions:
                    break
                continue
            active_t = incident_activity_mask(current["edge_index"], num_nodes)
            active_next = incident_activity_mask(following["edge_index"], num_nodes)
            additions += int((((~active_t) & active_next) & eligible).sum())
            removals += int(((active_t & (~active_next)) & eligible).sum())
            rows += int(eligible.sum())
            transitions += 1
        if transitions == 0:
            raise ValueError("No chronological training transitions for node-edit calibration")
        return {
            "fit_split": "train",
            "train_transitions": int(transitions),
            "train_node_rows": int(rows),
            "natural_additions": int(additions),
            "natural_removals": int(removals),
            "addition_rate": float(additions / rows),
            "removal_rate": float(removals / rows),
        }

    def _activity_masks(
        self, transition_id: int, transition: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_activity_mask_cache", None)
        if cache is None:
            cache = {}
            self._activity_mask_cache = cache
        cached = cache.get(int(transition_id))
        if cached is not None:
            return cached
        num_nodes = int(self.metadata["num_nodes"])
        masks = (
            incident_activity_mask(transition["edge_index_t"], num_nodes),
            incident_activity_mask(transition["edge_index_next"], num_nodes),
        )
        cache[int(transition_id)] = masks
        return masks

    def _observed_degrees(
        self,
        transition_id: int,
        edge_index_t: torch.Tensor,
        edge_index_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_observed_degree_cache", None)
        if cache is None:
            cache = {}
            self._observed_degree_cache = cache
        cached = cache.get(int(transition_id))
        if cached is not None:
            return cached
        num_nodes = int(self.metadata["num_nodes"])
        degrees = (
            incident_degree(edge_index_t, num_nodes),
            incident_degree(edge_index_next, num_nodes),
        )
        cache[int(transition_id)] = degrees
        return degrees

    def _masked_edges(
        self,
        transition_id: int,
        edge_index_t: torch.Tensor,
        edge_weight_t: torch.Tensor | None,
        edge_index_next: torch.Tensor,
        edge_weight_next: torch.Tensor | None,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        cache = getattr(self, "_masked_edge_cache", None)
        if cache is None:
            cache = {}
            self._masked_edge_cache = cache
        cached = cache.get(int(transition_id))
        if cached is not None:
            return cached
        source = (
            (edge_index_t, edge_weight_t)
            if not bool(source_mask.any())
            else mask_graph_nodes(edge_index_t, edge_weight_t, source_mask)
        )
        target = (
            (edge_index_next, edge_weight_next)
            if not bool(target_mask.any())
            else mask_graph_nodes(edge_index_next, edge_weight_next, target_mask)
        )
        result = (source[0], source[1], target[0], target[1])
        cache[int(transition_id)] = result
        return result

    def _controller_history_features(self, transition_id: int) -> torch.Tensor | None:

        if not bool(getattr(self, "include_controller_history_features", False)):
            return None
        cache = getattr(self, "_controller_history_cache", None)
        if cache is None:
            cache = {}
            self._controller_history_cache = cache
        cached = cache.get(int(transition_id))
        if cached is not None:
            return cached

        num_nodes = int(self.base.metadata["num_nodes"])
        history_start = max(
            0, int(transition_id) - int(getattr(self, "controller_history_window", 8))
        )
        strengths: list[torch.Tensor] = []
        activity: list[torch.Tensor] = []
        for snapshot_id in range(history_start, int(transition_id)):
            snapshot = self.base.snapshots[snapshot_id]
            strength = torch.log1p(
                incident_weight(
                    snapshot["edge_index"],
                    snapshot.get("edge_weight"),
                    num_nodes,
                ).clamp_min(0.0)
            )
            strengths.append(strength)
            activity.append(
                incident_activity_mask(snapshot["edge_index"], num_nodes).to(
                    dtype=torch.float32
                )
            )

        if not strengths:
            result = torch.zeros(
                (num_nodes, self._CONTROLLER_HISTORY_DIM), dtype=torch.float32
            )
        else:
            history_strength = torch.stack(strengths, dim=0)
            mean_strength = history_strength.mean(dim=0)
            latest_strength = history_strength[-1]
            if int(history_strength.shape[0]) >= 2:
                midpoint = max(1, int(history_strength.shape[0]) // 2)
                trend = (
                    history_strength[midpoint:].mean(dim=0)
                    - history_strength[:midpoint].mean(dim=0)
                )
            else:
                trend = torch.zeros_like(mean_strength)
            scale = history_strength.amax().clamp_min(1.0)
            active_fraction = torch.stack(activity, dim=0).mean(dim=0)
            result = torch.stack(
                [
                    mean_strength / scale,
                    latest_strength / scale,
                    trend / scale,
                    active_fraction,
                ],
                dim=-1,
            ).to(dtype=torch.float32)
        cache[int(transition_id)] = result
        return result

    def _activity_trajectory_features(
        self, transition_id: int
    ) -> torch.Tensor | None:

        if not bool(
            getattr(self, "include_activity_trajectory_features", False)
        ):
            return None
        cache = getattr(self, "_activity_trajectory_cache", None)
        if cache is None:
            cache = {}
            self._activity_trajectory_cache = cache
        cached = cache.get(int(transition_id))
        if cached is not None:
            return cached

        num_nodes = int(self.base.metadata["num_nodes"])
        history_start = max(
            0,
            int(transition_id)
            - int(getattr(self, "controller_history_window", 8)),
        )
        weighted_rows: list[torch.Tensor] = []
        degree_rows: list[torch.Tensor] = []
        for snapshot_id in range(history_start, int(transition_id)):
            snapshot = self.base.snapshots[snapshot_id]
            weighted_rows.append(
                torch.log1p(
                    incident_weight(
                        snapshot["edge_index"],
                        snapshot.get("edge_weight"),
                        num_nodes,
                    ).clamp_min(0.0)
                )
            )
            degree_rows.append(
                incident_degree(snapshot["edge_index"], num_nodes).to(
                    dtype=torch.float32
                )
            )

        def summarize(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if not rows:
                zeros = torch.zeros(num_nodes, dtype=torch.float32)
                return zeros, zeros
            values = torch.stack(rows, dim=0).to(dtype=torch.float32)
            mean = values.mean(dim=0)
            normalized_mean = mean / mean.amax().clamp_min(1.0)
            if int(values.shape[0]) < 4:
                relative_trend = torch.zeros_like(mean)
            else:
                midpoint = int(values.shape[0]) // 2
                early = values[:midpoint].mean(dim=0)
                recent = values[midpoint:].mean(dim=0)
                relative_trend = (recent - early) / (early + 1.0)
                relative_trend = relative_trend.clamp(-8.0, 8.0) / 8.0
            return normalized_mean, relative_trend

        weighted_mean, weighted_trend = summarize(weighted_rows)
        degree_mean, degree_trend = summarize(degree_rows)
        result = torch.stack(
            [weighted_mean, weighted_trend, degree_mean, degree_trend], dim=-1
        ).to(dtype=torch.float32)
        cache[int(transition_id)] = result
        return result

    def _constructed_masks(
        self,
        transition_id: int,
        active_t: torch.Tensor,
        active_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_constructed_mask_cache", None)
        if cache is None:
            cache = {}
            self._constructed_mask_cache = cache
        cached = cache.get(int(transition_id))
        if cached is not None:
            return cached

        def remember(
            masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cache[int(transition_id)] = masks
            return masks

        if not self.semi_synthetic_node_edits:
            return remember(
                (
                    torch.zeros_like(active_t),
                    torch.zeros_like(active_t),
                    torch.zeros_like(active_t),
                )
            )




        shared = active_t & active_next & self._node_edit_eligible
        shared_count = int(shared.sum())
        statistics = self.metadata["construction_train_statistics"]
        addition_rate = max(
            float(statistics["addition_rate"]) * self.construction_scale,
            self.construction_rate_floor,
        )
        removal_rate = max(
            float(statistics["removal_rate"]) * self.construction_scale,
            self.construction_rate_floor,
        )
        addition_count = round(shared_count * addition_rate)
        removal_count = round(shared_count * removal_rate)
        seed = self.construction_seed + int(transition_id) * 1_000_003
        if getattr(self, "construction_strategy", "past_observable") == "random":
            addition, removal = select_constructed_node_edits(
                shared,
                addition_count=addition_count,
                removal_count=removal_count,
                seed=seed,
            )
            if not bool(
                getattr(self, "construction_contrastive_addition_negatives", False)
            ):
                return remember((addition, removal, torch.zeros_like(active_t)))
            available = shared & ~addition & ~removal
            contrastive, _unused = select_constructed_node_edits(
                available,
                addition_count=addition_count,
                removal_count=0,
                seed=seed + 73_919,
            )
            return remember((addition, removal, contrastive))









        long_horizon = self.construction_strategy in {
            "long_horizon_stratified",
            "long_horizon_weighted_stratified",
            "long_horizon_hybrid_stratified",
        }
        weighted_long_horizon = (
            self.construction_strategy
            in {"long_horizon_weighted_stratified", "long_horizon_hybrid_stratified"}
        )
        hybrid_long_horizon = (
            self.construction_strategy == "long_horizon_hybrid_stratified"
        )
        history_start = max(
            0,
            int(transition_id)
            - (8 if long_horizon else 3),
        )
        if weighted_long_horizon:
            history_degrees = [
                torch.log1p(
                    incident_weight(
                        self.base.snapshots[index]["edge_index"],
                        self.base.snapshots[index].get("edge_weight"),
                        active_t.numel(),
                    )
                )
                for index in range(history_start, int(transition_id))
            ]
        else:
            history_degrees = [
                incident_degree(self.base.snapshots[index]["edge_index"], active_t.numel())
                for index in range(history_start, int(transition_id))
            ]
        removal_history_degrees = (
            [
                incident_degree(
                    self.base.snapshots[index]["edge_index"], active_t.numel()
                )
                for index in range(history_start, int(transition_id))
            ]
            if hybrid_long_horizon
            else history_degrees
        )
        history_mean = (
            torch.stack(history_degrees).mean(dim=0)
            if history_degrees
            else torch.zeros_like(active_t, dtype=torch.float32)
        )






        if (
            long_horizon and len(history_degrees) >= 4
        ):
            midpoint = len(history_degrees) // 2
            early_mean = torch.stack(history_degrees[:midpoint]).mean(dim=0)
            recent_mean = torch.stack(history_degrees[midpoint:]).mean(dim=0)


            long_horizon_trend = (recent_mean - early_mean) / (early_mean + 1.0)
            addition_priority = -long_horizon_trend
        elif self.construction_strategy in {
            "temporal_trend", "temporal_stratified", "temporal_stratified_sampled"
        } and history_degrees:
            recent_degree = history_degrees[-1]
            addition_priority = -(
                recent_degree - history_mean + 0.05 * recent_degree
            )
        else:
            addition_priority = history_mean
        current_degree = incident_degree(
            self.base.snapshots[int(transition_id)]["edge_index"], active_t.numel()
        )
        removal_history_mean = history_mean
        if (
            long_horizon and len(history_degrees) >= 4
        ):


            if hybrid_long_horizon:
                midpoint = len(removal_history_degrees) // 2
                early_degree = torch.stack(removal_history_degrees[:midpoint]).mean(dim=0)
                recent_degree = torch.stack(removal_history_degrees[midpoint:]).mean(dim=0)
                removal_priority = (recent_degree - early_degree) / (early_degree + 1.0)
                removal_history_mean = torch.stack(removal_history_degrees).mean(dim=0)
            else:
                removal_priority = long_horizon_trend
        elif self.construction_strategy in {
            "temporal_trend", "temporal_stratified", "temporal_stratified_sampled"
        } and history_degrees:




            removal_priority = current_degree - history_mean + 0.05 * current_degree
        else:
            removal_priority = current_degree
        if self.construction_strategy in {
            "temporal_stratified",
            "temporal_stratified_sampled",
            "long_horizon_stratified",
            "long_horizon_weighted_stratified",
            "long_horizon_hybrid_stratified",
        }:
            addition, removal = select_temporal_stratified_constructed_node_edits(
                shared,
                addition_priority,
                removal_priority,
                history_mean,
                addition_count=addition_count,
                removal_count=removal_count,
                seed=seed,
                removal_history_activity=(
                    removal_history_mean if hybrid_long_horizon else None
                ),
                sample_pool_fraction=(
                    0.5
                    if self.construction_strategy == "temporal_stratified_sampled"
                    else None
                ),
            )
        else:
            addition, removal = select_ranked_constructed_node_edits(
                shared,
                addition_priority,
                removal_priority,
                addition_count=addition_count,
                removal_count=removal_count,
                seed=seed,
            )
        contrastive = (
            select_contrastive_addition_non_edits(
                shared,
                addition,
                removal,
                addition_priority,
                count=addition_count,
                seed=seed,
            )
            if bool(getattr(self, "construction_contrastive_addition_negatives", False))
            else torch.zeros_like(active_t)
        )
        return remember((addition, removal, contrastive))

    def __len__(self) -> int:
        return len(self.base)

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        return self.base.snapshots

    def _augment(self, transition: dict[str, Any]) -> dict[str, Any]:
        result = dict(transition)
        controller_history_features = self._controller_history_features(
            int(transition["transition_id"])
        )
        activity_trajectory_features = self._activity_trajectory_features(
            int(transition["transition_id"])
        )
        natural_active_t, natural_active_next = self._activity_masks(
            int(transition["transition_id"]), transition
        )
        (
            constructed_addition,
            constructed_removal,
            constructed_addition_non_edit,
        ) = self._constructed_masks(
            int(transition["transition_id"]), natural_active_t, natural_active_next
        )

        source_mask = constructed_addition | constructed_addition_non_edit
        target_mask = constructed_removal | constructed_addition_non_edit
        (
            result["edge_index_t"],
            result["edge_weight_t"],
            result["edge_index_next"],
            result["edge_weight_next"],
        ) = self._masked_edges(
            int(transition["transition_id"]),
            result["edge_index_t"],
            result.get("edge_weight_t"),
            result["edge_index_next"],
            result.get("edge_weight_next"),
            source_mask,
            target_mask,
        )
        if bool(source_mask.any()):
            if bool(getattr(self, "defer_dense_masking", False)):
                result["dense_source_mask"] = source_mask
            else:
                for key in ("x_t", "x_t_raw", "property_state_t", "property_observation_mask_t"):
                    if key in result:
                        result[key] = _zero_rows(result[key], source_mask)
            _filter_property_rows(result, source_mask, source=True)
            if "property_current_target" in result:
                target_ids = result["property_node_ids"].long()
                target_rows = source_mask.index_select(0, target_ids)
                result["property_current_target"] = result["property_current_target"].clone()
                result["property_current_target"][target_rows] = 0
                result["property_current_observed_mask"] = (
                    result["property_current_observed_mask"].to(torch.bool) & (~target_rows)
                )

        if bool(target_mask.any()):
            if bool(getattr(self, "defer_dense_masking", False)):
                result["dense_target_mask"] = target_mask
            else:
                for key in ("x_next", "x_next_raw", "property_observation_mask_next"):
                    if key in result:
                        result[key] = _zero_rows(result[key], target_mask)
            _filter_property_rows(result, target_mask, source=False)

        active_t = natural_active_t & (~source_mask)
        active_next = natural_active_next & (~target_mask)
        observed_degree_t = observed_degree_next = None
        if bool(getattr(self, "include_observed_degree", False)):
            observed_degree_t, observed_degree_next = self._observed_degrees(
                int(transition["transition_id"]),
                result["edge_index_t"],
                result["edge_index_next"],
            )
        property_ids = result["property_node_ids"].long()
        comparable = result["property_current_observed_mask"].to(torch.bool)
        if property_ids.numel():
            comparable = comparable & (~source_mask.index_select(0, property_ids))
        if self.dataset_name == "trade":
            current_property = result["property_state_t"].index_select(0, property_ids)
        else:
            current_property = result["property_current_target"]
        if bool(getattr(self, "defer_dense_masking", False)) and property_ids.numel():
            hidden_property_rows = source_mask.index_select(0, property_ids)
            if bool(hidden_property_rows.any()):
                current_property = current_property.clone()
                current_property[hidden_property_rows] = 0
        result.update(
            {


                "x_t": append_observed_membership_features(
                    result["x_t"],
                    active_t,
                    result["edge_index_t"],
                    include_degree=bool(getattr(self, "include_observed_degree", False)),
                    degree=observed_degree_t,
                ),
                "x_next": append_observed_membership_features(
                    result["x_next"],
                    active_next,
                    result["edge_index_next"],
                    include_degree=bool(getattr(self, "include_observed_degree", False)),
                    degree=observed_degree_next,
                ),
                "node_active_t": active_t,
                "node_active_next": active_next,
                "node_edit_eligible_mask": self._node_edit_eligible,
                "node_added": ((~active_t) & active_next) & self._node_edit_eligible,
                "node_removed": (active_t & (~active_next)) & self._node_edit_eligible,
                "node_activated": ((~active_t) & active_next) & self._node_edit_eligible,
                "node_deactivated": (active_t & (~active_next)) & self._node_edit_eligible,
                "node_added_natural": ((~natural_active_t) & natural_active_next)
                & self._node_edit_eligible,
                "node_removed_natural": (natural_active_t & (~natural_active_next))
                & self._node_edit_eligible,
                "node_added_constructed": constructed_addition,
                "node_removed_constructed": constructed_removal,
                "node_addition_constructed_non_edit": constructed_addition_non_edit,
                "semantic_node_ids": property_ids,
                "semantic_current": current_property,
                "semantic_next": result["property_target"],
                "semantic_comparable_mask": comparable,
                "semantic_stable_activity_mask": (
                    active_t.index_select(0, property_ids)
                    & active_next.index_select(0, property_ids)
                ),
            }
        )
        if observed_degree_t is not None:
            result["node_observed_degree_t"] = observed_degree_t
        if controller_history_features is not None:
            result["controller_history_features"] = controller_history_features
        if activity_trajectory_features is not None:
            result["activity_trajectory_features"] = activity_trajectory_features
        if bool(getattr(self, "defer_dense_masking", False)):
            result["defer_dense_features"] = True
        if "dense_source_mask" in result:



            result["dense_source_feature_dim"] = int(result["x_t"].shape[1])
        if "dense_target_mask" in result:
            result["dense_target_feature_dim"] = int(result["x_next"].shape[1])
        return result

    def transition(self, transition_id: int) -> dict[str, Any]:
        transition_id = int(transition_id)
        cache_key = ("transition", transition_id)
        if self._transition_cache is not None:
            cached = self._transition_cache.get(cache_key)
            if cached is not None:
                return cached
        result = self._augment(self.base.transition(transition_id))
        if self._transition_cache is not None:
            return self._transition_cache.put(cache_key, result)
        return result

    def forecast_transition(self, transition_id: int) -> dict[str, Any]:
        transition_id = int(transition_id)
        if not hasattr(self.base, "forecast_transition"):
            return self.transition(transition_id)

        cache_key = ("forecast", transition_id)
        if self._transition_cache is not None:
            cached = self._transition_cache.get(cache_key)
            if cached is not None:
                return cached

        transition = dict(self.base.forecast_transition(transition_id))
        controller_history_features = self._controller_history_features(
            int(transition["transition_id"])
        )
        activity_trajectory_features = self._activity_trajectory_features(
            int(transition["transition_id"])
        )
        following = self.base.snapshots[int(transition_id) + 1]
        transition["edge_index_next"] = following["edge_index"]
        transition["edge_weight_next"] = following.get("edge_weight")

        natural_active_t, natural_active_next = self._activity_masks(
            int(transition["transition_id"]), transition
        )
        (
            constructed_addition,
            constructed_removal,
            constructed_addition_non_edit,
        ) = self._constructed_masks(
            int(transition["transition_id"]), natural_active_t, natural_active_next
        )

        source_mask = constructed_addition | constructed_addition_non_edit
        target_mask = constructed_removal | constructed_addition_non_edit
        (
            transition["edge_index_t"],
            transition["edge_weight_t"],
            transition["edge_index_next"],
            transition["edge_weight_next"],
        ) = self._masked_edges(
            int(transition["transition_id"]),
            transition["edge_index_t"],
            transition.get("edge_weight_t"),
            transition["edge_index_next"],
            transition.get("edge_weight_next"),
            source_mask,
            target_mask,
        )
        if bool(source_mask.any()):
            if bool(getattr(self, "defer_dense_masking", False)):
                transition["dense_source_mask"] = source_mask
            else:
                for key in ("x_t", "x_t_raw", "property_state_t", "property_observation_mask_t"):
                    if key in transition:
                        transition[key] = _zero_rows(transition[key], source_mask)
            _filter_property_rows(transition, source_mask, source=True)
            if "property_current_target" in transition:
                target_ids = transition["property_node_ids"].long()
                target_rows = source_mask.index_select(0, target_ids)
                transition["property_current_target"] = transition[
                    "property_current_target"
                ].clone()
                transition["property_current_target"][target_rows] = 0
                transition["property_current_observed_mask"] = (
                    transition["property_current_observed_mask"].to(torch.bool)
                    & (~target_rows)
                )

        if bool(target_mask.any()):
            _filter_property_rows(transition, target_mask, source=False)

        active_t = natural_active_t & (~source_mask)
        active_next = natural_active_next & (~target_mask)
        observed_degree_t = observed_degree_next = None
        if bool(getattr(self, "include_observed_degree", False)):
            observed_degree_t, observed_degree_next = self._observed_degrees(
                int(transition["transition_id"]),
                transition["edge_index_t"],
                transition["edge_index_next"],
            )
        property_ids = transition["property_node_ids"].long()
        comparable = transition["property_current_observed_mask"].to(torch.bool)
        if property_ids.numel():
            comparable = comparable & (~source_mask.index_select(0, property_ids))
        current_property = (
            transition["property_state_t"].index_select(0, property_ids)
            if self.dataset_name == "trade"
            else transition["property_current_target"]
        )
        if bool(getattr(self, "defer_dense_masking", False)) and property_ids.numel():
            hidden_property_rows = source_mask.index_select(0, property_ids)
            if bool(hidden_property_rows.any()):
                current_property = current_property.clone()
                current_property[hidden_property_rows] = 0
        transition.update(
            {
                "x_t": append_observed_membership_features(
                    transition["x_t"],
                    active_t,
                    transition["edge_index_t"],
                    include_degree=bool(getattr(self, "include_observed_degree", False)),
                    degree=observed_degree_t,
                ),
                "node_active_t": active_t,
                "node_active_next": active_next,
                "node_edit_eligible_mask": self._node_edit_eligible,
                "node_added": ((~active_t) & active_next) & self._node_edit_eligible,
                "node_removed": (active_t & (~active_next)) & self._node_edit_eligible,
                "node_activated": ((~active_t) & active_next) & self._node_edit_eligible,
                "node_deactivated": (active_t & (~active_next)) & self._node_edit_eligible,
                "node_added_natural": ((~natural_active_t) & natural_active_next)
                & self._node_edit_eligible,
                "node_removed_natural": (natural_active_t & (~natural_active_next))
                & self._node_edit_eligible,
                "node_added_constructed": constructed_addition,
                "node_removed_constructed": constructed_removal,
                "node_addition_constructed_non_edit": constructed_addition_non_edit,
                "semantic_node_ids": property_ids,
                "semantic_current": current_property,
                "semantic_next": transition["property_target"],
                "semantic_comparable_mask": comparable,
                "semantic_stable_activity_mask": (
                    active_t.index_select(0, property_ids)
                    & active_next.index_select(0, property_ids)
                ),
            }
        )
        if observed_degree_t is not None:
            transition["node_observed_degree_t"] = observed_degree_t
        if controller_history_features is not None:
            transition["controller_history_features"] = controller_history_features
        if activity_trajectory_features is not None:
            transition["activity_trajectory_features"] = activity_trajectory_features
        if bool(getattr(self, "defer_dense_masking", False)):
            transition["defer_dense_features"] = True
        if "dense_source_mask" in transition:
            transition["dense_source_feature_dim"] = int(transition["x_t"].shape[1])
        if self._transition_cache is not None:
            return self._transition_cache.put(cache_key, transition)
        return transition

    def __getitem__(self, item: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(len(self)))]


        base_transition = self.base[item]
        transition_id = int(base_transition["transition_id"])
        cache_key = ("transition", transition_id)
        if self._transition_cache is not None:
            cached = self._transition_cache.get(cache_key)
            if cached is not None:
                return cached
        result = self._augment(base_transition)
        if self._transition_cache is not None:
            return self._transition_cache.put(cache_key, result)
        return result

    def iter_all(self) -> Iterator[dict[str, Any]]:
        for transition_id in range(int(self.base.transition_count)):
            yield self.transition(transition_id)

    def iter_forecast_all(self) -> Iterator[dict[str, Any]]:
        if not hasattr(self.base, "forecast_transition"):
            yield from self.iter_all()
            return
        for transition_id in range(int(self.base.transition_count)):
            yield self.forecast_transition(transition_id)


def fit_semantic_change_threshold(
    dataset: NodePropertyTransitionDataset,
    *,
    topk: int = 10,
    quantile: float = 0.60,
) -> dict[str, float | int]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    distances: list[np.ndarray] = []
    transition_iterator = (
        dataset.iter_forecast_all()
        if hasattr(dataset, "iter_forecast_all")
        else dataset.iter_all()
    )
    for transition in transition_iterator:
        if transition["split"] != "train":
            break
        mask = transition["semantic_comparable_mask"]
        if not bool(mask.any()):
            continue
        distance = support_topk_jaccard_distance(
            transition["semantic_current"][mask],
            transition["semantic_next"][mask],
            topk=topk,
        )
        distances.append(distance.cpu().numpy())
    if not distances:
        raise ValueError("No consecutive official semantic rows were found in train")
    values = np.concatenate(distances)
    threshold = float(np.quantile(values, quantile))
    positive_rate = float((values > threshold).mean())
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        raise ValueError(
            "The selected semantic threshold creates a one-class train target; "
            "choose a lower quantile."
        )
    return {
        "topk": int(topk),
        "train_quantile": float(quantile),
        "threshold": threshold,
        "train_rows": int(values.size),
        "train_positive_rows": int((values > threshold).sum()),
        "train_positive_rate": positive_rate,
    }


def transition_statistics(
    dataset: NodePropertyTransitionDataset,
    *,
    semantic_threshold: float,
    topk: int = 10,
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, dict[str, float | int]] = {
        split: {
            "transitions": 0,
            "active_source_nodes": 0,
            "active_target_nodes": 0,
            "activated_nodes": 0,
            "deactivated_nodes": 0,
            "natural_added_nodes": 0,
            "natural_removed_nodes": 0,
            "constructed_added_nodes": 0,
            "constructed_removed_nodes": 0,
            "semantic_comparable_rows": 0,
            "semantic_stable_activity_rows": 0,
            "semantic_changed_rows": 0,
        }
        for split in ("train", "val", "test")
    }
    transition_iterator = (
        dataset.iter_forecast_all()
        if hasattr(dataset, "iter_forecast_all")
        else dataset.iter_all()
    )
    for transition in transition_iterator:
        split = str(transition["split"])
        bucket = buckets[split]
        bucket["transitions"] += 1
        bucket["active_source_nodes"] += int(transition["node_active_t"].sum())
        bucket["active_target_nodes"] += int(transition["node_active_next"].sum())
        bucket["activated_nodes"] += int(transition["node_activated"].sum())
        bucket["deactivated_nodes"] += int(transition["node_deactivated"].sum())
        bucket["natural_added_nodes"] += int(transition["node_added_natural"].sum())
        bucket["natural_removed_nodes"] += int(transition["node_removed_natural"].sum())
        bucket["constructed_added_nodes"] += int(
            transition["node_added_constructed"].sum()
        )
        bucket["constructed_removed_nodes"] += int(
            transition["node_removed_constructed"].sum()
        )
        comparable = transition["semantic_comparable_mask"]
        bucket["semantic_comparable_rows"] += int(comparable.sum())
        if bool(comparable.any()):
            labels, _ = semantic_change_labels(
                transition["semantic_current"][comparable],
                transition["semantic_next"][comparable],
                threshold=semantic_threshold,
                topk=topk,
            )
            bucket["semantic_changed_rows"] += int(labels.sum())
            bucket["semantic_stable_activity_rows"] += int(
                transition["semantic_stable_activity_mask"][comparable].sum()
            )
    num_nodes = int(dataset.metadata["num_nodes"])
    for bucket in buckets.values():
        transitions = max(int(bucket["transitions"]), 1)
        comparable = max(int(bucket["semantic_comparable_rows"]), 1)
        denominator = transitions * num_nodes
        bucket["activation_rate"] = float(bucket["activated_nodes"] / denominator)
        bucket["deactivation_rate"] = float(bucket["deactivated_nodes"] / denominator)
        bucket["node_addition_rate"] = bucket["activation_rate"]
        bucket["node_removal_rate"] = bucket["deactivation_rate"]
        bucket["semantic_change_rate"] = float(
            bucket["semantic_changed_rows"] / comparable
        )
        bucket["mean_active_source_nodes"] = float(
            bucket["active_source_nodes"] / transitions
        )
        bucket["mean_active_target_nodes"] = float(
            bucket["active_target_nodes"] / transitions
        )
    return buckets


__all__ = [
    "NodePropertyTransitionDataset",
    "fit_semantic_change_threshold",
    "incident_activity_mask",
    "incident_degree",
    "mask_graph_nodes",
    "semantic_change_labels",
    "select_constructed_node_edits",
    "select_ranked_constructed_node_edits",
    "support_topk_jaccard_distance",
    "transition_statistics",
]
