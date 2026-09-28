
from __future__ import annotations

from collections import defaultdict
import copy
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .data.node_property_transition import (
    NodePropertyTransitionDataset,
    semantic_change_labels,
    support_topk_jaccard_distance,
)
from .genre_training import _official_ndcg
from .latent_objective import latent_transition_terms
from .property_transition import decode_property_prediction, property_prediction_loss


T2_TENSOR_KEYS = {
    "x_t",
    "x_next",
    "edge_index_t",
    "edge_index_next",
    "edge_weight_t",
    "edge_weight_next",
    "node_active_t",
    "node_active_next",
    "node_edit_eligible_mask",
    "node_activated",
    "node_deactivated",
    "node_added",
    "node_removed",
    "semantic_node_ids",
    "semantic_current",
    "semantic_next",
    "semantic_comparable_mask",
    "semantic_stable_activity_mask",
    "controller_history_features",
    "property_node_ids_t",
    "property_observed_t",
    "dense_source_mask",
    "dense_target_mask",
    "semantic_history_source_features",
    "semantic_history_target_features",
    "node_addition_candidates",
    "node_removal_candidates",
    "node_property_candidates",
    "semantic_comparable_indices",
    "semantic_change_distance_cached",
    "node_observed_degree_t",
}


def t2_transition_to_device(
    transition: dict[str, Any], device: torch.device
) -> dict[str, Any]:





    prepared = dict(transition)
    if "node_active_t" in prepared:
        active = prepared["node_active_t"].to(torch.bool)
        eligible = prepared.get("node_edit_eligible_mask")
        if eligible is None:
            eligible = torch.ones_like(active)
        else:
            eligible = eligible.to(torch.bool)
        prepared.setdefault(
            "node_addition_candidates",
            ((~active) & eligible).nonzero(as_tuple=False).flatten(),
        )
        prepared.setdefault(
            "node_removal_candidates",
            (active & eligible).nonzero(as_tuple=False).flatten(),
        )
    if "property_node_ids_t" in prepared:
        prepared.setdefault(
            "node_property_candidates",
            torch.unique(prepared["property_node_ids_t"].long()),
        )
    if "semantic_comparable_mask" in prepared:
        prepared.setdefault(
            "semantic_comparable_indices",
            prepared["semantic_comparable_mask"].to(torch.bool).nonzero(
                as_tuple=False
            ).flatten(),
        )
    moved = {
        key: value.to(device, non_blocking=True) if key in T2_TENSOR_KEYS else value
        for key, value in prepared.items()
    }


    if "node_property_candidates" in prepared:
        moved["node_property_candidates_cpu"] = prepared["node_property_candidates"]
    for tensor_key, mask_key, dim_key in (
        ("x_t", "dense_source_mask", "dense_source_feature_dim"),
        ("x_next", "dense_target_mask", "dense_target_feature_dim"),
    ):
        if tensor_key not in moved or mask_key not in moved:
            continue
        mask = moved[mask_key].to(torch.bool)
        value = moved[tensor_key]


        if value is prepared[tensor_key]:
            value = value.clone()
        feature_dim = min(int(moved[dim_key]), int(value.shape[1]))
        value[:, :feature_dim].mul_(
            (~mask).to(dtype=value.dtype).unsqueeze(-1)
        )
        moved[tensor_key] = value
    for tensor_key, feature_key in (
        ("x_t", "semantic_history_source_features"),
        ("x_next", "semantic_history_target_features"),
    ):
        if tensor_key in moved and feature_key in moved:
            moved[tensor_key] = torch.cat(
                [moved[tensor_key], moved.pop(feature_key)], dim=-1
            )
    if "x_t" in moved:
        moved["node_source_visibility"] = (
            moved["x_t"].detach().abs().sum(dim=-1).gt(1e-8)
        )
    return moved


def _edge_weight_for_dataset(
    transition: dict[str, Any], dataset_name: str, *, suffix: str = "t"
) -> torch.Tensor | None:
    return transition[f"edge_weight_{suffix}"] if dataset_name in {"trade", "genre"} else None


def _binary_loss(logits: torch.Tensor, target: torch.Tensor, *, pos_weight: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        target.to(dtype=logits.dtype),
        pos_weight=logits.new_tensor(float(pos_weight)),
    )


def _candidate_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    pos_weight: float,
) -> torch.Tensor:
    if not bool(candidate_mask.any()):
        return logits.new_zeros(())
    return _binary_loss(
        logits[candidate_mask], target[candidate_mask], pos_weight=pos_weight
    )


def _candidate_focal_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    pos_weight: float,
    gamma: float,
    candidate_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if candidate_indices is None:
        candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return logits.new_zeros(())
    selected_logits = logits.index_select(0, candidate_indices)
    selected_target = target.index_select(0, candidate_indices).to(
        dtype=logits.dtype
    )
    bce = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_target,
        pos_weight=selected_logits.new_tensor(float(pos_weight)),
        reduction="none",
    )
    if gamma == 0:
        return bce.mean()
    probability = torch.sigmoid(selected_logits)
    probability_of_target = torch.where(
        selected_target.gt(0.5), probability, 1.0 - probability
    )
    return ((1.0 - probability_of_target).pow(float(gamma)) * bce).mean()


def _candidate_pairwise_ranking_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    max_hard_negatives: int = 64,
) -> torch.Tensor:
    if max_hard_negatives < 1:
        raise ValueError("max_hard_negatives must be positive")
    if not bool(candidate_mask.any()):
        return logits.new_zeros(())
    candidate_logits = logits[candidate_mask]
    candidate_target = target[candidate_mask].to(torch.bool)
    positive = candidate_logits[candidate_target]
    negative = candidate_logits[~candidate_target]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.new_zeros(())
    hard_negative_count = min(int(max_hard_negatives), int(negative.numel()))
    hard_negative = negative.topk(hard_negative_count).values


    return F.softplus(hard_negative.unsqueeze(0) - positive.unsqueeze(1)).mean()


def fit_t2_training_statistics(
    dataset: NodePropertyTransitionDataset,
    *,
    semantic_threshold: float,
    semantic_topk: int,
) -> dict[str, float | int]:
    activation_positive = 0
    activation_rows = 0
    activation_visible_positive = 0
    activation_visible_rows = 0
    activation_masked_positive = 0
    activation_masked_rows = 0
    deactivation_positive = 0
    deactivation_rows = 0
    semantic_positive = 0
    semantic_rows = 0
    transition_iterator = (
        dataset.iter_forecast_all()
        if hasattr(dataset, "iter_forecast_all")
        else dataset.iter_all()
    )
    for transition in transition_iterator:
        if transition["split"] != "train":
            break
        active_t = transition["node_active_t"]
        active_next = transition["node_active_next"]
        eligible = transition.get("node_edit_eligible_mask")
        if eligible is None:
            eligible = torch.ones_like(active_t, dtype=torch.bool)
        else:
            eligible = eligible.to(torch.bool)
        activation_candidates = (~active_t) & eligible
        deactivation_candidates = active_t & eligible
        activation_positive += int(active_next[activation_candidates].sum())
        activation_rows += int(activation_candidates.sum())
        source_visible = transition["x_t"].abs().sum(dim=-1).gt(1e-8)
        deferred_source_mask = transition.get("dense_source_mask")
        if deferred_source_mask is not None:
            source_visible = source_visible & ~deferred_source_mask.to(torch.bool)
        visible_candidates = activation_candidates & source_visible
        masked_candidates = activation_candidates & ~source_visible
        activation_visible_positive += int(active_next[visible_candidates].sum())
        activation_visible_rows += int(visible_candidates.sum())
        activation_masked_positive += int(active_next[masked_candidates].sum())
        activation_masked_rows += int(masked_candidates.sum())
        deactivation_positive += int((~active_next[deactivation_candidates]).sum())
        deactivation_rows += int(deactivation_candidates.sum())
        comparable = transition["semantic_comparable_mask"]
        if bool(comparable.any()):
            labels, _ = semantic_change_labels(
                transition["semantic_current"][comparable],
                transition["semantic_next"][comparable],
                threshold=semantic_threshold,
                topk=semantic_topk,
            )
            semantic_positive += int(labels.sum())
            semantic_rows += int(labels.numel())
    if activation_rows == 0 or deactivation_rows == 0 or semantic_rows == 0:
        raise ValueError("T2 needs both activity and consecutive semantic train rows.")
    if semantic_positive in {0, semantic_rows}:
        raise ValueError("Semantic T2 label is one-class on train; change the train quantile.")
    return {
        "activation_train_rows": int(activation_rows),
        "activation_train_positive_rows": int(activation_positive),
        "activation_train_positive_rate": float(activation_positive / activation_rows),
        "activation_pos_weight": float(
            (activation_rows - activation_positive) / max(activation_positive, 1)
        ),
        "activation_visible_train_rows": int(activation_visible_rows),
        "activation_visible_train_positive_rows": int(activation_visible_positive),
        "activation_visible_pos_weight": float(
            max(
                (activation_visible_rows - activation_visible_positive)
                / max(activation_visible_positive, 1),
                1.0,
            )
        ),
        "activation_masked_train_rows": int(activation_masked_rows),
        "activation_masked_train_positive_rows": int(activation_masked_positive),
        "activation_masked_pos_weight": float(
            max(
                (activation_masked_rows - activation_masked_positive)
                / max(activation_masked_positive, 1),
                1.0,
            )
        ),
        "deactivation_train_rows": int(deactivation_rows),
        "deactivation_train_positive_rows": int(deactivation_positive),
        "deactivation_train_positive_rate": float(
            deactivation_positive / deactivation_rows
        ),
        "deactivation_pos_weight": float(
            (deactivation_rows - deactivation_positive) / max(deactivation_positive, 1)
        ),
        "semantic_train_rows": int(semantic_rows),
        "semantic_train_positive_rows": int(semantic_positive),
        "semantic_train_positive_rate": float(semantic_positive / semantic_rows),
        "semantic_pos_weight": float((semantic_rows - semantic_positive) / max(semantic_positive, 1)),
    }


def _semantic_prediction(
    outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    *,
    property_mode: str,
    property_gate_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if property_mode not in {"logit_residual", "logit_residual_mixture"}:
        raise ValueError("T2 semantic property mode must be logit_residual or logit_residual_mixture.")
    node_ids = transition["semantic_node_ids"]
    decoded = outputs["node_property"]
    if "node_property_node_ids" not in outputs:
        decoded = decoded.index_select(0, node_ids)




    reference = outputs.get("_semantic_property_reference", transition["semantic_current"])
    reference_observed_mask = outputs.get(
        "_semantic_property_reference_observed_mask",
        transition["semantic_comparable_mask"],
    )
    if reference.shape != transition["semantic_current"].shape:
        raise ValueError("Semantic property reference must match semantic_current shape.")
    if reference_observed_mask.shape != transition["semantic_comparable_mask"].shape:
        raise ValueError("Semantic property reference mask must have one value per row.")
    transition_gate = None
    current_observed_mask = None
    if property_mode == "logit_residual_mixture":
        gate_logits = outputs.get(
            "node_property_transition_gate_logits", outputs["node_change_logits"]
        )
        if property_gate_bias != 0.0:



            gate_logits = gate_logits + gate_logits.new_tensor(
                float(property_gate_bias)
            )
        transition_gate = torch.sigmoid(
            gate_logits.index_select(0, node_ids)
        )
        current_observed_mask = reference_observed_mask
    prediction, delta = decode_property_prediction(
        decoded,
        reference,
        property_mode=property_mode,
        transition_gate=transition_gate,
        current_observed_mask=current_observed_mask,
    )
    return prediction, delta, decoded


def _current_semantic_reconstruction_loss(
    outputs: dict[str, torch.Tensor], transition: dict[str, Any]
) -> torch.Tensor:
    logits = outputs["node_property_current"]
    if "node_property_current_node_ids" not in outputs:
        logits = logits.index_select(0, transition["property_node_ids_t"])
    current = transition["property_observed_t"]
    if logits.numel() == 0:
        return outputs["node_change_logits"].new_zeros(())
    return -(current * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()


def _numpy_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if labels.size == 0:
        return {
            "rows": 0,
            "positive_rows": 0,
            "positive_rate": None,
            "auroc": None,
            "auprc": None,
            "f1": None,
            "threshold": threshold,
            "predicted_positive_rows": 0,
        }
    two_classes = np.unique(labels).size == 2
    prediction = (
        np.zeros_like(labels, dtype=bool)
        if threshold is None
        else scores >= float(threshold)
    )
    return {
        "rows": int(labels.size),
        "positive_rows": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)) if two_classes else None,
        "auprc": float(average_precision_score(labels, scores)) if two_classes else None,
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "threshold": None if threshold is None else float(threshold),
        "predicted_positive_rows": int(prediction.sum()),
    }


def select_binary_f1_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float | None, dict[str, float | str]]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if labels.size == 0:
        return None, {"selection": "empty validation rows"}
    if np.unique(labels).size != 2:

        return None, {"selection": "one-class validation rows"}
    if np.unique(scores).size < 2:
        all_negative = _numpy_binary_metrics(labels, scores, threshold=None)
        all_positive = _numpy_binary_metrics(labels, scores, threshold=float(scores[0]))
        if float(all_positive["f1"] or 0.0) > float(all_negative["f1"] or 0.0):
            return float(scores[0]), {
                "selection": "constant score -> all positive",
                "validation_f1": float(all_positive["f1"] or 0.0),
            }
        return None, {
            "selection": "constant score -> all negative",
            "validation_f1": float(all_negative["f1"] or 0.0),
        }
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.nanargmax(f1))
    return float(thresholds[index]), {
        "selection": "maximize validation F1",
        "validation_f1": float(f1[index]),
        "validation_precision": float(precision[index]),
        "validation_recall": float(recall[index]),
    }


class SemanticHistoryCache:

    def __init__(self) -> None:
        self.source_node_ids: dict[int, torch.Tensor] = {}
        self.source_index: dict[int, torch.Tensor] = {}
        self.source_valid: dict[int, torch.Tensor] = {}
        self.target_node_ids: dict[int, torch.Tensor] = {}
        self.target_index: dict[int, torch.Tensor] = {}
        self.target_valid: dict[int, torch.Tensor] = {}


def _positive_topk_support(
    values: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    values = values.detach().cpu().float()
    if values.ndim != 2:
        raise ValueError("semantic values must be a [rows, dimensions] matrix")
    k = min(int(topk), int(values.shape[1]))
    if values.shape[0] == 0:
        return (
            torch.empty((0, k), dtype=torch.long),
            torch.empty((0, k), dtype=torch.bool),
        )
    top_value, top_index = values.topk(k, dim=1)




    if int(values.shape[1]) < 2**15:
        top_index = top_index.to(torch.int16)
    return top_index, top_value.gt(0)


def _support_jaccard_distance(
    current_index: torch.Tensor,
    current_valid: torch.Tensor,
    following_index: torch.Tensor,
    following_valid: torch.Tensor,
) -> torch.Tensor:

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


def cached_semantic_change_labels(
    cache: SemanticHistoryCache,
    *,
    transition_id: int,
    node_ids: torch.Tensor,
    comparable: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:

    transition_id = int(transition_id)
    source_ids = cache.source_node_ids.get(transition_id)
    source_index = cache.source_index.get(transition_id)
    source_valid = cache.source_valid.get(transition_id)
    target_ids = cache.target_node_ids.get(transition_id)
    target_index = cache.target_index.get(transition_id)
    target_valid = cache.target_valid.get(transition_id)
    if any(
        value is None
        for value in (
            source_ids,
            source_index,
            source_valid,
            target_ids,
            target_index,
            target_valid,
        )
    ):
        return None

    ids_cpu = node_ids.detach().cpu().long()
    comparable_cpu = comparable.detach().cpu().to(torch.bool)
    assert source_ids is not None
    assert source_index is not None
    assert source_valid is not None
    assert target_ids is not None
    assert target_index is not None
    assert target_valid is not None
    if not torch.equal(target_ids, ids_cpu):
        return None
    selected_ids = ids_cpu[comparable_cpu]
    if selected_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty.to(torch.bool), empty

    max_node_id = max(
        int(source_ids.max().item()) if source_ids.numel() else -1,
        int(selected_ids.max().item()),
    )
    source_position = torch.full((max_node_id + 1,), -1, dtype=torch.long)
    source_position[source_ids] = torch.arange(source_ids.numel(), dtype=torch.long)
    positions = source_position.index_select(0, selected_ids)
    if bool(positions.lt(0).any()):
        return None
    distance = _support_jaccard_distance(
        source_index.index_select(0, positions),
        source_valid.index_select(0, positions),
        target_index[comparable_cpu],
        target_valid[comparable_cpu],
    )
    return distance.gt(float(threshold)), distance


class OnlineSemanticChangeHistory:

    def __init__(self, num_nodes: int, semantic_dim: int, *, topk: int) -> None:
        self.last_value = torch.zeros((int(num_nodes), int(semantic_dim)), dtype=torch.float32)
        self.last_seen = torch.zeros(int(num_nodes), dtype=torch.bool)
        self.last_distance = torch.zeros(int(num_nodes), dtype=torch.float32)
        self.distance_seen = torch.zeros(int(num_nodes), dtype=torch.bool)
        self.topk = int(topk)
        self._sum = 0.0
        self._count = 0
        self._cache: SemanticHistoryCache | None = None
        support_width = min(self.topk, int(semantic_dim))
        support_dtype = torch.int16 if int(semantic_dim) < 2**15 else torch.long
        self._last_support_index = torch.zeros(
            (int(num_nodes), support_width), dtype=support_dtype
        )
        self._last_support_valid = torch.zeros(
            (int(num_nodes), support_width), dtype=torch.bool
        )

    def attach_cache(self, cache: SemanticHistoryCache | None) -> None:

        self._cache = cache

    def clone(self) -> "OnlineSemanticChangeHistory":





        result = object.__new__(OnlineSemanticChangeHistory)
        result.last_value = (
            self.last_value
            if self._cache is not None
            else self.last_value.clone()
        )
        result.last_seen = self.last_seen.clone()
        result.last_distance = self.last_distance.clone()
        result.distance_seen = self.distance_seen.clone()
        result.topk = self.topk
        result._last_support_index = self._last_support_index.clone()
        result._last_support_valid = self._last_support_valid.clone()
        result._sum = self._sum
        result._count = self._count
        result._cache = self._cache
        return result

    def observe(
        self,
        node_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        transition_id: int | None = None,
    ) -> None:
        node_ids = node_ids.detach().cpu().long()
        values = values.detach().cpu().float()
        if not node_ids.numel():
            return
        prior_seen = self.last_seen.index_select(0, node_ids)
        if bool(prior_seen.any()):
            cached_index = None
            cached_valid = None
            if self._cache is not None and transition_id is not None:
                cached_ids = self._cache.source_node_ids.get(int(transition_id))
                candidate_index = self._cache.source_index.get(int(transition_id))
                candidate_valid = self._cache.source_valid.get(int(transition_id))
                if (
                    cached_ids is not None
                    and candidate_index is not None
                    and candidate_valid is not None
                    and torch.equal(cached_ids, node_ids)
                ):
                    cached_index = candidate_index
                    cached_valid = candidate_valid
            if (
                cached_index is not None
                and cached_valid is not None
            ):
                changed_ids = node_ids[prior_seen]
                distance = _support_jaccard_distance(
                    self._last_support_index.index_select(0, changed_ids),
                    self._last_support_valid.index_select(0, changed_ids),
                    cached_index[prior_seen],
                    cached_valid[prior_seen],
                )
            else:
                old = self.last_value.index_select(0, node_ids)[prior_seen]
                new = values[prior_seen]
                distance = support_topk_jaccard_distance(old, new, topk=self.topk)
                changed_ids = node_ids[prior_seen]
            self.last_distance.index_copy_(0, changed_ids, distance)
            self.distance_seen.index_fill_(0, changed_ids, True)
            self._sum += float(distance.sum())
            self._count += int(distance.numel())
        if self._cache is not None:
            cached_ids = (
                None
                if transition_id is None
                else self._cache.source_node_ids.get(int(transition_id))
            )
            support_index = (
                None
                if transition_id is None
                else self._cache.source_index.get(int(transition_id))
            )
            support_valid = (
                None
                if transition_id is None
                else self._cache.source_valid.get(int(transition_id))
            )
            if (
                cached_ids is None
                or support_index is None
                or support_valid is None
                or not torch.equal(cached_ids, node_ids)
            ):
                support_index, support_valid = _positive_topk_support(
                    values, topk=self.topk
                )
            self._last_support_index.index_copy_(
                0, node_ids, support_index.to(self._last_support_index.dtype)
            )
            self._last_support_valid.index_copy_(0, node_ids, support_valid)






        if self._cache is None:
            self.last_value.index_copy_(0, node_ids, values)
        self.last_seen.index_fill_(0, node_ids, True)

    def score(self, node_ids: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.detach().cpu().long()
        fallback = self._sum / self._count if self._count else 0.0
        return torch.where(
            self.distance_seen.index_select(0, node_ids),
            self.last_distance.index_select(0, node_ids),
            torch.full((node_ids.numel(),), float(fallback), dtype=torch.float32),
        )


def build_semantic_history_cache(
    dataset: NodePropertyTransitionDataset,
    *,
    topk: int = 10,
) -> SemanticHistoryCache:

    cache = SemanticHistoryCache()
    transition_count = int(dataset.base.transition_count)
    for transition_id in range(transition_count):
        if hasattr(dataset, "forecast_transition"):
            transition = dataset.forecast_transition(transition_id)
        else:
            transition = dataset.transition(transition_id)
        source_ids = transition["property_node_ids_t"].detach().cpu().long()
        source_values = transition["property_observed_t"].detach().cpu().float()
        target_ids = transition["property_node_ids"].detach().cpu().long()
        target_values = transition["property_target"].detach().cpu().float()
        source_index, source_valid = _positive_topk_support(
            source_values, topk=topk
        )
        target_index, target_valid = _positive_topk_support(
            target_values, topk=topk
        )
        cache.source_node_ids[transition_id] = source_ids
        cache.source_index[transition_id] = source_index
        cache.source_valid[transition_id] = source_valid
        cache.target_node_ids[transition_id] = target_ids
        cache.target_index[transition_id] = target_index
        cache.target_valid[transition_id] = target_valid
    return cache


class OnlineSemanticPropertyHistory:

    def __init__(self, num_nodes: int, semantic_dim: int) -> None:
        self.total = torch.zeros((int(num_nodes), int(semantic_dim)), dtype=torch.float32)
        self.count = torch.zeros(int(num_nodes), dtype=torch.long)

    def clone(self) -> "OnlineSemanticPropertyHistory":

        result = OnlineSemanticPropertyHistory(
            num_nodes=int(self.total.shape[0]),
            semantic_dim=int(self.total.shape[1]),
        )
        result.total = self.total.clone()
        result.count = self.count.clone()
        return result

    def observe(self, node_ids: torch.Tensor, values: torch.Tensor) -> None:
        node_ids = node_ids.detach().cpu().long()
        values = values.detach().cpu().float()
        if not node_ids.numel():
            return
        self.total.index_add_(0, node_ids, values)
        self.count.index_add_(
            0,
            node_ids,
            torch.ones_like(node_ids, dtype=self.count.dtype),
        )

    def mean(self, node_ids: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.detach().cpu().long()
        total = self.total.index_select(0, node_ids)
        count = self.count.index_select(0, node_ids).clamp_min(1).unsqueeze(-1)
        return total / count.to(dtype=total.dtype)

    def reference(self, node_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_ids = node_ids.detach().cpu().long()
        observed = self.count.index_select(0, node_ids) > 0
        return self.mean(node_ids), observed


def _new_buckets() -> dict[str, dict[str, list[np.ndarray]]]:
    names = ("activation", "deactivation", "semantic", "semantic_stable_activity")
    return {
        name: {
            "labels": [],
            "scores": [],
            "history_scores": [],
            "count_log1p": [],
            "group_ids": [],




            "source_visible": [],
        }
        for name in names
    }


def _append_bucket(
    bucket: dict[str, list[np.ndarray]],
    labels: torch.Tensor,
    scores: torch.Tensor,
    history_scores: torch.Tensor,
    *,
    count_log1p: torch.Tensor | float | None = None,
    source_visible: torch.Tensor | None = None,
) -> None:
    group_id = len(bucket["labels"])
    if count_log1p is None:
        count_value = float("nan")
    elif isinstance(count_log1p, torch.Tensor):
        count_value = float(count_log1p.detach().cpu().reshape(-1)[0].item())
    else:
        count_value = float(count_log1p)
    bucket["labels"].append(labels.detach().cpu().numpy().astype(bool, copy=False))
    bucket["scores"].append(scores.detach().cpu().numpy().astype(np.float32, copy=False))
    bucket["history_scores"].append(
        history_scores.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    bucket["count_log1p"].append(
        np.full(int(labels.numel()), count_value, dtype=np.float32)
    )
    bucket["group_ids"].append(
        np.full(int(labels.numel()), group_id, dtype=np.int64)
    )
    if source_visible is None:
        bucket["source_visible"].append(
            np.zeros(int(labels.numel()), dtype=bool)
        )
    else:
        source_visible = source_visible.reshape(-1)
        if source_visible.numel() != labels.numel():
            raise ValueError("source_visible must align with the bucket labels")
        bucket["source_visible"].append(
            source_visible.detach().cpu().numpy().astype(bool, copy=False)
        )


def _finish_bucket(bucket: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate(values, axis=0)
        if values
        else np.empty(
            (0,),
            dtype=(
                bool
                if name in {"labels", "source_visible"}
                else np.int64
                if name == "group_ids"
                else np.float32
            ),
        )
        for name, values in bucket.items()
    }


def _count_topk_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    count_log1p: np.ndarray,
    group_ids: np.ndarray,
    *,
    log_count_offset: float = 0.0,
) -> dict[str, float | int | str | None] | None:

    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    count_log1p = np.asarray(count_log1p, dtype=np.float32).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if not (
        labels.shape == scores.shape == count_log1p.shape == group_ids.shape
    ):
        raise ValueError("count-topk arrays must have identical shapes")
    if labels.size == 0 or not np.isfinite(count_log1p).all():
        return None
    prediction = np.zeros_like(labels, dtype=bool)
    for group in np.unique(group_ids):
        positions = np.flatnonzero(group_ids == group)
        if positions.size == 0:
            continue
        calibrated_log_count = float(
            np.clip(float(count_log1p[positions[0]]) + float(log_count_offset), 0.0, 12.0)
        )
        count = int(round(float(np.expm1(calibrated_log_count))))
        count = min(max(count, 0), int(positions.size))
        if count:
            selected = positions[np.argsort(scores[positions])[-count:]]
            prediction[selected] = True
    ranking = _numpy_binary_metrics(labels, scores, threshold=None)
    ranking.update(
        {
            "f1": float(f1_score(labels, prediction, zero_division=0)),
            "threshold": None,
            "predicted_positive_rows": int(prediction.sum()),
            "decision_rule": "validation_calibrated_predicted_operation_count_topk",
        }
    )
    return ranking


def _fit_count_log_offset(
    labels: np.ndarray,
    count_log1p: np.ndarray,
    group_ids: np.ndarray,
    scores: np.ndarray | None = None,
) -> tuple[float, dict[str, float | int]] | None:

    labels = np.asarray(labels, dtype=bool).reshape(-1)
    count_log1p = np.asarray(count_log1p, dtype=np.float32).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if not (labels.shape == count_log1p.shape == group_ids.shape):
        raise ValueError("count calibration arrays must have identical shapes")
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape != labels.shape:
            raise ValueError("scores must match count calibration labels")
    if labels.size == 0 or not np.isfinite(count_log1p).all():
        return None
    residuals: list[float] = []
    actual_total = 0
    predicted_total = 0.0
    for group in np.unique(group_ids):
        positions = np.flatnonzero(group_ids == group)
        if positions.size == 0:
            continue
        actual_count = int(labels[positions].sum())
        predicted_log_count = float(np.clip(count_log1p[positions[0]], 0.0, 12.0))
        residuals.append(float(np.log1p(actual_count) - predicted_log_count))
        actual_total += actual_count
        predicted_total += float(np.expm1(predicted_log_count))
    if not residuals:
        return None




    anchor = float(np.clip(np.mean(residuals), -1.5, 1.5))
    candidate_offsets = np.unique(
        np.concatenate([np.linspace(-1.5, 1.5, num=121), np.asarray([anchor])])
    )
    offset = anchor
    validation_f1: float | None = None
    if scores is not None:
        best_key: tuple[float, float, float] | None = None
        for candidate in candidate_offsets:
            metrics = _count_topk_binary_metrics(
                labels,
                scores,
                count_log1p,
                group_ids,
                log_count_offset=float(candidate),
            )
            assert metrics is not None


            key = (
                float(metrics["f1"]),
                -abs(float(candidate) - anchor),
                -abs(float(candidate)),
            )
            if best_key is None or key > best_key:
                best_key = key
                offset = float(candidate)
                validation_f1 = float(metrics["f1"])
    return offset, {
        "log_count_offset": offset,
        "transitions": int(len(residuals)),
        "actual_positive_rows": int(actual_total),
        "uncalibrated_predicted_rows": float(predicted_total),
        "selection": "maximize validation F1" if scores is not None else "count-scale residual",
        "validation_f1": validation_f1,
    }


@torch.no_grad()
def evaluate_t2_semantic_sequence(
    model: torch.nn.Module,
    dataset: NodePropertyTransitionDataset,
    *,
    split: str,
    device: torch.device,
    semantic_threshold: float,
    semantic_topk: int,
    semantic_property_mode: str = "logit_residual_mixture",
    semantic_gate_bias: float = 0.0,
    decision_thresholds: dict[str, float | None] | None = None,
    compute_latent: bool = True,
    minimal_metrics: bool = False,
    compute_baseline_diagnostics: bool = True,
    forward_step: Callable[[dict[str, Any], dict[str, Any], torch.Tensor], dict[str, torch.Tensor]]
    | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    model.eval()
    num_nodes = int(dataset.metadata["num_nodes"])
    property_dim = int(dataset.metadata["official_target_dim"])
    hidden = model.initial_hidden(num_nodes, device)
    history = OnlineSemanticChangeHistory(num_nodes, property_dim, topk=semantic_topk)
    history.attach_cache(getattr(dataset, "semantic_history_cache", None))
    property_history = (
        OnlineSemanticPropertyHistory(num_nodes, property_dim)
        if compute_baseline_diagnostics
        else None
    )
    buckets = _new_buckets()
    ndcg_total = 0.0
    ndcg_rows = 0
    mse_total = 0.0
    mse_entries = 0
    copy_last_ndcg_total = 0.0
    copy_last_mse_total = 0.0
    historical_mean_ndcg_total = 0.0
    historical_mean_mse_total = 0.0





    changed_ndcg_total = 0.0
    changed_ndcg_rows = 0
    changed_copy_last_ndcg_total = 0.0
    changed_historical_mean_ndcg_total = 0.0
    latent_squared_error = 0.0
    latent_entries = 0
    latent_cosine_sum = 0.0
    latent_cosine_rows = 0




    transition_iterator = (
        dataset.iter_all()
        if compute_latent or not hasattr(dataset, "iter_forecast_all")
        else dataset.iter_forecast_all()
    )
    for transition_cpu in transition_iterator:



        if split == "val" and transition_cpu["split"] == "test":
            break
        score_transition = transition_cpu["split"] == split



        history.observe(
            transition_cpu["property_node_ids_t"],
            transition_cpu["property_observed_t"],
            transition_id=int(transition_cpu["transition_id"]),
        )
        if property_history is not None:
            property_history.observe(
                transition_cpu["property_node_ids_t"], transition_cpu["property_observed_t"]
            )




        forward_transition_cpu = transition_cpu
        if forward_step is not None:
            prepare_transition = getattr(forward_step, "prepare_transition", None)
            if prepare_transition is not None:
                forward_transition_cpu = prepare_transition(transition_cpu)
        transition = t2_transition_to_device(forward_transition_cpu, device)
        outputs = (
            model(
                transition["x_t"],
                transition["edge_index_t"],
                hidden,
                action=None,
                edge_weight_t=_edge_weight_for_dataset(transition, dataset.dataset_name),
                decode_observables=score_transition,
            )
            if forward_step is None
            else forward_step(forward_transition_cpu, transition, hidden)
        )
        hidden = outputs["hidden_next"].detach()





        operation_count_log1p = outputs.get(
            "_decision_count_log1p", outputs.get("node_operation_count_log1p")
        )
        if operation_count_log1p is not None:
            operation_count_log1p = operation_count_log1p.reshape(-1)
            if operation_count_log1p.numel() != 3:
                raise ValueError(
                    "node_operation_count_log1p must contain addition, removal, "
                    "and semantic-change counts."
                )
        if not score_transition:
            continue

        active_t = transition["node_active_t"]
        active_next = transition["node_active_next"]
        eligible = transition.get("node_edit_eligible_mask")
        if eligible is None:
            eligible = torch.ones_like(active_t, dtype=torch.bool)
        else:
            eligible = eligible.to(torch.bool)
        activation_candidates = (~active_t) & eligible
        deactivation_candidates = active_t & eligible




        activation_source_visible = transition["x_t"].abs().sum(dim=-1).gt(1e-8)
        _append_bucket(
            buckets["activation"],
            active_next[activation_candidates],
            torch.sigmoid(outputs["node_activation_logits"])[activation_candidates],
            torch.zeros(int(activation_candidates.sum()), device=device),
            count_log1p=(
                None if operation_count_log1p is None else operation_count_log1p[0]
            ),
            source_visible=activation_source_visible[activation_candidates],
        )
        _append_bucket(
            buckets["deactivation"],
            ~active_next[deactivation_candidates],
            torch.sigmoid(outputs["node_deactivation_logits"])[deactivation_candidates],
            torch.zeros(int(deactivation_candidates.sum()), device=device),
            count_log1p=(
                None if operation_count_log1p is None else operation_count_log1p[1]
            ),
        )

        semantic_prediction = None
        if not minimal_metrics:
            semantic_prediction, _, _ = _semantic_prediction(
                outputs,
                transition,
                property_mode=semantic_property_mode,
                property_gate_bias=semantic_gate_bias,
            )
        semantic_ids = transition["semantic_node_ids"]
        comparable = transition["semantic_comparable_mask"]
        semantic_labels_for_ndcg: torch.Tensor | None = None
        if bool(comparable.any()):
            semantic_labels, _ = semantic_change_labels(
                transition["semantic_current"][comparable],
                transition["semantic_next"][comparable],
                threshold=semantic_threshold,
                topk=semantic_topk,
            )
            semantic_score = torch.sigmoid(
                outputs["node_change_logits"].index_select(0, semantic_ids)
            )[comparable]
            historical_score = history.score(semantic_ids.detach().cpu()).to(device)[comparable]
            _append_bucket(
                buckets["semantic"],
                semantic_labels,
                semantic_score,
                historical_score,
                count_log1p=(
                    None if operation_count_log1p is None else operation_count_log1p[2]
                ),
            )
            stable = transition["semantic_stable_activity_mask"][comparable]
            if bool(stable.any()):
                _append_bucket(
                    buckets["semantic_stable_activity"],
                    semantic_labels[stable],
                    semantic_score[stable],
                    historical_score[stable],




                    count_log1p=None,
                )


            semantic_labels_for_ndcg = semantic_labels




        target = transition["semantic_next"]
        target_rows = int(target.shape[0])
        if target_rows and not minimal_metrics:
            assert semantic_prediction is not None
            target_np = target.detach().cpu().numpy()
            prediction_np = semantic_prediction.detach().cpu().numpy()
            ndcg_total += _official_ndcg(
                target_np,
                prediction_np,
                dataset_name=f"tgbn-{dataset.dataset_name}",
            ) * target_rows
            copy_last_np = None
            historical_mean_np = None
            if compute_baseline_diagnostics:
                assert property_history is not None
                copy_last_np = transition["semantic_current"].detach().cpu().numpy()
                historical_mean_np = property_history.mean(semantic_ids).numpy()
                copy_last_ndcg_total += _official_ndcg(
                    target_np,
                    copy_last_np,
                    dataset_name=f"tgbn-{dataset.dataset_name}",
                ) * target_rows
                historical_mean_ndcg_total += _official_ndcg(
                    target_np,
                    historical_mean_np,
                    dataset_name=f"tgbn-{dataset.dataset_name}",
                ) * target_rows
            if semantic_labels_for_ndcg is not None:
                comparable_np = comparable.detach().cpu().numpy().astype(bool)
                changed_np = (
                    semantic_labels_for_ndcg.detach().cpu().numpy().astype(bool)
                )
                if bool(changed_np.any()):
                    changed_target_np = target_np[comparable_np][changed_np]
                    changed_prediction_np = prediction_np[comparable_np][changed_np]
                    changed_rows = int(changed_target_np.shape[0])
                    changed_ndcg_total += _official_ndcg(
                        changed_target_np,
                        changed_prediction_np,
                        dataset_name=f"tgbn-{dataset.dataset_name}",
                    ) * changed_rows
                    if compute_baseline_diagnostics:
                        assert copy_last_np is not None
                        assert historical_mean_np is not None
                        changed_copy_np = copy_last_np[comparable_np][changed_np]
                        changed_historical_np = historical_mean_np[comparable_np][changed_np]
                        changed_copy_last_ndcg_total += _official_ndcg(
                            changed_target_np,
                            changed_copy_np,
                            dataset_name=f"tgbn-{dataset.dataset_name}",
                        ) * changed_rows
                        changed_historical_mean_ndcg_total += _official_ndcg(
                            changed_target_np,
                            changed_historical_np,
                            dataset_name=f"tgbn-{dataset.dataset_name}",
                        ) * changed_rows
                    changed_ndcg_rows += changed_rows
            ndcg_rows += target_rows
            if compute_baseline_diagnostics:
                assert copy_last_np is not None and historical_mean_np is not None
                mse_total += float(np.square(prediction_np - target_np).sum(dtype=np.float64))
                copy_last_mse_total += float(np.square(copy_last_np - target_np).sum(dtype=np.float64))
                historical_mean_mse_total += float(
                    np.square(historical_mean_np - target_np).sum(dtype=np.float64)
                )
                mse_entries += int(target_np.size)

        if compute_latent:





            target_x_next = outputs.get("_target_x_next", transition["x_next"])
            target_z = model.encode_target(
                target_x_next,
                transition["edge_index_next"],
                edge_weight_next=_edge_weight_for_dataset(
                    transition, dataset.dataset_name, suffix="next"
                ),
            )
            difference = outputs["latent_mu"] - target_z
            latent_squared_error += float(difference.square().sum().item())
            latent_entries += int(difference.numel())
            latent_cosine_sum += float(
                F.cosine_similarity(outputs["latent_mu"], target_z, dim=-1).sum().item()
            )
            latent_cosine_rows += int(target_z.shape[0])

    finalized = {name: _finish_bucket(bucket) for name, bucket in buckets.items()}
    thresholds = decision_thresholds or {}
    observables: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    for name, values in finalized.items():



        observables[name] = _numpy_binary_metrics(
            values["labels"], values["scores"], threshold=thresholds.get(name)
        )
        persistence_scores = np.zeros_like(values["scores"])
        baselines[name] = {
            "persistence": _numpy_binary_metrics(
                values["labels"], persistence_scores, threshold=None
            ),
            "historical_semantic_change": (
                _numpy_binary_metrics(
                    values["labels"], values["history_scores"], threshold=thresholds.get(name)
                )
                if name.startswith("semantic")
                else None
            ),
        }
    return {
        "split": split,
        "semantic_threshold": float(semantic_threshold),
        "semantic_topk": int(semantic_topk),
        "activity": {
            "activation": observables["activation"],
            "deactivation": observables["deactivation"],
        },
        "node_addition": observables["activation"],
        "node_removal": observables["deactivation"],
        "semantic_change": observables["semantic"],
        "semantic_feature_change": observables["semantic"],
        "semantic_change_stable_activity": observables["semantic_stable_activity"],
        "semantic_prediction": {
            "official_ndcg": None if ndcg_rows == 0 else float(ndcg_total / ndcg_rows),
            "changed_ndcg_at_10": (
                None
                if changed_ndcg_rows == 0
                else float(changed_ndcg_total / changed_ndcg_rows)
            ),
            "changed_label_rows": int(changed_ndcg_rows),
            "mse": None if mse_entries == 0 else float(mse_total / mse_entries),
            "label_rows": int(ndcg_rows),
            "target_dim": property_dim,
            "decoder": f"{semantic_property_mode} from predicted future latent",
        },
        "latent": {
            "mse": None if latent_entries == 0 else float(latent_squared_error / latent_entries),
            "cosine_similarity": None
            if latent_cosine_rows == 0
            else float(latent_cosine_sum / latent_cosine_rows),
        },
        "baselines": {
            "activity_persistence": {
                "activation": baselines["activation"]["persistence"],
                "deactivation": baselines["deactivation"]["persistence"],
            },
            "node_edit_persistence": {
                "node_addition": baselines["activation"]["persistence"],
                "node_removal": baselines["deactivation"]["persistence"],
            },
            "semantic_persistence": baselines["semantic"]["persistence"],
            "historical_semantic_change": baselines["semantic"][
                "historical_semantic_change"
            ],
            "semantic_persistence_stable_activity": baselines["semantic_stable_activity"]["persistence"],
            "historical_semantic_change_stable_activity": baselines[
                "semantic_stable_activity"
            ]["historical_semantic_change"],
            "semantic_value": {
                "copy_last": {
                    "official_ndcg": None
                    if not compute_baseline_diagnostics or ndcg_rows == 0
                    else float(copy_last_ndcg_total / ndcg_rows),
                    "changed_ndcg_at_10": (
                        None
                        if not compute_baseline_diagnostics or changed_ndcg_rows == 0
                        else float(changed_copy_last_ndcg_total / changed_ndcg_rows)
                    ),
                    "mse": None
                    if mse_entries == 0
                    else float(copy_last_mse_total / mse_entries),
                },
                "historical_mean": {
                    "official_ndcg": None
                    if not compute_baseline_diagnostics or ndcg_rows == 0
                    else float(historical_mean_ndcg_total / ndcg_rows),
                    "changed_ndcg_at_10": (
                        None
                        if not compute_baseline_diagnostics or changed_ndcg_rows == 0
                        else float(
                            changed_historical_mean_ndcg_total / changed_ndcg_rows
                        )
                    ),
                    "mse": None
                    if mse_entries == 0
                    else float(historical_mean_mse_total / mse_entries),
                },
            },
        },
        "calibration_rows": {
            name: {
                "labels": values["labels"],
                "scores": values["scores"],
                "history_scores": values["history_scores"],
                "count_log1p": values["count_log1p"],
                "group_ids": values["group_ids"],
                "source_visible": values["source_visible"],
            }
            for name, values in finalized.items()
        },
    }


def _metrics_from_binary_prediction(
    labels: np.ndarray,
    scores: np.ndarray,
    prediction: np.ndarray,
    *,
    decision_rule: str,
) -> dict[str, float | int | str | None]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    if not (labels.shape == scores.shape == prediction.shape):
        raise ValueError("labels, scores, and prediction must have the same shape")
    metrics = _numpy_binary_metrics(labels, scores, threshold=None)
    metrics.update(
        {
            "f1": float(f1_score(labels, prediction, zero_division=0)),
            "threshold": None,
            "predicted_positive_rows": int(prediction.sum()),
            "decision_rule": decision_rule,
        }
    )
    return metrics


def _fit_source_visibility_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    source_visible: np.ndarray,
    *,
    fallback_threshold: float | None,
) -> tuple[dict[str, float | None], dict[str, Any]] | None:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    source_visible = np.asarray(source_visible, dtype=bool).reshape(-1)
    if not (labels.shape == scores.shape == source_visible.shape):
        raise ValueError("source-visible calibration arrays must have identical shapes")
    if labels.size == 0 or not source_visible.any() or bool(source_visible.all()):
        return None

    thresholds: dict[str, float | None] = {}
    details: dict[str, Any] = {}
    prediction = np.zeros_like(labels, dtype=bool)
    for name, mask in (
        ("visible", source_visible),
        ("masked", ~source_visible),
    ):
        subgroup_labels = labels[mask]
        subgroup_scores = scores[mask]
        if subgroup_labels.size == 0 or np.unique(subgroup_labels).size != 2:
            threshold = fallback_threshold
            selection: dict[str, Any] = {
                "selection": "fallback to global validation threshold",
            }
        else:
            threshold, selection = select_binary_f1_threshold(
                subgroup_labels, subgroup_scores
            )
        thresholds[name] = threshold
        details[name] = selection
        if threshold is not None:
            prediction[mask] = subgroup_scores >= float(threshold)

    metrics = _metrics_from_binary_prediction(
        labels,
        scores,
        prediction,
        decision_rule="source_visibility_validation_thresholds",
    )
    details["validation_f1"] = float(metrics["f1"] or 0.0)
    details["predicted_positive_rows"] = int(prediction.sum())
    return thresholds, details


def _source_visibility_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    source_visible: np.ndarray,
    thresholds: dict[str, float | None],
) -> dict[str, float | int | str | None]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    source_visible = np.asarray(source_visible, dtype=bool).reshape(-1)
    if not (labels.shape == scores.shape == source_visible.shape):
        raise ValueError("source-visible decision arrays must have identical shapes")
    prediction = np.zeros_like(labels, dtype=bool)
    for name, mask in (("visible", source_visible), ("masked", ~source_visible)):
        threshold = thresholds.get(name)
        if threshold is not None:
            prediction[mask] = scores[mask] >= float(threshold)
    return _metrics_from_binary_prediction(
        labels,
        scores,
        prediction,
        decision_rule="source_visibility_validation_thresholds",
    )


def select_t2_thresholds(
    validation: dict[str, Any],
    *,
    force_count_names: tuple[str, ...] = (),
    source_visibility_threshold_names: tuple[str, ...] = (),
    calibration_group_tail_fraction: float = 1.0,
) -> dict[str, Any]:
    tail_fraction = float(calibration_group_tail_fraction)
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("calibration_group_tail_fraction must lie in (0, 1].")
    raw = validation["calibration_rows"]
    forced = frozenset(str(name) for name in force_count_names)
    source_visibility_names = frozenset(
        str(name) for name in source_visibility_threshold_names
    )
    result: dict[str, Any] = {}
    for name, values in raw.items():
        calibration_values = values
        if tail_fraction < 1.0:
            group_ids = np.asarray(values["group_ids"]).reshape(-1)
            unique_groups = np.unique(group_ids)
            tail_count = max(1, int(np.ceil(unique_groups.size * tail_fraction)))
            keep_groups = unique_groups[-tail_count:]
            keep = np.isin(group_ids, keep_groups)
            row_count = int(np.asarray(values["labels"]).reshape(-1).size)
            calibration_values = {
                key: (
                    np.asarray(value)[keep]
                    if isinstance(value, np.ndarray)
                    and value.ndim > 0
                    and int(value.shape[0]) == row_count
                    else value
                )
                for key, value in values.items()
            }
        threshold, details = select_binary_f1_threshold(
            calibration_values["labels"], calibration_values["scores"]
        )
        details["calibration_group_tail_fraction"] = tail_fraction
        if name.startswith("semantic"):
            historical_threshold, historical_details = select_binary_f1_threshold(
                calibration_values["labels"], calibration_values["history_scores"]
            )
        else:
            historical_threshold, historical_details = None, None
        result[name] = {
            "decoder_threshold": threshold,
            "decoder_selection": details,
            "decoder_count_rule": None,
            "historical_threshold": historical_threshold,
            "historical_selection": historical_details,
        }
        if (
            calibration_values.get("count_log1p") is not None
            and np.isfinite(calibration_values["count_log1p"]).all()
        ):
            count_calibration = _fit_count_log_offset(
                calibration_values["labels"],
                calibration_values["count_log1p"],
                calibration_values["group_ids"],
                calibration_values["scores"],
            )
            if count_calibration is not None:
                offset, calibration_details = count_calibration
                count_f1 = calibration_details.get("validation_f1")
                threshold_f1 = details.get("validation_f1")








                if name in forced or (
                    count_f1 is not None
                    and threshold_f1 is not None
                    and float(count_f1) > float(threshold_f1) + 1e-8
                ):
                    result[name]["decoder_count_rule"] = (
                        "predicted_future_operation_count_topk"
                    )
                    result[name]["decoder_count_log_offset"] = offset
                    result[name]["decoder_count_calibration"] = calibration_details
        if name in source_visibility_names and result[name]["decoder_count_rule"] is None:
            fitted_visibility = _fit_source_visibility_thresholds(
                calibration_values["labels"],
                calibration_values["scores"],
                calibration_values.get(
                    "source_visible",
                    np.zeros_like(calibration_values["labels"], dtype=bool),
                ),
                fallback_threshold=threshold,
            )
            if fitted_visibility is not None:
                visibility_thresholds, visibility_details = fitted_visibility
                visibility_f1 = float(visibility_details["validation_f1"])
                threshold_f1 = float(details.get("validation_f1") or 0.0)
                if visibility_f1 > threshold_f1 + 1e-8:
                    result[name]["decoder_visibility_thresholds"] = visibility_thresholds
                    result[name]["decoder_visibility_calibration"] = visibility_details
    return result


def _strip_calibration_rows(result: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(result)
    clean.pop("calibration_rows", None)
    return clean


def apply_t2_thresholds(result: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    raw = result["calibration_rows"]
    clean = _strip_calibration_rows(result)
    activity: dict[str, Any] = {}
    semantic: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    for name, values in raw.items():
        cfg = calibration[name]
        count_metrics = (
            _count_topk_binary_metrics(
                values["labels"],
                values["scores"],
                values.get("count_log1p", np.empty((0,), dtype=np.float32)),
                values.get("group_ids", np.empty((0,), dtype=np.int64)),
                log_count_offset=float(cfg.get("decoder_count_log_offset", 0.0)),
            )
            if cfg.get("decoder_count_rule") is not None
            else None
        )
        visibility_metrics = (
            _source_visibility_binary_metrics(
                values["labels"],
                values["scores"],
                values.get("source_visible", np.zeros_like(values["labels"], dtype=bool)),
                cfg["decoder_visibility_thresholds"],
            )
            if cfg.get("decoder_visibility_thresholds") is not None
            else None
        )
        metrics = count_metrics or visibility_metrics or _numpy_binary_metrics(
            values["labels"], values["scores"], threshold=cfg["decoder_threshold"]
        )
        persistence = _numpy_binary_metrics(values["labels"], np.zeros_like(values["scores"]), threshold=None)
        history = (
            _numpy_binary_metrics(
                values["labels"], values["history_scores"], threshold=cfg["historical_threshold"]
            )
            if name.startswith("semantic")
            else None
        )
        if name in {"activation", "deactivation"}:
            activity[name] = metrics
            baselines[name] = persistence
        else:
            semantic[name] = metrics
            baselines[name] = {"persistence": persistence, "historical": history}
    clean["activity"] = activity
    clean["node_addition"] = activity["activation"]
    clean["node_removal"] = activity["deactivation"]
    clean["semantic_change"] = semantic["semantic"]
    clean["semantic_feature_change"] = semantic["semantic"]
    clean["semantic_change_stable_activity"] = semantic["semantic_stable_activity"]
    clean["baselines"] = {
        "activity_persistence": {
            "activation": baselines["activation"],
            "deactivation": baselines["deactivation"],
        },
        "node_edit_persistence": {
            "node_addition": baselines["activation"],
            "node_removal": baselines["deactivation"],
        },
        "semantic_persistence": baselines["semantic"]["persistence"],
        "historical_semantic_change": baselines["semantic"]["historical"],
        "semantic_persistence_stable_activity": baselines["semantic_stable_activity"]["persistence"],
        "historical_semantic_change_stable_activity": baselines["semantic_stable_activity"]["historical"],
        "semantic_value": clean["baselines"]["semantic_value"],
    }
    clean["decision_thresholds"] = calibration
    return clean


def train_t2_semantic_one_epoch(
    model: torch.nn.Module,
    dataset: NodePropertyTransitionDataset,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    semantic_threshold: float,
    semantic_topk: int,
    statistics: dict[str, float | int],
    lambda_activity: float = 1.0,
    lambda_semantic_change: float = 1.0,
    lambda_semantic_value: float = 1.0,
    lambda_current_semantic: float = 0.1,
    lambda_latent: float = 1.0,
    semantic_property_mode: str = "logit_residual_mixture",
    latent_cosine_weight: float = 0.0,
    latent_variance_weight: float = 0.0,
    bptt_steps: int = 1,
    grad_clip: float | None = 5.0,
    max_transitions: int | None = None,
) -> dict[str, float]:
    if bptt_steps < 1:
        raise ValueError("bptt_steps must be positive")
    if lambda_current_semantic < 0:
        raise ValueError("lambda_current_semantic must be non-negative")
    model.train()
    hidden = model.initial_hidden(int(dataset.metadata["num_nodes"]), device)
    optimizer.zero_grad(set_to_none=True)
    totals: defaultdict[str, float] = defaultdict(float)
    steps = 0
    latent_steps = 0
    accumulated: torch.Tensor | None = None
    accumulated_steps = 0

    def update() -> None:
        nonlocal accumulated, accumulated_steps, hidden
        if accumulated is None or accumulated_steps == 0:
            return
        (accumulated / accumulated_steps).backward()
        if grad_clip is not None and grad_clip > 0:
            clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        model.update_target_encoder()
        optimizer.zero_grad(set_to_none=True)
        hidden = hidden.detach()
        accumulated = None
        accumulated_steps = 0

    for transition_cpu in dataset.iter_all():
        if transition_cpu["split"] != "train":
            break
        if max_transitions is not None and steps >= int(max_transitions):
            break
        transition = t2_transition_to_device(transition_cpu, device)
        outputs = model(
            transition["x_t"],
            transition["edge_index_t"],
            hidden,
            action=None,
            edge_weight_t=_edge_weight_for_dataset(transition, dataset.dataset_name),
        )
        active_t = transition["node_active_t"]
        active_next = transition["node_active_next"]
        eligible = transition.get("node_edit_eligible_mask")
        if eligible is None:
            eligible = torch.ones_like(active_t, dtype=torch.bool)
        else:
            eligible = eligible.to(torch.bool)
        activation_candidates = (~active_t) & eligible
        deactivation_candidates = active_t & eligible
        activation_loss = _candidate_binary_loss(
            outputs["node_activation_logits"],
            active_next,
            activation_candidates,
            pos_weight=float(statistics["activation_pos_weight"]),
        )
        deactivation_loss = _candidate_binary_loss(
            outputs["node_deactivation_logits"],
            ~active_next,
            deactivation_candidates,
            pos_weight=float(statistics["deactivation_pos_weight"]),
        )
        activity_loss = 0.5 * (activation_loss + deactivation_loss)
        comparable = transition["semantic_comparable_mask"]
        semantic_loss = activity_loss.new_zeros(())
        if bool(comparable.any()):
            semantic_labels, _ = semantic_change_labels(
                transition["semantic_current"][comparable],
                transition["semantic_next"][comparable],
                threshold=semantic_threshold,
                topk=semantic_topk,
            )
            semantic_logits = outputs["node_change_logits"].index_select(
                0, transition["semantic_node_ids"]
            )[comparable]
            semantic_loss = _binary_loss(
                semantic_logits,
                semantic_labels,
                pos_weight=float(statistics["semantic_pos_weight"]),
            )
        semantic_prediction, _, decoded = _semantic_prediction(
            outputs,
            transition,
            property_mode=semantic_property_mode,
        )
        semantic_value_loss = property_prediction_loss(
            decoded,
            semantic_prediction,
            transition["semantic_next"],
            transition["semantic_current"],
            property_mode=semantic_property_mode,
        )
        current_semantic_loss = semantic_value_loss.new_zeros(())
        if lambda_current_semantic > 0:
            current_semantic_loss = _current_semantic_reconstruction_loss(outputs, transition)
        latent_loss = activity_loss.new_zeros(())
        if transition_cpu["latent_train_allowed"]:
            target_z = model.encode_target(
                transition["x_next"],
                transition["edge_index_next"],
                edge_weight_next=_edge_weight_for_dataset(
                    transition, dataset.dataset_name, suffix="next"
                ),
            )
            latent_loss = latent_transition_terms(
                outputs,
                target_z,
                cosine_weight=latent_cosine_weight,
                variance_weight=latent_variance_weight,
            )["total"]
            latent_steps += 1
        total = (
            float(lambda_activity) * activity_loss
            + float(lambda_semantic_change) * semantic_loss
            + float(lambda_semantic_value) * semantic_value_loss
            + float(lambda_current_semantic) * current_semantic_loss
            + float(lambda_latent) * latent_loss
        )
        accumulated = total if accumulated is None else accumulated + total
        accumulated_steps += 1
        hidden = outputs["hidden_next"]
        totals["total"] += float(total.detach())
        totals["activity"] += float(activity_loss.detach())
        totals["semantic_change"] += float(semantic_loss.detach())
        totals["semantic_value"] += float(semantic_value_loss.detach())
        totals["current_semantic"] += float(current_semantic_loss.detach())
        totals["latent"] += float(latent_loss.detach())
        steps += 1
        if accumulated_steps >= bptt_steps:
            update()
    update()
    divisor = max(steps, 1)
    return {
        "total": totals["total"] / divisor,
        "activity": totals["activity"] / divisor,
        "semantic_change": totals["semantic_change"] / divisor,
        "semantic_value": totals["semantic_value"] / divisor,
        "current_semantic": totals["current_semantic"] / divisor,
        "latent": totals["latent"] / divisor,
        "transitions": float(steps),
        "latent_supervised_transitions": float(latent_steps),
    }


__all__ = [
    "apply_t2_thresholds",
    "evaluate_t2_semantic_sequence",
    "fit_t2_training_statistics",
    "select_t2_thresholds",
    "t2_transition_to_device",
    "train_t2_semantic_one_epoch",
]
