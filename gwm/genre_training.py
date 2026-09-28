
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .data.tgbn_genre_dataset import TGBNGenreTransitionDataset
from .latent_objective import latent_transition_terms
from .property_transition import (
    decode_property_prediction,
    latent_geometry_loss,
    fit_residual_normalization,
    latent_prediction_metrics,
    property_prediction_loss,
    streaming_delta_diagnostics,
    validate_property_mode,
)


GENRE_TENSOR_KEYS = {
    "x_t",
    "x_t_raw",
    "edge_index_t",
    "edge_weight_t",
    "x_next",
    "x_next_raw",
    "edge_index_next",
    "edge_weight_next",
    "property_state_t",
    "property_node_ids_t",
    "property_observed_t",
    "property_node_ids",
    "property_target",
    "property_current_target",
    "property_current_observed_mask",
}


def genre_transition_to_device(transition: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if key in GENRE_TENSOR_KEYS else value
        for key, value in transition.items()
    }


def _validate_options(property_mode: str, graph_mode: str, property_change_gate: str) -> None:
    validate_property_mode(property_mode)
    if graph_mode not in {"binary", "weighted"}:
        raise ValueError("graph_mode must be binary or weighted")
    if property_change_gate not in {"ungated", "gated"}:
        raise ValueError("property_change_gate must be ungated or gated")
    if property_change_gate == "gated" and property_mode != "delta":
        raise ValueError("property change gating is meaningful only for delta prediction")


def _edge_weight_for_mode(
    transition: dict[str, Any], *, graph_mode: str, suffix: str = "t"
) -> torch.Tensor | None:
    return transition[f"edge_weight_{suffix}"] if graph_mode == "weighted" else None


def _property_prediction(
    outputs: dict[str, torch.Tensor],
    node_ids: torch.Tensor,
    current_target_values: torch.Tensor,
    *,
    property_mode: str,
    property_change_gate: str,
    current_observed_mask: torch.Tensor | None = None,
    residual_normalization: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if "node_property" not in outputs:
        raise ValueError("GraphWorldModel was not initialized with node_property_dim.")
    decoded = outputs["node_property"].index_select(0, node_ids)
    if property_mode == "absolute":
        prediction, delta = decode_property_prediction(
            decoded,
            current_target_values,
            property_mode=property_mode,
            residual_normalization=residual_normalization,
        )
        return prediction, delta, None, decoded
    if property_mode == "logit_mixture":
        if current_observed_mask is None:
            raise ValueError("logit_mixture requires the current-Y availability mask.")



        change_probability = outputs["node_change_probability"].index_select(0, node_ids)
        prediction, delta = decode_property_prediction(
            decoded,
            current_target_values,
            property_mode=property_mode,
            residual_normalization=residual_normalization,
            transition_gate=change_probability,
            current_observed_mask=current_observed_mask,
        )
        return prediction, delta, change_probability, decoded
    change_probability: torch.Tensor | None = None
    predicted_delta = decoded
    if property_change_gate == "gated":
        change_probability = outputs["node_change_probability"].index_select(0, node_ids)
        predicted_delta = change_probability.unsqueeze(-1) * decoded
    prediction, delta = decode_property_prediction(
        predicted_delta,
        current_target_values,
        property_mode=property_mode,
        residual_normalization=residual_normalization,
    )
    return prediction, delta, change_probability, predicted_delta


def _official_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dataset_name: str = "tgbn-genre",
    chunk_rows: int = 4096,
) -> float:
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("Official NDCG requires equally shaped [rows, coordinates] arrays.")
    if y_true.shape[0] == 0:
        raise ValueError("Official NDCG requires at least one labelled row.")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    from tgb.nodeproppred.evaluate import Evaluator

    evaluator = Evaluator(name=dataset_name)
    label_count = int(y_true.shape[1])
    k = min(10, label_count)
    discount = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    weighted_sum = 0.0
    total_rows = int(y_true.shape[0])
    for start in range(0, total_rows, int(chunk_rows)):
        stop = min(start + int(chunk_rows), total_rows)
        truth = np.asarray(y_true[start:stop])
        score = np.asarray(y_pred[start:stop])








        frontier = min(k + 1, label_count)
        frontier_indices = np.argpartition(
            -score, kth=frontier - 1, axis=1
        )[:, :frontier]
        frontier_scores = np.take_along_axis(score, frontier_indices, axis=1)
        frontier_scores.sort(axis=1)
        frontier_scores = frontier_scores[:, ::-1]
        tied = np.any(frontier_scores[:, :k - 1] == frontier_scores[:, 1:k], axis=1)
        if label_count > k:
            tied |= frontier_scores[:, k - 1] == frontier_scores[:, k]
        no_tie = ~tied

        if bool(no_tie.any()):
            no_tie_truth = truth[no_tie].astype(np.float64, copy=False)
            no_tie_score = score[no_tie]
            top_indices = np.argpartition(
                -no_tie_score, kth=k - 1, axis=1
            )[:, :k]
            top_scores = np.take_along_axis(no_tie_score, top_indices, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top_indices = np.take_along_axis(top_indices, order, axis=1)
            ranked_gain = np.take_along_axis(no_tie_truth, top_indices, axis=1)
            dcg = ranked_gain @ discount

            ideal_gain = np.partition(
                no_tie_truth, kth=label_count - k, axis=1
            )[:, -k:]
            ideal_gain.sort(axis=1)
            ideal_gain = ideal_gain[:, ::-1]
            ideal_dcg = ideal_gain @ discount
            row_ndcg = np.divide(
                dcg,
                ideal_dcg,
                out=np.zeros_like(dcg),
                where=ideal_dcg != 0,
            )
            weighted_sum += float(row_ndcg.sum(dtype=np.float64))

        if bool(tied.any()):

            result = evaluator.eval(
                {
                    "y_true": truth[tied],
                    "y_pred": score[tied],
                    "eval_metric": ["ndcg"],
                }
            )
            weighted_sum += float(result["ndcg"]) * int(tied.sum())
    return float(weighted_sum / total_rows)


def _delta_magnitudes(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return (target - current).abs().mean(dim=1)


def train_large_change_threshold(
    dataset: TGBNGenreTransitionDataset, *, quantile: float = 0.75
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    cache_key = f"_genre_large_change_q{quantile}"
    cached = getattr(dataset, cache_key, None)
    if cached is not None:
        return float(cached)
    magnitudes: list[np.ndarray] = []
    for transition in dataset.iter_property_only():
        if transition["split"] != "train":
            break
        magnitudes.append(
            _delta_magnitudes(
                transition["property_target"], transition["property_current_target"]
            ).numpy()
        )
    if not magnitudes:
        raise ValueError("Cannot derive a large-change threshold without train labels.")
    threshold = float(np.quantile(np.concatenate(magnitudes), quantile))
    setattr(dataset, cache_key, threshold)
    return threshold


def train_residual_normalization(
    dataset: TGBNGenreTransitionDataset, *, min_std: float = 1e-6
) -> dict[str, Any]:
    cache_key = f"_genre_residual_normalization_std{min_std}"
    cached = getattr(dataset, cache_key, None)
    if cached is not None:
        return cached

    def train_deltas() -> Any:
        for transition in dataset.iter_property_only():
            if transition["split"] != "train":
                break
            yield (
                transition["property_target"].float()
                - transition["property_current_target"].float()
            )

    result = fit_residual_normalization(train_deltas(), min_std=min_std)
    setattr(dataset, cache_key, result)
    return result


def train_property_change_statistics(
    dataset: TGBNGenreTransitionDataset, *, epsilon: float = 1e-8
) -> dict[str, Any]:
    cache_key = f"_genre_change_stats_eps{epsilon}"
    cached = getattr(dataset, cache_key, None)
    if cached is not None:
        return cached
    comparable = 0
    changed = 0
    target_rows = 0
    for transition in dataset.iter_property_only():
        if transition["split"] != "train":
            break
        target_rows += int(transition["property_target"].shape[0])
        mask = transition["property_current_observed_mask"]
        comparable += int(mask.sum())
        if mask.any():
            delta = transition["property_target"][mask] - transition["property_current_target"][mask]
            changed += int((delta.abs().amax(dim=1) > epsilon).sum())
    result = {
        "definition": (
            "same labelled user at consecutive official times and max(abs(Y_(t+1)-Y_t)) "
            f"> {epsilon}"
        ),
        "epsilon": float(epsilon),
        "train_target_rows": target_rows,
        "train_comparable_rows": comparable,
        "train_comparable_fraction": float(comparable / max(target_rows, 1)),
        "train_changed_comparable_rows": changed,
        "train_changed_positive_rate": float(changed / max(comparable, 1)),
        "gate_recommended": bool(
            comparable >= 100
            and 0.01 < (changed / max(comparable, 1)) < 0.99
        ),
    }
    setattr(dataset, cache_key, result)
    return result


def resolve_property_change_gate(
    requested: str, stats: dict[str, Any], *, property_mode: str
) -> str:
    if requested not in {"auto", "ungated", "gated"}:
        raise ValueError("property_change_gate must be auto/ungated/gated")
    if property_mode != "delta":
        return "ungated"
    if requested == "auto":
        return "gated" if bool(stats["gate_recommended"]) else "ungated"
    return requested


def _balanced_change_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positives = target.sum()
    negatives = target.numel() - positives
    if positives > 0 and negatives > 0:
        pos_weight = (negatives / positives).detach()
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(logits, target)


def _change_loss(
    outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    *,
    epsilon: float,
) -> tuple[torch.Tensor, int, int]:
    mask = transition["property_current_observed_mask"]
    if not bool(mask.any()):
        return outputs["node_change_logits"].new_zeros(()), 0, 0
    target = transition["property_target"]
    current = transition["property_current_target"]
    changed = (target[mask] - current[mask]).abs().amax(dim=1).gt(epsilon).float()
    logits = outputs["node_change_logits"].index_select(0, transition["property_node_ids"])[mask]
    return _balanced_change_bce(logits, changed), int(mask.sum()), int(changed.sum())


def _change_focused_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_current: np.ndarray,
    comparable_mask: np.ndarray,
    *,
    large_change_threshold: float,
    change_epsilon: float,
    predictor_name: str = "gwm_delta",
) -> dict[str, Any]:
    diagnostics = streaming_delta_diagnostics(
        y_true,
        y_pred,
        y_current,
        large_change_threshold=large_change_threshold,
        epsilon=change_epsilon,
        comparable_mask=comparable_mask,
    )
    all_rows = diagnostics["all"]
    assert all_rows is not None
    large = diagnostics["large"]
    meaningful_large = diagnostics["meaningful_large"]
    meaningful_changed = diagnostics["changed"]
    distribution = diagnostics["distribution"]
    result: dict[str, Any] = {
        "delta_mae": all_rows["prediction"]["mae"],
        "delta_rmse": all_rows["prediction"]["rmse"],
        "zero_delta_copy_last_mae": all_rows["zero_delta_copy_last"]["mae"],
        "zero_delta_copy_last_rmse": all_rows["zero_delta_copy_last"]["rmse"],
        "delta_magnitude": all_rows["delta_magnitude"],
        "property_transition_distribution": distribution,
        "change_focused": {
            "large_change_row_definition": "mean(abs(Y_(t+1)-Y_t)) across 513 official coordinates",
            "large_change_threshold_train_q75": float(large_change_threshold),
            "large_change_rows": int(distribution["large_change_rows"]),
            "large_change_row_fraction": float(distribution["large_change_ratio"]),
            "comparable_large_change_rows": int(distribution["meaningful_large_change_rows"]),
            "comparable_large_change_row_fraction": float(
                distribution["meaningful_large_change_ratio"]
            ),
            "comparable_changed_row_definition": (
                "official Y_t exists for the same user and max(abs(Y_(t+1)-Y_t)) "
                f"> {change_epsilon}"
            ),
            "comparable_changed_rows": int(distribution["meaningful_changed_rows"]),
            "comparable_changed_row_fraction": float(distribution["meaningful_changed_ratio"]),
        },
    }
    if large is not None:
        result["change_focused"]["large_change"] = {
            predictor_name: large["prediction"],
            "zero_delta_copy_last": large["zero_delta_copy_last"],
            "delta_magnitude": large["delta_magnitude"],
            "rows": large["rows"],
        }
    else:
        result["change_focused"]["large_change"] = None
    if meaningful_large is not None:
        result["change_focused"]["comparable_large_change"] = {
            predictor_name: meaningful_large["prediction"],
            "zero_delta_copy_last": meaningful_large["zero_delta_copy_last"],
            "delta_magnitude": meaningful_large["delta_magnitude"],
            "rows": meaningful_large["rows"],
            "row_definition": (
                "official Y_t exists for the same user and mean(abs(Y_(t+1)-Y_t)) "
                "is at least the train-only q75 threshold"
            ),
        }
    else:
        result["change_focused"]["comparable_large_change"] = None
    if meaningful_changed is not None:
        result["change_focused"]["comparable_changed"] = {
            predictor_name: meaningful_changed["prediction"],
            "zero_delta_copy_last": meaningful_changed["zero_delta_copy_last"],
            "delta_magnitude": meaningful_changed["delta_magnitude"],
            "rows": meaningful_changed["rows"],
        }
    else:
        result["change_focused"]["comparable_changed"] = None
    return result


def official_ndcg_by_current_y_availability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    comparable_mask: np.ndarray,
    *,
    dataset_name: str = "tgbn-genre",
) -> dict[str, Any]:
    def masked_ndcg(mask: np.ndarray, *, chunk_rows: int = 4096) -> float | None:
        indices = np.flatnonzero(mask)
        if not indices.size:
            return None
        weighted_total = 0.0
        for start in range(0, int(indices.size), chunk_rows):
            part = indices[start : start + chunk_rows]
            weighted_total += _official_ndcg(
                y_true[part], y_pred[part], dataset_name=dataset_name
            ) * len(part)
        return float(weighted_total / indices.size)

    result: dict[str, Any] = {
        "all_rows": _official_ndcg(y_true, y_pred, dataset_name=dataset_name),
        "current_y_available_rows": int(comparable_mask.sum()),
        "current_y_missing_rows": int((~comparable_mask).sum()),
        "availability_subset_evaluation": "rowwise NDCG@10 averaged in 4096-row chunks to bound memory",
    }
    result["current_y_available_ndcg"] = masked_ndcg(comparable_mask)
    result["current_y_missing_ndcg"] = masked_ndcg(~comparable_mask)
    return result


def ranking_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, *, topk: int = 10) -> dict[str, Any]:
    if y_true.ndim != 2 or y_pred.shape != y_true.shape:
        raise ValueError("ranking diagnostics require equally shaped dense target matrices")
    predicted_top1 = y_pred.argmax(axis=1)
    true_top1 = y_true.argmax(axis=1)
    predicted_counts = np.bincount(predicted_top1, minlength=y_pred.shape[1])
    true_counts = np.bincount(true_top1, minlength=y_true.shape[1])
    predicted_mode = int(predicted_counts.argmax())
    true_mode = int(true_counts.argmax())
    k = min(int(topk), int(y_true.shape[1]))
    predicted_topk = np.argpartition(y_pred, -k, axis=1)[:, -k:]
    true_topk = np.argpartition(y_true, -k, axis=1)[:, -k:]
    shared_mode_rows = (predicted_top1 == true_mode) & (true_top1 == true_mode)
    return {
        "topk": k,
        "predicted_top1_unique_coordinates": int(np.unique(predicted_top1).size),
        "true_top1_unique_coordinates": int(np.unique(true_top1).size),
        "predicted_topk_unique_coordinates": int(np.unique(predicted_topk).size),
        "true_topk_unique_coordinates": int(np.unique(true_topk).size),
        "predicted_top1_most_common_coordinate": predicted_mode,
        "true_top1_most_common_coordinate": true_mode,
        "predicted_top1_most_common_fraction": float(predicted_counts.max() / y_pred.shape[0]),
        "true_top1_most_common_fraction": float(true_counts.max() / y_true.shape[0]),
        "shared_most_common_top1_ratio": float(
            shared_mode_rows.mean() if predicted_mode == true_mode else 0.0
        ),
        "top1_row_agreement": float((predicted_top1 == true_top1).mean()),
    }


def train_tgbn_genre_one_epoch(
    model: torch.nn.Module,
    dataset: TGBNGenreTransitionDataset,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    lambda_latent: float = 1.0,
    latent_cosine_weight: float = 0.0,
    latent_variance_weight: float = 0.0,
    lambda_latent_geometry: float = 0.0,
    latent_geometry_nodes: int = 256,
    lambda_property: float = 1.0,
    lambda_current_property: float = 0.0,
    lambda_change: float = 1.0,
    property_mode: str = "delta",
    property_change_gate: str = "ungated",
    graph_mode: str = "binary",
    residual_normalization: dict[str, Any] | None = None,
    large_change_threshold: float | None = None,
    large_change_loss_weight: float = 1.0,
    change_epsilon: float = 1e-8,
    bptt_steps: int = 1,
    grad_clip: float | None = 5.0,
    max_transitions: int | None = None,
) -> dict[str, float]:
    _validate_options(property_mode, graph_mode, property_change_gate)
    if bptt_steps < 1:
        raise ValueError("bptt_steps must be at least one")
    if lambda_latent_geometry < 0:
        raise ValueError("lambda_latent_geometry must be non-negative")
    if lambda_current_property < 0:
        raise ValueError("lambda_current_property must be non-negative")
    if lambda_current_property > 0 and property_mode not in {
        "logit_absolute",
        "logit_mixture",
    }:
        raise ValueError(
            "Current-Y reconstruction is defined only for simplex absolute/mixture decoding."
        )
    model.train()
    hidden = model.initial_hidden(int(dataset.metadata["num_nodes"]), device)
    sums: defaultdict[str, float] = defaultdict(float)
    steps = 0
    latent_steps = 0
    change_rows = 0
    changed_rows = 0
    optimizer.zero_grad(set_to_none=True)
    accumulated_total: torch.Tensor | None = None
    accumulated_steps = 0

    def optimizer_update() -> None:
        nonlocal accumulated_total, accumulated_steps, hidden
        if accumulated_total is None or accumulated_steps == 0:
            return
        (accumulated_total / accumulated_steps).backward()
        if grad_clip is not None and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        update_target = getattr(model, "update_target_encoder", None)
        if callable(update_target):
            update_target()
        optimizer.zero_grad(set_to_none=True)



        hidden = hidden.detach()
        accumulated_total = None
        accumulated_steps = 0

    for transition_cpu in dataset.iter_all():
        if transition_cpu["split"] != "train":
            break
        if max_transitions is not None and steps >= max_transitions:
            break
        transition = genre_transition_to_device(transition_cpu, device)
        outputs = model(
            transition["x_t"],
            transition["edge_index_t"],
            hidden,
            action=None,
            edge_weight_t=_edge_weight_for_mode(transition, graph_mode=graph_mode),
        )
        prediction, _, _, decoded = _property_prediction(
            outputs,
            transition["property_node_ids"],
            transition["property_current_target"],
            property_mode=property_mode,
            property_change_gate=property_change_gate,
            current_observed_mask=transition["property_current_observed_mask"],
            residual_normalization=residual_normalization,
        )
        property_loss = property_prediction_loss(
            decoded,
            prediction,
            transition["property_target"],
            transition["property_current_target"],
            property_mode=property_mode,
            residual_normalization=residual_normalization,
            large_change_threshold=large_change_threshold,
            large_change_loss_weight=large_change_loss_weight,
        )
        current_property_loss = property_loss.new_zeros(())
        if lambda_current_property > 0:
            current_logits = outputs["node_property_current"].index_select(
                0, transition["property_node_ids_t"]
            )
            current_target = transition["property_observed_t"]
            current_property_loss = -(
                current_target * F.log_softmax(current_logits, dim=-1)
            ).sum(dim=1).mean()
        latent_loss = property_loss.new_zeros(())
        geometry_loss = property_loss.new_zeros(())
        if transition_cpu["latent_train_allowed"]:
            target_z = model.encode_target(
                transition["x_next"],
                transition["edge_index_next"],
                edge_weight_next=_edge_weight_for_mode(transition, graph_mode=graph_mode, suffix="next"),
            )
            latent_loss = latent_transition_terms(
                outputs,
                target_z,
                cosine_weight=latent_cosine_weight,
                variance_weight=latent_variance_weight,
            )["total"]
            if lambda_latent_geometry > 0:
                geometry_loss = latent_geometry_loss(
                    outputs["latent_mu"], target_z, max_nodes=latent_geometry_nodes
                )
            latent_steps += 1
        change_loss = property_loss.new_zeros(())
        if property_change_gate == "gated":
            change_loss, count, positive = _change_loss(
                outputs, transition, epsilon=change_epsilon
            )
            change_rows += count
            changed_rows += positive
        total = (
            lambda_property * property_loss
            + lambda_current_property * current_property_loss
            + lambda_latent * latent_loss
            + lambda_latent_geometry * geometry_loss
            + lambda_change * change_loss
        )
        accumulated_total = total if accumulated_total is None else accumulated_total + total
        accumulated_steps += 1
        hidden = outputs["hidden_next"]
        sums["total"] += float(total.detach().item())
        sums["property"] += float(property_loss.detach().item())
        sums["current_property"] += float(current_property_loss.detach().item())
        sums["latent"] += float(latent_loss.detach().item())
        sums["geometry"] += float(geometry_loss.detach().item())
        sums["change"] += float(change_loss.detach().item())
        steps += 1
        if accumulated_steps >= bptt_steps:
            optimizer_update()

    optimizer_update()
    divisor = max(steps, 1)
    return {
        "total": sums["total"] / divisor,
        "property": sums["property"] / divisor,
        "current_property": sums["current_property"] / divisor,
        "latent": sums["latent"] / divisor,
        "geometry": sums["geometry"] / divisor,
        "change": sums["change"] / divisor,
        "transitions": float(steps),
        "latent_supervised_transitions": float(latent_steps),
        "change_supervised_rows": float(change_rows),
        "change_positive_rate": float(changed_rows / max(change_rows, 1)),
        "bptt_steps": float(bptt_steps),
    }


@torch.no_grad()
def evaluate_tgbn_genre_sequence(
    model: torch.nn.Module,
    dataset: TGBNGenreTransitionDataset,
    *,
    split: str,
    device: torch.device,
    property_mode: str = "delta",
    property_change_gate: str = "ungated",
    graph_mode: str = "binary",
    residual_normalization: dict[str, Any] | None = None,
    change_epsilon: float = 1e-8,
    large_change_threshold: float | None = None,
    compute_latent: bool = True,
    return_cases: bool = False,
    include_availability_diagnostics: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    _validate_options(property_mode, graph_mode, property_change_gate)
    model.eval()
    hidden = model.initial_hidden(int(dataset.metadata["num_nodes"]), device)
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    current_states: list[torch.Tensor] = []
    comparable_masks: list[torch.Tensor] = []
    latent_targets: list[torch.Tensor] = []
    latent_predictions: list[torch.Tensor] = []
    latent_means: list[torch.Tensor] = []
    property_losses: list[float] = []
    cases: list[dict[str, Any]] = []
    for transition_cpu in dataset.iter_all():
        transition = genre_transition_to_device(transition_cpu, device)
        outputs = model(
            transition["x_t"],
            transition["edge_index_t"],
            hidden,
            action=None,
            edge_weight_t=_edge_weight_for_mode(transition, graph_mode=graph_mode),
        )
        hidden = outputs["hidden_next"].detach()
        if transition_cpu["split"] != split:
            continue
        prediction, _, change_probability, _ = _property_prediction(
            outputs,
            transition["property_node_ids"],
            transition["property_current_target"],
            property_mode=property_mode,
            property_change_gate=property_change_gate,
            current_observed_mask=transition["property_current_observed_mask"],
            residual_normalization=residual_normalization,
        )
        target = transition["property_target"]
        targets.append(target.cpu())
        predictions.append(prediction.cpu())
        current_states.append(transition["property_current_target"].cpu())
        comparable_masks.append(transition["property_current_observed_mask"].cpu())
        property_losses.append(float(F.mse_loss(prediction, target).item()))
        if compute_latent:
            target_z = model.encode_target(
                transition["x_next"],
                transition["edge_index_next"],
                edge_weight_next=_edge_weight_for_mode(transition, graph_mode=graph_mode, suffix="next"),
            )
            latent_targets.append(target_z.cpu())
            latent_predictions.append(outputs["latent_next"].cpu())
            latent_means.append(outputs["latent_mu"].cpu())
        if return_cases and len(cases) < 2 and target.shape[0]:
            k = min(10, int(prediction.shape[1]))
            cases.append(
                {
                    "transition_id": int(transition_cpu["transition_id"]),
                    "label_time": int(transition_cpu["label_time"]),
                    "labelled_node": int(transition_cpu["property_node_ids"][0]),
                    "top_predicted_target_coordinates": torch.topk(prediction[0], k=k).indices.tolist(),
                    "top_actual_target_coordinates": torch.topk(target[0], k=k).indices.tolist(),
                    "predicted_change_probability": (
                        None if change_probability is None else float(change_probability[0].item())
                    ),
                }
            )
    if not targets:
        raise ValueError(f"No official tgbn-genre labels found in split={split}.")
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(predictions).numpy()
    y_current = torch.cat(current_states).numpy()
    comparable = torch.cat(comparable_masks).numpy().astype(bool, copy=False)
    if large_change_threshold is None:
        large_change_threshold = train_large_change_threshold(dataset)
    result: dict[str, Any] = {
        "dataset": str(dataset.metadata.get("dataset_name", "tgbn-genre")),
        "split": split,
        "property_mode": property_mode,
        "property_change_gate": property_change_gate,
        "graph_mode": graph_mode,
        "official_ndcg": _official_ndcg(
            y_true, y_pred, dataset_name=str(dataset.metadata.get("dataset_name", "tgbn-genre"))
        ),
        "property_mse": float(np.mean((y_pred - y_true) ** 2)),
        "property_loss_mean_over_label_times": float(np.mean(property_losses)),
        "label_rows": int(y_true.shape[0]),
        "target_dim": int(y_true.shape[1]),
        "target_nonzero_fraction": float((y_true != 0).mean()),
        "current_y_available_fraction": float(comparable.mean()),
        "cases": cases,
    }
    if include_availability_diagnostics:
        result["official_ndcg_by_current_y_availability"] = official_ndcg_by_current_y_availability(
            y_true,
            y_pred,
            comparable,
            dataset_name=str(dataset.metadata.get("dataset_name", "tgbn-genre")),
        )
    else:
        result["official_ndcg_by_current_y_availability"] = {
            "not_computed": "disabled for memory-bounded property-transition evaluation",
            "current_y_available_rows": int(comparable.sum()),
            "current_y_missing_rows": int((~comparable).sum()),
        }
    result["ranking_diagnostics"] = ranking_diagnostics(y_true, y_pred)
    result.update(
        _change_focused_payload(
            y_true,
            y_pred,
            y_current,
            comparable,
            large_change_threshold=float(large_change_threshold),
            change_epsilon=change_epsilon,
        )
    )
    if latent_targets:
        z_true = torch.cat(latent_targets).numpy()
        z_pred = torch.cat(latent_predictions).numpy()
        z_mean = torch.cat(latent_means).numpy()
        sample_metrics = latent_prediction_metrics(z_true, z_pred)
        mean_metrics = latent_prediction_metrics(z_true, z_mean)
        result["latent_mse"] = sample_metrics["mse"]
        result["latent_cosine_similarity"] = sample_metrics["cosine_similarity"]
        result["latent_prediction"] = {
            "zhat_next": sample_metrics,
            "latent_mu": mean_metrics,
            "target": "stop-gradient z_(t+1)=GraphEncoder(g_(t+1))",
        }
    else:
        result["latent_mse"] = None
        result["latent_cosine_similarity"] = None
        result["latent_prediction"] = None
    return result


def format_tgbn_genre_epoch_log(
    epoch: int, train: dict[str, float], validation: dict[str, Any]
) -> str:
    return (
        f"epoch={epoch:03d} train_total={train['total']:.6f} "
        f"property_mse={train['property']:.6f} latent={train['latent']:.6f} "
        f"change={train['change']:.6f} | val_official_ndcg={validation['official_ndcg']:.6f} "
        f"val_property_mse={validation['property_mse']:.6f} "
        f"val_delta_rmse={validation['delta_rmse']:.6f}"
    )


__all__ = [
    "evaluate_tgbn_genre_sequence",
    "format_tgbn_genre_epoch_log",
    "genre_transition_to_device",
    "official_ndcg_by_current_y_availability",
    "ranking_diagnostics",
    "resolve_property_change_gate",
    "train_large_change_threshold",
    "train_property_change_statistics",
    "train_residual_normalization",
    "train_tgbn_genre_one_epoch",
]
