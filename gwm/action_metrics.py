
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def edge_codes(edge_index: torch.Tensor, num_nodes: int) -> set[int]:
    if edge_index.numel() == 0:
        return set()
    values = edge_index.detach().cpu().long()
    return set((values[0] * int(num_nodes) + values[1]).tolist())


@dataclass
class SetMetricAccumulator:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    predicted_count: int = 0
    true_count: int = 0

    def update(self, predicted: set[int], target: set[int]) -> None:
        self.true_positive += len(predicted & target)
        self.false_positive += len(predicted - target)
        self.false_negative += len(target - predicted)
        self.predicted_count += len(predicted)
        self.true_count += len(target)

    def metrics(self) -> dict[str, float | int]:
        precision = _safe_div(
            self.true_positive, self.true_positive + self.false_positive
        )
        recall = _safe_div(
            self.true_positive, self.true_positive + self.false_negative
        )
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        union = self.true_positive + self.false_positive + self.false_negative
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": _safe_div(self.true_positive, union),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "predicted_count": self.predicted_count,
            "true_count": self.true_count,
        }


def _binary_rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    two_classes = np.unique(labels).size > 1
    return {
        "auroc": float(roc_auc_score(labels, scores)) if two_classes else None,
        "auprc": float(average_precision_score(labels, scores)) if two_classes else None,
        "positive_rate": float(labels.mean()) if labels.size else None,
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    return (values - float(values.mean())) / max(std, 1e-8)


def _f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if not labels.size:
        return 0.0, 0.0





    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels, dtype=np.int64)
    predicted_positive = np.arange(1, labels.size + 1, dtype=np.int64)
    total_positive = int(sorted_labels.sum())

    tie_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    tp = true_positive[tie_ends].astype(np.float64)
    predicted = predicted_positive[tie_ends].astype(np.float64)
    denominator = predicted + float(total_positive)
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0.0,
    )
    best_index = int(np.argmax(f1))
    best_end = int(tie_ends[best_index])
    return float(sorted_scores[best_end]), float(f1[best_index])


def calibrate_node_change(
    labels: np.ndarray, logits: np.ndarray, past_frequency: np.ndarray
) -> dict[str, Any]:
    labels = labels.astype(np.int64, copy=False)
    logits_z = _standardize(logits)
    history_z = _standardize(np.log1p(np.maximum(past_frequency, 0.0)))
    curve: list[dict[str, float | None]] = []
    for alpha in np.linspace(0.0, 2.0, 17):
        score = logits_z + float(alpha) * history_z
        ranked = _binary_rank_metrics(labels, score)
        threshold, f1 = _f1_threshold(labels, score)
        curve.append(
            {
                "history_weight": float(alpha),
                "threshold": threshold,
                "f1": f1,
                "auprc": ranked["auprc"],
                "auroc": ranked["auroc"],
            }
        )
    selected = max(
        curve,
        key=lambda row: (
            -1.0 if row["auprc"] is None else float(row["auprc"]),
            float(row["f1"]),
            -float(row["history_weight"]),
        ),
    )
    return {
        "fit_split": "validation only",
        "selection_metric": "AUPRC then F1",
        "selected": selected,
        "curve": curve,
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "history_log1p_mean": float(np.log1p(np.maximum(past_frequency, 0.0)).mean()),
        "history_log1p_std": float(np.log1p(np.maximum(past_frequency, 0.0)).std()),
    }


def evaluate_node_change(
    labels: np.ndarray,
    logits: np.ndarray,
    past_frequency: np.ndarray,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    raw = _binary_rank_metrics(labels, logits)
    raw["f1_at_0_logit"] = float(
        f1_score(labels, logits >= 0.0, zero_division=0)
    )
    alpha = float(calibration["selected"]["history_weight"])
    logit_std = max(float(calibration["logit_std"]), 1e-8)
    history_std = max(float(calibration["history_log1p_std"]), 1e-8)
    score = (logits - float(calibration["logit_mean"])) / logit_std
    history = np.log1p(np.maximum(past_frequency, 0.0))
    score += alpha * (
        history - float(calibration["history_log1p_mean"])
    ) / history_std
    calibrated = _binary_rank_metrics(labels, score)
    threshold = float(calibration["selected"]["threshold"])
    calibrated["f1"] = float(
        f1_score(labels, score >= threshold, zero_division=0)
    )
    calibrated["threshold"] = threshold
    calibrated["history_weight"] = alpha
    return {"raw_decoder": raw, "validation_calibrated": calibrated}


def calibrate_node_change_history_blend(
    labels: np.ndarray,
    logits: np.ndarray,
    node_history_features: dict[str, np.ndarray],
    *,
    alpha_min: float = -2.0,
    alpha_max: float = 2.0,
    alpha_steps: int = 41,
) -> dict[str, Any]:
    labels = labels.astype(np.int64, copy=False)
    logits = logits.astype(np.float64, copy=False)
    if labels.ndim != 1 or logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("labels and logits must be matching one-dimensional arrays.")
    if alpha_steps < 2 or alpha_max < alpha_min:
        raise ValueError("history blend needs >=2 alpha steps and max >= min.")

    candidates: list[dict[str, float | str | None]] = []

    def add_candidate(
        feature: str,
        alpha: float,
        scores: np.ndarray,
        *,
        feature_mean: float,
        feature_std: float,
    ) -> None:
        ranked = _binary_rank_metrics(labels, scores)
        threshold, f1 = _f1_threshold(labels, scores)
        candidates.append(
            {
                "feature": feature,
                "alpha": float(alpha),
                "feature_mean": float(feature_mean),
                "feature_std": float(feature_std),
                "threshold": float(threshold),
                "f1": float(f1),
                "auprc": ranked["auprc"],
                "auroc": ranked["auroc"],
            }
        )

    add_candidate(
        "disabled",
        0.0,
        logits,
        feature_mean=0.0,
        feature_std=1.0,
    )
    for name, values in node_history_features.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.shape != logits.shape:
            raise ValueError(
                f"node-history feature {name!r} must match flattened logits."
            )
        mean = float(values.mean())
        std = max(float(values.std()), 1e-8)
        normalized = (values - mean) / std
        for alpha in np.linspace(alpha_min, alpha_max, alpha_steps):
            add_candidate(
                str(name),
                float(alpha),
                logits + float(alpha) * normalized,
                feature_mean=mean,
                feature_std=std,
            )





    selected = max(
        candidates,
        key=lambda row: (
            -1.0 if row["auprc"] is None else float(row["auprc"]),
            float(row["f1"]),
            row["feature"] == "disabled",
            -abs(float(row["alpha"])),
        ),
    )
    return {
        "fit_split": "validation only",
        "selection_metric": "AUPRC then F1",
        "selected": selected,
        "curve": candidates,
    }


def evaluate_node_change_history_blend(
    labels: np.ndarray,
    logits: np.ndarray,
    node_history_features: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    labels = labels.astype(np.int64, copy=False)
    logits = logits.astype(np.float64, copy=False)
    if labels.ndim != 1 or logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("labels and logits must be matching one-dimensional arrays.")
    raw = _binary_rank_metrics(labels, logits)
    raw["f1_at_0_logit"] = float(
        f1_score(labels, logits >= 0.0, zero_division=0)
    )
    selected = calibration["selected"]
    feature = str(selected["feature"])
    if feature == "disabled":
        score = logits
    else:
        if feature not in node_history_features:
            raise ValueError(
                f"Selected history feature {feature!r} is missing during evaluation."
            )
        values = np.asarray(node_history_features[feature], dtype=np.float64)
        if values.ndim != 1 or values.shape != logits.shape:
            raise ValueError(
                f"node-history feature {feature!r} must match flattened logits."
            )
        std = max(float(selected["feature_std"]), 1e-8)
        score = logits + float(selected["alpha"]) * (
            values - float(selected["feature_mean"])
        ) / std
    calibrated = _binary_rank_metrics(labels, score)
    threshold = float(selected["threshold"])
    calibrated["f1"] = float(
        f1_score(labels, score >= threshold, zero_division=0)
    )
    calibrated["threshold"] = threshold
    calibrated["history_feature"] = feature
    calibrated["history_weight"] = float(selected["alpha"])
    return {"raw_decoder": raw, "validation_calibrated": calibrated}


def node_change_calibrated_scores(
    logits: np.ndarray,
    node_history_features: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> np.ndarray:

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional.")
    selected = calibration["selected"]
    feature = str(selected.get("feature", "disabled"))
    if feature == "disabled":
        return logits.copy()
    if feature not in node_history_features:
        raise ValueError(
            f"Selected history feature {feature!r} is missing for calibration."
        )
    values = np.asarray(node_history_features[feature], dtype=np.float64)
    if values.shape != logits.shape:
        raise ValueError("history feature must match logits shape.")
    std = max(float(selected["feature_std"]), 1e-8)
    return logits + float(selected["alpha"]) * (
        values - float(selected["feature_mean"])
    ) / std


def _past_rate_topk_predictions(
    scores: np.ndarray,
    block_sizes: list[int] | tuple[int, ...],
    past_positive_rates: list[float] | tuple[float, ...],
    *,
    scale: float,
) -> np.ndarray:

    scores = np.asarray(scores, dtype=np.float64)
    if len(block_sizes) != len(past_positive_rates):
        raise ValueError("block_sizes and past_positive_rates must have equal length.")
    if sum(int(size) for size in block_sizes) != int(scores.size):
        raise ValueError("block_sizes must partition the score vector.")
    if scale < 0.0:
        raise ValueError("scale must be non-negative.")
    prediction = np.zeros(scores.size, dtype=np.int64)
    offset = 0
    for size, rate in zip(block_sizes, past_positive_rates):
        size = int(size)
        if size < 1:
            continue
        count = int(round(float(np.clip(float(rate) * float(scale), 0.0, 1.0)) * size))
        if count > 0:
            local = np.argsort(-scores[offset : offset + size], kind="mergesort")[:count]
            prediction[offset + local] = 1
        offset += size
    return prediction


def calibrate_past_rate_topk(
    labels: np.ndarray,
    scores: np.ndarray,
    block_sizes: list[int] | tuple[int, ...],
    past_positive_rates: list[float] | tuple[float, ...],
    *,
    scale_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
) -> dict[str, Any]:

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be matching one-dimensional arrays.")
    candidates: list[dict[str, float]] = []
    for scale in scale_grid:
        prediction = _past_rate_topk_predictions(
            scores, block_sizes, past_positive_rates, scale=float(scale)
        )
        true_positive = int(np.logical_and(prediction == 1, labels == 1).sum())
        predicted = int(prediction.sum())
        positive = int(labels.sum())
        precision = _safe_div(true_positive, predicted)
        recall = _safe_div(true_positive, positive)
        candidates.append(
            {
                "scale": float(scale),
                "precision": precision,
                "recall": recall,
                "f1": _safe_div(2.0 * precision * recall, precision + recall),
                "predicted_positive_rate": _safe_div(predicted, labels.size),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (float(row["f1"]), -abs(float(row["scale"]) - 1.0)),
    )
    return {
        "fit_split": "validation only",
        "count_source": "completed-prefix global changed-node rate",
        "selection_metric": "F1",
        "selected": selected,
        "curve": candidates,
    }


def evaluate_past_rate_topk(
    labels: np.ndarray,
    scores: np.ndarray,
    block_sizes: list[int] | tuple[int, ...],
    past_positive_rates: list[float] | tuple[float, ...],
    calibration: dict[str, Any],
) -> dict[str, float | int]:

    labels = np.asarray(labels, dtype=np.int64)
    prediction = _past_rate_topk_predictions(
        scores,
        block_sizes,
        past_positive_rates,
        scale=float(calibration["selected"]["scale"]),
    )
    true_positive = int(np.logical_and(prediction == 1, labels == 1).sum())
    predicted = int(prediction.sum())
    positive = int(labels.sum())
    precision = _safe_div(true_positive, predicted)
    recall = _safe_div(true_positive, positive)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "predicted_positive_rate": _safe_div(predicted, labels.size),
        "true_positive_rate": _safe_div(positive, labels.size),
        "predicted_count": predicted,
        "true_count": positive,
        "scale": float(calibration["selected"]["scale"]),
    }


@dataclass
class RegressionAccumulator:
    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    elements: int = 0
    rows: int = 0
    per_feature_absolute: torch.Tensor | None = None
    per_feature_squared: torch.Tensor | None = None

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("Regression rows must be equal [rows,features] tensors.")
        error = prediction.detach().cpu().float() - target.detach().cpu().float()
        absolute = error.abs().to(torch.float64)
        squared = error.square().to(torch.float64)
        self.absolute_sum += float(absolute.sum().item())
        self.squared_sum += float(squared.sum().item())
        self.elements += int(error.numel())
        self.rows += int(error.shape[0])
        feature_abs = absolute.sum(dim=0)
        feature_sq = squared.sum(dim=0)
        self.per_feature_absolute = (
            feature_abs
            if self.per_feature_absolute is None
            else self.per_feature_absolute + feature_abs
        )
        self.per_feature_squared = (
            feature_sq
            if self.per_feature_squared is None
            else self.per_feature_squared + feature_sq
        )

    def metrics(
        self,
        feature_names: list[str] | None = None,
        *,
        include_per_feature: bool = True,
    ) -> dict[str, Any]:
        mae = _safe_div(self.absolute_sum, self.elements)
        rmse = float(np.sqrt(_safe_div(self.squared_sum, self.elements)))
        result: dict[str, Any] = {"mae": mae, "rmse": rmse, "rows": self.rows}
        if include_per_feature and self.per_feature_absolute is not None and self.rows:
            dimensions = int(self.per_feature_absolute.numel())
            names = feature_names or [f"feature_{index}" for index in range(dimensions)]
            result["per_feature"] = {
                str(names[index]): {
                    "mae": float(self.per_feature_absolute[index].item() / self.rows),
                    "rmse": float(
                        np.sqrt(self.per_feature_squared[index].item() / self.rows)
                    ),
                }
                for index in range(dimensions)
            }
        return result


@dataclass
class PropertyMetricAccumulator:
    target_dim: int
    ndcg_sum: float = 0.0
    rows: int = 0
    property_squared_sum: float = 0.0
    property_elements: int = 0
    delta_absolute_sum: float = 0.0
    delta_squared_sum: float = 0.0
    delta_elements: int = 0
    predicted_top1_counts: np.ndarray = field(init=False)
    true_top1_counts: np.ndarray = field(init=False)
    shared_top1_counts: np.ndarray = field(init=False)
    top1_agreement: int = 0

    def __post_init__(self) -> None:
        self.predicted_top1_counts = np.zeros(self.target_dim, dtype=np.int64)
        self.true_top1_counts = np.zeros(self.target_dim, dtype=np.int64)
        self.shared_top1_counts = np.zeros(self.target_dim, dtype=np.int64)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        current: torch.Tensor,
    ) -> None:
        prediction = prediction.detach().cpu().float()
        target = target.detach().cpu().float()
        current = current.detach().cpu().float()
        if prediction.shape != target.shape or current.shape != target.shape:
            raise ValueError("Property prediction/current/target shapes must match.")
        k = min(10, self.target_dim)
        order = prediction.topk(k, dim=-1).indices
        ideal = target.topk(k, dim=-1).indices
        discount = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
        dcg = (target.gather(1, order) * discount).sum(dim=-1)
        idcg = (target.gather(1, ideal) * discount).sum(dim=-1)
        ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(1e-8), torch.zeros_like(dcg))
        self.ndcg_sum += float(ndcg.sum().item())
        self.rows += int(target.shape[0])
        error = prediction - target
        self.property_squared_sum += float(error.square().sum().item())
        self.property_elements += int(error.numel())
        delta_error = (prediction - current) - (target - current)
        self.delta_absolute_sum += float(delta_error.abs().sum().item())
        self.delta_squared_sum += float(delta_error.square().sum().item())
        self.delta_elements += int(delta_error.numel())
        predicted_top1 = prediction.argmax(dim=-1).numpy()
        true_top1 = target.argmax(dim=-1).numpy()
        self.predicted_top1_counts += np.bincount(
            predicted_top1, minlength=self.target_dim
        )
        self.true_top1_counts += np.bincount(true_top1, minlength=self.target_dim)
        agreement = predicted_top1 == true_top1
        self.shared_top1_counts += np.bincount(
            true_top1[agreement], minlength=self.target_dim
        )
        self.top1_agreement += int((predicted_top1 == true_top1).sum())

    def metrics(self) -> dict[str, Any]:
        predicted_mode = int(self.predicted_top1_counts.argmax())
        true_mode = int(self.true_top1_counts.argmax())
        return {
            "official_ndcg_at_10": _safe_div(self.ndcg_sum, self.rows),
            "property_mse": _safe_div(
                self.property_squared_sum, self.property_elements
            ),
            "delta_mae": _safe_div(self.delta_absolute_sum, self.delta_elements),
            "delta_rmse": float(
                np.sqrt(_safe_div(self.delta_squared_sum, self.delta_elements))
            ),
            "rows": self.rows,
            "ranking_diagnostics": {
                "predicted_top1_unique_count": int(
                    np.count_nonzero(self.predicted_top1_counts)
                ),
                "true_top1_unique_count": int(np.count_nonzero(self.true_top1_counts)),
                "predicted_top1_most_common_coordinate": predicted_mode,
                "true_top1_most_common_coordinate": true_mode,
                "predicted_top1_most_common_fraction": _safe_div(
                    int(self.predicted_top1_counts.max()), self.rows
                ),
                "true_top1_most_common_fraction": _safe_div(
                    int(self.true_top1_counts.max()), self.rows
                ),
                "shared_most_common_top1_ratio": (
                    _safe_div(
                        int(self.shared_top1_counts[true_mode]),
                        self.rows,
                    )
                    if predicted_mode == true_mode
                    else 0.0
                ),
                "top1_row_agreement": _safe_div(self.top1_agreement, self.rows),
            },
        }


__all__ = [
    "PropertyMetricAccumulator",
    "RegressionAccumulator",
    "SetMetricAccumulator",
    "calibrate_node_change",
    "calibrate_node_change_history_blend",
    "calibrate_past_rate_topk",
    "edge_codes",
    "evaluate_node_change",
    "evaluate_node_change_history_blend",
    "evaluate_past_rate_topk",
    "node_change_calibrated_scores",
]
