
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F










PROPERTY_MODES = {
    "absolute",
    "delta",
    "residual",
    "logit_absolute",
    "logit_residual",
    "logit_mixture",
    "logit_residual_mixture",
}


def validate_property_mode(property_mode: str) -> None:
    if property_mode not in PROPERTY_MODES:
        raise ValueError(f"property_mode must be one of {sorted(PROPERTY_MODES)}")


def fit_residual_normalization(
    delta_batches: Iterable[torch.Tensor], *, min_std: float = 1e-6
) -> dict[str, Any]:
    if min_std <= 0:
        raise ValueError("min_std must be positive")
    total: torch.Tensor | None = None
    square_total: torch.Tensor | None = None
    rows = 0
    for delta in delta_batches:
        if delta.ndim != 2:
            raise ValueError("Residual batches must have shape [rows, target_dim].")
        values = delta.detach().to(dtype=torch.float64, device="cpu")
        if total is None:
            total = torch.zeros(values.shape[1], dtype=torch.float64)
            square_total = torch.zeros_like(total)
        assert square_total is not None
        total += values.sum(dim=0)
        square_total += values.square().sum(dim=0)
        rows += int(values.shape[0])
    if total is None or square_total is None or rows == 0:
        raise ValueError("Cannot fit residual normalization without training deltas.")
    mean = total / rows
    variance = (square_total / rows - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(float(min_std))
    return {
        "mean": mean.to(torch.float32),
        "std": std.to(torch.float32),
        "rows": int(rows),
        "min_std": float(min_std),
    }


def residual_normalization_to_device(
    stats: dict[str, Any] | None, device: torch.device | str
) -> dict[str, Any] | None:
    if stats is None:
        return None
    if "mean" not in stats or "std" not in stats:
        raise ValueError("Residual normalization must contain mean and std tensors.")
    return {
        **stats,
        "mean": torch.as_tensor(stats["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(stats["std"], dtype=torch.float32, device=device),
    }


def residual_normalization_summary(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if stats is None:
        return None
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).cpu()
    std = torch.as_tensor(stats["std"], dtype=torch.float32).cpu()
    return {
        "fit_split": "train only",
        "rows": int(stats["rows"]),
        "target_dim": int(mean.numel()),
        "min_std_floor": float(stats["min_std"]),
        "mean_abs_mean": float(mean.abs().mean().item()),
        "mean_min": float(mean.min().item()),
        "mean_max": float(mean.max().item()),
        "std_mean": float(std.mean().item()),
        "std_min": float(std.min().item()),
        "std_max": float(std.max().item()),
    }


def decode_property_prediction(
    decoded: torch.Tensor,
    current: torch.Tensor,
    *,
    property_mode: str,
    residual_normalization: dict[str, Any] | None = None,
    transition_gate: torch.Tensor | None = None,
    current_observed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_property_mode(property_mode)
    if decoded.shape != current.shape:
        raise ValueError("Decoded property and current property must have identical shapes.")
    if property_mode == "absolute":
        return decoded, decoded - current
    if property_mode == "delta":
        return current + decoded, decoded
    if property_mode == "logit_absolute":
        prediction = torch.softmax(decoded, dim=-1)
        return prediction, prediction - current
    if property_mode == "logit_residual":





        base_logits = torch.log(current.clamp_min(1e-8))
        prediction = torch.softmax(base_logits + decoded, dim=-1)
        return prediction, prediction - current
    if property_mode == "logit_mixture":
        if transition_gate is None:
            raise ValueError("logit_mixture requires a transition gate from Zhat.")
        if transition_gate.ndim != 1 or transition_gate.shape[0] != decoded.shape[0]:
            raise ValueError("transition_gate must have one value per property row.")
        if current_observed_mask is None:
            raise ValueError("logit_mixture requires the observed-current-Y mask.")
        observed = current_observed_mask.to(device=decoded.device, dtype=torch.bool)
        if observed.ndim != 1 or observed.shape[0] != decoded.shape[0]:
            raise ValueError("current_observed_mask must have one value per property row.")



        absolute = torch.softmax(decoded, dim=-1)
        gate = transition_gate.clamp(0.0, 1.0).unsqueeze(-1)
        mixed = (1.0 - gate) * current + gate * absolute
        prediction = torch.where(observed.unsqueeze(-1), mixed, absolute)
        return prediction, prediction - current
    if property_mode == "logit_residual_mixture":
        if transition_gate is None:
            raise ValueError(
                "logit_residual_mixture requires a transition gate from Zhat."
            )
        if transition_gate.ndim != 1 or transition_gate.shape[0] != decoded.shape[0]:
            raise ValueError("transition_gate must have one value per property row.")
        if current_observed_mask is None:
            raise ValueError(
                "logit_residual_mixture requires the observed-current-Y mask."
            )
        observed = current_observed_mask.to(device=decoded.device, dtype=torch.bool)
        if observed.ndim != 1 or observed.shape[0] != decoded.shape[0]:
            raise ValueError(
                "current_observed_mask must have one value per property row."
            )




        residual = torch.softmax(
            torch.log(current.clamp_min(1e-8)) + decoded, dim=-1
        )
        absolute = torch.softmax(decoded, dim=-1)
        gate = transition_gate.clamp(0.0, 1.0).unsqueeze(-1)
        mixed = (1.0 - gate) * current + gate * residual
        prediction = torch.where(observed.unsqueeze(-1), mixed, absolute)
        return prediction, prediction - current
    if residual_normalization is None:
        raise ValueError("residual mode requires training-only residual normalization.")
    mean = torch.as_tensor(residual_normalization["mean"], device=decoded.device)
    std = torch.as_tensor(residual_normalization["std"], device=decoded.device)
    if mean.numel() != decoded.shape[1] or std.numel() != decoded.shape[1]:
        raise ValueError("Residual normalization target dimension does not match decoder output.")
    delta = decoded * std.unsqueeze(0) + mean.unsqueeze(0)
    return current + delta, delta


def property_prediction_loss(
    decoded: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    *,
    property_mode: str,
    residual_normalization: dict[str, Any] | None = None,
    large_change_threshold: float | None = None,
    large_change_loss_weight: float = 1.0,
) -> torch.Tensor:
    validate_property_mode(property_mode)
    if large_change_loss_weight <= 0:
        raise ValueError("large_change_loss_weight must be positive")
    if property_mode in {
        "logit_absolute",
        "logit_residual",
        "logit_mixture",
        "logit_residual_mixture",
    }:




        per_row = -(target * prediction.clamp_min(1e-8).log()).sum(dim=1)
    elif property_mode == "residual":
        if residual_normalization is None:
            raise ValueError("residual mode requires training-only residual normalization.")
        mean = torch.as_tensor(residual_normalization["mean"], device=decoded.device)
        std = torch.as_tensor(residual_normalization["std"], device=decoded.device)
        target_for_loss = (target - current - mean.unsqueeze(0)) / std.unsqueeze(0)
        per_row = (decoded - target_for_loss).square().mean(dim=1)
    else:
        per_row = (prediction - target).square().mean(dim=1)
    if large_change_loss_weight == 1.0:
        return per_row.mean()
    if large_change_threshold is None:
        raise ValueError("A train-derived large-change threshold is required for weighted loss.")
    magnitude = (target - current).abs().mean(dim=1)
    weights = torch.where(
        magnitude >= float(large_change_threshold),
        torch.full_like(magnitude, float(large_change_loss_weight)),
        torch.ones_like(magnitude),
    )
    return (weights * per_row).sum() / weights.sum().clamp_min(1.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mse": float(np.mean(error**2)),
    }


def latent_prediction_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> dict[str, float]:
    if z_true.shape != z_pred.shape or z_true.ndim != 2:
        raise ValueError("Latent targets/predictions must be equal rank-2 arrays.")
    mse = float(np.mean((z_pred - z_true) ** 2))
    denominator = np.linalg.norm(z_true, axis=1) * np.linalg.norm(z_pred, axis=1)
    cosine = np.divide(
        np.sum(z_true * z_pred, axis=1),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float32),
        where=denominator > 1e-12,
    )
    return {"mse": mse, "cosine_similarity": float(np.mean(cosine))}


def latent_geometry_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    *,
    max_nodes: int = 256,
) -> torch.Tensor:
    if z_pred.shape != z_target.shape or z_pred.ndim != 2:
        raise ValueError("Latent geometry inputs must have identical [N, D] shapes.")
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least two.")
    num_nodes = int(z_pred.shape[0])
    if num_nodes > max_nodes:
        indices = torch.linspace(
            0, num_nodes - 1, steps=max_nodes, device=z_pred.device
        ).round().long()
        z_pred = z_pred.index_select(0, indices)
        z_target = z_target.index_select(0, indices)
    predicted = F.normalize(z_pred, p=2, dim=-1, eps=1e-6)
    target = F.normalize(z_target.detach(), p=2, dim=-1, eps=1e-6)


    predicted_similarity = predicted @ predicted.transpose(0, 1)
    target_similarity = target @ target.transpose(0, 1)
    off_diagonal = ~torch.eye(
        predicted_similarity.shape[0], dtype=torch.bool, device=z_pred.device
    )
    return F.mse_loss(predicted_similarity[off_diagonal], target_similarity[off_diagonal])


def property_change_masks(
    y_true: np.ndarray,
    y_current: np.ndarray,
    *,
    large_change_threshold: float,
    epsilon: float = 1e-8,
    comparable_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if y_true.shape != y_current.shape or y_true.ndim != 2:
        raise ValueError("Property targets/current states must be equal rank-2 arrays.")
    delta = y_true - y_current
    mean_abs = np.mean(np.abs(delta), axis=1)
    max_abs = np.max(np.abs(delta), axis=1)
    changed = max_abs > float(epsilon)
    large = mean_abs >= float(large_change_threshold)
    masks: dict[str, np.ndarray] = {"changed": changed, "large": large}
    if comparable_mask is not None:
        comparable = np.asarray(comparable_mask, dtype=bool)
        if comparable.shape != changed.shape:
            raise ValueError("Comparable mask must have one value per property row.")
        masks["comparable"] = comparable
        masks["meaningful_changed"] = comparable & changed
    else:
        masks["meaningful_changed"] = changed
    summary: dict[str, Any] = {
        "row_delta_definition": "mean/max abs(Y_(t+1)-Y_t) across official target coordinates",
        "change_epsilon": float(epsilon),
        "large_change_threshold_train_q75": float(large_change_threshold),
        "row_mean_abs_delta_quantiles": {
            str(q): float(np.quantile(mean_abs, q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "unchanged_rows": int((~changed).sum()),
        "unchanged_ratio": float((~changed).mean()),
        "changed_rows": int(changed.sum()),
        "changed_ratio": float(changed.mean()),
        "large_change_rows": int(large.sum()),
        "large_change_ratio": float(large.mean()),
        "meaningful_changed_rows": int(masks["meaningful_changed"].sum()),
        "meaningful_changed_ratio": float(masks["meaningful_changed"].mean()),
    }
    if comparable_mask is not None:
        summary["current_y_available_rows"] = int(masks["comparable"].sum())
        summary["current_y_available_ratio"] = float(masks["comparable"].mean())
    return masks, summary


def delta_metrics_for_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_current: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return None
    true_delta = y_true[selected] - y_current[selected]
    predicted_delta = y_pred[selected] - y_current[selected]
    zero_delta = np.zeros_like(true_delta)
    return {
        "rows": int(selected.sum()),
        "prediction": regression_metrics(true_delta, predicted_delta),
        "zero_delta_copy_last": regression_metrics(true_delta, zero_delta),
    }


def streaming_delta_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_current: np.ndarray,
    *,
    large_change_threshold: float,
    epsilon: float = 1e-8,
    comparable_mask: np.ndarray | None = None,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    if y_true.shape != y_pred.shape or y_true.shape != y_current.shape or y_true.ndim != 2:
        raise ValueError("All property arrays must have equal rank-2 shapes.")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    row_count = y_true.shape[0]
    comparable = (
        np.ones(row_count, dtype=bool)
        if comparable_mask is None
        else np.asarray(comparable_mask, dtype=bool)
    )
    if comparable.shape != (row_count,):
        raise ValueError("Comparable mask must have one value per property row.")





    names = ("all", "changed", "large", "meaningful_large")
    totals = {
        name: {
            "rows": 0,
            "elements": 0,
            "prediction_abs": 0.0,
            "prediction_sq": 0.0,
            "zero_abs": 0.0,
            "zero_sq": 0.0,
            "predicted_delta_abs": 0.0,
            "true_delta_abs": 0.0,
        }
        for name in names
    }
    mean_magnitude_parts: list[np.ndarray] = []
    changed_rows = 0
    large_rows = 0
    meaningful_large_rows = 0
    unchanged_rows = 0
    meaningful_changed_rows = 0
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        true_delta = y_true[start:stop] - y_current[start:stop]
        predicted_delta = y_pred[start:stop] - y_current[start:stop]
        error = predicted_delta - true_delta
        abs_true = np.abs(true_delta)
        magnitude = abs_true.mean(axis=1)
        mean_magnitude_parts.append(magnitude.astype(np.float32, copy=False))
        changed = abs_true.max(axis=1) > float(epsilon)
        large = magnitude >= float(large_change_threshold)
        meaningful_changed = changed & comparable[start:stop]
        meaningful_large = large & comparable[start:stop]
        changed_rows += int(changed.sum())
        large_rows += int(large.sum())
        meaningful_large_rows += int(meaningful_large.sum())
        unchanged_rows += int((~changed).sum())
        meaningful_changed_rows += int(meaningful_changed.sum())
        for name, mask in (
            ("all", np.ones(stop - start, dtype=bool)),
            ("changed", meaningful_changed),
            ("large", large),
            ("meaningful_large", meaningful_large),
        ):
            if not mask.any():
                continue
            selected_error = error[mask]
            selected_true = true_delta[mask]
            selected_predicted_delta = predicted_delta[mask]
            total = totals[name]
            total["rows"] += int(mask.sum())
            total["elements"] += int(selected_error.size)
            total["prediction_abs"] += float(np.abs(selected_error).sum(dtype=np.float64))
            total["prediction_sq"] += float(np.square(selected_error).sum(dtype=np.float64))
            total["zero_abs"] += float(np.abs(selected_true).sum(dtype=np.float64))
            total["zero_sq"] += float(np.square(selected_true).sum(dtype=np.float64))
            total["predicted_delta_abs"] += float(
                np.abs(selected_predicted_delta).sum(dtype=np.float64)
            )
            total["true_delta_abs"] += float(np.abs(selected_true).sum(dtype=np.float64))

    def metric_payload(total: dict[str, Any], prefix: str) -> dict[str, Any] | None:
        if not total["elements"]:
            return None
        elements = float(total["elements"])
        mean_abs_true_delta = total["true_delta_abs"] / elements
        mean_abs_predicted_delta = total["predicted_delta_abs"] / elements
        return {
            "rows": int(total["rows"]),
            "prediction": {
                "mae": total["prediction_abs"] / elements,
                "rmse": float(np.sqrt(total["prediction_sq"] / elements)),
                "mse": total["prediction_sq"] / elements,
            },
            "zero_delta_copy_last": {
                "mae": total["zero_abs"] / elements,
                "rmse": float(np.sqrt(total["zero_sq"] / elements)),
                "mse": total["zero_sq"] / elements,
            },
            "delta_magnitude": {
                "mean_abs_true_delta": mean_abs_true_delta,
                "mean_abs_predicted_delta": mean_abs_predicted_delta,
                "predicted_to_true_mean_abs_ratio": (
                    mean_abs_predicted_delta / mean_abs_true_delta
                    if mean_abs_true_delta > 1e-12
                    else None
                ),
            },
            "predictor_name": prefix,
        }

    magnitudes = np.concatenate(mean_magnitude_parts) if mean_magnitude_parts else np.empty(0)
    return {
        "all": metric_payload(totals["all"], "prediction"),
        "changed": metric_payload(totals["changed"], "prediction"),
        "large": metric_payload(totals["large"], "prediction"),
        "meaningful_large": metric_payload(totals["meaningful_large"], "prediction"),
        "distribution": {
            "row_delta_definition": "mean/max abs(Y_(t+1)-Y_t) across official target coordinates",
            "change_epsilon": float(epsilon),
            "large_change_threshold_train_q75": float(large_change_threshold),
            "row_mean_abs_delta_quantiles": {
                str(q): float(np.quantile(magnitudes, q))
                for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
            },
            "unchanged_rows": int(unchanged_rows),
            "unchanged_ratio": float(unchanged_rows / max(row_count, 1)),
            "changed_rows": int(changed_rows),
            "changed_ratio": float(changed_rows / max(row_count, 1)),
            "large_change_rows": int(large_rows),
            "large_change_ratio": float(large_rows / max(row_count, 1)),
            "meaningful_large_change_rows": int(meaningful_large_rows),
            "meaningful_large_change_ratio": float(meaningful_large_rows / max(row_count, 1)),
            "meaningful_changed_rows": int(meaningful_changed_rows),
            "meaningful_changed_ratio": float(meaningful_changed_rows / max(row_count, 1)),
            "current_y_available_rows": int(comparable.sum()),
            "current_y_available_ratio": float(comparable.mean()),
            "chunk_rows": int(chunk_rows),
        },
    }


__all__ = [
    "PROPERTY_MODES",
    "decode_property_prediction",
    "delta_metrics_for_mask",
    "fit_residual_normalization",
    "latent_prediction_metrics",
    "property_change_masks",
    "property_prediction_loss",
    "regression_metrics",
    "residual_normalization_summary",
    "residual_normalization_to_device",
    "streaming_delta_diagnostics",
    "validate_property_mode",
]
