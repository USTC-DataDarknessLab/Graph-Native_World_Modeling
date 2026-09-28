
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DESCRIPTOR_NAMES = (
    "node_count",
    "edge_count",
    "density",
    "connected_components",
    "clustering_coefficient",
    "triangle_count",
)


def _adjacency(edge_index: torch.Tensor) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    if edge_index.numel():
        for source, destination in edge_index.detach().cpu().long().t().tolist():
            if int(source) == int(destination):
                continue
            adjacency.setdefault(int(source), set()).add(int(destination))
            adjacency.setdefault(int(destination), set()).add(int(source))
    return adjacency


TARGET_CACHE_FORMAT = "worldgraph_t3_targets_v3_causal_centres"
LEGACY_TARGET_CACHE_FORMAT = "worldgraph_t3_targets_v2_dynamic_ego"


def _ego_node_set(adjacency: dict[int, set[int]], centre: int) -> set[int]:
    if centre not in adjacency:
        return set()
    return {centre, *adjacency[centre]}


def _descriptor(adjacency: dict[int, set[int]], node_set: set[int]) -> torch.Tensor:



    active = sorted(node_set)
    active_set = set(active)
    local_neighbours = {
        node: adjacency.get(node, set()).intersection(active_set) for node in active
    }
    nodes = len(active)
    edges = sum(len(neighbours) for neighbours in local_neighbours.values()) // 2
    density = 2.0 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0
    components = 0
    unseen = set(active)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbour in local_neighbours[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    clustering_values: list[float] = []
    triangles_at_node = 0
    for node, neighbours in local_neighbours.items():
        degree = len(neighbours)
        if degree < 2:
            clustering_values.append(0.0)
        else:
            linked = sum(len(adjacency.get(left, set()).intersection(neighbours)) for left in neighbours) // 2
            clustering_values.append(2.0 * linked / (degree * (degree - 1)))

        triangles_at_node += sum(
            len(adjacency.get(left, set()).intersection(neighbours)) for left in neighbours
        ) // 2
    clustering = sum(clustering_values) / len(clustering_values) if clustering_values else 0.0
    triangles = triangles_at_node / 3.0
    return torch.tensor(
        [nodes, edges, density, components, clustering, triangles],
        dtype=torch.float32,
    )


def local_descriptors(
    edge_index: torch.Tensor, centres: torch.Tensor
) -> torch.Tensor:

    adjacency = _adjacency(edge_index)
    centre_ids = centres.detach().cpu().long().tolist()
    if not centre_ids:
        return torch.empty((0, len(DESCRIPTOR_NAMES)), dtype=torch.float32)
    return torch.stack(
        [
            _descriptor(adjacency, _ego_node_set(adjacency, int(centre)))
            for centre in centre_ids
        ]
    )


def compute_targets(transition: dict[str, Any]) -> dict[str, torch.Tensor]:
    source = _adjacency(transition["edge_index_t"])
    target = _adjacency(transition["edge_index_next"])
    published_centres = transition.get("t3_centres")
    if published_centres is None:
        centres = sorted(source)
    else:



        centres = sorted(
            int(value)
            for value in published_centres.detach().cpu().long().tolist()
            if int(value) in source
        )
    current: list[torch.Tensor] = []
    following: list[torch.Tensor] = []
    for centre in centres:
        current.append(_descriptor(source, _ego_node_set(source, centre)))
        following.append(_descriptor(target, _ego_node_set(target, centre)))
    if not centres:
        empty = torch.empty((0, len(DESCRIPTOR_NAMES)), dtype=torch.float32)
        return {"centres": torch.empty((0,), dtype=torch.long), "current": empty, "next": empty, "delta": empty}
    current_tensor = torch.stack(current)
    next_tensor = torch.stack(following)
    return {
        "centres": torch.tensor(centres, dtype=torch.long),
        "current": current_tensor,
        "next": next_tensor,
        "delta": next_tensor - current_tensor,
    }


@dataclass
class TargetCache:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[int, dict[str, torch.Tensor]] = {}
        if self.path.exists():
            saved = torch.load(self.path, map_location="cpu", weights_only=False)
            if isinstance(saved, dict) and saved.get("format") in {
                TARGET_CACHE_FORMAT,
                LEGACY_TARGET_CACHE_FORMAT,
            }:
                self.records = dict(saved.get("records", {}))

    def get(self, transition: dict[str, Any]) -> dict[str, torch.Tensor]:
        key = int(transition["transition_id"])
        if key not in self.records:
            self.records[key] = compute_targets(transition)
        return self.records[key]

    def save(self) -> None:




        temporary = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        try:
            torch.save(
                {"format": TARGET_CACHE_FORMAT, "descriptors": DESCRIPTOR_NAMES, "records": self.records},
                temporary,
            )
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


def fit_statistics(sequence: Sequence[dict[str, Any]], indices: Sequence[int], cache: TargetCache) -> dict[str, torch.Tensor]:
    values: list[torch.Tensor] = []
    current_values: list[torch.Tensor] = []
    for position, index in enumerate(indices, start=1):
        record = cache.get(sequence[index])
        values.append(record["delta"])
        current_values.append(record["current"])
        if position == 1 or position % 10 == 0 or position == len(indices):
            print(f"[T3-target-cache] train={position}/{len(indices)}", flush=True)
    values = [value for value in values if value.numel()]
    if not values:
        raise RuntimeError("No non-empty T3 training targets were found.")
    stacked = torch.cat(values, dim=0)
    current_stacked = torch.cat(
        [value for value in current_values if value.numel()], dim=0
    )
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False).clamp_min(1e-4)
    return {
        "mean": mean,
        "std": std,
        "tolerance": (0.10 * std).clamp_min(1e-4),
        "current_mean": current_stacked.mean(dim=0),
        "current_std": current_stacked.std(dim=0, unbiased=False).clamp_min(1e-4),
    }


def classes(delta: torch.Tensor, tolerance: torch.Tensor) -> torch.Tensor:
    tolerance = tolerance.to(device=delta.device, dtype=delta.dtype)
    return torch.where(
        delta < -tolerance,
        torch.zeros_like(delta, dtype=torch.long),
        torch.where(delta > tolerance, torch.full_like(delta, 2, dtype=torch.long), torch.ones_like(delta, dtype=torch.long)),
    )


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> dict[str, float]:
    error = prediction - target
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together.")
    if mean is not None and std is not None:
        mean = mean.to(device=prediction.device, dtype=prediction.dtype)
        std = std.to(device=prediction.device, dtype=prediction.dtype).clamp_min(1e-4)
        error = (prediction - mean) / std - (target - mean) / std
    return {
        "mae": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
    }


def change_regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    std: torch.Tensor,
) -> dict[str, float]:
    if prediction.shape != target.shape or target_classes.shape != target.shape:
        raise ValueError("prediction, target and target_classes must have equal shapes.")
    scale = std.to(device=prediction.device, dtype=prediction.dtype).clamp_min(1e-4)
    normalized_error = (prediction - target) / scale
    maes: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []
    for descriptor in range(target.shape[-1]):
        changed = target_classes[..., descriptor].ne(1)
        if not bool(changed.any()):
            continue
        error = normalized_error[..., descriptor][changed]
        maes.append(error.abs().mean())
        rmses.append(error.square().mean().sqrt())
    if not maes:
        return {"change_mae": 0.0, "change_rmse": 0.0}
    return {
        "change_mae": float(torch.stack(maes).mean().item()),
        "change_rmse": float(torch.stack(rmses).mean().item()),
    }


def macro_f1(target: torch.Tensor, prediction: torch.Tensor) -> float:
    scores: list[float] = []
    for label in (0, 1, 2):
        tp = ((target == label) & (prediction == label)).sum().item()
        fp = ((target != label) & (prediction == label)).sum().item()
        fn = ((target == label) & (prediction != label)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def change_macro_f1(target: torch.Tensor, prediction: torch.Tensor) -> float:
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes.")
    if target.ndim == 1:
        target = target.unsqueeze(-1)
        prediction = prediction.unsqueeze(-1)
    scores: list[float] = []
    for descriptor in range(target.shape[-1]):
        target_column = target[..., descriptor].reshape(-1)
        prediction_column = prediction[..., descriptor].reshape(-1)
        for label in (0, 2):
            target_positive = target_column == label
            predicted_positive = prediction_column == label
            if not bool(target_positive.any() or predicted_positive.any()):
                continue
            tp = (target_positive & predicted_positive).sum().item()
            fp = ((~target_positive) & predicted_positive).sum().item()
            fn = (target_positive & (~predicted_positive)).sum().item()
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            scores.append(
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
    return sum(scores) / len(scores) if scores else 0.0


def temporal_change_macro_f1(
    targets: Sequence[torch.Tensor], predictions: Sequence[torch.Tensor]
) -> float:
    if not targets or len(targets) != len(predictions):
        return 0.0
    values = [change_macro_f1(target, prediction) for target, prediction in zip(targets, predictions)]
    return float(sum(values) / len(values)) if values else 0.0
