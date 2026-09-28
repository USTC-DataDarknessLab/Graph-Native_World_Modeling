
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

FEATURE_NAMES = [
    "log_current_directed_degree",
    "log_current_interaction_count",
    "log_current_unique_neighbors",
    "log_time_since_last_interaction",
    "log_cumulative_interaction_count",
    "log_cumulative_unique_neighbors",
    "has_been_seen",
]




NODE_CHANGE_FEATURE_INDICES = [0, 1, 2, 4, 5, 6]
NODE_CHANGE_FEATURE_NAMES = [FEATURE_NAMES[index] for index in NODE_CHANGE_FEATURE_INDICES]


def _edge_index_from_codes(codes: np.ndarray, num_nodes: int) -> torch.Tensor:
    if len(codes) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(
        np.stack([codes // num_nodes, codes % num_nodes], axis=0).astype(np.int64)
    )


def _snapshot_ranges(
    stream: Any,
    window_size: int,
    window_stride: int,
) -> list[tuple[str, int, int]]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if window_stride < 1 or window_stride > window_size:
        raise ValueError("window_stride must lie in [1, window_size]")
    boundaries = [
        ("train", 0, stream.train_end),
        ("val", stream.train_end, stream.val_end),
        ("test", stream.val_end, stream.num_events),
    ]
    ranges: list[tuple[str, int, int]] = []
    for split, start, end in boundaries:
        if end <= start:
            continue
        split_size = end - start
        if split_size <= window_size:
            ranges.append((split, start, end))
            continue
        starts = list(range(start, end - window_size + 1, window_stride))
        final_start = end - window_size
        if starts[-1] != final_start:
            starts.append(final_start)
        ranges.extend(
            (split, chunk_start, chunk_start + window_size)
            for chunk_start in starts
        )
    if len(ranges) < 2:
        raise ValueError("Need at least two snapshots to build transitions.")
    return ranges


def _build_snapshot(
    stream: Any,
    split: str,
    start: int,
    end: int,
    cumulative_events: np.ndarray,
    cumulative_neighbors: list[set[int]],
    last_seen: np.ndarray,
    *,
    history_start: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    src = stream.src[start:end]
    dst = stream.dst[start:end]
    timestamps = stream.timestamp[start:end]
    num_nodes = stream.num_nodes
    encoded = src.astype(np.int64) * num_nodes + dst.astype(np.int64)
    edge_codes, edge_counts = np.unique(encoded, return_counts=True)
    edge_src = (edge_codes // num_nodes).astype(np.int64)
    edge_dst = (edge_codes % num_nodes).astype(np.int64)

    current_degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(current_degree, edge_src, 1.0)
    np.add.at(current_degree, edge_dst, 1.0)

    current_interactions = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(current_interactions, src, 1.0)
    np.add.at(current_interactions, dst, 1.0)

    current_neighbors = [set() for _ in range(num_nodes)]

    for left, right in zip(edge_src.tolist(), edge_dst.tolist()):
        if left != right:
            current_neighbors[left].add(right)
            current_neighbors[right].add(left)



    history_start = start if history_start is None else int(history_start)
    history_src = stream.src[history_start:end]
    history_dst = stream.dst[history_start:end]
    history_timestamps = stream.timestamp[history_start:end]
    history_codes = (
        history_src.astype(np.int64) * num_nodes + history_dst.astype(np.int64)
    )
    if len(history_codes):
        history_unique = np.unique(history_codes)
        for left, right in zip(
            (history_unique // num_nodes).tolist(),
            (history_unique % num_nodes).tolist(),
        ):
            if left != right:
                cumulative_neighbors[left].add(right)
                cumulative_neighbors[right].add(left)
        np.add.at(cumulative_events, history_src, 1.0)
        np.add.at(cumulative_events, history_dst, 1.0)
        np.maximum.at(last_seen, history_src, history_timestamps)
        np.maximum.at(last_seen, history_dst, history_timestamps)

    end_time = float(timestamps[-1])
    seen = np.isfinite(last_seen)
    recency = np.zeros(num_nodes, dtype=np.float64)
    recency[seen] = np.maximum(0.0, end_time - last_seen[seen])
    current_unique_neighbors = np.fromiter(
        (len(neighbors) for neighbors in current_neighbors), dtype=np.float64, count=num_nodes
    )
    cumulative_unique_neighbors = np.fromiter(
        (len(neighbors) for neighbors in cumulative_neighbors),
        dtype=np.float64,
        count=num_nodes,
    )


    x_raw = np.column_stack(
        [
            np.log1p(current_degree),
            np.log1p(current_interactions),
            np.log1p(current_unique_neighbors),
            np.log1p(recency),
            np.log1p(cumulative_events),
            np.log1p(cumulative_unique_neighbors),
            seen.astype(np.float64),
        ]
    ).astype(np.float32)
    snapshot: dict[str, Any] = {
        "split": split,
        "event_start": int(start),
        "event_end": int(end),
        "num_events": int(end - start),
        "time_start": float(timestamps[0]),
        "time_end": end_time,
        "edge_index": _edge_index_from_codes(edge_codes, num_nodes),
        "edge_weight": torch.from_numpy(edge_counts.astype(np.float32)),
        "x_raw": torch.from_numpy(x_raw),

        "_edge_codes": edge_codes,
    }
    return snapshot, edge_codes


def _normalize_and_add_transition_targets(
    snapshots: list[dict[str, Any]], epsilon: float
) -> tuple[list[float], list[float]]:
    train_x = torch.cat([snapshot["x_raw"] for snapshot in snapshots if snapshot["split"] == "train"])
    if train_x.numel() == 0:
        raise ValueError("No train snapshots are available for feature normalization.")
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    for snapshot in snapshots:
        snapshot["x"] = (snapshot["x_raw"] - feature_mean) / feature_std

    num_nodes = snapshots[0]["x"].shape[0]
    for index, snapshot in enumerate(snapshots):
        if index + 1 == len(snapshots):
            snapshot["edge_added"] = torch.empty((2, 0), dtype=torch.long)
            snapshot["edge_removed"] = torch.empty((2, 0), dtype=torch.long)
            snapshot["node_delta"] = torch.zeros_like(snapshot["x"])
            snapshot["node_delta_raw"] = torch.zeros_like(snapshot["x_raw"])
            snapshot["node_changed"] = torch.zeros(num_nodes, dtype=torch.float32)
            snapshot["node_changed_with_recency"] = torch.zeros(num_nodes, dtype=torch.float32)
            continue
        next_snapshot = snapshots[index + 1]
        current_codes = snapshot["_edge_codes"]
        next_codes = next_snapshot["_edge_codes"]
        added = np.setdiff1d(next_codes, current_codes, assume_unique=True)
        removed = np.setdiff1d(current_codes, next_codes, assume_unique=True)
        snapshot["edge_added"] = _edge_index_from_codes(added, num_nodes)
        snapshot["edge_removed"] = _edge_index_from_codes(removed, num_nodes)
        snapshot["node_delta"] = next_snapshot["x"] - snapshot["x"]
        snapshot["node_delta_raw"] = next_snapshot["x_raw"] - snapshot["x_raw"]


        snapshot["node_changed"] = (
            snapshot["node_delta_raw"][:, NODE_CHANGE_FEATURE_INDICES].abs().amax(dim=1)
            > epsilon
        ).to(torch.float32)

        snapshot["node_changed_with_recency"] = (
            snapshot["node_delta_raw"].abs().amax(dim=1) > epsilon
        ).to(torch.float32)

    for snapshot in snapshots:
        del snapshot["_edge_codes"]
    return feature_mean.tolist(), feature_std.tolist()


def _node_change_rates(snapshots: list[dict[str, Any]], key: str) -> dict[str, float]:
    totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for index, snapshot in enumerate(snapshots[:-1]):
        target_split = snapshots[index + 1]["split"]
        totals[target_split] += float(snapshot[key].sum().item())
        counts[target_split] += int(snapshot[key].numel())
    return {split: totals[split] / counts[split] for split in ("train", "val", "test") if counts[split]}


def build_snapshots(
    stream: Any,
    *,
    window_mode: str = "events",
    window_size: int = 500,
    window_stride: int | None = None,
    node_change_epsilon: float = 1e-8,
) -> dict[str, Any]:
    if window_mode != "events":
        raise NotImplementedError(
            "Only event-count windows are implemented in phase 1; use --window_mode events."
        )
    effective_stride = window_size if window_stride is None else int(window_stride)
    ranges = _snapshot_ranges(stream, window_size, effective_stride)
    cumulative_events = np.zeros(stream.num_nodes, dtype=np.float64)
    cumulative_neighbors = [set() for _ in range(stream.num_nodes)]
    last_seen = np.full(stream.num_nodes, -np.inf, dtype=np.float64)
    snapshots: list[dict[str, Any]] = []
    history_cursor = 0
    for snapshot_id, (split, start, end) in enumerate(ranges):
        snapshot, _ = _build_snapshot(
            stream,
            split,
            start,
            end,
            cumulative_events,
            cumulative_neighbors,
            last_seen,
            history_start=history_cursor,
        )
        history_cursor = end
        snapshot["snapshot_id"] = snapshot_id
        snapshots.append(snapshot)

    feature_mean, feature_std = _normalize_and_add_transition_targets(
        snapshots, node_change_epsilon
    )
    transition_counts = Counter(snapshot["split"] for snapshot in snapshots[1:])
    metadata = {
        **stream.metadata,
        "num_nodes": stream.num_nodes,
        "num_events": stream.num_events,
        "event_train_end": stream.train_end,
        "event_val_end": stream.val_end,
        "window_mode": window_mode,
        "window_size": window_size,
        "window_stride": effective_stride,
        "overlapping_windows": bool(effective_stride < window_size),
        "node_feature_names": FEATURE_NAMES,
        "node_feature_dim": len(FEATURE_NAMES),
        "node_change_epsilon": node_change_epsilon,
        "node_change_definition": "any non-recency graph-derived state delta exceeds epsilon",
        "node_change_feature_names": NODE_CHANGE_FEATURE_NAMES,
        "node_change_excluded_feature_names": ["log_time_since_last_interaction"],
        "node_change_positive_rate_by_target_split": _node_change_rates(snapshots, "node_changed"),
        "node_change_positive_rate_with_recency_by_target_split": _node_change_rates(
            snapshots, "node_changed_with_recency"
        ),
        "feature_mean_train_only": feature_mean,
        "feature_std_train_only": feature_std,
        "num_snapshots": len(snapshots),
        "num_transitions": len(snapshots) - 1,
        "snapshot_counts": dict(Counter(snapshot["split"] for snapshot in snapshots)),
        "transition_counts_by_target_split": dict(transition_counts),
    }
    return {
        "snapshots": snapshots,
        "metadata": metadata,


        "official_tgb": {
            "src": torch.from_numpy(stream.tgb_src.copy()),
            "dst": torch.from_numpy(stream.tgb_dst.copy()),
            "timestamp": torch.from_numpy(stream.tgb_timestamp.copy()),
            "id_to_compact": torch.from_numpy(stream.tgb_id_to_compact.copy()),
        },
    }


def save_processed_snapshots(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return path


def load_processed_snapshots(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def transition_sanity(snapshot: dict[str, Any]) -> dict[str, float | int]:
    return {
        "snapshot": int(snapshot["snapshot_id"]),
        "next_snapshot": int(snapshot["snapshot_id"]) + 1,
        "edges_t": int(snapshot["edge_index"].shape[1]),
        "edges_next": int(snapshot["edge_index"].shape[1]
                          + snapshot["edge_added"].shape[1]
                          - snapshot["edge_removed"].shape[1]),
        "added_edges": int(snapshot["edge_added"].shape[1]),
        "removed_edges": int(snapshot["edge_removed"].shape[1]),
        "changed_nodes": int(snapshot["node_changed"].sum().item()),
        "mean_abs_node_delta": float(snapshot["node_delta"].abs().mean().item()),
    }
