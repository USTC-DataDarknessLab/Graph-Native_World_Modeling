
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

GENRE_FEATURE_NAMES = [
    "log_current_incident_unique_edges",
    "log_current_incident_interaction_count",
    "log_current_unique_neighbors",
    "log_current_incident_abs_genre_weight",
    "log_time_since_last_interaction",
    "log_cumulative_incident_interaction_count",
    "log_cumulative_unique_neighbors",
    "log_cumulative_incident_abs_genre_weight",
    "has_been_seen",
]


def _edge_index_from_codes(codes: np.ndarray, num_nodes: int) -> torch.Tensor:
    if codes.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(
        np.stack([codes // num_nodes, codes % num_nodes], axis=0).astype(np.int64)
    )


def _label_split(label_time: int, stream: Any) -> str:
    if label_time <= stream.train_end_time:
        return "train"
    if label_time <= stream.val_end_time:
        return "val"
    return "test"


def _snapshot_from_interval(
    stream: Any,
    *,
    snapshot_id: int,
    event_start: int,
    event_end: int,
    end_time: int,
    label_time: int | None,
    cumulative_counts: np.ndarray,
    cumulative_weight: np.ndarray,
    cumulative_neighbors: list[set[int]],
    last_seen: np.ndarray,
) -> dict[str, Any]:
    num_nodes = stream.num_nodes
    src = stream.src[event_start:event_end]
    dst = stream.dst[event_start:event_end]
    timestamp = stream.timestamp[event_start:event_end]
    raw_weight = stream.weight[event_start:event_end]
    incident_weight = np.abs(raw_weight)
    codes = src * num_nodes + dst
    edge_codes, edge_inverse = np.unique(codes, return_inverse=True)
    edge_event_count = np.bincount(edge_inverse, minlength=edge_codes.size)
    edge_weight = np.bincount(edge_inverse, weights=raw_weight, minlength=edge_codes.size)
    edge_src = (edge_codes // num_nodes).astype(np.int64)
    edge_dst = (edge_codes % num_nodes).astype(np.int64)

    current_degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(current_degree, edge_src, 1.0)
    np.add.at(current_degree, edge_dst, 1.0)
    current_count = np.zeros(num_nodes, dtype=np.float64)
    current_weight = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(current_count, src, 1.0)
    np.add.at(current_count, dst, 1.0)
    np.add.at(current_weight, src, incident_weight)
    np.add.at(current_weight, dst, incident_weight)

    current_neighbors = [set() for _ in range(num_nodes)]
    for source, destination in zip(edge_src.tolist(), edge_dst.tolist()):
        if source != destination:
            current_neighbors[source].add(destination)
            current_neighbors[destination].add(source)
            cumulative_neighbors[source].add(destination)
            cumulative_neighbors[destination].add(source)
    np.add.at(cumulative_counts, src, 1.0)
    np.add.at(cumulative_counts, dst, 1.0)
    np.add.at(cumulative_weight, src, incident_weight)
    np.add.at(cumulative_weight, dst, incident_weight)
    if timestamp.size:
        np.maximum.at(last_seen, src, timestamp)
        np.maximum.at(last_seen, dst, timestamp)

    seen = np.isfinite(last_seen)
    recency = np.zeros(num_nodes, dtype=np.float64)
    recency[seen] = np.maximum(0.0, float(end_time) - last_seen[seen])
    current_unique_neighbors = np.fromiter(
        (len(values) for values in current_neighbors), dtype=np.float64, count=num_nodes
    )
    cumulative_unique_neighbors = np.fromiter(
        (len(values) for values in cumulative_neighbors), dtype=np.float64, count=num_nodes
    )
    x_raw = np.column_stack(
        [
            np.log1p(current_degree),
            np.log1p(current_count),
            np.log1p(current_unique_neighbors),
            np.log1p(current_weight),
            np.log1p(recency),
            np.log1p(cumulative_counts),
            np.log1p(cumulative_unique_neighbors),
            np.log1p(cumulative_weight),
            seen.astype(np.float64),
        ]
    ).astype(np.float32)
    split = "post_test" if label_time is None else _label_split(label_time, stream)
    return {
        "snapshot_id": int(snapshot_id),
        "split": split,
        "label_time": label_time,
        "event_start": int(event_start),
        "event_end": int(event_end),
        "num_events": int(event_end - event_start),
        "time_start": int(stream.timestamp[event_start]) if event_end > event_start else int(end_time),
        "time_end": int(end_time),
        "latest_observed_event_time": None if not timestamp.size else int(timestamp[-1]),
        "edge_index": _edge_index_from_codes(edge_codes, num_nodes),


        "edge_weight": torch.from_numpy(edge_weight.astype(np.float32)),
        "edge_event_count": torch.from_numpy(edge_event_count.astype(np.int64)),
        "x_raw": torch.from_numpy(x_raw),
    }


def _train_feature_statistics(snapshots: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.cat(
        [snapshot["x_raw"] for snapshot in snapshots[:-1] if snapshot["split"] == "train"],
        dim=0,
    )
    if train_x.numel() == 0:
        raise ValueError("No train-time snapshots available for input normalization.")
    return train_x.mean(dim=0), train_x.std(dim=0, unbiased=False).clamp_min(1e-6)


def _property_overlap_statistics(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    total_target_rows = 0
    comparable_rows = 0
    for current, following in zip(snapshots[:-2], snapshots[1:-1]):
        source_ids = current["property_node_ids"].numpy()
        target_ids = following["property_node_ids"].numpy()
        total_target_rows += int(target_ids.size)
        comparable_rows += int(np.isin(target_ids, source_ids, assume_unique=True).sum())
    return {
        "target_rows_with_current_observed_y": comparable_rows,
        "target_rows_without_current_official_y": total_target_rows - comparable_rows,
        "current_y_coverage_of_next_label_rows": float(comparable_rows / max(total_target_rows, 1)),
    }


def build_tgbn_genre_snapshots(stream: Any) -> dict[str, Any]:
    label_times = stream.label_times
    num_labels = int(label_times.size)
    event_ends = np.searchsorted(stream.timestamp, label_times, side="left")
    boundaries = np.concatenate([event_ends, np.asarray([stream.num_events], dtype=np.int64)])
    starts = np.concatenate([np.asarray([0], dtype=np.int64), event_ends])
    end_times = np.concatenate(
        [label_times, np.asarray([int(stream.timestamp[-1]) + 1], dtype=np.int64)]
    )
    if not np.all(starts <= boundaries):
        raise ValueError("Official label-aligned snapshot ranges are not chronological.")

    cumulative_counts = np.zeros(stream.num_nodes, dtype=np.float64)
    cumulative_weight = np.zeros(stream.num_nodes, dtype=np.float64)
    cumulative_neighbors = [set() for _ in range(stream.num_nodes)]
    last_seen = np.full(stream.num_nodes, -np.inf, dtype=np.float64)
    snapshots: list[dict[str, Any]] = []
    for snapshot_id in range(num_labels + 1):
        label_time = int(label_times[snapshot_id]) if snapshot_id < num_labels else None
        snapshot = _snapshot_from_interval(
            stream,
            snapshot_id=snapshot_id,
            event_start=int(starts[snapshot_id]),
            event_end=int(boundaries[snapshot_id]),
            end_time=int(end_times[snapshot_id]),
            label_time=label_time,
            cumulative_counts=cumulative_counts,
            cumulative_weight=cumulative_weight,
            cumulative_neighbors=cumulative_neighbors,
            last_seen=last_seen,
        )
        if snapshot_id < num_labels:



            snapshot["property_node_ids"] = torch.from_numpy(
                stream.label_nodes[snapshot_id].astype(np.int64, copy=False)
            )
            snapshot["property_target"] = torch.from_numpy(
                stream.label_values[snapshot_id].astype(np.float32, copy=False)
            )
        else:
            snapshot["property_node_ids"] = torch.empty(0, dtype=torch.long)
            snapshot["property_target"] = torch.empty(
                (0, stream.property_dim), dtype=torch.float32
            )
        snapshots.append(snapshot)

    feature_mean, feature_std = _train_feature_statistics(snapshots)
    for snapshot in snapshots:
        snapshot["x"] = (snapshot["x_raw"] - feature_mean) / feature_std

    latent_train_allowed: list[bool] = []
    for index, snapshot in enumerate(snapshots[:-2]):
        following = snapshots[index + 1]
        next_latest = following["latest_observed_event_time"]
        allowed = bool(
            following["split"] == "train"
            and (next_latest is None or next_latest <= stream.train_end_time)
        )
        snapshot["latent_train_allowed"] = allowed
        latent_train_allowed.append(allowed)
    snapshots[-2]["latent_train_allowed"] = False
    snapshots[-1]["latent_train_allowed"] = False

    target_snapshots = snapshots[1:-1]
    split_counts = Counter(snapshot["split"] for snapshot in target_snapshots)
    property_rows = {
        split: int(
            sum(
                snapshot["property_target"].shape[0]
                for snapshot in target_snapshots
                if snapshot["split"] == split
            )
        )
        for split in ("train", "val", "test")
    }
    target_nonzero = sum(
        int(torch.count_nonzero(snapshot["property_target"]).item())
        for snapshot in target_snapshots
    )
    target_entries = sum(int(snapshot["property_target"].numel()) for snapshot in target_snapshots)
    metadata = {
        **stream.metadata,
        "window_mode": "official_label_time",
        "num_snapshots": len(snapshots),
        "num_transitions": len(snapshots) - 2,
        "transition_counts_by_target_split": dict(split_counts),
        "property_rows_by_target_split": property_rows,
        "node_feature_names": GENRE_FEATURE_NAMES,
        "node_feature_dim": len(GENRE_FEATURE_NAMES),
        "model_input_dim": len(GENRE_FEATURE_NAMES) + stream.property_dim,
        "property_state_in_graph": True,
        "property_state_storage": (
            "sparse official label batches; lazily zero-scattered to [N, 513] per transition"
        ),
        "input_feature_definition": (
            "Leakage-safe graph-derived interaction/topology state from events strictly "
            "before each official genre label timestamp."
        ),
        "observed_property_state_definition": (
            "Y_i is the official tgbn-genre label batch at L_i scattered into [N, 513]; "
            "genre-node rows and users without a supplied current row are zero-filled. "
            "GraphEncoder receives concat(X_i, Y_i)."
        ),
        "feature_mean_train_only": feature_mean.tolist(),
        "feature_std_train_only": feature_std.tolist(),
        "property_target_normalization": "none; original TGB target values are used directly",
        "target_boundary_rule": (
            "For target Y_(i+1), source G_i contains only events with timestamp < L_i and "
            "observed Y_i. G_(i+1) and Y_(i+1) enter only target-latent/property supervision."
        ),
        "latent_train_allowed_transitions": int(sum(latent_train_allowed)),
        "latent_train_withheld_cross_boundary": int(
            sum(snapshot["split"] == "train" for snapshot in target_snapshots)
            - sum(latent_train_allowed)
        ),
        "target_nonzero_fraction_over_transition_targets": float(
            target_nonzero / max(target_entries, 1)
        ),
        **_property_overlap_statistics(snapshots),
    }
    return {"snapshots": snapshots, "metadata": metadata}


def save_processed_tgbn_genre(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return path


def load_processed_tgbn_genre(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(
            Path(path), map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        return torch.load(Path(path), map_location="cpu")
    except RuntimeError:
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def genre_transition_sanity(bundle: dict[str, Any], transition_id: int) -> dict[str, Any]:
    current = bundle["snapshots"][transition_id]
    following = bundle["snapshots"][transition_id + 1]
    current_ids = current["property_node_ids"].numpy()
    target_ids = following["property_node_ids"].numpy()
    return {
        "transition_id": int(transition_id),
        "source_label_time": current["label_time"],
        "target_label_time": following["label_time"],
        "target_split": following["split"],
        "source_events": int(current["num_events"]),
        "target_events": int(following["num_events"]),
        "source_unique_directed_edges": int(current["edge_index"].shape[1]),
        "target_unique_directed_edges": int(following["edge_index"].shape[1]),
        "source_label_rows": int(current_ids.size),
        "target_label_rows": int(target_ids.size),
        "target_rows_with_current_observed_y": int(
            np.isin(target_ids, current_ids, assume_unique=True).sum()
        ),
        "source_latest_observed_event_time": current["latest_observed_event_time"],
        "target_latest_observed_event_time": following["latest_observed_event_time"],
        "x_shape": list(current["x"].shape),
        "official_y_target_shape": list(following["property_target"].shape),
    }


__all__ = [
    "GENRE_FEATURE_NAMES",
    "build_tgbn_genre_snapshots",
    "genre_transition_sanity",
    "load_processed_tgbn_genre",
    "save_processed_tgbn_genre",
]
