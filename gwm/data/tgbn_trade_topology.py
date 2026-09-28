
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch

from .tgbn_trade_builder import load_processed_tgbn_trade


def _edge_codes(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(
        edge_index.detach().cpu().long()[0] * int(num_nodes)
        + edge_index.detach().cpu().long()[1],
        sorted=True,
    )


def _edge_index_from_codes(codes: torch.Tensor, num_nodes: int) -> torch.Tensor:
    codes = torch.unique(codes.detach().cpu().long(), sorted=True)
    if codes.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack([codes // int(num_nodes), codes % int(num_nodes)], dim=0)


def _set_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() == 0 or right.numel() == 0:
        return left.clone()
    positions = torch.searchsorted(right, left)
    valid = positions < right.numel()
    matched = torch.zeros(left.numel(), dtype=torch.bool)
    if valid.any():
        matched[valid] = right.index_select(0, positions[valid]).eq(left[valid])
    return left[~matched]


def build_tgbn_trade_topology_bundle(
    source_bundle: dict[str, Any],
    *,
    edge_weight_quantile: float | None = None,
) -> dict[str, Any]:
    source_metadata = source_bundle["metadata"]
    if source_metadata.get("dataset_name") != "tgbn-trade":
        raise ValueError("Expected the official tgbn-trade processed bundle.")
    num_nodes = int(source_metadata["num_nodes"])
    source_snapshots: list[dict[str, Any]] = source_bundle["snapshots"]
    if edge_weight_quantile is not None and not 0.0 <= float(edge_weight_quantile) < 1.0:
        raise ValueError("edge_weight_quantile must lie in [0, 1).")
    if len(source_snapshots) < 3:
        raise ValueError("Need at least three annual tgbn-trade snapshots.")

    edge_weight_threshold: float | None = None
    if edge_weight_quantile is not None:
        train_weights = [
            snapshot["edge_weight"].detach().cpu().float().reshape(-1)
            for index, snapshot in enumerate(source_snapshots[:-1])
            if str(source_snapshots[index + 1].get("split")) == "train"
        ]
        if not train_weights:
            raise ValueError(
                "Cannot fit a significant-edge threshold without train snapshots."
            )
        all_training_weights = torch.cat(train_weights)
        if all_training_weights.numel() == 0 or not torch.isfinite(all_training_weights).all():
            raise ValueError("Training edge weights must be finite and non-empty.")
        edge_weight_threshold = float(
            torch.quantile(all_training_weights, float(edge_weight_quantile)).item()
        )

    snapshots: list[dict[str, Any]] = []
    for index, source in enumerate(source_snapshots):
        required = ("x", "x_raw", "edge_index", "edge_weight", "property_state")
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"Trade source snapshot {index} misses {missing}.")
        edge_index = source["edge_index"]
        edge_weight = source["edge_weight"]
        edge_event_count = source.get("edge_event_count")
        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Snapshot {index} has misaligned edge_index/edge_weight fields."
            )
        if edge_weight_threshold is not None:
            keep = edge_weight.detach().cpu().float() >= edge_weight_threshold
            edge_index = edge_index[:, keep]
            edge_weight = edge_weight[keep]
            if edge_event_count is not None:
                edge_event_count = edge_event_count[keep]
        snapshots.append(
            {
                "snapshot_id": int(source["snapshot_id"]),
                "split": str(source["split"]),
                "label_time": source.get("label_time"),
                "time_start": int(source["time_start"]),
                "time_end": int(source["time_end"]),
                "num_events": int(source["num_events"]),


                "x": source["x"],
                "x_raw": source["x_raw"],
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "edge_event_count": edge_event_count,
                "property_state": source["property_state"],
                "property_node_ids": source.get("property_node_ids"),
                "property_target": source.get("property_target"),
            }
        )

    transition_splits: Counter[str] = Counter()
    for index, snapshot in enumerate(snapshots):
        if index + 1 == len(snapshots):
            snapshot["edge_added"] = torch.empty((2, 0), dtype=torch.long)
            snapshot["edge_removed"] = torch.empty((2, 0), dtype=torch.long)
            snapshot["target_split"] = None
            continue
        current = _edge_codes(snapshot["edge_index"], num_nodes)
        following = _edge_codes(snapshots[index + 1]["edge_index"], num_nodes)
        snapshot["edge_added"] = _edge_index_from_codes(
            _set_difference(following, current), num_nodes
        )
        snapshot["edge_removed"] = _edge_index_from_codes(
            _set_difference(current, following), num_nodes
        )
        target_split = str(snapshots[index + 1]["split"])


        snapshot["target_split"] = (
            str(snapshot["split"]) if target_split == "post_test" else target_split
        )
        transition_splits[str(snapshot["target_split"])] += 1

    comparable_transition_count = len(snapshots) - 2
    comparable_splits = Counter(
        str(snapshot["target_split"]) for snapshot in snapshots[:comparable_transition_count]
    )
    significant_track = edge_weight_threshold is not None
    variant_name = (
        f"tgbn-trade-topology-significant-q{float(edge_weight_quantile):g}"
        if significant_track
        else "tgbn-trade-topology"
    )
    metadata = {
        **source_metadata,
        "dataset_name": variant_name,
        "source_dataset_name": "tgbn-trade",
        "topology_source_processed_definition": (
            "Official-TGB annual-label-boundary trade states; binary edges are "
            + (
                "restricted to a training-fitted aggregate-weight quantile."
                if significant_track
                else "positive aggregate relations."
            )
        ),
        "topology_time_granularity": "annual official tgbn-trade boundaries",
        "topology_snapshot_years": [int(snapshot["time_start"]) for snapshot in snapshots],
        "num_topology_snapshots": len(snapshots),
        "num_topology_transitions_including_terminal": len(snapshots) - 1,
        "num_topology_transitions_common": comparable_transition_count,
        "topology_transition_counts_by_target_split_including_terminal": dict(transition_splits),
        "topology_transition_counts_by_target_split_common": dict(comparable_splits),
        "binary_topology_definition": (
            "A directed edge (u,v) is present iff the official annual observed interval has "
            + (
                f"aggregate raw tgbn-trade mass >= {edge_weight_threshold:.12g}; "
                "the threshold is the training-source "
                f"{float(edge_weight_quantile):g} quantile."
                if significant_track
                else "positive aggregate raw tgbn-trade mass for that country pair."
            )
        ),
        "topology_variant": (
            "significant_relation_train_weight_quantile"
            if significant_track
            else "any_positive_relation"
        ),
        "edge_weight_quantile_train_only": (
            None if edge_weight_quantile is None else float(edge_weight_quantile)
        ),
        "edge_weight_threshold_train_only": edge_weight_threshold,
        "edge_weight_available": True,
        "edge_weight_definition": source_metadata.get("edge_weight_definition"),
        "topology_default_observed_state": "g_t=(V,E_t,W_t,X_t); no Y_t/property input",
        "candidate_space_definition": "V x V directed country pairs, including observed self-trade pairs",
    }
    return {"snapshots": snapshots, "metadata": metadata}


def save_processed_tgbn_trade_topology(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return path


def load_processed_tgbn_trade_topology(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


class TGBNTradeTopologyDataset:

    def __init__(
        self,
        source: str | Path | dict[str, Any],
        *,
        edge_input: str = "weighted",
        include_terminal_transition: bool = False,
        split: str | None = None,
    ) -> None:
        self.bundle = (
            load_processed_tgbn_trade_topology(source)
            if not isinstance(source, dict)
            else source
        )
        self.snapshots: list[dict[str, Any]] = self.bundle["snapshots"]
        if len(self.snapshots) < 3:
            raise ValueError("Processed trade topology data has fewer than three snapshots.")
        if edge_input not in {"binary", "weighted"}:
            raise ValueError("edge_input must be binary or weighted")
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be None, train, val, or test")
        self.edge_input = edge_input
        self.include_terminal_transition = bool(include_terminal_transition)
        self.split = split
        self.transition_count = len(self.snapshots) - (1 if include_terminal_transition else 2)
        if self.transition_count < 1:
            raise ValueError("No valid topology transitions are available.")
        x_dim = int(self.snapshots[0]["x"].shape[1])
        self.metadata = dict(self.bundle["metadata"])
        self.metadata.update(
            {
                "num_nodes": int(self.snapshots[0]["x"].shape[0]),
                "node_feature_dim": x_dim,
                "model_input_dim": x_dim,
                "topology_input_mode": "topology_only",
                "topology_edge_input": self.edge_input,
                "active_transition_count": self.transition_count,
                "terminal_transition_included": self.include_terminal_transition,
            }
        )
        self.indices = [
            index
            for index in range(self.transition_count)
            if split is None or self._target_split(index) == split
        ]

        active_counts = Counter(
            self._target_split(index) for index in range(self.transition_count)
        )
        self.metadata.update(
            {
                "topology_active_transition_counts_by_target_split": dict(
                    active_counts
                ),
                "topology_active_transition_count": int(self.transition_count),
            }
        )

    def _target_split(self, transition_id: int) -> str:
        value = self.snapshots[transition_id].get("target_split")
        if value not in {"train", "val", "test"}:
            raise ValueError(f"Transition {transition_id} has no valid target split: {value!r}")
        return str(value)

    def __len__(self) -> int:
        return len(self.indices)

    def _features(self, snapshot: dict[str, Any]) -> torch.Tensor:
        return snapshot["x"]

    def transition(self, transition_id: int) -> dict[str, Any]:
        if transition_id < 0 or transition_id >= self.transition_count:
            raise IndexError(f"transition_id={transition_id} is out of range")
        current = self.snapshots[transition_id]
        following = self.snapshots[transition_id + 1]
        transition: dict[str, Any] = {
            "transition_id": transition_id,
            "split": self._target_split(transition_id),
            "snapshot_t": int(current["snapshot_id"]),
            "snapshot_next": int(following["snapshot_id"]),
            "year_t": int(current["time_start"]),
            "year_next": int(following["time_start"]),
            "time_t": int(current["time_end"]),
            "time_next": int(following["time_end"]),
            "x_t": self._features(current),
            "x_t_raw": current["x_raw"],
            "edge_index_t": current["edge_index"],
            "x_next": self._features(following),
            "x_next_raw": following["x_raw"],
            "edge_index_next": following["edge_index"],
            "edge_added": current["edge_added"],
            "edge_removed": current["edge_removed"],
        }
        if self.edge_input == "weighted":
            transition["edge_weight_t"] = current["edge_weight"]
            transition["edge_weight_next"] = following["edge_weight"]
        return transition

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.transition(self.indices[item])

    def iter_all(self) -> Iterator[dict[str, Any]]:
        for index in range(self.transition_count):
            yield self.transition(index)


def topology_transition_sanity(
    dataset: TGBNTradeTopologyDataset, transition_id: int
) -> dict[str, Any]:
    transition = dataset.transition(transition_id)
    num_nodes = int(dataset.metadata["num_nodes"])
    current = _edge_codes(transition["edge_index_t"], num_nodes)
    following = _edge_codes(transition["edge_index_next"], num_nodes)
    added = _edge_codes(transition["edge_added"], num_nodes)
    removed = _edge_codes(transition["edge_removed"], num_nodes)
    return {
        "transition_id": int(transition_id),
        "split": transition["split"],
        "year_t": int(transition["year_t"]),
        "year_next": int(transition["year_next"]),
        "current_edges": int(current.numel()),
        "future_edges": int(following.numel()),
        "edge_added": int(added.numel()),
        "edge_removed": int(removed.numel()),
        "has_current_weight": "edge_weight_t" in transition,
    }


def topology_transition_audit(dataset: TGBNTradeTopologyDataset) -> dict[str, Any]:
    num_nodes = int(dataset.metadata["num_nodes"])
    expected_x_dim = int(dataset.snapshots[0]["x"].shape[1])
    split_rank = {"train": 0, "val": 1, "test": 2}
    temporal_violations = 0
    split_violations = 0
    delta_violations = 0
    feature_violations = 0
    weight_violations = 0
    previous_split = -1
    counts = {"current_edges": 0, "future_edges": 0, "added_edges": 0, "removed_edges": 0}
    for transition in dataset.iter_all():
        current = _edge_codes(transition["edge_index_t"], num_nodes)
        following = _edge_codes(transition["edge_index_next"], num_nodes)
        added = _edge_codes(transition["edge_added"], num_nodes)
        removed = _edge_codes(transition["edge_removed"], num_nodes)
        reconstructed = torch.unique(
            torch.cat([_set_difference(current, removed), added]), sorted=True
        )
        if (
            torch.isin(added, current).any()
            or (~torch.isin(removed, current)).any()
            or not torch.equal(reconstructed, following)
        ):
            delta_violations += 1
        if int(transition["time_t"]) > int(transition["time_next"]):
            temporal_violations += 1
        rank = split_rank[str(transition["split"])]
        if rank < previous_split:
            split_violations += 1
        previous_split = rank
        if transition["x_t"].ndim != 2 or int(transition["x_t"].shape[1]) != expected_x_dim:
            feature_violations += 1
        if dataset.edge_input == "weighted":
            for edge_key, weight_key in (("edge_index_t", "edge_weight_t"), ("edge_index_next", "edge_weight_next")):
                weight = transition.get(weight_key)
                if (
                    weight is None
                    or weight.ndim != 1
                    or int(weight.shape[0]) != int(transition[edge_key].shape[1])
                    or not torch.isfinite(weight).all()
                    or (weight < 0).any()
                ):
                    weight_violations += 1
        counts["current_edges"] += int(current.numel())
        counts["future_edges"] += int(following.numel())
        counts["added_edges"] += int(added.numel())
        counts["removed_edges"] += int(removed.numel())
    violations = (
        temporal_violations
        + split_violations
        + delta_violations
        + feature_violations
        + weight_violations
    )
    return {
        "transitions_checked": int(dataset.transition_count),
        "input_mode": str(dataset.metadata["topology_input_mode"]),
        "edge_input": str(dataset.edge_input),
        "expected_input_dim": expected_x_dim,
        "counts": counts,
        "violations": {
            "time_order": temporal_violations,
            "chronological_split_order": split_violations,
            "delta_set_identity": delta_violations,
            "observed_input_shape": feature_violations,
            "observed_weight_shape_finiteness_nonnegativity": weight_violations,
        },
        "all_passed": bool(violations == 0),
    }


__all__ = [
    "TGBNTradeTopologyDataset",
    "build_tgbn_trade_topology_bundle",
    "load_processed_tgbn_trade_topology",
    "save_processed_tgbn_trade_topology",
    "topology_transition_audit",
    "topology_transition_sanity",
]
