
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .snapshot_builder import load_processed_snapshots


class GraphTransitionDataset:

    def __init__(self, source: str | Path | dict[str, Any], split: str | None = None):
        self.bundle = load_processed_snapshots(source) if not isinstance(source, dict) else source
        self.snapshots: list[dict[str, Any]] = self.bundle["snapshots"]
        self.metadata: dict[str, Any] = self.bundle["metadata"]
        if len(self.snapshots) < 2:
            raise ValueError("Processed data has fewer than two snapshots.")
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be one of None/train/val/test")
        self.split = split
        self.indices = [
            index
            for index in range(len(self.snapshots) - 1)
            if split is None or self.snapshots[index + 1]["split"] == split
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def transition(self, transition_index: int) -> dict[str, Any]:
        current = self.snapshots[transition_index]
        following = self.snapshots[transition_index + 1]
        return {
            "transition_id": transition_index,
            "split": following["split"],
            "x_t": current["x"],
            "x_t_raw": current["x_raw"],
            "edge_index_t": current["edge_index"],
            "edge_weight_t": current["edge_weight"],
            "x_next": following["x"],
            "x_next_raw": following["x_raw"],
            "edge_index_next": following["edge_index"],
            "edge_weight_next": following["edge_weight"],
            "node_delta": current["node_delta"],
            "node_delta_raw": current["node_delta_raw"],
            "node_changed": current["node_changed"],
            "node_changed_with_recency": current.get(
                "node_changed_with_recency", current["node_changed"]
            ),
            "edge_added": current["edge_added"],
            "edge_removed": current["edge_removed"],
            "time_t": current["time_end"],
            "time_next": following["time_end"],
            "snapshot_t": current["snapshot_id"],
            "snapshot_next": following["snapshot_id"],




            **(
                {"t3_centres": current["t3_centres"]}
                if "t3_centres" in current
                else {}
            ),
        }

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.transition(self.indices[item])

    def iter_all(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self.snapshots) - 1):
            yield self.transition(index)


def sample_candidate_edges(
    edge_index_next: torch.Tensor,
    num_nodes: int,
    neg_ratio: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if neg_ratio < 0:
        raise ValueError("neg_ratio must be non-negative")
    positives = edge_index_next.detach().cpu().to(torch.long)
    positive_count = int(positives.shape[1])
    if positive_count == 0:
        return positives, torch.empty(0, dtype=torch.float32)
    requested = int(np.ceil(positive_count * neg_ratio))
    pos_codes = {
        int(source) * num_nodes + int(destination)
        for source, destination in positives.t().tolist()
    }
    available = num_nodes * max(num_nodes - 1, 0) - sum(
        1 for code in pos_codes if (code // num_nodes) != (code % num_nodes)
    )
    target_negatives = min(requested, max(available, 0))
    negative_codes: set[int] = set()


    attempts = 0
    maximum_attempts = max(1000, target_negatives * 50)
    while len(negative_codes) < target_negatives and attempts < maximum_attempts:
        sources = rng.integers(0, num_nodes, size=max(16, target_negatives - len(negative_codes)))
        destinations = rng.integers(
            0, num_nodes, size=max(16, target_negatives - len(negative_codes))
        )
        for source, destination in zip(sources.tolist(), destinations.tolist()):
            if source == destination:
                continue
            code = source * num_nodes + destination
            if code not in pos_codes:
                negative_codes.add(code)
                if len(negative_codes) == target_negatives:
                    break
        attempts += len(sources)
    if len(negative_codes) < target_negatives:
        for source in range(num_nodes):
            for destination in range(num_nodes):
                if source == destination:
                    continue
                code = source * num_nodes + destination
                if code not in pos_codes:
                    negative_codes.add(code)
                    if len(negative_codes) == target_negatives:
                        break
            if len(negative_codes) == target_negatives:
                break
    if negative_codes:
        negatives = torch.tensor(
            [[code // num_nodes for code in negative_codes], [code % num_nodes for code in negative_codes]],
            dtype=torch.long,
        )
        candidates = torch.cat([positives, negatives], dim=1)
        labels = torch.cat(
            [torch.ones(positive_count, dtype=torch.float32), torch.zeros(len(negative_codes), dtype=torch.float32)]
        )
    else:
        candidates = positives
        labels = torch.ones(positive_count, dtype=torch.float32)
    return candidates, labels
