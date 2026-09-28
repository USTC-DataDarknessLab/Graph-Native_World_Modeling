
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch

from .tgbn_trade_builder import load_processed_tgbn_trade


class TGBNTradeTransitionDataset:

    def __init__(
        self,
        source: str | Path | dict[str, Any],
        split: str | None = None,
        *,
        include_property_observation_mask: bool = False,
    ):
        self.bundle = load_processed_tgbn_trade(source) if not isinstance(source, dict) else source
        self.snapshots: list[dict[str, Any]] = self.bundle["snapshots"]
        self.metadata: dict[str, Any] = dict(self.bundle["metadata"])
        if len(self.snapshots) < 2:
            raise ValueError("Processed tgbn-trade data has fewer than two snapshots.")
        self.property_state_in_graph = bool(self.metadata.get("property_state_in_graph", False))
        self.transition_count = len(self.snapshots) - (2 if self.property_state_in_graph else 1)
        if self.transition_count < 1:
            raise ValueError("Processed tgbn-trade data has no valid property transitions.")
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be None/train/val/test")
        self.split = split
        self.indices = [
            index
            for index in range(self.transition_count)
            if split is None
            or (
                self.snapshots[index + 1]["split"]
                if self.property_state_in_graph
                else self.snapshots[index]["split"]
            )
            == split
        ]
        self.include_property_observation_mask = bool(include_property_observation_mask)
        self.metadata["property_observation_mask_in_graph"] = (
            self.include_property_observation_mask
        )
        self.metadata["model_input_dim"] = int(self.metadata["model_input_dim"]) + int(
            self.include_property_observation_mask
        )

    def _property_observation_mask(self, snapshot: dict[str, Any]) -> torch.Tensor:
        mask = torch.zeros((int(self.metadata["num_nodes"]), 1), dtype=torch.float32)
        ids = snapshot["property_node_ids"].long()
        if ids.numel():
            mask.index_fill_(0, ids, 1.0)
        return mask

    def __len__(self) -> int:
        return len(self.indices)

    def transition(self, transition_id: int) -> dict[str, Any]:
        if transition_id < 0 or transition_id >= self.transition_count:
            raise IndexError(f"transition_id={transition_id} is out of range")
        current = self.snapshots[transition_id]
        following = self.snapshots[transition_id + 1]
        if self.property_state_in_graph:
            mask_t = self._property_observation_mask(current)
            mask_next = self._property_observation_mask(following)
            x_t = torch.cat([current["x"], current["property_state"]], dim=-1)
            x_next = torch.cat([following["x"], following["property_state"]], dim=-1)
            if self.include_property_observation_mask:
                x_t = torch.cat([x_t, mask_t], dim=-1)
                x_next = torch.cat([x_next, mask_next], dim=-1)
            return {
                "transition_id": transition_id,
                "split": following["split"],
                "source_label_time": current["label_time"],
                "label_time": following["label_time"],
                "x_t": x_t,
                "x_t_raw": current["x_raw"],
                "edge_index_t": current["edge_index"],
                "edge_weight_t": current["edge_weight"],
                "x_next": x_next,
                "x_next_raw": following["x_raw"],
                "edge_index_next": following["edge_index"],
                "edge_weight_next": following["edge_weight"],
                "property_state_t": current["property_state"],
                "property_observation_mask_t": mask_t,
                "property_observation_mask_next": mask_next,
                "property_node_ids_t": current["property_node_ids"],
                "property_observed_t": current["property_target"],
                "property_node_ids": following["property_node_ids"],
                "property_target": following["property_target"],




                "property_current_observed_mask": torch.isin(
                    following["property_node_ids"].long(),
                    current["property_node_ids"].long(),
                ),
                "time_t": current["time_end"],
                "time_next": following["time_end"],
                "latent_train_allowed": bool(current["latent_train_allowed"]),
            }
        return {
            "transition_id": transition_id,
            "split": current["split"],
            "label_time": current["label_time"],
            "x_t": current["x"],
            "x_t_raw": current["x_raw"],
            "edge_index_t": current["edge_index"],
            "edge_weight_t": current["edge_weight"],
            "x_next": following["x"],
            "x_next_raw": following["x_raw"],
            "edge_index_next": following["edge_index"],
            "edge_weight_next": following["edge_weight"],
            "property_node_ids": current["property_node_ids"],
            "property_target": current["property_target"],
            "time_t": current["time_end"],
            "time_next": following["time_end"],
            "latent_train_allowed": bool(current["latent_train_allowed"]),
        }

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.transition(self.indices[item])

    def iter_all(self) -> Iterator[dict[str, Any]]:
        for index in range(self.transition_count):
            yield self.transition(index)


__all__ = ["TGBNTradeTransitionDataset"]
