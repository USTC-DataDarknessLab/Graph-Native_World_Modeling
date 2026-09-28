
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import torch

from .tgbn_genre_builder import load_processed_tgbn_genre


class TGBNGenreTransitionDataset:

    def __init__(
        self,
        source: str | Path | dict[str, Any],
        split: str | None = None,
        *,
        include_property_observation_mask: bool = False,
    ):
        self.bundle = (
            load_processed_tgbn_genre(source) if not isinstance(source, dict) else source
        )
        self.snapshots: list[dict[str, Any]] = self.bundle["snapshots"]


        self.metadata: dict[str, Any] = dict(self.bundle["metadata"])
        if len(self.snapshots) < 3:
            raise ValueError("Processed tgbn-genre data has fewer than two labelled snapshots.")
        if not bool(self.metadata.get("property_state_in_graph", False)):
            raise ValueError("This adapter requires an observed official Y state in g_t.")
        self.transition_count = len(self.snapshots) - 2
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be None/train/val/test")
        self.split = split
        self.indices = [
            index
            for index in range(self.transition_count)
            if split is None or self.snapshots[index + 1]["split"] == split
        ]
        self.include_property_observation_mask = bool(include_property_observation_mask)
        self.metadata["property_observation_mask_in_graph"] = (
            self.include_property_observation_mask
        )
        self.metadata["model_input_dim"] = int(self.metadata["model_input_dim"]) + int(
            self.include_property_observation_mask
        )
        self._property_state_cache: OrderedDict[int, torch.Tensor] = OrderedDict()





        self._property_target_cache: OrderedDict[int, torch.Tensor] = OrderedDict()





        self._observed_graph_state_cache: OrderedDict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()

    def _property_target(self, snapshot_id: int) -> torch.Tensor:

        cached = self._property_target_cache.get(int(snapshot_id))
        if cached is not None:
            self._property_target_cache.move_to_end(int(snapshot_id))
            return cached
        snapshot = self.snapshots[int(snapshot_id)]
        dense = snapshot.get("property_target")
        if dense is None:
            shape = tuple(int(value) for value in snapshot["property_target_shape"])
            dense = torch.zeros(shape, dtype=torch.float32)
            flat_index = snapshot["property_target_flat_index"].long()
            if flat_index.numel():
                dense.reshape(-1).index_copy_(
                    0,
                    flat_index,
                    snapshot["property_target_nonzero"].to(torch.float32),
                )
        else:
            dense = dense.float()
        self._property_target_cache[int(snapshot_id)] = dense
        while len(self._property_target_cache) > 2:
            self._property_target_cache.popitem(last=False)
        return dense

    def __len__(self) -> int:
        return len(self.indices)

    def _property_state(self, snapshot_id: int) -> torch.Tensor:
        cached = self._property_state_cache.get(snapshot_id)
        if cached is not None:
            self._property_state_cache.move_to_end(snapshot_id)
            return cached
        snapshot = self.snapshots[snapshot_id]
        num_nodes = int(self.metadata["num_nodes"])
        target_dim = int(self.metadata["official_target_dim"])
        state = torch.zeros((num_nodes, target_dim), dtype=torch.float32)
        node_ids = snapshot["property_node_ids"].long()
        if node_ids.numel():
            state.index_copy_(0, node_ids, self._property_target(snapshot_id))
        self._property_state_cache[snapshot_id] = state


        while len(self._property_state_cache) > 2:
            self._property_state_cache.popitem(last=False)
        return state

    def _property_observation_mask(self, snapshot_id: int) -> torch.Tensor:
        mask = torch.zeros((int(self.metadata["num_nodes"]), 1), dtype=torch.float32)
        ids = self.snapshots[snapshot_id]["property_node_ids"].long()
        if ids.numel():
            mask.index_fill_(0, ids, 1.0)
        return mask

    def _observed_graph_state(
        self, snapshot_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._observed_graph_state_cache.get(snapshot_id)
        if cached is not None:
            self._observed_graph_state_cache.move_to_end(snapshot_id)
            return cached

        snapshot = self.snapshots[snapshot_id]
        property_state = self._property_state(snapshot_id)
        property_mask = self._property_observation_mask(snapshot_id)
        graph_input = torch.cat([snapshot["x"], property_state], dim=-1)
        if self.include_property_observation_mask:
            graph_input = torch.cat([graph_input, property_mask], dim=-1)
        cached = (graph_input, property_state, property_mask)
        self._observed_graph_state_cache[snapshot_id] = cached
        while len(self._observed_graph_state_cache) > 2:
            self._observed_graph_state_cache.popitem(last=False)
        return cached

    def property_transition(self, transition_id: int) -> dict[str, Any]:
        if transition_id < 0 or transition_id >= self.transition_count:
            raise IndexError(f"transition_id={transition_id} is out of range")
        current = self.snapshots[transition_id]
        following = self.snapshots[transition_id + 1]
        target_node_ids = following["property_node_ids"].long()
        current_node_ids = current["property_node_ids"].long()
        if current_node_ids.numel() != torch.unique(current_node_ids).numel():
            raise ValueError("Official property batch unexpectedly repeats a node ID.")


        lookup = torch.full((int(self.metadata["num_nodes"]),), -1, dtype=torch.long)
        if current_node_ids.numel():
            lookup[current_node_ids] = torch.arange(current_node_ids.numel(), dtype=torch.long)
        positions = lookup[target_node_ids]
        current_observed_mask = positions.ge(0)
        property_dim = int(self.metadata["official_target_dim"])
        current_target_values = torch.zeros(
            (target_node_ids.numel(), property_dim), dtype=torch.float32
        )
        if current_observed_mask.any():
            current_target_values[current_observed_mask] = self._property_target(
                transition_id
            ).index_select(
                0, positions[current_observed_mask]
            )
        current_property = self._property_target(transition_id)
        target_property = self._property_target(transition_id + 1)
        return {
            "transition_id": int(transition_id),
            "split": following["split"],
            "source_label_time": current["label_time"],
            "label_time": following["label_time"],
            "property_node_ids_t": current_node_ids,
            "property_observed_t": current_property,
            "property_node_ids": target_node_ids,
            "property_target": target_property,
            "property_current_target": current_target_values,
            "property_current_observed_mask": current_observed_mask,
        }

    def transition(self, transition_id: int) -> dict[str, Any]:
        if transition_id < 0 or transition_id >= self.transition_count:
            raise IndexError(f"transition_id={transition_id} is out of range")
        current = self.snapshots[transition_id]
        following = self.snapshots[transition_id + 1]
        x_t, property_state_t, property_observation_mask_t = self._observed_graph_state(
            transition_id
        )
        x_next, property_state_next, property_observation_mask_next = (
            self._observed_graph_state(transition_id + 1)
        )
        target_node_ids = following["property_node_ids"].long()
        current_node_ids = current["property_node_ids"].long()
        current_target_values = property_state_t.index_select(0, target_node_ids)



        current_observed_mask = torch.isin(target_node_ids, current_node_ids)
        return {
            "transition_id": int(transition_id),
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
            "property_state_t": property_state_t,
            "property_observation_mask_t": property_observation_mask_t,
            "property_observation_mask_next": property_observation_mask_next,
            "property_node_ids_t": current_node_ids,
            "property_observed_t": self._property_target(transition_id),
            "property_node_ids": target_node_ids,
            "property_target": self._property_target(transition_id + 1),
            "property_current_target": current_target_values,
            "property_current_observed_mask": current_observed_mask,
            "time_t": current["time_end"],
            "time_next": following["time_end"],
            "latent_train_allowed": bool(current["latent_train_allowed"]),
        }

    def forecast_transition(self, transition_id: int) -> dict[str, Any]:

        sparse = self.property_transition(transition_id)
        current = self.snapshots[transition_id]
        x_t, property_state_t, property_observation_mask_t = self._observed_graph_state(
            transition_id
        )
        return {
            **sparse,
            "x_t": x_t,
            "x_t_raw": current["x_raw"],
            "edge_index_t": current["edge_index"],
            "edge_weight_t": current["edge_weight"],
            "property_state_t": property_state_t,
            "property_observation_mask_t": property_observation_mask_t,
            "time_t": current["time_end"],
            "latent_train_allowed": bool(current["latent_train_allowed"]),
        }

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.transition(self.indices[item])

    def iter_all(self) -> Iterator[dict[str, Any]]:
        for index in range(self.transition_count):
            yield self.transition(index)

    def iter_property_only(self) -> Iterator[dict[str, Any]]:
        for index in range(self.transition_count):
            yield self.property_transition(index)


__all__ = ["TGBNGenreTransitionDataset"]
