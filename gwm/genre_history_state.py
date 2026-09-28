
from __future__ import annotations

from typing import Any

import torch


class GenreObservedPropertyHistory:

    def __init__(
        self,
        num_nodes: int,
        target_dim: int,
        *,
        num_genre_coordinate_nodes: int,
    ) -> None:
        if not 0 <= int(num_genre_coordinate_nodes) < int(num_nodes):
            raise ValueError("num_genre_coordinate_nodes must be in [0, num_nodes).")
        self.num_nodes = int(num_nodes)
        self.target_dim = int(target_dim)
        self.num_genre_coordinate_nodes = int(num_genre_coordinate_nodes)
        shape = (self.num_nodes, self.target_dim)
        self.running_sum = torch.zeros(shape, dtype=torch.float32)
        self.running_count = torch.zeros(self.num_nodes, dtype=torch.float32)
        self.global_sum = torch.zeros(self.target_dim, dtype=torch.float32)
        self.global_count = 0
        self.entity_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.entity_mask[self.num_genre_coordinate_nodes :] = True

    def _check_observation(self, node_ids: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = node_ids.detach().cpu().long()
        observed = values.detach().cpu().float()
        if ids.ndim != 1 or observed.ndim != 2 or observed.shape != (ids.numel(), self.target_dim):
            raise ValueError("Property observation ids/values have incompatible shapes.")
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.num_nodes):
            raise ValueError("Property observation IDs fall outside the fixed graph node universe.")
        if ids.numel() and torch.unique(ids).numel() != ids.numel():
            raise ValueError("Official property observation batch contains duplicate entities.")
        if ids.numel() and not bool(self.entity_mask.index_select(0, ids).all()):
            raise ValueError("Only user/property-entity nodes may carry a property-history state.")
        return ids, observed

    def observe(self, node_ids: torch.Tensor, values: torch.Tensor) -> None:
        ids, observed = self._check_observation(node_ids, values)
        if not ids.numel():
            return
        self.running_sum.index_add_(0, ids, observed)
        self.running_count.index_add_(0, ids, torch.ones(ids.numel(), dtype=torch.float32))
        self.global_sum += observed.sum(dim=0)
        self.global_count += int(ids.numel())

    def _state_from_statistics(
        self,
        running_sum: torch.Tensor,
        running_count: torch.Tensor,
        global_sum: torch.Tensor,
        global_count: int,
    ) -> torch.Tensor:
        if global_count:
            fallback = global_sum / float(global_count)
        else:
            fallback = torch.zeros(self.target_dim, dtype=torch.float32)
        state = fallback.unsqueeze(0).expand(self.num_nodes, -1).clone()
        observed = running_count > 0
        if observed.any():
            state[observed] = running_sum[observed] / running_count[observed].unsqueeze(1)


        state[~self.entity_mask] = 0.0
        return state

    def state(self) -> torch.Tensor:
        return self._state_from_statistics(
            self.running_sum, self.running_count, self.global_sum, self.global_count
        )

    def state_dict(self) -> dict[str, torch.Tensor | int]:

        return {
            "running_sum": self.running_sum.clone(),
            "running_count": self.running_count.clone(),
            "global_sum": self.global_sum.clone(),
            "global_count": int(self.global_count),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int]) -> None:

        self.running_sum.copy_(torch.as_tensor(state["running_sum"]))
        self.running_count.copy_(torch.as_tensor(state["running_count"]))
        self.global_sum.copy_(torch.as_tensor(state["global_sum"]))
        self.global_count = int(state["global_count"])

    def copy(self) -> "GenreObservedPropertyHistory":

        cloned = GenreObservedPropertyHistory(
            self.num_nodes,
            self.target_dim,
            num_genre_coordinate_nodes=self.num_genre_coordinate_nodes,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned

    def hypothetical_state_after(self, node_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        ids, observed = self._check_observation(node_ids, values)
        if not ids.numel():
            return self.state()



        global_sum = self.global_sum + observed.sum(dim=0)
        global_count = self.global_count + int(ids.numel())
        if global_count:
            fallback = global_sum / float(global_count)
        else:
            fallback = torch.zeros(self.target_dim, dtype=torch.float32)
        state = fallback.unsqueeze(0).expand(self.num_nodes, -1).clone()
        observed_before = self.running_count > 0
        if observed_before.any():
            state[observed_before] = (
                self.running_sum[observed_before]
                / self.running_count[observed_before].unsqueeze(1)
            )
        future_sum = self.running_sum.index_select(0, ids) + observed
        future_count = self.running_count.index_select(0, ids) + 1.0
        state.index_copy_(0, ids, future_sum / future_count.unsqueeze(1))
        state[~self.entity_mask] = 0.0
        return state


def augment_genre_transition_with_history(
    transition: dict[str, Any], history: GenreObservedPropertyHistory
) -> dict[str, Any]:
    history.observe(transition["property_node_ids_t"], transition["property_observed_t"])
    state_t = history.state()
    state_next_target = history.hypothetical_state_after(
        transition["property_node_ids"], transition["property_target"]
    )
    target_ids = transition["property_node_ids"].long()
    augmented = dict(transition)
    augmented["property_history_state_t"] = state_t
    augmented["property_history_reference"] = state_t.index_select(0, target_ids)
    augmented["property_history_state_next_target"] = state_next_target
    augmented["x_t"] = torch.cat([transition["x_t"], state_t], dim=-1)
    augmented["x_next"] = torch.cat([transition["x_next"], state_next_target], dim=-1)
    return augmented


def fit_genre_history_residual_normalization(
    dataset: Any, *, min_std: float = 1e-6
) -> dict[str, Any]:
    from .property_transition import fit_residual_normalization

    metadata = dataset.metadata
    coordinate_nodes = metadata.get(
        "num_genre_coordinate_nodes", metadata.get("num_target_coordinate_nodes")
    )
    if coordinate_nodes is None:
        raise ValueError("Property-history state requires a coordinate-node count in metadata.")
    history = GenreObservedPropertyHistory(
        int(metadata["num_nodes"]),
        int(metadata["official_target_dim"]),
        num_genre_coordinate_nodes=int(coordinate_nodes),
    )

    def residual_batches() -> Any:
        for transition in dataset.iter_all():
            if transition["split"] != "train":
                break
            augmented = augment_genre_transition_with_history(transition, history)
            yield (
                augmented["property_target"].float()
                - augmented["property_history_reference"].float()
            )

    return fit_residual_normalization(residual_batches(), min_std=min_std)


__all__ = [
    "GenreObservedPropertyHistory",
    "augment_genre_transition_with_history",
    "fit_genre_history_residual_normalization",
]
