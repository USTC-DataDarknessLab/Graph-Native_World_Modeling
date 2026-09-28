
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from .tgbn_genre_dataset import TGBNGenreTransitionDataset
from .tgbn_reddit_dataset import TGBNRedditTransitionDataset
from .tgbn_trade_dataset import TGBNTradeTransitionDataset
from .tgbn_trade_topology import TGBNTradeTopologyDataset
from .transition_dataset import GraphTransitionDataset
from .transition_cache import (
    DEFAULT_TRANSITION_CACHE_BYTES,
    TransitionRecordCache,
)


ACTION_BENCHMARK_DATASETS: dict[str, tuple[str, ...]] = {
    "topology": ("trade", "un_vote", "contact", "socialevo"),
    "node_property": ("trade", "genre", "reddit"),
}

ACTION_BENCHMARK_PATHS: dict[tuple[str, str], str] = {
    ("topology", "trade"): "data/processed/trade_T2.pt",
    ("topology", "un_vote"): "data/processed/un_vote_T2.pt",
    ("topology", "contact"): "data/processed/contact_T2.pt",
    ("topology", "socialevo"): "data/processed/socialevo_T2.pt",
    ("node_property", "trade"): (
        "data/processed/trade_T1.pt"
    ),
    ("node_property", "genre"): (
        "data/processed/genre_T1.pt"
    ),
    ("node_property", "reddit"): (
        "data/processed/reddit_T1.pt"
    ),
}


def resolve_action_benchmark_path(
    root: Path, task: str, dataset: str | None, processed: str | None
) -> Path:
    if processed is not None:
        return root / processed
    if dataset is None:
        dataset = ACTION_BENCHMARK_DATASETS[task][0]
    key = (task, dataset)
    if key not in ACTION_BENCHMARK_PATHS:
        allowed = ", ".join(ACTION_BENCHMARK_DATASETS[task])
        raise ValueError(f"task={task} supports datasets: {allowed}")
    return root / ACTION_BENCHMARK_PATHS[key]


def infer_action_benchmark_dataset(task: str, source: Path) -> str:
    name = source.name.lower()
    for dataset in ACTION_BENCHMARK_DATASETS[task]:
        expected = Path(ACTION_BENCHMARK_PATHS[(task, dataset)]).name.lower()
        if name == expected:
            return dataset
    if "trade" in name:
        return "trade"
    if "genre" in name:
        return "genre"
    if "reddit" in name:
        return "reddit"
    if "un_vote" in name:
        return "un_vote"
    if "flights" in name:
        return "flights"
    if "contact" in name:
        return "contact"
    raise ValueError(f"Cannot infer dataset alias from {source}.")


def _mmap_bundle(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


class ActionBenchmarkTransitionDataset(Sequence[dict[str, Any]]):

    def __init__(
        self,
        source: str | Path,
        *,
        task: str,
        dataset: str | None = None,
        split: str | None = "train",
        edge_input: str = "binary",
        include_property_observation_mask: bool = True,
        cache_transitions: bool = True,
        transition_cache_max_bytes: int = DEFAULT_TRANSITION_CACHE_BYTES,
    ) -> None:
        if task not in ACTION_BENCHMARK_DATASETS:
            raise ValueError(f"Unknown action benchmark task={task!r}.")
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be None/train/val/test")
        if edge_input not in {"binary", "weighted"}:
            raise ValueError("edge_input must be binary or weighted")
        self.source = Path(source)
        self.task = task
        self.dataset = dataset or infer_action_benchmark_dataset(task, self.source)
        self.edge_input = str(edge_input)
        if self.dataset not in ACTION_BENCHMARK_DATASETS[task]:
            raise ValueError(
                f"dataset={self.dataset} is not registered for task={task}."
            )
        self.split = split

        if task == "topology" and self.dataset == "trade":
            bundle = _mmap_bundle(self.source)
            self.base: Any = TGBNTradeTopologyDataset(
                bundle,
                edge_input=edge_input,
                include_terminal_transition=True,
                split=split,
            )
        elif task == "node_property":
            bundle = _mmap_bundle(self.source)
            dataset_type = {
                "trade": TGBNTradeTransitionDataset,
                "genre": TGBNGenreTransitionDataset,
                "reddit": TGBNRedditTransitionDataset,
            }[self.dataset]
            self.base = dataset_type(
                bundle,
                split=split,
                include_property_observation_mask=include_property_observation_mask,
            )
        else:
            self.base = GraphTransitionDataset(self.source, split=split)

        self._transition_cache = (
            TransitionRecordCache(int(transition_cache_max_bytes))
            if cache_transitions
            else None
        )

        self.metadata = dict(self.base.metadata)
        self.metadata.update(
            {
                "action_benchmark_task": task,
                "action_benchmark_dataset": self.dataset,
                "transition_access": "lazy chronological",
            }
        )
        if len(self.base):
            sample = self[0]
            self.metadata["model_input_dim"] = int(sample["x_t"].shape[1])
            self.metadata["node_feature_dim"] = int(sample["x_t"].shape[1])

    def __len__(self) -> int:
        return len(self.base)

    def _normalize_property(self, transition: dict[str, Any]) -> dict[str, Any]:
        current_state = transition["property_state_t"].float()
        target_ids = transition["property_node_ids"].long()
        target = transition["property_target"].float()
        current_target = transition.get("property_current_target")
        if current_target is None:
            current_target = current_state.index_select(0, target_ids)
        base_feature_dim = int(
            self.base.snapshots[int(transition["transition_id"])]["x"].shape[1]
        )
        normalized = dict(transition)
        normalized.update(
            {
                "property_t": current_state,
                "property_node_ids": target_ids,
                "property_target": target,
                "property_current_target": current_target.float(),
                "property_observed": transition["property_node_ids_t"].long(),
                "property_slice": slice(
                    base_feature_dim,
                    base_feature_dim + int(target.shape[1]),
                ),
            }
        )
        return normalized

    def _normalize(self, transition: dict[str, Any]) -> dict[str, Any]:
        normalized = (
            self._normalize_property(transition)
            if self.task == "node_property"
            else transition
        )





        if self.task == "topology":
            normalized = dict(normalized)
            if self.edge_input == "binary":
                if "edge_weight_t" in normalized:
                    normalized["edge_weight_t"] = None
                if "edge_weight_next" in normalized:
                    normalized["edge_weight_next"] = None
            else:
                for key in ("edge_weight_t", "edge_weight_next"):
                    if key in normalized and normalized[key] is None:
                        raise ValueError(
                            f"edge_input=weighted requires observed {key}."
                        )
        return normalized

    def __getitem__(self, item: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(len(self)))]
        item = int(item)
        cache_key = ("transition", item)
        if self._transition_cache is not None:
            cached = self._transition_cache.get(cache_key)
            if cached is not None:
                return cached
        if self.task == "node_property" and hasattr(
            self.base, "forecast_transition"
        ):
            result = self._normalize(
                self.base.forecast_transition(self.base.indices[item])
            )
        else:
            result = self._normalize(self.base[item])
        if self._transition_cache is not None:
            return self._transition_cache.put(cache_key, result)
        return result

    def target_graph(self, item: int) -> dict[str, Any]:

        if not 0 <= int(item) < len(self):
            raise IndexError(f"item={item} is out of range")
        cache_key = ("target", int(item))
        if self._transition_cache is not None:
            cached = self._transition_cache.get(cache_key)
            if cached is not None:
                return cached
        if self.task == "node_property" and hasattr(self.base, "transition"):
            transition_id = int(self.base.indices[int(item)])
            transition = self.base.transition(transition_id)
            result = {
                key: transition[key]
                for key in (
                    "x_next",
                    "edge_index_next",
                    "edge_weight_next",
                )
                if key in transition
            }
        else:
            transition = self[int(item)]
            result = {
                key: transition[key]
                for key in ("x_next", "edge_index_next", "edge_weight_next")
                if key in transition
            }
        if self._transition_cache is not None:
            return self._transition_cache.put(cache_key, result)
        return result

    def iter_all(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


__all__ = [
    "ACTION_BENCHMARK_DATASETS",
    "ACTION_BENCHMARK_PATHS",
    "ActionBenchmarkTransitionDataset",
    "infer_action_benchmark_dataset",
    "resolve_action_benchmark_path",
]
