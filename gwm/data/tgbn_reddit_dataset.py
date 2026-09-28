
from __future__ import annotations

from pathlib import Path
from typing import Any

from .tgbn_genre_dataset import TGBNGenreTransitionDataset
from .tgbn_reddit_builder import load_processed_tgbn_reddit


class TGBNRedditTransitionDataset(TGBNGenreTransitionDataset):

    def __init__(
        self,
        source: str | Path | dict[str, Any],
        split: str | None = None,
        *,
        include_property_observation_mask: bool = False,
    ):
        bundle = load_processed_tgbn_reddit(source) if not isinstance(source, dict) else source
        if bundle.get("metadata", {}).get("dataset_name") != "tgbn-reddit":
            raise ValueError("TGBNRedditTransitionDataset requires a tgbn-reddit artifact.")
        super().__init__(
            bundle,
            split=split,
            include_property_observation_mask=include_property_observation_mask,
        )


__all__ = ["TGBNRedditTransitionDataset"]
