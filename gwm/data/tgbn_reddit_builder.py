
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .tgbn_genre_builder import (
    build_tgbn_genre_snapshots,
    genre_transition_sanity,
    load_processed_tgbn_genre,
    save_processed_tgbn_genre,
)
REDDIT_FEATURE_NAMES = [
    "log_current_incident_unique_edges",
    "log_current_incident_interaction_count",
    "log_current_unique_neighbors",
    "log_current_incident_abs_edge_feature",
    "log_time_since_last_interaction",
    "log_cumulative_incident_interaction_count",
    "log_cumulative_unique_neighbors",
    "log_cumulative_incident_abs_edge_feature",
    "has_been_seen",
]


def build_tgbn_reddit_snapshots(stream: Any) -> dict[str, Any]:
    if stream.dataset_name != "tgbn-reddit":
        raise ValueError("tgbn-reddit builder received a different TGB stream.")
    bundle = build_tgbn_genre_snapshots(stream)
    metadata = dict(bundle["metadata"])
    metadata.update(
        {
            "dataset_name": "tgbn-reddit",
            "node_feature_names": REDDIT_FEATURE_NAMES,
            "input_feature_definition": (
                "Leakage-safe graph-derived Reddit interaction/topology state from events strictly "
                "before each official Reddit label timestamp."
            ),
            "observed_property_state_definition": (
                "Y_i is the official tgbn-reddit label batch at L_i scattered into [N, 698]; "
                "subreddit-coordinate rows and users without a supplied current row are zero-filled. "
                "GraphEncoder receives concat(X_i, Y_i)."
            ),
            "target_boundary_rule": (
                "For target Y_(i+1), source G_i contains only events with timestamp < L_i and "
                "observed Y_i. G_(i+1) and Y_(i+1) enter only target-latent/property supervision."
            ),
            "property_state_storage": (
                "sparse official Reddit label batches; lazily zero-scattered to [N, 698] per transition"
            ),
            "edge_weight_used_by_encoder": False,
            "graph_input_recommendation": (
                "binary: TGB's Reddit TemporalData edge feature is not treated as a semantic "
                "non-negative relation weight by this passive-GWM adapter."
            ),
        }
    )
    bundle["metadata"] = metadata
    return bundle


def save_processed_tgbn_reddit(bundle: dict[str, Any], path: str | Path) -> Path:
    return save_processed_tgbn_genre(bundle, path)


def load_processed_tgbn_reddit(path: str | Path) -> dict[str, Any]:
    return load_processed_tgbn_genre(path)


def reddit_transition_sanity(bundle: dict[str, Any], transition_id: int) -> dict[str, Any]:
    return genre_transition_sanity(bundle, transition_id)


__all__ = [
    "REDDIT_FEATURE_NAMES",
    "build_tgbn_reddit_snapshots",
    "load_processed_tgbn_reddit",
    "reddit_transition_sanity",
    "save_processed_tgbn_reddit",
]
