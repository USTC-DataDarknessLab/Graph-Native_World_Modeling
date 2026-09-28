
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


_PROTOCOL_ID = "WorldGraph-task-defaults-v1"
_SHARED_EXECUTION = {
    "seeds": {"count": 5, "default_start": 1, "policy": "consecutive"},
    "train_transitions": "complete chronological source split",
    "validation_transitions": "complete chronological validation split",
    "test_transitions": "complete chronological test split",
    "checkpoint_selection": "validation only",
    "observability": "current snapshot and released history only",
}



_TASK_EXECUTION = {
    "T1": {"epochs": 60, "patience": 20, "val_every": 1},
    "T2": {"epochs": 60, "patience": 20, "val_every": 1},
    "T3": {"epochs": 60, "patience": 20, "val_every": 1},
}


def _task_protocol() -> dict[str, Any]:

    return {
        "protocol_id": _PROTOCOL_ID,
        "paper_task_mapping": {"T1": "node", "T2": "edge", "T3": "graph"},
        "shared_execution": deepcopy(_SHARED_EXECUTION),
        "tasks": {
            "T1": {
                "name": "node",
                "execution": deepcopy(_TASK_EXECUTION["T1"]),
                "benchmark_datasets": ["trade", "genre", "reddit"],
                "metrics": [
                    "node_addition_f1",
                    "node_removal_f1",
                    "semantic_feature_change_f1",
                    "official_ndcg_at_10",
                    "changed_ndcg_at_10",
                ],
                "semantic_change": {
                    "label": "top10_property_support_jaccard_distance",
                    "train_quantile": 0.6,
                    "candidate_rows": "all comparable official property rows",
                },
                "common_source_features": {
                    "property_observation_mask": True,
                    "observed_degree": True,
                },
                "datasets": {
                    "trade": {
                        "artifact": "data/processed/trade_T1.pt",
                        "edge_input": "weighted",
                        "node_edit_construction": {
                            "semi_synthetic": True,
                            "scale": 1.0,
                            "rate_floor": 0.0,
                            "strategy": "long_horizon_weighted_stratified",
                            "contrastive_addition_negatives": True,
                        },
                    },
                    "genre": {
                        "artifact": "data/processed/genre_T1.pt",
                        "edge_input": "weighted",
                        "node_edit_construction": {
                            "semi_synthetic": True,
                            "scale": 1.0,
                            "rate_floor": 0.0,
                            "strategy": "long_horizon_hybrid_stratified",
                            "contrastive_addition_negatives": True,
                            "candidate_scope": "dynamic_entity_nodes",
                        },
                    },
                    "reddit": {
                        "artifact": "data/processed/reddit_T1.pt",
                        "edge_input": "binary",
                        "node_edit_construction": {
                            "semi_synthetic": True,
                            "scale": 1.0,
                            "rate_floor": 0.0,
                            "strategy": "long_horizon_hybrid_stratified",
                            "contrastive_addition_negatives": True,
                            "candidate_scope": "dynamic_entity_nodes",
                        },
                    },
                },
            },
            "T2": {
                "name": "edge",
                "execution": deepcopy(_TASK_EXECUTION["T2"]),
                "history_length": 8,
                "benchmark_datasets": [
                    "trade",
                    "un_vote",
                    "contact",
                    "socialevo",
                ],
                "metrics": [
                    "edge_addition_f1",
                    "edge_removal_f1",
                    "edge_change_only_macro_f1",
                    "reconstructed_topology_f1",
                ],
                "primary_metrics": [
                    "edge_addition_f1",
                    "edge_removal_f1",
                    "edge_change_only_macro_f1",
                ],
                "auxiliary_metrics": ["reconstructed_topology_f1"],
                "candidate_space": {
                    "addition": "all legal currently absent pairs",
                    "removal": "all current edges",
                    "future_changed_pairs_or_edge_count": False,
                },
                "datasets": {
                    "trade": {
                        "artifact": "data/processed/trade_T2.pt",
                        "edge_input": "weighted",
                    },
                    "un_vote": {
                        "artifact": "data/processed/un_vote_T2.pt",
                        "edge_input": "weighted",
                        "execution": {"epochs": 150},
                    },
                    "contact": {
                        "artifact": "data/processed/contact_T2.pt",
                        "edge_input": "binary",
                        "execution": {"epochs": 150},
                    },
                    "socialevo": {
                        "artifact": "data/processed/socialevo_T2.pt",
                        "edge_input": "binary",
                    },
                },
            },
            "T3": {
                "name": "graph",
                "execution": deepcopy(_TASK_EXECUTION["T3"]),
                "history_length": 8,
                "benchmark_datasets": ["flights", "contact", "enron"],
                "metrics": ["change_mae", "change_rmse", "temporal_change_macro_f1"],
                "primary_metrics": [
                    "change_mae",
                    "change_rmse",
                    "temporal_change_macro_f1",
                ],
                "local_structure": {
                    "centre_set": "all nodes in current snapshot",
                    "node_set": "time-specific one-hop ego node set for source-observed centres",
                    "descriptors": [
                        "node_count",
                        "edge_count",
                        "density",
                        "connected_components",
                        "clustering_coefficient",
                        "triangle_count",
                    ],
                    "class_labels": ["decrease", "unchanged", "increase"],
                    "tolerances": "fitted on training split only",
                },
                "datasets": {
                    "flights": {
                        "artifact": "data/processed/flights_T3.pt",
                        "edge_input": "binary",
                    },
                    "contact": {
                        "artifact": "data/processed/contact_T3.pt",
                        "edge_input": "binary",
                    },
                    "enron": {
                        "artifact": "data/processed/enron_T3.pt",
                        "edge_input": "binary",
                    },
                },
            },
        },
    }


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:

    if path is None:
        return _task_protocol()
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict) or "tasks" not in protocol:
        raise ValueError(f"Invalid WorldGraph protocol: {source}")
    return protocol


def protocol_digest(protocol: dict[str, Any]) -> str:

    payload = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_task_dataset(
    task: str,
    dataset: str,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:

    protocol = load_protocol(path)
    try:
        task_spec = protocol["tasks"][task]
        raw_dataset_spec = protocol["tasks"][task]["datasets"][dataset]
    except KeyError as error:
        available = ", ".join(sorted(protocol.get("tasks", {})))
        raise ValueError(
            f"Unknown protocol entry task={task!r}, dataset={dataset!r}; "
            f"tasks={available}"
        ) from error
    dataset_spec = deepcopy(raw_dataset_spec)
    execution = deepcopy(
        task_spec.get("execution", protocol.get("shared_execution", {}))
    )


    execution.update(deepcopy(dataset_spec.pop("execution", {})))
    resolved_task_spec = deepcopy(
        {
            key: value
            for key, value in task_spec.items()
            if key not in {"datasets", "execution"}
        }
    )
    local_structure_override = dataset_spec.pop("local_structure_override", None)
    if local_structure_override is not None:
        if task != "T3":
            raise ValueError("local_structure_override is valid only for T3 datasets.")
        resolved_task_spec["local_structure"].update(deepcopy(local_structure_override))
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_digest": protocol_digest(protocol),
        "task": task,
        "task_name": task_spec["name"],
        "dataset": dataset,
        "shared_execution": execution,
        "task_spec": resolved_task_spec,
        "dataset_spec": dataset_spec,
    }


def common_cli_arguments(
    task: str,
    dataset: str,
    *,
    seed: int,
    path: str | Path | None = None,
) -> list[str]:

    entry = resolve_task_dataset(task, dataset, path=path)
    execution = entry["shared_execution"]
    dataset_spec = entry["dataset_spec"]
    arguments = [
        "--epochs", str(execution["epochs"]),
        "--patience", str(execution["patience"]),
        "--val_every", str(execution["val_every"]),
        "--seed", str(seed),
    ]
    if task == "T1":
        construction = dataset_spec["node_edit_construction"]
        arguments.extend(
            [
                "--construction_seed", str(seed),
                "--construction_scale", str(construction["scale"]),
                "--construction_rate_floor", str(construction["rate_floor"]),
                "--construction_strategy", str(construction["strategy"]),
            ]
        )
        if construction["contrastive_addition_negatives"]:
            arguments.append("--construction_contrastive_addition_negatives")
        if entry["task_spec"]["common_source_features"]["observed_degree"]:
            arguments.append("--observed_degree_feature")
    return arguments


def assert_common_runtime(
    task: str,
    dataset: str,
    *,
    seed: int,
    epochs: int,
    patience: int,
    val_every: int,
    path: str | Path | None = None,
) -> dict[str, Any]:

    entry = resolve_task_dataset(task, dataset, path=path)
    execution = entry["shared_execution"]
    actual = {"epochs": int(epochs), "patience": int(patience), "val_every": int(val_every)}
    expected = {
        "epochs": int(execution["epochs"]),
        "patience": int(execution["patience"]),
        "val_every": int(execution["val_every"]),
    }
    if actual != expected:
        raise ValueError(
            f"{entry['protocol_id']} requires {expected} for {task}/{dataset}; "
            f"received {actual}."
        )
    if int(seed) < 1:
        raise ValueError("The benchmark seed must be positive.")
    return entry


__all__ = [
    "assert_common_runtime",
    "common_cli_arguments",
    "load_protocol",
    "protocol_digest",
    "resolve_task_dataset",
]
