

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwm.training.runtime_cpu import (
    configure_cpu_environment,
    configure_torch_runtime,
    subprocess_cpu_environment,
)





_CPU_THREADS = configure_cpu_environment(overwrite=True)

from gwm.benchmark_protocol import (
    common_cli_arguments,
    resolve_task_dataset,
)




import torch

configure_torch_runtime(torch, _CPU_THREADS)




WORLDGRAPH_PYTHON = Path(sys.executable)



GRAPHSAGE_ROOT = ROOT / "third_party" / "GraphSAGE"
GRAPHGPS_ROOT = ROOT / "third_party" / "GraphGPS"
GCN_ROOT = ROOT / "third_party" / "pygcn"
GAT_ROOT = ROOT / "third_party" / "pyGAT"
SGFORMER_ROOT = ROOT / "third_party" / "SGFormer"
NODEFORMER_ROOT = ROOT / "third_party" / "NodeFormer"
TGN_ROOT = ROOT / "third_party" / "tgn"
TIDFORMER_ROOT = ROOT / "third_party" / "TIDFormer"
GWME_ROOT = ROOT / "third_party" / "GWM-E"



ARTIFACT_ROOT = ROOT / "benchmark_artifacts" / "unified_v1"




BASELINE_HISTORY_LENGTH = 8




WORLDGRAPH_SGT_TOPOLOGY_CACHE_BYTES = 384 * 1024**2






DEFAULT_WORLDGRAPH_PRETRAIN_BACKBONES = {
    ("T1", "trade"): ROOT / "checkpoints/pretraining/t1_trade.pt",
    ("T1", "genre"): ROOT / "checkpoints/pretraining/t1_genre.pt",
    ("T1", "reddit"): ROOT / "checkpoints/pretraining/t1_reddit.pt",
    ("T2", "trade"): ROOT / "checkpoints/pretraining/t2_trade.pt",
    ("T2", "un_vote"): ROOT / "checkpoints/pretraining/t2_un_vote.pt",
    ("T2", "contact"): ROOT / "checkpoints/pretraining/t2_contact.pt",
    ("T2", "socialevo"): ROOT / "checkpoints/pretraining/t2_socialevo.pt",
    ("T3", "flights"): ROOT / "checkpoints/pretraining/t3_flights.pt",
    ("T3", "contact"): ROOT / "checkpoints/pretraining/t3_contact.pt",
    ("T3", "enron"): ROOT / "checkpoints/pretraining/t3_enron.pt",
}
DEFAULT_WORLDGRAPH_PRETRAIN_BACKBONE_BLENDS = {
    ("T3", "enron"): 0.05,
}
DEFAULT_WORLDGRAPH_PRETRAIN_TASK_BLENDS = {
    ("T2", "trade"): 0.00,
    ("T3", "enron"): 0.10,
}
DEFAULT_WORLDGRAPH_PRETRAIN_COMPONENT_BLENDS = {
    ("T1", "trade"): {
        "graph_encoder": 0.20,
        "state_model": 0.15,
        "latent_predictor": 0.05,
    },
    ("T2", "trade"): {
        "graph_encoder": 0.35,
        "state_model": 0.00,
        "latent_predictor": 0.00,
    },
}
DEFAULT_WORLDGRAPH_PRETRAIN_SELECTION_PROPERTY_WEIGHTS = {
    ("T1", "trade"): 1.30,
}
DEFAULT_WORLDGRAPH_PRETRAIN_GRPO_UPDATE_INTERVALS = {
    ("T2", "trade"): 4,
}
DEFAULT_WORLDGRAPH_PRETRAIN_DIRECTION_BYPASS: set[tuple[str, str]] = set()
DEFAULT_WORLDGRAPH_PRETRAIN_DIRECTION_TRANSFER_MIX = {
    ("T3", "enron"): 0.50,
}






_WORLDGRAPH_T1_CAUSAL_DEFAULT_ARGS = [
    "--lambda_activation_ranking", "0.2",
    "--lambda_deactivation_ranking", "0.2",
    "--lambda_semantic_distance", "0.2",
    "--semantic_pos_weight_scale", "1.4",
    "--node_activity_addition_source_context",
    "--node_activity_history_context",
    "--node_activity_trajectory_features",
    "--semantic_causal_expert",
    "--node_change_history_prior",
    "--node_change_source_context",
    "--semantic_transition_gate_bias", "0.25",
    "--semantic_volatility_residual_bias", "0.5",
    "--node_removal_history_bias", "0.0",
    "--node_removal_history_bias_grid",
    "0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0",
    "--semantic_property_change_bias_grid",
    "-2.0", "-1.0", "-0.5", "0.0", "0.25", "0.5", "1.0", "2.0", "4.0",
]

MODEL_FOLDERS = {
    "worldgraph": "worldgraph",


    "worldgraph_pretrain": "worldgraph_pretrain",
}

TASK_REPORT_METRICS = {



    "T1": ("Add. F1", "Rem. F1", "Sem. F1", "Changed-NDCG@10"),


    "T2": ("Add. F1", "Rem. F1", "Change Macro-F1"),
    "T3": ("Change MAE", "Change RMSE", "Change Macro-F1"),
}






_LEGACY_WORLDGRAPH_T1_ARGS: dict[str, list[str]] = {
    "genre": [
        "--train_clip_transitions", "128",



        "--sgt_topology_cache_bytes", str(WORLDGRAPH_SGT_TOPOLOGY_CACHE_BYTES),
        "--fused_adamw",
        "--defer_dense_masking",
        "--clip_history_burn_in", "8",
        "--semantic_history_feature",
        "--property_transition_gate_decoder",






        "--controller_history_features",
        "--controller_history_window", "8",
        "--controller_addition_source_visibility",
        "--controller_addition_source_visibility_balance", "per_transition",
        "--controller_remove_observed_degree",



        "--node_activity_decoder_input", "future_concat_observed",
        "--node_activity_current_state_context",
        "--node_activity_action_context",
        "--node_activity_separate_heads",
        "--node_activity_decoder_activation", "gelu",
        "--lambda_activity", "2.0",
        "--action_observable_decoder_update",
        "--action_use_property_magnitude",
        "--property_action_mode", "expected",
        "--action_count_support_mode", "remove",
        "--controller_count_supervision_mode", "all",
        "--controller_count_decision_mode", "remove",
        "--controller_node_count_weight", "0.05",
        "--controller_property_magnitude_weight", "0.05",
        "--property_history_action_weight", "0.05",



        "--semantic_reference_mode", "historical_mean",
        "--selection_metric", "f1",
        "--selection_property_weight", "1.0",
        "--action_warmup_epochs", "10",
        "--grpo_warmup_epochs", "10",
        "--grpo_interval", "32",
        "--grpo_rollout_budget", "12",
        "--grpo_weight", "0.05",
        "--no-evaluate_latent",
    ],
    "reddit": [
        "--sgt_topology_cache_bytes", str(WORLDGRAPH_SGT_TOPOLOGY_CACHE_BYTES),





        "--train_clip_transitions", "64",
        "--clip_history_burn_in", "32",
        "--semantic_history_feature",
        "--property_transition_gate_decoder",
        "--lambda_property_transition_gate", "1.0",



        "--controller_history_features",
        "--controller_history_window", "8",
        "--controller_addition_source_visibility",
        "--controller_addition_source_visibility_balance", "per_transition",
        "--controller_remove_observed_degree",
        "--node_activity_decoder_input", "future_concat_observed",
        "--node_activity_current_state_context",
        "--node_activity_action_context",
        "--node_activity_separate_heads",
        "--node_activity_decoder_activation", "gelu",


        "--lambda_activity", "2.0",
        "--lambda_semantic_change", "1.25",
        "--lambda_semantic_distance", "0.20",
        "--action_observable_decoder_update",
        "--action_use_property_magnitude",
        "--action_count_support_mode", "remove",
        "--controller_count_supervision_mode", "all",
        "--controller_count_decision_mode", "remove",
        "--controller_node_count_weight", "0.05",
        "--controller_property_magnitude_weight", "0.05",
        "--property_history_action_weight", "0.05",





        "--semantic_reference_mode", "historical_mean",
        "--selection_metric", "f1",
        "--selection_property_weight", "1.0",
        "--action_warmup_epochs", "10",
        "--grpo_warmup_epochs", "10",
        "--grpo_interval", "32",
        "--grpo_rollout_budget", "12",
        "--grpo_weight", "0.05",
        "--no-evaluate_latent",
    ],
    "trade": [
        "--sgt_topology_cache_bytes", str(WORLDGRAPH_SGT_TOPOLOGY_CACHE_BYTES),
        "--clip_history_burn_in", "32",
        "--semantic_history_feature",
        "--controller_history_features",
        "--controller_history_window", "8",
        "--controller_addition_source_visibility",
        "--controller_addition_source_visibility_balance", "per_transition",
        "--controller_remove_observed_degree",
        "--node_activity_decoder_input", "future_concat_observed",
        "--node_activity_current_state_context",
        "--node_activity_action_context",
        "--node_activity_separate_heads",
        "--node_activity_decoder_activation", "gelu",
        "--node_activity_score_source", "future_latent",
        "--node_addition_score_source", "controller",


        "--node_removal_score_source", "controller",


        "--node_removal_history_bias", "2.7",
        "--node_removal_history_bias_grid",
        "0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "2.7", "3.0",


        "--semantic_gate_bias", "-3.0",
        "--semantic_gate_bias_grid",
        "0.0", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-3.0", "-3.5", "-4.0", "-5.0",
        "--action_count_support_mode", "add",
        "--controller_count_supervision_mode", "all",
        "--controller_count_decision_mode", "add",
        "--node_operation_count_history_window", "4",
        "--node_operation_count_source", "controller",
        "--changed_node_latent_weight", "2.0",
        "--property_history_action_weight", "0.05",
        "--sgt_isolated_zero_hop",
        "--selection_metric", "f1",
        "--selection_property_weight", "1.0",
        "--action_warmup_epochs", "10",
        "--grpo_warmup_epochs", "10",
        "--grpo_interval", "32",
        "--grpo_rollout_budget", "12",
        "--grpo_weight", "0.01",
    ],
}



DEFAULT_WORLDGRAPH_HISTORY_WINDOW = 8




_LEGACY_WORLDGRAPH_T2_STEPS = {
    "trade": 0,
    "un_vote": 0,
    "contact": 0,
    "socialevo": 0,
}

_LEGACY_WORLDGRAPH_T2_ARGS: dict[str, list[str]] = {
    "trade": [
        "--rollout_budget", "16", "--world_intervention_rollouts", "8",
        "--world_warmup_epochs", "5", "--grpo_update_interval", "1",
        "--validation_transitions", "64", "--validation_history_burn_in", "8",
        "--lambda_synthetic_intervention", "0.0", "--lambda_forecast", "1.0",
        "--forecast_latent_weight", "0.1",
        "--action_conditioning_warmup_epochs", "5",
        "--soft_edge_action_conditioning", "--confidence_gated_soft_action",
        "--soft_edge_action_topk", "0", "--soft_edge_action_mode", "multilabel",
        "--separate_action_query", "--world_action_lr_scale", "0.1",
        "--controller_supervised_weight", "1.0", "--controller_topology_count_weight", "0.2",
        "--forecast_change_weight", "0.0", "--forecast_state_weight", "2.0",
        "--topology_addition_neg_ratio", "1",
        "--topology_addition_negative_strategy", "uniform", "--topology_pair_state",
    ],
    "un_vote": [
        "--rollout_budget", "16", "--world_intervention_rollouts", "8",
        "--world_warmup_epochs", "5", "--grpo_update_interval", "1",
        "--validation_transitions", "64", "--validation_history_burn_in", "8",
        "--lambda_synthetic_intervention", "0.0", "--lambda_forecast", "1.0",
        "--forecast_latent_weight", "0.1",
        "--action_conditioning_warmup_epochs", "5",
        "--soft_edge_action_conditioning", "--confidence_gated_soft_action",
        "--soft_edge_action_topk", "0", "--soft_edge_action_mode", "multilabel",
        "--separate_action_query", "--world_action_lr_scale", "0.1",
        "--controller_supervised_weight", "1.0", "--controller_topology_count_weight", "0.2",
        "--forecast_change_weight", "0.0", "--forecast_state_weight", "2.0",
        "--topology_addition_neg_ratio", "5",
        "--topology_addition_negative_strategy", "mixed", "--topology_pair_state",
    ],
    "contact": [
        "--rollout_budget", "16", "--world_intervention_rollouts", "8",
        "--world_warmup_epochs", "5", "--grpo_update_interval", "1",
        "--validation_transitions", "64", "--validation_history_burn_in", "8",
        "--lambda_synthetic_intervention", "0.0", "--lambda_forecast", "1.0",
        "--forecast_latent_weight", "0.1",
        "--action_conditioning_warmup_epochs", "5",
        "--soft_edge_action_conditioning", "--confidence_gated_soft_action",
        "--soft_edge_action_topk", "0", "--soft_edge_action_mode", "multilabel",
        "--separate_action_query", "--world_action_lr_scale", "0.1",
        "--controller_supervised_weight", "1.0", "--controller_topology_count_weight", "0.2",
        "--forecast_change_weight", "0.0", "--forecast_state_weight", "2.0",
        "--topology_addition_neg_ratio", "5",
        "--topology_addition_negative_strategy", "mixed", "--topology_pair_state",
    ],
}




_LEGACY_WORLDGRAPH_T2_ARGS["socialevo"] = list(
    _LEGACY_WORLDGRAPH_T2_ARGS["contact"]
)


def _python() -> str:
    return str(WORLDGRAPH_PYTHON if WORLDGRAPH_PYTHON.is_file() else Path(sys.executable))


def _task_folder(task: str) -> str:
    return task


def _paths(task: str, model: str, dataset: str, seed: int, timestamp: str) -> dict[str, Path]:
    root = ROOT / "results" / _task_folder(task) / MODEL_FOLDERS[model] / dataset
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = ARTIFACT_ROOT / _task_folder(task) / MODEL_FOLDERS[model] / dataset
    artifact_root.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_s{seed}_{timestamp}"
    return {
        "root": root,
        "checkpoint": artifact_root / f"{stem}.pt",
        "results": artifact_root / f"{stem}.json",
        "log": root / f"{stem}.log",
        "manifest": artifact_root / f"{stem}.manifest.json",
        "artifacts": artifact_root,
    }


def _repo_relative_command(command: list[str]) -> list[str]:
    root = str(ROOT)
    prefix = root + os.sep
    result: list[str] = []
    for value in command:
        if value.startswith(prefix):
            result.append(str(Path(value).relative_to(ROOT)))
        else:
            result.append(value)
    return result


def _worldgraph_command(
    *,
    task: str,
    dataset: str,
    seed: int,
    device: str,
    paths: dict[str, Path],
    protocol: Path,
    entry: dict[str, Any],
    history_window: int | None = None,
    sgt_num_hops: int | None = None,
    sgt_num_walks: int | None = None,
    grpo_rollout_budget: int | None = None,
    group_beta: float | None = None,
    degree_power: float | None = None,
    volatility_power: float | None = None,
    no_train_rotation: bool = False,
    train_clip_transitions: int | None = None,
    semantic_reference_mode: str | None = None,
    controller_history_window: int | None = None,
    pretrained_backbone: Path | None = None,
    pretrained_backbone_blend: float | None = None,
    pretrained_graph_encoder_blend: float | None = None,
    pretrained_state_model_blend: float | None = None,
    pretrained_latent_predictor_blend: float | None = None,
    pretrained_controller_blend: float | None = None,
    pretrained_task_blend: float | None = None,
    pretrained_magnitude_head_blend: float | None = None,
    pretrained_direction_head_blend: float | None = None,
    pretrained_transfer_gate: bool = False,
    pretrained_transfer_gate_direction_bypass: bool = False,
    pretrained_direction_transfer_mix: float | None = None,
    lr: float | None = None,
    controller_lr: float | None = None,
    direction_loss_weight: float | None = None,
    direction_class_balance_power: float | None = None,
    selection_change_f1_weight: float | None = None,
    selection_change_rmse_weight: float | None = None,
    batch_grpo_rollout_forward: bool = True,
    batch_topology_grpo_rollouts: bool | None = None,
    batch_topology_grpo_rewards: bool | None = None,
    topology_composite_embedding_reward: bool = True,
    grpo_update_interval: int | None = None,
    t3_grpo_interval: int | None = None,
    t1_composite_structure_reward: bool = True,
    t3_magnitude_reward_weight: float | None = None,
    cache_deterministic_topology_candidates: bool = False,
    cache_device_topology_candidates: bool = True,
    inprocess_validation: bool = False,
    persistent_validation_worker: bool = False,
    cache_action_candidates: bool = False,
    batch_edge_grpo_rollouts: bool = False,
    tensorized_grpo_rewards: bool = False,
    cache_released_history: bool | None = None,
    reuse_controller_policy_states: bool = False,
    cache_static_action_candidates: bool = False,
    vectorized_controller_forward: bool = False,
    cache_validation_preparation: bool = False,
    lambda_activation_ranking: float | None = None,
    lambda_deactivation_ranking: float | None = None,
    lambda_deactivation: float | None = None,
    deactivation_focal_gamma: float | None = None,
    lambda_semantic_distance: float | None = None,
    semantic_pos_weight_scale: float | None = None,
    node_activity_addition_source_context: bool | None = None,
    node_activity_full_observed_context: bool | None = None,
    node_activity_source_visibility_context: bool | None = None,
    node_activity_history_context: bool | None = None,
    node_activity_trajectory_features: bool | None = None,
    controller_count_decision_mode: str | None = None,
    node_operation_count_history_window: int | None = None,
    node_operation_count_source: str | None = None,
    node_operation_count_decoder: bool | None = None,
    lambda_node_operation_count: float | None = None,
    decision_calibration_tail_fraction: float | None = None,
    selection_metric: str | None = None,
    selection_property_weight: float | None = None,
    node_activity_history_gate: bool | None = None,
    node_activity_history_prior: bool | None = None,
    node_activity_history_prior_remove_only: bool | None = None,
    semantic_causal_expert: bool | None = None,
    node_change_source_context: bool | None = None,
    node_change_history_prior: bool | None = None,
    dynamic_change_hard_fraction: float | None = None,
    dynamic_change_focal_gamma: float | None = None,
    semantic_hard_pair_weight: float | None = None,
    semantic_hard_pair_margin: float | None = None,
    node_activity_score_source: str | None = None,
    node_addition_score_source: str | None = None,
    node_removal_score_source: str | None = None,
    node_removal_history_bias: float | None = None,
    node_removal_group_calibration_mode: str | None = None,
    node_removal_group_calibration_weight: float | None = None,
    semantic_history_bias: float | None = None,
    semantic_property_change_bias: float | None = None,
    semantic_transition_gate_bias: float | None = None,
    semantic_volatility_residual_bias: float | None = None,
    activity_trajectory_supervision_bias: float | None = None,
    removal_history_supervision_bias: float | None = None,
    node_addition_source_visibility_threshold: bool | None = None,
    node_removal_history_bias_grid: list[float] | None = None,
    node_removal_group_calibration_weight_grid: list[float] | None = None,
    semantic_property_change_bias_grid: list[float] | None = None,
    semantic_gate_bias_grid: list[float] | None = None,
) -> tuple[list[str], Path]:
    common = common_cli_arguments(task, dataset, seed=seed, path=protocol)
    if task == "T1":




        if dataset.lower() == "genre":
            no_train_rotation = True
        effective_history_window = (
            history_window
            if history_window is not None
            else DEFAULT_WORLDGRAPH_HISTORY_WINDOW
        )



        if (pretrained_backbone is not None or dataset.lower() == "trade") and history_window is None:
            effective_history_window = 12
        command = [
                _python(),
                str(ROOT / "gwm" / "training" / "train_node_level_action_rl.py"),
                "--dataset", dataset,
                "--processed", str((ROOT / entry["dataset_spec"]["artifact"]).resolve()),
                "--device", device,
                "--semi_synthetic_node_edits",



                "--cache_transitions",
                "--semantic_history_cache",
                *common,
                *_WORLDGRAPH_T1_CAUSAL_DEFAULT_ARGS,
                *_LEGACY_WORLDGRAPH_T1_ARGS[dataset],
                "--checkpoint", str(paths["checkpoint"]),
                "--results", str(paths["results"]),
            ]
        if pretrained_backbone is not None or dataset.lower() == "trade":
            command.extend(
                [



                    "--latent_dim", "96",
                    "--hidden_dim", "96",
                    "--action_dim", "48",
                ]
            )
        if dataset.lower() == "trade":
            command.extend(
                [
                    "--lambda_latent", "1.5",
                    "--latent_normalization", "none",
                    "--weight_decay", "5e-05",
                ]
            )
        if effective_history_window is not None:
            command.extend(["--history_window", str(effective_history_window)])
        if sgt_num_hops is not None:
            command.extend(["--sgt_num_hops", str(sgt_num_hops)])
        if sgt_num_walks is not None:
            command.extend(["--sgt_num_walks", str(sgt_num_walks)])
        if grpo_rollout_budget is not None:
            budget_index = max(
                index
                for index, value in enumerate(command)
                if value == "--grpo_rollout_budget"
            )
            command[budget_index + 1] = str(grpo_rollout_budget)
        if no_train_rotation:




            for clip_value in ("128", "64"):
                marker = ["--train_clip_transitions", clip_value]
                if all(item in command for item in marker):
                    index = next(
                        i for i in range(len(command) - 1)
                        if command[i:i + 2] == marker
                    )
                    del command[index:index + 2]
                    break
        if semantic_reference_mode is not None:
            command.extend(["--semantic_reference_mode", semantic_reference_mode])
        if controller_history_window is not None:
            command.extend(["--controller_history_window", str(controller_history_window)])
        if batch_grpo_rollout_forward:
            command.append("--batch_grpo_rollout_forward")
        command.append(
            "--t1_composite_structure_reward"
            if t1_composite_structure_reward
            else "--no-t1_composite_structure_reward"
        )
        if cache_static_action_candidates:
            command.append("--cache_static_action_candidates")
        if vectorized_controller_forward:
            command.append("--vectorized_controller_forward")
        if cache_validation_preparation:
            command.append("--cache_validation_preparation")



        for option, value in (
            ("--lambda_activation_ranking", lambda_activation_ranking),
            ("--lambda_deactivation_ranking", lambda_deactivation_ranking),
            ("--lambda_deactivation", lambda_deactivation),
            ("--deactivation_focal_gamma", deactivation_focal_gamma),
            ("--lambda_semantic_distance", lambda_semantic_distance),
            ("--semantic_pos_weight_scale", semantic_pos_weight_scale),
            ("--dynamic_change_hard_fraction", dynamic_change_hard_fraction),
            ("--dynamic_change_focal_gamma", dynamic_change_focal_gamma),
            ("--semantic_hard_pair_weight", semantic_hard_pair_weight),
            ("--semantic_hard_pair_margin", semantic_hard_pair_margin),
            ("--lambda_node_operation_count", lambda_node_operation_count),
            ("--selection_metric", selection_metric),
            ("--selection_property_weight", selection_property_weight),
        ):
            if value is not None:
                command.extend([option, str(value)])
        if node_activity_addition_source_context is not None:
            command.append(
                "--node_activity_addition_source_context"
                if node_activity_addition_source_context
                else "--no-node_activity_addition_source_context"
            )
        if node_activity_full_observed_context is not None:
            command.append(
                "--node_activity_full_observed_context"
                if node_activity_full_observed_context
                else "--no-node_activity_full_observed_context"
            )
        if node_activity_source_visibility_context is not None:
            command.append(
                "--node_activity_source_visibility_context"
                if node_activity_source_visibility_context
                else "--no-node_activity_source_visibility_context"
            )
        if node_addition_source_visibility_threshold is not None:
            command.append(
                "--node_addition_source_visibility_threshold"
                if node_addition_source_visibility_threshold
                else "--no-node_addition_source_visibility_threshold"
            )
        if node_operation_count_decoder is not None:
            command.append(
                "--node_operation_count_decoder"
                if node_operation_count_decoder
                else "--no-node_operation_count_decoder"
            )
        for option, value in (
            ("--node_activity_score_source", node_activity_score_source),
            ("--node_addition_score_source", node_addition_score_source),
            ("--node_removal_score_source", node_removal_score_source),
            ("--node_removal_history_bias", node_removal_history_bias),
            (
                "--node_removal_group_calibration_mode",
                node_removal_group_calibration_mode,
            ),
            (
                "--node_removal_group_calibration_weight",
                node_removal_group_calibration_weight,
            ),
            ("--semantic_history_bias", semantic_history_bias),
            ("--semantic_property_change_bias", semantic_property_change_bias),
            ("--semantic_transition_gate_bias", semantic_transition_gate_bias),
            ("--semantic_volatility_residual_bias", semantic_volatility_residual_bias),
            ("--activity_trajectory_supervision_bias", activity_trajectory_supervision_bias),
            ("--removal_history_supervision_bias", removal_history_supervision_bias),
        ):
            if value is not None:
                command.extend([option, str(value)])
        if node_removal_history_bias_grid is not None:
            command.append("--node_removal_history_bias_grid")
            command.extend(str(value) for value in node_removal_history_bias_grid)
        if node_removal_group_calibration_weight_grid is not None:
            command.append("--node_removal_group_calibration_weight_grid")
            command.extend(
                str(value)
                for value in node_removal_group_calibration_weight_grid
            )
        if semantic_property_change_bias_grid is not None:
            command.append("--semantic_property_change_bias_grid")
            command.extend(
                str(value) for value in semantic_property_change_bias_grid
            )
        if semantic_gate_bias_grid is not None:
            command.append("--semantic_gate_bias_grid")
            command.extend(str(value) for value in semantic_gate_bias_grid)
        if node_activity_history_context is not None:
            command.append(
                "--node_activity_history_context"
                if node_activity_history_context
                else "--no-node_activity_history_context"
            )
        if node_activity_trajectory_features is not None:
            command.append(
                "--node_activity_trajectory_features"
                if node_activity_trajectory_features
                else "--no-node_activity_trajectory_features"
            )
        if controller_count_decision_mode is not None:
            command.extend(
                ["--controller_count_decision_mode", controller_count_decision_mode]
            )
        if node_operation_count_history_window is not None:
            command.extend(
                [
                    "--node_operation_count_history_window",
                    str(node_operation_count_history_window),
                ]
            )
        if node_operation_count_source is not None:
            command.extend(
                ["--node_operation_count_source", node_operation_count_source]
            )
        if decision_calibration_tail_fraction is not None:
            command.extend(
                [
                    "--decision_calibration_tail_fraction",
                    str(decision_calibration_tail_fraction),
                ]
            )
        if semantic_causal_expert is not None:
            command.append(
                "--semantic_causal_expert"
                if semantic_causal_expert
                else "--no-semantic_causal_expert"
            )
        if node_change_history_prior is not None:
            command.append(
                "--node_change_history_prior"
                if node_change_history_prior
                else "--no-node_change_history_prior"
            )
        if node_change_source_context is not None:
            command.append(
                "--node_change_source_context"
                if node_change_source_context
                else "--no-node_change_source_context"
            )
        if node_activity_history_gate is not None:
            command.append(
                "--node_activity_history_gate"
                if node_activity_history_gate
                else "--no-node_activity_history_gate"
            )
        if node_activity_history_prior is not None:
            command.append(
                "--node_activity_history_prior"
                if node_activity_history_prior
                else "--no-node_activity_history_prior"
            )
        if node_activity_history_prior_remove_only is not None:
            command.append(
                "--node_activity_history_prior_remove_only"
                if node_activity_history_prior_remove_only
                else "--no-node_activity_history_prior_remove_only"
            )
        if pretrained_backbone is not None:
            command.extend(
                [
                    "--pretrained_backbone", str(pretrained_backbone),
                    "--pretrained_transfer_mode", "structural",
                    "--pretrained_transfer_state_model",
                    "--pretrained_backbone_blend",




                    str(0.20 if pretrained_backbone_blend is None else pretrained_backbone_blend),
                ]
            )
            for option, value in (
                ("--pretrained_graph_encoder_blend", pretrained_graph_encoder_blend),
                ("--pretrained_state_model_blend", pretrained_state_model_blend),
                ("--pretrained_latent_predictor_blend", pretrained_latent_predictor_blend),
            ):
                if value is not None:
                    command.extend([option, str(value)])
            if pretrained_controller_blend is not None:
                command.extend(["--pretrained_controller_blend", str(pretrained_controller_blend)])
            command.extend([
                "--pretrained_task_blend",
                str(0.5 if pretrained_task_blend is None else pretrained_task_blend),
            ])
        return command, ROOT
    if task == "T2":


        effective_steps = 0 if no_train_rotation else _LEGACY_WORLDGRAPH_T2_STEPS[dataset]
        command = [
                _python(),
                str(ROOT / "gwm" / "training" / "train_action_rl.py"),
                "--task", "topology",
                "--dataset", dataset,
                "--device", device,
                "--steps", str(effective_steps),
                "--clip_history_burn_in", "8",
                "--history_window", str(
                    DEFAULT_WORLDGRAPH_HISTORY_WINDOW
                    if history_window is None
                    else history_window
                ),
                "--sgt_topology_cache_bytes", str(WORLDGRAPH_SGT_TOPOLOGY_CACHE_BYTES),
                "--edge_input", str(entry["dataset_spec"]["edge_input"]),
                *common,
                *_LEGACY_WORLDGRAPH_T2_ARGS[dataset],
                "--checkpoint", str(paths["checkpoint"]),
                "--results", str(paths["results"]),
            ]
        if sgt_num_hops is not None:
            command.extend(["--sgt_num_hops", str(sgt_num_hops)])
        if sgt_num_walks is not None:
            command.extend(["--sgt_num_walks", str(sgt_num_walks)])
        if group_beta is not None:
            command.extend(["--group_beta", str(group_beta)])
        if grpo_rollout_budget is not None:
            budget_index = max(
                index
                for index, value in enumerate(command)
                if value == "--rollout_budget"
            )
            command[budget_index + 1] = str(grpo_rollout_budget)
        command.append(
            "--batch_topology_grpo_rollouts"
            if batch_topology_grpo_rollouts is not False
            else "--no-batch_topology_grpo_rollouts"
        )
        command.append(
            "--batch_topology_grpo_rewards"
            if batch_topology_grpo_rewards is not False
            else "--no-batch_topology_grpo_rewards"
        )
        if not topology_composite_embedding_reward:
            command.append("--no-topology_composite_embedding_reward")
        if grpo_update_interval is not None:
            interval_index = max(
                index
                for index, value in enumerate(command)
                if value == "--grpo_update_interval"
            )
            command[interval_index + 1] = str(grpo_update_interval)
        if cache_deterministic_topology_candidates:
            command.append("--cache_deterministic_topology_candidates")
        if cache_device_topology_candidates:
            command.append("--cache_device_topology_candidates")
        if inprocess_validation:
            command.append("--inprocess_validation")
        if persistent_validation_worker:
            command.append("--persistent_validation_worker")
        if pretrained_backbone is not None:
            command.extend(
                [
                    "--pretrained_backbone", str(pretrained_backbone),
                    "--pretrained_transfer_mode", "structural",
                    "--pretrained_transfer_state_model",
                    "--pretrained_backbone_blend",
                    str(0.25 if pretrained_backbone_blend is None else pretrained_backbone_blend),
                ]
            )
            for option, value in (
                ("--pretrained_graph_encoder_blend", pretrained_graph_encoder_blend),
                ("--pretrained_state_model_blend", pretrained_state_model_blend),
                ("--pretrained_latent_predictor_blend", pretrained_latent_predictor_blend),
            ):
                if value is not None:
                    command.extend([option, str(value)])
            command.extend([
                "--pretrained_task_blend",
                str(0.5 if pretrained_task_blend is None else pretrained_task_blend),
            ])
            if pretrained_transfer_gate:
                command.append("--pretrained_transfer_gate")
        return command, ROOT
    if task == "T3":





        effective_direction_loss_weight = (
            0.25 if direction_loss_weight is None else direction_loss_weight
        )
        effective_direction_class_balance_power = (
            0.25
            if direction_class_balance_power is None
            else direction_class_balance_power
        )
        command = [
                _python(),
                str(ROOT / "gwm" / "training" / "train_worldgraph_graph.py"),
                "--dataset", dataset,
                "--processed", str((ROOT / entry["dataset_spec"]["artifact"]).resolve()),
                "--target_cache", str(
                    (paths["artifacts"] / "t3_targets_v2.pt").resolve()
                ),
                "--device", device,
                "--epochs", str(entry["shared_execution"]["epochs"]),
                "--patience", str(entry["shared_execution"]["patience"]),
                "--val_every", str(entry["shared_execution"]["val_every"]),
                "--history_window", str(
                    DEFAULT_WORLDGRAPH_HISTORY_WINDOW
                    if history_window is None
                    else history_window
                ),
                "--sgt_topology_cache_bytes", str(WORLDGRAPH_SGT_TOPOLOGY_CACHE_BYTES),
                "--grpo_rollout_budget", "12",




                "--causal_structure_history_context",
                "--group_aware_decoder_context",
                "--direction_group_aware_context_only",
                "--decoupled_direction_head",
                "--direction_loss_weight", str(effective_direction_loss_weight),
                "--direction_class_balance_power", str(
                    effective_direction_class_balance_power
                ),
                "--direction_projection_margin", "0.02",
                "--direction_projection_confidence", "0.70",
                "--selection_change_f1_weight", str(
                    2.0
                    if selection_change_f1_weight is None
                    else selection_change_f1_weight
                ),
                "--selection_change_rmse_weight", str(
                    0.25
                    if selection_change_rmse_weight is None
                    else selection_change_rmse_weight
                ),
                "--checkpoint", str(paths["checkpoint"]),
                "--results", str(paths["results"]),
                "--seed", str(seed),
            ]
        if sgt_num_hops is not None:
            command.extend(["--sgt_num_hops", str(sgt_num_hops)])
        if sgt_num_walks is not None:
            command.extend(["--sgt_num_walks", str(sgt_num_walks)])
        if grpo_rollout_budget is not None:
            budget_index = max(
                index
                for index, value in enumerate(command)
                if value == "--grpo_rollout_budget"
            )
            command[budget_index + 1] = str(grpo_rollout_budget)
        if group_beta is not None:
            command.extend(["--group_beta", str(group_beta)])
        if degree_power is not None:
            command.extend(["--degree_power", str(degree_power)])
        if volatility_power is not None:
            command.extend(["--volatility_power", str(volatility_power)])
        if lr is not None:
            command.extend(["--lr", str(lr)])
        if controller_lr is not None:
            command.extend(["--controller_lr", str(controller_lr)])
        if t3_grpo_interval is not None:
            command.extend(["--grpo_interval", str(t3_grpo_interval)])
        if t3_magnitude_reward_weight is not None:
            command.extend(
                ["--magnitude_reward_weight", str(t3_magnitude_reward_weight)]
            )
        if train_clip_transitions is not None:
            command.extend(["--train_clip_transitions", str(train_clip_transitions)])
        command.append(
            "--cache_action_candidates"
            if cache_action_candidates
            else "--no-cache_action_candidates"
        )
        command.append(
            "--batch_edge_grpo_rollouts"
            if batch_edge_grpo_rollouts
            else "--no-batch_edge_grpo_rollouts"
        )
        if tensorized_grpo_rewards:
            command.append("--tensorized_grpo_rewards")
        command.append(
            "--cache_released_history"
            if cache_released_history is not False
            else "--no-cache_released_history"
        )
        if reuse_controller_policy_states:
            command.append("--reuse_controller_policy_states")
        if pretrained_backbone is not None:
            command.extend(
                [
                    "--pretrained_backbone", str(pretrained_backbone),
                    "--pretrained_transfer_mode", "structural",
                    "--pretrained_transfer_state_model",
                    "--pretrained_backbone_blend",
                    str(0.25 if pretrained_backbone_blend is None else pretrained_backbone_blend),
                ]
            )
            if pretrained_transfer_gate:
                command.append("--pretrained_transfer_gate")
            if pretrained_transfer_gate_direction_bypass:
                command.append("--pretrained_transfer_gate_direction_bypass")
            if pretrained_direction_transfer_mix is not None:
                command.extend([
                    "--pretrained_direction_transfer_mix",
                    str(pretrained_direction_transfer_mix),
                ])
            command.extend([
                "--pretrained_task_blend",
                str(0.5 if pretrained_task_blend is None else pretrained_task_blend),
            ])
            if pretrained_magnitude_head_blend is not None:
                command.extend([
                    "--pretrained_magnitude_head_blend",
                    str(pretrained_magnitude_head_blend),
                ])
            if pretrained_direction_head_blend is not None:
                command.extend([
                    "--pretrained_direction_head_blend",
                    str(pretrained_direction_head_blend),
                ])
        return command, ROOT
    raise NotImplementedError(
        "WorldGraph's legacy node-state trainer is not a T3 local-subgraph "
        "predictor. A dedicated T3 adapter must be completed before this "
        "launcher permits a unified WorldGraph T3 run."
    )


def _worldgraph_passive_command(
    *,
    task: str,
    dataset: str,
    seed: int,
    device: str,
    paths: dict[str, Path],
    protocol: Path,
    entry: dict[str, Any],
) -> tuple[list[str], Path]:
    if task != "T1":
        raise NotImplementedError(
            "WorldGraph-passive is currently available only for the unified "
            "T1 node-level protocol."
        )
    common = common_cli_arguments(task, dataset, seed=seed, path=protocol)
    return (
        [
            _python(),
            str(ROOT / "scripts" / "train_node_level_transition.py"),
            "--dataset", dataset,
            "--device", device,
            "--semi_synthetic_node_edits",
            *common,
            "--checkpoint", str(paths["checkpoint"]),
            "--results", str(paths["results"]),
        ],
        ROOT,
    )


def _gwme_command(
    *,
    task: str,
    dataset: str,
    seed: int,
    device: str,
    paths: dict[str, Path],
    protocol: Path,
    entry: dict[str, Any],
) -> tuple[list[str], Path]:
    common = common_cli_arguments(task, dataset, seed=seed, path=protocol)
    if task == "T1":



        gwme_common: list[str] = []
        iterator = iter(common)
        for argument in iterator:
            if argument == "--construction_seed":
                next(iterator)
                continue
            gwme_common.append(argument)
        return (
            [



                _python(), "-m", "temporal_extension.train_llm_node_action",
                "--dataset", dataset,
                "--processed", str((ROOT / entry["dataset_spec"]["artifact"]).resolve()),
                "--llm_checkpoint", str(
                    (ROOT / "third_party" / "GWM"
                     / "checkpoints/NousResearch-Meta-Llama-3-8B-Instruct").resolve()
                ),
                "--device", device,
                "--state_dim", "64",
                "--hops", "2",
                "--history_length", str(BASELINE_HISTORY_LENGTH),
                *gwme_common,
                "--selection_metric", "f1",
                "--adapter_checkpoint", str(paths["checkpoint"]),
                "--results", str(paths["results"]),
            ],
            GWME_ROOT,
        )
    if task == "T2":




        return (
            [
                _python(), "-m", "temporal_extension.train_llm_topology_action",
                "--dataset", dataset,
                "--processed", str((ROOT / entry["dataset_spec"]["artifact"]).resolve()),
                "--llm_checkpoint", str(
                    (ROOT / "third_party" / "GWM"
                     / "checkpoints/NousResearch-Meta-Llama-3-8B-Instruct").resolve()
                ),
                "--device", device,
                "--epochs", str(entry["shared_execution"]["epochs"]),
                "--patience", str(entry["shared_execution"]["patience"]),
                "--val_every", str(entry["shared_execution"]["val_every"]),
                "--state_dim", "64",
                "--hops", "2",
                "--history_length", str(BASELINE_HISTORY_LENGTH),
                "--edge_input", str(entry["dataset_spec"]["edge_input"]),
                "--seed", str(seed),
                "--count_weight", "0.2",
                "--adapter_checkpoint", str(paths["checkpoint"]),
                "--results", str(paths["results"]),
            ],
            GWME_ROOT,
        )
    if task == "T3":



        return (
            [
                _python(), "-m", "temporal_extension.train_llm_t3_structure",
                "--dataset", dataset,
                "--processed", str((ROOT / entry["dataset_spec"]["artifact"]).resolve()),
                "--target_cache", str((paths["artifacts"] / "t3_targets_v2.pt").resolve()),
                "--llm_checkpoint", str(
                    (ROOT / "third_party" / "GWM"
                     / "checkpoints/NousResearch-Meta-Llama-3-8B-Instruct").resolve()
                ),
                "--device", device,
                "--epochs", str(entry["shared_execution"]["epochs"]),
                "--patience", str(entry["shared_execution"]["patience"]),
                "--val_every", str(entry["shared_execution"]["val_every"]),
                "--state_dim", "64",
                "--hops", "2",
                "--history_length", str(BASELINE_HISTORY_LENGTH),
                "--adapter_checkpoint", str(paths["checkpoint"]),
                "--results", str(paths["results"]),
                "--seed", str(seed),
            ],
            GWME_ROOT,
        )
    raise NotImplementedError(f"Unsupported GWM-E task: {task}")


def _graphsage_command(
    *,
    task: str,
    dataset: str,
    seed: int,
    device: str,
    paths: dict[str, Path],
    protocol: Path,
    entry: dict[str, Any],
    history_length: int | None = None,
) -> tuple[list[str], Path]:
    task_name = entry["task_name"]
    common = common_cli_arguments(task, dataset, seed=seed, path=protocol)


    filtered: list[str] = []
    iterator = iter(common)
    for argument in iterator:
        if argument == "--construction_seed":
            next(iterator)
            continue
        filtered.append(argument)
    command = [
            _python(),
            str(GRAPHSAGE_ROOT / "scripts" / "train_worldgraph.py"),
            "--task", task_name,
            "--dataset", dataset,
            "--benchmark_root", str(ROOT),
            "--device", device,
            "--num_seeds", "1",
            "--history_length", str(
                BASELINE_HISTORY_LENGTH if history_length is None else history_length
            ),
            "--output_root", str(paths["artifacts"] / "graphsage"),
            "--edge_input", str(entry["dataset_spec"]["edge_input"]),
            *filtered,
        ]
    return command, GRAPHSAGE_ROOT


def _graphgps_command(
    *,
    task: str,
    dataset: str,
    seed: int,
    device: str,
    paths: dict[str, Path],
    protocol: Path,
    entry: dict[str, Any],
    history_length: int | None = None,
) -> tuple[list[str], Path]:

    task_name = entry["task_name"]
    common = common_cli_arguments(task, dataset, seed=seed, path=protocol)


    filtered: list[str] = []
    iterator = iter(common)
    for argument in iterator:
        if argument == "--construction_seed":
            next(iterator)
            continue
        filtered.append(argument)
    return (
        [
            _python(),
            str(GRAPHGPS_ROOT / "scripts" / "train_worldgraph.py"),
            "--task", task_name,
            "--dataset", dataset,
            "--benchmark_root", str(ROOT),
            "--device", device,
            "--num_seeds", "1",
            "--history_length", str(
                BASELINE_HISTORY_LENGTH if history_length is None else history_length
            ),
            "--output_root", str(paths["artifacts"] / "graphgps"),
            "--edge_input", str(entry["dataset_spec"]["edge_input"]),
            *filtered,
        ],
        GRAPHGPS_ROOT,
    )


def _external_encoder_command(
    *,
    model: str,
    task: str,
    dataset: str,
    seed: int,
    device: str,
    paths: dict[str, Path],
    protocol: Path,
    entry: dict[str, Any],
    history_length: int | None = None,
) -> tuple[list[str], Path]:

    roots = {
        "gcn": GCN_ROOT,
        "gat": GAT_ROOT,
        "sgformer": SGFORMER_ROOT,
        "nodeformer": NODEFORMER_ROOT,
        "tgn": TGN_ROOT,
        "tidformer": TIDFORMER_ROOT,
    }
    baseline_root = roots[model]
    task_name = entry["task_name"]
    common = common_cli_arguments(task, dataset, seed=seed, path=protocol)
    filtered: list[str] = []
    iterator = iter(common)
    for argument in iterator:
        if argument == "--construction_seed":
            next(iterator)
            continue
        filtered.append(argument)
    return (
        [
            _python(),
            str(baseline_root / "scripts" / "train_worldgraph.py"),
            "--task", task_name,
            "--dataset", dataset,
            "--benchmark_root", str(ROOT),
            "--device", device,
            "--num_seeds", "1",
            "--history_length", str(
                BASELINE_HISTORY_LENGTH if history_length is None else history_length
            ),
            "--output_root", str(paths["artifacts"] / model),
            "--edge_input", str(entry["dataset_spec"]["edge_input"]),
            *filtered,
        ],
        baseline_root,
    )


def _gcn_command(**kwargs: Any) -> tuple[list[str], Path]:
    return _external_encoder_command(model="gcn", **kwargs)


def _gat_command(**kwargs: Any) -> tuple[list[str], Path]:
    return _external_encoder_command(model="gat", **kwargs)


def _sgformer_command(**kwargs: Any) -> tuple[list[str], Path]:
    return _external_encoder_command(model="sgformer", **kwargs)


def _nodeformer_command(**kwargs: Any) -> tuple[list[str], Path]:
    return _external_encoder_command(model="nodeformer", **kwargs)


def _tgn_command(**kwargs: Any) -> tuple[list[str], Path]:
    return _external_encoder_command(model="tgn", **kwargs)


def _tidformer_command(**kwargs: Any) -> tuple[list[str], Path]:
    return _external_encoder_command(model="tidformer", **kwargs)


def _command_for(
    model: str,
    **kwargs: Any,
) -> tuple[list[str], Path]:
    builders = {
        "worldgraph": _worldgraph_command,
        "worldgraph_pretrain": _worldgraph_command,
        "worldgraph-passive": _worldgraph_passive_command,
        "gwm-e": _gwme_command,
        "graphsage": _graphsage_command,
        "graphgps": _graphgps_command,
        "gcn": _gcn_command,
        "gat": _gat_command,
        "sgformer": _sgformer_command,
        "nodeformer": _nodeformer_command,
        "tgn": _tgn_command,
        "tidformer": _tidformer_command,
    }
    return builders[model](**kwargs)


def _write_manifest(
    path: Path,
    *,
    command: list[str],
    cwd: Path,
    entry: dict[str, Any],
    model: str,
    seed: int,
    timestamp: str,
) -> None:
    payload = {
        "protocol_id": entry["protocol_id"],
        "protocol_digest": entry["protocol_digest"],
        "task": entry["task"],
        "dataset": entry["dataset"],
        "model": model,
        "seed": int(seed),
        "timestamp": timestamp,
        "command": command,
        "cwd": str(cwd),
        "shared_execution": entry["shared_execution"],
        "task_spec": entry["task_spec"],
        "dataset_spec": entry["dataset_spec"],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_test_metrics_line(line: str) -> dict[str, float] | None:

    if not line.startswith("[TEST-MEAN/") or "] " not in line:
        return None
    _header, payload = line.split("] ", 1)
    metrics: dict[str, float] = {}
    for field in payload.split("|"):
        if "=" not in field:
            continue
        name, value = field.rsplit("=", 1)
        try:
            metrics[name.strip()] = float(value.strip().split("±", 1)[0])
        except ValueError:
            continue
    return metrics or None


def _format_metrics(metrics: dict[str, float], names: list[str] | tuple[str, ...]) -> str:
    return " | ".join(f"{name}={metrics[name]:.4f}" for name in names)


def _emit(message: str, log: TextIO) -> None:
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()


def _command_value(command: list[str], option: str) -> str | None:

    for index in range(len(command) - 2, -1, -1):
        if command[index] == option:
            return command[index + 1]
    return None


def _evaluate_worldgraph_t2(
    command: list[str],
    *,
    log: TextIO,
    task_name: str,
    dataset: str,
    seed: int,
) -> tuple[int, dict[str, float] | None]:

    checkpoint = _command_value(command, "--checkpoint")
    training_results = _command_value(command, "--results")
    device = _command_value(command, "--device") or "cpu"
    history_burn_in = (
        _command_value(command, "--validation_history_burn_in")
        or _command_value(command, "--clip_history_burn_in")
        or "0"
    )
    if checkpoint is None or training_results is None:
        return 2, None

    test_results = str(Path(training_results).with_suffix(".test.json"))
    evaluation_command = [
        _python(),
        str(ROOT / "gwm" / "training" / "evaluate_action_rl.py"),
        "--checkpoint", checkpoint,
        "--dataset", dataset,
        "--split", "test",
        "--device", device,
        "--history_burn_in", history_burn_in,
        "--results", test_results,
    ]
    controller_count_weight = float(
        _command_value(command, "--controller_topology_count_weight") or "0"
    )
    if controller_count_weight > 0.0:
        evaluation_command.extend(
            ["--topology_decode_policy", "controller_count_topk"]
        )
        if "--topology_history_prior" in command:
            evaluation_command.append("--topology_history_prior")
    evaluation = subprocess.run(
        evaluation_command,
        cwd=ROOT,
        env=subprocess_cpu_environment(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if evaluation.returncode != 0:



        if evaluation.stdout:
            print(evaluation.stdout, end="", flush=True)
            log.write(evaluation.stdout)
            log.flush()
        return evaluation.returncode, None

    try:
        payload = json.loads(Path(test_results).read_text(encoding="utf-8"))
        formal = payload["formal_metrics"]
        metrics = {
            "Add. F1": float(formal["edge_addition"]["f1"]),
            "Rem. F1": float(formal["edge_deletion"]["f1"]),
            "Change Macro-F1": float(formal["edge_change_only_macro_f1"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return 2, None
    _emit(
        f"[TEST/{task_name}/{dataset}/seed={seed}] "
        + _format_metrics(metrics, TASK_REPORT_METRICS["T2"]),
        log,
    )
    return 0, metrics


def _execute(
    command: list[str],
    *,
    cwd: Path,
    log: TextIO,
    task: str,
    task_name: str,
    dataset: str,
    seed: int,
) -> tuple[int, dict[str, float] | None]:

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=subprocess_cpu_environment(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    test_metrics: dict[str, float] | None = None
    for line in process.stdout:
        parsed = _parse_test_metrics_line(line.rstrip("\n"))
        if parsed is None:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            continue
        names = TASK_REPORT_METRICS.get(task, tuple(parsed))
        if any(name not in parsed for name in names):


            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            continue
        test_metrics = {name: parsed[name] for name in names}
        _emit(
            f"[TEST/{task_name}/{dataset}/seed={seed}] "
            + _format_metrics(test_metrics, names),
            log,
        )
    return_code = process.wait()
    if (
        return_code == 0
        and test_metrics is None
        and task == "T2"
        and any(Path(part).name == "train_action_rl.py" for part in command)
    ):
        return _evaluate_worldgraph_t2(
            command,
            log=log,
            task_name=task_name,
            dataset=dataset,
            seed=seed,
        )
    return return_code, test_metrics


def _existing_test_metrics(
    log_path: Path, *, task_name: str, dataset: str
) -> dict[int, dict[str, float]]:

    if not log_path.exists():
        return {}
    prefix = f"[TEST/{task_name}/{dataset}/seed="
    records: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(prefix) or "] " not in line:
            continue
        header, payload = line.split("] ", 1)
        try:
            seed = int(header.rsplit("seed=", 1)[1])
        except ValueError:
            continue
        metrics: dict[str, float] = {}
        for field in payload.split("|"):
            if "=" not in field:
                continue
            name, value = field.rsplit("=", 1)
            try:
                metrics[name.strip()] = float(value.strip().split("±", 1)[0])
            except ValueError:
                continue
        if metrics:
            records[seed] = metrics
    return records


def _last_test_metrics(log_path: Path) -> tuple[str, dict[str, float]] | None:

    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.startswith("[TEST-MEAN/") or "] " not in line:
            continue
        header, payload = line.split("] ", 1)
        prefix = header.rsplit("/n=", 1)[0]
        metrics = _parse_test_metrics_line(line)
        if metrics is None:
            continue
        return (prefix, metrics) if metrics else None
    return None


def _write_batch_summary(
    *, log_paths: list[Path], dataset: str, timestamp: str
) -> Path | None:

    records = [record for path in log_paths if (record := _last_test_metrics(path))]
    if len(records) < 2:
        return None
    prefix = records[0][0]
    per_seed = [metrics for record_prefix, metrics in records if record_prefix == prefix]
    if len(per_seed) < 2:
        return None
    metric_names = [name for name in per_seed[0] if all(name in row for row in per_seed)]
    if not metric_names:
        return None
    fields: list[str] = []
    for name in metric_names:
        values = [row[name] for row in per_seed]
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        fields.append(f"{name}={mean:.4f}±{std:.4f}")
    seed_values: list[int] = []
    for path in log_paths:
        match = re.search(r"_s(\d+)_", path.stem)
        if match is not None:
            seed_values.append(int(match.group(1)))
    seed_suffix = (
        f"_s{min(seed_values)}_to_s{max(seed_values)}"
        if seed_values
        else ""
    )
    summary_path = log_paths[0].parent / (
        f"summary_{dataset}{seed_suffix}_{timestamp}.log"
    )
    summary_path.write_text(
        f"{prefix}/n={len(per_seed)}] "
        + " | ".join(fields)
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_FOLDERS), required=True)
    parser.add_argument("--task", choices=["T1", "T2", "T3"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more explicit seeds. Omit for seeds 1--5. A single seed "
            "combined with --num_seeds retains the legacy consecutive range."
        ),
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Legacy consecutive-seed count; use --seed S1 S2 ... for an explicit set.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--history_window", type=int, default=None,
        help=(
            "WorldGraph temporal history window. Defaults to 8 for ordinary "
            "runs and to the compatible pretrained architecture when a "
            "pretrained backbone is selected."
        ),
    )
    parser.add_argument(
        "--sgt_num_hops", type=int, default=None,
        help="Optional WorldGraph message-passing hop-count override.",
    )
    parser.add_argument(
        "--sgt_num_walks", type=int, default=None,
        help="Optional WorldGraph random-walk count override.",
    )
    parser.add_argument(
        "--grpo_rollout_budget", type=int, default=None,
        help="Optional WorldGraph GRPO rollout-budget override.",
    )
    parser.add_argument(
        "--group_beta", type=float, default=None,
        help="Optional dynamic action-group rarity exponent override.",
    )
    parser.add_argument(
        "--degree_power", type=float, default=None,
        help="Optional degree-importance exponent theta override.",
    )
    parser.add_argument(
        "--volatility_power", type=float, default=None,
        help="Optional historical-variability exponent eta override.",
    )
    parser.add_argument(
        "--pretrained_backbone",
        type=Path,
        default=None,
        help="Optional WorldGraph LOO-pretraining checkpoint (WorldGraph only).",
    )
    parser.add_argument(
        "--pretrained_backbone_blend", type=float, default=None,
        help="Optional fraction of compatible pretrained weights to load.",
    )
    parser.add_argument(
        "--pretrained_graph_encoder_blend", type=float, default=None,
        help="Optional graph-encoder-specific pretrained blend fraction (T2).",
    )
    parser.add_argument(
        "--pretrained_state_model_blend", type=float, default=None,
        help="Optional history-state-specific pretrained blend fraction (T2).",
    )
    parser.add_argument(
        "--pretrained_latent_predictor_blend", type=float, default=None,
        help="Optional latent-predictor-specific pretrained blend fraction (T2).",
    )
    parser.add_argument(
        "--pretrained_controller_blend", type=float, default=None,
        help="Optional fraction of pretrained ADD_NODE Controller initialization to load.",
    )
    parser.add_argument(
        "--pretrained_task_blend", type=float, default=None,
        help="Optional fraction of task-aligned pretrained module initialization to load.",
    )
    parser.add_argument(
        "--pretrained_transfer_gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable conservative learnable transfer gates for WorldGraph-pretrain. "
            "The T3 residual gates remain enabled by default; T2 requires this "
            "explicit flag and uses a frozen LOO graph branch."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_gate_direction_bypass",
        action="store_true",
        help=(
            "For T3 pretraining transfer, keep the gated representation for "
            "regression but bypass it in the direction-classification head."
        ),
    )
    parser.add_argument(
        "--pretrained_direction_transfer_mix",
        type=float,
        default=None,
        help="Optional T3 direction-head mixture of pre/post-transfer states (0=pre, 1=post).",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Optional WorldGraph optimizer learning-rate override.",
    )
    parser.add_argument(
        "--controller_lr", type=float, default=None,
        help="Optional WorldGraph Controller learning-rate override.",
    )
    parser.add_argument(
        "--direction_loss_weight", type=float, default=None,
        help="Optional T3 direction-loss weight override.",
    )
    parser.add_argument(
        "--pretrained_magnitude_head_blend", type=float, default=None,
        help="Optional T3 pretrained magnitude-head transfer fraction.",
    )
    parser.add_argument(
        "--pretrained_direction_head_blend", type=float, default=None,
        help="Optional T3 pretrained direction-head transfer fraction.",
    )
    parser.add_argument(
        "--direction_class_balance_power", type=float, default=None,
        help="Optional T3 direction class-balance exponent override.",
    )
    parser.add_argument(
        "--selection_change_f1_weight", type=float, default=None,
        help="Optional T3 validation Change Macro-F1 weight override.",
    )
    parser.add_argument(
        "--selection_change_rmse_weight", type=float, default=None,
        help="Optional T3 validation changed-RMSE weight override.",
    )
    parser.add_argument(
        "--no_train_rotation", action="store_true",
        help=(
            "For WorldGraph T1, remove the dataset-specific rotating training "
            "clip (the default for Genre)."
        ),
    )
    parser.add_argument(
        "--train_clip_transitions",
        type=int,
        default=None,
        help="WorldGraph T3 contiguous training clip length; omit for full chronological training.",
    )
    parser.add_argument(
        "--semantic_reference_mode",
        choices=["current", "historical_mean"],
        default=None,
        help="Override the WorldGraph T1 causal semantic-property reference.",
    )
    parser.add_argument(
        "--controller_history_window",
        type=int,
        default=None,
        help="Optional T1 Controller causal-history length override.",
    )
    parser.add_argument(
        "--cache_action_candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the WorldGraph T3 deterministic action-candidate cache; "
            "T3 enables it by default."
        ),
    )
    parser.add_argument(
        "--batch_edge_grpo_rollouts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override T3 same-group edge GRPO batching; enabled by default."
        ),
    )
    parser.add_argument(
        "--tensorized_grpo_rewards",
        action="store_true",
        help=(
            "Experimental T3 on-device verifier-reward aggregation. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--cache_released_history",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Cache deterministic released edits and causal edge-history "
            "updates. Enabled by default for T3; use the --no- form for comparison."
        ),
    )
    parser.add_argument(
        "--reuse_controller_policy_states",
        action="store_true",
        help=(
            "Experimental T3 reuse of differentiable Controller policy "
            "states between action proposal and Controller training."
        ),
    )
    parser.add_argument(
        "--batch_grpo_rollout_forward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Batch independent T1 GRPO reward forwards. Enabled by default "
            "for T1; use --no-batch_grpo_rollout_forward only for controlled "
            "compatibility experiments."
        ),
    )
    parser.add_argument(
        "--batch_topology_grpo_rollouts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Batch mixed add/remove-edge Controller and GRPO evaluation. "
            "Enabled by default for T2; use the --no- form for comparison."
        ),
    )
    parser.add_argument(
        "--batch_topology_grpo_rewards",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Batch T2 decoder-only verifier rewards on-device. Enabled by "
            "default for T2; use the --no- form for compatibility runs."
        ),
    )
    parser.add_argument(
        "--topology_composite_embedding_reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the Tex composite T2 verifier reward. The --no- form is "
            "reserved for controlled old-reward comparisons."
        ),
    )
    parser.add_argument(
        "--grpo_update_interval",
        type=int,
        default=None,
        help=(
            "Experimental T2 GRPO update interval override. Omit to retain "
            "the dataset-specific default."
        ),
    )
    parser.add_argument(
        "--t3_grpo_interval",
        type=int,
        default=None,
        help=(
            "Experimental T3 GRPO update interval override. Set to 1 to "
            "update on every eligible graph transition."
        ),
    )
    parser.add_argument(
        "--t1_composite_structure_reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the four-target task-specific verifier reward for T1 GRPO "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--t3_magnitude_reward_weight",
        type=float,
        default=None,
        help=(
            "Experimental T3 mixture weight for the changed-node magnitude "
            "verifier. Omit to retain the established default."
        ),
    )
    parser.add_argument(
        "--cache_deterministic_topology_candidates",
        action="store_true",
        help=(
            "Experimentally cache complete deterministic T2 candidate sets "
            "on CPU; randomly truncated candidates are never cached."
        ),
    )
    parser.add_argument(
        "--cache_device_topology_candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Experimentally retain complete deterministic T2 action "
            "candidates and causal pair features on the training device. "
            "Enabled by default for T2."
        ),
    )
    parser.add_argument(
        "--inprocess_validation",
        action="store_true",
        help=(
            "Experimentally reuse the training process for repeated WorldGraph "
            "validation; currently intended for T2 timing experiments."
        ),
    )
    parser.add_argument(
        "--persistent_validation_worker",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reuse an external validation worker without sharing training "
            "RNG or recurrent state. Enabled by default for T2."
        ),
    )
    parser.add_argument(
        "--cache_static_action_candidates",
        action="store_true",
        help="Opt-in bounded cache for immutable T1 action-candidate indices.",
    )
    parser.add_argument(
        "--vectorized_controller_forward",
        action="store_true",
        help="Opt-in batched execution of independent T1 Controller trunks.",
    )
    parser.add_argument(
        "--cache_validation_preparation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse deterministic T1 validation features across epochs.",
    )
    parser.add_argument("--lambda_activation_ranking", type=float, default=None)
    parser.add_argument("--lambda_deactivation_ranking", type=float, default=None)
    parser.add_argument("--lambda_deactivation", type=float, default=None)
    parser.add_argument("--deactivation_focal_gamma", type=float, default=None)
    parser.add_argument("--lambda_semantic_distance", type=float, default=None)
    parser.add_argument("--semantic_pos_weight_scale", type=float, default=None)
    parser.add_argument(
        "--node_activity_addition_source_context",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_activity_full_observed_context",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_activity_source_visibility_context",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_addition_source_visibility_threshold",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_activity_history_context",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_activity_trajectory_features",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--controller_count_decision_mode",
        choices=["same", "none", "add", "edits", "all", "remove"],
        default=None,
        help="Optional T1 evaluation count-decision override.",
    )
    parser.add_argument(
        "--node_operation_count_history_window",
        type=int,
        default=None,
        help="Optional causal history window for T1 operation-count decisions.",
    )
    parser.add_argument(
        "--node_operation_count_source",
        choices=["action_history", "controller", "future_latent"],
        default=None,
        help="Optional causal source for T1 ranked edit cardinalities.",
    )
    parser.add_argument(
        "--node_operation_count_decoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optionally train the future-state node-operation count decoder.",
    )
    parser.add_argument("--lambda_node_operation_count", type=float, default=None)
    parser.add_argument(
        "--decision_calibration_tail_fraction",
        type=float,
        default=None,
        help="Optional latest-validation fraction used for T1 decision calibration.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["auprc", "f1", "min_f1"],
        default=None,
        help="Optional T1 validation-only checkpoint selection metric.",
    )
    parser.add_argument("--selection_property_weight", type=float, default=None)
    parser.add_argument(
        "--semantic_causal_expert",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_change_history_prior",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_change_source_context",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_activity_history_gate",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_activity_history_prior",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--node_activity_history_prior_remove_only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dynamic_change_hard_fraction", type=float, default=None)
    parser.add_argument("--dynamic_change_focal_gamma", type=float, default=None)
    parser.add_argument("--semantic_hard_pair_weight", type=float, default=None)
    parser.add_argument("--semantic_hard_pair_margin", type=float, default=None)
    parser.add_argument(
        "--node_activity_score_source", choices=["future_latent", "controller"], default=None
    )
    parser.add_argument(
        "--node_addition_score_source", choices=["same", "future_latent", "controller"], default=None
    )
    parser.add_argument(
        "--node_removal_score_source", choices=["same", "future_latent", "controller"], default=None
    )
    parser.add_argument("--node_removal_history_bias", type=float, default=None)
    parser.add_argument(
        "--node_removal_group_calibration_mode",
        choices=["none", "zscore", "percentile", "stratified_percentile"],
        default=None,
    )
    parser.add_argument(
        "--node_removal_group_calibration_weight", type=float, default=None
    )
    parser.add_argument("--semantic_history_bias", type=float, default=None)
    parser.add_argument("--semantic_property_change_bias", type=float, default=None)
    parser.add_argument("--semantic_transition_gate_bias", type=float, default=None)
    parser.add_argument("--semantic_volatility_residual_bias", type=float, default=None)
    parser.add_argument("--activity_trajectory_supervision_bias", type=float, default=None)
    parser.add_argument("--removal_history_supervision_bias", type=float, default=None)
    parser.add_argument("--node_removal_history_bias_grid", type=float, nargs="+", default=None)
    parser.add_argument(
        "--node_removal_group_calibration_weight_grid",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--semantic_property_change_bias_grid",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--semantic_gate_bias_grid",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=None,
        help="Override the temporal history length for external baselines.",
    )
    parser.add_argument(
        "--artifact_tag",
        type=str,
        default=None,
        help="Optional run tag appended to checkpoint/manifest timestamps to isolate parallel launches.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Override the business-log path when resuming an interrupted batch.",
    )
    parser.add_argument(
        "--append_log",
        action="store_true",
        help="Append to --log_file and include its existing tests in the final mean.",
    )
    parser.add_argument("--execute", action="store_true", help="Run commands; omit to print and write manifests only.")
    args = parser.parse_args()
    if args.grpo_update_interval is not None:
        if args.grpo_update_interval < 1:
            parser.error("--grpo_update_interval must be positive")
        if args.task != "T2" or args.model not in {"worldgraph", "worldgraph_pretrain"}:
            parser.error("--grpo_update_interval is only supported for WorldGraph T2")
    if args.t3_grpo_interval is not None:
        if args.t3_grpo_interval < 1:
            parser.error("--t3_grpo_interval must be positive")
        if args.task != "T3" or args.model not in {"worldgraph", "worldgraph_pretrain"}:
            parser.error("--t3_grpo_interval is only supported for WorldGraph T3")




    native_flights_alias = bool(
        args.task == "T3" and args.dataset.lower() == "flights_native"
    )
    dataset = "flights" if native_flights_alias else args.dataset
    output_dataset = "flights" if native_flights_alias else dataset
    if args.num_seeds is not None and args.num_seeds < 1:
        parser.error("--num_seeds must be positive")
    if args.seed is None:
        seeds = list(range(1, 1 + (args.num_seeds or 5)))
    elif len(args.seed) == 1 and args.num_seeds is not None:
        seeds = list(range(args.seed[0], args.seed[0] + args.num_seeds))
    elif args.num_seeds is not None:
        parser.error("--num_seeds cannot be combined with multiple explicit --seed values")
    else:
        seeds = list(args.seed)
    if len(set(seeds)) != len(seeds):
        parser.error("--seed values must be unique")
    protocol = None
    entry = resolve_task_dataset(args.task, dataset)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if args.artifact_tag is not None:
        if not args.artifact_tag or not all(
            char.isalnum() or char in "_-" for char in args.artifact_tag
        ):
            parser.error("--artifact_tag must contain only letters, digits, '_' or '-'")
        timestamp = f"{timestamp}_{args.artifact_tag}"
    log_root = ROOT / "results" / _task_folder(args.task) / MODEL_FOLDERS[args.model] / output_dataset
    log_root.mkdir(parents=True, exist_ok=True)


    log_name = (
        f"{output_dataset}_s{seeds[0]}_{timestamp}.log"
        if len(seeds) == 1
        else f"{output_dataset}_{timestamp}.log"
    )
    batch_log = args.log_file.resolve() if args.log_file is not None else log_root / log_name
    if args.append_log and args.log_file is None:
        parser.error("--append_log requires --log_file")
    existing_metrics = (
        _existing_test_metrics(
            batch_log, task_name=entry["task_name"], dataset=output_dataset
        )
        if args.append_log
        else {}
    )
    completed_metrics: list[dict[str, float]] = list(existing_metrics.values())



    batch_log.parent.mkdir(parents=True, exist_ok=True)
    with batch_log.open("a" if args.append_log else "w", encoding="utf-8") as log:
        if args.append_log:
            _emit(
                f"resume_seeds={','.join(str(seed) for seed in seeds)} "
                f"existing_test_seeds={','.join(str(seed) for seed in sorted(existing_metrics))}",
                log,
            )
        _emit(f"seeds={','.join(str(seed) for seed in seeds)}", log)
        _emit(f"log={batch_log}", log)
        for seed in seeds:
            paths = _paths(args.task, args.model, output_dataset, seed, timestamp)
            try:
                command_kwargs = dict(
                    model=args.model,
                    task=args.task,
                    dataset=dataset,
                    seed=seed,
                    device=args.device,
                    paths=paths,
                    protocol=protocol,
                    entry=entry,
                )
                if args.model in {"worldgraph", "worldgraph_pretrain"}:
                    command_kwargs["history_window"] = args.history_window
                    command_kwargs["sgt_num_hops"] = args.sgt_num_hops
                    command_kwargs["sgt_num_walks"] = args.sgt_num_walks
                    command_kwargs["grpo_rollout_budget"] = args.grpo_rollout_budget
                    command_kwargs["group_beta"] = args.group_beta
                    command_kwargs["degree_power"] = args.degree_power
                    command_kwargs["volatility_power"] = args.volatility_power




                    command_kwargs["no_train_rotation"] = (
                        args.no_train_rotation
                        or (args.task == "T1" and dataset.lower() == "genre")
                    )
                    command_kwargs["train_clip_transitions"] = args.train_clip_transitions
                    command_kwargs["semantic_reference_mode"] = args.semantic_reference_mode
                    command_kwargs["controller_history_window"] = args.controller_history_window
                    default_pretrained_backbone = DEFAULT_WORLDGRAPH_PRETRAIN_BACKBONES.get(
                        (args.task, dataset.lower())
                    )
                    command_kwargs["pretrained_backbone"] = (
                        args.pretrained_backbone.resolve()
                        if args.pretrained_backbone is not None
                        else default_pretrained_backbone
                        if args.model == "worldgraph_pretrain"
                        else None
                    )
                    default_pretrained_key = (args.task, dataset.lower())
                    use_dataset_pretrained_defaults = (
                        args.model == "worldgraph_pretrain"
                        and args.pretrained_backbone is None
                    )
                    command_kwargs["pretrained_backbone_blend"] = (
                        args.pretrained_backbone_blend
                        if args.pretrained_backbone_blend is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_BACKBONE_BLENDS.get(
                            default_pretrained_key
                        )
                        if use_dataset_pretrained_defaults
                        else None
                    )
                    command_kwargs["pretrained_graph_encoder_blend"] = (
                        args.pretrained_graph_encoder_blend
                        if args.pretrained_graph_encoder_blend is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_COMPONENT_BLENDS.get(
                            default_pretrained_key, {}
                        ).get("graph_encoder")
                    )
                    command_kwargs["pretrained_state_model_blend"] = (
                        args.pretrained_state_model_blend
                        if args.pretrained_state_model_blend is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_COMPONENT_BLENDS.get(
                            default_pretrained_key, {}
                        ).get("state_model")
                    )
                    command_kwargs["pretrained_latent_predictor_blend"] = (
                        args.pretrained_latent_predictor_blend
                        if args.pretrained_latent_predictor_blend is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_COMPONENT_BLENDS.get(
                            default_pretrained_key, {}
                        ).get("latent_predictor")
                    )
                    command_kwargs["pretrained_controller_blend"] = args.pretrained_controller_blend
                    command_kwargs["pretrained_task_blend"] = (
                        args.pretrained_task_blend
                        if args.pretrained_task_blend is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_TASK_BLENDS.get(
                            default_pretrained_key
                        )
                        if use_dataset_pretrained_defaults
                        else None
                    )
                    command_kwargs["pretrained_transfer_gate"] = (
                        args.model == "worldgraph_pretrain"
                        and args.task == "T3"
                        if args.pretrained_transfer_gate is None
                        else args.pretrained_transfer_gate
                    )
                    command_kwargs["pretrained_transfer_gate_direction_bypass"] = (
                        args.pretrained_transfer_gate_direction_bypass
                        or (
                            args.model == "worldgraph_pretrain"
                            and args.pretrained_backbone is None
                            and (args.task, dataset.lower())
                            in DEFAULT_WORLDGRAPH_PRETRAIN_DIRECTION_BYPASS
                        )
                    )
                    command_kwargs["pretrained_direction_transfer_mix"] = (
                        args.pretrained_direction_transfer_mix
                        if args.pretrained_direction_transfer_mix is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_DIRECTION_TRANSFER_MIX.get(
                            default_pretrained_key
                        )
                        if args.model == "worldgraph_pretrain"
                        else None
                    )
                    command_kwargs["lr"] = args.lr
                    command_kwargs["controller_lr"] = args.controller_lr
                    command_kwargs["direction_loss_weight"] = args.direction_loss_weight
                    command_kwargs["pretrained_magnitude_head_blend"] = (
                        args.pretrained_magnitude_head_blend
                    )
                    command_kwargs["pretrained_direction_head_blend"] = (
                        args.pretrained_direction_head_blend
                    )
                    command_kwargs["direction_class_balance_power"] = (
                        args.direction_class_balance_power
                    )
                    command_kwargs["selection_change_f1_weight"] = (
                        args.selection_change_f1_weight
                    )
                    command_kwargs["selection_change_rmse_weight"] = (
                        args.selection_change_rmse_weight
                    )
                    command_kwargs["cache_action_candidates"] = (
                        args.task == "T3"
                        if args.cache_action_candidates is None
                        else args.cache_action_candidates
                    )
                    command_kwargs["batch_edge_grpo_rollouts"] = (
                        args.task == "T3"
                        if args.batch_edge_grpo_rollouts is None
                        else args.batch_edge_grpo_rollouts
                    )
                    command_kwargs["tensorized_grpo_rewards"] = (
                        args.tensorized_grpo_rewards
                    )
                    command_kwargs["cache_released_history"] = (
                        args.task == "T3"
                        if args.cache_released_history is None
                        else args.cache_released_history
                    )
                    command_kwargs["reuse_controller_policy_states"] = (
                        args.reuse_controller_policy_states
                    )
                    command_kwargs["cache_static_action_candidates"] = (
                        args.cache_static_action_candidates
                    )
                    command_kwargs["vectorized_controller_forward"] = (
                        args.vectorized_controller_forward
                    )
                    command_kwargs["cache_validation_preparation"] = (
                        args.cache_validation_preparation
                    )
                    command_kwargs["batch_grpo_rollout_forward"] = (
                        args.task == "T1"
                        if args.batch_grpo_rollout_forward is None
                        else args.batch_grpo_rollout_forward
                    )
                    command_kwargs["batch_topology_grpo_rollouts"] = (
                        args.task == "T2"
                        if args.batch_topology_grpo_rollouts is None
                        else args.batch_topology_grpo_rollouts
                    )
                    command_kwargs["batch_topology_grpo_rewards"] = (
                        args.task == "T2"
                        if args.batch_topology_grpo_rewards is None
                        else args.batch_topology_grpo_rewards
                    )
                    command_kwargs["topology_composite_embedding_reward"] = (
                        args.topology_composite_embedding_reward
                    )
                    command_kwargs["grpo_update_interval"] = (
                        args.grpo_update_interval
                        if args.grpo_update_interval is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_GRPO_UPDATE_INTERVALS.get(
                            default_pretrained_key
                        )
                        if args.model == "worldgraph_pretrain"
                        else None
                    )
                    command_kwargs["t3_grpo_interval"] = args.t3_grpo_interval
                    command_kwargs["t1_composite_structure_reward"] = (
                        args.t1_composite_structure_reward
                    )
                    command_kwargs["t3_magnitude_reward_weight"] = (
                        args.t3_magnitude_reward_weight
                    )
                    command_kwargs["cache_deterministic_topology_candidates"] = (
                        args.cache_deterministic_topology_candidates
                    )
                    command_kwargs["cache_device_topology_candidates"] = (
                        args.task == "T2"
                        if args.cache_device_topology_candidates is None
                        else args.cache_device_topology_candidates
                    )
                    command_kwargs["inprocess_validation"] = (
                        args.inprocess_validation
                    )
                    command_kwargs["persistent_validation_worker"] = (
                        False
                        if args.inprocess_validation
                        else (
                            args.task == "T2"
                            if args.persistent_validation_worker is None
                            else args.persistent_validation_worker
                        )
                    )
                    command_kwargs["lambda_activation_ranking"] = args.lambda_activation_ranking
                    command_kwargs["lambda_deactivation_ranking"] = args.lambda_deactivation_ranking
                    command_kwargs["lambda_deactivation"] = args.lambda_deactivation
                    command_kwargs["deactivation_focal_gamma"] = args.deactivation_focal_gamma
                    command_kwargs["lambda_semantic_distance"] = args.lambda_semantic_distance
                    command_kwargs["semantic_pos_weight_scale"] = args.semantic_pos_weight_scale
                    command_kwargs["node_activity_addition_source_context"] = (
                        args.node_activity_addition_source_context
                    )
                    command_kwargs["node_activity_full_observed_context"] = (
                        args.node_activity_full_observed_context
                    )
                    command_kwargs["node_activity_source_visibility_context"] = (
                        args.node_activity_source_visibility_context
                    )
                    command_kwargs["node_addition_source_visibility_threshold"] = (
                        args.node_addition_source_visibility_threshold
                    )
                    command_kwargs["node_activity_history_context"] = (
                        args.node_activity_history_context
                    )
                    command_kwargs["node_activity_trajectory_features"] = (
                        args.node_activity_trajectory_features
                    )
                    command_kwargs["controller_count_decision_mode"] = (
                        args.controller_count_decision_mode
                    )
                    command_kwargs["node_operation_count_history_window"] = (
                        args.node_operation_count_history_window
                    )
                    command_kwargs["node_operation_count_source"] = (
                        args.node_operation_count_source
                    )
                    command_kwargs["node_operation_count_decoder"] = (
                        args.node_operation_count_decoder
                    )
                    command_kwargs["lambda_node_operation_count"] = (
                        args.lambda_node_operation_count
                    )
                    command_kwargs["decision_calibration_tail_fraction"] = (
                        args.decision_calibration_tail_fraction
                    )
                    command_kwargs["selection_metric"] = args.selection_metric
                    command_kwargs["selection_property_weight"] = (
                        args.selection_property_weight
                        if args.selection_property_weight is not None
                        else DEFAULT_WORLDGRAPH_PRETRAIN_SELECTION_PROPERTY_WEIGHTS.get(
                            default_pretrained_key
                        )
                        if args.model == "worldgraph_pretrain"
                        else None
                    )
                    command_kwargs["semantic_causal_expert"] = (
                        args.semantic_causal_expert
                    )
                    command_kwargs["node_change_history_prior"] = (
                        args.node_change_history_prior
                    )
                    command_kwargs["node_change_source_context"] = (
                        args.node_change_source_context
                    )
                    command_kwargs["node_activity_history_gate"] = (
                        args.node_activity_history_gate
                    )
                    command_kwargs["node_activity_history_prior"] = (
                        args.node_activity_history_prior
                    )
                    command_kwargs["node_activity_history_prior_remove_only"] = (
                        args.node_activity_history_prior_remove_only
                    )
                    command_kwargs["dynamic_change_hard_fraction"] = (
                        args.dynamic_change_hard_fraction
                    )
                    command_kwargs["dynamic_change_focal_gamma"] = (
                        args.dynamic_change_focal_gamma
                    )
                    command_kwargs["semantic_hard_pair_weight"] = (
                        args.semantic_hard_pair_weight
                    )
                    command_kwargs["semantic_hard_pair_margin"] = (
                        args.semantic_hard_pair_margin
                    )
                    command_kwargs["node_activity_score_source"] = (
                        args.node_activity_score_source
                    )
                    command_kwargs["node_addition_score_source"] = (
                        args.node_addition_score_source
                    )
                    command_kwargs["node_removal_score_source"] = (
                        args.node_removal_score_source
                    )
                    command_kwargs["node_removal_history_bias"] = (
                        args.node_removal_history_bias
                    )
                    command_kwargs["node_removal_group_calibration_mode"] = (
                        args.node_removal_group_calibration_mode
                    )
                    command_kwargs["node_removal_group_calibration_weight"] = (
                        args.node_removal_group_calibration_weight
                    )
                    command_kwargs["semantic_history_bias"] = (
                        args.semantic_history_bias
                    )
                    command_kwargs["semantic_property_change_bias"] = (
                        args.semantic_property_change_bias
                    )
                    command_kwargs["semantic_transition_gate_bias"] = (
                        args.semantic_transition_gate_bias
                    )
                    command_kwargs["semantic_volatility_residual_bias"] = (
                        args.semantic_volatility_residual_bias
                    )
                    command_kwargs["activity_trajectory_supervision_bias"] = (
                        args.activity_trajectory_supervision_bias
                    )
                    command_kwargs["removal_history_supervision_bias"] = (
                        args.removal_history_supervision_bias
                    )
                    command_kwargs["node_removal_history_bias_grid"] = (
                        args.node_removal_history_bias_grid
                    )
                    command_kwargs[
                        "node_removal_group_calibration_weight_grid"
                    ] = args.node_removal_group_calibration_weight_grid
                    command_kwargs["semantic_property_change_bias_grid"] = (
                        args.semantic_property_change_bias_grid
                    )
                    command_kwargs["semantic_gate_bias_grid"] = (
                        args.semantic_gate_bias_grid
                    )
                elif args.model in {
                    "graphsage", "graphgps", "gcn", "gat", "sgformer", "nodeformer",
                    "tgn", "tidformer"
                }:
                    command_kwargs["history_length"] = args.history_length
                command, cwd = _command_for(**command_kwargs)
            except NotImplementedError as error:
                raise SystemExit(f"{args.model}/{args.task}: {error}") from error
            command = _repo_relative_command(command)
            _write_manifest(
                paths["manifest"], command=command, cwd=cwd, entry=entry,
                model=args.model, seed=seed, timestamp=timestamp,
            )
            _emit(f"[{args.model}/{args.task}/{output_dataset}/s{seed}] {shlex.join(command)}", log)
            _emit(f"manifest={paths['manifest']}", log)
            if not args.execute:
                continue
            code, metrics = _execute(
                command,
                cwd=cwd,
                log=log,
                task=args.task,
                task_name=entry["task_name"],
                dataset=output_dataset,
                seed=seed,
            )
            if code != 0:
                _emit(f"[ERROR/seed={seed}] return_code={code}", log)
                raise SystemExit(code)
            if metrics is None:
                _emit(f"[ERROR/seed={seed}] missing canonical test metrics", log)
                raise SystemExit(2)
            completed_metrics.append(metrics)

        if args.execute:
            metric_names = list(completed_metrics[0])
            summary_fields = []
            for name in metric_names:
                values = [row[name] for row in completed_metrics]
                summary_fields.append(
                    f"{name}={statistics.mean(values):.4f}±"
                    f"{(statistics.pstdev(values) if len(values) > 1 else 0.0):.4f}"
                )
            _emit(
                f"[TEST-MEAN/{entry['task_name']}/{output_dataset}/n={len(completed_metrics)}] "
                + " | ".join(summary_fields),
                log,
            )


if __name__ == "__main__":
    main()
