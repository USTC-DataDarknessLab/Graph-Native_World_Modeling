

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gwm.action_rl import (
    ActionAwareController,
    calibrated_soft_node_action_probability,
    structure_node_weights,
)
from gwm.action_metrics import (
    PropertyMetricAccumulator,
    RegressionAccumulator,
    SetMetricAccumulator,
    calibrate_node_change,
    calibrate_node_change_history_blend,
    calibrate_past_rate_topk,
    edge_codes,
    evaluate_node_change,
    evaluate_node_change_history_blend,
    evaluate_past_rate_topk,
    node_change_calibrated_scores,
)
from gwm.actions import GraphActionEncoder, GraphOperation
from gwm.data.action_benchmark import (
    ACTION_BENCHMARK_DATASETS,
    infer_action_benchmark_dataset,
    resolve_action_benchmark_path,
)
from gwm.model import GraphWorldModel
from gwm.transfer import FrozenGraphTransferBranch, GatedResidualTransferAdapter
from gwm.pair_state import CausalPairState, PAIR_STATE_DIM
from gwm.genre_history_state import GenreObservedPropertyHistory
from gwm.property_transition import decode_property_prediction
from gwm.utils import resolve_device, save_json, seed_everything
from gwm.training.train_action_rl import (
    TASK_OPERATIONS,
    _decoded_reward,
    _edge_weight,
    _graph_transitions,
    _encode_action_or_zero,
    _past_change_observation_count,
    _past_change_update,
    _topk_edges,
    _topology_candidates,
    _topology_count_context,
    _topology_reward_context,
    _transition_to_device,
)





NODE_HISTORY_DECAYS = (0.5, 0.8, 0.9, 0.95)


def _new_node_change_history(num_nodes: int) -> dict[str, Any]:
    return {
        "previous_change": torch.zeros(num_nodes, dtype=torch.float32),
        "cumulative_change": torch.zeros(num_nodes, dtype=torch.float32),
        "completed": 0,
        "ema": {
            decay: torch.zeros(num_nodes, dtype=torch.float32)
            for decay in NODE_HISTORY_DECAYS
        },
    }


def _observe_node_change_history(
    history: dict[str, Any], change: torch.Tensor
) -> None:
    change = change.detach().cpu().to(dtype=torch.float32)
    history["previous_change"] = change.clone()
    history["cumulative_change"].add_(change)
    for decay, value in history["ema"].items():
        value.mul_(float(decay)).add_(change, alpha=1.0 - float(decay))
    history["completed"] = int(history["completed"]) + 1


def _node_change_history_features(history: dict[str, Any]) -> dict[str, torch.Tensor]:
    completed = int(history["completed"])
    return {
        "previous_change": history["previous_change"],
        "cumulative_rate": (
            history["cumulative_change"] + 1.0
        ) / float(completed + 2),
        **{
            f"ema_{decay:g}": value
            for decay, value in history["ema"].items()
        },
    }


def _bucket_node_history_features(
    bucket: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {
        name: torch.cat(values).float().numpy()
        for name, values in bucket["node_history"].items()
        if values
    }


def _calibrate_node_change_readout(
    validation: dict[str, Any],
    requested: dict[str, Any],
    *,
    logits_key: str,
    use_node_history_blend: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation_labels = torch.cat(validation["labels"]).numpy()
    validation_logits = torch.cat(validation[logits_key]).numpy()
    requested_labels = torch.cat(requested["labels"]).numpy()
    requested_logits = torch.cat(requested[logits_key]).numpy()
    if use_node_history_blend:
        validation_features = _bucket_node_history_features(validation)
        requested_features = _bucket_node_history_features(requested)
        if validation_features and requested_features:
            calibration = calibrate_node_change_history_blend(
                validation_labels,
                validation_logits,
                validation_features,
            )
            metrics = evaluate_node_change_history_blend(
                requested_labels,
                requested_logits,
                requested_features,
                calibration,
            )
            return calibration, metrics
    calibration = calibrate_node_change(
        validation_labels,
        validation_logits,
        torch.cat(validation["history"]).numpy(),
    )
    metrics = evaluate_node_change(
        requested_labels,
        requested_logits,
        torch.cat(requested["history"]).numpy(),
        calibration,
    )
    return calibration, metrics


def _checkpoint_value(
    checkpoint_args: dict[str, Any], key: str, default: Any
) -> Any:
    value = checkpoint_args.get(key, default)
    return default if value is None else value


def _load_backward_compatible_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    optional_missing_prefixes: tuple[str, ...] = (),
) -> None:

    incompatible = module.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key
        for key in missing
        if key.startswith(optional_missing_prefixes)
    }
    disallowed_missing = missing - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the requested model: "
            f"missing={sorted(disallowed_missing)}, "
            f"unexpected={sorted(unexpected)}."
        )


def _calibrate_state_gate(
    rows: dict[str, list[torch.Tensor]],
    requested_split: str,
    feature_names: list[str] | None,
) -> tuple[dict[str, Any], RegressionAccumulator, RegressionAccumulator]:
    fit_split = "val" if requested_split == "test" else requested_split
    if not rows.get(fit_split, {}).get("target"):
        raise RuntimeError("State-gate calibration requires validation rows.")
    logits_fit = torch.cat(rows[fit_split]["logits"]).float()
    raw_fit = torch.cat(rows[fit_split]["raw"]).float()
    target_fit = torch.cat(rows[fit_split]["target"]).float()
    labels_fit = torch.cat(rows[fit_split]["labels"]).bool()



    threshold_candidates = torch.unique(
        torch.cat(
            [
                logits_fit.new_tensor([0.0]),
                logits_fit.flatten().quantile(
                    torch.linspace(0.05, 0.95, 19, device=logits_fit.device)
                ).reshape(-1),
            ]
        )
    ).sort().values
    alpha_candidates = torch.linspace(
        0.0, 1.0, 21, device=raw_fit.device, dtype=raw_fit.dtype
    )


    copy_all_mae = target_fit.abs().mean().item()
    best_feasible: tuple[float, float, float] | None = None
    best_any: tuple[float, float, float] | None = None
    for threshold in threshold_candidates:
        gate = (logits_fit >= threshold).to(raw_fit.dtype).unsqueeze(-1)
        candidate_prediction = (
            gate.unsqueeze(0)
            * raw_fit.unsqueeze(0)
            * alpha_candidates[:, None, None]
        )
        candidate_abs = (candidate_prediction - target_fit.unsqueeze(0)).abs()
        all_error = candidate_abs.mean(dim=(1, 2))
        if bool(labels_fit.any()):
            changed_abs = candidate_abs[:, labels_fit]
            changed_error = changed_abs.mean(dim=(1, 2))
            changed_rmse = torch.sqrt(changed_abs.square().mean(dim=(1, 2)))
        else:
            changed_error = torch.zeros_like(all_error)
            changed_rmse = torch.zeros_like(all_error)
        objective = (
            changed_error
            + 0.01 * changed_rmse
            + 0.01 * all_error
        )
        for alpha_index in range(alpha_candidates.numel()):
            candidate = (
                float(objective[alpha_index].item()),
                float(threshold.item()),
                float(alpha_candidates[alpha_index].item()),
            )
            if best_any is None or candidate < best_any:
                best_any = candidate
            if float(all_error[alpha_index].item()) <= 1.02 * copy_all_mae:
                if best_feasible is None or candidate < best_feasible:
                    best_feasible = candidate
    best = best_feasible or best_any
    assert best is not None
    _, threshold_value, alpha_value = best

    def evaluate(split: str) -> tuple[RegressionAccumulator, RegressionAccumulator]:
        if not rows.get(split, {}).get("target"):
            raise RuntimeError(f"No node-state rows for split={split}.")
        logits = torch.cat(rows[split]["logits"]).float()
        raw = torch.cat(rows[split]["raw"]).float()
        target = torch.cat(rows[split]["target"]).float()
        labels = torch.cat(rows[split]["labels"]).bool()
        pred = (
            (logits >= threshold_value).to(raw.dtype).unsqueeze(-1)
            * raw
            * alpha_value
        )
        all_acc = RegressionAccumulator()
        changed_acc = RegressionAccumulator()
        all_acc.update(pred, target)
        if bool(labels.any()):
            changed_acc.update(pred[labels], target[labels])
        return all_acc, changed_acc

    all_acc, changed_acc = evaluate(requested_split)
    calibration = {
        "fit_split": fit_split,
        "threshold_logit": threshold_value,
        "delta_scale": alpha_value,
        "objective": "min changed_mae (+0.01*changed_rmse +0.01*all_mae) subject to all_mae <= 1.02 * copy_last_mae",
        "copy_last_validation_mae": copy_all_mae,
        "feasible": best_feasible is not None,
    }
    return calibration, all_acc, changed_acc


def _build_components(
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
    *,
    task: str,
    device: torch.device,
) -> tuple[GraphWorldModel, GraphActionEncoder, ActionAwareController]:
    saved = checkpoint["args"]
    node_feature_dim = int(metadata["model_input_dim"])
    property_dim = (
        int(metadata["official_target_dim"]) if task == "node_property" else None
    )
    history_setting = str(_checkpoint_value(saved, "property_history_input", "auto"))
    dataset_name = str(metadata.get("action_benchmark_dataset", ""))
    property_history_input = task == "node_property" and (
        history_setting in {"on", "True", "true"}
        or (history_setting == "auto" and dataset_name in {"genre", "reddit"})
    )
    input_dim = node_feature_dim + (
        (property_dim or 0) if property_history_input else 0
    )
    latent_dim = int(_checkpoint_value(saved, "latent_dim", 32))
    hidden_dim = int(_checkpoint_value(saved, "hidden_dim", 32))
    action_dim = int(_checkpoint_value(saved, "action_dim", 16))
    policy_dim = int(_checkpoint_value(saved, "policy_dim", 32))
    model = GraphWorldModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        node_property_dim=property_dim,
        property_decoder_zero_init=(
            bool(_checkpoint_value(saved, "property_decoder_zero_init", True))
            if task == "node_property"
            else False
        ),
        state_decoder_zero_init=bool(
            _checkpoint_value(saved, "state_decoder_zero_init", False)
        ),
        state_prediction_mode=str(
            _checkpoint_value(saved, "state_prediction_mode", "ungated")
        ),
        current_state_reconstruction=(
            task in {"node_change", "node_state"}
            and float(_checkpoint_value(saved, "forecast_current_state_weight", 0.0))
            > 0.0
        ),
        topology_transition_heads=(
            task == "topology"
            or bool(_checkpoint_value(saved, "joint_action_groups", False))
            or bool(_checkpoint_value(saved, "synthetic_topology_actions", False))
        ),
        topology_count_decoder=bool(
            float(_checkpoint_value(saved, "topology_count_weight", 0.0)) > 0.0
        ),
        topology_pair_state_dim=(
            PAIR_STATE_DIM
            if bool(_checkpoint_value(saved, "topology_pair_state", False))
            else 0
        ),
        dropout=float(_checkpoint_value(saved, "dropout", 0.0)),
        observable_latent_mode=str(
            _checkpoint_value(saved, "observable_latent_mode", "mean")
        ),
        target_encoder_momentum=float(
            _checkpoint_value(saved, "target_encoder_momentum", 0.99)
        ),
        latent_normalization=str(
            _checkpoint_value(saved, "latent_normalization", "layernorm")
        ),
        latent_min_logvar=float(
            _checkpoint_value(saved, "latent_min_logvar", -6.0)
        ),
        latent_max_logvar=float(
            _checkpoint_value(saved, "latent_max_logvar", 2.0)
        ),




        graph_encoder_type=str(
            _checkpoint_value(saved, "graph_encoder_type", "mentor_sgt_gwm")
        ),
        state_model_type=str(
            _checkpoint_value(
                saved, "state_model_type", "mentor_sgt_gwm_transformer"
            )
        ),
        sgt_num_hops=int(_checkpoint_value(saved, "sgt_num_hops", 1)),
        sgt_num_walks=int(_checkpoint_value(saved, "sgt_num_walks", 1)),
        sgt_walk_length=int(_checkpoint_value(saved, "sgt_walk_length", 1)),



        sgt_deterministic_walks=bool(
            _checkpoint_value(saved, "sgt_deterministic_walks", True)
        ),
        history_window=int(_checkpoint_value(saved, "history_window", 4)),
        history_num_heads=int(_checkpoint_value(saved, "history_num_heads", 4)),
        separate_action_query=bool(
            _checkpoint_value(saved, "separate_action_query", False)
        ),
        bounded_action_residual=bool(
            _checkpoint_value(saved, "bounded_action_residual", False)
        ),
        zero_init_action_adapters=bool(
            _checkpoint_value(saved, "zero_init_action_adapters", False)
        ),
        action_adapter_initial_scale=float(
            _checkpoint_value(saved, "action_adapter_initial_scale", 0.10)
        ),
        action_residual_max_scale=float(
            _checkpoint_value(saved, "action_residual_max_scale", 0.25)
        ),
        action_adapter_squash=bool(
            _checkpoint_value(saved, "action_adapter_squash", False)
        ),
        action_adapter_temperature=float(
            _checkpoint_value(saved, "action_adapter_temperature", 0.25)
        ),
        action_injection_mode=str(
            _checkpoint_value(saved, "action_injection_mode", "full")
        ),






        input_adapter_type=str(
            _checkpoint_value(
                saved,
                "input_adapter_type",
                "residual_domain_invariant"
                if (
                    _checkpoint_value(saved, "pretrained_backbone", None)
                    and str(
                        _checkpoint_value(
                            saved, "pretrained_transfer_mode", "legacy"
                        )
                    )
                    == "legacy"
                )
                else "none",
            )
        ),
        action_plan_decoder=float(
            _checkpoint_value(saved, "action_plan_consistency_weight", 0.0)
        ) > 0.0,
        action_plan_decoder_dim=(
            2
            if task == "topology"
            and float(
                _checkpoint_value(saved, "action_plan_consistency_weight", 0.0)
            )
            > 0.0
            else 1
        ),
        node_change_decoder_input=str(
            _checkpoint_value(saved, "node_change_decoder_input", "future_concat_abs_delta")
        ),
    ).to(device)
    if bool(_checkpoint_value(saved, "pretrained_transfer_gate", False)):
        initial_gate = float(
            _checkpoint_value(saved, "pretrained_transfer_gate_initial", 0.1)
        )



        source_shell = GraphWorldModel(
            input_dim=latent_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            dropout=float(_checkpoint_value(saved, "dropout", 0.0)),
            observable_latent_mode=str(
                _checkpoint_value(saved, "observable_latent_mode", "mean")
            ),
            latent_normalization=str(
                _checkpoint_value(saved, "latent_normalization", "layernorm")
            ),
            graph_encoder_type=str(
                _checkpoint_value(saved, "graph_encoder_type", "mentor_sgt_gwm")
            ),
            state_model_type=str(
                _checkpoint_value(
                    saved, "state_model_type", "mentor_sgt_gwm_transformer"
                )
            ),
            sgt_num_hops=int(_checkpoint_value(saved, "sgt_num_hops", 1)),
            sgt_num_walks=int(_checkpoint_value(saved, "sgt_num_walks", 1)),
            sgt_walk_length=int(_checkpoint_value(saved, "sgt_walk_length", 1)),
            sgt_deterministic_walks=bool(
                _checkpoint_value(saved, "sgt_deterministic_walks", True)
            ),
            history_window=int(_checkpoint_value(saved, "history_window", 4)),
            history_num_heads=int(
                _checkpoint_value(saved, "history_num_heads", 4)
            ),
            separate_action_query=bool(
                _checkpoint_value(saved, "separate_action_query", False)
            ),
            bounded_action_residual=bool(
                _checkpoint_value(saved, "bounded_action_residual", False)
            ),
            zero_init_action_adapters=bool(
                _checkpoint_value(saved, "zero_init_action_adapters", False)
            ),
            action_adapter_initial_scale=float(
                _checkpoint_value(saved, "action_adapter_initial_scale", 0.10)
            ),
            action_residual_max_scale=float(
                _checkpoint_value(saved, "action_residual_max_scale", 0.25)
            ),
            action_adapter_squash=bool(
                _checkpoint_value(saved, "action_adapter_squash", False)
            ),
            action_adapter_temperature=float(
                _checkpoint_value(saved, "action_adapter_temperature", 0.25)
            ),
            action_injection_mode=str(
                _checkpoint_value(saved, "action_injection_mode", "full")
            ),
            input_adapter_type="domain_invariant",
        )
        assert source_shell.input_adapter is not None
        model.pretrained_graph_transfer_branch = FrozenGraphTransferBranch(
            source_shell.input_adapter,
            source_shell.graph_encoder,
            latent_dim,
            initial_gate=initial_gate,
        ).to(device)
        model.topology_add_transfer_adapter = GatedResidualTransferAdapter(
            latent_dim,
            bottleneck=max(4, latent_dim // 2),
            initial_gate=initial_gate,
        ).to(device)
        model.topology_remove_transfer_adapter = GatedResidualTransferAdapter(
            latent_dim,
            bottleneck=max(4, latent_dim // 2),
            initial_gate=initial_gate,
        ).to(device)
    soft_node_magnitude_conditioning = bool(
        _checkpoint_value(saved, "soft_node_magnitude_conditioning", False)
    )
    change_gated_action_value = bool(
        _checkpoint_value(saved, "change_gated_action_value", False)
    )
    action_full_value_conditioning = bool(
        _checkpoint_value(saved, "action_full_value_conditioning", False)
    )
    action_encoder = GraphActionEncoder(
        latent_dim,
        action_dim,
        node_state_dim=(
            node_feature_dim
            if task == "node_state"
            and (action_full_value_conditioning or soft_node_magnitude_conditioning)
            else None
        ),
        node_property_dim=(
            property_dim
            if task == "node_property"
            and (action_full_value_conditioning or soft_node_magnitude_conditioning)
            else None
        ),
    ).to(device)
    controller = ActionAwareController(
        latent_dim,
        hidden_dim,
        policy_dim=policy_dim,
        node_state_dim=node_feature_dim,
        node_property_dim=property_dim,
        topology_count_context_dim=(
            4
            if bool(_checkpoint_value(saved, "topology_count_context", False))
            else 0
        ),
    ).to(device)




    _load_backward_compatible_state(
        model,
        checkpoint["model_state"],
        optional_missing_prefixes=(
            "edge_addition_decoder.context_linear.",
            "edge_deletion_decoder.context_linear.",
        ),
    )
    _load_backward_compatible_state(
        action_encoder,
        checkpoint["action_encoder_state"],
        optional_missing_prefixes=("topology_count_projection.",),
    )
    _load_backward_compatible_state(
        controller,
        checkpoint["controller_state"],
        optional_missing_prefixes=(
            "topology_count_head.",
            "node_count_head.",
        ),
    )
    model.eval()
    action_encoder.eval()
    controller.eval()
    return model, action_encoder, controller


def _append_capped(
    bucket: list[torch.Tensor], value: torch.Tensor, *, current_rows: int, cap: int
) -> int:
    remaining = max(int(cap) - int(current_rows), 0)
    if remaining:
        selected = value.detach().cpu()[:remaining]
        bucket.append(selected)
        return current_rows + int(selected.shape[0])
    return current_rows


def _score_topology_chunks(
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    candidates: torch.Tensor,
    *,
    pair_features: torch.Tensor | None = None,
    chunk_size: int = 65536,
) -> torch.Tensor:
    scores = []
    for start in range(0, candidates.shape[1], int(chunk_size)):
        pair = candidates[:, start : start + int(chunk_size)]
        context = (
            None
            if pair_features is None
            else pair_features[start : start + int(chunk_size)]
        )
        scores.append(decoder(latent[pair[0]], latent[pair[1]], context))
    return torch.cat(scores) if scores else latent.new_empty(0)


TOPOLOGY_HISTORY_READOUT = {


    "trade": {"add_weight": 1.0, "remove_weight": 3.0, "add_scale": 1.0, "remove_scale": 1.0},
    "un_vote": {"add_weight": 1.0, "remove_weight": 0.25, "add_scale": 1.0, "remove_scale": 1.0},
    "contact": {"add_weight": 0.5, "remove_weight": 0.5, "add_scale": 1.0, "remove_scale": 1.25},
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--processed", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max_eval_transitions", type=int, default=0)
    parser.add_argument(
        "--history_burn_in",
        type=int,
        default=0,
        help=(
            "If positive, skip neural rollout over the old prefix and rebuild "
            "hidden state from only this many immediately preceding snapshots. "
            "Past-only edge/property statistics are still accumulated over "
            "the complete chronological prefix."
        ),
    )
    parser.add_argument(
        "--node_history_blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For T2/T3 localisation reporting, validation-select one past-only "
            "per-node change-propensity residual (previous change, cumulative "
            "rate, or EMA).  This matches the established passive readout and "
            "never reads a current target before scoring it."
        ),
    )
    parser.add_argument(
        "--skip_controller_diagnostics",
        action="store_true",
        help=(
            "Skip Controller-only localization calibration. This is intended "
            "for intermediate validation used solely to select a world-model "
            "checkpoint; it does not change the future-latent decoder metric."
        ),
    )
    parser.add_argument("--max_actions", type=int, default=None)
    parser.add_argument("--min_actions", type=int, default=None)
    parser.add_argument("--stop_threshold", type=float, default=0.5)
    parser.add_argument("--max_addition_candidates", type=int, default=None)
    parser.add_argument(
        "--topology_candidate_policy",
        choices=["all_pairs", "seen_pairs"],
        default=None,
        help="Override the checkpoint topology candidate universe.",
    )
    parser.add_argument("--pair_chunk_size", type=int, default=65536)
    parser.add_argument(
        "--topology_decode_policy",
        choices=["controller_count_topk", "past_history_topk", "latent_count_topk"],
        default="past_history_topk",
    )
    parser.add_argument(
        "--topology_history_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a causal pair-history prior to Controller topology scores. "
            "Only completed source snapshots are used."
        ),
    )
    parser.add_argument("--topology_add_history_weight", type=float, default=None)
    parser.add_argument("--topology_remove_history_weight", type=float, default=None)
    parser.add_argument("--topology_add_count_scale", type=float, default=None)
    parser.add_argument("--topology_remove_count_scale", type=float, default=None)
    parser.add_argument(
        "--topology_count_history_blend",
        type=float,
        default=0.0,
        help=(
            "Optional causal blend for Controller cardinality decoding. "
            "A value of 0 keeps the learned count; a value of 1 uses the "
            "mean of the last eight released add/remove counts."
        ),
    )
    parser.add_argument(
        "--topology_controller_score_weight",
        type=float,
        default=0.0,
        help="Validation-selected weight for future-free Controller pair logits.",
    )
    parser.add_argument(
        "--action_residual_scale_override",
        type=float,
        default=None,
        help=(
            "Diagnostic-only override for the saved bounded action residual "
            "scale. The checkpoint is never modified."
        ),
    )



    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results", default="results/action_rl_future_free_eval.json")
    args = parser.parse_args(argv)
    if args.max_eval_transitions < 0:
        parser.error("--max_eval_transitions must be non-negative.")
    if args.history_burn_in < 0:
        parser.error("--history_burn_in must be non-negative.")
    if not 0.0 <= args.stop_threshold <= 1.0:
        parser.error("--stop_threshold must lie in [0, 1].")
    if args.pair_chunk_size < 1:
        parser.error("--pair_chunk_size must be positive.")
    if not 0.0 <= args.topology_count_history_blend <= 1.0:
        parser.error("--topology_count_history_blend must lie in [0,1].")
    if args.topology_controller_score_weight < 0.0:
        parser.error("--topology_controller_score_weight must be non-negative.")
    if (
        args.action_residual_scale_override is not None
        and args.action_residual_scale_override < 0.0
    ):
        parser.error("--action_residual_scale_override must be non-negative.")

    device = resolve_device(args.device)
    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint["args"]
    task = str(saved["task"])
    if task not in TASK_OPERATIONS:
        raise ValueError(f"Checkpoint has unsupported task={task!r}.")
    seed = int(args.seed if args.seed is not None else saved.get("seed", 1))
    seed_everything(seed)
    dataset_name = str(
        args.dataset
        or saved.get("dataset")
        or ACTION_BENCHMARK_DATASETS[task][0]
    )
    topology_readout = dict(TOPOLOGY_HISTORY_READOUT.get(dataset_name, {}))
    for key, override in (
        ("add_weight", args.topology_add_history_weight),
        ("remove_weight", args.topology_remove_history_weight),
        ("add_scale", args.topology_add_count_scale),
        ("remove_scale", args.topology_remove_count_scale),
    ):
        if override is not None:
            if override < 0.0:
                parser.error("Topology history weights/count scales must be non-negative.")
            topology_readout[key] = float(override)







    if (
        task == "topology"
        and dataset_name == "trade"
        and bool(_checkpoint_value(saved, "topology_pair_state", False))
        and args.topology_remove_history_weight is None
    ):
        topology_readout["remove_weight"] = 0.0
    processed = resolve_action_benchmark_path(
        ROOT,
        task,
        dataset_name,
        args.processed or saved.get("processed"),
    )
    if (args.processed or saved.get("processed")) and not (args.dataset or saved.get("dataset")):
        dataset_name = infer_action_benchmark_dataset(task, processed)
    edge_input = str(_checkpoint_value(saved, "edge_input", "binary"))
    metadata, transitions = _graph_transitions(
        processed,
        task,
        split=None,
        dataset=dataset_name,
        edge_input=edge_input,
    )
    if not transitions:
        raise RuntimeError("No chronological transitions are available.")
    model, action_encoder, controller = _build_components(
        checkpoint, metadata, task=task, device=device
    )
    if args.action_residual_scale_override is not None:
        state_model = model.state_model
        gate = getattr(state_model, "action_residual_gate", None)
        if gate is None or not bool(getattr(state_model, "bounded_action_residual", False)):
            raise ValueError(
                "--action_residual_scale_override requires a bounded action-residual checkpoint."
            )
        scale = float(args.action_residual_scale_override)
        maximum = float(getattr(state_model, "action_residual_max_scale", 0.25))
        if scale >= maximum:
            raise ValueError(
                "--action_residual_scale_override must be smaller than the "
                f"checkpoint residual cap ({maximum:g})."
            )
        raw_gate = float(np.arctanh(scale / maximum)) if scale > 0.0 else 0.0
        with torch.no_grad():
            gate.copy_(torch.tensor(raw_gate, device=gate.device, dtype=gate.dtype))
    action_conditioning = True
    soft_node_action_conditioning = bool(
        _checkpoint_value(saved, "soft_node_action_conditioning", False)
    )
    soft_node_action_mode = str(
        _checkpoint_value(saved, "soft_node_action_mode", "expected")
    )
    soft_node_action_probability_calibration = str(
        _checkpoint_value(
            saved, "soft_node_action_probability_calibration", "none"
        )
    )
    soft_node_action_prior_smoothing = float(
        _checkpoint_value(saved, "soft_node_action_prior_smoothing", 1.0)
    )
    soft_node_action_topk = int(
        _checkpoint_value(saved, "soft_node_action_topk", 0)
    )
    soft_node_action_count_support = bool(
        _checkpoint_value(saved, "soft_node_action_count_support", False)
    )
    soft_edge_action_conditioning = bool(
        _checkpoint_value(saved, "soft_edge_action_conditioning", False)
    )
    soft_edge_action_topk = int(
        _checkpoint_value(saved, "soft_edge_action_topk", 0)
    )
    soft_edge_action_count_support = bool(
        _checkpoint_value(saved, "soft_edge_action_count_support", False)
    )
    soft_edge_action_mode = str(
        _checkpoint_value(saved, "soft_edge_action_mode", "categorical")
    )
    controller_topology_count_weight = float(
        _checkpoint_value(saved, "controller_topology_count_weight", 0.0)
    )
    controller_node_count_weight = float(
        _checkpoint_value(saved, "controller_node_count_weight", 0.0)
    )
    confidence_gated_soft_action = bool(
        _checkpoint_value(saved, "confidence_gated_soft_action", False)
    )
    soft_node_magnitude_conditioning = bool(
        _checkpoint_value(saved, "soft_node_magnitude_conditioning", False)
    )
    change_gated_action_value = bool(
        _checkpoint_value(saved, "change_gated_action_value", False)
    )

    num_nodes = int(metadata["num_nodes"])
    min_actions = int(
        args.min_actions
        if args.min_actions is not None
        else _checkpoint_value(saved, "min_actions", 1)
    )
    max_actions = int(
        args.max_actions
        if args.max_actions is not None
        else _checkpoint_value(saved, "max_actions", 8)
    )
    if min_actions < 1 or max_actions < min_actions:
        parser.error("Require 1 <= min_actions <= max_actions.")
    max_addition_candidates = int(
        args.max_addition_candidates
        if args.max_addition_candidates is not None
        else 0
    )
    topology_candidate_policy = str(
        args.topology_candidate_policy
        or _checkpoint_value(saved, "topology_candidate_policy", "all_pairs")
    )
    property_mode = str(_checkpoint_value(saved, "property_mode", "logit_residual"))
    if property_mode == "auto":
        property_mode = "logit_residual"
    property_change_threshold = _checkpoint_value(
        saved, "property_change_threshold_train_q75", None
    )
    exact_reward_weight = float(
        _checkpoint_value(saved, "topology_exact_reward_weight", 0.5)
    )
    hidden = model.initial_hidden(num_nodes, device)
    historical_change_counts = torch.zeros(num_nodes, device=device)
    historical_change_observations = torch.zeros((), device=device)
    generator = torch.Generator().manual_seed(seed)
    records: list[dict[str, Any]] = []
    scored_transition_count = 0




    soft_action_masses: list[float] = []
    soft_action_support_counts: list[int] = []
    topology_pair_count = (
        torch.zeros(num_nodes * num_nodes, device=device)
        if task == "topology"
        else None
    )
    topology_pair_state = (
        CausalPairState(num_nodes)
        if task == "topology"
        and (
            bool(_checkpoint_value(saved, "topology_pair_state", False))
            or topology_candidate_policy == "seen_pairs"
        )
        else None
    )
    topology_add_count_history: list[int] = []
    topology_remove_count_history: list[int] = []
    controller_topology_count_predictions: list[torch.Tensor] = []
    controller_topology_count_targets: list[torch.Tensor] = []
    topology_formal = {
        "addition": SetMetricAccumulator(),
        "deletion": SetMetricAccumulator(),
        "reconstruction": SetMetricAccumulator(),
    }
    topology_reactivation = SetMetricAccumulator()
    topology_reactivation_target_count = 0
    topology_novel_target_count = 0
    topology_all_target_count = 0
    topology_persistence = {
        "addition": SetMetricAccumulator(),
        "deletion": SetMetricAccumulator(),
        "reconstruction": SetMetricAccumulator(),
    }
    topology_frequency = {
        "addition": SetMetricAccumulator(),
        "deletion": SetMetricAccumulator(),
        "reconstruction": SetMetricAccumulator(),
    }
    topology_random = {
        "addition": SetMetricAccumulator(),
        "deletion": SetMetricAccumulator(),
        "reconstruction": SetMetricAccumulator(),
    }
    node_change_buckets: dict[str, dict[str, Any]] = {
        split: {
            "labels": [],
            "logits": [],
            "controller_logits": [],
            "history": [],
            "node_history": {},



            "block_sizes": [],
            "past_positive_rates": [],
        }
        for split in ("val", "test")
    }
    state_all = RegressionAccumulator()
    state_changed = RegressionAccumulator()
    state_copy_all = RegressionAccumulator()
    state_copy_changed = RegressionAccumulator()
    state_rows: dict[str, dict[str, list[torch.Tensor]]] = {
        split: {
            key: []
            for key in (
                "logits",
                "raw",
                "target",
                "labels",
                "controller_mu",
            )
        }
        for split in ("val", "test")
    }
    history_setting = str(_checkpoint_value(saved, "property_history_input", "auto"))
    property_history_input = task == "node_property" and (
        history_setting in {"on", "True", "true"}
        or (history_setting == "auto" and dataset_name in {"genre", "reddit"})
    )
    property_graph_history: GenreObservedPropertyHistory | None = None
    if property_history_input:
        coordinate_nodes = metadata.get(
            "num_genre_coordinate_nodes",
            metadata.get("num_target_coordinate_nodes"),
        )
        if coordinate_nodes is None:
            raise ValueError("Property-history input requires coordinate-node metadata.")
        property_graph_history = GenreObservedPropertyHistory(
            num_nodes,
            int(metadata["official_target_dim"]),
            num_genre_coordinate_nodes=int(coordinate_nodes),
        )
    property_model_names = ("gwm", "future_latent_only", "copy_last")
    property_metrics = {
        name: PropertyMetricAccumulator(int(metadata.get("official_target_dim", 1)))
        for name in property_model_names
    }
    property_changed = {
        name: RegressionAccumulator()
        for name in property_model_names
    }
    property_large = {
        name: RegressionAccumulator()
        for name in property_model_names
    }
    train_property_magnitudes: list[torch.Tensor] = []
    node_change_history = _new_node_change_history(num_nodes)

    model_start_index = 0
    if args.history_burn_in:



        first_model_split = (
            "val"
            if args.split == "test"
            and task in {"node_change", "node_state", "node_property"}
            else args.split
        )
        first_target_index = next(
            (
                index
                for index in range(len(transitions))
                if str(transitions[index]["split"]) == first_model_split
            ),
            0,
        )
        model_start_index = max(
            0, int(first_target_index) - int(args.history_burn_in)
        )




    if model_start_index:
        prefix_change_counts = torch.zeros(num_nodes, dtype=torch.float32)
        prefix_change_observations = torch.zeros((), dtype=torch.float32)
        prefix_pair_count: torch.Tensor | None = None
        snapshots = getattr(transitions.base, "snapshots", None)
        if task != "node_property" and snapshots is not None:
            prefix_snapshots = snapshots[:model_start_index]
            if task == "topology":
                pair_chunks = [
                    snapshot["edge_index"][0].long() * num_nodes
                    + snapshot["edge_index"][1].long()
                    for snapshot in prefix_snapshots
                    if snapshot["edge_index"].numel()
                ]
                prefix_pair_count = torch.bincount(
                    torch.cat(pair_chunks) if pair_chunks else torch.empty(0, dtype=torch.long),
                    minlength=num_nodes * num_nodes,
                ).to(torch.float32)




                topology_add_count_history.extend(
                    int(snapshot["edge_added"].shape[1])
                    for snapshot in prefix_snapshots
                )
                topology_remove_count_history.extend(
                    int(snapshot["edge_removed"].shape[1])
                    for snapshot in prefix_snapshots
                )
                if topology_pair_state is not None:
                    for prefix_index, snapshot in enumerate(prefix_snapshots):
                        topology_pair_state.observe(
                            snapshot["edge_index"], time_index=prefix_index
                        )
            else:
                prefix_change_counts.copy_(
                    torch.stack(
                        [snapshot["node_changed"].to(torch.float32) for snapshot in prefix_snapshots]
                    ).sum(dim=0)
                )
                prefix_change_observations.fill_(
                    float(model_start_index * num_nodes)
                )
        else:
            property_graph_cache_loaded = False
            property_graph_cache_path: Path | None = None
            if property_graph_history is not None:
                property_graph_cache_path = (
                    processed.parent
                    / ".action_history_cache"
                    / f"{processed.stem}_prefix{model_start_index}.pt"
                )
                if property_graph_cache_path.exists():
                    cached_history = torch.load(
                        property_graph_cache_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    property_graph_history.load_state_dict(cached_history)
                    property_graph_cache_loaded = True
                elif hasattr(transitions.base, "snapshots"):




                    prefix_property_snapshots = transitions.base.snapshots[
                        :model_start_index
                    ]
                    for chunk_start in range(0, len(prefix_property_snapshots), 16):
                        chunk = prefix_property_snapshots[
                            chunk_start : chunk_start + 16
                        ]
                        ids = torch.cat(
                            [snapshot["property_node_ids"].long() for snapshot in chunk]
                        )
                        values = torch.cat(
                            [snapshot["property_target"].float() for snapshot in chunk]
                        )
                        if ids.numel():
                            property_graph_history.running_sum.index_add_(
                                0, ids, values
                            )
                            property_graph_history.running_count.index_add_(
                                0,
                                ids,
                                torch.ones(ids.numel(), dtype=torch.float32),
                            )
                            property_graph_history.global_sum.add_(
                                values.sum(dim=0)
                            )
                            property_graph_history.global_count += int(ids.numel())
                    property_graph_cache_loaded = True
                    property_graph_cache_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    torch.save(
                        property_graph_history.state_dict(),
                        property_graph_cache_path,
                    )



            for prefix_index in range(model_start_index):


                prefix = (
                    transitions.base.property_transition(prefix_index)
                    if hasattr(transitions.base, "property_transition")
                    else transitions[prefix_index]
                )
                if property_graph_history is not None and not property_graph_cache_loaded:
                    property_graph_history.observe(
                        prefix["property_node_ids_t"],
                        prefix["property_observed_t"],
                    )
                prefix_change_counts.add_(
                    _past_change_update(
                        task,
                        prefix,
                        torch.device("cpu"),
                        num_nodes=num_nodes,
                    )
                )
                prefix_change_observations.add_(
                    _past_change_observation_count(
                        task, prefix, torch.device("cpu")
                    )
                )
                if str(prefix["split"]) == "train":
                    comparable_prefix = prefix[
                        "property_current_observed_mask"
                    ].to(torch.bool)
                    if bool(comparable_prefix.any()):
                        train_property_magnitudes.append(
                            (
                                prefix["property_target"][comparable_prefix]
                                - prefix["property_current_target"][comparable_prefix]
                            )
                            .abs()
                            .mean(dim=-1)
                            .cpu()
                        )
        historical_change_counts.copy_(prefix_change_counts.to(device))
        historical_change_observations.copy_(
            prefix_change_observations.to(device)
        )
        if topology_pair_count is not None and prefix_pair_count is not None:
            topology_pair_count.copy_(prefix_pair_count.to(device))
        if topology_pair_state is not None:
            for prefix_index in range(model_start_index):
                topology_pair_state.observe(
                    transitions[prefix_index]["edge_index_t"],
                    time_index=prefix_index,
                )






    if task in {"node_change", "node_state"} and model_start_index:
        for prefix_index in range(model_start_index):
            _observe_node_change_history(
                node_change_history,
                transitions[prefix_index]["node_changed"],
            )








    model_end_index = len(transitions)
    if args.split == "val":
        validation_indices = [
            index
            for index, transition in enumerate(transitions)
            if str(transition["split"]) == "val"
        ]
        if not validation_indices:
            raise RuntimeError("No validation transitions are available.")
        model_end_index = validation_indices[-1] + 1

    with torch.inference_mode():
        for chronological_index in range(model_start_index, model_end_index):
            transition_cpu = transitions[chronological_index]
            if property_graph_history is not None:
                property_graph_history.observe(
                    transition_cpu["property_node_ids_t"],
                    transition_cpu["property_observed_t"],
                )
            transition = _transition_to_device(
                transition_cpu, device, task=task
            )
            if property_graph_history is not None:
                transition["x_t"] = torch.cat(
                    [transition["x_t"], property_graph_history.state().to(device)],
                    dim=-1,
                )
            x_t = transition["x_t"]
            edge_index_t = transition["edge_index_t"]
            topology_count_context = (
                _topology_count_context(
                    edge_index_t,
                    num_nodes=num_nodes,
                    dtype=x_t.dtype,
                )
                if bool(_checkpoint_value(saved, "topology_count_context", False))
                else None
            )
            current_edge_count = max(int(edge_index_t.shape[1]), 1)
            edge_weight_t = _edge_weight(transition, edge_input)
            if edge_weight_t is not None:
                edge_weight_t = edge_weight_t.to(device)
            if topology_pair_count is not None:
                current_pair_codes = torch.unique(
                    edge_index_t[0].long() * num_nodes + edge_index_t[1].long()
                )
                topology_pair_count.index_add_(
                    0,
                    current_pair_codes,
                    torch.ones_like(current_pair_codes, dtype=torch.float32),
                )

            candidate_edges: dict[GraphOperation, torch.Tensor] = {}
            candidate_nodes: dict[GraphOperation, torch.Tensor] = {}
            if task == "topology":
                candidate_edges = {
                    operation: values.to(device)
                    for operation, values in _topology_candidates(
                        edge_index_t,
                        num_nodes=num_nodes,
                        metadata=metadata,
                        max_addition_candidates=max_addition_candidates,
                        generator=generator,
                        candidate_policy=topology_candidate_policy,
                        seen_codes=(
                            topology_pair_state.observed_codes()
                            if topology_candidate_policy == "seen_pairs"
                            and topology_pair_state is not None
                            else None
                        ),
                    ).items()
                }
            elif task == "node_property":




                candidate_nodes[GraphOperation.MODIFY_NODE_PROPERTY] = torch.unique(
                    transition_cpu["property_node_ids_t"].to(
                        device=device, dtype=torch.long
                    )
                )
            topology_pair_context = (
                {
                    GraphOperation.ADD_EDGE: topology_pair_state.features(
                        candidate_edges[GraphOperation.ADD_EDGE],
                        time_index=chronological_index,
                        currently_present=False,
                    ),
                    GraphOperation.REMOVE_EDGE: topology_pair_state.features(
                        candidate_edges[GraphOperation.REMOVE_EDGE],
                        time_index=chronological_index,
                        currently_present=True,
                    ),
                }
                if task == "topology" and topology_pair_state is not None
                else {}
            )











            z_policy_raw = model.encode_observed_graph(
                x_t, edge_index_t, edge_weight_t
            )
            z_policy = model._normalize_latent(z_policy_raw)


            hidden_for_action = hidden
            controller_node_state = controller.node_state(z_policy, hidden_for_action)
            controller_target_logits = controller.node_target_head(
                controller_node_state
            ).squeeze(-1)
            controller_target_mu = None
            soft_action_mass: float | None = None
            soft_action_support_count: int | None = None
            if (
                task in {"node_state", "node_property"}
                and soft_node_magnitude_conditioning
            ):
                magnitude_operation = (
                    GraphOperation.MODIFY_NODE_PROPERTY
                    if task == "node_property"
                    else GraphOperation.MODIFY_NODE_STATE
                )
                controller_target_mu = controller._magnitude_distribution(
                    controller_node_state,
                    magnitude_operation,
                )[0]






            uses_soft_unary_action = bool(
                task in {"node_change", "node_state", "node_property"}
                and action_conditioning
                and soft_node_action_conditioning
            )
            selected = None
            if task == "topology" or not uses_soft_unary_action:


                selected = controller.greedy_action_sequence(
                    z_policy,
                    hidden_for_action,
                    candidate_edges=candidate_edges,
                    candidate_nodes=candidate_nodes,
                    valid_operations=TASK_OPERATIONS[task],
                    min_actions=min_actions,
                    max_actions=max_actions,
                    predict_magnitude=task not in {"topology", "node_change"},
                    stop_threshold=args.stop_threshold,
                    edge_chunk_size=args.pair_chunk_size,
                )
            if (
                task == "topology"
                and action_conditioning
                and soft_edge_action_conditioning
            ):
                controller_log_counts = (
                    controller.topology_log_counts(
                        controller_node_state,
                        context=topology_count_context,
                    )
                    if controller_topology_count_weight > 0.0
                    else None
                )
                if (
                    controller_log_counts is not None
                    and str(
                        _checkpoint_value(
                            saved, "topology_count_target_mode", "absolute"
                        )
                    )
                    == "edge_rate"
                ):
                    controller_count = torch.expm1(
                        controller_log_counts.detach().clamp(0.0, 12.0)
                    ) * float(current_edge_count)
                    controller_log_counts = torch.log1p(
                        controller_count
                    ).clamp(0.0, 12.0)
                controller_support_count_by_operation = None
                if (
                    soft_edge_action_count_support
                    and controller_log_counts is not None
                ):
                    controller_count = torch.expm1(
                        controller_log_counts.detach().clamp(0.0, 12.0)
                    ).round().to(torch.long)
                    controller_support_count_by_operation = {
                        GraphOperation.ADD_EDGE: min(
                            max(int(controller_count[0].item()), 0),
                            int(candidate_edges[GraphOperation.ADD_EDGE].shape[1]),
                        ),
                        GraphOperation.REMOVE_EDGE: min(
                            max(int(controller_count[1].item()), 0),
                            int(candidate_edges[GraphOperation.REMOVE_EDGE].shape[1]),
                        ),
                    }
                controller_edge_probability = (
                    controller.soft_edge_action_probabilities(
                        controller_node_state,
                        candidate_edges,
                        chunk_size=args.pair_chunk_size,
                        top_k_per_operation=soft_edge_action_topk,
                        support_count_by_operation=controller_support_count_by_operation,
                        mode=soft_edge_action_mode,
                    )
                )
                encoded_action_details = action_encoder.encode_soft_edge_actions(
                    z_policy,
                    candidate_edges,
                    controller_edge_probability,
                    chunk_size=args.pair_chunk_size,
                    confidence_gate=confidence_gated_soft_action,
                    operation_log_counts=(
                        {
                            GraphOperation.ADD_EDGE: controller_log_counts[0],
                            GraphOperation.REMOVE_EDGE: controller_log_counts[1],
                        }
                        if controller_log_counts is not None
                        else None
                    ),
                )
                encoded_action = encoded_action_details["nodewise"]
            elif (
                task in {"node_change", "node_state", "node_property"}
                and action_conditioning
                and soft_node_action_conditioning
            ):
                soft_operation = (
                    GraphOperation.MODIFY_NODE_PROPERTY
                    if task == "node_property"
                    else GraphOperation.MODIFY_NODE_STATE
                )
                soft_probability = calibrated_soft_node_action_probability(
                    controller_target_logits,
                    historical_positive_count=historical_change_counts.sum(),
                    historical_observation_count=historical_change_observations,
                    mode=soft_node_action_probability_calibration,
                    smoothing=soft_node_action_prior_smoothing,
                )
                if task == "node_property":
                    observed_nodes = candidate_nodes.get(soft_operation)
                    if observed_nodes is not None:
                        observed_mask = torch.zeros_like(soft_probability)
                        observed_mask[observed_nodes] = 1.0
                        soft_probability = soft_probability * observed_mask
                controller_node_support_count = None
                if (
                    soft_node_action_count_support
                    and controller_node_count_weight > 0.0
                ):
                    controller_count = torch.expm1(
                        controller.node_log_count(controller_node_state)
                        .detach()
                        .clamp(0.0, 12.0)
                    ).round().to(torch.long)
                    eligible_nodes = candidate_nodes.get(soft_operation)
                    maximum_count = (
                        int(eligible_nodes.numel())
                        if eligible_nodes is not None
                        else int(soft_probability.numel())
                    )
                    controller_node_support_count = min(
                        max(int(controller_count.item()), 0), maximum_count
                    )
                encoded_action_details = action_encoder.encode_soft_node_action(
                    z_policy,
                    soft_probability,
                    soft_operation,
                    node_value=(
                        controller_target_mu
                        if task in {"node_state", "node_property"}
                        and soft_node_magnitude_conditioning
                        else None
                    ),
                    change_gated_value=change_gated_action_value,
                    confidence_gate=confidence_gated_soft_action,
                    localization_mode=soft_node_action_mode,
                    top_k=soft_node_action_topk,
                    support_count=controller_node_support_count,
                )
                encoded_action = encoded_action_details["nodewise"]
                soft_action_mass = float(
                    encoded_action_details["action_mass"].detach().cpu()
                )
                support = encoded_action_details["support_count"]
                soft_action_support_count = (
                    None if support is None else int(support)
                )
            else:
                assert selected is not None
                encoded_action = _encode_action_or_zero(
                    action_encoder,
                    z_policy,
                    selected.action,
                    enabled=action_conditioning,
                )
            outputs = model(
                x_t,
                edge_index_t,
                hidden,
                action=encoded_action,
                edge_weight_t=edge_weight_t,
                addition_candidate_edges=None,
                deletion_candidate_edges=None,
                decode_node_state=task != "topology",
                commit_history=True,
                precomputed_z_t_raw=z_policy_raw,
            )
            if task == "topology":
                assert model.edge_addition_decoder is not None
                assert model.edge_deletion_decoder is not None
                outputs["edge_addition_logits"] = _score_topology_chunks(
                    model.edge_addition_decoder,
                    outputs["latent_next"],
                    candidate_edges[GraphOperation.ADD_EDGE],
                    pair_features=topology_pair_context.get(GraphOperation.ADD_EDGE),
                    chunk_size=args.pair_chunk_size,
                )
                outputs["edge_deletion_logits"] = _score_topology_chunks(
                    model.edge_deletion_decoder,
                    outputs["latent_next"],
                    candidate_edges[GraphOperation.REMOVE_EDGE],
                    pair_features=topology_pair_context.get(GraphOperation.REMOVE_EDGE),
                    chunk_size=args.pair_chunk_size,
                )
            hidden = outputs["hidden_next"].detach()

            target_split = str(transition["split"])
            if task == "topology" and target_split == args.split:
                add_candidates = candidate_edges[GraphOperation.ADD_EDGE]
                remove_candidates = candidate_edges[GraphOperation.REMOVE_EDGE]
                add_logits = outputs["edge_addition_logits"]
                remove_logits = outputs["edge_deletion_logits"]
                if args.topology_controller_score_weight > 0.0:
                    add_logits = add_logits + float(
                        args.topology_controller_score_weight
                    ) * controller.edge_logits_chunked(
                        controller_node_state,
                        add_candidates,
                        chunk_size=args.pair_chunk_size,
                    )
                    remove_logits = remove_logits + float(
                        args.topology_controller_score_weight
                    ) * controller.edge_logits_chunked(
                        controller_node_state,
                        remove_candidates,
                        chunk_size=args.pair_chunk_size,
                    )
                assert selected is not None
                add_count = selected.action.operation_count(GraphOperation.ADD_EDGE)
                remove_count = selected.action.operation_count(GraphOperation.REMOVE_EDGE)
                if args.topology_decode_policy == "controller_count_topk":
                    if controller_topology_count_weight <= 0.0:
                        raise ValueError(
                            "controller_count_topk requires a checkpoint trained "
                            "with --controller_topology_count_weight > 0."
                        )


                    predicted_counts = controller.topology_edit_counts(
                        controller_node_state,
                        context=topology_count_context,
                    ).detach().float()
                    if str(
                        _checkpoint_value(
                            saved, "topology_count_target_mode", "absolute"
                        )
                    ) == "edge_rate":
                        predicted_counts = predicted_counts * float(current_edge_count)






                    add_scale = float(
                        args.topology_add_count_scale
                        if args.topology_add_count_scale is not None
                        else 1.0
                    )
                    remove_scale = float(
                        args.topology_remove_count_scale
                        if args.topology_remove_count_scale is not None
                        else 1.0
                    )
                    add_count = round(add_scale * float(predicted_counts[0]))
                    remove_count = round(remove_scale * float(predicted_counts[1]))
                    if args.topology_history_prior and topology_pair_count is not None:
                        add_codes_tensor = (
                            add_candidates[0] * num_nodes + add_candidates[1]
                        )
                        remove_codes_tensor = (
                            remove_candidates[0] * num_nodes + remove_candidates[1]
                        )
                        add_logits = add_logits + float(
                            topology_readout.get("add_weight", 0.0)
                        ) * torch.log1p(
                            topology_pair_count.index_select(0, add_codes_tensor)
                        )
                        remove_logits = remove_logits - float(
                            topology_readout.get("remove_weight", 0.0)
                        ) * torch.log1p(
                            topology_pair_count.index_select(0, remove_codes_tensor)
                        )






                    history_blend = float(args.topology_count_history_blend)
                    if history_blend > 0.0:
                        if topology_add_count_history:
                            history_add = float(
                                np.mean(topology_add_count_history[-8:])
                            )
                            add_count = round(
                                (1.0 - history_blend) * add_count
                                + history_blend * history_add
                            )
                        if topology_remove_count_history:
                            history_remove = float(
                                np.mean(topology_remove_count_history[-8:])
                            )
                            remove_count = round(
                                (1.0 - history_blend) * remove_count
                                + history_blend * history_remove
                            )
                    controller_topology_count_predictions.append(predicted_counts.cpu())
                    controller_topology_count_targets.append(
                        torch.tensor(
                            [
                                float(transition["edge_added"].shape[1]),
                                float(transition["edge_removed"].shape[1]),
                            ]
                        )
                    )
                if args.topology_decode_policy in {"past_history_topk", "latent_count_topk"}:
                    assert topology_pair_count is not None
                    readout = topology_readout
                    add_codes_tensor = add_candidates[0] * num_nodes + add_candidates[1]
                    remove_codes_tensor = (
                        remove_candidates[0] * num_nodes + remove_candidates[1]
                    )
                    add_logits = add_logits + float(readout["add_weight"]) * torch.log1p(
                        topology_pair_count.index_select(0, add_codes_tensor)
                    )
                    remove_logits = remove_logits - float(
                        readout["remove_weight"]
                    ) * torch.log1p(
                        topology_pair_count.index_select(0, remove_codes_tensor)
                    )
                    if args.topology_decode_policy == "latent_count_topk":
                        if "topology_count_log1p" not in outputs:
                            raise ValueError(
                                "latent_count_topk requires a trained topology count decoder."
                            )
                        predicted_counts = torch.expm1(
                            outputs["topology_count_log1p"].detach().float()
                        ).clamp_min(0.0)
                        add_count = round(
                            float(readout["add_scale"]) * float(predicted_counts[0])
                        )
                        remove_count = round(
                            float(readout["remove_scale"]) * float(predicted_counts[1])
                        )
                    else:
                        if topology_add_count_history:
                            add_count = round(
                                float(readout["add_scale"])
                                * float(np.mean(topology_add_count_history[-8:]))
                            )
                        if topology_remove_count_history:
                            remove_count = round(
                                float(readout["remove_scale"])
                                * float(np.mean(topology_remove_count_history[-8:]))
                            )
                predicted_add = _topk_edges(
                    add_logits,
                    add_candidates,
                    add_count,
                )
                predicted_remove = _topk_edges(
                    remove_logits,
                    remove_candidates,
                    remove_count,
                )
                predicted_add_codes = edge_codes(predicted_add, num_nodes)
                predicted_remove_codes = edge_codes(predicted_remove, num_nodes)
                target_add_codes_all = edge_codes(
                    transition["edge_added"], num_nodes
                )
                seen_before: set[int] = set()
                if topology_candidate_policy == "seen_pairs":
                    seen_before = set(
                        topology_pair_state.observed_codes().tolist()
                        if topology_pair_state is not None
                        else []
                    )
                    target_add_codes = {
                        int(code)
                        for code in target_add_codes_all
                        if int(code) in seen_before
                    }
                else:
                    target_add_codes = target_add_codes_all
                target_remove_codes = edge_codes(transition["edge_removed"], num_nodes)
                current_codes = edge_codes(edge_index_t, num_nodes)
                predicted_future = (
                    current_codes - predicted_remove_codes
                ) | predicted_add_codes
                target_future = (
                    current_codes - target_remove_codes
                ) | target_add_codes_all
                topology_formal["addition"].update(
                    predicted_add_codes, target_add_codes
                )
                topology_formal["deletion"].update(
                    predicted_remove_codes, target_remove_codes
                )
                topology_formal["reconstruction"].update(
                    predicted_future, target_future
                )
                if topology_candidate_policy == "seen_pairs":
                    reactivation_target = target_add_codes
                    novel_target = {
                        int(code)
                        for code in target_add_codes_all
                        if int(code) not in seen_before
                    }
                    topology_reactivation.update(
                        predicted_add_codes, reactivation_target
                    )
                    topology_reactivation_target_count += int(
                        len(reactivation_target)
                    )
                    topology_novel_target_count += int(len(novel_target))
                    topology_all_target_count += int(len(target_add_codes_all))
                topology_persistence["addition"].update(set(), target_add_codes)
                topology_persistence["deletion"].update(set(), target_remove_codes)
                topology_persistence["reconstruction"].update(
                    current_codes, target_future
                )
                assert topology_pair_count is not None
                history_add_count = (
                    round(float(np.mean(topology_add_count_history[-8:])))
                    if topology_add_count_history
                    else 0
                )
                history_remove_count = (
                    round(float(np.mean(topology_remove_count_history[-8:])))
                    if topology_remove_count_history
                    else 0
                )
                frequency_add = _topk_edges(
                    torch.log1p(
                        topology_pair_count.index_select(
                            0, add_candidates[0] * num_nodes + add_candidates[1]
                        )
                    ),
                    add_candidates,
                    history_add_count,
                )
                frequency_remove = _topk_edges(
                    -torch.log1p(
                        topology_pair_count.index_select(
                            0,
                            remove_candidates[0] * num_nodes
                            + remove_candidates[1],
                        )
                    ),
                    remove_candidates,
                    history_remove_count,
                )
                frequency_add_codes = edge_codes(frequency_add, num_nodes)
                frequency_remove_codes = edge_codes(frequency_remove, num_nodes)
                topology_frequency["addition"].update(
                    frequency_add_codes, target_add_codes
                )
                topology_frequency["deletion"].update(
                    frequency_remove_codes, target_remove_codes
                )
                topology_frequency["reconstruction"].update(
                    (current_codes - frequency_remove_codes) | frequency_add_codes,
                    target_future,
                )





                random_add_count = min(
                    max(int(history_add_count), 0),
                    int(add_candidates.shape[1]),
                )
                random_remove_count = min(
                    max(int(history_remove_count), 0),
                    int(remove_candidates.shape[1]),
                )
                if random_add_count:
                    random_add_indices = torch.randperm(
                        int(add_candidates.shape[1]),
                        generator=generator,
                        device="cpu",
                    )[:random_add_count].to(add_candidates.device)
                    random_add = add_candidates.index_select(1, random_add_indices)
                else:
                    random_add = add_candidates[:, :0]
                if random_remove_count:
                    random_remove_indices = torch.randperm(
                        int(remove_candidates.shape[1]),
                        generator=generator,
                        device="cpu",
                    )[:random_remove_count].to(remove_candidates.device)
                    random_remove = remove_candidates.index_select(
                        1, random_remove_indices
                    )
                else:
                    random_remove = remove_candidates[:, :0]
                random_add_codes = edge_codes(random_add, num_nodes)
                random_remove_codes = edge_codes(random_remove, num_nodes)
                topology_random["addition"].update(
                    random_add_codes, target_add_codes
                )
                topology_random["deletion"].update(
                    random_remove_codes, target_remove_codes
                )
                topology_random["reconstruction"].update(
                    (current_codes - random_remove_codes) | random_add_codes,
                    target_future,
                )

            if task in {"node_change", "node_state"} and target_split in {"val", "test"}:
                bucket = node_change_buckets[target_split]
                bucket["labels"].append(transition["node_changed"].detach().cpu())
                bucket["logits"].append(outputs["node_change_logits"].detach().cpu())
                bucket["controller_logits"].append(
                    controller_target_logits.detach().cpu()
                )
                bucket["history"].append(
                    (
                        historical_change_counts
                        / max(int(chronological_index), 1)
                    ).detach().cpu()
                )
                past_observations = historical_change_observations.clamp_min(1.0)
                bucket["block_sizes"].append(int(transition["node_changed"].numel()))
                bucket["past_positive_rates"].append(
                    float((historical_change_counts.sum() / past_observations).item())
                )
                if args.node_history_blend:
                    for name, value in _node_change_history_features(
                        node_change_history
                    ).items():
                        bucket["node_history"].setdefault(name, []).append(
                            value.detach().cpu()
                        )

            if task == "node_state" and target_split == args.split:
                target_delta = transition["node_delta"]




                prediction_delta = outputs["node_delta"]
                state_all.update(prediction_delta, target_delta)
                state_copy_all.update(torch.zeros_like(target_delta), target_delta)
                changed = transition["node_changed"].to(torch.bool)
                if bool(changed.any()):
                    state_changed.update(
                        prediction_delta[changed], target_delta[changed]
                    )
                    state_copy_changed.update(
                        torch.zeros_like(target_delta[changed]), target_delta[changed]
                    )
            if task == "node_state" and target_split in {"val", "test"}:
                state_rows[target_split]["logits"].append(
                    outputs["node_change_logits"].detach().cpu()
                )
                state_rows[target_split]["raw"].append(
                    outputs["node_delta_ungated"].detach().cpu()
                )
                state_rows[target_split]["target"].append(
                    transition["node_delta"].detach().cpu()
                )
                state_rows[target_split]["labels"].append(
                    transition["node_changed"].detach().cpu()
                )
                if controller_target_mu is not None:
                    state_rows[target_split]["controller_mu"].append(
                        controller_target_mu.detach().cpu()
                    )

            if task == "node_property":
                target_ids = transition["property_node_ids"].long()
                target_rows = transition["property_target"].float()
                current_rows = transition["property_current_target"].float()
                comparable = transition["property_current_observed_mask"].to(torch.bool)
                raw_full, _ = decode_property_prediction(
                    outputs["node_property"],
                    transition["property_t"],
                    property_mode=property_mode,
                    transition_gate=(
                        torch.sigmoid(outputs["node_change_logits"])
                        if property_mode in {"logit_mixture", "logit_residual_mixture"}
                        else None
                    ),
                    current_observed_mask=(
                        transition["property_observation_mask_t"].squeeze(-1).bool()
                        if property_mode in {"logit_mixture", "logit_residual_mixture"}
                        else None
                    ),
                )
                raw_rows = raw_full.index_select(0, target_ids)
                if target_split == "train" and bool(comparable.any()):
                    train_property_magnitudes.append(
                        (target_rows[comparable] - current_rows[comparable])
                        .abs()
                        .mean(dim=-1)
                        .detach()
                        .cpu()
                    )
                if target_split == "test" and args.split == "test":
                    primary_rows = raw_rows
                    predictions = {




                        "gwm": primary_rows,
                        "future_latent_only": raw_rows,
                        "copy_last": current_rows,
                    }
                    threshold = (
                        float(torch.quantile(torch.cat(train_property_magnitudes), 0.75))
                        if train_property_magnitudes
                        else float("inf")
                    )
                    delta_magnitude = (target_rows - current_rows).abs().mean(dim=-1)
                    changed_mask = comparable & delta_magnitude.gt(1e-8)
                    large_mask = comparable & delta_magnitude.ge(threshold)
                    for name, prediction_rows in predictions.items():
                        property_metrics[name].update(
                            prediction_rows, target_rows, current_rows
                        )
                        if bool(changed_mask.any()):
                            property_changed[name].update(
                                prediction_rows[changed_mask], target_rows[changed_mask]
                            )
                        if bool(large_mask.any()):
                            property_large[name].update(
                                prediction_rows[large_mask], target_rows[large_mask]
                            )
                elif target_split == "val" and args.split == "val":





                    validation_predictions = {
                        "gwm": raw_rows,
                        "future_latent_only": raw_rows,
                    }
                    validation_predictions["copy_last"] = current_rows
                    for name, prediction_rows in validation_predictions.items():
                        property_metrics[name].update(
                            prediction_rows, target_rows, current_rows
                        )
                        delta_magnitude = (
                            target_rows - current_rows
                        ).abs().mean(dim=-1)
                        changed_mask = comparable & delta_magnitude.gt(1e-8)
                        large_threshold = (
                            float(
                                torch.quantile(
                                    torch.cat(train_property_magnitudes), 0.75
                                )
                            )
                            if train_property_magnitudes
                            else float("inf")
                        )
                        large_mask = comparable & delta_magnitude.ge(
                            large_threshold
                        )
                        if bool(changed_mask.any()):
                            property_changed[name].update(
                                prediction_rows[changed_mask],
                                target_rows[changed_mask],
                            )
                        if bool(large_mask.any()):
                            property_large[name].update(
                                prediction_rows[large_mask],
                                target_rows[large_mask],
                            )

            if transition["split"] == args.split:
                scored_transition_count += 1





                if soft_action_mass is not None:
                    soft_action_masses.append(soft_action_mass)
                if soft_action_support_count is not None:
                    soft_action_support_counts.append(soft_action_support_count)


                if not args.skip_controller_diagnostics:




                    if selected is None:
                        selected = controller.greedy_action_sequence(
                            z_policy,
                            hidden_for_action,
                            candidate_edges=candidate_edges,
                            candidate_nodes=candidate_nodes,
                            valid_operations=TASK_OPERATIONS[task],
                            min_actions=min_actions,
                            max_actions=max_actions,
                            predict_magnitude=task not in {"topology", "node_change"},
                            stop_threshold=args.stop_threshold,
                            edge_chunk_size=args.pair_chunk_size,
                        )
                    node_weights = structure_node_weights(
                        edge_index_t,
                        num_nodes=num_nodes,
                        historical_change_counts=historical_change_counts,
                        history_steps=chronological_index,
                    )
                    topology_context = (
                        _topology_reward_context(
                            transition,
                            candidate_edges,
                            node_weights,
                            device=device,
                        )
                        if task == "topology"
                        else None
                    )
                    topology_target_z = None
                    if task == "topology":
                        target_edge_weight = (
                            transition.get("edge_weight_next")
                            if edge_input == "weighted"
                            else None
                        )
                        topology_target_z = model.encode_target(
                            transition["x_next"],
                            transition["edge_index_next"],
                            edge_weight_next=target_edge_weight,
                            topology_cache_key=int(transition["transition_id"]) + 1,
                        )
                    reward, details = _decoded_reward(
                        task,
                        selected.action,
                        outputs,
                        transition,
                        node_weights,
                        candidate_edges=candidate_edges,
                        candidate_nodes=candidate_nodes,
                        device=device,
                        topology_exact_reward_weight=exact_reward_weight,
                        topology_context=topology_context,
                        topology_target_z=topology_target_z,
                        node_action_reward_weight=float(
                            _checkpoint_value(saved, "node_action_reward_weight", 0.0)
                        ),
                        property_mode=property_mode,
                        property_change_threshold=(
                            None
                            if property_change_threshold is None
                            else float(property_change_threshold)
                        ),
                    )
                    records.append(
                        {
                            "transition_id": int(transition["transition_id"]),
                            "split": str(transition["split"]),
                            "policy_log_probability": float(
                                selected.old_log_prob.item()
                            ),
                            "action_count": len(selected.action),
                            "actions": [
                                {
                                    "operation": edit.operation.name,
                                    "target": list(edit.target),
                                }
                                for edit in selected.action
                            ],
                            "decoded_reward": reward,
                            "metrics": details,
                            "soft_action_mass": soft_action_mass,
                            "soft_action_support_count": soft_action_support_count,
                        }
                    )




            if task == "topology":
                topology_add_count_history.append(
                    int(transition["edge_added"].shape[1])
                )
                topology_remove_count_history.append(
                    int(transition["edge_removed"].shape[1])
                )
                if topology_pair_state is not None:
                    topology_pair_state.observe(
                        transition_cpu["edge_index_t"],
                        time_index=chronological_index,
                    )
            historical_change_counts = (
                historical_change_counts
                + _past_change_update(
                    task, transition, device, num_nodes=num_nodes
                )
            )
            historical_change_observations = (
                historical_change_observations
                + _past_change_observation_count(task, transition, device)
            )
            if task in {"node_change", "node_state"}:
                _observe_node_change_history(
                    node_change_history,
                    transition_cpu["node_changed"],
                )
            if (
                args.max_eval_transitions
                and scored_transition_count >= args.max_eval_transitions
            ):
                break

    if not scored_transition_count:
        raise RuntimeError(f"No transitions were evaluated for split={args.split}.")
    metric_names = sorted(
        {name for record in records for name in record["metrics"]}
    )



    aggregate_metrics = {}
    for name in metric_names:
        values = [
            float(record["metrics"][name])
            for record in records
            if isinstance(record["metrics"].get(name), (int, float))
        ]
        if values:
            aggregate_metrics[name] = float(sum(values) / len(values))
    formal_metrics: dict[str, Any]
    calibration: dict[str, Any] | None = None
    if task == "topology":
        formal_metrics = {
            "edge_addition": topology_formal["addition"].metrics(),
            "edge_deletion": topology_formal["deletion"].metrics(),
            "edge_change_only_macro_f1": 0.5 * (
                topology_formal["addition"].metrics()["f1"]
                + topology_formal["deletion"].metrics()["f1"]
            ),
            "future_graph_reconstruction": topology_formal[
                "reconstruction"
            ].metrics(),
            "candidate_protocol": topology_candidate_policy,
            "reactivation_addition": (
                topology_reactivation.metrics()
                if topology_candidate_policy == "seen_pairs"
                else None
            ),
            "addition_target_composition": (
                {
                    "reactivation_targets": topology_reactivation_target_count,
                    "novel_targets": topology_novel_target_count,
                    "novel_fraction": (
                        topology_novel_target_count
                        / max(topology_all_target_count, 1)
                    ),
                }
                if topology_candidate_policy == "seen_pairs"
                else None
            ),
            "decode_policy": args.topology_decode_policy,
            "cardinality_source": (
                (
                    "Controller future-free log churn-rate head scaled by |E_t|"
                    if str(
                        _checkpoint_value(
                            saved, "topology_count_target_mode", "absolute"
                        )
                    )
                    == "edge_rate"
                    else "Controller future-free log add/remove-count head"
                )
                if args.topology_decode_policy == "controller_count_topk"
                else (
                    "predicted future latent count decoder"
                    if args.topology_decode_policy == "latent_count_topk"
                    else "mean of the last eight released operation counts"
                )
            ),
            "controller_count_scales": (
                {
                    "addition": float(
                        args.topology_add_count_scale
                        if args.topology_add_count_scale is not None
                        else 1.0
                    ),
                    "deletion": float(
                        args.topology_remove_count_scale
                        if args.topology_remove_count_scale is not None
                        else 1.0
                    ),
                }
                if args.topology_decode_policy == "controller_count_topk"
                else None
            ),
            "controller_count_history_blend": (
                float(args.topology_count_history_blend)
                if args.topology_decode_policy == "controller_count_topk"
                else None
            ),
            "topology_history_prior": bool(args.topology_history_prior),
            "controller_count_target_mode": (
                str(
                    _checkpoint_value(
                        saved, "topology_count_target_mode", "absolute"
                    )
                )
                if args.topology_decode_policy == "controller_count_topk"
                else None
            ),
            "controller_count_context": bool(
                _checkpoint_value(saved, "topology_count_context", False)
            ),
            "pair_history": (
                None
                if args.topology_decode_policy == "controller_count_topk"
                else topology_readout
            ),
            "future_edge_count_used": False,
            "controller_pair_score_weight": float(
                args.topology_controller_score_weight
            ),
            "controller_count_diagnostics": (
                {
                    "mean_predicted_add_count": float(
                        torch.stack(controller_topology_count_predictions)[:, 0].mean()
                    ),
                    "mean_predicted_remove_count": float(
                        torch.stack(controller_topology_count_predictions)[:, 1].mean()
                    ),
                    "mean_true_add_count": float(
                        torch.stack(controller_topology_count_targets)[:, 0].mean()
                    ),
                    "mean_true_remove_count": float(
                        torch.stack(controller_topology_count_targets)[:, 1].mean()
                    ),
                    "log1p_count_mae": float(
                        (
                            torch.log1p(torch.stack(controller_topology_count_predictions))
                            - torch.log1p(torch.stack(controller_topology_count_targets))
                        )
                        .abs()
                        .mean()
                    ),
                }
                if controller_topology_count_predictions
                else None
            ),
            "baselines": {
                "previous_snapshot_persistence": {
                    "edge_addition": topology_persistence["addition"].metrics(),
                    "edge_deletion": topology_persistence["deletion"].metrics(),
                    "future_graph_reconstruction": topology_persistence[
                        "reconstruction"
                    ].metrics(),
                },
                "historical_edge_frequency": {
                    "edge_addition": topology_frequency["addition"].metrics(),
                    "edge_deletion": topology_frequency["deletion"].metrics(),
                    "future_graph_reconstruction": topology_frequency[
                        "reconstruction"
                    ].metrics(),
                },
                "random": {
                    "edge_addition": topology_random["addition"].metrics(),
                    "edge_deletion": topology_random["deletion"].metrics(),
                    "future_graph_reconstruction": topology_random[
                        "reconstruction"
                    ].metrics(),
                    "cardinality_source": (
                        "mean of completed-prefix released operation counts"
                    ),
                    "random_seed": int(seed),
                },
            },
            "primary_metrics": [
                "edge_addition.f1",
                "edge_deletion.f1",
            ],
            "secondary_metrics": [
                "future_graph_reconstruction.f1",
                "future_graph_reconstruction.jaccard",
            ],
            "jaccard_relation": (
                "For globally accumulated set counts, Jaccard = F1 / (2 - F1); "
                "it is retained for compatibility but is not an independent "
                "primary metric."
            ),
        }
    elif task == "node_change":
        validation = node_change_buckets["val"]
        requested = node_change_buckets[args.split]
        if not validation["labels"] or not requested["labels"]:
            raise RuntimeError("Node-change formal evaluation requires validation and requested rows.")
        validation_labels = torch.cat(validation["labels"]).numpy().astype(np.int64)
        requested_labels = torch.cat(requested["labels"]).numpy().astype(np.int64)
        calibration, formal_metrics = _calibrate_node_change_readout(
            validation,
            requested,
            logits_key="logits",
            use_node_history_blend=args.node_history_blend,
        )
        if not args.skip_controller_diagnostics:
            controller_calibration, controller_metrics = _calibrate_node_change_readout(
                validation,
                requested,
                logits_key="controller_logits",
                use_node_history_blend=args.node_history_blend,
            )
            formal_metrics["controller_action_localization"] = controller_metrics
            formal_metrics["controller_history_calibration"] = controller_calibration






        if args.node_history_blend:
            validation_features = _bucket_node_history_features(validation)
            requested_features = _bucket_node_history_features(requested)
            if validation_features and requested_features:
                validation_scores = node_change_calibrated_scores(
                    torch.cat(validation["logits"]).numpy(),
                    validation_features,
                    calibration,
                )
                requested_scores = node_change_calibrated_scores(
                    torch.cat(requested["logits"]).numpy(),
                    requested_features,
                    calibration,
                )
                rate_calibration = calibrate_past_rate_topk(
                    validation_labels,
                    validation_scores,
                    validation["block_sizes"],
                    validation["past_positive_rates"],
                )
                formal_metrics["past_rate_topk"] = evaluate_past_rate_topk(
                    requested_labels,
                    requested_scores,
                    requested["block_sizes"],
                    requested["past_positive_rates"],
                    rate_calibration,
                )
                formal_metrics["past_rate_topk_calibration"] = rate_calibration
        decoder_logits = torch.cat(requested["logits"]).float().numpy()
        controller_logits = torch.cat(requested["controller_logits"]).float().numpy()
        if decoder_logits.size > 1 and np.std(decoder_logits) > 0 and np.std(controller_logits) > 0:
            score_correlation = float(
                np.corrcoef(decoder_logits.reshape(-1), controller_logits.reshape(-1))[0, 1]
            )
        else:
            score_correlation = None
        formal_metrics["controller_decoder_score_correlation"] = score_correlation
        formal_metrics["always_unchanged"] = {
            "auroc": 0.5 if np.unique(requested_labels).size > 1 else None,
            "auprc": float(requested_labels.mean()),
            "f1": 0.0,
            "positive_rate": float(requested_labels.mean()),
        }
    elif task == "node_state":
        feature_names = metadata.get("node_feature_names")
        validation = node_change_buckets["val"]
        requested = node_change_buckets[args.split]
        state_change_metrics: dict[str, Any] | None = None
        state_change_calibration: dict[str, Any] | None = None
        if validation["labels"] and requested["labels"]:
            state_change_calibration, state_change_readout = (
                _calibrate_node_change_readout(
                    validation,
                    requested,
                    logits_key="logits",
                    use_node_history_blend=args.node_history_blend,
                )
            )
            state_change_metrics = state_change_readout["validation_calibrated"]
        controller_action_localization: dict[str, Any] | None = None
        controller_change_calibration: dict[str, Any] | None = None
        if (
            not args.skip_controller_diagnostics
            and
            validation["labels"]
            and requested["labels"]
            and validation["controller_logits"]
            and requested["controller_logits"]
        ):
            controller_change_calibration, controller_change_readout = (
                _calibrate_node_change_readout(
                    validation,
                    requested,
                    logits_key="controller_logits",
                    use_node_history_blend=args.node_history_blend,
                )
            )
            controller_action_localization = controller_change_readout[
                "validation_calibrated"
            ]
        controller_magnitude = RegressionAccumulator()
        if requested["labels"] and state_rows[args.split]["controller_mu"]:
            requested_state_rows = state_rows[args.split]
            controller_mu = torch.cat(requested_state_rows["controller_mu"])
            controller_target = torch.cat(requested_state_rows["target"])
            controller_changed = torch.cat(requested_state_rows["labels"]).bool()
            if bool(controller_changed.any()):
                controller_magnitude.update(
                    controller_mu[controller_changed],
                    controller_target[controller_changed],
                )
        state_gate_calibration, calibrated_all, calibrated_changed = (
            _calibrate_state_gate(state_rows, args.split, feature_names)
        )
        formal_metrics = {






            "all_nodes": state_all.metrics(feature_names),
            "changed_nodes": state_changed.metrics(feature_names),
            "validation_gated_readout": {
                "all_nodes": calibrated_all.metrics(feature_names),
                "changed_nodes": calibrated_changed.metrics(feature_names),
            },
            "uncalibrated_model": {
                "all_nodes": state_all.metrics(feature_names),
                "changed_nodes": state_changed.metrics(feature_names),
            },
            "state_gate_calibration": state_gate_calibration,
            "node_change_history_calibration": state_change_calibration,
            "node_change": state_change_metrics,
            "controller_action_localization": controller_action_localization,
            "controller_history_calibration": controller_change_calibration,
            "controller_changed_node_magnitude": controller_magnitude.metrics(
                feature_names
            ),
            "copy_last_zero_delta": {
                "all_nodes": state_copy_all.metrics(feature_names),
                "changed_nodes": state_copy_changed.metrics(feature_names),
            },
            "prediction": (
                f"{model.state_prediction_mode} conditional magnitude decoded "
                "from future latent"
            ),
        }
    else:
        requested_property_split = "test" if args.split == "test" else "val"
        threshold = (
            float(torch.quantile(torch.cat(train_property_magnitudes), 0.75))
            if train_property_magnitudes
            else None
        )
        formal_metrics = {
            "property_mode": property_mode,
            "train_q75_large_change_threshold": threshold,
            requested_property_split: {
                name: {
                    **accumulator.metrics(),
                    "changed_rows": property_changed[name].metrics(
                        include_per_feature=False
                    ),
                    "large_change_rows": property_large[name].metrics(
                        include_per_feature=False
                    ),
                }
                for name, accumulator in property_metrics.items()
            },
        }
    result = {
        "experiment": "action_aware_gwm_future_free_greedy_inference",
        "task": task,
        "dataset": dataset_name,
        "split": args.split,
        "history_burn_in": int(args.history_burn_in),
        "model_rollout_start_index": int(model_start_index),
        "prefix_observable_statistics": "complete chronological past",
        "soft_node_action_probability_calibration": (
            soft_node_action_probability_calibration
        ),
        "soft_node_action_prior_smoothing": soft_node_action_prior_smoothing,
        "processed": str(processed.relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "selection_rule": (
            "greedy operation + greedy legal target + Gaussian mean + "
            f"STOP(probability >= {args.stop_threshold})"
        ),
        "future_reward_reranking": False,
        "controller_diagnostics_skipped": bool(args.skip_controller_diagnostics),
        "num_evaluated_transitions": int(scored_transition_count),
        "mean_decoded_reward": float(
            sum(record["decoded_reward"] for record in records) / len(records)
            if records
            else 0.0
        ),
        "mean_action_count": float(
            sum(record["action_count"] for record in records) / len(records)
            if records
            else 0.0
        ),
        "mean_soft_action_mass": (
            None
            if not soft_action_masses
            else float(sum(soft_action_masses) / len(soft_action_masses))
        ),
        "mean_soft_action_support_count": (
            None
            if not soft_action_support_counts
            else float(sum(soft_action_support_counts) / len(soft_action_support_counts))
        ),
        "metrics": aggregate_metrics,
        "formal_metrics": formal_metrics,
        "validation_calibration": calibration,
        "predictions": records,
    }
    result_path = ROOT / args.results
    save_json(result, result_path)
    print(f"task={task} split={args.split} transitions={len(records)}")
    print(f"mean_action_count={result['mean_action_count']:.4f}")
    for name, value in aggregate_metrics.items():
        print(f"{name}={value:.6f}")
    if task == "topology":
        print(
            "formal_summary="
            f"add_f1={formal_metrics['edge_addition']['f1']:.6f} "
            f"delete_f1={formal_metrics['edge_deletion']['f1']:.6f} "
            f"change_macro_f1={formal_metrics['edge_change_only_macro_f1']:.6f} "
            "reconstruction_f1="
            f"{formal_metrics['future_graph_reconstruction']['f1']:.6f}"
        )
    elif task == "node_change":
        calibrated = formal_metrics["validation_calibrated"]
        print(
            "formal_summary="
            f"auroc={calibrated['auroc']:.6f} "
            f"auprc={calibrated['auprc']:.6f} "
            f"f1={calibrated['f1']:.6f}"
        )
    elif task == "node_state":
        print(
            "formal_summary="
            f"all_mae={formal_metrics['all_nodes']['mae']:.6f} "
            f"changed_mae={formal_metrics['changed_nodes']['mae']:.6f}"
        )
    else:
        property_split = "test" if args.split == "test" else "val"
        test_metrics = formal_metrics[property_split]["gwm"]
        print(
            "formal_summary="
            f"ndcg_at_10={test_metrics['official_ndcg_at_10']:.6f} "
            f"mse={test_metrics['property_mse']:.6f} "
            f"delta_mae={test_metrics['delta_mae']:.6f}"
        )
    try:
        display_result_path = result_path.relative_to(ROOT)
    except ValueError:
        display_result_path = result_path
    print(f"saved_results={display_result_path}")


if __name__ == "__main__":
    main()
