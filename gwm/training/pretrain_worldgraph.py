

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from gwm.training.runtime_cpu import configure_cpu_environment, configure_torch_runtime

_CPU_THREADS = configure_cpu_environment(overwrite=True)

import torch
from torch.nn import functional as F

configure_torch_runtime(torch, _CPU_THREADS)

from gwm.benchmark_protocol import protocol_digest, resolve_task_dataset
from gwm.pretraining import (
    PRETRAINING_FORMAT,
    TARGET_ARCHITECTURE,
    TemporalPretrainingSource,
    WorldGraphContrastivePretrainer,
    discover_loo_sources,
    graph_queue_info_nce,
    load_snapshot_bundle,
    resolve_target_architecture,
    sample_aligned_nodes,
    symmetric_info_nce,
)
from gwm.t3_structure import DESCRIPTOR_NAMES
from gwm.utils import resolve_device, seed_everything


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TARGET_ARCHITECTURE), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--steps_per_epoch", type=_positive_int, default=256)
    parser.add_argument("--val_steps_per_domain", type=_positive_int, default=8)
    parser.add_argument(
        "--history_window", type=_positive_int, default=None,
        help="Defaults to the finalized downstream protocol for this task/dataset.",
    )
    parser.add_argument("--history_num_heads", type=_positive_int, default=4)
    parser.add_argument("--latent_dim", type=_positive_int)
    parser.add_argument("--hidden_dim", type=_positive_int)
    parser.add_argument("--action_dim", type=_positive_int)
    parser.add_argument("--policy_dim", type=_positive_int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--contrastive_dim", type=_positive_int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--feature_mask_probability", type=float, default=0.1)
    parser.add_argument("--max_contrastive_nodes", type=_positive_int, default=512)
    parser.add_argument("--node_view_weight", type=float, default=1.0)
    parser.add_argument("--graph_view_weight", type=float, default=0.5)
    parser.add_argument("--future_weight", type=float, default=1.0)
    parser.add_argument("--state_weight", type=float, default=0.25)



    parser.add_argument("--prototype_weight", type=float, default=0.0)
    parser.add_argument(
        "--action_prediction_weight", type=float, default=0.25,
        help="Auxiliary weight for predicting the past-only edit summary.",
    )
    parser.add_argument(
        "--activity_prediction_weight", type=float, default=0.25,
        help="Auxiliary weight for source node activation/deactivation prediction.",
    )
    parser.add_argument(
        "--task_alignment_weight", type=float, default=1.0,
        help="Weight of the task-level LOO objective transferred downstream.",
    )
    parser.add_argument(
        "--same_task_probability", type=float, default=1.0,
        help=(
            "Probability of sampling a source from the target task; 1.0 is "
            "the strict same-task LOO protocol."
        ),
    )
    parser.add_argument("--future_cosine_weight", type=float, default=0.5)
    parser.add_argument("--target_momentum", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--sgt_num_hops", type=_positive_int, default=2)
    parser.add_argument("--sgt_num_walks", type=_positive_int, default=4)
    parser.add_argument("--sgt_walk_length", type=_positive_int, default=3)
    parser.add_argument(
        "--sgt_topology_cache_bytes", type=int, default=384 * 1024**2
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Output checkpoint (default: benchmark_artifacts/pretraining/...).",
    )
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    architecture = resolve_target_architecture(args.task, args.dataset)
    for name in ("latent_dim", "hidden_dim", "action_dim", "policy_dim", "history_window"):
        if getattr(args, name) is None:
            setattr(args, name, architecture[name])
    if not 0.0 <= args.feature_mask_probability < 1.0:
        parser.error("--feature_mask_probability must lie in [0, 1)")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if not 0.0 <= args.target_momentum < 1.0:
        parser.error("--target_momentum must lie in [0, 1)")
    if args.sgt_topology_cache_bytes < 0:
        parser.error("--sgt_topology_cache_bytes must be non-negative")
    if args.action_prediction_weight < 0.0:
        parser.error("--action_prediction_weight must be non-negative")
    if args.activity_prediction_weight < 0.0:
        parser.error("--activity_prediction_weight must be non-negative")
    if args.task_alignment_weight < 0.0:
        parser.error("--task_alignment_weight must be non-negative")
    if not 0.0 <= args.same_task_probability <= 1.0:
        parser.error("--same_task_probability must lie in [0, 1]")
    return args


def _trainable_parameters(model: WorldGraphContrastivePretrainer):
    modules = (
        model.world_model.input_adapter,
        model.world_model.graph_encoder,
        model.world_model.state_model,
        model.world_model.latent_predictor,
        model.action_adapter,
        model.action_predictor,
        model.activity_policy_encoder,
        model.activity_policy_head,
        model.semantic_change_decoder,
        model.semantic_delta_decoder,
        model.topology_controller,
        model.structure_trunk,
        model.structure_magnitude_head,
        model.structure_direction_head,
        model.node_projector,
        model.graph_projector,
        model.state_projector,
    )
    return [
        parameter
        for module in modules
        if module is not None
        for parameter in module.parameters()
    ]


def _cosine_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"cosine inputs have different shapes: {left.shape}, {right.shape}")
    return (1.0 - F.cosine_similarity(left, right, dim=-1)).mean()


def _future_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    cosine_weight: float,
    maximum_nodes: int,
) -> torch.Tensor:
    prediction, target = sample_aligned_nodes(
        prediction,
        target,
        maximum=min(int(maximum_nodes), int(prediction.shape[0])),
    )
    cosine = _cosine_loss(prediction, target)



    del temperature
    regression = F.smooth_l1_loss(prediction, target)
    return float(cosine_weight) * cosine + regression


def _domain_transition(
    source: TemporalPretrainingSource,
    *,
    split: str,
    rng: random.Random,
) -> int:
    candidates = source.indices.get(split, [])
    if not candidates:
        candidates = source.indices["train"]
    return int(rng.choice(candidates))


def _sample_source(
    sources: list[TemporalPretrainingSource],
    *,
    target_task: str,
    same_task_probability: float,
    rng: random.Random,
) -> TemporalPretrainingSource:

    same = [source for source in sources if source.spec.task == target_task]
    if same and rng.random() < float(same_task_probability):
        return rng.choice(same)
    return rng.choice(sources)


def _step_loss(
    model: WorldGraphContrastivePretrainer,
    source: TemporalPretrainingSource,
    transition_index: int,
    *,
    device: torch.device,
    args: argparse.Namespace,
    graph_queue: deque[torch.Tensor],
    domain_prototypes: dict[str, torch.Tensor],
    training: bool,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    if training:
        model.train()
    else:
        model.eval()
    view_one = model.encode_history(
        source,
        transition_index,
        device=device,
        feature_mask_probability=0.0,
        topology_namespace="online",
    )
    view_two = model.encode_history(
        source,
        transition_index,
        device=device,
        feature_mask_probability=args.feature_mask_probability,



        topology_namespace="augment",
    )
    target = model.encode_future_target(
        source,
        transition_index,
        device=device,
    )

    node_one = model.node_projector(view_one["node"])
    node_two = model.node_projector(view_two["node"])
    node_one, node_two = sample_aligned_nodes(
        node_one, node_two, maximum=args.max_contrastive_nodes
    )
    node_loss = symmetric_info_nce(
        node_one, node_two, temperature=args.temperature
    )
    graph_one = model.graph_projector(view_one["graph"].unsqueeze(0))




    graph_two_view = model.graph_projector(view_two["graph"].unsqueeze(0))
    graph_loss = _cosine_loss(graph_one, graph_two_view)
    future_prediction = model.node_projector(view_one["future"])
    future_target = model.target_node_projector(target["node"])
    future_loss = _future_loss(
        future_prediction,
        future_target,
        temperature=args.temperature,
        cosine_weight=args.future_cosine_weight,
        maximum_nodes=args.max_contrastive_nodes,
    )
    state_one = model.state_projector(view_one["hidden"])
    state_two = model.state_projector(view_two["hidden"])
    state_one, state_two = sample_aligned_nodes(
        state_one, state_two, maximum=args.max_contrastive_nodes
    )
    state_loss = _cosine_loss(state_one, state_two)
    action_summary = source.action_summary(transition_index).to(device=device)


    action_target = torch.tanh(action_summary / 5.0)
    action_features = torch.cat(
        [view_one["hidden"].mean(dim=0), view_one["graph"]], dim=-1
    )
    action_prediction = model.action_predictor(action_features)
    action_loss = F.smooth_l1_loss(action_prediction, action_target)



    current_snapshot = source.snapshots[int(transition_index)]
    next_snapshot = source.snapshots[int(transition_index) + 1]
    num_nodes = int(current_snapshot["x"].shape[0])
    current_active = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    next_active = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    current_edges = current_snapshot["edge_index"].to(device=device, dtype=torch.long)
    next_edges = next_snapshot["edge_index"].to(device=device, dtype=torch.long)
    if current_edges.numel():
        current_active[current_edges.reshape(-1).unique()] = True
    if next_edges.numel():
        next_active[next_edges.reshape(-1).unique()] = True
    activity_target = torch.stack(
        [
            (next_active & ~current_active).to(dtype=view_one["future"].dtype),
            (current_active & ~next_active).to(dtype=view_one["future"].dtype),
        ],
        dim=-1,
    )




    activity_policy_input = torch.cat(
        [view_one["node"].detach(), view_one["hidden"].detach()], dim=-1
    )
    activity_logits = model.activity_policy_head(
        model.activity_policy_encoder(activity_policy_input)
    )
    positive = activity_target.sum(dim=0)
    negative = activity_target.shape[0] - positive
    pos_weight = ((negative + 1.0) / (positive + 1.0)).clamp(1.0, 20.0)
    activity_loss = F.binary_cross_entropy_with_logits(
        activity_logits,
        activity_target,
        pos_weight=pos_weight,
    )
    previous = domain_prototypes.get(source.spec.key)
    if previous is None:
        prototype_loss = graph_loss.new_zeros(())
    else:
        prototype_loss = _cosine_loss(
            graph_one, previous.to(device=graph_one.device).unsqueeze(0)
        )
    task_alignment_loss = graph_loss.new_zeros(())
    task_terms: dict[str, float] = {}
    if model.target_task == "T1" and source.spec.task == "T1":
        assert model.semantic_change_decoder is not None
        assert model.semantic_delta_decoder is not None
        target_semantic = source.semantic_transition_target(transition_index)
        known = target_semantic["known"].to(device=device)
        if bool(known.any()):
            semantic_latent = view_one["future"].index_select(0, known.nonzero().flatten())
            changed = target_semantic["changed"].to(device=device, dtype=semantic_latent.dtype)[known]
            logits = model.semantic_change_decoder(semantic_latent)
            positive = changed.sum()
            negative = changed.numel() - positive
            pos_weight = ((negative + 1.0) / (positive + 1.0)).clamp(1.0, 20.0)
            change_loss = F.binary_cross_entropy_with_logits(
                logits, changed, pos_weight=pos_weight
            )
            delta_target = target_semantic["delta"].to(
                device=device, dtype=semantic_latent.dtype
            )[known]
            delta_loss = F.smooth_l1_loss(
                model.semantic_delta_decoder(semantic_latent), delta_target
            )
            task_alignment_loss = change_loss + 0.25 * delta_loss
            task_terms = {
                "task_semantic_change": float(change_loss.detach().cpu()),
                "task_semantic_delta": float(delta_loss.detach().cpu()),
            }
    elif model.target_task == "T2" and source.spec.task == "T2":
        assert model.topology_controller is not None
        topology = source.topology_transition_target(transition_index)
        policy_state = model.topology_controller.node_state(
            view_one["node"], view_one["hidden"]
        )
        pair_losses: list[torch.Tensor] = []
        for prefix in ("add", "remove"):
            positive_pairs = topology[f"{prefix}_positive"].to(device=device)
            negative_pairs = topology[f"{prefix}_negative"].to(device=device)
            candidates = torch.cat((positive_pairs, negative_pairs), dim=1)
            if candidates.shape[1] == 0:
                continue
            labels = torch.cat(
                (
                    torch.ones(positive_pairs.shape[1], device=device),
                    torch.zeros(negative_pairs.shape[1], device=device),
                )
            )
            logits = model.topology_controller.edge_logits_chunked(
                policy_state, candidates
            )
            pair_losses.append(F.binary_cross_entropy_with_logits(logits, labels))
        pair_loss = (
            torch.stack(pair_losses).mean()
            if pair_losses
            else policy_state.new_zeros(())
        )
        count_target = torch.log1p(
            topology["counts"].to(device=device, dtype=policy_state.dtype)
        )
        count_loss = F.smooth_l1_loss(
            model.topology_controller.topology_log_counts(policy_state), count_target
        )
        task_alignment_loss = pair_loss + 0.25 * count_loss
        task_terms = {
            "task_edge_pair": float(pair_loss.detach().cpu()),
            "task_edge_count": float(count_loss.detach().cpu()),
        }
    elif model.target_task == "T3" and source.spec.task == "T3":
        assert model.structure_trunk is not None
        assert model.structure_magnitude_head is not None
        assert model.structure_direction_head is not None
        structure = source.structure_transition_target(transition_index)
        centres_cpu = structure["centres"].to(dtype=torch.long)
        positions = torch.arange(centres_cpu.numel(), dtype=torch.long)
        if positions.numel() > args.max_contrastive_nodes:
            positions = positions[: args.max_contrastive_nodes]
        centres = centres_cpu.index_select(0, positions).to(
            device=device, dtype=torch.long
        )
        if centres.numel():
            feature = torch.cat(
                (
                    view_one["node"].index_select(0, centres),
                    view_one["future"].index_select(0, centres),
                    view_one["hidden"].index_select(0, centres),
                ),
                dim=-1,
            )
            trunk = model.structure_trunk(feature)
            target_delta = structure["delta"].index_select(0, positions).to(
                device=device, dtype=trunk.dtype
            )
            robust_target = torch.asinh(target_delta)
            magnitude_loss = F.smooth_l1_loss(
                model.structure_magnitude_head(trunk), robust_target
            )
            direction_target = torch.where(
                target_delta < -1e-6,
                torch.zeros_like(target_delta, dtype=torch.long),
                torch.where(
                    target_delta > 1e-6,
                    torch.full_like(target_delta, 2, dtype=torch.long),
                    torch.ones_like(target_delta, dtype=torch.long),
                ),
            )
            direction_logits = model.structure_direction_head(trunk).view(
                trunk.shape[0], len(DESCRIPTOR_NAMES), 3
            )
            flat_target = direction_target.reshape(-1)
            frequencies = torch.bincount(flat_target, minlength=3).to(trunk.dtype)
            class_weight = (frequencies.sum() / frequencies.clamp_min(1.0)).sqrt()
            class_weight = class_weight / class_weight.mean().clamp_min(1e-8)
            direction_loss = F.cross_entropy(
                direction_logits.reshape(-1, 3), flat_target, weight=class_weight
            )
            task_alignment_loss = magnitude_loss + 0.5 * direction_loss
            task_terms = {
                "task_structure_magnitude": float(magnitude_loss.detach().cpu()),
                "task_structure_direction": float(direction_loss.detach().cpu()),
            }

    total = (
        float(args.node_view_weight) * node_loss
        + float(args.graph_view_weight) * graph_loss
        + float(args.future_weight) * future_loss
        + float(args.state_weight) * state_loss
        + float(args.prototype_weight) * prototype_loss
        + float(args.action_prediction_weight) * action_loss
        + float(args.activity_prediction_weight) * activity_loss
        + float(args.task_alignment_weight) * task_alignment_loss
    )
    with torch.no_grad():
        graph_snapshot = graph_one[0].detach()
    terms = {
        "total": float(total.detach().cpu()),
        "node": float(node_loss.detach().cpu()),
        "graph": float(graph_loss.detach().cpu()),
        "future": float(future_loss.detach().cpu()),
        "state": float(state_loss.detach().cpu()),
        "prototype": float(prototype_loss.detach().cpu()),
        "action_prediction": float(action_loss.detach().cpu()),
        "activity_prediction": float(activity_loss.detach().cpu()),
        "task_alignment": float(task_alignment_loss.detach().cpu()),
        **task_terms,
    }
    return total, terms, graph_snapshot


def _save_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    protocol_digest_value: str,
    source_specs: list[Any],
    model: WorldGraphContrastivePretrainer,
    best_epoch: int,
    best_val_loss: float,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": PRETRAINING_FORMAT,
        "version": 4,
        "target_task": args.task,
        "target_dataset": args.dataset,
        "protocol_digest": protocol_digest_value,
        "sources": [
            {
                "key": spec.key,
                "task": spec.task,
                "dataset": spec.dataset,
                "path": str(spec.path),
                "weighted_edges": bool(spec.weighted_edges),
            }
            for spec in source_specs
        ],
        "config": vars(args),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "transferable_state": model.transferable_state_dict(),
        "activity_controller_state": (
            model.transferable_activity_controller_state_dict()
        ),
        "task_transferable_state": model.transferable_task_state_dict(),
    }


    payload["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload["config"].items()
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _default_checkpoint(args: argparse.Namespace) -> Path:
    return ROOT / "benchmark_artifacts" / "pretraining" / (
        f"worldgraph_loo_{args.task.lower()}_{args.dataset.lower()}.pt"
    )


def main() -> None:
    args = _parse_args()
    if args.seed < 1:
        raise ValueError("--seed must be positive")
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    protocol_path = None
    entry = resolve_task_dataset(args.task, args.dataset)
    source_specs = discover_loo_sources(
        args.dataset,
        target_task=args.task,
        protocol_path=protocol_path,
        root=ROOT,
    )




    if args.same_task_probability >= 0.999:
        same_task_specs = [spec for spec in source_specs if spec.task == args.task]
        if same_task_specs:
            source_specs = same_task_specs
    loaded_sources = [
        TemporalPretrainingSource(spec, load_snapshot_bundle(spec.path))
        for spec in source_specs
    ]
    validation_sources = (
        [source for source in loaded_sources if source.spec.task == args.task]
        if args.same_task_probability >= 0.999
        else loaded_sources
    ) or loaded_sources
    input_dims = {source.spec.key: source.input_dim for source in loaded_sources}
    model = WorldGraphContrastivePretrainer(
        input_dims,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        policy_dim=args.policy_dim,
        history_window=args.history_window,
        history_num_heads=args.history_num_heads,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
        target_momentum=args.target_momentum,
        sgt_num_hops=args.sgt_num_hops,
        sgt_num_walks=args.sgt_num_walks,
        sgt_walk_length=args.sgt_walk_length,
        sgt_topology_cache_bytes=args.sgt_topology_cache_bytes,
        target_task=args.task,
    ).to(device)
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    graph_queue: deque[torch.Tensor] = deque(maxlen=256)
    domain_prototypes: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    best_epoch = 0
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else _default_checkpoint(args)
    results_path = args.results.expanduser().resolve() if args.results else checkpoint.with_suffix(".json")
    print(
        json.dumps(
            {
                "format": PRETRAINING_FORMAT,
                "task": args.task,
                "target_dataset": args.dataset,
                "device": str(device),
                "sources": [spec.key for spec in source_specs],
                "validation_sources": [source.spec.key for source in validation_sources],
                "checkpoint": str(checkpoint),
                "protocol_digest": entry["protocol_digest"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        sums: dict[str, float] = {}
        for _ in range(args.steps_per_epoch):
            source = _sample_source(
                loaded_sources,
                target_task=args.task,
                same_task_probability=args.same_task_probability,
                rng=rng,
            )
            transition_index = _domain_transition(source, split="train", rng=rng)
            optimizer.zero_grad(set_to_none=True)
            total, terms, graph_snapshot = _step_loss(
                model,
                source,
                transition_index,
                device=device,
                args=args,
                graph_queue=graph_queue,
                domain_prototypes=domain_prototypes,
                training=True,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"non-finite pretraining loss at epoch={epoch}: {terms}"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), 5.0)
            optimizer.step()
            model.update_targets()
            graph_queue.append(graph_snapshot)
            previous = domain_prototypes.get(source.spec.key)
            if previous is None:
                domain_prototypes[source.spec.key] = graph_snapshot
            else:
                domain_prototypes[source.spec.key] = (
                    args.target_momentum * previous
                    + (1.0 - args.target_momentum) * graph_snapshot
                ).detach()
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + value
        train_terms = {key: value / args.steps_per_epoch for key, value in sums.items()}

        model.eval()
        validation_sums: dict[str, float] = {}
        validation_count = 0





        with torch.no_grad():
            for source in validation_sources:
                candidates = source.indices.get("val", [])
                if not candidates:
                    continue
                for transition_index in candidates[: args.val_steps_per_domain]:
                    _total, terms, _graph_snapshot = _step_loss(
                        model,
                        source,
                        transition_index,
                        device=device,
                        args=args,
                        graph_queue=graph_queue,
                        domain_prototypes=domain_prototypes,
                        training=False,
                    )
                    for key, value in terms.items():
                        validation_sums[key] = validation_sums.get(key, 0.0) + value
                    validation_count += 1
        if validation_count:
            validation_terms = {
                key: value / validation_count
                for key, value in validation_sums.items()
            }
            val_loss = float(validation_terms["total"])
        else:
            validation_terms = {"total": train_terms["total"]}
            val_loss = train_terms["total"]
        record = {
            "epoch": epoch,
            "train": train_terms,
            "validation": validation_terms,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_total={train_terms['total']:.6f} "
            f"val_total={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            _save_checkpoint(
                checkpoint,
                args=args,
                protocol_digest_value=entry["protocol_digest"],
                source_specs=source_specs,
                model=model,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                history=history,
            )
            print(f"saved_best epoch={epoch:03d} path={checkpoint}", flush=True)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "format": PRETRAINING_FORMAT,
                "target_task": args.task,
                "target_dataset": args.dataset,
                "checkpoint": str(checkpoint),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "history": history,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[PRETRAIN][SUMMARY] best_epoch={best_epoch} "
        f"best_val_loss={best_val_loss:.6f} checkpoint={checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
